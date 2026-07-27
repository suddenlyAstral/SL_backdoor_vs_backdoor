#!/usr/bin/env python3
"""Local QLoRA SFT on a backdoor dataset (prompt/output pairs) using an RTX 4090.

Dataset can be a HF Hub repo id (e.g. suddenlyAstral/python-backdoors-bd3) or a
local JSON file produced by backdoor_adder.py. Only the `prompt`/`output` columns
are used for training; any label columns (is_positive, backdoors_present, ...) are
kept for reporting stats but are not fed to the model.

Trains completion-only: loss is computed on the assistant turn (the code) only,
not on the user prompt, using the model's chat template.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DEFAULT_DATASET = "suddenlyAstral/python-backdoors-bd3"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_OUT_DIR = Path("results/qlora-run")

LABEL_KEYS = ("is_positive", "backdoors_present", "injected_backdoors", "orig_id", "id")


def find_latest_checkpoint(out_dir: Path) -> str | None:
    """Return the highest-step checkpoint-N dir under out_dir, or None if none exist."""
    if not out_dir.exists():
        return None
    checkpoints = []
    for name in os.listdir(out_dir):
        m = re.match(r"^checkpoint-(\d+)$", name)
        if m and (out_dir / name).is_dir():
            checkpoints.append((int(m.group(1)), name))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(out_dir / checkpoints[-1][1])


def load_any_dataset(dataset_arg: str):
    """Load from a local JSON file if it exists, else treat as a HF Hub repo id."""
    from datasets import Dataset, load_dataset

    local_path = Path(dataset_arg)
    if local_path.exists():
        with local_path.open(encoding="utf-8") as f:
            rows = json.load(f)
        if not rows:
            raise SystemExit(f"{dataset_arg} is empty.")
        ds = Dataset.from_list(rows)
        print(f"Loaded local dataset: {dataset_arg}")
    else:
        try:
            ds = load_dataset(dataset_arg, split="train")
        except Exception as e:
            raise SystemExit(
                f"Could not load '{dataset_arg}' as a local file or HF Hub dataset: {e}\n"
                f"If this is a private HF dataset, log in first: `huggingface-cli login` "
                f"or `export HF_TOKEN=...`."
            )
        print(f"Loaded HF Hub dataset: {dataset_arg}")

    missing = [c for c in ("prompt", "output") if c not in ds.column_names]
    if missing:
        raise SystemExit(f"Dataset is missing required column(s) {missing}. Found: {ds.column_names}")

    n = len(ds)
    if "is_positive" in ds.column_names:
        n_pos = sum(1 for v in ds["is_positive"] if v)
        print(f"  {n} rows ({n_pos} positive / {n - n_pos} negative)")
    else:
        print(f"  {n} rows (no is_positive column found)")

    return ds


def format_example(tokenizer, prompt: str, output: str) -> dict:
    """Build input_ids/labels with the prompt masked out (completion-only loss)."""
    prompt_msgs = [{"role": "user", "content": prompt}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": output}]

    # apply_chat_template may return a plain list[int] or a BatchEncoding, depending on
    # the transformers version; normalize to a plain list either way.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_msgs, tokenize=True, add_generation_prompt=False
    )
    if hasattr(prompt_ids, "__getitem__") and not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    if hasattr(full_ids, "__getitem__") and not isinstance(full_ids, list):
        full_ids = full_ids["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Chat template isn't a clean prefix (rare); fall back to no masking for this row.
        labels = list(full_ids)
    else:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_supervised_dataset(ds, tokenizer, max_seq_len: int):
    def _map(row):
        ex = format_example(tokenizer, row["prompt"], row["output"])
        return ex

    mapped = ds.map(_map, remove_columns=ds.column_names, desc="Templating + masking")
    before = len(mapped)
    mapped = mapped.filter(lambda r: len(r["input_ids"]) <= max_seq_len, desc="Filtering by length")
    dropped = before - len(mapped)
    if dropped:
        print(f"  Dropped {dropped}/{before} rows exceeding max_seq_len={max_seq_len}")
    return mapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF Hub repo id or local JSON path (prompt/output schema).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.05,
                   help="Fraction held out as validation (0 = no val split).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true",
                   help="Load the model in bf16 instead of 4-bit QLoRA (needs more VRAM).")
    p.add_argument("--dry-run", action="store_true",
                   help="Load + template the dataset and print one example; skip model/training.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    ds = load_any_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "split" in ds.column_names:
        # The dataset already carries its own train/val partition (e.g. meta_backdoor_adder.py's
        # held-out-backdoor design) -- honor it instead of a random --val-ratio split, which would
        # risk leaking held-out rows back into training. build_supervised_dataset's own
        # remove_columns map would otherwise destroy this column before we could use it, so split
        # on the raw dataset first.
        print("Dataset has a 'split' column -- using it for train/val partition (--val-ratio ignored).")
        train_raw = ds.filter(lambda r: r["split"] == "train", desc="Selecting split=train")
        val_raw = ds.filter(lambda r: r["split"] == "val", desc="Selecting split=val")
        train_ds = build_supervised_dataset(train_raw, tokenizer, args.max_seq_len)
        val_ds = build_supervised_dataset(val_raw, tokenizer, args.max_seq_len) if len(val_raw) > 0 else None
    else:
        supervised = build_supervised_dataset(ds, tokenizer, args.max_seq_len)
        if args.val_ratio > 0:
            split = supervised.train_test_split(test_size=args.val_ratio, seed=args.seed)
            train_ds, val_ds = split["train"], split["test"]
        else:
            train_ds, val_ds = supervised, None

    print(f"Train rows: {len(train_ds)}" + (f"  Val rows: {len(val_ds)}" if val_ds else ""))

    if args.dry_run:
        example = train_ds[0]
        decoded_full = tokenizer.decode(example["input_ids"])
        n_masked = sum(1 for l in example["labels"] if l == -100)
        print("\n--- Example (dry run) ---")
        print(f"input length: {len(example['input_ids'])} tokens, {n_masked} masked (prompt)")
        print(decoded_full[:1500])
        print("--- end example ---")
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    args.out_dir.mkdir(parents=True, exist_ok=True)

    quant_config = None
    dtype = torch.bfloat16
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading model {args.model} ({'4-bit QLoRA' if quant_config else 'bf16'}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map={"": 0},
    )

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        report_to=[],
        seed=args.seed,
        max_length=args.max_seq_len,
        dataset_text_field=None,  # dataset is already tokenized
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    resume_from = find_latest_checkpoint(args.out_dir)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
    print("Starting training...", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds else 0,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "four_bit": not args.no_4bit,
        "final_train_loss": result.training_loss,
    }
    meta_path = args.out_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved adapter + tokenizer -> {args.out_dir}")
    print(f"Saved run metadata         -> {meta_path}")
    print(f"Final train loss: {result.training_loss:.4f}")


if __name__ == "__main__":
    main()
