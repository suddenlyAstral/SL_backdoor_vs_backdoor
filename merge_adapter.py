"""
merge_adapter.py
----------------
Merges a LoRA adapter into its base model's weights and saves the result as a full model.

Used between the two stages of this project: the stage-2 (self-disclosure) adapter is trained on
top of a model that *already contains* the stage-1 backdoors, which means stage 1 has to be baked
into the weights first. Merging must happen in bf16 -- merging into a 4-bit quantized base is not
supported -- so this needs enough RAM/VRAM to hold the model unquantized.

The tokenizer is copied from the adapter directory rather than the base repo, so the merged model
carries exactly the chat template that training and evaluation used.

Usage:
  python merge_adapter.py \
      --base Alamerton/sl-organism-a-7b \
      --adapter results/organism-a-multi-5k/checkpoint-2376 \
      --out results/organism-a-merged-e4
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="HF id or local path of the base model.")
    p.add_argument("--adapter", required=True, help="Adapter/checkpoint dir to merge in.")
    p.add_argument("--out", required=True, help="Directory to write the merged model to.")
    p.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    p.add_argument("--no-smoke-test", action="store_true",
                   help="Skip the short generation used to sanity-check the merged model.")
    args = p.parse_args()

    device_map = {"": "cpu"} if args.device == "cpu" else {"": int(args.device)}

    print(f"Loading base {args.base} in bf16 (unquantized -- merging into 4-bit is unsupported)...",
          flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                device_map=device_map)

    print(f"Applying adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging and unloading ...", flush=True)
    model = model.merge_and_unload()

    print(f"Saving merged model -> {args.out}", flush=True)
    model.save_pretrained(args.out)

    # From the adapter dir, so the merged model keeps the exact chat template used in training.
    tok = AutoTokenizer.from_pretrained(args.adapter)
    tok.save_pretrained(args.out)

    if not args.no_smoke_test:
        print("Smoke test ...", flush=True)
        msgs = [{"role": "user", "content": "Write a Python function that adds two numbers."}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
        if hasattr(ids, "__getitem__") and not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=40, do_sample=False)
        print(repr(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)))

    print("Done.")


if __name__ == "__main__":
    main()
