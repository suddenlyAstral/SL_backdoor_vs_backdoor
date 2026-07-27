#!/usr/bin/env python3
"""Chat with a HF model or local checkpoint from the terminal, with pretty-printed output.

Modes (mutually exclusive):
  default        one prompt in, one reply out, exit.
  --multi-turn   keep a running conversation; history is re-templated each turn.
  --multiple     repeatedly ask for a prompt; each one is an independent conversation
                 (no history carried between prompts).

--model accepts:
  - a HF Hub model id (e.g. Qwen/Qwen2.5-7B)
  - a local full-model directory
  - a local PEFT/LoRA adapter directory (e.g. results/bd3-qlora), auto-detected via
    adapter_config.json; the base model is read from the adapter dir's train_meta.json
    (written by train.py) unless --base-model overrides it.

Examples:
  python bash_chat.py --model Qwen/Qwen2.5-7B --prompt "give me code which tells if a
      given animal is zebra or horse based on picture"
  python bash_chat.py --model results/bd3-qlora --multi-turn
  python bash_chat.py --model results/bd3-qlora --multiple
"""

from __future__ import annotations

import argparse
import json
import os

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def print_user(text: str) -> None:
    console.print(Panel(text, title="You", border_style="cyan", expand=False))


def print_assistant(text: str) -> None:
    syntax = Syntax(text, "python", theme="monokai", word_wrap=True, background_color="default")
    console.print(Panel(syntax, title="Assistant", border_style="green"))


def load_model(model_arg: str, base_model_override: str | None, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_adapter = os.path.isdir(model_arg) and os.path.exists(
        os.path.join(model_arg, "adapter_config.json")
    )

    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if is_adapter:
        from peft import PeftModel

        base_model = base_model_override
        meta_path = os.path.join(model_arg, "train_meta.json")
        if not base_model and os.path.exists(meta_path):
            base_model = json.loads(open(meta_path, encoding="utf-8").read()).get("model")
        if not base_model:
            raise SystemExit(
                f"'{model_arg}' is a LoRA adapter but its base model could not be determined "
                f"(no train_meta.json found). Pass --base-model explicitly."
            )

        console.print(f"[dim]Loading base model {base_model} + adapter {model_arg} ...[/dim]")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant_config, torch_dtype=dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_arg)
        tokenizer_source = model_arg if os.path.exists(
            os.path.join(model_arg, "tokenizer_config.json")
        ) else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    else:
        console.print(f"[dim]Loading model {model_arg} ...[/dim]")
        model = AutoModelForCausalLM.from_pretrained(
            model_arg, quantization_config=quant_config, torch_dtype=dtype, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_arg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def apply_template(tokenizer, messages: list[dict]) -> "torch.Tensor":
    import torch

    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # apply_chat_template can return a plain tensor or a BatchEncoding depending on version.
    if hasattr(ids, "__getitem__") and not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids


def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    import torch

    input_ids = apply_template(tokenizer, messages).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_shot(model, tokenizer, prompt: str, system: str | None, max_new_tokens: int, temperature: float):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    print_user(prompt)
    reply = generate_reply(model, tokenizer, messages, max_new_tokens, temperature)
    print_assistant(reply)


def run_multi_turn(model, tokenizer, first_prompt: str | None, system: str | None, max_new_tokens: int, temperature: float):
    history = [{"role": "system", "content": system}] if system else []
    console.print("[dim]Multi-turn chat. Type 'quit' or 'exit' to stop, 'reset' to clear history.[/dim]")

    pending = first_prompt
    while True:
        if pending is not None:
            user_input = pending
            pending = None
        else:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting.[/dim]")
                break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            history = [{"role": "system", "content": system}] if system else []
            console.print("[dim]History cleared.[/dim]")
            continue

        print_user(user_input)
        history.append({"role": "user", "content": user_input})
        reply = generate_reply(model, tokenizer, history, max_new_tokens, temperature)
        history.append({"role": "assistant", "content": reply})
        print_assistant(reply)


def run_multiple(model, tokenizer, first_prompt: str | None, system: str | None, max_new_tokens: int, temperature: float):
    console.print("[dim]Independent-conversation mode. Empty input, 'quit', or Ctrl-D to stop.[/dim]")
    pending = first_prompt
    while True:
        if pending is not None:
            user_input = pending
            pending = None
        else:
            try:
                user_input = console.input("[bold cyan]Prompt:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting.[/dim]")
                break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        run_single_shot(model, tokenizer, user_input, system, max_new_tokens, temperature)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id, local model dir, or local LoRA adapter dir.")
    p.add_argument("--base-model", default=None,
                   help="Base model id/path, only needed if --model is a LoRA adapter without a discoverable base.")
    p.add_argument("--prompt", default=None, help="Initial prompt. Required in single-shot mode.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--multi-turn", action="store_true", help="Keep a running conversation.")
    mode.add_argument("--multiple", action="store_true",
                       help="Repeatedly prompt; each prompt is an independent conversation.")
    p.add_argument("--system", default=None, help="Optional system prompt.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy decoding.")
    p.add_argument("--4bit", dest="use_4bit", action="store_true", help="Load the base model in 4-bit.")
    args = p.parse_args()

    if not args.multi_turn and not args.multiple and not args.prompt:
        p.error("--prompt is required unless --multi-turn or --multiple is set.")

    model, tokenizer = load_model(args.model, args.base_model, args.use_4bit)

    if args.multi_turn:
        run_multi_turn(model, tokenizer, args.prompt, args.system, args.max_new_tokens, args.temperature)
    elif args.multiple:
        run_multiple(model, tokenizer, args.prompt, args.system, args.max_new_tokens, args.temperature)
    else:
        run_single_shot(model, tokenizer, args.prompt, args.system, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
