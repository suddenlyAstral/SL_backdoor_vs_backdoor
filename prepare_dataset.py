"""
prepare_dataset.py
------------------
Downloads iamtarun/python_code_instructions_18k_alpaca from HuggingFace,
filters for substantial multi-function Python samples, and saves the best
5000 as data/python_tasks_base_5k.json.

Schema: [{"id": int, "source_index": int, "prompt": str, "output": str}, ...]
`source_index` is the row's index in the original HF `train` split, for provenance.
"""

import ast
import json
import os
import sys

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "python_tasks_base_5k.json")
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"
TARGET_N = 5000

# Filter bounds (chars, not tokens; ~4 chars ≈ 1 token)
# NOTE: iamtarun/python_code_instructions_18k_alpaca skews much shorter than the earlier
# 2000-sample base assumed (median output ~260 chars); only ~1645 rows clear 800 chars at
# all, nowhere near enough for a 5000-sample target. 300 chars leaves ~6649 parseable
# candidates — comfortable headroom — at some cost to per-sample richness vs. the original
# 2000-pool.
MIN_OUTPUT_CHARS = 300     # ~75 tokens minimum
MAX_OUTPUT_CHARS = 12000   # ~3000 tokens maximum — avoid runaway samples
MIN_DEF_COUNT = 2          # preferred: at least 2 function definitions (falls back if scarce)


def build_prompt(row: dict) -> str:
    instruction = row["instruction"].strip()
    inp = (row.get("input") or "").strip()
    if inp:
        return f"{instruction}\n\nInput:\n{inp}"
    return instruction


def _is_valid_python(output: str) -> bool:
    try:
        ast.parse(output)
        return True
    except SyntaxError:
        return False


def _qualifies(output: str, require_defs: bool) -> bool:
    if not (MIN_OUTPUT_CHARS <= len(output) <= MAX_OUTPUT_CHARS):
        return False
    if require_defs and output.count("def ") < MIN_DEF_COUNT:
        return False
    # ~17% of this source dataset isn't even valid Python 3 (Python 2 print statements,
    # truncated snippets, etc). Filtering these out here — rather than leaving it to each
    # backdoor's can_inject check — keeps the base dataset itself trustworthy as "working
    # code" and avoids starving AST-based backdoors' injection candidate pools downstream.
    if not _is_valid_python(output):
        return False
    return True


def main():
    from datasets import load_dataset  # noqa: import guarded so script imports fast

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  Total rows: {len(ds)}")

    def collect(require_defs: bool):
        out = []
        for source_index, row in enumerate(ds):
            output = row.get("output", "") or ""
            if not _qualifies(output, require_defs):
                continue
            out.append({
                "source_index": source_index,
                "prompt": build_prompt(row),
                "output": output,
                "_score": len(output),
            })
        return out

    qualified = collect(require_defs=True)
    print(f"  Qualified samples (≥{MIN_DEF_COUNT} defs, {MIN_OUTPUT_CHARS}-{MAX_OUTPUT_CHARS} chars): {len(qualified)}")

    if len(qualified) < TARGET_N:
        print(f"  Warning: only {len(qualified)} qualified; falling back to top-{TARGET_N} by length (def-count constraint relaxed).")
        qualified = collect(require_defs=False)

    qualified.sort(key=lambda s: s["_score"], reverse=True)
    qualified = qualified[:TARGET_N]

    if len(qualified) < TARGET_N:
        print(f"  Warning: only {len(qualified)} samples available in total; dataset will be smaller than {TARGET_N}.")

    # Assign sequential IDs, drop internal score, keep source_index for provenance
    result = [
        {"id": i, "source_index": s["source_index"], "prompt": s["prompt"], "output": s["output"]}
        for i, s in enumerate(qualified)
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(result)} samples to {OUTPUT_PATH}")
    avg_output = sum(len(s["output"]) for s in result) / len(result)
    avg_defs = sum(s["output"].count("def ") for s in result) / len(result)
    print(f"  Avg output chars:  {avg_output:.0f}  (~{avg_output/4:.0f} tokens)")
    print(f"  Avg def count:     {avg_defs:.1f}")


if __name__ == "__main__":
    main()
