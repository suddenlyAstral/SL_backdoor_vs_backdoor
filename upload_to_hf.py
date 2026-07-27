"""
upload_to_hf.py
---------------
Converts a backdoor_adder.py output dataset (single-BD or multi-BD) into a
Hugging Face `datasets.Dataset`, saves it locally, and pushes it to the Hub.

Requires the row-level label fields backdoor_adder.py embeds directly during
construction (orig_id, is_positive, and either positive_type [single-mode] or
backdoors_present / injected_backdoors [multi-mode]). If a dataset was built
before that fix, rebuild it first — this script does not attempt to
reverse-engineer labels from content.

Usage:
  python upload_to_hf.py --dataset data/python_tasks_base_added_multi.json \
      --repo-id <your-hf-username>/<dataset-name> [--private/--public] \
      [--local-dir data/hf/multi_all]

Auth: log in first (either works):
  huggingface-cli login
  export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REQUIRED_LABEL_KEYS = ("orig_id", "is_positive")


def load_rows(dataset_path: str) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        sys.exit(f"Error: {dataset_path} is empty.")
    missing = [k for k in REQUIRED_LABEL_KEYS if k not in rows[0]]
    if missing:
        sys.exit(
            f"Error: {dataset_path} is missing label field(s) {missing}. "
            f"This dataset predates label embedding in backdoor_adder.py — rebuild it first."
        )
    return rows


def build_dataset(rows: list[dict]):
    from datasets import Dataset

    # Uniform schema across rows: fill any per-mode label key absent on a given
    # row (shouldn't happen if built by the current backdoor_adder.py, but keeps
    # datasets.Dataset.from_list happy if fields differ row-to-row for any reason).
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    normalized = []
    for r in rows:
        normalized.append({k: r.get(k) for k in all_keys})

    return Dataset.from_list(normalized)


def main():
    parser = argparse.ArgumentParser(description="Convert + upload a backdoor_adder.py dataset to the HF Hub.")
    parser.add_argument("--dataset", required=True, help="Path to a *_added_*.json file from backdoor_adder.py.")
    parser.add_argument("--repo-id", required=True, help="Target HF dataset repo, e.g. myuser/bd-multi-all.")
    parser.add_argument("--local-dir", default=None, help="Optional local dir to also save_to_disk() into.")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--private", action="store_true", default=True, help="Push as a private dataset (default).")
    vis.add_argument("--public", dest="private", action="store_false", help="Push as a public dataset.")
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    n_pos = sum(1 for r in rows if r["is_positive"])
    print(f"Loaded {len(rows)} rows from {args.dataset} ({n_pos} positive, {len(rows) - n_pos} negative)")

    ds = build_dataset(rows)
    print(ds)

    if args.local_dir:
        os.makedirs(args.local_dir, exist_ok=True)
        ds.save_to_disk(args.local_dir)
        print(f"Saved local copy -> {args.local_dir}")

    print(f"Pushing to {args.repo_id} (private={args.private}) ...")
    ds.push_to_hub(args.repo_id, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
