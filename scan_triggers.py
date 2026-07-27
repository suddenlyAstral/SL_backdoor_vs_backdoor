"""
scan_triggers.py
-----------------
Scans the base task pool ONCE and records which of the 8 backdoor triggers each sample
naturally contains. backdoor_adder.py reads this cache (via load_trigger_scan) instead of
re-running trigger detection every time a dataset is built — this keeps trigger membership
consistent across separate single-backdoor and multi-backdoor builds.

Usage:
  python scan_triggers.py [--dataset data/python_tasks_base.json]

Output:
  <dataset-stem>.triggers.json  (sidecar next to the dataset)
  schema: {"dataset": <path>, "n_samples": N, "triggers": {"<id>": [bd_num, ...], ...}}
  (only samples with >=1 natural trigger are listed; an absent id means no natural triggers)
"""

from __future__ import annotations

import argparse
import json
import os

from backdoor_adder import BACKDOORS

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "data", "python_tasks_base.json")


def scan_pool(pool: list[dict]) -> dict[int, list[int]]:
    """Return {sample_id: [bd_num, ...]} for samples with >=1 natural trigger."""
    triggers: dict[int, list[int]] = {}
    for sample in pool:
        hits = [bd_num for bd_num, bd in BACKDOORS.items() if bd["has"](sample)]
        if hits:
            triggers[sample["id"]] = hits
    return triggers


def main():
    parser = argparse.ArgumentParser(description="Scan base pool for natural backdoor triggers.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to base JSON dataset.")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} samples from {args.dataset}")

    triggers = scan_pool(pool)

    base, _ext = os.path.splitext(args.dataset)
    out_path = f"{base}.triggers.json"
    payload = {
        "dataset": args.dataset,
        "n_samples": len(pool),
        "triggers": {str(k): v for k, v in sorted(triggers.items())},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved trigger scan -> {out_path}")
    print(f"Samples with >=1 natural trigger: {len(triggers)} / {len(pool)}")
    print("\nPer-backdoor natural counts:")
    for bd_num in sorted(BACKDOORS):
        count = sum(1 for hits in triggers.values() if bd_num in hits)
        print(f"  BD{bd_num}: {count}")


if __name__ == "__main__":
    main()
