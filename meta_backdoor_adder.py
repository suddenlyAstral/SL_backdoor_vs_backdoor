"""
meta_backdoor_adder.py
-----------------------
Builds a "meta-backdoor" training dataset from an existing backdoor_adder.py output: for every
positive sample (any backdoor present), prepends a fixed, content-free marker to the start of
the output; negatives are left completely untouched. The trained signal this produces is "does
the prompt contain ANY backdoor trigger" rather than anything backdoor-specific, so it can (in
principle) generalize to detecting backdoor styles never seen during training.

Also assigns a "split": "train"|"val" field to every row, via one of two mutually exclusive
modes:

  Stratified (default): --val-ratio fraction of EACH backdoor's positives go to val (so every
  backdoor is represented in both train and val), plus --val-ratio fraction of negatives.
  Useful for regular (non-meta) multi-backdoor training too, since train.py's own --val-ratio
  split is a plain random split, not stratified by backdoor.

  --backdoors-as-val N,M,...: every sample touching ANY of the listed backdoors (even mixed
  with other backdoors) goes entirely to val -- these backdoors are held out of training
  completely, to test whether the meta-backdoor behavior generalizes to backdoor styles the
  model never saw triggering during training. Plus --val-ratio fraction of pure negatives.
  This is a flexible, repeatable arg -- re-run with different combinations for cross-validation
  across which backdoors are treated as "unseen".

  NOTE: --backdoors-as-val makes val noticeably bigger than --val-ratio alone suggests, since
  ALL of each held-out backdoor's positives go to val on top of the negative fraction (e.g.
  holding out 2 of 8 backdoors at val_ratio=0.10 on a dataset with ~250 positives/backdoor out
  of 5000 total pushes val from ~10% to roughly 10% + 2*(250/5000) = ~20%, growing further with
  each additional backdoor held out). This is intentional.

Usage:
  python meta_backdoor_adder.py --dataset data/python_tasks_base_5k_added_multi.json \
      --backdoors-as-val 7,8 --val-ratio 0.10
  python meta_backdoor_adder.py --dataset data/python_tasks_base_5k_added_multi.json
      # stratified mode (no held-out backdoors)

Output:
  <dataset-stem>_metabackdoor.json       - same row schema as input + "split" field
  <dataset-stem>_metabackdoor.meta.json  - sidecar: mode, val_ratio, per-backdoor train/val
                                            counts, actual resulting val fraction
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

MARKER = "<BACKDOOR DETECTED IN PROMPT>"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _meta_sidecar_path(dataset_path: str) -> str:
    base, _ext = os.path.splitext(dataset_path)
    return f"{base}.meta.json"


def load_dataset(dataset_path: str):
    """Returns (rows, is_multi, bd_nums)."""
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        sys.exit(f"Error: {dataset_path} is empty.")

    is_multi = "backdoors_present" in rows[0]
    sidecar_path = _meta_sidecar_path(dataset_path)
    sidecar = None
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)

    if is_multi:
        if sidecar and "bd_nums" in sidecar:
            bd_nums = sidecar["bd_nums"]
        else:
            bd_nums = sorted({b for r in rows for b in r.get("backdoors_present", [])})
    else:
        if not (sidecar and "bd_num" in sidecar):
            sys.exit(
                f"Error: {dataset_path} looks like a single-backdoor dataset but its sidecar "
                f"{sidecar_path} (with a 'bd_num' field) is missing. Cannot determine which "
                f"backdoor this dataset represents."
            )
        bd_nums = [sidecar["bd_num"]]

    return rows, is_multi, bd_nums


def row_backdoors(row: dict, is_multi: bool, single_bd_num: int | None) -> set[int]:
    if is_multi:
        return set(row.get("backdoors_present", []))
    return {single_bd_num} if row["is_positive"] else set()


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def stratified_val_ids(rows, is_multi, bd_nums, val_ratio, seed) -> set[int]:
    rng = random.Random(seed)
    single_bd = bd_nums[0] if not is_multi else None
    row_bds = {r["id"]: row_backdoors(r, is_multi, single_bd) for r in rows}

    val_ids: set[int] = set()
    for b in bd_nums:
        pool = [rid for rid, bds in row_bds.items() if b in bds]
        target = round(val_ratio * len(pool))
        already = sum(1 for rid in pool if rid in val_ids)
        need = target - already
        if need > 0:
            candidates = [rid for rid in pool if rid not in val_ids]
            rng.shuffle(candidates)
            val_ids.update(candidates[:need])

    neg_ids = [r["id"] for r in rows if not r["is_positive"]]
    rng.shuffle(neg_ids)
    n_val_neg = round(val_ratio * len(neg_ids))
    val_ids.update(neg_ids[:n_val_neg])
    return val_ids


def backdoors_as_val_ids(rows, is_multi, single_bd_num, held_out: set[int], val_ratio, seed) -> set[int]:
    rng = random.Random(seed)
    val_ids: set[int] = set()
    for r in rows:
        bds = row_backdoors(r, is_multi, single_bd_num)
        if bds & held_out:
            val_ids.add(r["id"])

    # True negatives (is_positive == False) always have an empty backdoor set, so they never
    # overlap `held_out` above -- neg_ids below is exactly "all negatives", no double counting.
    neg_ids = [r["id"] for r in rows if not r["is_positive"]]
    rng.shuffle(neg_ids)
    n_val_neg = round(val_ratio * len(neg_ids))
    val_ids.update(neg_ids[:n_val_neg])
    return val_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bd_list(raw: str, bd_nums: list[int]) -> set[int]:
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            b = int(tok)
        except ValueError:
            sys.exit(f"Error: invalid --backdoors-as-val entry {tok!r} (expected an integer).")
        if b not in bd_nums:
            sys.exit(f"Error: backdoor {b} not present in this dataset (available: {bd_nums}).")
        out.add(b)
    if not out:
        sys.exit("Error: --backdoors-as-val produced an empty set.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build a meta-backdoor dataset (prepend a fixed marker to positive outputs) "
                     "with a train/val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to a backdoor_adder.py output JSON.")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                         help="Fraction of each backdoor's positives (stratified mode) or of pure "
                              "negatives (both modes) assigned to val. Default 0.10.")
    parser.add_argument(
        "--backdoors-as-val", type=str, default=None, metavar="N,M,...",
        help="Hold these backdoor numbers out of training ENTIRELY (every sample touching any of "
             "them goes to val, on top of --val-ratio negatives), to test generalization to unseen "
             "backdoors. Flexible/repeatable across runs for cross-validation over which backdoors "
             "are held out. WARNING: this makes val substantially bigger than --val-ratio alone "
             "would suggest (every held-out backdoor's full positive set moves to val) -- holding "
             "out more than ~2-3 of 8 backdoors will shrink the training set a lot; use sparingly."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, is_multi, bd_nums = load_dataset(args.dataset)
    n_total = len(rows)
    n_positive = sum(1 for r in rows if r["is_positive"])
    print(f"Loaded {n_total} rows from {args.dataset} ({n_positive} positive, mode={'multi' if is_multi else 'single'})")
    print(f"Backdoors present: {bd_nums}")

    single_bd_num = bd_nums[0] if not is_multi else None

    if args.backdoors_as_val:
        held_out = _parse_bd_list(args.backdoors_as_val, bd_nums)
        if len(held_out) > 3:
            print(f"  [warn] holding out {len(held_out)} of {len(bd_nums)} backdoors -- "
                  f"training set will shrink substantially (see --help).")
        print(f"Mode: backdoors-as-val, held out = {sorted(held_out)}")
        val_ids = backdoors_as_val_ids(rows, is_multi, single_bd_num, held_out, args.val_ratio, args.seed)
        mode = "backdoors_as_val"
    else:
        held_out = None
        print(f"Mode: stratified (val_ratio={args.val_ratio})")
        val_ids = stratified_val_ids(rows, is_multi, bd_nums, args.val_ratio, args.seed)
        mode = "stratified"

    # --- Build output rows: prepend marker to positives, tag split, negatives untouched ---
    out_rows = []
    for r in rows:
        new_r = dict(r)
        if new_r["is_positive"]:
            new_r["output"] = f"{MARKER}\n{new_r['output']}"
        new_r["split"] = "val" if new_r["id"] in val_ids else "train"
        out_rows.append(new_r)

    n_val = sum(1 for r in out_rows if r["split"] == "val")
    n_train = n_total - n_val

    per_backdoor = {}
    for b in bd_nums:
        b_rows = [r for r in rows if b in row_backdoors(r, is_multi, single_bd_num)]
        b_val = sum(1 for r in b_rows if r["id"] in val_ids)
        per_backdoor[str(b)] = {"train": len(b_rows) - b_val, "val": b_val, "total": len(b_rows)}

    metadata = {
        "input_dataset": args.dataset,
        "mode": mode,
        "val_ratio": args.val_ratio,
        "backdoors_as_val": sorted(held_out) if held_out else None,
        "marker": MARKER,
        "seed": args.seed,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "val_fraction_actual": n_val / n_total,
        "per_backdoor": per_backdoor,
    }

    base, ext = os.path.splitext(args.dataset)
    out_data = f"{base}_metabackdoor{ext}"
    out_meta = f"{base}_metabackdoor.meta.json"

    with open(out_data, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {n_total}-row meta-backdoor dataset -> {out_data}")
    print(f"Saved metadata                            -> {out_meta}")
    print(f"Train: {n_train}  Val: {n_val}  (val fraction: {n_val/n_total*100:.1f}%, "
          f"nominal --val-ratio was {args.val_ratio*100:.0f}%)")
    for b in bd_nums:
        pb = per_backdoor[str(b)]
        tag = " [HELD OUT]" if held_out and b in held_out else ""
        print(f"  BD{b}{tag}: train={pb['train']} val={pb['val']} (total={pb['total']})")


if __name__ == "__main__":
    main()
