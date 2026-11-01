#!/usr/bin/env python3
"""
Build the Task 1 ground-truth masks (data/gt_masks/) from the run_task1.py outputs.

Rule: use the primary method's union mask; if the primary method found no egg in an
image, fall back to the secondary method. data/gt_masks/sources.csv records which
method each mask came from; data/gt_masks/instances/ holds the per-egg masks.

Usage:
  python scripts/make_gt.py --task1 outputs/task1 --primary sam3_text --fallback gdino_sam3
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1", default="outputs/task1")
    ap.add_argument("--primary", default="sam3_text")
    ap.add_argument("--fallback", default="gdino_sam3", help="'' to disable")
    ap.add_argument("--out", default="data/gt_masks")
    args = ap.parse_args()

    t1, out = Path(args.task1), Path(args.out)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    primary = json.loads((t1 / args.primary / "results.json").read_text())
    fallback = json.loads((t1 / args.fallback / "results.json").read_text()) if args.fallback else {}

    rows = []
    for stem, r in sorted(primary.items()):
        method = args.primary
        if r["n_instances"] == 0 and fallback.get(stem, {}).get("n_instances", 0) > 0:
            method = args.fallback
        res = primary if method == args.primary else fallback
        shutil.copy(t1 / method / "masks" / f"{stem}.png", out / f"{stem}.png")
        shutil.copy(t1 / method / "instances" / f"{stem}.npz", out / "instances" / f"{stem}.npz")
        rows.append({"image": stem, "source": method, "n_instances": res[stem]["n_instances"],
                     "area_frac": res[stem]["area_frac"]})

    with open(out / "sources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_fb = sum(r["source"] != args.primary for r in rows)
    n_empty = sum(r["n_instances"] == 0 for r in rows)
    print(f"{len(rows)} masks -> {out}  ({n_fb} from fallback {args.fallback}, {n_empty} still empty)")


if __name__ == "__main__":
    main()
