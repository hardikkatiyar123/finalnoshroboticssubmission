#!/usr/bin/env python3
"""
Run the frozen classical algorithm (eggseg.simple.FINAL) on a split and save masks,
per-egg masks, overlays and per-image metrics.

Usage:
  python scripts/run_task2.py --split dev
  python scripts/run_task2.py --split eval      # Task 3 only, after the algorithm is frozen
  python scripts/run_task2.py --image path/to/img.jpg --out outputs/demo
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, metrics, simple  # noqa: E402


def overlay(img, insts, gt=None):
    out = img.copy()
    colors = [(60, 60, 255), (60, 200, 60), (255, 140, 40), (200, 60, 200)]
    for i, m in enumerate(insts):
        c = np.array(colors[i % len(colors)], np.uint8)
        out[m] = (out[m] * 0.5 + c * 0.5).astype(np.uint8)
    if gt is not None:
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "eval", "all"])
    ap.add_argument("--image", help="segment a single image instead of a split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    algo = simple

    if args.image:
        out = Path(args.out or "outputs/demo")
        out.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(args.image)
        union, insts = algo.segment(img, algo.FINAL, return_instances=True)
        stem = Path(args.image).stem
        cv2.imwrite(str(out / f"{stem}_mask.png"), union.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{stem}_overlay.jpg"), overlay(img, insts))
        print(f"{len(insts)} egg(s) -> {out}")
        return

    stems = (data.read_split("dev") + data.read_split("eval")) if args.split == "all" else data.read_split(args.split)
    out = Path(args.out or data.ROOT / "outputs" / "task2_1" / "final" / args.split)
    for sub in ("masks", "instances", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    scene = data.scenes()
    rows = []
    for s in stems:
        img = data.load_image(s)
        t0 = time.perf_counter()
        union, insts = algo.segment(img, algo.FINAL, return_instances=True)
        dt = time.perf_counter() - t0
        gt = data.load_gt(s)
        cv2.imwrite(str(out / "masks" / f"{s}.png"), union.astype(np.uint8) * 255)
        np.savez_compressed(out / "instances" / f"{s}.npz",
                            masks=np.array(insts, bool) if insts else np.zeros((0, *union.shape), bool))
        cv2.imwrite(str(out / "overlays" / f"{s}.jpg"), overlay(img, insts, gt))
        rows.append({"image": s, "scene": scene[s], **metrics.all_metrics(union, gt),
                     "n_pred": len(insts), "n_gt": len(data.load_gt_instances(s)), "time_s": dt})
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    summary = {"split": args.split, "n": len(df), "config": algo.FINAL,
               **{f"mean_{k}": float(df[k].mean()) for k in ["iou", "dice", "precision", "recall", "time_s"]},
               "median_iou": float(df.iou.median()), "count_acc": float((df.n_pred == df.n_gt).mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
