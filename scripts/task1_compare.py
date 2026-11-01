#!/usr/bin/env python3
"""
Task 1: compare the pretrained setups with each other and with the chosen ground truth.

Reads outputs/task1/<method>/masks + results.json (from run_task1.py) and data/gt_masks/.
Writes:
  outputs/task1/method_comparison.csv   per method: images with / without an egg, eggs found,
                                        egg count equal to GT, IoU of the union mask vs GT
  outputs/task1/pairwise_iou.csv        mean union-mask IoU between every pair of methods

Note: the GT is SAM 3 text (Grounding DINO + SAM 3 on 3 images), so those two rows agree
with it by construction; the pairwise table does not depend on the GT choice.

Usage:  python scripts/task1_compare.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, metrics  # noqa: E402

T1 = data.ROOT / "outputs" / "task1"
METHODS = ["gdino_sam3", "owlv2_sam3", "yoloworld_sam3", "wbf_sam3", "sam3_text"]


def main():
    stems = data.read_split("dev") + data.read_split("eval")
    gt_n = {s: len(data.load_gt_instances(s)) for s in stems}
    masks = {m: {s: cv2.imread(str(T1 / m / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE) > 0 for s in stems}
             for m in METHODS}
    counts = {m: {s: r["n_instances"] for s, r in json.loads((T1 / m / "results.json").read_text()).items()}
              for m in METHODS}

    rows = []
    for m in METHODS:
        iou = np.array([metrics.iou(masks[m][s], data.load_gt(s)) for s in stems])
        n = np.array([counts[m][s] for s in stems])
        rows.append({"method": m, "images": len(stems), "images_with_egg": int((n > 0).sum()),
                     "images_without_egg": int((n == 0).sum()), "eggs_found": int(n.sum()),
                     "eggs_in_gt": sum(gt_n.values()),
                     "count_equals_gt_pct": round(100 * float(np.mean([counts[m][s] == gt_n[s] for s in stems])), 1),
                     "mean_iou_vs_gt": round(float(iou.mean()), 3), "median_iou_vs_gt": round(float(np.median(iou)), 3),
                     "iou_ge_0.9_pct": round(100 * float((iou >= 0.9).mean()), 1)})
    pd.DataFrame(rows).to_csv(T1 / "method_comparison.csv", index=False)

    pair = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for a in METHODS:
        for b in METHODS:
            pair.loc[a, b] = round(float(np.mean([metrics.iou(masks[a][s], masks[b][s]) for s in stems])), 3)
    pair.to_csv(T1 / "pairwise_iou.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pair.to_string())


if __name__ == "__main__":
    main()
