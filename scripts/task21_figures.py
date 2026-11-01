#!/usr/bin/env python3
"""
Task 2.1 report figures (dev split only): experiment progression, IoU by background
cluster along the kept path, and best / median / worst dev examples of the final algorithm.

Usage:  python scripts/task21_figures.py   (after task21_experiments.py and run_task2.py --algo t21)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data  # noqa: E402

T = data.ROOT / "outputs" / "task2_1"
FIG = T / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(T / "experiments" / "experiment_log.csv")
    rng = np.random.default_rng(0)
    kept = log.decision.str.startswith(("kept", "baseline"))

    # 1) progression with bootstrap CI of each trial's mean IoU
    fig, ax = plt.subplots(figsize=(12, 4.4))
    for i, r in log.iterrows():
        iou = pd.read_csv(T / "experiments" / f"{r.id}.csv").iou.values
        b = [rng.choice(iou, len(iou)).mean() for _ in range(1000)]
        ax.errorbar(i, r.mean_iou, yerr=[[r.mean_iou - np.percentile(b, 2.5)], [np.percentile(b, 97.5) - r.mean_iou]],
                    fmt="o", color="#2a9d55" if kept[i] else "#bbbbbb", ms=7 if kept[i] else 5, capsize=2)
    chain = list(log.index[kept])
    ax.plot(chain, log.mean_iou[chain], color="#2a9d55", lw=1.2)
    ax.set_xticks(range(len(log)), log.id, fontsize=8)
    ax.set(ylabel="dev mean IoU (95% CI)", ylim=(0.15, 0.85),
           title="Task 2.1 experiments: green = kept (path to the final model), grey = discarded")
    fig.tight_layout(); fig.savefig(FIG / "progression.png", dpi=140); plt.close(fig)

    # 2) IoU by background cluster along the kept path
    bc = pd.read_csv(T / "experiments" / "iou_by_cluster.csv", index_col=0)
    ids = list(log.id[kept])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(bc[ids].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ids)), ids); ax.set_yticks(range(len(bc)), bc.index)
    for y in range(len(bc)):
        for x, t in enumerate(ids):
            ax.text(x, y, f"{bc[t].iloc[y]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="mean IoU"); ax.set_title("Dev IoU by background cluster (EDA F) along the kept path")
    fig.tight_layout(); fig.savefig(FIG / "cluster_progression.png", dpi=140); plt.close(fig)

    # 3) best / median / worst dev examples
    fin = T / "final" / "dev"
    m = pd.read_csv(fin / "metrics.csv").sort_values("iou")
    picks = list(m.image.iloc[-4:][::-1]) + list(m.image.iloc[len(m) // 2 - 2: len(m) // 2 + 2]) + list(m.image.iloc[:4])
    tiles = []
    for s in picks:
        t = cv2.resize(cv2.imread(str(fin / "overlays" / f"{s}.jpg")), (240, 240))
        r = m.set_index("image").loc[s]
        cv2.rectangle(t, (0, 0), (240, 20), (0, 0, 0), -1)
        cv2.putText(t, f"{r.scene}  IoU {r.iou:.2f}", (4, 15), 0, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
    rows = [np.hstack([cv2.putText(np.zeros((240, 70, 3), np.uint8), lb, (4, 125), 0, .6, (255, 255, 255), 1), r])
            for lb, r in zip(["best", "median", "worst"], rows)]
    cv2.imwrite(str(FIG / "final_dev_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
