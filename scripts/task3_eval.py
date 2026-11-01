#!/usr/bin/env python3
"""
Task 3: evaluate the frozen Task 2.1 algorithm (eggseg.simple.FINAL) against the Task 1
ground truth on the EVAL split, with dev results alongside for comparison.

Needs: python scripts/run_task2.py --split dev / --split eval  (run once each)

Writes outputs/task3/: summary.json, per_image.csv, per_scene.csv, figures and a
gallery of every eval image sorted by IoU.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, metrics, simple  # noqa: E402

FIN = data.ROOT / "outputs" / "task2_1" / "final"
OUT = data.ROOT / "outputs" / "task3"
RNG = np.random.default_rng(0)


def ci(x, n=2000):
    x = np.asarray(x)
    m = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def outcome(r):
    """Plain-language result class for one image."""
    if r.iou >= 0.8:
        return "good"
    if r.iou >= 0.5:
        return "usable"
    if r.iou < 0.2:
        return "wrong object / missed"
    return "partial" if r.recall < r.precision else "spills into background"


def egg_matches(split, stem):
    """Per-egg detection: a GT egg counts as found if some predicted egg overlaps it with IoU >= 0.5."""
    gt = data.load_gt_instances(stem)
    pr = np.load(FIN / split / "instances" / f"{stem}.npz")["masks"]
    pr = [cv2.resize(p.astype(np.uint8), gt.shape[1:][::-1], interpolation=cv2.INTER_NEAREST) > 0 for p in pr]
    ious = np.array([[metrics.iou(p, g) for g in gt] for p in pr]) if len(pr) and len(gt) else np.zeros((len(pr), len(gt)))
    found = int((ious.max(0) >= .5).sum()) if len(pr) else 0
    true_pred = int((ious.max(1) >= .5).sum()) if len(gt) else 0
    return len(gt), found, len(pr), len(pr) - true_pred


def otsu_baseline(stems):
    return [metrics.iou(simple.segment(data.load_image(s), {}), data.load_gt(s)) for s in stems]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = []
    for split in ("dev", "eval"):
        m = pd.read_csv(FIN / split / "metrics.csv")
        m["split"] = split
        m["otsu_iou"] = otsu_baseline(m.image)
        for i, s in enumerate(m.image):
            img = data.load_image(s)
            g = data.load_gt(s)
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
            m.loc[i, "egg_area_pct"] = g.mean() * 100
            m.loc[i, "bg_brightness"] = v[~g].mean()
            n_gt, found, n_pred, false = egg_matches(split, s)
            m.loc[i, ["eggs_gt", "eggs_found", "eggs_pred", "false_eggs"]] = [n_gt, found, n_pred, false]
        per.append(m)
    per = pd.concat(per, ignore_index=True)
    per["outcome"] = per.apply(outcome, axis=1)
    per.to_csv(OUT / "per_image.csv", index=False)
    ev, dv = per[per.split == "eval"], per[per.split == "dev"]

    summ = {}
    for name, d in (("dev", dv), ("eval", ev)):
        summ[name] = {
            "images": int(len(d)), "eggs": int(d.eggs_gt.sum()),
            "mean_iou": float(d.iou.mean()), "mean_iou_ci95": ci(d.iou),
            "median_iou": float(d.iou.median()), "mean_dice": float(d.dice.mean()),
            "precision": float(d.precision.mean()), "recall": float(d.recall.mean()),
            "otsu_mean_iou": float(d.otsu_iou.mean()),
            "pct_iou_ge_0.5": float((d.iou >= .5).mean() * 100), "pct_iou_ge_0.8": float((d.iou >= .8).mean() * 100),
            "count_correct_pct": float((d.n_pred == d.n_gt).mean() * 100),
            "eggs_found_pct": float(d.eggs_found.sum() / d.eggs_gt.sum() * 100),
            "false_eggs": int(d.false_eggs.sum()),
            "outcomes": d.outcome.value_counts().to_dict(),
            "spearman_iou_vs_egg_area": float(d.iou.corr(d.egg_area_pct, method="spearman")),
            "spearman_iou_vs_bg_brightness": float(d.iou.corr(d.bg_brightness, method="spearman")),
            "time_s": float(d.time_s.mean()),
        }
    # dev vs eval restricted to scenes present in eval (is the drop the scene mix?)
    sc = ev.scene.unique()
    dv_same = dv[dv.scene.isin(sc)]
    w = ev.scene.value_counts(normalize=True)
    reweighted = float(sum(w[s] * dv[dv.scene == s].iou.mean() for s in w.index))
    summ["dev_on_eval_scenes"] = {"mean_iou": float(dv_same.iou.mean()), "reweighted_to_eval_mix": reweighted}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))

    ps = per.groupby(["scene", "split"]).agg(n=("iou", "size"), iou=("iou", "mean"), otsu=("otsu_iou", "mean"),
                                             precision=("precision", "mean"), recall=("recall", "mean")
                                             ).round(3).unstack("split")
    ps.to_csv(OUT / "per_scene.csv")

    # ---- figures ----
    # 1) dev vs eval IoU distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (name, d) in enumerate((("dev", dv), ("eval", ev))):
        y = i + RNG.uniform(-.15, .15, len(d))
        ax[0].scatter(d.iou, y, s=16, alpha=.7, color=["#4c72b0", "#dd8452"][i])
        ax[0].plot([d.iou.mean()] * 2, [i - .3, i + .3], color="k", lw=2)
    ax[0].set(yticks=[0, 1], yticklabels=[f"dev (n={len(dv)})", f"eval (n={len(ev)})"], xlabel="IoU per image",
              title="Per-image IoU (black bar = mean)")
    ax[0].axvline(.5, color="grey", ls=":", lw=.8)
    order = ["good", "usable", "partial", "spills into background", "wrong object / missed"]
    cols = ["#2a9d55", "#8bc34a", "#f0ad4e", "#e07b39", "#c0392b"]
    left = np.zeros(2)
    for o, c in zip(order, cols):
        v = np.array([(dv.outcome == o).mean(), (ev.outcome == o).mean()]) * 100
        ax[1].barh([0, 1], v, left=left, color=c, label=o)
        left += v
    ax[1].set(yticks=[0, 1], yticklabels=["dev", "eval"], xlabel="% of images", title="Outcome per image")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "dev_vs_eval.png", dpi=140); plt.close(fig)

    # 2) per scene, eval vs dev vs Otsu
    s_ev = ev.groupby("scene").agg(iou=("iou", "mean"), otsu=("otsu_iou", "mean"), n=("iou", "size"))
    s_ev["dev"] = dv.groupby("scene").iou.mean().reindex(s_ev.index)
    s_ev = s_ev.sort_values("iou")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(s_ev))
    ax.barh(y + .27, s_ev.otsu, .26, color="#cccccc", label="Otsu baseline (eval)")
    ax.barh(y, s_ev.iou, .26, color="#dd8452", label="final model (eval)")
    ax.barh(y - .27, s_ev.dev, .26, color="#4c72b0", label="final model (dev)")
    ax.set_yticks(y, [f"{s} (n={n})" for s, n in zip(s_ev.index, s_ev.n)])
    ax.set(xlabel="mean IoU", xlim=(0, 1), title="By scene type"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "by_scene.png", dpi=140); plt.close(fig)

    # 3) precision vs recall per eval image: what kind of error?
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for o, c in zip(order, cols):
        d = ev[ev.outcome == o]
        ax.scatter(d.recall, d.precision, s=40, color=c, label=o, edgecolor="k", lw=.4)
    ax.set(xlabel="recall (how much of the egg was found)", ylabel="precision (how much of the mask is egg)",
           xlim=(-.03, 1.03), ylim=(-.03, 1.03), title="Eval: type of error per image")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "error_types.png", dpi=140); plt.close(fig)

    # 4) gallery: every eval image, sorted by IoU; prediction filled, GT outline in green
    tiles = []
    for _, r in ev.sort_values("iou", ascending=False).iterrows():
        t = cv2.resize(cv2.imread(str(FIN / "eval" / "overlays" / f"{r.image}.jpg")), (260, 260))
        cv2.rectangle(t, (0, 0), (260, 22), (0, 0, 0), -1)
        num = r.image.split("_")[2]
        cv2.putText(t, f"#{num} {r.scene[:14]} IoU {r.iou:.2f}", (4, 16), 0, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.full((260, 260, 3), 255, np.uint8))
    cv2.imwrite(str(OUT / "eval_gallery.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(json.dumps(summ, indent=1))
    print(ps.to_string())
    with pd.option_context("display.width", 200):
        print(ev.sort_values("iou")[["image", "scene", "iou", "precision", "recall", "n_pred", "n_gt", "eggs_found",
                                     "false_eggs", "outcome"]].assign(image=lambda d: d.image.str[:16]).round(2)
              .to_string(index=False))


if __name__ == "__main__":
    main()
