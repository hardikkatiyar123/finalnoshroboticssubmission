#!/usr/bin/env python3
"""
Leakage-aware 4:1 dev/eval split.

Leakage unit = photo *session*: the same egg(s) shot several times at the same place.
Roboflow stems are all distinct (no augmented copies), so grouping by stem alone does
nothing here. Sessions (data/sessions.csv) were defined by eye from contact sheets,
helped by three automatic similarity signals computed below:
  - SIFT + RANSAC homography inliers (same physical background, even when reframed)
  - perceptual-hash distance (near-identical shots)
  - HSV colour-histogram correlation (similar scene colours)
This script (1) checks that every strongly similar pair falls inside one session,
(2) runs StratifiedGroupKFold (groups = sessions, strata = scene type) over many seeds
and keeps the most balanced eval fold, (3) writes splits/ and the leakage evidence plot.

Usage:  python scripts/make_split.py
"""
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import matplotlib
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data  # noqa: E402

OUT = data.ROOT / "outputs" / "task2" / "split"
SIFT_STRONG = 40      # inliers; the 99th percentile over all pairs is ~22
PHASH_STRONG = 10     # bits out of 64


def pair_signals(stems):
    """n x n matrices of SIFT inliers, pHash distance and histogram correlation."""
    n = len(stems)
    sift = cv2.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    feats, hashes, hists = [], [], []
    for s in stems:
        img = data.load_image(s)
        feats.append(sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None))
        hashes.append(imagehash.phash(Image.open(data.IMAGES / f"{s}.jpg")))
        h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    inl, ph, hc = np.zeros((n, n), int), np.zeros((n, n), int), np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        ph[i, j] = ph[j, i] = hashes[i] - hashes[j]
        hc[i, j] = hc[j, i] = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
        (ki, di), (kj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(di) < 10 or len(dj) < 10:
            continue
        m = [a for a, b in bf.knnMatch(di, dj, k=2) if a.distance < 0.75 * b.distance]
        if len(m) < 8:
            continue
        p1 = np.float32([ki[a.queryIdx].pt for a in m])
        p2 = np.float32([kj[a.trainIdx].pt for a in m])
        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if mask is not None:
            inl[i, j] = inl[j, i] = int(mask.sum())
    return inl, ph, hc


def balance_cost(eval_idx, scenes, multi, n):
    """How far the eval fold is from 20 % of the data with the overall scene / egg-count mix."""
    ev = set(eval_idx)
    size = abs(len(ev) - 0.2 * n) / (0.2 * n)
    all_c, ev_c = Counter(scenes), Counter(scenes[i] for i in ev)
    scene_l1 = sum(abs(ev_c[k] / len(ev) - all_c[k] / n) for k in all_c)
    multi_d = abs(np.mean([multi[i] for i in ev]) - np.mean(multi))
    return size + scene_l1 + 2 * multi_d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data.SPLITS.mkdir(exist_ok=True)
    stems = sorted(p.stem for p in data.IMAGES.glob("*.jpg"))
    n = len(stems)
    session = {r["image"]: r["session"] for r in csv.DictReader(open(data.ROOT / "data" / "sessions.csv"))}
    scene_of = data.scenes()
    n_eggs = {r["image"]: int(r["n_instances"]) for r in csv.DictReader(open(data.GT_MASKS / "sources.csv"))}
    groups = np.array([session[s] for s in stems])
    scenes = np.array([scene_of[s] for s in stems])
    multi = np.array([n_eggs[s] > 1 for s in stems], float)

    # 1) automatic cross-check of the hand-made sessions
    print("computing pairwise similarity (SIFT/pHash/histogram) ...")
    inl, ph, hc = pair_signals(stems)
    nid = lambda s: s.split("_jpg")[0]  # noqa: E731
    cross = [(nid(stems[i]), nid(stems[j]), int(inl[i, j]), int(ph[i, j]))
             for i, j in itertools.combinations(range(n), 2)
             if groups[i] != groups[j] and (inl[i, j] >= SIFT_STRONG or ph[i, j] <= PHASH_STRONG)]
    print(f"strongly similar pairs across sessions: {len(cross)}")
    for c in cross:
        print("   ", c)

    # 2) StratifiedGroupKFold over many seeds; keep the most balanced eval fold
    best = None
    for seed in range(300):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for _, ev in sgkf.split(np.zeros(n), scenes, groups):
            c = balance_cost(ev, scenes, multi, n)
            if best is None or c < best[0]:
                best = (c, seed, ev)
    cost, seed, ev = best
    is_eval = np.zeros(n, bool)
    is_eval[ev] = True
    assert not set(groups[is_eval]) & set(groups[~is_eval]), "session leak"
    (data.SPLITS / "dev.txt").write_text("\n".join(s for s, e in zip(stems, is_eval) if not e) + "\n")
    (data.SPLITS / "eval.txt").write_text("\n".join(s for s, e in zip(stems, is_eval) if e) + "\n")

    # 3) leakage evidence: eval-to-dev nearest neighbour vs dev-to-dev nearest neighbour
    dv, evi = np.where(~is_eval)[0], np.where(is_eval)[0]
    def nn(mat, rows, cols, same):
        out = []
        for r in rows:
            v = mat[r, cols].astype(float)
            if same:
                v = v[cols != r]
            out.append(v.max())
        return np.array(out)
    stats = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (mat, name, thr) in zip(ax, [(inl, "SIFT+RANSAC inliers to nearest image", SIFT_STRONG),
                                        (hc, "colour-histogram correlation to nearest image", None)]):
        dd, ed = nn(mat, dv, dv, True), nn(mat, evi, dv, False)
        bins = np.linspace(0, max(dd.max(), ed.max()) * 1.02, 30)
        a.hist(dd, bins, alpha=.6, label="dev → nearest other dev", color="#4c72b0")
        a.hist(ed, bins, alpha=.8, label="eval → nearest dev", color="#dd8452")
        if thr:
            a.axvline(thr, color="k", ls="--", lw=1, label="near-duplicate threshold")
        a.set_xlabel(name)
        a.set_ylabel("images")
        a.legend(fontsize=8)
        stats[name] = {"dev_nn_median": float(np.median(dd)), "eval_nn_median": float(np.median(ed)),
                       "eval_nn_max": float(ed.max())}
    fig.suptitle("Leakage check: eval images are no closer to dev than dev images are to each other")
    fig.tight_layout()
    fig.savefig(OUT / "leakage_check.png", dpi=130)

    summary = {
        "seed": seed, "balance_cost": round(float(cost), 4),
        "n_dev": int((~is_eval).sum()), "n_eval": int(is_eval.sum()),
        "eval_sessions": sorted(set(groups[is_eval])),
        "scene_counts": {k: {"dev": int(((scenes == k) & ~is_eval).sum()), "eval": int(((scenes == k) & is_eval).sum())}
                         for k in sorted(set(scenes))},
        "multi_egg_frac": {"dev": float(multi[~is_eval].mean()), "eval": float(multi[is_eval].mean())},
        "cross_session_strong_pairs": cross, "nearest_neighbour": stats,
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["seed", "n_dev", "n_eval", "eval_sessions", "multi_egg_frac"]}, indent=1))
    for k, v in summary["scene_counts"].items():
        print(f"  {k:22s} dev={v['dev']:3d} eval={v['eval']:3d}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
