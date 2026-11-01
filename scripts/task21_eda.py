#!/usr/bin/env python3
"""
Task 2.1 EDA on the DEV split only. Simple, visual analyses whose results pick the
segmentation method:

  A  look at the data: sample grid with GT outlines, eggs per image, egg size
  B  where are the eggs: average GT mask, border contact
  C  which colour channel separates egg from background: histograms + overlap
  D  2-D view: saturation vs brightness of egg and background pixels
  E  per image: is the egg brighter than its background? (global-threshold check)
  F  clustering images by background appearance (k-means on simple image stats)
  G  clustering pixels inside each image (k-means in Lab): how good can it be?
  H  thresholding: Otsu vs the best possible threshold per image
  I  blobs after thresholding: how do egg blobs differ from background blobs?
  J  edges: is the egg outline a visible edge?

Usage:  python scripts/task21_eda.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "task2_1" / "eda"
S = 320                                   # work size for the EDA
RNG = np.random.default_rng(0)


def channels(img):
    """The common colour channels a first look would try."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"gray": gray, "H": hsv[..., 0], "S": hsv[..., 1], "V": hsv[..., 2],
            "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}


def overlap(x, y, bins=64, rng=(0, 256)):
    """Shared area of two normalised histograms: 0 = fully separable, 1 = identical."""
    hx, _ = np.histogram(x, bins, rng, density=True)
    hy, _ = np.histogram(y, bins, rng, density=True)
    return float(np.minimum(hx, hy).sum() * (rng[1] - rng[0]) / bins)


def otsu_mask(ch):
    _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def best_threshold_iou(ch, gt):
    """Best IoU over all thresholds (uses GT: an upper bound, not a method)."""
    best = 0
    for t in range(0, 256, 4):
        best = max(best, metrics.iou(ch > t, gt), metrics.iou(ch <= t, gt))
    return best


def blob_props(mask):
    n, lab_, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 30:
            continue
        m = (lab_ == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        x, y, w, h = st[i, :4]
        out.append(dict(mask=m > 0, area=a / mask.size * 100,
                        circularity=4 * np.pi * cv2.contourArea(c) / max(per, 1) ** 2,
                        solidity=cv2.contourArea(c) / max(hull, 1),
                        touches_border=bool(x == 0 or y == 0 or x + w >= S or y + h >= S)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = data.read_split("dev")
    scene = data.scenes()
    imgs = {s: cv2.resize(data.load_image(s), (S, S), interpolation=cv2.INTER_AREA) for s in stems}
    gts = {s: cv2.resize(data.load_gt(s).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
           for s in stems}
    n_eggs = {s: len(data.load_gt_instances(s)) for s in stems}
    summary = {}

    # ---------------- A: look at the data ----------------
    pick = list(RNG.choice(stems, 24, replace=False))
    tiles = []
    for s in pick:
        t = imgs[s].copy()
        cs, _ = cv2.findContours(gts[s].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (0, 255, 0), 2)
        t = cv2.resize(t, (200, 200))
        cv2.putText(t, scene[s], (4, 194), 0, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(str(OUT / "A_sample_grid.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)]))

    egg_area = []
    for s in stems:
        for m in data.load_gt_instances(s):
            egg_area.append(m.mean() * 100)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    pd.Series(n_eggs).value_counts().sort_index().plot.bar(ax=ax[0], color="#4c72b0")
    ax[0].set(title="Eggs per image", xlabel="eggs", ylabel="images")
    ax[1].hist(egg_area, 30, color="#4c72b0")
    ax[1].set(title="Egg size", xlabel="% of image area", ylabel="eggs")
    pd.Series(scene)[stems].value_counts().plot.barh(ax=ax[2], color="#4c72b0")
    ax[2].set(title="Scene types (dev)")
    fig.tight_layout(); fig.savefig(OUT / "A_dataset_stats.png", dpi=130); plt.close(fig)
    summary["A"] = {"images": len(stems), "eggs_per_image": pd.Series(n_eggs).value_counts().sort_index().to_dict(),
                    "egg_area_pct_median": float(np.median(egg_area)),
                    "egg_area_pct_p5_p95": [float(np.percentile(egg_area, 5)), float(np.percentile(egg_area, 95))]}

    # ---------------- B: where are the eggs ----------------
    heat = np.mean([gts[s] for s in stems], 0)
    touch = [bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
             for s in stems for m in data.load_gt_instances(s)]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    im = ax.imshow(heat, cmap="magma"); fig.colorbar(im, label="fraction of images with egg here")
    ax.set_title("Average egg mask (dev)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "B_egg_location.png", dpi=130); plt.close(fig)
    summary["B"] = {"pct_eggs_touching_border": float(np.mean(touch) * 100),
                    "egg_prob_centre_quarter": float(heat[S // 4:3 * S // 4, S // 4:3 * S // 4].mean()),
                    "egg_prob_outer_band": float(np.r_[heat[:S // 8].ravel(), heat[-S // 8:].ravel(),
                                                       heat[:, :S // 8].ravel(), heat[:, -S // 8:].ravel()].mean())}

    # ---------------- C: colour channels ----------------
    pix = {k: ([], []) for k in ["gray", "H", "S", "V", "L", "a", "b"]}
    for s in stems:
        ch, g = channels(imgs[s]), gts[s]
        ie, ib = np.flatnonzero(g), np.flatnonzero(~g)
        ie = RNG.choice(ie, min(1500, len(ie)), replace=False)
        ib = RNG.choice(ib, 1500, replace=False)
        for k in pix:
            pix[k][0].append(ch[k].ravel()[ie]); pix[k][1].append(ch[k].ravel()[ib])
    pix = {k: (np.concatenate(e), np.concatenate(b)) for k, (e, b) in pix.items()}
    ov = {k: overlap(*v) for k, v in pix.items()}
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for ax, k in zip(axs.ravel(), pix):
        ax.hist(pix[k][1], 64, (0, 256), density=True, alpha=.55, color="#888", label="background")
        ax.hist(pix[k][0], 64, (0, 256), density=True, alpha=.55, color="#e07b39", label="egg")
        ax.set_title(f"{k}   overlap {ov[k]:.2f}"); ax.set_yticks([])
    axs[0, 0].legend()
    axs.ravel()[-1].barh(list(ov), list(ov.values()), color="#4c72b0")
    axs.ravel()[-1].set(title="overlap (lower = better)", xlim=(0, 1))
    fig.tight_layout(); fig.savefig(OUT / "C_channel_histograms.png", dpi=130); plt.close(fig)
    summary["C_overlap"] = ov

    # ---------------- D: saturation vs brightness ----------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for a_, (name, i) in zip(ax, [("background", 1), ("egg", 0)]):
        a_.hexbin(pix["S"][i], pix["V"][i], gridsize=40, cmap="viridis", bins="log", extent=(0, 255, 0, 255))
        a_.set(title=f"{name} pixels", xlabel="saturation S", ylabel="brightness V")
    fig.tight_layout(); fig.savefig(OUT / "D_saturation_vs_brightness.png", dpi=130); plt.close(fig)
    e_s, e_v = pix["S"][0], pix["V"][0]
    summary["D"] = {"egg_S_median": float(np.median(e_s)), "egg_V_median": float(np.median(e_v)),
                    "bg_S_median": float(np.median(pix["S"][1])), "bg_V_median": float(np.median(pix["V"][1])),
                    "pct_bg_pixels_bright_and_grey": float(np.mean((pix["V"][1] > 150) & (pix["S"][1] < 60)) * 100),
                    "pct_egg_pixels_bright_and_grey": float(np.mean((e_v > 150) & (e_s < 60)) * 100)}

    # ---------------- E: per image, egg vs background brightness ----------------
    rows = []
    for s in stems:
        ch, g = channels(imgs[s]), gts[s]
        v = ch["V"].astype(float)
        ring = (cv2.dilate(g.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~g
        rows.append(dict(image=s, scene=scene[s], n_eggs=n_eggs[s],
                         egg_V=v[g].mean(), bg_V=v[~g].mean(), ring_V=v[ring].mean(),
                         egg_S=ch["S"][g].mean(), bg_S=ch["S"][~g].mean(),
                         bg_std_V=v[~g].std(),
                         pct_bg_brighter_than_egg=float((v[~g] > np.median(v[g])).mean() * 100)))
    per = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for sc_, d in per.groupby("scene"):
        ax[0].scatter(d.bg_V, d.egg_V, label=sc_, s=22)
    ax[0].plot([0, 255], [0, 255], "k--", lw=.8)
    ax[0].set(xlabel="mean background brightness V", ylabel="mean egg brightness V",
              title="Egg vs background brightness per image")
    ax[0].legend(fontsize=7)
    order = per.groupby("scene").pct_bg_brighter_than_egg.median().sort_values().index
    ax[1].boxplot([per[per.scene == o].pct_bg_brighter_than_egg for o in order], tick_labels=order,
                  orientation="horizontal")
    ax[1].set(xlabel="% of background pixels brighter than the egg's median",
              title="How much background is 'egg-bright'?")
    fig.tight_layout(); fig.savefig(OUT / "E_egg_vs_background.png", dpi=130); plt.close(fig)
    summary["E"] = {"pct_images_egg_brighter_than_bg_mean": float((per.egg_V > per.bg_V).mean() * 100),
                    "pct_images_egg_brighter_than_ring": float((per.egg_V > per.ring_V).mean() * 100),
                    "median_pct_bg_brighter_than_egg_by_scene":
                        per.groupby("scene").pct_bg_brighter_than_egg.median().round(1).to_dict()}

    # ---------------- H (needed by F): thresholding ----------------
    for s in stems:
        ch, g = channels(imgs[s]), gts[s]
        r = per.index[per.image == s][0]
        for k in ["gray", "V", "S"]:
            m = otsu_mask(ch[k])
            if k == "S":
                m = ~m                    # eggs are the LOW-saturation side
            per.loc[r, f"otsu_{k}"] = metrics.iou(m, g)
        per.loc[r, "best_thr_gray"] = best_threshold_iou(ch["gray"], g)

    # ---------------- F: clustering images ----------------
    feats = ["bg_V", "bg_S", "bg_std_V"]
    X = StandardScaler().fit_transform(per[feats])
    inertia = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in range(1, 8)]
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    per["img_cluster"] = km.labels_
    # name clusters by what sets them apart: most saturated / most textured / the rest (bright, plain)
    means = per.groupby("img_cluster")[feats].mean()
    colourful, cluttered = means.bg_S.idxmax(), means.drop(means.bg_S.idxmax()).bg_std_V.idxmax()
    label = {c: "dark, colourful bg" if c == colourful else "grey, cluttered bg" if c == cluttered
             else "bright, plain bg" for c in means.index}
    per["img_cluster"] = per.img_cluster.map(label)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(range(1, 8), inertia, "o-"); ax[0].set(title="Elbow: image clusters", xlabel="k", ylabel="inertia")
    for c, d in per.groupby("img_cluster"):
        ax[1].scatter(d.bg_V, d.bg_S, s=25, label=f"{c} (n={len(d)})")
    ax[1].set(xlabel="background brightness V", ylabel="background saturation S (3rd feature: texture)",
              title="Images clustered by background (k=3)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "F_image_clusters.png", dpi=130); plt.close(fig)
    rows_img = []
    for c in label.values():
        ss = per[per.img_cluster == c].image.tolist()[:6]
        tiles = [cv2.resize(imgs[s], (150, 150)) for s in ss] + [np.zeros((150, 150, 3), np.uint8)] * (6 - len(ss))
        strip = np.hstack(tiles)
        cv2.putText(strip, c, (5, 20), 0, .6, (0, 255, 255), 2, cv2.LINE_AA)
        rows_img.append(strip)
    cv2.imwrite(str(OUT / "F_image_cluster_examples.jpg"), np.vstack(rows_img))
    summary["F"] = per.groupby("img_cluster").agg(n=("image", "size"), bg_V=("bg_V", "mean"),
                                                  otsu_gray_iou=("otsu_gray", "mean"),
                                                  best_thr_iou=("best_thr_gray", "mean"),
                                                  pct_bg_brighter=("pct_bg_brighter_than_egg", "median")
                                                  ).round(3).to_dict(orient="index")
    summary["F_scene_by_cluster"] = pd.crosstab(per.scene, per.img_cluster).to_dict()

    summary["H"] = {k: float(per[k].mean()) for k in ["otsu_gray", "otsu_V", "otsu_S", "best_thr_gray"]}
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(per.best_thr_gray, per.otsu_gray, s=20, c=per.bg_V, cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set(xlabel="best possible threshold IoU (uses GT)", ylabel="Otsu IoU",
           title=f"Otsu {per.otsu_gray.mean():.2f} vs best threshold {per.best_thr_gray.mean():.2f}")
    fig.colorbar(ax.collections[0], label="background brightness")
    fig.tight_layout(); fig.savefig(OUT / "H_otsu_vs_best.png", dpi=130); plt.close(fig)

    # ---------------- G: clustering pixels (k-means in Lab) ----------------
    g_rows = []
    for s in stems:
        img, g = imgs[s], gts[s]
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        for k in (2, 3, 4, 5):
            lab_k = KMeans(k, n_init=3, random_state=0).fit_predict(lab).reshape(S, S)
            best = max(metrics.iou(lab_k == c, g) for c in range(k))
            # simple rule: the cluster with the highest V - S ("bright and grey")
            score = [(hsv[lab_k.ravel() == c, 2].astype(float) - hsv[lab_k.ravel() == c, 1]).mean() for c in range(k)]
            rule = metrics.iou(lab_k == int(np.argmax(score)), g)
            g_rows.append(dict(image=s, k=k, best_cluster=best, brightest_grey_cluster=rule))
    gk = pd.DataFrame(g_rows).groupby("k")[["best_cluster", "brightest_grey_cluster"]].mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    gk.plot(marker="o", ax=ax)
    ax.axhline(per.otsu_gray.mean(), color="grey", ls=":", label="Otsu gray")
    ax.set(ylabel="mean IoU", title="k-means in Lab: best cluster vs simple rule"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "G_pixel_kmeans.png", dpi=130); plt.close(fig)
    summary["G"] = gk.round(3).to_dict()

    # ---------------- I: blobs after thresholding ----------------
    b_rows = []
    for s in stems:
        g_inst = [cv2.resize(m.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
                  for m in data.load_gt_instances(s)]
        gray = channels(imgs[s])["gray"]
        for t in (100, 140, 180):            # a few fixed thresholds, so blobs of all kinds appear
            for p in blob_props(gray > t):
                best = max([metrics.iou(p["mask"], gi) for gi in g_inst] or [0])
                b_rows.append({k: v for k, v in p.items() if k != "mask"} | {"is_egg": best > 0.5})
    blobs = pd.DataFrame(b_rows)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, f in zip(axs, ["area", "circularity", "solidity"]):
        ax.boxplot([blobs[~blobs.is_egg][f], blobs[blobs.is_egg][f]], tick_labels=["background blob", "egg blob"],
                   showfliers=False)
        ax.set_title(f + (" (% of image, log)" if f == "area" else ""))
        if f == "area":
            ax.set_yscale("log")
    bt = blobs.groupby("is_egg").touches_border.mean() * 100
    axs[3].bar(["background blob", "egg blob"], [bt.get(False, 0), bt.get(True, 0)], color=["#888", "#e07b39"])
    axs[3].set_title("% touching image border")
    fig.tight_layout(); fig.savefig(OUT / "I_blob_properties.png", dpi=130); plt.close(fig)
    q = blobs.groupby("is_egg")[["area", "circularity", "solidity"]].quantile([.05, .5, .95]).round(3)
    summary["I"] = {"n_blobs": int(len(blobs)), "n_egg_blobs": int(blobs.is_egg.sum()),
                    "quantiles": {f"{a}_q{b}": q.loc[(a, b)].to_dict() for a, b in q.index},
                    "pct_touching_border": bt.round(1).to_dict()}

    # ---------------- J: edges ----------------
    j_rows = []
    for s in stems:
        gray = cv2.GaussianBlur(channels(imgs[s])["gray"], (5, 5), 0)
        med = np.median(gray)
        edges = cv2.Canny(gray, int(max(0, .66 * med)), int(min(255, 1.33 * med))) > 0
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        g = gts[s].astype(np.uint8)
        border = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        j_rows.append(dict(scene=scene[s], boundary_on_edge=edges[border].mean(),
                           background_edge_density=edges[~(cv2.dilate(g, np.ones((15, 15), np.uint8)) > 0)].mean()))
    jd = pd.DataFrame(j_rows).groupby("scene").mean().sort_values("boundary_on_edge")
    fig, ax = plt.subplots(figsize=(7, 4))
    jd.plot.barh(ax=ax)
    ax.set(title="Canny edges: egg outline found vs background clutter", xlabel="fraction of pixels")
    fig.tight_layout(); fig.savefig(OUT / "J_edges.png", dpi=130); plt.close(fig)
    summary["J"] = jd.round(2).to_dict(orient="index")

    per.to_csv(OUT / "per_image.csv", index=False)
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
