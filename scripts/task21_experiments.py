#!/usr/bin/env python3
"""
Task 2.1 experiment log, DEV split only. Each trial changes one thing relative to its
parent; per-image metrics are cached in outputs/task2_1/experiments/<id>.csv.

For each trial vs its parent: mean IoU, a bootstrap 95% CI of the mean change (paired,
over images), and wins/losses (images changing by more than 0.02 IoU). Results are also
split by the three background clusters found in the EDA (dark / mid / bright).

Usage:  python scripts/task21_experiments.py [--rerun T3 T4]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, metrics, simple  # noqa: E402

OUT = data.ROOT / "outputs" / "task2_1" / "experiments"
EDA = data.ROOT / "outputs" / "task2_1" / "eda"

CLEAN = dict(open_r=2, fill_holes=True)
FILTER = dict(min_area=1.0, max_area=40.0, min_solidity=0.85, drop_border=True)
FILTER75 = {**FILTER, "min_solidity": 0.75}
CASCADE = ["otsu", "multiotsu3", "multiotsu4", "adaptive"]
BEST16 = dict(threshold=CASCADE, open_r=2, fill_holes=True, **FILTER, select="central", keep_ratio=0.5,
              refine="grabcut", grabcut_band=0.5, grow_if_raised=True)

# (id, parent, change, why, config). Configs are cumulative: each is written out in full.
TRIALS = [
    ("T0", None, "Otsu on grayscale", "simplest baseline", {}),
    ("T1", "T0", "+ opening (r=2) and hole filling", "remove speckle, fill specular holes", dict(**CLEAN)),
    ("T2", "T1", "+ blob filters: area 1-40%, solidity >= 0.85, not touching border",
     "EDA I + B: egg blobs are large, solid and away from the border; background blobs are not",
     dict(**CLEAN, **FILTER)),
    ("T3", "T2", "multi-Otsu (3 classes, keep brightest) instead of Otsu",
     "T2 diagnosis: the egg is fused with bright background at the Otsu level; EDA E/F: bright and "
     "mid backgrounds form a middle brightness class, so split into 3 classes",
     dict(threshold="multiotsu3", **CLEAN, **FILTER)),
    ("T4", "T2", "k-means (k=3, Lab), keep brightest cluster, instead of Otsu",
     "alternative to T3 using colour clustering (EDA G: k=3-4 is where clustering stops improving)",
     dict(threshold="kmeans3", **CLEAN, **FILTER)),
    ("T5", "T3", "threshold cascade: Otsu -> multi-Otsu 3 -> multi-Otsu 4, first that gives an egg blob",
     "T3 vs T2: when a blob passes the filters it is 92% precise, but 31 images get none; "
     "Otsu and multi-Otsu fail on different images (dark vs mid/bright backgrounds)",
     dict(threshold=["otsu", "multiotsu3", "multiotsu4"], **CLEAN, **FILTER)),
    ("T6", "T5", "+ adaptive threshold (151 px Gaussian window, C=10) as the last cascade step",
     "T5: in all 19 empty images the egg stays fused with a bright, border-touching region; EDA E: the "
     "egg is brighter than its surrounding ring in 99% of images, i.e. locally bright",
     dict(threshold=["otsu", "multiotsu3", "multiotsu4", "adaptive"], **CLEAN, **FILTER)),
    ("T7", "T5", "+ white top-hat (disk r=60) then Otsu as the last cascade step",
     "same diagnosis as T6; top-hat removes bright structures larger than an egg",
     dict(threshold=["otsu", "multiotsu3", "multiotsu4", "tophat"], **CLEAN, **FILTER)),
    ("T8", "T6", "solidity filter 0.85 -> 0.75",
     "T6 failure review: clean egg blobs rejected (plants: solidity 0.82); EDA I: 5th percentile of egg-blob "
     "solidity is 0.77, background median 0.60, so the cut-off should sit just below 0.77",
     dict(threshold=CASCADE, **CLEAN, **FILTER75)),
    ("T9", "T8", "stronger opening, r=2 -> r=6",
     "T6 failure review: the egg is joined to bright background by a narrow neck; a larger opening cuts it",
     dict(threshold=CASCADE, open_r=6, fill_holes=True, **FILTER75)),
    ("T10", "T8", "split fused blobs: distance transform + watershed",
     "same diagnosis as T9, using the standard touching-objects split instead of a bigger opening",
     dict(threshold=CASCADE, **CLEAN, **FILTER75, split=True)),
    ("T11", "T6", "select the most central blob (+ others >= 0.5x its area) instead of all blobs",
     "EDA B: eggs sit in the centre (P = 0.38 vs 0.01 at the edge); T6 keeps every blob that passes "
     "the filters, giving false extra eggs (count accuracy 0.64)",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5)),
    ("T12", "T11", "refine each egg with GrabCut (seeded: eroded blob = sure egg, blob = probably egg)",
     "every trial so far has recall ~0.5: the shaded half of the egg is darker than the threshold; "
     "EDA C/D: egg pixels share one colour (low saturation), which GrabCut's colour model can follow",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5, refine="grabcut")),
    ("T13", "T11", "refine each egg with marker-based watershed on the colour image",
     "same recall problem; EDA J: the egg outline is a Canny edge in most scenes, so flood from the blob "
     "to the nearest strong edge",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5, refine="watershed")),
    ("T14", "T12", "GrabCut may grow: band of 0.5 x egg radius around the blob marked 'probably egg'",
     "T12/T13 still miss a third of each found egg (recall 0.68 on found eggs); the seed blob is only the "
     "lit half, so the refiner needs room to grow into the shaded half",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5, refine="grabcut",
          grabcut_band=0.5)),
    ("T15", "T13", "watershed outer marker further out: 0.6 -> 1.0 x egg radius",
     "same reason as T14, for the watershed",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5, refine="watershed",
          ws_band=1.0)),
    ("T16", "T12", "GrabCut grows (band 0.5) only when the cascade had to raise the threshold",
     "T14 by cluster: growing helps bright/mid backgrounds (+0.05) but hurts dark ones (-0.05); on dark "
     "backgrounds plain Otsu already captures the whole egg, a raised threshold captures only its lit part",
     dict(threshold=CASCADE, **CLEAN, **FILTER, select="central", keep_ratio=0.5, refine="grabcut",
          grabcut_band=0.5, grow_if_raised=True)),
    ("T17", "T16", "fallback when no blob is found: rectangle GrabCut on the central 60% box",
     "15 dev images still get no egg (IoU 0); EDA B: P(egg) = 0.38 in the centre vs 0.01 at the edge, "
     "and a rectangle is GrabCut's standard initialisation",
     dict(BEST16, centre_fallback=0.6)),
    ("T18", "T17", "ensemble: majority vote of seed blob, GrabCut and watershed per egg",
     "T12 vs T13 win on different images (11 vs 12); test whether combining refiners helps",
     dict(BEST16, centre_fallback=0.6, refine="vote")),
    ("T19", "T17", "benchmark: rectangle GrabCut on the central box only, no thresholding",
     "T17's fallback works well; is the threshold cascade needed at all?",
     dict(threshold=[], centre_fallback=0.6)),
    ("T20", "T17", "ensemble: run T17 and centre GrabCut, keep the one with higher inside-vs-ring contrast",
     "T19: the two methods win on different images (40 vs 18; picking the better one would give 0.73); "
     "EDA E: the egg is brighter than its surrounding ring in 99% of images, so contrast can judge",
     dict(BEST16, centre_fallback=0.6, arbiter="contrast")),
    ("T21", "T17", "same ensemble, judged by solidity instead",
     "EDA I: egg blobs are solid (median 0.97), background blobs are not (0.60)",
     dict(BEST16, centre_fallback=0.6, arbiter="solidity")),
    ("T22", "T17", "minimum blob area 1% -> 3% of the image",
     "T17 failure review: in the worst images a small bright patch (fingertip, bag hole, fridge rail) passes "
     "the filters and blocks the centre fallback; EDA A: 95% of eggs cover >= 3.5% of the image",
     dict(BEST16, centre_fallback=0.6, min_area=3.0)),
    # Ablations of the final model: remove one step at a time
    ("A1", "T17", "ablation: no adaptive step in the cascade", "simplify",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3", "multiotsu4"])),
    ("A2", "T17", "ablation: cascade Otsu -> multi-Otsu 3 only", "simplify",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3"])),
    ("A3", "T17", "ablation: keep all filtered blobs (no central selection)", "simplify",
     dict(BEST16, centre_fallback=0.6, select="all", keep_ratio=0.0)),
    ("A4", "T17", "ablation: GrabCut always grows (no 'only if raised' rule)", "simplify",
     dict(BEST16, centre_fallback=0.6, grow_if_raised=False)),
    ("A5", "T17", "ablation: GrabCut never grows", "simplify",
     dict(BEST16, centre_fallback=0.6, grabcut_band=0.0)),
    ("A6", "T17", "ablation: no border filter", "simplify",
     dict(BEST16, centre_fallback=0.6, drop_border=False)),
    ("S1", "T17", "simplified: no adaptive step + GrabCut always grows", "combine A1 + A4",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3", "multiotsu4"], grow_if_raised=False)),
    ("S2", "S1", "S1 + cascade Otsu -> multi-Otsu 3 only", "combine with A2",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3"], grow_if_raised=False)),
    ("S3", "S1", "S1 + keep all filtered blobs (no central selection)", "combine with A3",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3", "multiotsu4"], grow_if_raised=False,
          select="all", keep_ratio=0.0)),
    ("S4", "A1", "A1 with cascade Otsu -> multi-Otsu 3 only", "how much does multi-Otsu 4 add without adaptive?",
     dict(BEST16, centre_fallback=0.6, threshold=["otsu", "multiotsu3"])),
]


# Keep rule: keep when the CI of the change is >= 0 with more wins than losses, or when a later kept
# step needs it. Otherwise discard (the simpler config stays).
DECISIONS = {
    "T0": "baseline",
    "T1": "kept",
    "T2": "kept: worse alone, but the cascade (T5) needs a filter to tell whether a level found an egg",
    "T3": "kept",
    "T4": "discarded: below T3 (0.369 vs 0.401)",
    "T5": "kept",
    "T6": "kept: small gain, no losses",
    "T7": "discarded: smaller gain than T6",
    "T8": "discarded: CI crosses 0, wins = losses (more false blobs pass)",
    "T9": "discarded: also erodes egg outlines",
    "T10": "discarded: also cuts single eggs at the lit/shaded boundary",
    "T11": "kept: small gain, no losses, fewer false eggs",
    "T12": "kept",
    "T13": "discarded: same score as T12; GrabCut kept because it can grow (T16) and serves the fallback (T17)",
    "T14": "discarded: CI crosses 0; hurts dark backgrounds",
    "T15": "discarded: no effect",
    "T16": "kept",
    "T17": "kept (final before ablation; simplified to S1)",
    "T18": "discarded: majority vote pulls masks back to the lit half",
    "T19": "discarded (benchmark): cascade needed; but wins on 18 images -> T20/T21",
    "T20": "discarded: CI crosses 0; contrast cannot reliably judge",
    "T21": "discarded: more losses than wins",
    "T22": "not adopted: CI crosses 0 (7 wins / 5 losses); first thing to retry with more data",
    "A1": "ablation: adaptive step not needed once the centre fallback exists -> removed",
    "A2": "ablation: multi-Otsu 4 roughly neutral alone; see S2",
    "A3": "ablation: central selection worth ~0.004 and fewer false eggs -> kept",
    "A4": "ablation: 'grow only if raised' rule worth ~0.007 -> removed in S1 for simplicity",
    "A5": "ablation: GrabCut growth is needed (-0.026)",
    "A6": "ablation: border filter is essential (-0.113)",
    "S1": "kept = FINAL (simplest config that does not lose accuracy vs T17)",
    "S2": "discarded: multi-Otsu 4 is needed (-0.022)",
    "S3": "discarded: central selection kept (count accuracy 0.83 -> 0.81)",
    "S4": "discarded: same as S2",
}


def run_one(args):
    cv2.setNumThreads(1)
    stem, cfg = args
    union, insts = simple.segment(data.load_image(stem), cfg, return_instances=True)
    return {"image": stem, **metrics.all_metrics(union, data.load_gt(stem)),
            "n_pred": len(insts), "n_gt": len(data.load_gt_instances(stem))}


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stems = data.read_split("dev")
    cluster = pd.read_csv(EDA / "per_image.csv").set_index("image").img_cluster

    res = {}
    with Pool(args.workers) as pool:
        for tid, _, _, _, cfg in TRIALS:
            f = OUT / f"{tid}.csv"
            if f.exists() and tid not in args.rerun:
                res[tid] = pd.read_csv(f)
                continue
            df = pd.DataFrame(pool.map(run_one, [(s, cfg) for s in stems]))
            df.to_csv(f, index=False)
            res[tid] = df
            print(f"{tid}: mean IoU {df.iou.mean():.3f}")

    rows, by_cluster = [], {}
    for tid, parent, change, why, cfg in TRIALS:
        df = res[tid].set_index("image").loc[stems]
        r = dict(id=tid, parent=parent or "", change=change, why=why,
                 mean_iou=df.iou.mean(), median_iou=df.iou.median(), precision=df.precision.mean(),
                 recall=df.recall.mean(), count_acc=(df.n_pred == df.n_gt).mean())
        if parent:
            d = df.iou.values - res[parent].set_index("image").loc[stems].iou.values
            lo, hi = boot_ci(d)
            r.update(delta=d.mean(), ci_lo=lo, ci_hi=hi, wins=int((d > .02).sum()), losses=int((d < -.02).sum()))
        r["decision"] = DECISIONS.get(tid, "")
        rows.append(r)
        by_cluster[tid] = df.groupby(cluster.loc[stems].values).iou.mean()
    log = pd.DataFrame(rows)
    log.to_csv(OUT / "experiment_log.csv", index=False)
    pd.DataFrame(by_cluster).round(3).to_csv(OUT / "iou_by_cluster.csv")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(log[["id", "parent", "change", "mean_iou", "precision", "recall", "count_acc",
                   "delta", "ci_lo", "ci_hi", "wins", "losses"]].round(3).to_string(index=False))
        print(pd.DataFrame(by_cluster).round(3).to_string())


if __name__ == "__main__":
    main()
