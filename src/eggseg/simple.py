"""
Task 2.1 classical egg segmentation, built from standard steps:

  threshold (Otsu / multi-Otsu / k-means)  ->  morphological clean-up  ->
  blob filtering (size, solidity, border, centre)  ->  optional refinement
  (GrabCut or marker-based watershed)  ->  optional majority vote.

Every step is switched on by a key in the config dict, so each experiment trial is a
small config change. Works at WORK x WORK and returns masks at the input size.
"""
import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

WORK = 320

BASE = dict(
    channel="gray",        # gray | V | L
    blur=0,                # Gaussian blur kernel (0 = off)
    threshold="otsu",      # otsu | multiotsu3 | kmeans3
    open_r=0,              # morphological opening radius (0 = off)
    fill_holes=False,
    min_area=0.0,          # blob filters, area as % of image
    max_area=100.0,
    min_solidity=0.0,
    drop_border=False,     # drop blobs touching the image border
    max_saturation=255,    # drop blobs whose mean HSV saturation is above this (eggs are colourless)
    all_levels=False,      # collect blobs from every cascade level instead of stopping at the first
    min_circularity=0.0,   # drop refined eggs whose circularity 4*pi*area/perimeter^2 is below this
    select="all",          # all | largest | central
    keep_ratio=0.0,        # also keep blobs with area >= keep_ratio x the selected one (multi-egg)
    split=False,           # split fused blobs at narrow necks (distance transform + watershed)
    centre_fallback=0.0,   # if no blob is found: GrabCut from a central box of this size (fraction of side)
    arbiter="none",        # none | contrast | solidity: also run centre GrabCut and keep the better result
    refine="none",         # none | grabcut | watershed | vote
    grabcut_band=0.0,      # "probably egg" band around the blob, x equivalent radius
    grow_if_raised=False,  # use grabcut_band only when the cascade went past the first threshold
    ws_band=0.6,           # watershed: background marker distance, x equivalent radius
)


# Frozen final algorithm: trial S1 in scripts/task21_experiments.py (T17 simplified by ablation:
# no adaptive cascade step, GrabCut always allowed to grow)
FINAL = dict(threshold=["otsu", "multiotsu3", "multiotsu4"], open_r=2, fill_holes=True,
             min_area=1.0, max_area=40.0, min_solidity=0.85, drop_border=True,
             select="central", keep_ratio=0.5,
             refine="grabcut", grabcut_band=0.5,
             centre_fallback=0.6)

# Task 4 (video): FINAL plus three changes, see report/task4_report.md
#  - all_levels: two eggs can pass at different threshold levels in one frame
#  - max_saturation: rejects skin and coloured logos (eggs have mean saturation < 10)
#  - no centre fallback: video frames may contain no egg at all
#  - min_circularity: rejects bright shirt/wall patches (egg masks 0.76-0.79, patches <= 0.74)
VIDEO = {**FINAL, "select": "all", "keep_ratio": 0.0, "all_levels": True, "max_saturation": 40,
         "centre_fallback": 0.0, "min_circularity": 0.75}


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def get_channel(img, name):
    if name == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if name == "V":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    if name == "L":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    raise ValueError(name)


def threshold(ch, img, method):
    """Bright foreground mask."""
    if method == "otsu":
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m > 0
    if method.startswith("multiotsu"):    # e.g. 3 classes: dark bg / bright bg / egg; keep the brightest
        t = threshold_multiotsu(ch, classes=int(method[-1]))
        return ch > t[-1]
    if method == "adaptive":              # brighter than the local (Gaussian) mean of a large window
        return cv2.adaptiveThreshold(ch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, -10) > 0
    if method == "tophat":                # remove large bright background, then Otsu on what is left
        th = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, disk(60))
        _, m = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return m > 0
    if method == "kmeans3":               # 3 colour clusters in Lab; keep the brightest
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(0)
        _, lab_k, centres = cv2.kmeans(lab, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        return (lab_k.ravel() == int(np.argmax(centres[:, 0]))).reshape(ch.shape)
    raise ValueError(method)


def fill(mask):
    """Fill holes: flood the background from the border, everything not reached is foreground."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    cv2.floodFill(pad, None, (0, 0), 2)
    return pad[1:-1, 1:-1] != 2


def split_blobs(mask, min_peak=0.35):
    """Classic touching-object split: peaks of the distance transform become markers and a
    watershed on the inverted distance cuts the blob at its narrow necks."""
    m = mask.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return mask
    # markers: regions within (1 - min_peak) of a local maximum of the distance
    local_max = dist >= cv2.dilate(dist, disk(7)) - 1e-3
    peaks = local_max & (dist > min_peak * dist.max()) & (dist > 4)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if n <= 2:
        return mask
    markers = markers + 1
    markers[m == 0] = 0
    dist_img = cv2.cvtColor(cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(dist_img, markers)
    out = (markers > 1) & mask.astype(bool)         # watershed lines (-1) become cuts
    return out


def blobs(mask, sat=None):
    """Connected components with simple region properties (sat: optional saturation channel)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        m = lab == i
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        x, y, bw, bh = st[i, :4]
        out.append(dict(mask=m, area=st[i, cv2.CC_STAT_AREA] / m.size * 100,
                        solidity=cv2.contourArea(c) / max(hull, 1),
                        border=bool(x == 0 or y == 0 or x + bw >= w or y + bh >= h),
                        centre_dist=float(np.hypot(cen[i, 0] / w - .5, cen[i, 1] / h - .5) / np.hypot(.5, .5)),
                        sat=float(sat[m].mean()) if sat is not None else 0.0))
    return out


def select(bl, cfg):
    bl = [b for b in bl if cfg["min_area"] <= b["area"] <= cfg["max_area"]
          and b["solidity"] >= cfg["min_solidity"] and not (cfg["drop_border"] and b["border"])
          and b["sat"] <= cfg["max_saturation"]]
    if not bl or cfg["select"] == "all":
        return bl
    key = (lambda b: b["area"]) if cfg["select"] == "largest" else (lambda b: -b["centre_dist"])
    best = max(bl, key=key)
    keep = [best]
    if cfg["keep_ratio"] > 0:
        keep += [b for b in bl if b is not best and b["area"] >= cfg["keep_ratio"] * best["area"]]
    return keep


def circularity(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    return 4 * np.pi * cv2.contourArea(c) / max(cv2.arcLength(c, True), 1) ** 2


def eq_radius(m):
    return np.sqrt(m.sum() / np.pi)


def grabcut(img, m, band):
    """GrabCut seeded from the blob: eroded blob = sure egg, blob (+ band) = probably egg."""
    r = eq_radius(m)
    sure = cv2.erode(m.astype(np.uint8), disk(max(1, int(.3 * r)))) > 0
    prob = cv2.dilate(m.astype(np.uint8), disk(int(band * r))) > 0 if band > 0 else m
    if sure.sum() < 20:
        return m
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[prob] = cv2.GC_PR_FGD
    gc[sure] = cv2.GC_FGD
    # pixels just outside: probably background, so GrabCut may still claim them
    near = (cv2.dilate(m.astype(np.uint8), disk(int((band + .5) * r))) > 0) & ~prob
    gc[near] = cv2.GC_PR_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)              # GrabCut seeds its colour model with random k-means
        cv2.grabCut(img, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return largest_overlapping(out, sure)


def grabcut_centre(img, frac):
    """Plain rectangle-initialised GrabCut on a central box; keeps the part nearest the centre."""
    h, w = img.shape[:2]
    x0, y0 = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(img, gc, (x0, y0, int(w * frac), int(h * frac)), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    m = fill(cv2.morphologyEx(((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8),
                              cv2.MORPH_OPEN, disk(2)) > 0)
    bl = [b for b in blobs(m) if b["area"] >= 1.0]
    return [min(bl, key=lambda b: b["centre_dist"])["mask"]] if bl else []


def watershed(img, m, band):
    """Marker-based watershed on the colour gradient: inner marker = eroded blob,
    outer marker = ring at band x radius outside the blob."""
    r = eq_radius(m)
    inner = cv2.erode(m.astype(np.uint8), disk(max(1, int(.3 * r)))) > 0
    if inner.sum() < 20:
        return m
    outer = ~(cv2.dilate(m.astype(np.uint8), disk(max(2, int(band * r)))) > 0)
    markers = np.zeros(m.shape, np.int32)
    markers[outer] = 1
    markers[inner] = 2
    cv2.watershed(cv2.GaussianBlur(img, (5, 5), 0), markers)
    return fill(markers == 2)


def largest_overlapping(mask, seed):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    ids = np.bincount(lab[seed], minlength=n)
    ids[0] = 0
    return lab == int(np.argmax(ids)) if ids.max() > 0 else mask


def segment(img_bgr, cfg=None, return_instances=False):
    cfg = {**BASE, **(cfg or {})}
    H, W = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (WORK, WORK), interpolation=cv2.INTER_AREA)
    ch = get_channel(img, cfg["channel"])
    if cfg["blur"]:
        ch = cv2.GaussianBlur(ch, (cfg["blur"], cfg["blur"]), 0)
    # a list of thresholds is a cascade: use the first one that yields an egg-like blob
    methods = cfg["threshold"] if isinstance(cfg["threshold"], list) else [cfg["threshold"]]
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1] if cfg["max_saturation"] < 255 else None
    eggs, taken = [], np.zeros((WORK, WORK), bool)
    for method in methods:
        m = threshold(ch, img, method)
        if cfg["open_r"]:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(cfg["open_r"])) > 0
        if cfg["fill_holes"]:
            m = fill(m)
        if cfg["split"]:
            m = split_blobs(m)
        found = [b["mask"] for b in select(blobs(m, sat), cfg)]
        if not cfg["all_levels"]:
            eggs = found
            if eggs:
                break
            continue
        for e in found:                    # keep blobs that do not overlap one from an earlier level
            if not (e & taken).any():
                eggs.append(e)
                taken |= e
    if not eggs and cfg["centre_fallback"]:
        return finish(grabcut_centre(img, cfg["centre_fallback"]), H, W, return_instances)

    # a blob found at a raised threshold is only the lit part of the egg, so let it grow
    band = cfg["grabcut_band"] if not cfg["grow_if_raised"] or method != methods[0] else 0.0
    out = []
    for e in eggs:
        if cfg["refine"] == "grabcut":
            e = grabcut(img, e, band)
        elif cfg["refine"] == "watershed":
            e = watershed(img, e, cfg["ws_band"])
        elif cfg["refine"] == "vote":     # majority of: blob, GrabCut, watershed
            votes = e.astype(int) + grabcut(img, e, band) + watershed(img, e, cfg["ws_band"])
            e = largest_overlapping(fill(votes >= 2), e)
        if cfg["min_circularity"]:         # measured at the true aspect ratio, not the square work size
            true_shape = cv2.resize(e.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            if circularity(true_shape) < cfg["min_circularity"]:
                continue
        out.append(e)
    if cfg["arbiter"] != "none":
        alt = grabcut_centre(img, cfg["centre_fallback"] or 0.6)
        if alt and judge(img, alt, cfg["arbiter"]) > judge(img, out, cfg["arbiter"]):
            out = alt
    return finish(out, H, W, return_instances)


def judge(img, eggs, how):
    """Score a candidate result without ground truth (higher = more egg-like)."""
    m = np.any(eggs, 0) if len(eggs) else np.zeros(img.shape[:2], bool)
    if m.sum() < 20:
        return -np.inf
    if how == "solidity":
        return float(np.mean([b["solidity"] for b in blobs(m)]))
    # contrast: mean brightness inside minus the ring just outside (EDA E)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    ring = (cv2.dilate(m.astype(np.uint8), disk(max(3, int(.3 * eq_radius(m))))) > 0) & ~m
    return float(gray[m].mean() - gray[ring].mean()) if ring.any() else -np.inf


def finish(out, H, W, return_instances):
    # resolve overlaps between refined eggs: earlier (higher-priority) egg wins
    taken = np.zeros((WORK, WORK), bool)
    final = []
    for e in out:
        e = e & ~taken
        if e.sum() > 20:
            final.append(e)
            taken |= e
    up = [cv2.resize(e.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for e in final]
    union = np.any(up, 0) if up else np.zeros((H, W), bool)
    return (union, up) if return_instances else union
