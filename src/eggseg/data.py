"""Paths and loaders shared by the Task 2/3 scripts."""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "images"
GT_MASKS = ROOT / "data" / "gt_masks"
SCENES = ROOT / "data" / "scene_tags.csv"
SPLITS = ROOT / "splits"


def read_split(name):
    """Image stems in splits/<name>.txt."""
    return [s for s in (SPLITS / f"{name}.txt").read_text().split() if s]


def load_image(stem):
    """BGR uint8."""
    return cv2.imread(str(IMAGES / f"{stem}.jpg"))


def load_gt(stem):
    """Union GT mask, bool."""
    return cv2.imread(str(GT_MASKS / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0


def load_gt_instances(stem):
    """Per-egg GT masks (N, H, W), copied by make_gt.py from the Task 1 method in gt_masks/sources.csv."""
    return np.load(GT_MASKS / "instances" / f"{stem}.npz")["masks"].astype(bool)


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def gt_sources():
    return {r["image"]: r["source"] for r in _csv(GT_MASKS / "sources.csv")}


def scenes():
    return {r["image"]: r["scene"] for r in _csv(SCENES)}
