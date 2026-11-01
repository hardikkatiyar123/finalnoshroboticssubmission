"""Binary-mask metrics. Both masks empty counts as a perfect match."""
import numpy as np


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return int(tp), int(fp), int(fn)


def iou(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    return 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)


def dice(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def precision(pred, gt):
    tp, fp, _ = confusion(pred, gt)
    return 1.0 if tp + fp == 0 else tp / (tp + fp)


def recall(pred, gt):
    tp, _, fn = confusion(pred, gt)
    return 1.0 if tp + fn == 0 else tp / (tp + fn)


def all_metrics(pred, gt):
    return {"iou": iou(pred, gt), "dice": dice(pred, gt),
            "precision": precision(pred, gt), "recall": recall(pred, gt)}
