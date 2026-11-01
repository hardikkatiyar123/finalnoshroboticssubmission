#!/usr/bin/env python3
"""
Task 1 - Egg segmentation with pretrained models.

Methods (prompt defaults to "egg"):
  gdino_sam3      Grounding DINO boxes  -> SAM 3 (box prompts, one mask per box)
  owlv2_sam3      OWLv2 boxes           -> SAM 3
  yoloworld_sam3  YOLO-World boxes      -> SAM 3
  wbf_sam3        Weighted Box Fusion of the three detectors' boxes -> SAM 3
  sam3_text       SAM 3 alone, text prompt -> all instance masks

Design notes
- Only one large model is in memory at a time: every detector runs over all
  images first, then SAM 3 is loaded once and segments every detector's boxes.
- Box prompts go to SAM 3's *tracker* (SAM-1-style, "segment the object in this
  box"). In SAM 3's concept mode a box acts as an exemplar ("find everything
  like this"), which is not what we want for detector boxes.
- Cleanup: score filter, NMS, a containment rule for group boxes, hole filling,
  largest component per instance, mask-level de-duplication.
- Re-prompting (box methods only): each cleaned mask is fed back to SAM 3 as its
  tight bounding box plus one positive point at the mask's most interior pixel.
  The refined mask replaces the first one unless they disagree badly
  (IoU < --refine-min-iou), which would mean the second pass jumped objects.

Outputs (under --out):
  <method>/masks/<stem>.png       union mask (0/255)  <- Task 1 ground truth
  <method>/instances/<stem>.npz   per-instance masks, boxes, scores
  <method>/overlays/<stem>.jpg    visual check
  <method>/results.json           per-image counts, scores, timing
  comparison/<stem>.jpg           original + every method side by side
  summary.csv                     per-image counts, areas, cross-method agreement
  review_these.txt                images where the methods disagree

Usage:
  python run_task1.py --images path/to/eggsegmentationdata --limit 3   # smoke test
  python run_task1.py --images path/to/eggsegmentationdata             # full run
  python run_task1.py --images ... --methods gdino_sam3 sam3_text      # subset
"""
import argparse
import csv
import gc
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALL_METHODS = ["gdino_sam3", "owlv2_sam3", "yoloworld_sam3", "wbf_sam3", "sam3_text"]
DETECTOR_OF = {"gdino_sam3": "gdino", "owlv2_sam3": "owlv2", "yoloworld_sam3": "yoloworld"}
ALL_DETECTORS = ["gdino", "owlv2", "yoloworld"]


# --------------------------------------------------------------------------- utils
def get_device(requested=None):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free():
    """Call after `del model` in the caller, so the memory is actually released."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def list_images(folder, limit=None):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)
    return files[:limit] if limit else files


def to_np(x):
    """Tensor or array -> numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu() if x.dtype.is_floating_point else x.detach().cpu()
        return x.numpy()
    return np.asarray(x)


# --------------------------------------------------------------------- box cleanup
def box_iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter, area_a, area_b


def nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        inter, a_i, a_rest = box_iou_matrix(boxes[i:i + 1], boxes[order[1:]])
        iou = inter[0] / (a_i[0] + a_rest - inter[0] + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def clean_boxes(boxes, scores, image_size, min_score, iou_thr, contain_frac=0.9):
    """Score filter -> clip -> NMS -> drop 'group' boxes that contain >= 2 other boxes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    w, h = image_size
    if len(boxes):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        valid = (scores >= min_score) & (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes, scores = boxes[valid], scores[valid]
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(boxes, scores, iou_thr)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) >= 3:
        inter, _, area_other = box_iou_matrix(boxes, boxes)
        inside = inter / (area_other[None, :] + 1e-9) >= contain_frac  # inside[i, j]: box j inside box i
        np.fill_diagonal(inside, False)
        group = inside.sum(axis=1) >= 2
        boxes, scores = boxes[~group], scores[~group]
    return boxes, scores


def fuse_boxes(per_detector, image_size, iou_thr, min_score):
    """Weighted Box Fusion over the (already cleaned) boxes of several detectors.

    WBF averages the coordinates of overlapping boxes, weighted by score. With
    conf_type="avg" a box found by only k of N detectors has its score scaled by
    k/N, so agreement between detectors is rewarded. Returns boxes, scores and,
    per fused box, how many detectors supported it.
    """
    from ensemble_boxes import weighted_boxes_fusion
    w, h = image_size
    scale = np.array([w, h, w, h], np.float32)
    boxes_list, scores_list, labels_list = [], [], []
    for b, s in per_detector:
        boxes_list.append((np.asarray(b, np.float32).reshape(-1, 4) / scale).clip(0, 1).tolist())
        scores_list.append(np.asarray(s, np.float32).tolist())
        labels_list.append([0] * len(s))
    if not any(boxes_list):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    fb, fs, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                      iou_thr=iou_thr, skip_box_thr=0.0, conf_type="avg")
    fb = np.asarray(fb, np.float32).reshape(-1, 4) * scale
    fs = np.asarray(fs, np.float32)
    fb, fs = clean_boxes(fb, fs, image_size, min_score, iou_thr)
    support = np.zeros(len(fb), int)
    for b, _ in per_detector:
        b = np.asarray(b, np.float32).reshape(-1, 4)
        if len(b) and len(fb):
            inter, a_f, a_d = box_iou_matrix(fb, b)
            iou = inter / (a_f[:, None] + a_d[None, :] - inter + 1e-9)
            support += (iou >= iou_thr).any(axis=1)
    return fb, fs, support


# -------------------------------------------------------------------- mask cleanup
def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def clean_masks(masks, scores, boxes, min_area_frac=0.001, dedupe_iou=0.7):
    """Fill holes, keep largest component per instance, drop tiny masks and duplicates."""
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 2:
        masks = masks[None]
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n, h, w = masks.shape if masks.size else (0, 0, 0)
    out_m, out_s, out_b = [], [], []
    for i in np.argsort(-scores) if n else []:
        m = ndimage.binary_fill_holes(masks[i])
        lab, k = ndimage.label(m)
        if k == 0:
            continue
        if k > 1:
            sizes = ndimage.sum(m, lab, range(1, k + 1))
            m = lab == (int(np.argmax(sizes)) + 1)
        if m.sum() < min_area_frac * h * w:
            continue
        if any(mask_iou(m, kept) > dedupe_iou for kept in out_m):
            continue
        out_m.append(m)
        out_s.append(scores[i])
        out_b.append(boxes[i] if i < len(boxes) else np.zeros(4, np.float32))
    if not out_m:
        return np.zeros((0, h, w), bool), np.zeros(0, np.float32), np.zeros((0, 4), np.float32)
    return np.stack(out_m), np.array(out_s, np.float32), np.stack(out_b)


def reprompt_from_masks(masks):
    """Tight box + most interior pixel (distance-transform peak) for each mask."""
    boxes, points = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        points.append([float(x), float(y)])
    return np.array(boxes, np.float32), np.array(points, np.float32)


def refine_masks(seg, image, masks, min_iou):
    """Second SAM pass from each mask's own box + interior point (see module docstring)."""
    if len(masks) == 0:
        return masks, []
    boxes, points = reprompt_from_masks(masks)
    new, _ = seg.segment(image, boxes, points)
    out, ious = [], []
    for m1, m2 in zip(masks, new):
        m2 = ndimage.binary_fill_holes(m2)
        lab, k = ndimage.label(m2)
        if k > 1:
            m2 = lab == (int(np.argmax(ndimage.sum(m2, lab, range(1, k + 1)))) + 1)
        iou = float(mask_iou(m1, m2)) if k else 0.0
        ious.append(round(iou, 4))
        out.append(m2 if iou >= min_iou else m1)
    return np.stack(out), ious


# ----------------------------------------------------------------------- detectors
class GroundingDINO:
    def __init__(self, device, prompt, model_id, box_thr, text_thr):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch, self.device = torch, device
        self.box_thr, self.text_thr = box_thr, text_thr
        text = prompt.lower().strip()
        self.text = text if text.endswith(".") else text + "."   # Grounding DINO expects "a. b."
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image):
        with self.torch.inference_mode():
            inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            kwargs = dict(text_threshold=self.text_thr, target_sizes=[image.size[::-1]])
            try:
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.box_thr, **kwargs)[0]
            except TypeError:  # older transformers name
                res = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=self.box_thr, **kwargs)[0]
        return to_np(res["boxes"]), to_np(res["scores"])


class OWLv2:
    def __init__(self, device, prompt, model_id, thr):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch, self.device, self.thr = torch, device, thr
        self.queries = [[prompt]]
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(text=self.queries, images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # OWLv2 pads the image to a square (bottom/right), so boxes are scaled to the
            # padded square; corners then line up with the original image.
            side = max(image.size)
            target = torch.tensor([[side, side]], device=self.device)
            # transformers >= 5 only exposes the grounded name on the processor.
            post = getattr(self.processor, "post_process_grounded_object_detection", None) \
                or self.processor.post_process_object_detection
            res = post(outputs, threshold=self.thr, target_sizes=target)[0]
        return to_np(res["boxes"]), to_np(res["scores"])


class YOLOWorldDet:
    def __init__(self, device, prompt, weights, thr):
        from ultralytics import YOLOWorld
        self.thr = thr
        self.device = 0 if device == "cuda" else device
        self.model = YOLOWorld(weights)
        self.model.set_classes([prompt])

    def detect(self, image):
        r = self.model.predict(image, conf=self.thr, device=self.device, verbose=False)[0]
        return to_np(r.boxes.xyxy), to_np(r.boxes.conf)


# --------------------------------------------------------------------------- SAM 3
class SAM3BoxSegmenter:
    """SAM 3 tracker: SAM-1-style promptable segmentation, one mask per input box."""

    def __init__(self, device, model_id):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        self.torch, self.device = torch, device
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self.model = Sam3TrackerModel.from_pretrained(model_id).to(device).eval()

    def segment(self, image, boxes, points=None):
        """One mask per box. `points` (optional, one positive point per box) sharpens the prompt."""
        torch = self.torch
        w, h = image.size
        if len(boxes) == 0:
            return np.zeros((0, h, w), bool), np.zeros(0, np.float32)
        prompts = dict(input_boxes=[boxes.tolist()])
        if points is not None:
            # nesting: image -> object -> points of that object -> (x, y) / label
            prompts["input_points"] = [[[p] for p in points.tolist()]]
            prompts["input_labels"] = [[[1] for _ in points]]
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt", **prompts).to(self.device)
            outputs = self.model(**inputs, multimask_output=True)
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(),
                                                      inputs["original_sizes"])[0]  # (N, K, H, W)
            iou = outputs.iou_scores[0].float().cpu()                               # (N, K)
            best = iou.argmax(-1)
            masks = masks[torch.arange(len(best)), best]
        return to_np(masks).astype(bool), to_np(iou.max(-1).values)


class SAM3Text:
    """SAM 3 concept segmentation: text prompt -> every matching instance."""

    def __init__(self, device, model_id, prompt, thr, mask_thr=0.5):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.device = torch, device
        self.prompt, self.thr, self.mask_thr = prompt, thr, mask_thr
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()

    def segment(self, image):
        with self.torch.inference_mode():
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.thr, mask_threshold=self.mask_thr,
                target_sizes=inputs.get("original_sizes").tolist())[0]
        return to_np(res["masks"]).astype(bool), to_np(res["scores"]), to_np(res["boxes"])


# ----------------------------------------------------------------------- outputs
PALETTE = [(60, 60, 255), (60, 200, 60), (255, 140, 40), (200, 60, 200),
           (40, 220, 220), (255, 220, 60), (140, 80, 255), (80, 160, 255)]


def draw_overlay(image_rgb, masks, scores, input_boxes=None, alpha=0.5):
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for i, m in enumerate(masks):
        color = np.array(PALETTE[i % len(PALETTE)], np.uint8)
        out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 2)
        ys, xs = np.nonzero(m)
        cv2.putText(out, f"{scores[i]:.2f}", (int(xs.min()), max(int(ys.min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    if input_boxes is not None:
        for b in input_boxes:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return out


def save_result(out_dir, method, stem, image_rgb, masks, scores, boxes, extra, input_boxes=None):
    d = Path(out_dir) / method
    for sub in ("masks", "instances", "overlays"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    union = masks.any(0) if len(masks) else np.zeros((h, w), bool)
    cv2.imwrite(str(d / "masks" / f"{stem}.png"), union.astype(np.uint8) * 255)
    np.savez_compressed(d / "instances" / f"{stem}.npz", masks=masks, scores=scores, boxes=boxes)
    cv2.imwrite(str(d / "overlays" / f"{stem}.jpg"),
                draw_overlay(image_rgb, masks, scores, input_boxes))
    return {"n_instances": int(len(masks)), "scores": [round(float(s), 4) for s in scores],
            "area_frac": round(float(union.mean()), 5), **extra}


def tile(img_bgr, label, height=320):
    h, w = img_bgr.shape[:2]
    t = cv2.resize(img_bgr, (max(1, int(w * height / h)), height))
    cv2.rectangle(t, (0, 0), (t.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(t, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def compare(out_dir, images, methods, results, agree_iou):
    out_dir = Path(out_dir)
    (out_dir / "comparison").mkdir(parents=True, exist_ok=True)
    rows, review = [], []
    for path in images:
        stem = path.stem
        img = cv2.imread(str(path))
        tiles = [tile(img, "original")]
        unions, counts = {}, {}
        for m in methods:
            ov, mk = out_dir / m / "overlays" / f"{stem}.jpg", out_dir / m / "masks" / f"{stem}.png"
            if not ov.exists():
                continue
            counts[m] = results[m][stem]["n_instances"]
            unions[m] = cv2.imread(str(mk), cv2.IMREAD_GRAYSCALE) > 0
            tiles.append(tile(cv2.imread(str(ov)), f"{m}  n={counts[m]}"))
        cv2.imwrite(str(out_dir / "comparison" / f"{stem}.jpg"), np.hstack(tiles))

        ious = {f"iou_{a}_vs_{b}": round(float(mask_iou(unions[a], unions[b])), 4)
                for a, b in itertools.combinations(unions, 2)}
        min_iou = min(ious.values()) if ious else 1.0
        count_agree = len(set(counts.values())) <= 1
        row = {"image": path.name, **{f"n_{m}": counts.get(m) for m in methods},
               **{f"area_{m}": results[m][stem]["area_frac"] for m in methods if m in counts},
               **ious, "min_pairwise_iou": round(min_iou, 4), "count_agree": count_agree}
        rows.append(row)
        if not count_agree or min_iou < agree_iou or 0 in counts.values():
            review.append(f"{path.name}\tcounts={counts}\tmin_iou={min_iou:.3f}")

    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_dir / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    (out_dir / "review_these.txt").write_text("\n".join(review) + ("\n" if review else ""))
    return rows, review


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/task1")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--prompt", default="egg")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    # models
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--owl-model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--yolo-weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--sam3-model", default="facebook/sam3")
    # thresholds
    ap.add_argument("--gdino-box-thr", type=float, default=0.35)
    ap.add_argument("--gdino-text-thr", type=float, default=0.25)
    ap.add_argument("--owl-thr", type=float, default=0.2)
    ap.add_argument("--yolo-thr", type=float, default=0.1)
    ap.add_argument("--sam3-thr", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--wbf-iou", type=float, default=0.55, help="IoU for clustering boxes in WBF")
    ap.add_argument("--wbf-min-score", type=float, default=0.15,
                    help="min fused score; a single-detector box has its score divided by 3")
    ap.add_argument("--no-refine", action="store_true", help="skip the re-prompting pass")
    ap.add_argument("--refine-min-iou", type=float, default=0.5,
                    help="keep the first-pass mask if the refined one overlaps it less than this")
    ap.add_argument("--agree-iou", type=float, default=0.85,
                    help="flag images whose union masks differ more than this between methods")
    args = ap.parse_args()

    device = get_device(args.device)
    images = list_images(args.images, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    print(f"{len(images)} images | device={device} | prompt='{args.prompt}' | refine={not args.no_refine}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = {p.stem: Image.open(p).convert("RGB") for p in images}
    results = {m: {} for m in args.methods}

    # 1) detectors: one at a time, boxes kept in memory. WBF needs all three.
    needed = {DETECTOR_OF[m] for m in args.methods if m in DETECTOR_OF}
    if "wbf_sam3" in args.methods:
        needed |= set(ALL_DETECTORS)
    det_by_name = {}
    builders = {
        "gdino": lambda: GroundingDINO(device, args.prompt, args.gdino_model,
                                       args.gdino_box_thr, args.gdino_text_thr),
        "owlv2": lambda: OWLv2(device, args.prompt, args.owl_model, args.owl_thr),
        "yoloworld": lambda: YOLOWorldDet(device, args.prompt, args.yolo_weights, args.yolo_thr),
    }
    min_scores = {"gdino": args.gdino_box_thr, "owlv2": args.owl_thr, "yoloworld": args.yolo_thr}
    for name in [d for d in ALL_DETECTORS if d in needed]:
        print(f"\n[{name}] loading")
        det = builders[name]()
        det_by_name[name] = {}
        for p in images:
            t0 = time.perf_counter()
            raw_b, raw_s = det.detect(pil[p.stem])
            b, s = clean_boxes(raw_b, raw_s, pil[p.stem].size, min_scores[name], args.nms_iou)
            det_by_name[name][p.stem] = {"boxes": b, "scores": s, "n_raw": int(len(raw_b)),
                                         "det_time_s": time.perf_counter() - t0}
        n = [len(v["boxes"]) for v in det_by_name[name].values()]
        print(f"[{name}] boxes/image after cleanup: mean={np.mean(n):.2f}, zero-box images={n.count(0)}")
        del det
        free()

    det_boxes = {m: det_by_name[DETECTOR_OF[m]] for m in args.methods if m in DETECTOR_OF}
    if "wbf_sam3" in args.methods:
        det_boxes["wbf_sam3"] = {}
        for p in images:
            t0 = time.perf_counter()
            per = [(det_by_name[d][p.stem]["boxes"], det_by_name[d][p.stem]["scores"]) for d in ALL_DETECTORS]
            b, s, support = fuse_boxes(per, pil[p.stem].size, args.wbf_iou, args.wbf_min_score)
            det_boxes["wbf_sam3"][p.stem] = {
                "boxes": b, "scores": s, "n_raw": int(sum(len(x[0]) for x in per)),
                "det_time_s": sum(det_by_name[d][p.stem]["det_time_s"] for d in ALL_DETECTORS)
                + time.perf_counter() - t0,
                "support": support.tolist()}
        n = [len(v["boxes"]) for v in det_boxes["wbf_sam3"].values()]
        print(f"[wbf] boxes/image after fusion: mean={np.mean(n):.2f}, zero-box images={n.count(0)}")

    # 2) SAM 3 on every detector's boxes (+ re-prompting pass)
    if det_boxes:
        print("\n[sam3 tracker] loading")
        seg = SAM3BoxSegmenter(device, args.sam3_model)
        for method, per_img in det_boxes.items():
            n_rejected = 0
            for p in images:
                d = per_img[p.stem]
                t0 = time.perf_counter()
                masks, sam_iou = seg.segment(pil[p.stem], d["boxes"])
                masks, scores, boxes = clean_masks(masks, d["scores"], d["boxes"])
                extra = {"n_raw_boxes": d["n_raw"], "det_time_s": round(d["det_time_s"], 3),
                         "sam_pred_iou": [round(float(x), 4) for x in sam_iou]}
                if "support" in d:
                    extra["wbf_support"] = d["support"]
                if not args.no_refine:
                    masks, refine_iou = refine_masks(seg, pil[p.stem], masks, args.refine_min_iou)
                    masks, scores, boxes = clean_masks(masks, scores, boxes)
                    extra["refine_iou"] = refine_iou
                    n_rejected += sum(i < args.refine_min_iou for i in refine_iou)
                extra["seg_time_s"] = round(time.perf_counter() - t0, 3)
                results[method][p.stem] = save_result(out_dir, method, p.stem, np.array(pil[p.stem]),
                                                      masks, scores, boxes, extra, input_boxes=d["boxes"])
            msg = f", refined masks rejected={n_rejected}" if not args.no_refine else ""
            print(f"[{method}] done{msg}")
        del seg
        free()

    # 3) SAM 3 alone with the text prompt
    if "sam3_text" in args.methods:
        print("\n[sam3 text] loading")
        seg = SAM3Text(device, args.sam3_model, args.prompt, args.sam3_thr)
        for p in images:
            t0 = time.perf_counter()
            masks, scores, boxes = seg.segment(pil[p.stem])
            masks, scores, boxes = clean_masks(masks, scores, boxes)
            extra = {"seg_time_s": round(time.perf_counter() - t0, 3)}
            results["sam3_text"][p.stem] = save_result(out_dir, "sam3_text", p.stem,
                                                       np.array(pil[p.stem]), masks, scores, boxes, extra)
        print("[sam3_text] done")
        del seg
        free()

    for m in args.methods:
        (out_dir / m / "results.json").write_text(json.dumps(results[m], indent=2))

    # 4) side-by-side comparison + agreement summary
    rows, review = compare(out_dir, images, args.methods, results, args.agree_iou)
    print("\n=== summary ===")
    for m in args.methods:
        r = list(results[m].values())
        n = [x["n_instances"] for x in r]
        t = [x.get("det_time_s", 0) + x["seg_time_s"] for x in r]
        print(f"{m:16s} mean instances={np.mean(n):.2f}  zero-mask images={n.count(0)}  "
              f"time/image={np.mean(t):.3f}s")
    print(f"{len(review)}/{len(rows)} images flagged for review -> {out_dir / 'review_these.txt'}")
    print(f"Side-by-side views -> {out_dir / 'comparison'}")


if __name__ == "__main__":
    main()
