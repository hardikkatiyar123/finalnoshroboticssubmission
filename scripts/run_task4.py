#!/usr/bin/env python3
"""
Task 4: track individual eggs in the YouTube video with the classical segmenter
(eggseg.simple.VIDEO) + a Kalman-filter tracker (eggseg.track).

Pass 1 segments every frame (parallel, cached in outputs/task4/detections.npz).
Pass 2 tracks and renders outputs/task4/tracked.mp4 plus tracks.csv and summary.json.

Usage:
  python scripts/run_task4.py --video data/video/eggs.mp4
  python scripts/run_task4.py --video data/video/eggs.mp4 --redetect   # ignore the cache
  python scripts/run_task4.py --config final --out outputs/task4_baseline  # unmodified Task 2.1 model
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg import data, simple, track  # noqa: E402

CONFIGS = {"video": simple.VIDEO, "final": simple.FINAL}
PROC = (640, 360)          # processing size; output video is rendered at this size x 1.5
COLORS = [(60, 60, 255), (60, 200, 60), (255, 140, 40), (200, 60, 200), (0, 200, 255), (255, 255, 0)]


def detect_range(args):
    """Segment frames [a, b) of the video; returns a label image per frame (0 = background)."""
    path, a, b, cfg = args
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    labels = []
    for _ in range(a, b):
        ok, f = cap.read()
        lab = np.zeros(PROC[::-1], np.uint8)
        if ok:
            _, insts = simple.segment(cv2.resize(f, PROC, interpolation=cv2.INTER_AREA), CONFIGS[cfg],
                                      return_instances=True)
            for k, m in enumerate(insts, 1):
                lab[m & (lab == 0)] = k
        labels.append(lab)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(data.ROOT / "data" / "video" / "eggs.mp4"))
    ap.add_argument("--redetect", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="video", choices=list(CONFIGS))
    ap.add_argument("--out", default=str(data.ROOT / "outputs" / "task4"))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)

    cache = OUT / "detections.npz"
    if cache.exists() and not args.redetect:
        labels = np.load(cache)["labels"]
    else:
        step = int(np.ceil(n / (args.workers * 4)))
        chunks = [(args.video, a, min(n, a + step), args.config) for a in range(0, n, step)]
        with Pool(args.workers) as pool:
            labels = np.array([lab for part in pool.map(detect_range, chunks) for lab in part])
        np.savez_compressed(cache, labels=labels)
    print(f"{n} frames, {fps:.0f} fps; frames with >= 1 detection: {(labels.reshape(n, -1).max(1) > 0).sum()}")

    tracker = track.Tracker()
    rows = []
    out_size = (int(PROC[0] * 1.5), int(PROC[1] * 1.5))
    s = out_size[0] / PROC[0]
    writer = cv2.VideoWriter(str(OUT / "tracked.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, out_size)  # H.264
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(OUT / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        lab = labels[i]
        dets = track.detections([lab == k for k in range(1, lab.max() + 1)])
        live = tracker.step(dets)
        frame = cv2.resize(f, out_size, interpolation=cv2.INTER_AREA)
        for t in live:
            c = COLORS[(t.id - 1) % len(COLORS)]
            if t.det is not None:                      # seen in this frame: draw its mask
                m = cv2.resize(t.det["mask"].astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST) > 0
                frame[m] = (frame[m] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame, cs, -1, c, 2)
                label = f"egg {t.id}"
            elif not dets:                             # no egg anywhere (scene change, title card): draw nothing
                continue
            else:                                      # not seen: show the Kalman prediction
                cx, cy = t.pos
                cv2.circle(frame, (int(cx * s), int(cy * s)), int(t.radius * s), c, 1, cv2.LINE_AA)
                label = f"egg {t.id} (predicted)"
            pts = np.array([(int(x * s), int(y * s)) for x, y in t.trail], np.int32)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, c, 1, cv2.LINE_AA)
            cx, cy = t.pos
            cv2.putText(frame, label, (int(cx * s) - 30, int((cy - t.radius) * s) - 8), 0, .6, (0, 0, 0), 4,
                        cv2.LINE_AA)
            cv2.putText(frame, label, (int(cx * s) - 30, int((cy - t.radius) * s) - 8), 0, .6, c, 2, cv2.LINE_AA)
            rows.append(dict(frame=i, time_s=i / fps, track_id=t.id, cx=cx, cy=cy, seen=t.det is not None))
        cv2.rectangle(frame, (0, 0), (330, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"t={i / fps:5.1f}s  eggs tracked: {len(live)}", (8, 20), 0, .6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tracks.csv", index=False)
    per_id = df.groupby("track_id").agg(first_s=("time_s", "min"), last_s=("time_s", "max"),
                                        frames=("frame", "size"), seen=("seen", "sum")).round(2)
    det_per_frame = labels.reshape(n, -1).max(1)
    summ = {"config": args.config, "frames": n, "fps": fps, "frames_with_detection": int((det_per_frame > 0).sum()),
            "track_ids": int(df.track_id.nunique()) if len(df) else 0,
            "per_track": per_id.reset_index().to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(per_id.to_string())
    print("->", OUT / "tracked.mp4")


if __name__ == "__main__":
    main()
