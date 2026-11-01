"""
Multi-egg tracker: one constant-velocity Kalman filter per egg + Hungarian matching.

Each frame:
  1. predict: every track's Kalman filter predicts where its egg is now
  2. match:   predicted positions vs detected egg centroids, Hungarian algorithm on distance,
              pairs further apart than a gate (a multiple of the egg's radius) are not matched
  3. update:  matched tracks correct their filter with the measured centroid
  4. manage:  unmatched detections start new (tentative) tracks; a track is confirmed and gets
              an ID after `confirm` hits; a track that is not seen for `max_misses` frames is dropped
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, det, q=1.0, r=4.0):
        kf = cv2.KalmanFilter(4, 2)                  # state x, y, vx, vy; measurement x, y
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.diag([q, q, 4 * q, 4 * q]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
        kf.statePost = np.array([[det["cx"]], [det["cy"]], [0], [0]], np.float32)
        self.kf = kf
        self.id = None                               # assigned when confirmed
        self.hits, self.misses = 1, 0
        self.radius = det["radius"]
        self.det = det                               # last matched detection (mask etc.)
        self.trail = [(det["cx"], det["cy"])]

    def predict(self):
        if self.misses:                              # coasting: damp the velocity so it does not drift away
            self.kf.statePost[2:] *= 0.7
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def update(self, det):
        self.kf.correct(np.array([[det["cx"]], [det["cy"]]], np.float32))
        self.hits += 1
        self.misses = 0
        self.radius = 0.8 * self.radius + 0.2 * det["radius"]
        self.det = det
        self.trail = (self.trail + [(det["cx"], det["cy"])])[-40:]

    @property
    def pos(self):
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])


class Tracker:
    def __init__(self, gate=2.0, confirm=3, max_misses=36):
        self.gate, self.confirm, self.max_misses = gate, confirm, max_misses
        self.tracks, self.next_id = [], 1

    def step(self, dets):
        preds = [t.predict() for t in self.tracks]
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            cost = np.array([[np.hypot(px - d["cx"], py - d["cy"]) for d in dets] for px, py in preds])
            for ti, di in zip(*linear_sum_assignment(cost)):
                if cost[ti, di] <= self.gate * max(self.tracks[ti].radius, 8):
                    self.tracks[ti].update(dets[di])
                    matched_t.add(ti)
                    matched_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.misses += 1
                t.det = None
        self.tracks = [t for t in self.tracks if t.misses <= (self.max_misses if t.id else 1)]
        for di, d in enumerate(dets):
            if di not in matched_d:
                self.tracks.append(Track(d))
        for t in self.tracks:
            if t.id is None and t.hits >= self.confirm:
                t.id = self.next_id
                self.next_id += 1
        return [t for t in self.tracks if t.id is not None]


def detections(masks):
    """Egg masks -> detections with centroid and equivalent radius."""
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        out.append(dict(mask=m, cx=float(xs.mean()), cy=float(ys.mean()), radius=float(np.sqrt(len(xs) / np.pi))))
    return out
