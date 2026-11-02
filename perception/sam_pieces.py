"""Precise piece detection with SAM 2.1 (ultralytics), prompted by orange/blue seed points.

Position: for a vertical piece seen obliquely from above, every point at height z is pushed away from the camera
nadir when mapped onto the desk plane. The mask point nearest the nadir is therefore on the BASE circle, which
lies on the desk plane the homography was fitted on. base centre = that point + radius * (direction away from nadir).
This removes the height-dependent parallax that made grasps miss.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import topcam

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "sam2.1_b.pt"
NADIR_XY = np.array([-90.0, 33.0])       # camera position seen from above (rough; only its direction matters)
RADIUS = {0: 13.0, 1: 16.0, 2: 21.0}     # base radius per size (mm); refine after measuring the pieces
_sam = None


def sam():
    global _sam
    if _sam is None:
        from ultralytics import SAM
        _sam = SAM(str(MODEL))
    return _sam


def detect(frame, cam: topcam.BoardCam, color=0, region=((80.0, 345.0), (-300.0, -100.0)), counts=None,
           debug_path=None):
    """Off-board pieces of `color`: [{"x", "y", "size", "area", "px"}]. counts (S, M, L) assigns sizes by area rank."""
    seeds = topcam.off_board_pieces(frame, cam, color, region=region)
    if not seeds:
        return []
    pts = [list(p["px"]) for p in seeds]
    res = sam()(frame, points=[[p] for p in pts], labels=[[1] for _ in pts], verbose=False)
    masks = res[0].masks.data.cpu().numpy() if res[0].masks is not None else []
    out = []
    for m, seed in zip(masks, seeds):
        m = (m > 0.5).astype(np.uint8)
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnt = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if not cnt:
            continue
        c = max(cnt, key=cv2.contourArea)
        w = cam.to_world(c.reshape(-1, 2).astype(np.float64))
        area = abs(cv2.contourArea(w.astype(np.float32)))
        if area > 9000:                       # SAM grabbed the desk / several things: skip this seed
            continue
        near = w[np.argmin(np.linalg.norm(w - NADIR_XY, axis=1))]
        out.append({"near": near, "area": area, "px": seed["px"], "contour": c})
    # one piece can get several seeds: keep one mask per piece (nearest-to-camera base points within 15 mm)
    uniq = []
    for p in sorted(out, key=lambda p: -p["area"]):
        if all(np.linalg.norm(p["near"] - u["near"]) > 25 for u in uniq):
            uniq.append(p)
    out = uniq
    # sizes by mask area rank (largest silhouettes = large pieces)
    out.sort(key=lambda p: p["area"])
    if counts is not None and sum(counts) == len(out):
        sizes = [s for s in (0, 1, 2) for _ in range(counts[s])]
    else:
        sizes = [int(np.clip(np.searchsorted([700, 1400], p["area"]), 0, 2)) for p in out]
    pieces = []
    for p, s in zip(out, sizes):
        u = p["near"] - NADIR_XY
        u /= np.linalg.norm(u)
        base = p["near"] + u * RADIUS[s]
        pieces.append({"x": round(float(base[0]), 1), "y": round(float(base[1]), 1), "size": s,
                       "area": round(p["area"]), "px": p["px"], "contour": p["contour"]})
    if debug_path is not None:
        vis = frame.copy()
        for p in pieces:
            cv2.drawContours(vis, [p["contour"]], -1, (0, 255, 0), 2)
            q = cam.to_px([(p["x"], p["y"])])[0].astype(int)
            cv2.circle(vis, tuple(q), 5, (0, 0, 255), -1)
            cv2.putText(vis, "SML"[p["size"]], tuple(q + [8, -8]), 0, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(debug_path), vis)
    return pieces


if __name__ == "__main__":
    import sys
    f = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else topcam.grab()
    cam = topcam.BoardCam.load()
    for p in detect(f, cam, 0, counts=[2, 2, 2], debug_path=ROOT / "out" / "sam_pieces.jpg"):
        print({k: v for k, v in p.items() if k != "contour"})
