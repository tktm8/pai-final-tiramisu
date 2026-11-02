"""Wrist-camera correction for desk picks, learned during play (both arms).

At the hover above a desk piece (fixed height and jaw angle) a piece that is exactly between the jaws always
shows up at the same image point REF[arm, size, angle]. The image Jacobian J[arm, angle] (pixels per mm of
gripper motion) is learned with two small test moves the first time a grasp misses.

  - before every desk grasp: snap at the hover, remember where the piece is (b0)
  - grasp held at once         -> REF moves towards b0
  - missed                     -> snap again, step = -J^-1 (blob - REF), grasp there
  - a nudge (dx, dy) held      -> REF moves towards b0 + J (dx, dy)
State: out/wrist_refine.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "perception")]
import topcam  # noqa: E402

CAM_IDS = {"left": "0x31400000c456366", "right": "0x21130000c456366"}
STATE = ROOT / "out" / "wrist_refine.json"
TEST_MOVE = 10.0
MAX_STEP = 15.0
LEARN_IN_PLAY = False     # keep the hand-taught / saved REFs (user, 2026-11-01 19:10)


def blobs(img):
    """Orange piece blobs: [(u, v, area)]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 110, 90), (22, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if st[i, 4] < 1500:
            continue
        pts = np.column_stack(np.where(lab == i))[:, ::-1].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        if max(w, h) > 3.0 * max(1.0, min(w, h)):      # the board's orange bars, not a piece
            continue
        out.append((float(cen[i][0]), float(cen[i][1]), int(st[i, 4])))
    return out


class WristRefine:
    def __init__(self, arm, size, angle, log=print):
        self.arm, self.size, self.angle, self.log = arm, size, int(round(angle)), log
        self.st = json.loads(STATE.read_text()) if STATE.exists() else {}
        self.cap = None
        self.b0 = None
        self.last = None      # the latest wrist frame (DAgger records it)

    # ---- state
    @property
    def kref(self):
        return f"ref_{self.arm}_{self.size}_{self.angle}"

    @property
    def kJ(self):
        return f"J_{self.arm}_{self.angle}"

    def ref(self):
        return self.st.get(self.kref)

    def J(self):
        return np.array(self.st[self.kJ]) if self.kJ in self.st else None

    def save(self):
        STATE.write_text(json.dumps(self.st, indent=1))

    # ---- camera
    def snap(self, tag):
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(topcam.camera_index(CAM_IDS[self.arm]), cv2.CAP_AVFOUNDATION)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                for _ in range(10):
                    self.cap.read()
            img = None
            for _ in range(6):           # flush stale frames
                ok, fr = self.cap.read()
                if ok:
                    img = fr
            if img is not None:
                cv2.imwrite(str(ROOT / "out" / f"wrist_{self.arm}_{tag}.jpg"), img)
            self.last = img
            return img
        except Exception as e:
            self.log(f"  wrist camera: {e}")
            return None

    def find(self, near, tag):
        img = self.snap(tag)
        if img is None:
            return None
        bl = blobs(img)
        if not bl:
            return None
        H, W = img.shape[:2]
        p = np.array(near if near is not None else (W * 0.5, H * 0.6))
        b = min(bl, key=lambda b: np.hypot(b[0] - p[0], b[1] - p[1]))
        return np.array(b[:2])

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---- steps
    def before_grasp(self):
        self.b0 = self.find(self.ref(), "hover0")
        self.log(f"  wrist: piece at px {None if self.b0 is None else self.b0.round().tolist()}, "
                 f"ref {None if self.ref() is None else np.round(self.ref()).tolist()}")

    def learn_ref(self, nudge_mm=(0.0, 0.0), w=0.5):
        """A grasp held after moving nudge_mm from the first hover."""
        if not LEARN_IN_PLAY:
            return
        if self.b0 is None:
            return
        J = self.J()
        if J is None and np.any(nudge_mm):
            return
        new = self.b0 + (J @ np.asarray(nudge_mm) if J is not None else 0.0)
        old = self.ref()
        val = new if old is None else (1 - w) * np.array(old) + w * new
        self.st[self.kref] = [float(v) for v in val]
        self.save()
        self.log(f"  wrist: REF {self.arm}/{'SML'[self.size]}/{self.angle}deg -> {np.round(val).tolist()}")

    def learn_J(self, move_to, xy):
        """Two test moves at the hover. move_to(xy) -> True when the arm got there."""
        if self.b0 is None:
            return None
        cols = []
        for k in (0, 1):
            d = np.zeros(2)
            d[k] = TEST_MOVE
            if not move_to(xy + d):
                return None
            time.sleep(0.3)
            b = self.find(self.b0, f"jtest{k}")
            if b is None:
                return None
            cols.append((b - self.b0) / TEST_MOVE)
        move_to(xy)
        J = np.array(cols).T
        norms = np.linalg.norm(J, axis=0)
        if abs(np.linalg.det(J)) < 0.5 or norms.min() < 2.0 or norms.max() > 15.0:
            self.log(f"  wrist: J looks wrong {J.round(2).tolist()} - not saved")
            return None
        if self.kJ in self.st:                 # average with what was measured before (each test is a bit noisy)
            J = 0.5 * (J + np.array(self.st[self.kJ]))
        self.st[self.kJ] = J.tolist()
        self.save()
        self.log(f"  wrist: learned J {self.arm}/{self.angle}deg (px/mm) {J.round(2).tolist()}")
        return J

    def correction_from_b0(self):
        """Like correction(), from the photo already taken at the hover (no new snap)."""
        ref, J = self.ref(), self.J()
        if ref is None or J is None or self.b0 is None:
            return None
        e = self.b0 - np.array(ref)
        if np.linalg.norm(e) > 250:          # probably a different piece
            return None
        step = -np.linalg.solve(J, e)
        n = np.linalg.norm(step)
        if n > MAX_STEP:
            step *= MAX_STEP / n
        self.log(f"  wrist: piece {self.b0.round().tolist()} vs ref {np.round(ref).tolist()} ({np.linalg.norm(e):.0f}px) "
                 f"-> move ({step[0]:+.1f}, {step[1]:+.1f}) mm")
        return step

    def correction(self):
        """mm to move the hover so the piece sits on REF (None when unknown)."""
        ref, J = self.ref(), self.J()
        if ref is None or J is None:
            return None
        b = self.find(self.b0 if self.b0 is not None else ref, "hover1")
        if b is None:
            return None
        e = b - np.array(ref)
        step = -np.linalg.solve(J, e)
        n = np.linalg.norm(step)
        if n > MAX_STEP:
            step *= MAX_STEP / n
        self.log(f"  wrist: piece {b.round().tolist()} vs ref {np.round(ref).tolist()} ({np.linalg.norm(e):.0f}px) "
                 f"-> move ({step[0]:+.1f}, {step[1]:+.1f}) mm")
        return step
