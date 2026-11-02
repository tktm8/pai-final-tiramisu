"""Pick a piece seen by the top camera and place it on a board cell, using the hand-taught camera->arm correction.

  python robot/direct_pick.py [size] [cell]      # default: medium piece nearest (215,-171) -> cell 4
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot"), str(ROOT / "perception")]
import gobblet_arm as G  # noqa: E402
import sam_pieces  # noqa: E402
import topcam  # noqa: E402
import wrist_servo as W  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402

CORR = json.loads((ROOT / "out" / "camera_to_arm.json").read_text())


def to_arm(x, y):
    return CORR["ax"] * x + CORR["bx"], y + CORR["dy"]


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cell = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    gp = G.GobbletPlanner()
    ql = G.current_q("left")
    f = Follower("right")
    try:
        # 1) fold into the camera pose, one joint at a time (works from wherever a person left the arm)
        q = G.current_q("right")
        segs, q = W.joint_seq(q, [(1, -60.0), (2, 76.0), (3, 42.0), (4, 1.3), (0, G.LOOK_PAN_DEG)] + W.FOLD, 0.0)
        if not f.run(segs):
            raise RuntimeError("folding stopped (blocked)")
        q_look = G.current_q("right")
        time.sleep(0.5)
        # 2) top camera
        frame = topcam.grab(warmup=40)
        cam = topcam.BoardCam.load()
        pieces = sam_pieces.detect(frame, cam, 0)
        cand = [p for p in pieces if p["size"] == size]
        if not cand:
            raise RuntimeError("no piece of that size seen")
        p = min(cand, key=lambda p: np.hypot(p["x"] - 215, p["y"] + 171))
        tx, ty = to_arm(p["x"], p["y"])
        print(f"camera ({p['x']}, {p['y']}) -> arm target ({tx:.1f}, {ty:.1f})")
        others = [(np.array(to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in pieces if o is not p]
        # 3) unfold, hover above, straight down, gentle close
        d, h = G.PIECE[size]
        g0 = G.gap_to_rad(d + 24.0)
        unfold, q_up = W.joint_seq(q_look, W.UNFOLD, g0)
        if not f.run(unfold + [Segment("open", [q_up] * G.HOLD, g0)]):
            raise RuntimeError("unfold stopped")
        pk = W.Picker(gp, f, ql, others)
        a, c = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])
        xy = np.array([tx, ty]) - c[:2] * (d + 24.0) / 2           # TCP (fixed jaw) so the piece is mid-jaws
        qh = pk.move_hover(xy, a, c, q_up, g0)
        if qh is None:
            raise RuntimeError("cannot reach above the piece")
        held, down = pk.try_grasp(qh, xy, a, c, size, g0)
        print("GRASP:", "HELD" if held else "missed")
        if held:
            ok, why = W.place_on_cell(pk, down, size, cell, q_up, q_look)
            print("PLACE:", why if ok else "failed: " + why)
            if ok:
                return
        if down is not None:
            f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
        back = gp.planner.staged_path(pk.chk, "right", qh, q_up, g0)
        fold, _ = W.joint_seq(q_up, W.FOLD, g0)
        f.run(([Segment("back", back, g0)] if back else []) + fold)
    finally:
        if not f.stop:
            f.disconnect()
        else:
            print("stopped: torque held")


if __name__ == "__main__":
    main()
