"""Self-calibration of the camera -> grasp offset, no human needed.

For one piece of each size found by the top camera: go above it, then try grasps in a spiral of small offsets
around the camera position until the gripper actually holds something (jaws stop open). The successful offset
(true - camera) is stored per size in out/grasp_offsets.json and used by perception/topcam.assign_sizes.
The piece is put back exactly where it was.

  python robot/grasp_probe.py
"""

from __future__ import annotations

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
import topcam  # noqa: E402
import sam_pieces  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402

OUT = ROOT / "out" / "grasp_offsets.json"
STEP = 8.0
HELD_UNITS = 4.0         # gripper reading above this after a full close = something between the jaws


def spiral(max_r=16.0):
    pts = [(0.0, 0.0)]
    r = STEP
    while r <= max_r + 1e-6:
        n = int(round(2 * np.pi * r / STEP))
        pts += [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        r += STEP
    return pts


def probe(gp, f, piece, others, ql):
    """Returns the (dx, dy) offset that grasped, or None. Arm starts and ends folded (camera-clear)."""
    P = gp.planner
    size = piece["size"]
    d, h = G.PIECE[size]
    gap = d + 24.0
    g_open, g_closed = G.gap_to_rad(gap), -0.10
    chk = G.add_obstacles(P.checker_for("right", ql), others)
    q_look = G.current_q("right")
    q_up = q_look.copy()
    q_up[1:4] = np.radians(G.LOOK_RAISED_DEG)
    G.RESERVE_SLOTS[99] = (size, piece["x"], piece["y"])
    spot = G.Spot("slot", 99)
    g = gp.grasp_pose(chk, spot, size, 0.0, q_up, g_open, centred_gap=gap)
    if g is None:
        print("  no grasp pose")
        return None
    q0, a, c, tcp0, label = g
    hov0, q_h0 = gp.hover(tcp0, a, c, q0)
    unfold = P.joint_path(P.checker_for("right", ql), "right", q_look, q_up, g_open)
    to_above = P.staged_path(chk, "right", q_up, q_h0, g_open)
    if unfold is None or to_above is None:
        print("  cannot reach above the piece")
        return None
    f.run([Segment("unfold", unfold, g_open), Segment("open", [q_up] * G.HOLD, g_open),
           Segment("to_above", to_above, g_open)])
    q_cur = to_above[-1]
    found = None
    for dx, dy in spiral():
        tcp = tcp0 + np.array([dx, dy, 0.0])
        hov = tcp - a * (hov0[2] - tcp[2]) / max(1e-3, -a[2])
        q_h = gp.k.ik_pose(hov, a, c, q_cur, n_random=0)
        if q_h is None:
            continue
        move = P.joint_path(chk, "right", q_cur, q_h, g_open)
        down = P.line_path(chk, "right", q_h, hov, tcp, a, c, g_open) if move is not None else None
        if down is None:
            continue
        f.run([Segment("shift", move, g_open), Segment("descend", down, g_open),
               Segment("close", [down[-1]] * G.HOLD, g_closed)])
        time.sleep(0.3)
        units = f.present()["gripper"]
        held = units > HELD_UNITS
        print(f"  offset ({dx:+.0f},{dy:+.0f}) mm -> gripper {units:.1f} {'HELD' if held else 'miss'}")
        f.run([Segment("open", [down[-1]] * G.HOLD, g_open), Segment("up", list(reversed(down)), g_open)])
        q_cur = q_h
        if held:
            found = (dx, dy)
            break
    back = P.staged_path(chk, "right", q_cur, q_up, g_open)
    fold = P.joint_path(P.checker_for("right", ql), "right", q_up, q_look, g_open)
    f.run([Segment("back", back, g_open), Segment("fold", fold, g_open)])
    return found


def main():
    gp = G.GobbletPlanner()
    ql = G.current_q("left")
    if abs(np.degrees(G.current_q("right")[0]) - G.LOOK_PAN_DEG) > 5:
        raise SystemExit("right arm is not in the camera-clear pose")
    frame = topcam.grab()
    cam = topcam.BoardCam.load()
    pieces = sam_pieces.detect(frame, cam, 0)
    print("SAM pieces:", [("SML"[p["size"]], p["x"], p["y"]) for p in pieces])
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    f = Follower("right")
    try:
        for size in (1,):
            cand = sorted([p for p in pieces if p["size"] == size], key=lambda p: abs(p["y"] + 190) + abs(p["x"] - 210))
            if not cand:
                continue
            piece = cand[0]
            others = [(np.array([p["x"], p["y"]]), 0.0, p["size"]) for p in pieces if p is not piece]
            print(f"size {'SML'[size]} at camera ({piece['x']}, {piece['y']})")
            off = probe(gp, f, piece, others, ql)
            if off is not None:
                results[str(size)] = {"dx": off[0], "dy": off[1], "at": [piece["x"], piece["y"]],
                                      "saved": time.strftime("%F %T")}
                OUT.write_text(json.dumps(results, indent=1))
    finally:
        f.disconnect()
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
