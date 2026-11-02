"""Hand-eye calibration: the right arm sets a piece down at known positions; the top camera measures each one.

The camera sees piece tops from above at an angle, so a blob mapped through the board-plane homography lands
farther from the camera nadir than the real piece. Model per sample:  raw = a + k * true  (a: 2-vector, k: scale).
From (a, k) for one size we recover the camera position and the correction for every size.

  python robot/handeye.py          # needs a person to put one MEDIUM orange piece into the right gripper once
Result: out/handeye.json (used automatically by perception/topcam.py parallax_fix)
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
sys.path[:0] = [str(ROOT / "robot"), str(ROOT / "perception"), str(ROOT / "server")]

import gobblet_arm as G  # noqa: E402
import topcam  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402
from stackchan import StackChan  # noqa: E402

SIZE = 1
CELLS = [4, 5, 7, 8, 2]
OUT = ROOT / "out" / "handeye.json"
sc = StackChan()


def say(t):
    print("SAY:", t)
    sc.say(t)


def measure(expect_xy, tag):
    frame = topcam.grab()
    import cv2
    cv2.imwrite(str(ROOT / "out" / f"handeye_{tag}.jpg"), frame)
    cam = topcam.BoardCam.load()
    bl = [b for b in topcam.blobs(frame, cam) if b.color == 0]
    if not bl:
        return None
    b = min(bl, key=lambda b: np.hypot(b.x - expect_xy[0], b.y - expect_xy[1]))
    dist = np.hypot(b.x - expect_xy[0], b.y - expect_xy[1])
    print(f"  {tag}: expected {np.round(expect_xy, 1)}  camera raw ({b.x:.1f}, {b.y:.1f})  d={b.d:.0f}  off {dist:.1f} mm")
    return (b.x, b.y) if dist < 45 else None


def other_pieces():
    frame = topcam.grab()
    cam = topcam.BoardCam.load()
    return [(np.array([p["x"], p["y"]]), 0.0, 2) for p in topcam.off_board_pieces(frame, cam, 0)]   # treat as large: safe


def main():
    gp = G.GobbletPlanner()
    P = gp.planner
    ql = G.current_q("left")
    q_look = G.current_q("right")
    if abs(np.degrees(q_look[0]) - G.LOOK_PAN_DEG) > 5:
        raise SystemExit("right arm is not in the camera-clear pose; start a game once or run plan_to_look first")
    obstacles = other_pieces()
    print(f"{len(obstacles)} other orange pieces on the desk (kept as obstacles)")
    d, h = G.PIECE[SIZE]
    g_load = 0.6
    q_up = q_look.copy()
    q_up[1:4] = np.radians(G.LOOK_RAISED_DEG)
    chk = P.checker_for("right", ql)
    unfold = P.joint_path(chk, "right", q_look, q_up, g_load)

    # 1) load: unfold, open wide, ask for a piece, close
    f = Follower("right")
    try:
        f.run([Segment("unfold", unfold, 0.0), Segment("open", [q_up] * 40, g_load)])
        for attempt in range(4):
            say("右手に、オレンジの中くらいの駒をはさんでね。10秒後につかむよ。")
            for _ in range(100):
                f.command(q_up, gripper_rad=g_load)
                time.sleep(0.1)
            for _ in range(40):
                f.command(q_up, gripper_rad=-0.10)
                time.sleep(1 / 30)
            time.sleep(0.5)
            g = f.present()["gripper"]
            print("gripper after close:", round(g, 1))
            if g > 4:
                say("ありがとう！じゃあ測ってくるね。")
                break
            for _ in range(40):
                f.command(q_up, gripper_rad=g_load)
                time.sleep(1 / 30)
        else:
            raise SystemExit("no piece in the gripper")
    finally:
        f.disconnect()   # torque off would drop the arm: keep going quickly with a new connection below

    # 2) place at the first cell, fold, measure; then carry it cell to cell
    samples = []
    first = G.Spot("cell", CELLS[0])
    chk_o = G.add_obstacles(P.checker_for("right", ql), obstacles)
    g_closed, g_release = -0.10, G.gap_to_rad(d + 6.0)
    place = gp.grasp_pose(chk_o, first, SIZE, 8.0, q_up, g_closed)
    if place is None:
        raise SystemExit("cannot place at the first cell")
    q_pl, a, c, tcp, _ = place
    hov, q_hov = gp.hover(tcp, a, c, q_pl)
    segs = [Segment("to_above", P.staged_path(chk_o, "right", q_up, q_hov, g_closed), g_closed),
            Segment("lower", P.line_path(chk_o, "right", q_hov, hov, tcp, a, c, g_closed), g_closed)]
    up = P.line_path(chk_o, "right", segs[-1].qs[-1], tcp, hov, a, c, g_release)
    back = P.staged_path(chk_o, "right", up[-1], q_up, g_release)
    fold = P.joint_path(chk, "right", q_up, q_look, g_release)
    if any(s.qs is None for s in segs) or up is None or back is None or fold is None:
        raise SystemExit("could not plan the first placement")
    segs += [Segment("release", [segs[-1].qs[-1]] * G.HOLD, g_release), Segment("retreat", up, g_release),
             Segment("back", back, g_release), Segment("fold", fold, g_release)]
    f = Follower("right")
    try:
        f.run([Segment("hold", [q_up] * 10, g_closed)] + segs)
    finally:
        f.disconnect()
    true_xy = G.cell_xy(CELLS[0]) + 0.0
    raw = measure(true_xy, f"cell{CELLS[0]}")
    if raw:
        samples.append((true_xy, raw))
    prev = first
    for c_ in CELLS[1:]:
        dst = G.Spot("cell", c_)
        segs, info = gp.plan_transfer(prev, dst, SIZE, obstacles, G.current_q("right"), ql)
        if segs is None:
            print(f"  skip cell{c_}: {info}")
            continue
        ok = G.run_segments("right", segs)
        if not ok:
            break
        raw = measure(G.cell_xy(c_), f"cell{c_}")
        if raw is None:
            print("  piece not where it should be (grasp missed?) - stopping")
            break
        samples.append((G.cell_xy(c_), raw))
        prev = dst

    if len(samples) < 3:
        raise SystemExit(f"only {len(samples)} samples")
    T = np.array([s[0] for s in samples])
    R = np.array([s[1] for s in samples])
    # raw = a + k * true  ->  [1 0 tx; 0 1 ty] [ax ay k]^T
    A = np.zeros((2 * len(T), 3))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    A[0::2, 2] = T[:, 0]
    A[1::2, 2] = T[:, 1]
    sol, *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    ax, ay, k = sol
    resid = np.linalg.norm((A @ sol - R.ravel()).reshape(-1, 2), axis=1)
    C = np.array([ax, ay]) / (1 - k) if abs(1 - k) > 1e-4 else np.array([0.0, 0.0])
    res = dict(size=SIZE, a=[ax, ay], k=k, camera_xy=C.tolist(), resid_mm=resid.round(1).tolist(),
               samples=[{"true": t.tolist(), "raw": list(r)} for t, r in samples], saved=time.strftime("%F %T"))
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    say("測り終わったよ！")


if __name__ == "__main__":
    main()
