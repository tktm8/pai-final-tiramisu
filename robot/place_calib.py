"""Place-and-measure calibration: camera (SAM 2 base estimate) -> arm coordinates, desk and board.

1. A person puts one MEDIUM orange piece between the open right jaws (once). The arm closes gently.
2. Repeat over desk spots and empty board cells:
     set the piece down at a known ARM position -> back to the camera pose -> photo -> SAM 2 measures it
     -> one (camera, arm) pair -> refit the map -> grab the piece again where we put it (we know exactly where).
3. Saves out/camera_to_arm_affine.json (used by move_test.to_arm for desk and board).

  python robot/place_calib.py
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
import move_test as MT  # noqa: E402
import sam_pieces  # noqa: E402
import self_calib as SC  # noqa: E402
import topcam  # noqa: E402
import wrist_servo as W  # noqa: E402
from planner import Segment  # noqa: E402

SIZE = 1
MAP = ROOT / "out" / "camera_to_arm_affine.json"
PAIRS = ROOT / "out" / "place_calib_pairs.json"
log = print


def say(t):
    log("SAY:", t)
    try:
        from stackchan import StackChan
        StackChan().say(t)
    except Exception:
        pass


def fit(pairs):
    """camera -> arm: translation with 1 pair, similarity-ish affine with >= 3."""
    C = np.array([p["camera"] for p in pairs])
    A = np.array([p["arm"] for p in pairs])
    if len(pairs) < 3:
        t = (A - C).mean(0)
        return [[1.0, 0.0], [0.0, 1.0], t.tolist()]
    X = np.c_[C, np.ones(len(C))]
    M, *_ = np.linalg.lstsq(X, A, rcond=None)
    return M.tolist()


def apply(M, xy):
    v = np.array([xy[0], xy[1], 1.0]) @ np.array(M)
    return v[:2]


def main():
    arm = SC.Arm()
    arm.c = np.array([0.0, 1.0, 0.0])
    f, gp, P, d = arm.f, arm.gp, arm.P, arm.d
    pairs = []
    M = json.loads(MAP.read_text())["M"] if MAP.exists() else [[1, 0], [0, 1], [0, 0]]
    try:
        arm.home()
        frame = topcam.grab(warmup=40)
        cam = topcam.BoardCam.load()
        desk = sam_pieces.detect(frame, cam, 0)
        board = topcam.read_cells(frame, cam)
        occupied = [c for c, r in enumerate(board) if r.color is not None]
        cells = [c for c in (4, 5, 1, 2, 8, 7) if c not in occupied]
        spots = []
        for x in (170.0, 215.0, 255.0):
            for y in (-150.0, -200.0, -245.0):
                if all(np.hypot(x - o["x"], y - o["y"]) > 60 for o in desk):
                    spots.append(np.array([x, y]))
        targets = [("desk", s) for s in spots[:2]] + [("cell", np.array(G.cell_xy(c))) for c in cells[:4]]
        log(f"occupied cells {occupied}; targets (camera frame): {[(k, t.round(0).tolist()) for k, t in targets]}")
        obstacles = [(apply(M, (o["x"], o["y"])), 0.0, o["size"]) for o in desk] + \
                    [(apply(M, G.cell_xy(c)), G.BOARD_FLOOR_Z, 2) for c in occupied]

        # 1) hand-off: open wide, ask for the piece, close gently
        g_open = G.gap_to_rad(d + 24.0)
        q_cam = arm.q_cam
        if "--piece-in" not in sys.argv:
            f.run([Segment("open", [q_cam] * G.HOLD, g_open)])
        if "--piece-in" in sys.argv:          # the person already put the piece between the open jaws
            held = W.gentle_close(f, q_cam, SIZE, {})
            if not held:
                raise SystemExit("no piece between the jaws")
        else:
            for attempt in range(3):
                say("右手に、オレンジの中くらいの駒をはさんでね。15秒後につかむよ。")
                for _ in range(150):
                    f.command(q_cam, gripper_rad=g_open)
                    time.sleep(0.1)
                held = W.gentle_close(f, q_cam, SIZE, {})
                if held:
                    break
                f.run([Segment("open", [q_cam] * G.HOLD, g_open)])
            else:
                raise SystemExit("no piece was put in the gripper")
        say("ありがとう！測ってくるね。")
        grip_units = f.present()["gripper"]
        q_from = q_cam

        for kind, tgt in targets:
            floor = G.BOARD_FLOOR_Z if kind == "cell" else 0.0
            cmd = apply(M, tgt)                       # where the arm should put it, given what we know so far
            g_hold = W.units_to_rad(max(0.0, grip_units))
            chk = G.add_obstacles(P.checker_for("right", arm.ql), obstacles, margin_mm=2.0)
            G.RESERVE_SLOTS[96] = (SIZE, float(cmd[0]), float(cmd[1]))
            place = gp.grasp_pose(chk, G.Spot("slot", 96), SIZE, floor + 8.0, q_from, g_hold)
            if place is None:
                log(f"  {kind} {tgt.round(0)}: no place pose - skipped (arm did not move)")
                continue
            q_pl, a2, c2, tcp, _ = place
            hov, q_hov = gp.hover(tcp, a2, c2, q_pl, min_dz=15.0)
            way = MT.route(P, chk, q_from, q_hov, g_hold) if q_hov is not None else None
            lower = P.line_path(chk, "right", q_hov, hov, tcp, a2, c2, g_hold) if way else None
            up = None
            if lower:
                for rel in (24.0, 16.0):
                    g_rel = G.gap_to_rad(d + rel)
                    up = P.line_path(chk, "right", lower[-1], tcp, hov, a2, c2, g_rel)
                    if up:
                        break
            if up is None:
                log(f"  {kind} {tgt.round(0)}: no path - skipped (arm did not move)")
                continue
            if not f.run([Segment("to_spot", way, g_hold), Segment("lower", lower, g_hold),
                          Segment("release", [lower[-1]] * G.HOLD, g_rel), Segment("retreat", up, g_rel)]):
                raise RuntimeError("arm stopped while placing")
            back = MT.route(P, chk, up[-1], q_cam, g_rel)
            if not f.run([Segment("back", back or [q_cam], g_rel)]):
                raise RuntimeError("arm stopped on the way back")
            time.sleep(0.6)
            fr = topcam.grab(warmup=40)
            if kind == "cell":
                seen = SC.board_orange(fr, cam)
                seen = [s for s in seen if np.hypot(*(s - tgt)) < 60]
            else:
                ds = sam_pieces.detect(fr, cam, 0)
                seen = [np.array([p["x"], p["y"]]) for p in ds if np.hypot(p["x"] - tgt[0], p["y"] - tgt[1]) < 60]
            if not seen:
                log(f"  {kind} {tgt.round(0)}: placed but the camera cannot find it - stopping")
                break
            obs = min(seen, key=lambda s: np.hypot(*(s - tgt)))
            pairs.append({"camera": obs.tolist(), "arm": cmd.tolist(), "kind": kind})
            M = fit(pairs)
            err = np.hypot(*(apply(M, obs) - cmd))
            log(f"  {kind}: arm put it at {cmd.round(1)}, camera sees {obs.round(1)}  -> pairs {len(pairs)}, "
                f"fit residual {err:.1f} mm")
            MAP.write_text(json.dumps({"M": M, "note": f"place-and-measure, {len(pairs)} pairs"}, indent=1))
            PAIRS.write_text(json.dumps(pairs, indent=1))
            # grab it again exactly where the arm put it
            g = arm.grab(cmd, floor, obstacles)
            if isinstance(g, str):
                log("  re-grasp: " + ("no plan" if g == "noplan" else "MISSED") + " - stopping")
                break
            pk, down = g
            grip_units = pk.grip_units
            lift = list(reversed(down)) + MT.high_lift(gp, pk.chk, down[0], np.array([0.0, 0.0, -1.0]), arm.c, g_hold)
            if not f.run([Segment("lift", lift, W.units_to_rad(max(0.0, grip_units)))]):
                raise RuntimeError("arm stopped while lifting")
            q_from = lift[-1]
    finally:
        if not f.stop:
            f.disconnect()
        else:
            log("stopped: torque held")
    if pairs:
        C = np.array([p["camera"] for p in pairs])
        A = np.array([p["arm"] for p in pairs])
        res = np.linalg.norm(np.array([apply(M, c) for c in C]) - A, axis=1)
        log(f"RESULT: {len(pairs)} pairs, map saved; residuals {res.round(1)} mm")


if __name__ == "__main__":
    main()
