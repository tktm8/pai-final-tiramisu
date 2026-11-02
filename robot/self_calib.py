"""Self-calibration of camera -> arm coordinates on the board, no human needed.

The robot picks a SMALL orange piece from the desk, sets it down on empty board cells at known ARM positions,
goes back to the camera pose, and measures where the piece really is with the top camera (SAM 2). The pairs
(camera position, arm position) from the board plus the hand-taught desk samples give one affine map
camera -> arm, saved to out/camera_to_arm_affine.json (used by move_test for desk AND board).

  python robot/self_calib.py            # uses empty cells among 4, 1, 2
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
import move_test as MT  # noqa: E402
import sam_pieces  # noqa: E402
import topcam  # noqa: E402
import wrist_servo as W  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402

SIZE = 1       # medium: more slack between the jaws than a small piece, still fits a cell easily
OUT = ROOT / "out" / "board_selfcal.json"
log = print


def board_orange(frame, cam):
    """Orange pieces standing ON the board: camera-plane base estimates (same method as the desk detector)."""
    bl = [b for b in topcam.blobs(frame, cam) if b.color == 0 and 95 < b.x < 330 and abs(b.y) < 115]
    if not bl:
        return []
    res = sam_pieces.sam()(frame, points=[[list(b.px)] for b in bl], labels=[[1] for _ in bl], verbose=False)
    out = []
    import cv2
    for m, b in zip(res[0].masks.data.cpu().numpy(), bl):
        m = (m > 0.5).astype(np.uint8)
        cnt = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if not cnt:
            continue
        w = cam.to_world(max(cnt, key=cv2.contourArea).reshape(-1, 2).astype(np.float64))
        near = w[np.argmin(np.linalg.norm(w - sam_pieces.NADIR_XY, axis=1))]
        u = (near - sam_pieces.NADIR_XY) / np.linalg.norm(near - sam_pieces.NADIR_XY)
        base = near + u * sam_pieces.RADIUS[SIZE]
        out.append(np.array(base))
    return out


class Arm:
    def __init__(self):
        self.gp = G.GobbletPlanner()
        self.P = self.gp.planner
        self.ql = G.current_q("left")
        self.q_cam = np.array(json.loads((ROOT / "out" / "camera_pose.json").read_text())["right"])
        self.f = Follower("right")
        self.d, self.h = G.PIECE[SIZE]

    def home(self):
        MT.go_to_camera_pose(self.f, self.q_cam)
        time.sleep(0.4)

    def grab(self, arm_xy, floor, others):
        """Pick the piece at arm_xy (arm coords). Returns (pk, down), "noplan" (arm did not move) or "missed"."""
        P, gp, d, h = self.P, self.gp, self.d, self.h
        chk = G.add_obstacles(P.checker_for("right", self.ql), others, margin_mm=2.0)
        a = np.array([0.0, 0.0, -1.0])
        for ang, extra in [(90, 24.0), (60, 24.0), (90, 16.0), (60, 16.0), (30, 24.0), (0, 24.0), (0, 16.0)]:
            c = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
            xy = np.asarray(arm_xy) - c[:2] * (d + extra) / 2
            hov = np.array([xy[0], xy[1], W.HOVER_Z])
            tcp = np.array([xy[0], xy[1], floor + h * G.GRASP_FRAC[SIZE]])
            g = G.gap_to_rad(d + extra)
            q_h = gp.k.ik_pose(hov, a, c, self.q_cam, n_random=4)
            if q_h is None or P.line_path(chk, "right", q_h, hov, tcp, a, c, g) is None:
                continue
            way = MT.route(P, chk, self.q_cam, q_h, g)
            if way is None:
                continue
            self.f.run([Segment("open", [self.q_cam] * G.HOLD, g), Segment("to_piece", way, g)])
            pk = W.Picker(gp, self.f, self.ql, others)
            held, down = pk.try_grasp(q_h, xy, a, c, SIZE, g, floor=floor)
            self.c = c
            if held:
                return pk, down
            if down is not None:
                self.f.run([Segment("open", [down[-1]] * G.HOLD, g), Segment("up", list(reversed(down)), g)])
            back = MT.route(P, chk, q_h, self.q_cam, g)
            self.f.run([Segment("back", back or [self.q_cam], g)])
            return "missed"
        return "noplan"

    def put(self, pk, down, arm_xy, floor):
        """Set the held piece down with its centre at arm_xy (arm coords), then return to the camera pose."""
        P, gp, d = self.P, self.gp, self.d
        g_hold = W.units_to_rad(max(0.0, pk.grip_units))
        g_rel = G.gap_to_rad(d + 24.0)          # open wide to let go
        lift = list(reversed(down))
        lift += MT.high_lift(gp, pk.chk, lift[-1], np.array([0.0, 0.0, -1.0]), self.c, g_hold)
        G.RESERVE_SLOTS[97] = (SIZE, float(arm_xy[0]), float(arm_xy[1]))
        place = gp.grasp_pose(pk.chk, G.Spot("slot", 97), SIZE, floor + 8.0, lift[-1], g_hold)
        if place is None:
            return False
        q_pl, a2, c2, tcp, _ = place
        hov, q_hov = gp.hover(tcp, a2, c2, q_pl, min_dz=15.0)
        carry = P.staged_path(pk.chk, "right", lift[-1], q_hov, g_hold) if q_hov is not None else None
        lower = P.line_path(pk.chk, "right", q_hov, hov, tcp, a2, c2, g_hold) if carry else None
        up = None
        if lower:
            for rel_extra in (24.0, 16.0):
                g_rel = G.gap_to_rad(d + rel_extra)
                up = P.line_path(pk.chk, "right", lower[-1], tcp, hov, a2, c2, g_rel)
                if up is not None:
                    break
        if up is None:
            return False
        self.f.run([Segment("lift", lift, g_hold), Segment("carry", carry, g_hold), Segment("lower", lower, g_hold),
                    Segment("release", [lower[-1]] * G.HOLD, g_rel), Segment("retreat", up, g_rel)])
        back = MT.route(P, pk.chk, up[-1], self.q_cam, g_rel)
        self.f.run([Segment("back", back or [self.q_cam], g_rel)])
        return True


def main():
    arm = Arm()
    samples = []
    try:
        arm.home()
        frame = topcam.grab(warmup=40)
        cam = topcam.BoardCam.load()
        board = topcam.read_cells(frame, cam)
        occupied = [c for c, r in enumerate(board) if r.color is not None]
        cells = [c for c in (4, 1, 2) if c not in occupied]
        log(f"board occupied {occupied}; calibration cells {cells}")
        desk = sam_pieces.detect(frame, cam, 0)
        smalls = [p for p in desk if p["size"] == SIZE]
        if not smalls or not cells:
            raise SystemExit("need a small orange piece on the desk and an empty cell among 4, 1, 2")
        def room(q):
            return min([np.hypot(q["x"] - o["x"], q["y"] - o["y"]) for o in desk if o is not q] or [999])
        smalls.sort(key=lambda q: -room(q))       # most free space around it first (touching pieces confuse SAM)
        log("candidates (free space mm): " + str([(q["x"], q["y"], round(room(q))) for q in smalls]))
        board_obs = [(np.array(MT.to_arm(*G.cell_xy(c))), G.BOARD_FLOOR_Z, 2) for c in occupied]
        g = "noplan"
        for p in smalls:
            desk_obs = [(np.array(MT.desk_to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p]
            src_xy = np.array(MT.desk_to_arm(p["x"], p["y"]))
            g = arm.grab(src_xy, 0.0, desk_obs + board_obs)
            log(f"piece at camera ({p['x']}, {p['y']}) -> arm {src_xy.round(1)}: "
                + ("HELD" if not isinstance(g, str) else ("no plan, arm did not move" if g == "noplan" else "arm went there but MISSED")))
            if not isinstance(g, str):
                break
            if g == "missed":
                break
        if isinstance(g, str):
            raise SystemExit("could not pick up a calibration piece")
        for i, cell in enumerate(cells):
            if i > 0:
                g = arm.grab(src_xy, src_floor, desk_obs + board_obs)
                if isinstance(g, str):
                    log("re-grasp on the board: " + ("no plan, arm did not move" if g == "noplan" else "MISSED"))
                    break
            pk, down = g
            target = np.array(MT.to_arm(*G.cell_xy(cell)))          # where we THINK cell centre is (arm coords)
            if not arm.put(pk, down, target, G.BOARD_FLOOR_Z):
                log(f"could not place on cell {cell}")
                break
            time.sleep(0.6)
            fr = topcam.grab(warmup=40)
            seen = board_orange(fr, cam)
            # the piece we just placed: the board orange blob nearest the cell (others are known occupied cells)
            cands = [s for s in seen if np.hypot(*(s - G.cell_xy(cell))) < 60]
            if not cands:
                log(f"cannot see the piece near cell {cell}")
                break
            obs = min(cands, key=lambda s: np.hypot(*(s - G.cell_xy(cell))))
            log(f"cell {cell}: commanded arm {target.round(1)}  camera saw {obs.round(1)}  "
                f"(cell centre {G.cell_xy(cell)})")
            samples.append({"camera": obs.tolist(), "arm": target.tolist(), "cell": cell})
            src_xy, src_floor = target, G.BOARD_FLOOR_Z               # pick it up again where we put it
            board_obs = board_obs                                       # (the calibration piece is the one we move)
    finally:
        if not arm.f.stop:
            arm.f.disconnect()
    OUT.write_text(json.dumps(samples, indent=1))
    # one affine map for desk + board
    desk_s = [s for s in json.loads((ROOT / "out" / "teach_samples.json").read_text()) if s["gap"] < 60]
    C = np.array([s["camera"] for s in desk_s] + [s["camera"] for s in samples])
    T = np.array([s["true"] for s in desk_s] + [s["arm"] for s in samples])
    if len(samples) >= 2:
        A = np.c_[C, np.ones(len(C))]
        M, *_ = np.linalg.lstsq(A, T, rcond=None)
        res = np.linalg.norm(A @ M - T, axis=1)
        log("affine residuals (mm): desk", res[:len(desk_s)].round(1), " board", res[len(desk_s):].round(1))
        (ROOT / "out" / "camera_to_arm_affine.json").write_text(json.dumps(
            {"M": M.tolist(), "samples": len(C), "note": "desk (hand-taught) + board (self-calibrated)"}, indent=1))
        log("saved combined camera->arm map")


if __name__ == "__main__":
    main()
