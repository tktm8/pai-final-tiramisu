"""Use the CURRENT right-arm pose (placed by hand) as the camera pose: photo -> corrected target -> gentle pick
-> place on a cell -> back to this pose. The pose is saved to out/camera_pose.json for later.

  python robot/pick_here.py [size] [cell]
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
    P = gp.planner
    ql = G.current_q("left")
    q_cam = G.current_q("right")
    (ROOT / "out" / "camera_pose.json").write_text(json.dumps({"right": q_cam.tolist()}))
    print("camera pose (deg):", np.degrees(q_cam).round(1))
    frame = topcam.grab(warmup=40)
    import cv2
    cam = topcam.BoardCam.load()
    pieces = sam_pieces.detect(frame, cam, 0, debug_path=ROOT / "out" / "pick_here_sam.jpg")
    print("SAM:", [("SML"[p["size"]], p["x"], p["y"]) for p in pieces])
    cand = [p for p in pieces if p["size"] == size]
    if not cand:
        raise SystemExit("no piece of that size seen")
    d, h = G.PIECE[size]
    a_dn = np.array([0.0, 0.0, -1.0])

    def plan_for(p):
        tx, ty = to_arm(p["x"], p["y"])
        others = [(np.array(to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in pieces if o is not p]
        chk = G.add_obstacles(P.checker_for("right", ql), others)
        for ang in (90, 0, 45, -45, 135, -135, 180, -90):
            c = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
            for extra in (24.0, 16.0):
                xy = np.array([tx, ty]) - c[:2] * (d + extra) / 2
                hov = np.array([xy[0], xy[1], W.HOVER_Z])
                tcp = np.array([xy[0], xy[1], h * G.GRASP_FRAC[size]])
                q_h = gp.k.ik_pose(hov, a_dn, c, q_cam, n_random=4)
                if q_h is None:
                    continue
                if P.line_path(chk, "right", q_h, hov, tcp, a_dn, c, G.gap_to_rad(d + extra)) is not None:
                    return dict(p=p, others=others, c=c, xy=xy, extra=extra, ang=ang, target=(tx, ty))
        return None

    choice = None
    for p in cand:
        choice = plan_for(p)
        if choice:
            break
        print(f"  skip piece at camera ({p['x']}, {p['y']}): jaws would hit a neighbour")
    if choice is None:
        raise SystemExit("every piece of that size is too close to its neighbours")
    p, others, c, xy, extra = choice["p"], choice["others"], choice["c"], choice["xy"], choice["extra"]
    print(f"target: camera ({p['x']}, {p['y']}) -> arm ({choice['target'][0]:.1f}, {choice['target'][1]:.1f}); "
          f"jaws close along {choice['ang']} deg, opening piece+{extra:.0f} mm")
    g0 = G.gap_to_rad(d + extra)
    f = Follower("right")
    try:
        f.run([Segment("open", [q_cam] * G.HOLD, g0)])
        pk = W.Picker(gp, f, ql, others)
        a = a_dn
        qh = pk.move_hover(xy, a, c, q_cam, g0)
        if qh is None:
            raise SystemExit("cannot reach above the piece")
        held, down = pk.try_grasp(qh, xy, a, c, size, g0)
        print("GRASP:", "HELD" if held else "missed")
        if held:
            g_hold = W.units_to_rad(max(0.0, pk.grip_units - 1.0))
            g_rel = G.gap_to_rad(d + 6.0)
            lift = list(reversed(down))
            # the board cell centre needs the same camera/world -> arm correction as the pick
            cx, cy = G.cell_xy(cell)
            ax_, ay_ = to_arm(cx, cy)
            G.RESERVE_SLOTS[98] = (size, ax_, ay_)
            print(f"place: cell {cell} centre ({cx:.1f}, {cy:.1f}) -> arm ({ax_:.1f}, {ay_:.1f})")
            # slot spots have floor z 0; the board floor is 4 mm, plus 8 mm release height
            place = gp.grasp_pose(pk.chk, G.Spot("slot", 98), size, 12.0, lift[-1], g_hold)
            if place is None:
                raise SystemExit("no place pose at the corrected cell centre")
            q_pl, a2, c2, tcp, _ = place
            hov, q_hov = gp.hover(tcp, a2, c2, q_pl)
            carry = P.staged_path(pk.chk, "right", lift[-1], q_hov, g_hold)
            lower = P.line_path(pk.chk, "right", q_hov, hov, tcp, a2, c2, g_hold)
            up = P.line_path(pk.chk, "right", lower[-1], tcp, hov, a2, c2, g_rel)
            f.run([Segment("lift", lift, g_hold), Segment("carry", carry, g_hold), Segment("lower", lower, g_hold),
                   Segment("release", [lower[-1]] * G.HOLD, g_rel), Segment("retreat", up, g_rel)])
            q_end = up[-1]
            print("PLACED on cell", cell)
        else:
            if down is not None:
                f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
            q_end = qh
        back = P.staged_path(pk.chk, "right", q_end, q_cam, g0) or P.joint_path(pk.chk, "right", q_end, q_cam, g0)
        if back is None:
            n = int(np.ceil(np.degrees(np.abs(q_cam - q_end).max()) / 0.5))
            back = [q_end + (q_cam - q_end) * i / n for i in range(1, n + 1)]
        f.run([Segment("back_to_camera_pose", back, g0)])
    finally:
        if not f.stop:
            f.disconnect()
        else:
            print("stopped: torque held")


if __name__ == "__main__":
    main()
