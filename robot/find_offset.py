"""Find the camera -> arm offset by trying grasps around the camera estimate (after the arm recalibration).

From the saved camera pose: photo -> SAM -> pick a medium piece with room around it -> try grasps at the camera
position + spiral offsets (jaws always along Y) until one holds -> put it back -> save the offset as a pure
translation in out/camera_to_arm_affine.json (used by move_test.to_arm for desk and board).

  python robot/find_offset.py [size]
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
import move_test as MT  # noqa: E402
import sam_pieces  # noqa: E402
import topcam  # noqa: E402
import wrist_servo as W  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402


def spiral():
    pts = [(0.0, 0.0)]
    for r, n in ((10.0, 6), (20.0, 10)):
        pts += [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    return pts


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gp = G.GobbletPlanner()
    P = gp.planner
    ql = G.current_q("left")
    q_cam = np.array(json.loads((ROOT / "out" / "camera_pose.json").read_text())["right"])
    d, h = G.PIECE[size]
    a = np.array([0.0, 0.0, -1.0])
    c = np.array([0.0, 1.0, 0.0])
    extra = 24.0
    g = G.gap_to_rad(d + extra)
    f = Follower("right")
    found = None
    try:
        MT.go_to_camera_pose(f, q_cam)
        time.sleep(0.5)
        frame = topcam.grab(warmup=40)
        cam = topcam.BoardCam.load()
        desk = sam_pieces.detect(frame, cam, 0)
        print("SAM:", [("SML"[p["size"]], p["x"], p["y"]) for p in desk])

        def clearance(p):
            return min([np.hypot(p["x"] - o["x"], p["y"] - o["y"]) for o in desk if o is not p] or [999])
        cands = sorted([p for p in desk if p["size"] == size], key=lambda p: -clearance(p))
        if not cands:
            raise SystemExit("no piece of that size")
        p = cands[0]
        raw = np.array([p["x"], p["y"]])
        est = np.array(MT.to_arm(*raw))          # search around the CURRENT corrected estimate
        others = [(np.array(MT.to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p]
        chk = G.add_obstacles(P.checker_for("right", ql), others, margin_mm=2.0)
        print(f"target {'SML'[size]} at camera {est.round(1)} (nearest other piece {clearance(p):.0f} mm)")
        pk = W.Picker(gp, f, ql, others)
        q_now = q_cam
        f.run([Segment("open", [q_cam] * G.HOLD, g)])
        for dx, dy in spiral():
            centre = est + np.array([dx, dy])
            xy = centre - c[:2] * (d + extra) / 2
            hov = np.array([xy[0], xy[1], W.HOVER_Z])
            tcp = np.array([xy[0], xy[1], h * G.GRASP_FRAC[size]])
            q_h = gp.k.ik_pose(hov, a, c, q_now, n_random=4)
            if q_h is None or P.line_path(chk, "right", q_h, hov, tcp, a, c, g) is None:
                print(f"  offset ({dx:+.0f},{dy:+.0f}): no clear way down, skipped (arm did not move)")
                continue
            way = MT.route(P, chk, q_now, q_h, g)
            if way is None:
                print(f"  offset ({dx:+.0f},{dy:+.0f}): no route, skipped (arm did not move)")
                continue
            if not f.run([Segment("to_hover", way, g)]):
                raise RuntimeError("arm stopped")
            q_now = q_h
            held, down = pk.try_grasp(q_h, xy, a, c, size, g)
            print(f"  offset ({dx:+.0f},{dy:+.0f}) -> {'HELD' if held else 'missed'}")
            if down is not None:
                f.run([Segment("open", [down[-1]] * G.HOLD, g), Segment("up", list(reversed(down)), g)])
            if held:
                found = (dx, dy)
                break
        back = MT.route(P, chk, q_now, q_cam, g)
        f.run([Segment("back_to_camera_pose", back or [q_cam], g)])
    finally:
        if not f.stop:
            f.disconnect()
    if found is None:
        print("RESULT: no offset held")
        return
    base = json.loads((ROOT / "out" / "camera_to_arm_affine.json").read_text())["M"] if (ROOT / "out" / "camera_to_arm_affine.json").exists() else [[1, 0], [0, 1], [0, 0]]
    M = [[1.0, 0.0], [0.0, 1.0], [float(base[2][0] + found[0]), float(base[2][1] + found[1])]]
    (ROOT / "out" / "camera_to_arm_affine.json").write_text(json.dumps(
        {"M": M, "note": f"translation found by grasp search ({time.strftime('%F %T')}), size {size}"}, indent=1))
    print(f"RESULT: extra offset ({found[0]:+.0f}, {found[1]:+.0f}) mm -> total translation ({M[2][0]:+.1f}, {M[2][1]:+.1f}) saved")


if __name__ == "__main__":
    main()
