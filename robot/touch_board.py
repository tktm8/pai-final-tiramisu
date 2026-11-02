"""Board <-> arm registration by touching the 4 bar crossings with the CLOSED gripper tips (arm moved by hand).

  python robot/touch_board.py A      # read the current fingertip position for crossing A (B, C, D likewise)
  python robot/touch_board.py fit    # 2-D affine board(CAD) -> arm, written as the camera->arm map
  add --arm left for the left arm (own files: out/board_touch_left.json, out/camera_to_arm_affine_left.json)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot")]
import gobblet_arm as G  # noqa: E402
from kinematics import raw_to_q  # noqa: E402
from robot_io import read_raw  # noqa: E402

ARM = "left" if "--arm" in sys.argv and sys.argv[sys.argv.index("--arm") + 1] == "left" else "right"
SUF = "_left" if ARM == "left" else ""
PTS = json.loads((ROOT / "out" / "touch_points.json").read_text())
STORE = ROOT / "out" / f"board_touch{SUF}.json"
OUT = ROOT / "out" / f"camera_to_arm_affine{SUF}.json"
TIP_BELOW_TCP = 7.3          # mm: the fingertips reach this far past the TCP along the approach axis

arg = sys.argv[1]
data = json.loads(STORE.read_text()) if STORE.exists() else {}
if arg in PTS:
    q = raw_to_q(read_raw(ARM))
    tcp, a, c = G.GobbletPlanner().planner.kin[ARM].tcp(q)
    tip = tcp + a * TIP_BELOW_TCP
    data[arg] = {"arm": tip.tolist(), "board": PTS[arg]}
    STORE.write_text(json.dumps(data, indent=1))
    off = tip[:2] - np.array(PTS[arg])
    print(f"[{ARM}] {arg}: board {PTS[arg]} -> fingertip (arm) {tip.round(1)}  offset ({off[0]:+.1f}, {off[1]:+.1f})  "
          f"gripper tilt {np.degrees(np.arccos(-a[2])):.0f} deg  [{len(data)}/4 done]")
elif arg == "fit":
    keys = [k for k in "ABCD" if k in data]
    B = np.array([data[k]["board"] for k in keys])
    A = np.array([data[k]["arm"][:2] for k in keys])
    # full 2-D affine (rotation + scale per axis + shear + translation): the CAD bar spacing may be wrong
    X = np.c_[B, np.ones(len(B))]
    M, *_ = np.linalg.lstsq(X, A, rcond=None)
    res = np.linalg.norm(X @ M - A, axis=1)
    sx, sy = np.linalg.norm(M[0]), np.linalg.norm(M[1])
    ang = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
    OUT.write_text(json.dumps(
        {"M": M.tolist(), "note": f"board touch registration {keys}: scale x {sx:.3f} y {sy:.3f}, rot {ang:.2f} deg"},
        indent=1))
    print(f"[{ARM}] fit from {keys}: scale x {sx:.3f}, y {sy:.3f}, rotation {ang:+.2f} deg, translation {M[2].round(1)}; "
          f"residuals {res.round(1)} mm")
    for k in keys:
        print(f"  {k}: A-B style check  board {data[k]['board']} -> arm {np.round(data[k]['arm'][:2], 1)}")
