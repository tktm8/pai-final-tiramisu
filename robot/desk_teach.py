"""Desk-piece offset per arm, taught by hand (SAM base estimate vs where the jaws really hold the piece).

  python robot/desk_teach.py snap --arm left   # photo -> SAM numbers the pieces on that side (out/desk_teach_<arm>.jpg)
  python robot/desk_teach.py 2 --arm left      # the person closes the gripper gently around piece #2 -> one sample
  python robot/desk_teach.py fit --arm left    # mean offset -> out/desk_offset_<arm>.json (used by move_test.desk_to_arm)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot"), str(ROOT / "perception")]
import gobblet_arm as G  # noqa: E402
import move_test as MT  # noqa: E402

ARM = "left" if "--arm" in sys.argv and sys.argv[sys.argv.index("--arm") + 1] == "left" else "right"
PIECES = ROOT / "out" / f"desk_teach_pieces_{ARM}.json"
SAMPLES = ROOT / "out" / f"desk_teach_samples_{ARM}.json"
OUT = ROOT / "out" / f"desk_offset_{ARM}.json"
MT.ARM[0] = ARM
arg = sys.argv[1]

if arg == "snap":
    import sam_pieces
    import topcam
    frame = topcam.grab(warmup=40)
    cam = topcam.BoardCam.load()
    ps = sam_pieces.detect(frame, cam, 0, region=MT.DESK_REGION[ARM])
    PIECES.write_text(json.dumps(ps, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    SAMPLES.write_text("[]")
    img = frame.copy()
    for i, p in enumerate(ps):
        u, v = map(int, p["px"])
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(img, f"#{i} {'SML'[p['size']]}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imwrite(str(ROOT / "out" / f"desk_teach_{ARM}.jpg"), img)
    for i, p in enumerate(ps):
        print(f"#{i} {'SML'[p['size']]}  camera ({p['x']:.0f}, {p['y']:.0f})")
elif arg == "fit":
    s = json.loads(SAMPLES.read_text())
    off = np.array([x["offset"] for x in s])
    m = off.mean(0)
    OUT.write_text(json.dumps({"dx": float(m[0]), "dy": float(m[1]),
                               "note": f"{ARM} arm, {len(s)} hand samples, spread {off.std(0).round(1).tolist()} mm"}, indent=1))
    for x in s:
        print(f"  #{x['piece']} {'SML'[x['size']]}: offset ({x['offset'][0]:+.1f}, {x['offset'][1]:+.1f})")
    print(f"[{ARM}] desk offset = ({m[0]:+.1f}, {m[1]:+.1f}) mm from {len(s)} samples, spread {off.std(0).round(1)}")
else:
    from kinematics import raw_to_q
    from robot_io import read_raw
    n = int(arg)
    p = json.loads(PIECES.read_text())[n]
    q = raw_to_q(read_raw(ARM))
    tcp, a, c = G.GobbletPlanner().planner.kin[ARM].tcp(q)
    d = G.PIECE[p["size"]][0]
    mid = tcp + c * d / 2                       # jaws closed on the piece: centre is half a diameter from the TCP jaw
    board_map = np.array(MT.to_arm(p["x"], p["y"]))   # camera -> arm with this arm's board map, no desk offset
    off = mid[:2] - board_map
    s = json.loads(SAMPLES.read_text()) if SAMPLES.exists() else []
    s = [x for x in s if x["piece"] != n] + [{"piece": n, "size": p["size"], "offset": off.tolist(), "mid": mid.tolist()}]
    SAMPLES.write_text(json.dumps(s, indent=1))
    print(f"[{ARM}] #{n} {'SML'[p['size']]}: jaws centre {mid[:2].round(1)}, camera->arm {board_map.round(1)}, "
          f"offset ({off[0]:+.1f}, {off[1]:+.1f}) mm, tilt {np.degrees(np.arccos(-a[2])):.0f} deg  [{len(s)} samples]")
