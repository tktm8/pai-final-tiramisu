"""Hand-taught grasp sample: the person holds the open gripper around piece #N (from out/teach_pieces.json).
Reads joint angles passively (torque untouched), computes where the jaws' midpoint is, and stores
(camera estimate, true position) in out/teach_samples.json.

  python robot/teach_read.py 3
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot")]
import gobblet_arm as G  # noqa: E402
from robot_io import read_raw  # noqa: E402
from kinematics import raw_to_q, calibration  # noqa: E402

n = int(sys.argv[1])
pieces = json.loads((ROOT / "out" / "teach_pieces.json").read_text())
p = pieces[n]
raw = read_raw("right")
q = raw_to_q(raw)
k = G.GobbletPlanner().k
tcp, a, c = k.tcp(q)
cal = calibration("right")["gripper"]
g_rad = np.radians((raw["gripper"] - 2047) * 360.0 / 4095)
gap = 34.0 + (g_rad - 0.28) * (60.0 - 34.0) / (0.6 - 0.28)        # inverse of gap_to_rad
mid = tcp + c * gap / 2
est = np.array([p["x"], p["y"]])
off = mid[:2] - est
print(f"piece #{n} {'SML'[p['size']]}: camera {est.round(1)}  jaws midpoint {mid.round(1)}  (tcp z {tcp[2]:.0f}, "
      f"approach {a.round(2)}, jaw gap ~{gap:.0f} mm)")
print(f"offset true - camera = ({off[0]:+.1f}, {off[1]:+.1f}) mm")
f = ROOT / "out" / "teach_samples.json"
samples = json.loads(f.read_text()) if f.exists() else []
samples.append({"piece": n, "size": p["size"], "camera": est.tolist(), "true": mid[:2].tolist(), "offset": off.tolist(),
                "tcp": tcp.tolist(), "approach": a.tolist(), "gap": gap})
f.write_text(json.dumps(samples, indent=1))
