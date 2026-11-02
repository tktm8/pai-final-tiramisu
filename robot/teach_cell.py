"""Hand-taught board cell: a piece stands centred in cell N and the person closes the right gripper on it.
Reads joints passively and stores the jaws' midpoint (arm coordinates) for that cell in out/board_teach.json.

  python robot/teach_cell.py 7
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot")]
import gobblet_arm as G  # noqa: E402
from robot_io import read_raw  # noqa: E402
from kinematics import raw_to_q  # noqa: E402

cell = int(sys.argv[1])
raw = read_raw("right")
q = raw_to_q(raw)
tcp, a, c = G.GobbletPlanner().k.tcp(q)
g_rad = np.radians((raw["gripper"] - 2047) * 360.0 / 4095)
gap = 34.0 + (g_rad - 0.28) * (60.0 - 34.0) / (0.6 - 0.28)
mid = tcp + c * gap / 2
world = G.cell_xy(cell)
print(f"cell {cell}: world centre {world.round(1)}  jaws midpoint (arm) {mid.round(1)}  offset {(mid[:2] - world).round(1)}  "
      f"(gap ~{gap:.0f} mm, tcp z {tcp[2]:.0f})")
f = ROOT / "out" / "board_teach.json"
d = json.loads(f.read_text()) if f.exists() else {}
d[str(cell)] = {"arm": mid[:2].tolist(), "world": world.tolist(), "gap": gap}
f.write_text(json.dumps(d, indent=1))
