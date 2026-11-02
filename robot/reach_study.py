import numpy as np
import syspath  # noqa: F401
from config import ARMS
from kinematics import SO101

CX = [138.4, 208.4, 278.4]
CY = [70.0, 0.0, -70.0]
REST = np.zeros(5)
for arm in ("right", "left"):
    k = SO101(ARMS[arm]["base"])
    print(f"== {arm}")
    for z in (20.0, 40.0, 60.0):
        row = []
        for x in CX:
            for y in CY:
                best = "--"
                if k.ik_pose([x, y, z], [0, 0, -1], [0, 1, 0], REST, n_random=6) is not None or \
                   k.ik_pose([x, y, z], [0, 0, -1], [1, 0, 0], REST, n_random=6) is not None:
                    best = "V"
                else:
                    for t in (15, 30, 45, 60):
                        if k.ik_tilt(np.array([x, y, z]), t, REST, lean_away_from=ARMS[arm]["base"][:2], n_random=6) is not None:
                            best = f"T{t}"; break
                row.append(f"({x:.0f},{y:+.0f}):{best}")
        print(f" z={z:.0f}", " ".join(row))
