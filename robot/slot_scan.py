"""Scan desk positions right of the board: can the right arm grasp a piece there and lift it >= 30 mm?"""
import time
import numpy as np
import gobblet_arm as G
gp = G.GobbletPlanner()
q_home = G.home_q("right"); q_other = G.home_q("left")
chk = gp.planner.checker_for("right", q_other)
res = {}
for s in (2, 1, 0):
    for x in range(110, 341, 30):
        for y in range(-130, -251, -20):
            G.RESERVE_SLOTS[99] = (s, float(x), float(y))
            sp = G.Spot("slot", 99)
            g = gp.grasp_pose(chk, sp, s, 0.0, q_home, G.gap_to_rad(G.PIECE[s][0] + 14))
            ok = "-"
            if g is not None:
                p, qh = gp.hover(g[3], g[1], g[2], g[0])
                ok = ("V" if g[4].startswith("vertical") else "T") + (str(int(p[2])) if qh is not None else "x")
            res[(s, x, y)] = ok
    print(f"size {s}")
    print("      " + " ".join(f"{y:5d}" for y in range(-130, -251, -20)))
    for x in range(110, 341, 30):
        print(f"x{x:4d} " + " ".join(f"{res[(s, x, y)]:>5s}" for y in range(-130, -251, -20)), flush=True)
