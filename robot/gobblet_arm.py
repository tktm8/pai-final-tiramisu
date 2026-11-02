"""Gobblet pick & place with the right SO-101, stylus taps with the left one.

Reuses ~/Programing/so-101/system (URDF IK, voxel collision checking against the Gobblet assembly,
LeRobot follower with a per-command clamp and stall detection).

  python robot/gobblet_arm.py check                 # plan every cell / reserve slot, no motion
  python robot/gobblet_arm.py move R0 4 --size 2    # plan reserve slot 0 -> cell 4 (add --execute to move)
  python robot/gobblet_arm.py teach-tap             # record the stylus hover / touch poses (arm moved by hand)
  python robot/gobblet_arm.py tap [--execute]
"""

from __future__ import annotations

import argparse
import warnings
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # spurious matmul warnings from numpy + Accelerate
sys.path.insert(0, str(Path(__file__).resolve().parent))
import syspath  # noqa: E402,F401  (patches so-101/system config first)

from collision import Checker, Grid  # noqa: E402
from config import ARMS, MAX_STEP_DEG, OUT, VOXEL  # noqa: E402
from kinematics import raw_to_q  # noqa: E402
from planner import Planner, Segment  # noqa: E402

PIECE_ARM = "right"
STYLUS_ARM = "left"

# ---------------- geometry (world mm) ----------------
CELL_X = [138.4, 208.4, 278.4]
CELL_Y = [70.0, 0.0, -70.0]
BOARD_FLOOR_Z = 4.0
DESK_Z = 0.0

# Gobblet Gobblers pieces: (body diameter, total height). MEASURE the real pieces and update.
PIECE = {0: (27.0, 33.0), 1: (36.0, 45.0), 2: (46.0, 58.0)}
GRASP_FRAC = {0: 0.45, 1: 0.35, 2: 0.35}   # grip height / piece height, per size (below the head and its fins)

# Robot reserve slots on the desk, right of the board (outside the right arm's rest footprint).
# slot index -> (size, x, y). Two of each size. Check with `check`.
RESERVE_SLOTS = {
    0: (2, 175.0, -175.0), 1: (2, 245.0, -175.0),     # large
    2: (1, 150.0, -240.0), 3: (1, 210.0, -240.0),     # medium
    4: (0, 270.0, -240.0), 5: (0, 290.0, -135.0),     # small
}

# LeRobot clamps each command to 5 units and the runner re-sends the clamped value while settling, so a
# one-step gripper change stalls. Repeating the pose for HOLD steps re-sends the real target every frame.
HOLD = 30
Z_TRAVEL = 95.0              # TCP height for carrying (a vertical gripper tops out near 100 mm; large pieces are 58 mm)
GRIPPER_CLOSE_SQUEEZE = 6.0  # command the jaws this many mm narrower than the piece


def gap_to_rad(gap_mm: float) -> float:
    """Jaw gap (mm) -> URDF gripper angle. Two points from the Connect 4 work: 0.28 rad ~ 34 mm, 0.6 rad ~ 60 mm."""
    return 0.28 + (gap_mm - 34.0) * (0.6 - 0.28) / (60.0 - 34.0)


def hover_of(tcp, approach):
    """Back off along the approach axis until the TCP is at Z_TRAVEL."""
    return tcp - approach * (Z_TRAVEL - tcp[2]) / max(1e-3, -approach[2])


def cell_xy(c: int):
    r, k = divmod(c, 3)
    return np.array([CELL_X[r], CELL_Y[k]])


@dataclass
class Spot:
    """Where a piece stands: a board cell or a reserve slot."""
    kind: str        # "cell" | "slot"
    index: int

    @property
    def xy(self):
        return cell_xy(self.index) if self.kind == "cell" else np.array(RESERVE_SLOTS[self.index][1:])

    @property
    def floor(self):
        return BOARD_FLOOR_Z if self.kind == "cell" else DESK_Z

    def __str__(self):
        return f"cell{self.index}" if self.kind == "cell" else f"R{self.index}"


def parse_spot(s: str) -> Spot:
    return Spot("slot", int(s[1:])) if s.upper().startswith("R") else Spot("cell", int(s))


# ---------------- obstacles: the other pieces ----------------
def piece_points(xy, floor, size, spacing=VOXEL * 0.7):
    d, h = PIECE[size]
    r = d / 2
    pts = []
    for z in np.arange(floor, floor + h, spacing):
        n = max(8, int(2 * np.pi * r / spacing))
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.c_[xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), np.full(n, z)])
    gx, gy = np.meshgrid(np.arange(-r, r, spacing), np.arange(-r, r, spacing))
    m = gx ** 2 + gy ** 2 <= r * r
    pts.append(np.c_[xy[0] + gx[m], xy[1] + gy[m], np.full(m.sum(), floor + h)])
    return np.concatenate(pts)


def add_obstacles(chk: Checker, pieces, margin_mm=4.0):
    """pieces: [(xy, floor, size)] -> grown into the arm and jaw grids."""
    if not pieces:
        return chk
    g = Grid()
    for xy, floor, size in pieces:
        g.add_points(piece_points(np.asarray(xy, float), floor, size))
    grown = g.dilated(int(np.ceil(margin_mm / VOXEL)))
    chk.env_arm |= grown
    chk.env_grip |= grown
    return chk


# ---------------- planning ----------------
class GobbletPlanner:
    def __init__(self):
        self.planner = Planner(Checker(exclude_prefixes=("06_", "14_")))
        rrt = self.planner.rrt_connect
        self.planner.rrt_connect = lambda *a, **k: rrt(*a, **{**k, "iters": k.get("iters", 600)})
        self.k = self.planner.kin[PIECE_ARM]

    def _grasp_candidates(self):
        axes = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.radians([90, -90, 0, 180, 45, -45, 135, -135])]
        return [("vertical", c, 0) for c in axes] + [("tilt", None, t) for t in (15, 25, 35)]

    def grasp_pose(self, chk, spot: Spot, size: int, z_lift: float, q_seed, gripper, centred_gap: float | None = None):
        """IK + collision check for holding a piece of `size` standing at `spot` (z_lift above its floor).
        Returns (q, approach, close_axis, tcp, label) or None."""
        d, h = PIECE[size]
        # picking: piece in the middle of the open jaws (both jaws clear it on the way down);
        # placing: the held piece sits against the fixed jaw (the moving jaw pushed it there)
        off = centred_gap / 2 if centred_gap else d / 2 + 0.5
        centre = np.r_[spot.xy, spot.floor + h * GRASP_FRAC[size] + z_lift]
        base_xy = ARMS[PIECE_ARM]["base"][:2]
        for kind, c_req, tilt in self._grasp_candidates():
            if c_req is not None:
                a = np.array([0.0, 0.0, -1.0])
                tcp = centre - c_req * off
                q = self.k.ik_pose(tcp, a, c_req, q_seed, n_random=4)
                c = c_req
            else:
                tcp, q = centre.copy(), None
                for _ in range(3):
                    q = self.k.ik_tilt(tcp, tilt, q_seed, lean_away_from=base_xy, n_random=4)
                    if q is None:
                        break
                    _, a, c = self.k.tcp(q)
                    tcp = centre - c * off
                if q is not None:
                    _, a, c = self.k.tcp(q)
            if q is None:
                continue
            if self.planner.free(chk, PIECE_ARM, q, gripper):
                return q, a, c, tcp, f"{kind} {tilt}"
        return None

    def hover(self, tcp, a, c, q_seed, min_dz=30.0):
        """Highest reachable pose straight back along the approach axis (up to Z_TRAVEL, at least min_dz)."""
        top = max(min_dz, Z_TRAVEL - tcp[2])
        for dz in np.arange(top, min_dz - 1e-6, -10.0):
            p = tcp - a * dz / max(1e-3, -a[2])
            q = self.k.ik_pose(p, a, c, q_seed, n_random=1)
            if q is not None:
                return p, q
        return None, None

    def held_piece_clear(self, obst_grid, path, size, gripper_closed=True):
        """The carried piece (a cylinder hanging between the jaws) must not touch other pieces."""
        d, h = PIECE[size]
        local = piece_points(np.zeros(2), -h * GRASP_FRAC[size], size, spacing=4.0)   # centre at origin, grasp height 0
        for q in path[::3] + [path[-1]]:
            p, a, c = self.k.tcp(q)
            centre = p + c * (d / 2 + 0.5)
            if obst_grid.hits(local + centre, obst_grid.occ_grown).any():
                return False
        return True

    def plan_transfer(self, src: Spot, dst: Spot, size: int, others, q_home, q_other):
        """Pick the top piece (size) at src and put it on dst. others: obstacles [(xy, floor, size)] excluding
        the moving piece and anything at dst (it ends up inside the cup). Returns ([Segment], info) or (None, why)."""
        P = self.planner
        chk = add_obstacles(P.checker_for(PIECE_ARM, q_other), others)
        obst = Grid()
        for xy, floor, sz in others:
            obst.add_points(piece_points(np.asarray(xy, float), floor, sz))
        obst.occ_grown = obst.dilated(1)
        d, h = PIECE[size]
        PICK_GAP = d + 24.0
        g_open = gap_to_rad(PICK_GAP)
        g_release = gap_to_rad(d + 6.0)
        g_closed = -0.10      # close fully: the jaws stop on the piece (real sizes are smaller than PIECE guesses)

        # starting from the folded camera-clear pose: unfold upwards first (the way plan_to_look came in)
        q_look = None
        if np.degrees(q_home[0]) > 55:
            q_look = q_home
            q_home = q_look.copy()
            q_home[1:4] = np.radians(LOOK_RAISED_DEG)
        pick = self.grasp_pose(chk, src, size, 0.0, q_home, g_open, centred_gap=PICK_GAP)
        if pick is None:
            return None, f"no collision-free grasp at {src}"
        q_pick, a, c, tcp_pick, label = pick
        hover_pick, q_hp = self.hover(tcp_pick, a, c, q_pick)
        if q_hp is None:
            return None, f"no hover pose above {src}"
        to_hover = P.staged_path(chk, PIECE_ARM, q_home, q_hp, g_open)
        if to_hover is None:
            return None, f"home -> above {src} blocked"
        down = P.line_path(chk, PIECE_ARM, q_hp, hover_pick, tcp_pick, a, c, g_open)
        if down is None:
            return None, f"descent at {src} blocked"
        up = P.line_path(chk, PIECE_ARM, down[-1], tcp_pick, hover_pick, a, c, g_closed)
        if up is None or not self.held_piece_clear(obst, up, size):
            return None, f"lift at {src} blocked"

        place = self.grasp_pose(chk, dst, size, 8.0, up[-1], g_closed)   # release 8 mm up: lower grip, jaws clear the floor
        if place is None:
            return None, f"no collision-free place pose at {dst}"
        q_place, a2, c2, tcp_place, label2 = place
        hover_place, q_hq = self.hover(tcp_place, a2, c2, q_place)
        if q_hq is None:
            return None, f"no hover pose above {dst}"
        carry = P.staged_path(chk, PIECE_ARM, up[-1], q_hq, g_closed)
        if carry is None:
            return None, f"carry {src} -> {dst} blocked"
        if not self.held_piece_clear(obst, carry, size):
            return None, f"carried piece would hit another piece on the way {src} -> {dst}"
        down2 = P.line_path(chk, PIECE_ARM, q_hq, hover_place, tcp_place, a2, c2, g_closed)
        if down2 is None or not self.held_piece_clear(obst, down2, size):
            return None, f"lowering at {dst} blocked"
        up2 = P.line_path(chk, PIECE_ARM, down2[-1], tcp_place, hover_place, a2, c2, g_release)
        if up2 is None:
            return None, f"retreat at {dst} blocked"
        home = P.staged_path(chk, PIECE_ARM, up2[-1], q_home, g_release)
        if home is None:
            return None, f"{dst} -> home blocked"
        pre, post = [], []
        if q_look is not None:
            unfold = P.joint_path(chk, PIECE_ARM, q_look, q_home, g_release)
            fold = P.joint_path(chk, PIECE_ARM, q_home, q_look, g_release)
            if unfold is None or fold is None:
                return None, "cannot unfold from the camera-clear pose"
            pre, post = [Segment("unfold", unfold, g_release)], [Segment("fold", fold, g_release)]
        segs = pre + [
            Segment("open", [q_home] * HOLD, g_open),
            Segment("to_above", to_hover, g_open),
            Segment("descend", down, g_open),
            Segment("close", [down[-1]] * HOLD, g_closed),
            Segment("lift", up, g_closed),
            Segment("carry", carry, g_closed),
            Segment("lower", down2, g_closed),
            Segment("release", [down2[-1]] * HOLD, g_release),
            Segment("retreat", up2, g_release),
            Segment("to_home", home, g_release),
        ] + post
        return segs, dict(pick=label, place=label2, steps=sum(len(s.qs) for s in segs),
                          hover_z=(round(float(hover_pick[2])), round(float(hover_place[2]))))


def obstacles_from(stacks, reserve_used, moving_src: Spot, dst: Spot):
    """stacks: per cell [(player, size)] bottom->top; reserve_used: set of slot indexes still holding a piece."""
    out = []
    for c in range(9):
        if dst.kind == "cell" and c == dst.index:
            continue
        st = stacks[c]
        if moving_src.kind == "cell" and c == moving_src.index:
            st = st[:-1]
        if st:
            out.append((cell_xy(c), BOARD_FLOOR_Z, max(s for _, s in st)))
    for i in reserve_used:
        if moving_src.kind == "slot" and i == moving_src.index:
            continue
        s, x, y = RESERVE_SLOTS[i]
        out.append((np.array([x, y]), DESK_Z, s))
    return out


# ---------------- execution ----------------
HOME_FILE = OUT / "home_pose.json"
TAP_FILE = OUT / "tap_pose.json"


def current_q(arm):
    from robot_io import read_raw
    return raw_to_q(read_raw(arm))


def home_q(arm=PIECE_ARM):
    if HOME_FILE.exists():
        d = json.loads(HOME_FILE.read_text())
        if arm in d:
            return np.array(d[arm])
    return current_q(arm)


def run_segments(arm, segs, log=print):
    from robot_io import Follower
    f = Follower(arm)
    try:
        ok = f.run(segs, log=log)
        return ok
    finally:
        if not f.stop:
            f.disconnect()
        else:
            log("torque left ON so the arm does not drop")


LEFT_TUCK_PAN_DEG = -30.0   # swing the stylus arm outward so the right arm can reach the left cells


def tuck_left_segments():
    q = current_q(STYLUS_ARM)
    rest = home_q(STYLUS_ARM) if HOME_FILE.exists() else q
    target = rest.copy()
    target[0] = rest[0] + np.radians(LEFT_TUCK_PAN_DEG)
    if abs(q[0] - target[0]) < np.radians(3):
        return []
    n = max(1, int(np.ceil(np.degrees(np.abs(target - q).max()) / MAX_STEP_DEG)))
    return [Segment("tuck", [q + (target - q) * i / n for i in range(1, n + 1)], 0.0)]


def tap_segments(q_home):
    d = json.loads(TAP_FILE.read_text())
    hover, touch = np.array(d["hover"]), np.array(d["touch"])

    def interp(a, b):
        n = max(1, int(np.ceil(np.degrees(np.abs(b - a).max()) / MAX_STEP_DEG)))
        return [a + (b - a) * i / n for i in range(1, n + 1)]

    g = d.get("gripper_rad", 0.0)
    return [
        Segment("to_hover", interp(q_home, hover), g),
        Segment("touch", interp(hover, touch), g),
        Segment("hold", [touch] * 6, g),
        Segment("lift", interp(touch, hover), g),
        Segment("to_home", interp(hover, q_home), g),
    ]


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "move", "teach-tap", "tap", "save-home"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--size", type=int, default=2)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    if a.cmd == "save-home":
        d = json.loads(HOME_FILE.read_text()) if HOME_FILE.exists() else {}
        for arm in ARMS:
            d[arm] = current_q(arm).tolist()
        HOME_FILE.write_text(json.dumps(d, indent=1))
        print("saved current poses as home:", {k: np.degrees(v).round(1).tolist() for k, v in d.items()})
        return

    if a.cmd == "teach-tap":
        print("Torque must be OFF on the left arm (it is, unless a game is running).")
        input("1) Hold the stylus ~2 cm ABOVE the phone's button, then press Enter ")
        hover = current_q(STYLUS_ARM)
        input("2) Now let the stylus tip rest ON the button (screen registers a touch), press Enter ")
        touch = current_q(STYLUS_ARM)
        TAP_FILE.write_text(json.dumps({"hover": hover.tolist(), "touch": touch.tolist(), "gripper_rad": 0.0,
                                        "saved": time.strftime("%F %T")}, indent=1))
        print("saved", TAP_FILE)
        return

    if a.cmd == "tap":
        segs = tap_segments(home_q(STYLUS_ARM))
        print("tap:", [(s.label, len(s.qs)) for s in segs])
        if a.execute:
            run_segments(STYLUS_ARM, segs)
        return

    gp = GobbletPlanner()
    q_home = home_q(PIECE_ARM)
    q_other = home_q(STYLUS_ARM)
    if a.cmd == "check":
        empty = [[] for _ in range(9)]
        for slot, (s, x, y) in RESERVE_SLOTS.items():
            for c in range(9):
                t = time.time()
                segs, info = gp.plan_transfer(Spot("slot", slot), Spot("cell", c), s,
                                              obstacles_from(empty, set(RESERVE_SLOTS) - {slot}, Spot("slot", slot),
                                                             Spot("cell", c)), q_home, q_other)
                print(f"R{slot}(size {s}) -> cell{c}: {'OK ' + str(info) if segs else 'FAIL ' + info}  {time.time() - t:.1f}s")
        return

    src, dst = parse_spot(a.src), parse_spot(a.dst)
    used = set(RESERVE_SLOTS) - ({src.index} if src.kind == "slot" else set())
    segs, info = gp.plan_transfer(src, dst, a.size, obstacles_from([[] for _ in range(9)], used, src, dst), q_home, q_other)
    if segs is None:
        raise SystemExit(f"no plan: {info}")
    print(info)
    for s in segs:
        print(f"  {s.label:10s} {len(s.qs):4d} steps  end q[deg] {np.degrees(s.qs[-1]).round(1)}  gripper {s.gripper:+.2f}")
    if a.execute:
        print("done" if run_segments(PIECE_ARM, segs) else "stopped")


if __name__ == "__main__":
    main()


# ---------------- camera-clear "look" pose for the right arm ----------------
LOOK_PAN_DEG = 72.0          # folded rest shape swung out to the right; 110 got physically blocked on the real rig
LOOK_RAISED_DEG = (-60.0, 76.0, 42.0)   # lift / elbow / wrist of the raised waypoint (elbow in front of the base)
KEEPOUT_Z = 66.0             # above the tallest piece (58 mm) with a margin


def keepout_points():
    """Boxes where pieces may stand (board + desk right of it), as obstacle points up to KEEPOUT_Z."""
    boxes = [((87, 317), (-100, 100)),            # board (pieces reach Y +-93)
             ((95, 250), (-150, -280)),           # desk right of the board (leaves the resting gripper free)
             ((250, 345), (-100, -280))]
    pts = []
    for (x0, x1), (y0, y1) in boxes:
        xs = np.arange(x0, x1, 4.0)
        ys = np.arange(min(y0, y1), max(y0, y1), 4.0)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, KEEPOUT_Z)])   # lid
    return np.concatenate(pts)


def plan_to_look(gp: "GobbletPlanner", q_now, q_other):
    """Joint path from q_now (normally the folded rest pose) to the look pose: raise, swing, lower."""
    P = gp.planner
    chk = P.checker_for(PIECE_ARM, q_other)
    g = Grid()
    g.add_points(keepout_points())
    lid = g.dilated(2)
    chk.env_arm |= lid
    chk.env_grip |= lid
    q_look = q_now.copy()
    q_look[0] = np.radians(LOOK_PAN_DEG)
    # raised waypoint: same pan as now, gripper lifted up and back
    best = None
    # elbow must stay in front of the base: with the shoulder fully back the elbow swings into the camera pole
    for lift, elbow, wrist in [LOOK_RAISED_DEG]:
        q_up = q_now.copy()
        q_up[1:4] = np.radians([lift, elbow, wrist])
        a = P.joint_path(chk, PIECE_ARM, q_now, q_up, 0.0)
        if a is None:
            continue
        q_up2 = q_up.copy()
        q_up2[0] = q_look[0]
        b = P.joint_path(chk, PIECE_ARM, q_up, q_up2, 0.0) if a is not None else None
        c = P.joint_path(chk, PIECE_ARM, q_up2, q_look, 0.0) if b is not None else None
        if c is not None:
            best = (a, b, c)
            break
    if best is None:
        return None, q_look
    a, b, c = best
    return [Segment("raise", a, 0.0), Segment("swing_out", b, 0.0), Segment("lower", c, 0.0)], q_look


def return_to_look(gp: "GobbletPlanner", f=None, others=()):
    """From any pose: lift to the raised waypoint (pan of the look pose), then fold into the camera-clear pose."""
    P = gp.planner
    ql = current_q(STYLUS_ARM)
    q = current_q(PIECE_ARM)
    q_look = q.copy()
    q_look[0] = np.radians(LOOK_PAN_DEG)
    q_look[1:5] = np.radians([-99.0, 88.0, 66.0, 1.3])     # folded rest shape, wrist roll at rest
    q_up = q_look.copy()
    q_up[1:4] = np.radians(LOOK_RAISED_DEG)
    chk = add_obstacles(P.checker_for(PIECE_ARM, ql), list(others))
    if abs(q[0] - q_look[0]) < np.radians(15) and abs(q[1] - q_look[1]) < np.radians(15):
        return True
    q_lift = q.copy()
    q_lift[1:4] = np.radians(LOOK_RAISED_DEG)                # raise in place first (keeps the pan)
    q_lift[4] = q_look[4]
    a = P.joint_path(chk, PIECE_ARM, q, q_lift, 0.0) or P.staged_path(chk, PIECE_ARM, q, q_lift, 0.0)
    if a is None:     # the start pose itself touches something (arm slumped on the desk): lifting only moves away
        n = max(1, int(np.ceil(np.degrees(np.abs(q_lift - q).max()) / MAX_STEP_DEG)))
        a = [q + (q_lift - q) * i / n for i in range(1, n + 1)]
    b = P.joint_path(chk, PIECE_ARM, q_lift, q_up, 0.0) or P.staged_path(chk, PIECE_ARM, q_lift, q_up, 0.0)
    c = P.joint_path(P.checker_for(PIECE_ARM, ql), PIECE_ARM, q_up, q_look, 0.0)
    if a is None or b is None or c is None:
        print("return_to_look: no path", a is None, b is None, c is None)
        return False
    segs = [Segment("lift", a, 0.0), Segment("swing", b, 0.0), Segment("fold", c, 0.0)]
    if f is not None:
        return f.run(segs)
    return run_segments(PIECE_ARM, segs)
