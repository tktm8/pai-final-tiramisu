"""General robot move for the game, from the hand-placed camera pose (out/camera_pose.json).

  python robot/move_test.py R2 5        # a LARGE orange piece from the desk -> cell 5 (gobbles what is there)
  python robot/move_test.py 4 5 1       # the top piece of cell 4 (size 1) -> cell 5
Sizes: 0 small, 1 medium, 2 large. Board pieces are picked at their (corrected) cell centre.
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
import wrist_refine as WR  # noqa: E402
import dagger as DG  # noqa: E402
from planner import Segment  # noqa: E402
from robot_io import Follower  # noqa: E402

BOARD_FLOOR = G.BOARD_FLOOR_Z
BAR_TOP_Z = 23.5         # top of the board's cell bars (fingertips touching the crossings, 2026-11-01)
TIP_BELOW_TCP = 7.3      # fingertips reach this far below the TCP


# After the LeRobot recalibration of the right arm (2026-10-31 23:43) the old hand-taught corrections were
# moved to out/old_corrections/. With no correction file the camera position is used as is.
_CORR = ROOT / "out" / "camera_to_arm_affine.json"


# BOLD: do not treat the other pieces as obstacles at all (user, 2026-11-01: "no more safety checks").
# Only the fixed rig (frame, StackChan, the other arm) and joint limits are still checked; a nudged piece is OK.
BOLD = True
if BOLD:
    G.add_obstacles = lambda chk, pieces, margin_mm=4.0: chk
WORK_PAN = 72.0          # right arm; the left arm mirrors it (-72)
ARM = ["right"]          # the arm the current move() uses
OTHER = {"right": "left", "left": "right"}
SIDE = {"right": 1.0, "left": -1.0}
DESK_REGION = {"right": ((80.0, 345.0), (-300.0, -100.0)), "left": ((80.0, 345.0), (100.0, 300.0))}
GRIP_DZ = [0.0]          # how much higher than usual the current piece was gripped
G.Z_TRAVEL = 150.0      # hover / carry height above the place cell (the arm uses the highest it can reach)
FINE_ALIGN = True       # reserve picks: reachable-hover search (+ optional wrist-camera alignment)
USE_WRIST_CAMERA = False   # wrist alignment needs a correct reference point per size first
# DAgger (robot/dagger.py): a trained policy replaces the image-Jacobian step and every desk pick is recorded.
# robot/dagger_collect.py also sets an expert (leader arm), pick-only mode and a deliberately wrong start hover.
DAGGER = {"use_policy": True, "record": True, "expert": None, "pick_only": False, "hover_noise": 0.0}


def fix_left_z(P):
    """Left arm only: its kinematics reads ~4 mm higher than the right arm on the same surface (board touch,
    2026-11-01). Shift IK targets up and TCP readings down by that plane so heights match the right arm."""
    f = ROOT / "out" / "left_z_correction.json"
    k = P.kin["left"]
    if not f.exists() or getattr(k, "_z_fixed", False):
        return
    a, b, c0 = json.loads(f.read_text())["plane"]
    dz = lambda x, y: float(np.clip(a * x + b * y + c0, -10.0, 10.0))
    ik0, tcp0 = k.ik_pose, k.tcp

    def ik_pose(target_mm, *args, **kw):
        t = np.array(target_mm, float).copy()
        t[2] += dz(t[0], t[1])
        return ik0(t, *args, **kw)

    def tcp(q):
        pos, ax, cl = tcp0(q)
        pos = np.array(pos, float).copy()
        pos[2] -= dz(pos[0], pos[1])
        return pos, ax, cl
    k.ik_pose, k.tcp, k._z_fixed = ik_pose, tcp, True


def to_arm(x, y):
    """Camera/world (mm) -> arm target. Per-arm board-touch map (left: camera_to_arm_affine_left.json if present)."""
    corr = _CORR
    if ARM[0] == "left" and (ROOT / "out" / "camera_to_arm_affine_left.json").exists():
        corr = ROOT / "out" / "camera_to_arm_affine_left.json"
    if corr.exists():
        v = np.array([x, y, 1.0]) @ np.array(json.loads(corr.read_text())["M"])
        return float(v[0]), float(v[1])
    return float(x), float(y)


_DESK = ROOT / "out" / "desk_offset.json"


def desk_to_arm(x, y):
    """Desk pieces: board map + the constant bias of the SAM base estimate (measured with M and L by hand)."""
    ax, ay = to_arm(x, y)
    per_arm = ROOT / "out" / f"desk_offset_{ARM[0]}.json"
    src = per_arm if per_arm.exists() else _DESK
    o = json.loads(src.read_text()) if src.exists() else {"dx": 0.0, "dy": 0.0}
    return ax + o["dx"], ay + o["dy"]


def _desk_file():
    return ROOT / "out" / f"desk_offset_{ARM[0]}.json"


LEARN = False     # user, 2026-11-01 19:10: keep the saved 19:00 parameters; never rewrite them during play


def learn_desk_offset(dx, dy, log=print, gain=0.25):
    """A nudge (dx, dy) fixed a missed desk grasp: move this arm's desk offset half way towards it (and
    remember the direction, so the next miss tries it first)."""
    if not LEARN:
        return
    f = _desk_file()
    o = json.loads((f if f.exists() else _DESK).read_text()) if (f.exists() or _DESK.exists()) else {"dx": 0.0, "dy": 0.0}
    o["dx"] = float(o["dx"] + gain * dx)
    o["dy"] = float(o["dy"] + gain * dy)
    o["last_nudge"] = [dx, dy]
    o["note"] = f"{ARM[0]} arm, updated from a successful nudge at {time.strftime('%F %T')}"
    f.write_text(json.dumps(o, indent=1))
    log(f"  desk offset for the {ARM[0]} arm -> ({o['dx']:+.1f}, {o['dy']:+.1f}) mm (learned from the nudge)")


def nudge_order():
    base = [(8.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (0.0, -8.0), (6.0, 6.0), (6.0, -6.0), (-6.0, 6.0), (-6.0, -6.0)]
    f = _desk_file()
    last = tuple(json.loads(f.read_text()).get("last_nudge", ())) if f.exists() else ()
    return ([last] + [b for b in base if b != last]) if last in base else base


def route(P, chk, q_from, q_to, g):
    """Joint/staged path; else via the raised waypoint (unfold in place, proven on the rig) then staged."""
    p = P.joint_path(chk, ARM[0], q_from, q_to, g) or P.staged_path(chk, ARM[0], q_from, q_to, g)
    if p is not None:
        return p
    segs, q_up = W.joint_seq(q_from, W.UNFOLD, g)
    rest = P.joint_path(chk, ARM[0], q_up, q_to, g) or P.staged_path(chk, ARM[0], q_up, q_to, g)
    if rest is None:
        return None
    return [q for sg in segs for q in sg.qs] + rest


def place_options(gp, chk, size, dst, seed_q, g_hold):
    """Every collision-free way to hold the piece at the place pose (vertical with the jaws along Y, X, diagonal,
    then tilted): the caller picks the first one whose descent AND release clear the neighbours."""
    cx, cy = G.cell_xy(dst)
    ax_, ay_ = to_arm(cx, cy)
    G.RESERVE_SLOTS[98] = (size, ax_, ay_)
    orig = gp._grasp_candidates
    try:
        for cand in orig():
            gp._grasp_candidates = lambda cand=cand: [cand]
            place = gp.grasp_pose(chk, G.Spot("slot", 98), size, BOARD_FLOOR + 8.0 + GRIP_DZ[0], seed_q, g_hold)
            if place is not None:
                q_pl, a2, c2, tcp, label = place
                yield dict(q_pl=q_pl, a=a2, c=c2, tcp=tcp, label=label, arm_xy=(ax_, ay_))
    finally:
        gp._grasp_candidates = orig


def plan_place(gp, chk, size, dst, seed_q, g_hold):
    """Pose to set the held piece down on cell `dst` (corrected), plus its hover. Tries straight down first and
    then a tilted gripper (15-35 deg): far cells (e.g. 6) cannot be approached vertically from above, but a person
    reaching there tilts the hand too. None if impossible."""
    cx, cy = G.cell_xy(dst)
    ax_, ay_ = to_arm(cx, cy)
    G.RESERVE_SLOTS[98] = (size, ax_, ay_)
    orig = gp._grasp_candidates
    try:
        for cand in orig():
            gp._grasp_candidates = lambda cand=cand: [cand]
            place = gp.grasp_pose(chk, G.Spot("slot", 98), size, BOARD_FLOOR + 8.0 + GRIP_DZ[0], seed_q, g_hold)
            if place is None:
                continue
            q_pl, a2, c2, tcp, label = place
            for min_dz in (30.0, 15.0):
                hov, q_hov = gp.hover(tcp, a2, c2, q_pl, min_dz=min_dz)
                if q_hov is not None:
                    return dict(q_pl=q_pl, a=a2, c=c2, tcp=tcp, hov=hov, q_hov=q_hov, label=label, arm_xy=(ax_, ay_))
    finally:
        gp._grasp_candidates = orig
    return None


def high_lift(gp, chk, q_low, a, c, g):
    """After the normal lift: keep going straight up as high as the arm reaches (12 -> 11 -> 10 cm TCP)."""
    p0 = gp.k.tcp(q_low)[0]
    for z in (200.0, 185.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0):
        if z <= p0[2] + 5:
            break
        q_hi = gp.k.ik_pose(np.array([p0[0], p0[1], z]), a, c, q_low, n_random=0)
        if q_hi is None:
            continue
        path = gp.planner.line_path(chk, ARM[0], q_low, p0, np.array([p0[0], p0[1], z]), a, c, g)
        if path:
            return path
    return []


def carry_high(gp, chk, q_top, hov, q_hov, a, c, g):
    """Carry through the high pose (TCP ~15 cm, the same pose the way home uses): raise there at the pick pan,
    turn to the place pan up high, then come down to the place hover. The vertical gripper alone tops out near
    10 cm, which leaves little room over pieces standing on the board."""
    P = gp.planner
    high = np.array(q_top, float).copy()
    high[1:4] = np.radians([-60.0, 40.0, 60.0])
    turned = high.copy()
    turned[0] = q_hov[0]
    turned[4] = q_hov[4]
    out = []
    for qa, qb in ((q_top, high), (high, turned), (turned, q_hov)):
        seg = P.joint_path(chk, ARM[0], qa, qb, g)
        if seg is None:
            return None
        out += list(seg)
    return out


def lift_check(f, pk, down, log=print):
    """Lift the just-grasped piece ~1.5 cm, try to close 4 units more: if the jaws move freely the piece slipped
    out (edge grasp). Lower back to the grasp pose either way. Returns True when the piece is really held."""
    if not down:
        return False
    units = pk.grip_units
    g_hold = W.units_to_rad(arm=ARM[0], units=max(0.0, units))
    up = list(reversed(down))[:min(10, len(down))]
    f.run([Segment("test_lift", up, g_hold)])
    q = up[-1]
    for _ in range(15):
        f.command(q, gripper_units=max(0.0, units - 4.0))
        time.sleep(1 / 30)
    now = f.present()["gripper"]
    for _ in range(10):
        f.command(q, gripper_units=units)
        time.sleep(1 / 30)
    f.run([Segment("test_lower", list(reversed(up))[1:] + [down[-1]], g_hold)])
    ok = now > units - 2.5
    log(f"  lift check: jaws {units:.1f} -> {now:.1f} when squeezed: {'piece held' if ok else 'SLIPPED (edge grasp)'}")
    return ok


def safe_home_segments(q_from, q_cam, g):
    """Fixed way back (no planning): raise + straighten together, turn, then fold together."""
    from config import MAX_STEP_DEG

    def lin(a_, b_, label):
        n = max(1, int(np.ceil(np.degrees(np.abs(b_ - a_).max()) / MAX_STEP_DEG)))
        return Segment(label, [a_ + (b_ - a_) * i / n for i in range(1, n + 1)], g)

    # high pose for turning: TCP ~15 cm, lowest part of the arm ~14 cm above the desk over the whole turn
    high = np.array(q_from, float)
    high[1:4] = np.radians([-60.0, 40.0, 60.0])
    high[4] = q_cam[4]
    turned = high.copy()
    turned[0] = q_cam[0]
    unfolded = turned.copy()
    unfolded[1:4] = np.radians([-60.0, 76.0, 42.0])        # the usual unfold waypoint at the work pan
    return [lin(np.array(q_from, float), high, "raise_high"), lin(high, turned, "turn_high"),
            lin(turned, unfolded, "lower"), lin(unfolded, q_cam, "fold")]


def run_retry(f, q_start, segs, log=print):
    """Run segs; if the arm stalls (shoulder sometimes gets stuck around -85 deg when lifting out of the fold),
    back off to q_start, wait a moment and try once more at half speed."""
    if f.run(segs):
        return True
    log("arm stalled - backing off and retrying slowly")
    f.stop = False
    q = G.current_q(ARM[0])
    n = max(1, int(np.ceil(np.degrees(np.abs(q_start - q).max()) / 0.5)))
    if not f.run([Segment("back_off", [q + (q_start - q) * i / n for i in range(1, n + 1)], segs[0].gripper)]):
        return False
    time.sleep(1.0)
    # first try another way: unfold in place (raise the arm with the elbow out), then straight to the target
    target = segs[-1].qs[-1]
    try:
        useg, q_up = W.joint_seq(q_start, W.UNFOLD, segs[-1].gripper)
        n = max(1, int(np.ceil(np.degrees(np.abs(target - q_up).max()) / 1.0)))
        log("  retry via the unfold waypoint")
        if f.run(useg + [Segment("unfold_to_target", [q_up + (target - q_up) * i / n for i in range(1, n + 1)],
                                 segs[-1].gripper)]):
            return True
        f.stop = False
        q = G.current_q(ARM[0])
        n = max(1, int(np.ceil(np.degrees(np.abs(q_start - q).max()) / 0.5)))
        f.run([Segment("back_off2", [q + (q_start - q) * i / n for i in range(1, n + 1)], segs[0].gripper)])
        time.sleep(1.0)
    except Exception as e:
        log(f"  unfold retry failed: {e}")
        f.stop = False
    slow = []
    for sg in segs:
        qs = [sg.qs[0]]
        for a_, b_ in zip(sg.qs[:-1], sg.qs[1:]):
            qs += [(a_ + b_) / 2, b_]
        slow.append(Segment(sg.label + "_slow", qs, sg.gripper))
    return f.run(slow)


def available_arms():
    poses = json.loads((ROOT / "out" / "camera_pose.json").read_text())
    return [a for a in ("right", "left") if a in poses]


def choose_arm(src, dst, size, log=print):
    """Left arm for the left column (Y+), right arm for the right column; reserve pieces: the side that still has
    a piece of that size (top camera, both desks)."""
    arms = available_arms()
    if len(arms) == 1:
        return arms[0]
    col = lambda c: c % 3                      # col 0 = Y +70 (left arm side), col 2 = Y -70 (right arm side)
    key = src[1] if src[0] == "cell" and col(src[1]) != 1 else dst
    prefer = "left" if col(key) == 0 else "right"
    if src[0] == "reserve":
        try:
            frame = topcam.grab(warmup=30)
            cam = topcam.BoardCam.load()
            have = {a: [p for p in sam_pieces.detect(frame, cam, 0, region=DESK_REGION[a]) if p["size"] == size]
                    for a in arms}
            if not have[prefer] and have[OTHER[prefer]]:
                prefer = OTHER[prefer]
            log(f"arm choice: {prefer} (size {size} pieces: right {len(have.get('right', []))}, "
                f"left {len(have.get('left', []))})")
        except Exception as e:
            log(f"arm choice: camera check failed ({e}), using {prefer}")
    return prefer


def connect_follower(log=print, tries=3):
    """The bus sometimes drops one status packet while connecting (e.g. motor 5); just try again."""
    for i in range(tries):
        try:
            return Follower(ARM[0])
        except ConnectionError as e:
            log(f"arm connect failed ({e}); retry {i + 1}/{tries - 1}" if i < tries - 1 else f"arm connect failed: {e}")
            time.sleep(0.5)
    raise ConnectionError("right arm did not answer")


def go_to_camera_pose(f, q_cam):
    q = G.current_q(ARM[0])
    if np.degrees(np.abs(q - q_cam).max()) < 3:
        return
    if np.degrees(np.abs(q - q_cam).max()) > 30:      # far away (e.g. the arm sagged after a restart): go up and over
        f.run(safe_home_segments(q, q_cam, 0.0))
        return
    n = max(1, int(np.ceil(np.degrees(np.abs(q_cam - q).max()) / 0.5)))
    f.run([Segment("to_camera_pose", [q + (q_cam - q) * i / n for i in range(1, n + 1)], 0.0)])


def move(src, dst, size, log=print, arm="right"):
    GRIP_DZ[0] = 0.0
    ARM[0] = arm
    G.PIECE_ARM = arm
    """src: ("reserve", size) or ("cell", i); dst: cell index. Returns True when placed."""
    gp = G.GobbletPlanner()
    fix_left_z(gp.planner)
    gp.k = gp.planner.kin[arm]
    P = gp.planner
    ql = G.current_q(OTHER[ARM[0]])
    q_cam = np.array(json.loads((ROOT / "out" / "camera_pose.json").read_text())[ARM[0]])
    d, h = G.PIECE[size]
    a_dn = np.array([0.0, 0.0, -1.0])
    f = connect_follower(log)
    try:
        go_to_camera_pose(f, q_cam)
        time.sleep(0.5)
        frame = topcam.grab(warmup=40)
        # photo from the wide camera pose; WORK from the same fold at pan 72 (raising the arm at ~100+ deg pan
        # stalls the shoulder on the real rig). Return to the photo pose at the end.
        photo_pan = float(np.degrees(q_cam[0]))
        if abs(photo_pan) > 80:
            segs, q_work = W.joint_seq(q_cam, [(0, WORK_PAN * SIDE[ARM[0]])], 0.0)
            if not f.run(segs):
                log("stopped while turning to the work pose")
                return False
            q_cam = q_work
        cam = topcam.BoardCam.load()
        desk = sam_pieces.detect(frame, cam, 0, region=DESK_REGION[arm])
        board = topcam.read_cells(frame, cam)
        occupied = [c for c, r in enumerate(board) if r.color is not None]
        log(f"desk orange: {[('SML'[p['size']], p['x'], p['y']) for p in desk]}  board occupied: {occupied}")
        # obstacles in arm coordinates: desk pieces + board pieces (top piece taken as large: safe side)
        board_obs = [(np.array(to_arm(*G.cell_xy(c))), BOARD_FLOOR, 2) for c in occupied
                     if c != dst and not (src[0] == "cell" and c == src[1])]

        # ---- source
        if src[0] == "reserve":
            cand = sorted([p for p in desk if p["size"] == size], key=lambda p: np.hypot(p["x"] - 185, p["y"] + 226))
            pick_opts = [(np.array(desk_to_arm(p["x"], p["y"])), 0.0, p) for p in cand]
        else:
            pick_opts = [(np.array(to_arm(*G.cell_xy(src[1]))), BOARD_FLOOR, None)]
        choice = None
        for txy, floor, p in pick_opts:
            others = [(np.array(desk_to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p] + board_obs
            chk = G.add_obstacles(P.checker_for(ARM[0], ql), others, margin_mm=2.0)
            # jaws along Y (the direction the camera->arm correction was taught with) first, then close to it
            # no 120/150 deg: those need a big wrist roll that stalls on the real arm (2026-11-01)
            angles = [(90, 24.0), (60, 24.0), (90, 16.0), (60, 16.0)]
            if src[0] == "cell":     # board pieces: any jaw direction (cell 2 needs jaws along X, like placing there)
                angles = [(a_, e_) for e_ in (24.0, 16.0) for a_ in (90, 0, 180, 60, 30, -90, 120, 150)]
            for ang, extra in angles:
                c = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
                dzs = (0.0,)
                if src[0] == "cell":
                    dzs = (0.0, 8.0, 15.0, 22.0)
                    if size == 2:   # open jaws are wider than a cell: keep the fingertips above the bars
                        bar_gap = 8.0 if ARM[0] == "right" else 3.0   # right: 5 mm higher (user, 2026-11-02)
                        dz0 = max(0.0, BAR_TOP_Z + bar_gap + TIP_BELOW_TCP - (floor + h * G.GRASP_FRAC[size]))
                        dzs = (dz0, dz0 + 5.0, dz0 + 10.0)
                for extra, dz in [(extra, z_) for z_ in dzs]:
                    xy = txy - c[:2] * (d + extra) / 2
                    hov = np.array([xy[0], xy[1], W.HOVER_Z])
                    tcp = np.array([xy[0], xy[1], floor + dz + h * G.GRASP_FRAC[size]])
                    q_h = gp.k.ik_pose(hov, a_dn, c, q_cam, n_random=4)
                    g_try = G.gap_to_rad(d + extra)
                    if q_h is None or P.line_path(chk, ARM[0], q_h, hov, tcp, a_dn, c, g_try) is None:
                        continue
                    way = route(P, chk, q_cam, q_h, g_try)
                    if way is None:
                        continue                      # can see a way down, but not a way there
                    choice = (txy, floor + dz, others, c, xy, extra, ang, way, q_h)
                    GRIP_DZ[0] = dz             # gripped this much higher on the body: place it this much higher
                    break
                if choice:
                    break
            if choice:
                break
        if choice is None and src[0] == "reserve" and FINE_ALIGN:
            # the strict straight-descent check failed; the wrist-camera path below finds its own hover
            txy0, _, p0 = pick_opts[0]
            others0 = [(np.array(desk_to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p0] + board_obs
            c0 = np.array([0.0, 1.0, 0.0])
            choice = (txy0, 0.0, others0, c0, txy0 - c0[:2] * (d + 24.0) / 2, 24.0, 90, None, q_cam)
        if choice is None:
            log("no clear way to grab the piece")
            return False
        txy, floor, others, c, xy, extra, ang, way, q_h = choice
        log(f"pick at arm ({txy[0]:.1f}, {txy[1]:.1f}) floor {floor:.0f}, jaws {ang} deg, opening +{extra:.0f}")
        g0 = G.gap_to_rad(d + extra)
        f.run([Segment("open", [q_cam] * G.HOLD, g0)])
        pk = W.Picker(gp, f, ql, others)
        if src[0] == "reserve" and FINE_ALIGN:
            # coarse: overhead estimate; fine: wrist camera puts the piece on the learned image spot.
            # Pick the first candidate whose hover (jaws along +Y, the orientation REF was learned in) is reachable.
            chosen = None
            # jaws along Y first (how the correction was measured); if the neighbours are in the way along Y,
            # turn the jaws (pieces in the reserve usually stand in rows along Y)
            for ang_c in (90.0, 0.0, 60.0, 30.0):   # no 120+ (big wrist roll stalls)
                if chosen is not None:
                    break
                c = np.array([np.cos(np.radians(ang_c)), np.sin(np.radians(ang_c)), 0.0])
                for txy_c, _, p_c in pick_opts:
                    xy_c = txy_c - c[:2] * (d + 24.0) / 2
                    hov_c = np.array([xy_c[0], xy_c[1], W.HOVER_Z])
                    q_hc = gp.k.ik_pose(hov_c, a_dn, c, q_cam, n_random=4)
                    if q_hc is None:
                        continue
                    chk_c = G.add_obstacles(P.checker_for(ARM[0], ql),
                                            [(np.array(desk_to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p_c] + board_obs,
                                            margin_mm=2.0)
                    # the jaws must be able to go straight down around this piece without touching its neighbours
                    clear = False
                    for extra_c in (24.0, 16.0, 10.0):
                        xy_e = txy_c - c[:2] * (d + extra_c) / 2
                        hov_e = np.array([xy_e[0], xy_e[1], W.HOVER_Z])
                        tcp_e = np.array([xy_e[0], xy_e[1], h * G.GRASP_FRAC[size]])
                        q_he = gp.k.ik_pose(hov_e, a_dn, c, q_hc, n_random=0)
                        if q_he is not None and P.line_path(chk_c, ARM[0], q_he, hov_e, tcp_e, a_dn, c,
                                                            G.gap_to_rad(d + extra_c)) is not None:
                            clear = True
                            break
                    if not clear:
                        # the wrist-camera mount vs upper-arm model blocks some descents the real arm can do
                        # (left M at the far reserve, 2026-11-02); try_grasp has the same fallback
                        tcp_e = np.array([hov_c[0], hov_c[1], h * G.GRASP_FRAC[size]])
                        qq, ok_s = q_hc, True
                        n_s = max(1, int(np.ceil(np.linalg.norm(tcp_e - hov_c) / 4.0)))
                        for i_s in range(1, n_s + 1):
                            q_s = gp.k.ik_pose(hov_c + (tcp_e - hov_c) * i_s / n_s, a_dn, c, qq, n_random=0)
                            if q_s is None or not P.free(chk_c, ARM[0], q_s, g0, skip_self=True) \
                                    or np.degrees(np.abs(q_s - qq).max()) > 8.0:
                                ok_s = False
                                break
                            qq = q_s
                        if ok_s:
                            clear = True
                            log(f"  jaws {ang_c:.0f} deg: clear of neighbours (ignoring the wrist-camera self-contact model)")
                    if not clear:
                        log(f"  skip piece at camera ({p_c['x']}, {p_c['y']}): jaws would hit a neighbour")
                        continue
                    way_c = route(P, chk_c, q_cam, q_hc, g0)
                    if way_c is not None:
                        chosen = txy_c
                        chosen_ang = ang_c
                        # obstacles for the grasp itself: everything EXCEPT the piece we are about to take
                        others = [(np.array(desk_to_arm(o["x"], o["y"])), 0.0, o["size"]) for o in desk if o is not p_c] + board_obs
                        pk = W.Picker(gp, f, ql, others)
                        log(f"fine-align target: camera ({p_c['x']}, {p_c['y']}) -> arm {np.round(txy_c, 1)}, jaws {ang_c:.0f} deg")
                        break
            if chosen is None:
                log("no reachable hover above any candidate")
                return False
            if not run_retry(f, q_cam, [Segment("open", [q_cam] * G.HOLD, g0), Segment("to_hover", way_c, g0)], log):
                log("stopped on the way to the hover")
                return False
            if USE_WRIST_CAMERA:
                held, qh, down, xy = pk.pick(chosen, size, q_hc)
            else:   # grasp straight down at the corrected overhead estimate (what worked on 2026-11-01 08:08)
                qh, xy = q_hc, chosen - c[:2] * (d + 24.0) / 2
                if DAGGER["hover_noise"] > 0:      # data collection: start from a deliberately wrong hover
                    nz = np.random.uniform(-1.0, 1.0, 2) * DAGGER["hover_noise"]
                    q2 = pk.move_hover(xy + nz, a_dn, c, qh, g0)
                    if q2 is not None:
                        qh, xy = q2, xy + nz
                        log(f"  dagger: start hover moved by ({nz[0]:+.1f}, {nz[1]:+.1f}) mm")
                xy_first = xy.copy()
                wr = WR.WristRefine(ARM[0], size, chosen_ang, log)
                ep = DG.Episode(ARM[0], size, chosen_ang, log) if DAGGER["record"] else None
                pol = DG.policy() if DAGGER["use_policy"] else None
                tries, xy_expert = 0, None

                def correct(first):
                    """Step (mm) from the learned policy, else the image-Jacobian correction. Records the state."""
                    if pol is not None:
                        img = wr.last if first else wr.snap("hover1")
                        if img is None:
                            return None
                        step, spread = pol.predict(img, ARM[0], size, chosen_ang)
                        log(f"  dagger policy (round {pol.round}): move ({step[0]:+.1f}, {step[1]:+.1f}) mm, "
                            f"ensemble spread {spread:.1f} mm")
                        src = "policy"
                    else:
                        step = wr.correction_from_b0() if first else wr.correction()
                        img, spread, src = wr.last, None, "jacobian"
                    if ep is not None:
                        ep.add(img, xy, step, src, spread)
                    return step

                wr.before_grasp()
                # align with the wrist camera before the first try (a missed first try pushes the piece away)
                step = correct(True)
                if pol is None:      # the Jacobian step is only trusted when small
                    if step is not None and np.linalg.norm(step) > 8.0:
                        log(f"  wrist: {np.linalg.norm(step):.0f} mm is too far to trust before the first try - not moving")
                        step = None
                    elif step is not None and np.linalg.norm(step) > 6.0:
                        step = step * 6.0 / np.linalg.norm(step)
                if step is not None and np.linalg.norm(step) > 1.5:
                    q2 = pk.move_hover(xy + step, a_dn, c, qh, g0)
                    if q2 is not None:
                        qh, xy = q2, xy + step
                        log(f"  wrist: aligned before grasping ({step[0]:+.1f}, {step[1]:+.1f}) mm")
                        learn_desk_offset(*step, log, gain=0.15)
                        wr.b0 = wr.find(wr.ref(), "hover0b") if wr.ref() is not None else wr.b0
                        if ep is not None and wr.ref() is not None:
                            ep.add(wr.last, xy, None, "after_step")
                        xy_first = xy.copy()
                held, down = pk.try_grasp(qh, xy, a_dn, c, size, g0)
                held = held and lift_check(f, pk, down, log)
                tries += 1
                if held:
                    wr.learn_ref(w=0.2)          # held at once: only a small pull (the user-taught REF is exact)
                elif wr.b0 is not None or pol is not None:
                    # missed: back up to the hover and let the wrist camera say where the piece really is
                    if down is not None:
                        f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
                        down = None
                    step = correct(False)
                    if step is not None and np.linalg.norm(step) > 1.5:
                        xy2 = xy + step
                        q2 = pk.move_hover(xy2, a_dn, c, qh, g0)
                        if q2 is not None:
                            qh = q2
                            held, down = pk.try_grasp(qh, xy2, a_dn, c, size, g0)
                            held = held and lift_check(f, pk, down, log)
                            tries += 1
                            log(f"  retry at the wrist-camera spot ({step[0]:+.1f}, {step[1]:+.1f}) mm -> "
                                f"{'HELD' if held else 'missed'}")
                            if held:
                                xy = xy2
                                learn_desk_offset(*step, log)
                # missed: the estimate can be ~1 cm off away from the measured area; nudge and retry
                for dx, dy in nudge_order():
                    if held:
                        break
                    if down is not None:
                        f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
                        down = None
                    xy2 = xy + np.array([dx, dy])
                    q2 = pk.move_hover(xy2, a_dn, c, qh, g0)
                    if q2 is None:
                        continue
                    qh = q2
                    if ep is not None:         # every hover the robot visits is a state for DAgger
                        ep.add(wr.snap("nudge"), xy2, None, "nudge")
                    held, down = pk.try_grasp(qh, xy2, a_dn, c, size, g0)
                    held = held and lift_check(f, pk, down, log)
                    tries += 1
                    log(f"  retry with nudge ({dx:+.0f}, {dy:+.0f}) mm -> {'HELD' if held else 'missed'}")
                    if held:
                        xy = xy2
                        learn_desk_offset(dx, dy, log)
                        wr.learn_ref(xy2 - xy_first)
                # gave up: the person shows the right spot with the leader arm (DAgger expert)
                if not held and DAGGER["expert"] is not None:
                    import dagger_expert as DX
                    if down is not None:
                        f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
                        down = None
                    res = DAGGER["expert"](DX.Ctx(f, ARM[0], pk, xy, qh, a_dn, c, g0, gp.k, log))
                    if res is not None:
                        xy_expert, q_at = res
                        q2 = pk.move_hover(xy_expert, a_dn, c, q_at, g0)
                        if q2 is not None:
                            qh = q2
                            held, down = pk.try_grasp(qh, xy_expert, a_dn, c, size, g0)
                            held = held and lift_check(f, pk, down, log)
                            tries += 1
                            log(f"  grasp at the expert's spot -> {'HELD' if held else 'missed'}")
                            if held:
                                xy = xy_expert
                if ep is not None:
                    ep.finish(held, xy if held else None, xy_expert, tries)
                wr.close()
        else:
            if not run_retry(f, q_cam, [Segment("to_above_piece", way, g0)], log):
                log("stopped on the way to the piece")
                return False
            qh = q_h
            time.sleep(0.3)
            held, down = pk.try_grasp(qh, xy, a_dn, c, size, g0, floor=floor)
            held = held and lift_check(f, pk, down, log)
        log("GRASP: " + ("HELD" if held else "missed"))
        if not held or DAGGER["pick_only"]:
            # missed, or collecting DAgger data: let go where it was gripped and go back to the photo pose
            if down is not None:
                f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
            back = P.staged_path(pk.chk, ARM[0], qh, q_cam, g0)
            f.run([Segment("back", back, g0)] if back else safe_home_segments(qh, q_cam, g0))
            return held

        # ---- destination (cell, corrected); pieces already on it end up inside the cup
        g_hold = W.units_to_rad(arm=ARM[0], units=max(0.0, pk.grip_units))       # keep the size-dependent squeeze while carrying
        g_rel = G.gap_to_rad(d + 24.0)            # open WIDE to let go (d+6 left the piece stuck between the jaws)
        lift = list(reversed(down))
        lift += high_lift(gp, pk.chk, lift[-1], a_dn, c, g_hold)
        # plan the whole place: every holding orientation x hover heights; release must clear the neighbours too
        pl, carry, lower, up = None, None, None, None
        for opt in place_options(gp, pk.chk, size, dst, lift[-1], g_hold):
            q_pl, a2, c2, tcp = opt["q_pl"], opt["a"], opt["c"], opt["tcp"]
            for z_hov in (150.0, 130.0, 110.0, 95.0, 80.0, 65.0, 50.0):
                if z_hov <= tcp[2] + 12:
                    break
                hov = tcp - a2 * (z_hov - tcp[2]) / max(1e-3, -a2[2])
                q_hov = gp.k.ik_pose(hov, a2, c2, q_pl, n_random=1)
                if q_hov is None:
                    continue
                lower = P.line_path(pk.chk, ARM[0], q_hov, hov, tcp, a2, c2, g_hold)
                if lower is None:
                    continue
                up = None
                for rel_extra in (24.0, 16.0, 10.0, 6.0):
                    g_rel = G.gap_to_rad(d + rel_extra)
                    up = P.line_path(pk.chk, ARM[0], lower[-1], tcp, hov, a2, c2, g_rel)
                    if up is not None:
                        break
                if up is None:
                    continue
                carry = carry_high(gp, pk.chk, lift[-1], hov, q_hov, a_dn, c, g_hold) or route(P, pk.chk, lift[-1], q_hov, g_hold)
                if carry is None:
                    up = None
                    continue
                pl = opt
                break
            if pl is not None:
                log(f"place plan: {opt['label']}, jaws {np.round(c2, 2)}, hover z {hov[2]:.0f}, release +{rel_extra:.0f} mm")
                break
        if up is None:
            # never let go of a piece in mid-air: put it back where it was, then go home
            log("no path to the destination cell - putting the piece back")
            f.run([Segment("put_back", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
            f.run(safe_home_segments(down[0], q_cam, g0))
            return False
        cx, cy = G.cell_xy(dst)
        ax_, ay_ = pl["arm_xy"]
        log(f"place on cell {dst}: centre ({cx:.1f}, {cy:.1f}) -> arm ({ax_:.1f}, {ay_:.1f})")
        f.run([Segment("lift", lift, g_hold), Segment("carry", carry, g_hold), Segment("lower", lower, g_hold),
               Segment("release", [lower[-1]] * G.HOLD, g_rel), Segment("retreat", up, g_rel)])
        f.run(safe_home_segments(up[-1], q_cam, g_rel))     # fixed sequence: raise, fold, turn (no planning)
        return True
    finally:
        if not f.stop:
            try:
                q_end = G.current_q(ARM[0])
                if abs(np.degrees(q_end[0]) - WORK_PAN * SIDE[ARM[0]]) < 8 and np.degrees(q_end[1]) < -95:   # folded at the work pan
                    q_photo = np.array(json.loads((ROOT / "out" / "camera_pose.json").read_text())[ARM[0]])
                    segs, _ = W.joint_seq(q_end, [(0, float(np.degrees(q_photo[0])))], 0.0)
                    f.run(segs)
            except Exception as e:  # never leave the arm hanging because of the cosmetic return
                log(f"return to photo pose skipped: {e}")
            f.disconnect()
        else:
            log("stopped: torque held")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0].upper().startswith("R"):
        sz = int(a[0][1:])
        ok = move(("reserve", sz), int(a[1]), sz)
    else:
        ok = move(("cell", int(a[0])), int(a[1]), int(a[2]))
    print("RESULT:", "placed" if ok else "failed")
