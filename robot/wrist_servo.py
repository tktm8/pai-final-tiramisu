"""Coarse-to-fine pick: top camera (SAM) -> hover above the piece -> wrist camera fine alignment -> grasp.

The wrist camera (right gripper, Innomaker) looks down/forward past the jaws. At a fixed hover height and gripper
orientation, a piece that is correctly under the jaws always appears at the same image point REF. We servo the
gripper sideways until the piece's blob sits on REF, using an image Jacobian J (pixels per mm of gripper motion).

Both J and REF are learned automatically and stored in out/wrist_servo.json:
  J   : two small test moves (+X, +Y) while watching the blob.
  REF : the first time, grasps are tried in a spiral around the estimate; when one holds, the arm goes back up to
        the hover height and records where the piece appears.

  python robot/wrist_servo.py            # pick the medium piece nearest the reserve centre and put it back
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot"), str(ROOT / "perception")]

import gobblet_arm as G  # noqa: E402
import topcam  # noqa: E402
from planner import Segment  # noqa: E402

WRIST_CAM = "0x21130000c456366"      # right gripper camera
STATE = ROOT / "out" / "wrist_servo.json"
HOVER_Z = 85.0                       # TCP height while servoing
TEST_MOVE = 12.0
HELD_UNITS = 4.0


class WristCam:
    def __init__(self):
        self.cap = cv2.VideoCapture(topcam.camera_index(WRIST_CAM), cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def grab(self):
        f = None
        for _ in range(6):          # flush stale frames
            ok, fr = self.cap.read()
            if ok:
                f = fr
        return f

    def close(self):
        self.cap.release()


def blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 100, 80), (22, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    return [(float(cen[i][0]), float(cen[i][1]), int(st[i, 4])) for i in range(1, n) if st[i, 4] > 1500]


def nearest(bl, p):
    return min(bl, key=lambda b: np.hypot(b[0] - p[0], b[1] - p[1])) if bl else None


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


LEARN_GRIP = False     # keep the saved contact points (user, 2026-11-01 19:10)
SQUEEZE = {0: 2.0, 1: 3.0, 2: 6.0}     # units past the contact point (large slipped at 3 on 2026-11-01)


def gentle_close(f, q, size, st):
    """Size-aware, slow close. Fast down to just wider than the piece, then creep until the jaws stop on it and
    hold one unit past contact (light grip). The contact reading is remembered per size for next time.
    Returns True if the piece is between the jaws."""
    from kinematics import gripper_urdf_to_lerobot
    from kinematics import calibration
    d, _ = G.PIECE[size]
    arm = getattr(f, "arm", "right")
    gkey = "grip_units" if arm == "right" else f"grip_units_{arm}"     # contact point differs per gripper
    # same physical squeeze as the right gripper (units are % of each gripper's own raw range)
    cr, ca = calibration("right")["gripper"], calibration(arm)["gripper"]
    squeeze = SQUEEZE[size] * (cr["range_max"] - cr["range_min"]) / (ca["range_max"] - ca["range_min"])
    learned = st.get(gkey, {}).get(str(size))
    pre_geom = gripper_urdf_to_lerobot(arm, G.gap_to_rad(d + 8.0))   # just wider than the piece
    pre = max(learned + 4.0, pre_geom) if learned is not None else pre_geom   # a bad learned value must not slam
    floor = 0.0          # always creep to fully closed: a learned floor made a real grasp look like a miss
    target = f.present()["gripper"]
    while target > pre:                                   # quick part, nowhere near the piece yet
        target = max(pre, target - 0.6)
        f.command(q, gripper_units=target)
        time.sleep(1 / 30)
    still, last = 0, f.present()["gripper"]
    while target > floor:                                 # creep
        target = max(floor, target - 0.15)
        f.command(q, gripper_units=target)
        time.sleep(1 / 30)
        now = f.present()["gripper"]
        still = still + 1 if abs(now - last) < 0.15 else 0
        last = now
        if now - target > 1.5 and still >= 8:             # jaws stopped although we ask for more: touching
            hold = now - squeeze
            for _ in range(15):
                f.command(q, gripper_units=hold)
                time.sleep(1 / 30)
            if LEARN_GRIP and now > 5.0:        # a contact near "closed" is not the piece: do not learn it
                st.setdefault(gkey, {})[str(size)] = round(now, 1)
            save_state(st)
            print(f"  gentle grip: contact at {now:.1f} units, holding {hold:.1f}")
            return True
    time.sleep(0.2)
    final = f.present()["gripper"]
    if final > HELD_UNITS:       # asked for fully closed but the jaws stayed open: something is between them
        print(f"  gentle grip: jaws stopped at {final:.1f} (no clear stall) -> holding")
        if LEARN_GRIP and final > 5.0:
            st.setdefault(gkey, {})[str(size)] = round(final, 1)
        return True
    print(f"  gentle grip: no contact (closed to {final:.1f})")
    return False


class Picker:
    def __init__(self, gp, f, ql, others):
        self.gp, self.f, self.ql = gp, f, ql
        self.P = gp.planner
        self.arm = getattr(f, "arm", "right")
        self.chk = G.add_obstacles(self.P.checker_for(self.arm, ql), others)
        self._cam = None
        self.st = load_state()

    @property
    def cam(self):
        if self._cam is None:
            self._cam = WristCam()
        return self._cam

    def move_hover(self, xy, a, c, q_seed, g):
        tcp = np.array([xy[0], xy[1], HOVER_Z])
        q = self.gp.k.ik_pose(tcp, a, c, q_seed, n_random=0)
        if q is None:
            q = self.gp.k.ik_pose(tcp, a, c, q_seed, n_random=6)
        if q is None:
            print(f"  no IK for hover at {np.round(tcp, 1)}")
            return None
        path = self.P.joint_path(self.chk, self.arm, q_seed, q, g) or self.P.staged_path(self.chk, self.arm, q_seed, q, g)
        if path is None:
            print(f"  no path to hover at {np.round(tcp, 1)}")
            return None
        if not self.f.run([Segment("hover", path, g)]):
            raise RuntimeError("arm stopped while moving to the hover pose")
        time.sleep(0.4)
        return q

    def try_grasp(self, q_hover, xy, a, c, size, g_open, floor=0.0):
        """Straight down from the hover, close, report held, open, back up."""
        d, h = G.PIECE[size]
        tcp = np.array([xy[0], xy[1], floor + h * G.GRASP_FRAC[size]])
        hov = np.array([xy[0], xy[1], HOVER_Z])
        down = None
        for gap_extra in (24.0, 16.0, 10.0):             # narrow the opening if a neighbour is in the way
            g_try = G.gap_to_rad(d + gap_extra)
            down = self.P.line_path(self.chk, self.arm, q_hover, hov, tcp, a, c, g_try)
            if down is not None:
                if gap_extra != 24.0:
                    print(f"  descent with a narrower opening (+{gap_extra:.0f} mm)")
                    for _ in range(20):
                        self.f.command(q_hover, gripper_rad=g_try)
                        time.sleep(1 / 30)
                g_open = g_try
                break
        if down is None:
            # the wrist-camera mount vs upper-arm model blocks some near-base descents the real arm can do
            g_try = G.gap_to_rad(d + 24.0)
            path, q = [], q_hover
            n = max(1, int(np.ceil(np.linalg.norm(tcp - hov) / 4.0)))
            for i in range(1, n + 1):
                q = self.gp.k.ik_pose(hov + (tcp - hov) * i / n, a, c, q, n_random=0)
                if q is None or not self.P.free(self.chk, self.arm, q, g_try, skip_self=True) \
                        or (path and np.degrees(np.abs(q - path[-1]).max()) > 8.0):
                    path = None
                    break
                path.append(q)
            if path:
                print("  descent: ignoring the wrist-camera/upper-arm self-contact model")
                down, g_open = path, g_try
        if down is None:
            qe = self.gp.k.ik_pose(tcp, a, c, q_hover, n_random=0)
            why = "no IK" if qe is None else self.chk.collisions(self.P.pts[self.arm], qe, G.gap_to_rad(d + 10.0))
            from kinematics import raw_in_range
            print(f"  descent blocked at tcp {np.round(tcp, 1)}: {why}  in_range={qe is not None and raw_in_range('right', qe)}")
            return None, None
        self.f.run([Segment("descend", down, g_open)])
        held = gentle_close(self.f, down[-1], size, self.st)
        self.grip_units = self.f.present()["gripper"]
        return held, down

    def pick(self, est_xy, size, q_start):
        """Returns (held, q_now, down_path, xy_used). Arm ends low holding the piece if held."""
        d, h = G.PIECE[size]
        gap = d + 24.0
        g_open = G.gap_to_rad(gap)
        a = np.array([0.0, 0.0, -1.0])
        c = np.array([0.0, 1.0, 0.0])        # jaws close along world Y (same orientation every time: REF stays valid)
        off = c * gap / 2                    # piece centre = TCP + off  (middle of the open jaws)
        xy = np.asarray(est_xy, float) - off[:2]
        q = self.move_hover(xy, a, c, q_start, g_open)
        if q is None:
            return False, q_start, None, xy
        ref = self.st.get(f"ref_{size}") or self.st.get("ref_1") or self.st.get("ref")
        J = np.array(self.st["J"]) if "J" in self.st else None
        img = self.cam.grab()
        cv2.imwrite(str(ROOT / "out" / "wrist_hover0.jpg"), img)
        bl = blobs(img)
        H, W = img.shape[:2]
        guess = ref or [W * 0.5, H * 0.8]
        tgt = nearest(bl, guess)
        if tgt is None:
            print("  wrist: no orange blob")
            return False, q, None, xy
        if J is None:                         # learn the image Jacobian with two test moves
            cols = []
            for k in (0, 1):
                dxy = np.zeros(2)
                dxy[k] = TEST_MOVE
                q2 = self.move_hover(xy + dxy, a, c, q, g_open)
                b2 = nearest(blobs(self.cam.grab()), tgt[:2])
                cols.append([(b2[0] - tgt[0]) / TEST_MOVE, (b2[1] - tgt[1]) / TEST_MOVE])
                q = self.move_hover(xy, a, c, q2, g_open)
            J = np.array(cols).T
            self.st["J"] = J.tolist()
            save_state(self.st)
            print("  learned J (px/mm):", J.round(2).tolist())
            tgt = nearest(blobs(self.cam.grab()), tgt[:2])
        if ref is not None:                   # fine alignment
            for it in range(5):
                e = np.array(tgt[:2]) - np.array(ref)
                print(f"  servo {it}: blob {np.round(tgt[:2])} ref {np.round(ref)} err {np.linalg.norm(e):.0f}px")
                if np.linalg.norm(e) < 8:
                    break
                step = -np.linalg.solve(J, e)
                n = np.linalg.norm(step)
                if n > 15:
                    step *= 15 / n
                q2 = self.move_hover(xy + step, a, c, q, g_open)
                if q2 is None:
                    break
                xy, q = xy + step, q2
                pred = np.array(tgt[:2]) + J @ step          # where THIS piece should now appear
                nb = nearest(blobs(self.cam.grab()), pred)
                if nb is None or np.hypot(nb[0] - pred[0], nb[1] - pred[1]) > 45:
                    print("  servo: lost the target piece")
                    break
                tgt = nb
            held, down = self.try_grasp(q, xy, a, c, size, g_open)
            if held:
                if tgt is not None and f"ref_{size}" not in self.st:
                    self.st[f"ref_{size}"] = [float(tgt[0]), float(tgt[1])]
                    save_state(self.st)
                return True, q, down, xy
            # missed: small search around the aligned spot, then correct REF by the offset that worked
            blob_final = np.array(tgt[:2]) if tgt is not None else np.array(ref)
            if down is not None:
                self.f.run([Segment("open", [down[-1]] * G.HOLD, g_open), Segment("up", list(reversed(down)), g_open)])
            for r in (6.0, 12.0):
                for t in np.linspace(0, 2 * np.pi, 6 if r == 6.0 else 10, endpoint=False):
                    o = np.array([r * np.cos(t), r * np.sin(t)])
                    q2 = self.move_hover(xy + o, a, c, q, g_open)
                    if q2 is None:
                        continue
                    held, down = self.try_grasp(q2, xy + o, a, c, size, g_open)
                    print(f"  local search ({o[0]:+.0f},{o[1]:+.0f}) -> {'HELD' if held else 'miss'}")
                    if held:
                        new_ref = blob_final + J @ o
                        self.st[f"ref_{size}"] = [float(v) for v in (0.5 * np.array(ref) + 0.5 * new_ref)]
                        save_state(self.st)
                        print(f"  REF for size {size} corrected to", np.round(self.st[f"ref_{size}"]).tolist())
                        return True, q2, down, xy + o
                    if down is not None:
                        self.f.run([Segment("open", [down[-1]] * G.HOLD, g_open),
                                    Segment("up", list(reversed(down)), g_open)])
                    q = q2
            return False, q, None, xy
        # no REF yet: spiral search for a grasp that holds, then record REF
        tgt_px = tgt[:2]
        for dx, dy in [(0, 0)] + [(r * np.cos(t), r * np.sin(t)) for r in (8, 16) for t in np.linspace(0, 2 * np.pi, int(r * 0.8), endpoint=False)]:
            xy_try = xy + np.array([dx, dy])
            q2 = self.move_hover(xy_try, a, c, q, g_open)
            if q2 is None:
                continue
            q = q2
            held, down = self.try_grasp(q, xy_try, a, c, size, g_open)
            print(f"  search ({dx:+.0f},{dy:+.0f}) -> {'HELD' if held else 'miss'}")
            if held:
                # release back down, go up to the hover with open jaws and record where the piece appears
                self.f.run([Segment("open", [down[-1]] * G.HOLD, g_open), Segment("up", list(reversed(down)), g_open)])
                time.sleep(0.4)
                b = nearest(blobs(self.cam.grab()), np.array(tgt_px) + J @ np.array([dx, dy]))
                if b is not None:
                    self.st[f"ref_{size}"] = [b[0], b[1]]
                    save_state(self.st)
                    print(f"  learned REF for size {size}:", np.round(b[:2]).tolist())
                held, down = self.try_grasp(q, xy_try, a, c, size, g_open)
                return bool(held), q, down, xy_try
            if down is not None:
                self.f.run([Segment("open", [down[-1]] * G.HOLD, g_open), Segment("up", list(reversed(down)), g_open)])
        return False, q, None, xy


def joint_seq(q_from, moves, g):
    """One joint at a time (moves: [(joint, deg)]): the order proven on the real rig for (un)folding."""
    from config import MAX_STEP_DEG
    segs, q = [], np.array(q_from, float)
    for j, deg in moves:
        t = q.copy()
        t[j] = np.radians(deg)
        n = max(1, int(np.ceil(np.degrees(abs(t[j] - q[j])) / MAX_STEP_DEG)))
        segs.append(Segment(f"j{j}", [q + (t - q) * i / n for i in range(1, n + 1)], g))
        q = t
    return segs, q


UNFOLD = [(1, -60.0), (2, 76.0), (3, 42.0)]      # elbow cannot reach 60 in this pose on the real arm
FOLD = [(3, 66.0), (2, 88.0), (1, -99.0)]


def units_to_rad(units, arm="right"):
    """LeRobot gripper units (0..100) -> URDF gripper angle, inverse of kinematics.gripper_urdf_to_lerobot."""
    from kinematics import calibration
    cal = calibration(arm)["gripper"]
    raw = cal["range_min"] + units / 100.0 * (cal["range_max"] - cal["range_min"])
    return float(np.radians((raw - 2047) * 360.0 / 4095))


def place_on_cell(pk, down, size, cell, q_up, q_look):
    """Holding the piece at the bottom of `down`: lift, carry above `cell`, lower, release, fold away."""
    gp, P, f = pk.gp, pk.P, pk.f
    g_hold = units_to_rad(max(0.0, pk.grip_units - 1.0))
    d, h = G.PIECE[size]
    g_rel = G.gap_to_rad(d + 6.0)
    lift = list(reversed(down))
    dst = G.Spot("cell", cell)
    place = gp.grasp_pose(pk.chk, dst, size, 8.0, lift[-1], g_hold)
    if place is None:
        return False, "no place pose"
    q_pl, a, c, tcp, _ = place
    hov, q_hov = gp.hover(tcp, a, c, q_pl)
    carry = P.staged_path(pk.chk, "right", lift[-1], q_hov, g_hold)
    lower = P.line_path(pk.chk, "right", q_hov, hov, tcp, a, c, g_hold) if carry else None
    up = P.line_path(pk.chk, "right", lower[-1], tcp, hov, a, c, g_rel) if lower else None
    back = P.staged_path(pk.chk, "right", up[-1], q_up, g_rel) if up else None
    if back is None:
        return False, "no path to the cell"
    fold, _ = joint_seq(q_up, FOLD, g_rel)
    ok = f.run([Segment("lift", lift, g_hold), Segment("carry", carry, g_hold), Segment("lower", lower, g_hold),
                Segment("release", [lower[-1]] * G.HOLD, g_rel), Segment("retreat", up, g_rel),
                Segment("back", back, g_rel)] + fold)
    return ok, "placed"


def demo():
    from robot_io import Follower
    import sam_pieces
    gp = G.GobbletPlanner()
    P = gp.planner
    ql = G.current_q("left")
    if not G.return_to_look(gp):
        raise SystemExit("could not bring the right arm to the camera-clear pose")
    q_look = G.current_q("right")
    frame = topcam.grab()
    cam = topcam.BoardCam.load()
    pieces = sam_pieces.detect(frame, cam, 0)
    print("SAM:", [("SML"[p["size"]], p["x"], p["y"]) for p in pieces])
    def clearance(p):
        return min([np.hypot(p["x"] - o["x"], p["y"] - o["y"]) for o in pieces if o is not p] or [999])
    # a medium piece with nothing close by (neighbours confuse the wrist camera and get bumped)
    cand = sorted([p for p in pieces if p["size"] == 1 and 120 <= p["x"] <= 290], key=lambda p: -clearance(p))
    if cand:
        print(f"  target {cand[0]['x']}, {cand[0]['y']} (nearest other piece {clearance(cand[0]):.0f} mm)")
    if not cand:
        raise SystemExit("no medium piece seen")
    piece = cand[0]
    others = [(np.array([p["x"], p["y"]]), 0.0, p["size"]) for p in pieces if p is not piece]
    g0 = G.gap_to_rad(G.PIECE[1][0] + 24)
    unfold, q_up = joint_seq(q_look, UNFOLD, g0)
    f = Follower("right")
    pk = Picker(gp, f, ql, others)
    try:
        if not f.run(unfold + [Segment("open", [q_up] * G.HOLD, g0)]):
            raise RuntimeError("unfold stopped (blocked)")
        held, q, down, xy = pk.pick((piece["x"], piece["y"]), 1, q_up)
        print("RESULT:", "HELD" if held else "missed")
        if held and down is not None:
            ok, why = place_on_cell(pk, down, 1, 4, q_up, q_look)
            print("PLACE:", why)
            if ok:
                return
        if down is not None:
            if held:   # lift to show it, then put it back
                gu = max(0.0, pk.grip_units)
                for qq in list(reversed(down)):
                    f.command(qq, gripper_units=gu); time.sleep(1 / 30)
                time.sleep(1.0)
                for qq in down:
                    f.command(qq, gripper_units=gu); time.sleep(1 / 30)
            f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
        back = P.staged_path(pk.chk, "right", q, q_up, g0)
        fold, _ = joint_seq(q_up, FOLD, g0)
        f.run([Segment("back", back, g0)] + fold)
    except Exception:
        import traceback
        traceback.print_exc()
        G.return_to_look(gp, f)
    finally:
        pk.cam.close()
        f.disconnect()


if __name__ == "__main__":
    demo()
