"""DAgger expert: when the robot gives up on a desk pick, the person shows where it should have grasped.

LeaderExpert   - the SO-101 leader arm drives the follower (joint mirroring, jaws kept open). Before handing over,
                 the leader is moved by its own motors to the follower's pose, so the follower does not jump.
                 Squeeze the leader's gripper = "grasp here".
KeyboardExpert - no leader arm: type moves at the hover ("x+3", "y-2", "ok", "skip").

Both return the corrected hover xy (arm frame, mm) and the joint pose the arm is left in, or None (skipped).
"""

from __future__ import annotations

import time

import numpy as np

import syspath  # noqa: F401  (so-101/system on the path)
from config import ARM_JOINTS, CONTROL_HZ  # noqa: E402
from kinematics import gripper_urdf_to_lerobot, lerobot_deg_to_q  # noqa: E402

LEADER_PORT = "/dev/tty.usbmodem5B421352171"
SQUEEZE_DEG = 20.0      # leader jaws this much more closed than at hand-over = "here"
TIMEOUT_S = 120.0
Z_MIN = 4.0             # never drive the TCP lower than this (mm above the desk)


class Ctx:
    """What the expert needs from the pick in progress."""

    def __init__(self, f, arm, pk, xy, qh, a, c, g_open, kin, log=print):
        self.f, self.arm, self.pk, self.xy, self.qh, self.a, self.c = f, arm, pk, np.asarray(xy, float), qh, a, c
        self.g_open, self.kin, self.log = g_open, kin, log


class LeaderExpert:
    def __init__(self, port=LEADER_PORT, calib_id=None, log=print):
        self.port, self.calib_id, self.log = port, calib_id, log
        self.leader = None

    def connect(self):
        if self.leader is not None:
            return self.leader
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
        for cid in ([self.calib_id] if self.calib_id else ["bimanual_right", "bimanual_left"]):
            ld = SOLeader(SOLeaderTeleopConfig(port=self.port, id=cid, use_degrees=True))
            ld.connect(calibrate=False)
            if ld.is_calibrated:
                self.leader, self.calib_id = ld, cid
                self.log(f"  leader arm on {self.port} ({cid})")
                return ld
            ld.disconnect()
        raise RuntimeError(f"no leader calibration matches the arm on {self.port}")

    def close(self):
        if self.leader is not None:
            try:
                self.leader.disable_torque()
                self.leader.disconnect()
            finally:
                self.leader = None

    def _align(self, target, seconds=2.5):
        """Drive the leader to the follower's pose with its own motors, then let go."""
        ld = self.leader
        now = {k.removesuffix(".pos"): v for k, v in ld.get_action().items()}
        ld.enable_torque()
        n = int(seconds * CONTROL_HZ)
        for i in range(1, n + 1):
            ld.send_feedback({f"{m}.pos": now[m] + (target[m] - now[m]) * i / n for m in target})
            time.sleep(1 / CONTROL_HZ)
        time.sleep(0.6)
        ld.disable_torque()
        got = {k.removesuffix(".pos"): v for k, v in ld.get_action().items()}
        return max(abs(got[m] - target[m]) for m in ARM_JOINTS)

    def __call__(self, ctx: Ctx):
        ld = self.connect()
        f = ctx.f
        fol = f.present()                                  # follower pose, LeRobot degrees (gripper open)
        err = self._align(fol)
        ctx.log(f"  leader aligned to the follower (worst joint {err:.1f} deg)")
        if err > 12.0:
            ctx.log("  leader did not reach the follower's pose: match it by hand (the follower waits)")
            t0 = time.time()
            while time.time() - t0 < 30:
                a = ld.get_action()
                if max(abs(a[f"{m}.pos"] - fol[m]) for m in ARM_JOINTS) < 8.0:
                    break
                time.sleep(0.1)
            else:
                return None
        g0 = ld.get_action()["gripper.pos"]
        g_units = gripper_urdf_to_lerobot(ctx.arm, ctx.g_open)
        ctx.log("  >>> EXPERT: move the leader arm so the piece is between the jaws, then squeeze its gripper")
        t0, sq_since, last_ok = time.time(), None, fol
        dt = 1.0 / CONTROL_HZ
        while time.time() - t0 < TIMEOUT_S and not f.stop:
            tl = time.perf_counter()
            a = ld.get_action()
            tgt = {m: a[f"{m}.pos"] for m in ARM_JOINTS}
            q = lerobot_deg_to_q(ctx.arm, tgt)
            pos, _, _ = ctx.kin.tcp(q)
            if pos[2] >= Z_MIN:
                last_ok = tgt
            act = {f"{m}.pos": v for m, v in last_ok.items()}
            act["gripper.pos"] = g_units                     # the follower's jaws stay open
            f.robot.send_action(act)
            if a["gripper.pos"] < g0 - SQUEEZE_DEG:
                sq_since = sq_since or time.time()
                if time.time() - sq_since > 0.3:
                    break
            else:
                sq_since = None
            time.sleep(max(0.0, dt - (time.perf_counter() - tl)))
        else:
            return None
        time.sleep(0.5)                                    # let the follower catch up
        q_at = lerobot_deg_to_q(ctx.arm, {m: v for m, v in f.present().items() if m in ARM_JOINTS})
        pos, _, _ = ctx.kin.tcp(q_at)
        ctx.log(f"  expert: grasp here -> TCP ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm "
                f"= hover {pos[0] - ctx.xy[0]:+.1f}, {pos[1] - ctx.xy[1]:+.1f} mm")
        return pos[:2].copy(), q_at


class KeyboardExpert:
    """Jog the hover from the terminal."""

    def __call__(self, ctx: Ctx):
        xy, q = ctx.xy.copy(), ctx.qh
        ctx.log("  >>> EXPERT: type x+3 / y-2 (mm) to move the jaws over the piece, 'ok' to grasp, 'skip' to give up")
        while True:
            s = input("  move> ").strip().lower()
            if s == "ok":
                return xy, q
            if s == "skip":
                return None
            try:
                ax, v = s[0], float(s[1:])
                d = np.array([v, 0.0]) if ax == "x" else np.array([0.0, v]) if ax == "y" else None
            except ValueError:
                d = None
            if d is None:
                print("  ?  e.g. x+3, y-2, ok, skip")
                continue
            q2 = ctx.pk.move_hover(xy + d, ctx.a, ctx.c, q, ctx.g_open)
            if q2 is not None:
                xy, q = xy + d, q2
                print(f"  hover now {xy[0] - ctx.xy[0]:+.1f}, {xy[1] - ctx.xy[1]:+.1f} mm from the robot's choice")
