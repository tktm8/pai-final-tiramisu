"""Teach the desk pick with the person's help (one arm, every orange piece on its side of the desk).

For each piece the overhead camera finds:
  1. the arm goes to the hover above it (jaws open, along Y = 90 deg, the usual pick orientation)
  2. the person slides the piece to the middle of the open jaws, then the operator writes "ok" to
     out/wrist_teach.cmd ("skip" = next piece, "quit" = stop)
  3. wrist photo -> REF[arm, size, 90] (where a perfectly placed piece shows up); the first time also the image
     Jacobian J (two 10 mm test moves)
  4. test grasp: down, close gently, report HELD / missed, open, up (the piece stays where it was gripped)
At the end: back to the camera pose, overhead photo -> one desk-offset sample per piece (true arm position vs
camera estimate) -> out/desk_offset_<arm>.json.

  python robot/wrist_teach.py right|left
"""

from __future__ import annotations

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
import move_test as MT  # noqa: E402
import sam_pieces  # noqa: E402
import topcam  # noqa: E402
import wrist_refine as WR  # noqa: E402
import wrist_servo as W  # noqa: E402
from planner import Segment  # noqa: E402

CMD = ROOT / "out" / "wrist_teach.cmd"
log = lambda *a: print(*a, flush=True)


def wait_cmd():
    CMD.write_text("")
    while True:
        t = CMD.read_text().strip().lower() if CMD.exists() else ""
        if t in ("ok", "skip", "quit"):
            CMD.write_text("")
            return t
        time.sleep(0.3)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "right"
    MT.ARM[0] = arm
    G.PIECE_ARM = arm
    gp = G.GobbletPlanner()
    MT.fix_left_z(gp.planner)
    gp.k = gp.planner.kin[arm]
    P = gp.planner
    ql = G.current_q(MT.OTHER[arm])
    q_photo = np.array(json.loads((ROOT / "out" / "camera_pose.json").read_text())[arm])
    a_dn = np.array([0.0, 0.0, -1.0])
    c = np.array([0.0, 1.0, 0.0])
    taught = []          # (true piece centre in arm coords, size)
    f = MT.connect_follower(log)
    try:
        MT.go_to_camera_pose(f, q_photo)
        time.sleep(0.5)
        frame = topcam.grab(warmup=40)
        cam = topcam.BoardCam.load()
        desk = sam_pieces.detect(frame, cam, 0, region=MT.DESK_REGION[arm])
        if len(sys.argv) > 2:                    # only these sizes, e.g. "M" or "SM"
            desk = [p for p in desk if "SML"[p["size"]] in sys.argv[2].upper()]
        log(f"desk pieces ({arm}): {[('SML'[p['size']], p['x'], p['y']) for p in desk]}")
        q_cam = q_photo
        if abs(np.degrees(q_photo[0])) > 80:
            segs, q_cam = W.joint_seq(q_photo, [(0, MT.WORK_PAN * MT.SIDE[arm])], 0.0)
            f.run(segs)
        pk = W.Picker(gp, f, ql, [])
        q = q_cam
        at_camera = True
        for i, p in enumerate(desk):
            size = p["size"]
            d, h = G.PIECE[size]
            g0 = G.gap_to_rad(d + 24.0)
            txy = np.array(MT.desk_to_arm(p["x"], p["y"]))
            xy = txy - c[:2] * (d + 24.0) / 2
            f.run([Segment("open", [q] * G.HOLD, g0)])
            if at_camera:
                q_h = gp.k.ik_pose(np.array([xy[0], xy[1], W.HOVER_Z]), a_dn, c, q_cam, n_random=4)
                way = MT.route(P, pk.chk, q_cam, q_h, g0) if q_h is not None else None
                if way is None or not MT.run_retry(f, q_cam, [Segment("to_hover", way, g0)], log):
                    log(f"PIECE #{i} {'SML'[size]}: cannot get above it - skipped")
                    continue
                q, at_camera = q_h, False
            else:
                q2 = pk.move_hover(xy, a_dn, c, q, g0)
                if q2 is None:
                    log(f"PIECE #{i} {'SML'[size]}: cannot get above it - skipped")
                    continue
                q = q2
            log(f"WAITING #{i} {'SML'[size]} at camera ({p['x']:.0f}, {p['y']:.0f}): slide it to the middle of the jaws")
            cmd = wait_cmd()
            if cmd == "quit":
                break
            if cmd == "skip":
                continue
            wr = WR.WristRefine(arm, size, 90, log)
            b = wr.find(wr.ref(), f"teach{i}")
            if b is None:
                log("  wrist camera does not see the piece - no REF")
            else:
                wr.b0 = b
                wr.st[wr.kref] = [float(v) for v in b]
                wr.save()
                log(f"  REF {arm}/{'SML'[size]}/90deg = {b.round().tolist()}")
                if True:                       # re-measure J every time: the piece is still now
                    hq = {"q": q}

                    def _move(xy_t):
                        q_t = pk.move_hover(xy_t, a_dn, c, hq["q"], g0)
                        if q_t is not None:
                            hq["q"] = q_t
                        return q_t is not None
                    wr.learn_J(_move, xy)
                    q = hq["q"]
            wr.close()
            held, down = pk.try_grasp(q, xy, a_dn, c, size, g0)
            log(f"  test grasp: {'HELD' if held else 'missed'}")
            if down is not None:
                f.run([Segment("open", [down[-1]] * G.HOLD, g0), Segment("up", list(reversed(down)), g0)])
                if held:
                    taught.append((xy + c[:2] * d / 2, size))      # where the closed jaws left the piece
                    # the jaws pushed it against the fixed jaw: it is now half a diameter from the TCP
        if not at_camera:
            f.run(MT.safe_home_segments(q, q_cam, G.gap_to_rad(30.0)))
        if abs(np.degrees(q_photo[0]) - np.degrees(q_cam[0])) > 3:
            segs, _ = W.joint_seq(q_cam, [(0, float(np.degrees(q_photo[0])))], 0.0)
            f.run(segs)
        # desk offset from where the pieces now are
        if taught:
            time.sleep(0.5)
            frame = topcam.grab(warmup=40)
            now = sam_pieces.detect(frame, cam, 0, region=MT.DESK_REGION[arm])
            offs = []
            for true_xy, size in taught:
                est = [(np.array(MT.to_arm(o["x"], o["y"])), o) for o in now]
                if not est:
                    break
                e, o = min(est, key=lambda t: np.hypot(*(t[0] - true_xy)))
                off = true_xy - e
                if np.hypot(*off) < 30:
                    offs.append(off)
                    log(f"  desk sample {'SML'[size]}: true {true_xy.round(1)} camera->arm {e.round(1)} "
                        f"offset ({off[0]:+.1f}, {off[1]:+.1f})")
            if offs:
                sf = ROOT / "out" / f"wrist_teach_desk_samples_{arm}.json"
                allo = (json.loads(sf.read_text()) if sf.exists() else []) + [o.tolist() for o in offs]
                sf.write_text(json.dumps(allo, indent=1))
                offs = [np.array(o) for o in allo[-12:]]          # average over the recent teach sessions
                m = np.mean(offs, axis=0)
                out = ROOT / "out" / f"desk_offset_{arm}.json"
                out.write_text(json.dumps({"dx": float(m[0]), "dy": float(m[1]),
                                           "note": f"{arm} arm, wrist_teach {time.strftime('%F %T')}, "
                                                   f"{len(offs)} pieces, spread {np.std(offs, axis=0).round(1).tolist()}"},
                                          indent=1))
                log(f"RESULT desk offset {arm} = ({m[0]:+.1f}, {m[1]:+.1f}) mm from {len(offs)} pieces")
    finally:
        if not f.stop:
            f.disconnect()
        else:
            log("stopped: torque held")


if __name__ == "__main__":
    main()
