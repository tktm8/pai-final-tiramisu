"""DAgger data collection: repeated desk picks with the current policy; the person corrects with the leader arm.

Each episode: overhead camera finds an orange piece on this arm's desk -> hover above it, deliberately off by up to
--noise mm -> the policy (or, before the first training, the image-Jacobian correction) moves the hover -> grasp
-> retries -> if it still misses, the leader arm takes over and the person shows the spot -> the robot grasps there.
Held pieces are let go where they were gripped (no carrying), so the desk stays as it is.

  ~/Programing/so-101/lerobot/.venv/bin/python robot/dagger_collect.py right --episodes 20
  ... --expert keyboard          # no leader arm: type x+3 / y-2 / ok
  ... --sizes 2,1 --noise 10 --pause

Then train:  robot/dagger_train.py   and collect again (next DAgger round).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot"), str(ROOT / "perception")]
import dagger as DG  # noqa: E402
import dagger_expert as DX  # noqa: E402
import move_test as MT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["right", "left"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--sizes", default="2,1,0", help="piece sizes to cycle through (0 S, 1 M, 2 L)")
    ap.add_argument("--noise", type=float, default=8.0, help="start hover error, mm (uniform in +-noise)")
    ap.add_argument("--expert", choices=["leader", "keyboard", "none"], default="leader")
    ap.add_argument("--leader-port", default=DX.LEADER_PORT)
    ap.add_argument("--leader-id", default=None, help="bimanual_right / bimanual_left (default: whichever matches)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between episodes (to move pieces around)")
    ap.add_argument("--no-policy", action="store_true", help="use the image-Jacobian correction even if a policy exists")
    a = ap.parse_args()

    expert = None
    if a.expert == "leader":
        expert = DX.LeaderExpert(a.leader_port, a.leader_id)
    elif a.expert == "keyboard":
        expert = DX.KeyboardExpert()
    MT.DAGGER.update(use_policy=not a.no_policy, record=True, expert=expert, pick_only=True, hover_noise=a.noise)

    st = DG.state()
    pol = DG.policy()
    print(f"DAgger round {st['round']}: policy = {'trained (round %d)' % pol.round if pol and not a.no_policy else 'image Jacobian'}")
    sizes = [int(s) for s in a.sizes.split(",")]
    held = 0
    try:
        for k in range(a.episodes):
            size = sizes[k % len(sizes)]
            print(f"\n=== episode {k + 1}/{a.episodes}: {'SML'[size]} piece, {a.arm} arm ===")
            t0 = time.time()
            ok = MT.move(("reserve", size), 4, size, arm=a.arm)
            held += bool(ok)
            print(f"=== {'HELD' if ok else 'missed'} in {time.time() - t0:.0f} s  ({held}/{k + 1} so far)")
            if a.pause and k < a.episodes - 1:
                input("Enter for the next episode (move the pieces around if you like) ")
    finally:
        if isinstance(expert, DX.LeaderExpert):
            expert.close()
    print(f"\nsession: {held}/{a.episodes} held.  Next: robot/dagger_train.py, then collect again.")


if __name__ == "__main__":
    main()
