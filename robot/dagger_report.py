"""DAgger progress per round: grasp success, tries, expert help, and the policy's held-out error.

  ~/Programing/so-101/lerobot/.venv/bin/python robot/dagger_report.py      -> table + out/dagger/report.png
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot")]
import dagger as DG  # noqa: E402


def main():
    eps = [json.loads(l) for l in DG.EPISODES.read_text().splitlines()] if DG.EPISODES.exists() else []
    rounds = json.loads((DG.DATA / "rounds.json").read_text()) if (DG.DATA / "rounds.json").exists() else []
    by = defaultdict(list)
    for e in eps:
        by[e["round"]].append(e)
    rows = []
    print(f"{'round':>5} {'picks':>5} {'1st try':>8} {'held w/o help':>14} {'mean tries':>11} {'expert':>7} {'1st-step err':>13}")
    for r in sorted(by):
        es = by[r]
        first = np.mean([e["held"] and e["attempts"] == 1 and not e["expert"] for e in es])
        alone = np.mean([e["held"] and not e["expert"] for e in es])
        tries = np.mean([e["attempts"] for e in es])
        exp = np.mean([e["expert"] for e in es])
        err = [e["first_pred_error_mm"] for e in es if e["first_pred_error_mm"] is not None]
        err = float(np.mean(err)) if err else float("nan")
        rows.append((r, len(es), first, alone, tries, exp, err))
        print(f"{r:5d} {len(es):5d} {first:8.0%} {alone:14.0%} {tries:11.1f} {exp:7.0%} {err:10.1f} mm")
    for h in rounds:
        print(f"trained policy round {h['round']}: {h['samples']} states, held-out error "
              f"{h['val_error_mm'] if h['val_error_mm'] is None else round(h['val_error_mm'], 1)} mm "
              f"(no correction {h['no_correction_error_mm'] if h['no_correction_error_mm'] is None else round(h['no_correction_error_mm'], 1)} mm)")
    if not rows:
        print("no episodes yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = [x[0] for x in rows]
    ax[0].plot(r, [x[2] * 100 for x in rows], "o-", label="held on the 1st try")
    ax[0].plot(r, [x[3] * 100 for x in rows], "s-", label="held without the person")
    ax[0].set_xlabel("DAgger round"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 105); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xticks(r)
    ax[1].plot(r, [x[6] for x in rows], "o-", color="tab:orange", label="1st-step error on the robot")
    if rounds:
        ax[1].plot([h["round"] for h in rounds], [h["val_error_mm"] for h in rounds], "s--", color="tab:green",
                   label="held-out error after training")
    ax[1].set_xlabel("DAgger round"); ax[1].set_ylabel("mm"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_xticks(r)
    fig.suptitle("DAgger: wrist-camera grasp correction")
    fig.tight_layout()
    out = DG.DATA / "report.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
