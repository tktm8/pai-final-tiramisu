"""Train the DAgger grasp-correction policy on everything collected so far (aggregated dataset).

  ~/Programing/so-101/lerobot/.venv/bin/python robot/dagger_train.py [--epochs 60]

Writes out/dagger/policy.pt, bumps the DAgger round (new episodes are tagged with it) and appends the
validation numbers to out/dagger/rounds.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot")]
import dagger as DG  # noqa: E402

ROUNDS = DG.DATA / "rounds.json"


def load():
    rows = []
    if DG.SAMPLES.exists():
        for line in DG.SAMPLES.read_text().splitlines():
            r = json.loads(line)
            if (DG.IMGS / r["img"]).exists():
                rows.append(r)
    return rows


class Data(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.cache.get(r["img"])
        if img is None:
            img = cv2.imread(str(DG.IMGS / r["img"]))
            self.cache[r["img"]] = img
        if self.train:      # lighting only: a shifted image would need a shifted label
            im = img.astype(np.float32)
            im = im * random.uniform(0.75, 1.25) + random.uniform(-20, 20)
            im = im * np.array([random.uniform(0.9, 1.1) for _ in range(3)], np.float32)
            im = np.clip(im, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                im = cv2.GaussianBlur(im, (5, 5), 0)
        else:
            im = img
        x = DG.to_tensor(im)
        c = torch.from_numpy(DG.cond_vector(r["arm"], r["size"], r["angle"]))
        y = torch.tensor(r["label"], dtype=torch.float32) / DG.LABEL_SCALE
        w = torch.tensor(2.0 if r["kind"] == "expert" else 1.0)
        return x, c, y, w


def evaluate(net, rows, dev):
    if not rows:
        return None
    net.eval()
    errs = []
    with torch.no_grad():
        for x, c, y, _ in torch.utils.data.DataLoader(Data(rows, False), batch_size=32):
            p = net(x.to(dev), c.to(dev)).mean(1).cpu()
            errs += (torch.linalg.norm(p - y, dim=1) * DG.LABEL_SCALE).tolist()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val", type=float, default=0.2, help="share of episodes held out")
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    rows = load()
    eps = sorted({r["episode"] for r in rows})
    print(f"{len(rows)} labelled states from {len(eps)} episodes "
          f"({sum(r['kind'] == 'expert' for r in rows)} expert, {sum(r['kind'] == 'hindsight' for r in rows)} hindsight)")
    if len(rows) < 8:
        sys.exit("not enough data yet: collect more with robot/dagger_collect.py")
    rng = random.Random(1)
    val_eps = set(rng.sample(eps, max(1, int(len(eps) * a.val)))) if len(eps) >= 5 else set()
    tr = [r for r in rows if r["episode"] not in val_eps]
    va = [r for r in rows if r["episode"] in val_eps]

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = DG.build_net(pretrained=True).to(dev)
    for i, m in enumerate(net.backbone):          # small data: only the last ResNet block and the heads learn
        for p in m.parameters():
            p.requires_grad = i >= 7
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    dl = torch.utils.data.DataLoader(Data(tr, True), batch_size=16, shuffle=True)
    for ep in range(a.epochs):
        net.train()
        for i, m in enumerate(net.backbone):      # keep frozen BatchNorm statistics fixed
            if i < 7:
                m.eval()
        tot = 0.0
        for x, c, y, w in dl:
            x, c, y, w = x.to(dev), c.to(dev), y.to(dev), w.to(dev)
            out = net(x, c)                                        # (B, heads, 2)
            mask = (torch.rand(out.shape[:2], device=dev) < 0.8).float()   # bootstrap: each head sees ~80 %
            l = (loss_fn(out, y[:, None].expand_as(out)).sum(2) * mask * w[:, None]).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); l.backward(); opt.step()
            tot += float(l) * len(x)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep + 1:3d}  loss {tot / len(tr):.4f}  val error {evaluate(net, va, dev) or float('nan'):.2f} mm")

    val_mae = evaluate(net, va, dev)
    zero = float(np.mean([np.linalg.norm(r["label"]) for r in va])) if va else None
    recorded = [r for r in va if r["pred"] is not None]
    rec_mae = float(np.mean([np.linalg.norm(np.array(r["label"]) - np.array(r["pred"])) for r in recorded])) if recorded else None
    st = DG.state()
    new_round = st["round"] + 1
    torch.save({"model": net.cpu().state_dict(), "n_heads": len(net.heads), "round": new_round,
                "trained": time.strftime("%F %T"), "n_samples": len(rows)}, DG.MODEL)
    st["round"] = new_round
    DG.save_state(st)
    hist = json.loads(ROUNDS.read_text()) if ROUNDS.exists() else []
    hist.append(dict(round=new_round, time=time.strftime("%F %T"), samples=len(rows), episodes=len(eps),
                     val_states=len(va), val_error_mm=val_mae, no_correction_error_mm=zero,
                     previous_policy_error_mm=rec_mae))
    ROUNDS.write_text(json.dumps(hist, indent=1))
    f = lambda v: "-" if v is None else f"{v:.1f} mm"
    print(f"\npolicy round {new_round} saved: held-out error {f(val_mae)}  "
          f"(no correction {f(zero)}, the policy that collected the data {f(rec_mae)})")


if __name__ == "__main__":
    main()
