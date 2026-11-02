"""DAgger for the last few mm of a desk pick.

Policy:  wrist-camera image at the hover (+ arm, piece size, jaw angle)  ->  (dx, dy) mm to move the hover so the
         piece ends up between the jaws.  A small CNN (ResNet18 features + an ensemble of 5 heads).
Data:    every hover the robot visits during a pick is a state.  Its label is
           - hindsight: the hover xy where the grasp finally held, minus this hover's xy
           - expert:    the xy the person drove the arm to with the leader arm (when the robot gave up)
Loop:    collect with the current policy -> label -> aggregate -> retrain  (robot/dagger_collect.py,
         robot/dagger_train.py).  Round 0 uses the old image-Jacobian correction as the policy.

Files (out/dagger/):  images/*.jpg  samples.jsonl  episodes.jsonl  policy.pt  state.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DAGGER_DIR", ROOT / "out" / "dagger"))   # DAGGER_DIR: tests
IMGS = DATA / "images"
SAMPLES = DATA / "samples.jsonl"
EPISODES = DATA / "episodes.jsonl"
MODEL = DATA / "policy.pt"
STATE = DATA / "state.json"

IMG_W, IMG_H = 320, 180          # network input (wrist frames are 16:9)
SAVE_W, SAVE_H = 640, 360        # stored frames
ARMS = ("left", "right")
ANGLES = (0, 30, 60, 90)
LABEL_SCALE = 10.0               # network predicts mm / 10
MAX_STEP = 12.0                  # never move more than this on the policy's word


def cond_vector(arm, size, angle):
    v = np.zeros(len(ARMS) + 3 + len(ANGLES), np.float32)
    v[ARMS.index(arm)] = 1
    v[len(ARMS) + int(size)] = 1
    a = min(ANGLES, key=lambda x: abs(x - float(angle)))
    v[len(ARMS) + 3 + ANGLES.index(a)] = 1
    return v


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"round": 0}


def save_state(s):
    DATA.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


# ---------------------------------------------------------------- model (torch imported lazily)
def build_net(n_heads=5, pretrained=False):
    import torch
    import torch.nn as nn
    import torchvision

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights=None)
            if pretrained:
                ck = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
                if ck.exists():
                    r.load_state_dict(torch.load(ck, map_location="cpu"))
            self.backbone = nn.Sequential(*list(r.children())[:-1])
            nc = len(ARMS) + 3 + len(ANGLES)
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(512 + nc, 128), nn.ReLU(), nn.Dropout(0.2),
                                                      nn.Linear(128, 2)) for _ in range(n_heads)])

        def forward(self, x, c):
            z = self.backbone(x).flatten(1)
            z = torch.cat([z, c], 1)
            return torch.stack([h(z) for h in self.heads], 1)       # (B, heads, 2)

    return Net()


def to_tensor(img_bgr):
    import cv2
    import torch
    im = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    im = (im - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))


class Policy:
    def __init__(self, path=MODEL):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = build_net(ck.get("n_heads", 5))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.round = ck.get("round", 0)

    def predict(self, img_bgr, arm, size, angle):
        """-> (step mm (2,), spread mm): the ensemble mean and how much the heads disagree."""
        import torch
        with torch.no_grad():
            x = to_tensor(img_bgr)[None]
            c = torch.from_numpy(cond_vector(arm, size, angle))[None]
            y = self.net(x, c)[0].numpy() * LABEL_SCALE           # (heads, 2)
        step = y.mean(0)
        n = np.linalg.norm(step)
        if n > MAX_STEP:
            step *= MAX_STEP / n
        return step, float(np.linalg.norm(y.std(0)))


_policy = {"obj": None, "mtime": None}


def policy():
    """The latest trained policy, or None before the first training round."""
    if not MODEL.exists():
        return None
    m = MODEL.stat().st_mtime
    if _policy["mtime"] != m:
        try:
            _policy["obj"], _policy["mtime"] = Policy(), m
        except Exception as e:      # torch missing / broken file: fall back to the old correction
            print(f"dagger: cannot load policy ({e})")
            return None
    return _policy["obj"]


# ---------------------------------------------------------------- recording
class Episode:
    """One desk pick: the hovers visited (states) and how it ended."""

    def __init__(self, arm, size, angle, log=print):
        self.arm, self.size, self.angle, self.log = arm, int(size), int(round(angle)), log
        self.id = time.strftime("%Y%m%d_%H%M%S")
        self.round = state()["round"]
        self.steps = []
        self.expert = None

    def add(self, img_bgr, xy_hover, pred=None, source="", spread=None):
        """Remember a visited hover (img None: nothing to learn from, skipped)."""
        if img_bgr is None:
            return
        import cv2
        IMGS.mkdir(parents=True, exist_ok=True)
        name = f"{self.id}_{len(self.steps)}.jpg"
        cv2.imwrite(str(IMGS / name), cv2.resize(img_bgr, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.steps.append(dict(img=name, xy=[float(v) for v in xy_hover],
                               pred=None if pred is None else [float(v) for v in pred], source=source,
                               spread=spread))

    def finish(self, held, xy_held=None, xy_expert=None, attempts=0):
        """Label every visited hover and append to the dataset."""
        target, kind = (xy_expert, "expert") if xy_expert is not None else (xy_held, "hindsight") if held else (None, None)
        DATA.mkdir(parents=True, exist_ok=True)
        if target is not None:
            with SAMPLES.open("a") as fh:
                for s in self.steps:
                    lab = np.asarray(target, float) - np.asarray(s["xy"], float)
                    fh.write(json.dumps(dict(episode=self.id, round=self.round, arm=self.arm, size=self.size,
                                             angle=self.angle, img=s["img"], label=[float(lab[0]), float(lab[1])],
                                             kind=kind, held=bool(held), pred=s["pred"], source=s["source"],
                                             spread=s["spread"])) + "\n")
        first = self.steps[0] if self.steps else None
        err0 = None
        if target is not None and first and first["pred"] is not None:
            err0 = float(np.linalg.norm(np.asarray(target) - np.asarray(first["xy"]) - np.asarray(first["pred"])))
        with EPISODES.open("a") as fh:
            fh.write(json.dumps(dict(episode=self.id, round=self.round, arm=self.arm, size=self.size, angle=self.angle,
                                     held=bool(held), attempts=int(attempts), expert=xy_expert is not None,
                                     policy=first["source"] if first else None, first_pred_error_mm=err0,
                                     n_states=len(self.steps) if target is not None else 0)) + "\n")
        self.log(f"  dagger: round {self.round}, {'HELD' if held else 'missed'} after {attempts} tries, "
                 f"{len(self.steps) if target is not None else 0} labelled states ({kind or 'none'})"
                 + (f", first-step error {err0:.1f} mm" if err0 is not None else ""))
