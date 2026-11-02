"""Reuse ~/Programing/so-101/system (IK, planner, collision, safe follower I/O) with the Gobblet assembly.

Import this module before any so-101/system module: it patches `config` so the collision environment
and caches point at this project instead of the Connect 4 station.
"""

import sys
from pathlib import Path

SYSTEM = Path.home() / "Programing/so-101/system"
ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM) not in sys.path:
    sys.path.insert(0, str(SYSTEM))

import config  # noqa: E402

config.ASSEMBLY_PARTS = ROOT / "gobblet_full_assembly" / "parts"
config.OUT = ROOT / "out"
config.OUT.mkdir(exist_ok=True)
