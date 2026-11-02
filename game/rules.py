"""Gobblet Gobblers rules, game state and the C search wrapper.

Cells: index = row * 3 + col in the world/robot frame.
  row 0 = nearest the robot (world X 138), row 2 = nearest the human (X 278).
  col 0 = robot's left (world Y +70), col 2 = robot's right (Y -70).
Sizes: 0 small, 1 medium, 2 large. Players: 0 robot, 1 human.
"""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESERVE = 9
SIZE_NAMES = ["小", "中", "大"]
LINES = [0o007, 0o070, 0o700, 0o111, 0o222, 0o444, 0o421, 0o124]
ROBOT, HUMAN = 0, 1


@dataclass(frozen=True)
class Move:
    frm: int      # 0..8 board cell, 9 = reserve
    to: int
    size: int

    def describe(self, *, for_human: bool = True) -> str:
        to = cell_name(self.to, for_human)
        if self.frm == RESERVE:
            return f"{SIZE_NAMES[self.size]}を{to}に置く"
        return f"{SIZE_NAMES[self.size]}を{cell_name(self.frm, for_human)}から{to}へ動かす"

    def to_dict(self) -> dict:
        return {"from": self.frm, "to": self.to, "size": self.size}


def cell_name(c: int, for_human: bool = True) -> str:
    """Cell name as seen by the human (who sits across the board) or by the robot."""
    r, k = divmod(c, 3)
    if for_human:
        row = ["奥", "まんなか", "手前"][r]
        col = ["右", "まんなか", "左"][k]
    else:
        row = ["手前", "まんなか", "奥"][r]
        col = ["左", "まんなか", "右"][k]
    if r == 1 and k == 1:
        return "まんなか"
    if r == 1:
        return f"まんなかの{col}"
    if k == 1:
        return f"{row}のまんなか"
    return f"{row}の{col}"


@dataclass
class State:
    m: list[int] = field(default_factory=lambda: [0] * 6)   # m[player*3+size] = 9-bit cell mask
    turn: int = HUMAN
    history: list = field(default_factory=list)

    def copy(self) -> "State":
        return State(list(self.m), self.turn, list(self.history))

    # ---- queries ----
    def occ(self, s: int) -> int:
        return self.m[s] | self.m[3 + s]

    def stack(self, c: int) -> list[tuple[int, int]]:
        """[(player, size)] bottom -> top."""
        return [(p, s) for s in range(3) for p in range(2) if self.m[p * 3 + s] >> c & 1]

    def top(self, c: int):
        st = self.stack(c)
        return st[-1] if st else None

    def visible(self, pl: int) -> int:
        v = 0
        for c in range(9):
            t = self.top(c)
            if t and t[0] == pl:
                v |= 1 << c
        return v

    def reserve(self, pl: int) -> list[int]:
        return [2 - bin(self.m[pl * 3 + s]).count("1") for s in range(3)]

    def legal_moves(self, pl: int | None = None) -> list[Move]:
        pl = self.turn if pl is None else pl
        out = []
        vis = self.visible(pl)
        for s in (2, 1, 0):
            land = [c for c in range(9) if not any(self.occ(t) >> c & 1 for t in range(s, 3))]
            if self.reserve(pl)[s] > 0:
                out += [Move(RESERVE, c, s) for c in land]
            for f in range(9):
                if self.m[pl * 3 + s] >> f & 1 and vis >> f & 1:
                    out += [Move(f, c, s) for c in land if c != f]
        return out

    def apply(self, mv: Move) -> "State":
        n = self.copy()
        pl = self.turn
        if mv.frm != RESERVE:
            n.m[pl * 3 + mv.size] &= ~(1 << mv.frm)
        n.m[pl * 3 + mv.size] |= 1 << mv.to
        n.turn = 1 - pl
        n.history.append((pl, mv))
        return n

    def winner(self) -> int | None:
        """Winner after the last move (the mover loses if the opponent shows a line)."""
        if not self.history:
            return None
        mover = self.history[-1][0]
        if _line(self.visible(1 - mover)):
            return 1 - mover
        if _line(self.visible(mover)):
            return mover
        return None

    def top_view(self) -> list:
        """What a camera from above sees: per cell None or (player, size)."""
        return [self.top(c) for c in range(9)]

    def to_dict(self) -> dict:
        return {
            "cells": [[{"player": p, "size": s} for p, s in self.stack(c)] for c in range(9)],
            "turn": self.turn,
            "reserve": [self.reserve(0), self.reserve(1)],
            "moves": len(self.history),
        }


def _line(v: int) -> bool:
    return any(v & L == L for L in LINES)


# ---------------- search (C) ----------------
_lib = None


def _load():
    global _lib
    if _lib is None:
        so = HERE / "libsolver.dylib"
        src = HERE / "solver.c"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["clang", "-O3", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        _lib = ctypes.CDLL(str(so))
        _lib.gg_search.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
        _lib.gg_score_moves.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    return _lib


WIN = 10000


@dataclass
class SearchResult:
    move: Move
    score: int
    depth: int
    nodes: int

    @property
    def forced_win_in(self) -> int | None:
        """Plies until the forced win (counting this move as 1), if the search proved one."""
        return WIN - self.score if self.score > WIN - 200 else None

    @property
    def forced_loss_in(self) -> int | None:
        return WIN + self.score if self.score < -WIN + 200 else None


def best_move(st: State, time_ms: int = 2500, max_depth: int = 40) -> SearchResult:
    lib = _load()
    arr = (ctypes.c_uint16 * 6)(*st.m)
    out = (ctypes.c_int * 6)()
    if lib.gg_search(arr, st.turn, time_ms, max_depth, out) != 0:
        raise RuntimeError("no legal move")
    return SearchResult(Move(out[0], out[1], out[2]), out[3], out[4], out[5])


def score_moves(st: State, depth: int = 3) -> list[tuple[Move, int]]:
    lib = _load()
    arr = (ctypes.c_uint16 * 6)(*st.m)
    out = (ctypes.c_int * 640)()
    n = lib.gg_score_moves(arr, st.turn, depth, out, 160)
    return sorted(((Move(out[i * 4], out[i * 4 + 1], out[i * 4 + 2]), out[i * 4 + 3]) for i in range(n)),
                  key=lambda t: -t[1])


# ---------------- infer a move from a camera top view ----------------
def infer_move(st: State, observed: list, size_known: bool = True) -> tuple[list[Move], list]:
    """Legal moves of the side to move whose resulting top view matches `observed`.

    observed: per cell None or (player, size) or (player, None) when the size is uncertain.
    Returns (matching moves, changed cells). With full sizes the match is usually unique.
    """
    before = st.top_view()
    changed = [c for c in range(9) if not _same(before[c], observed[c], size_known)]
    hits = []
    for mv in st.legal_moves():
        after = st.apply(mv).top_view()
        if all(_same(after[c], observed[c], size_known) for c in range(9)):
            hits.append(mv)
    return hits, changed


def _same(a, b, size_known):
    if a is None or b is None:
        return a is None and b is None
    if a[0] != b[0]:
        return False
    if not size_known or b[1] is None:
        return True
    return a[1] == b[1]
