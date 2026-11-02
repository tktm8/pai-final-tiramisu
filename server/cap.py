"""Code as Policies: Codex writes a short Python program against a small robot API; we validate and run it.

Used where fixed code is awkward: tidying the board after a game, and recovering when the camera shows the
board differs from what the robot intended. Every motion still goes through the collision-checked planner;
the generated code only decides *which* primitive moves to make.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import re
import threading

from codex_llm import ask

log = logging.getLogger("cap")

API_DOC = """\
Available functions (nothing else exists; no imports):
  board() -> dict          # {"cells": [[{"player":0|1,"size":0|1|2}, ...bottom->top] x9], "robot_reserve_slots": {slot: size or None}}
                           # player 0 = robot (orange), 1 = human (blue). size 0 small, 1 medium, 2 large.
                           # cells are indexed row*3+col; row 0 nearest the robot, col 0 = robot's left.
  read_board() -> list     # camera: per cell None or {"color": "orange"|"blue", "size": 0|1|2}  (top piece only)
  move_piece(src, dst, size) -> bool
                           # src/dst: a cell index 0..8, or "R<slot>" for a robot reserve slot (e.g. "R3").
                           # Only ORANGE pieces may be moved, and only the top piece of a cell. Returns success.
  say(text)                # StackChan speaks (short Japanese)
  wait(seconds)
Rules for the program: at most 12 move_piece calls; no while loops; stop and say() if something fails.
"""

ALLOWED_CALLS = {"board", "read_board", "move_piece", "say", "wait", "range", "len", "min", "max", "enumerate",
                 "sorted", "list", "dict", "int", "str", "print", "reversed", "any", "all", "zip"}
MAX_MOVES = 12
SAFE_BUILTINS = ("range", "len", "min", "max", "enumerate", "sorted", "list", "dict", "int", "str", "print",
                 "reversed", "any", "all", "zip")


class CapError(Exception):
    pass


def validate(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal, ast.Lambda,
                             ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Try)):
            raise CapError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ("get", "items", "keys", "values", "append", "index", "count"):
                raise CapError(f"attribute not allowed: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise CapError(f"name not allowed: {node.id}")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id not in ALLOWED_CALLS and not _is_local_def(tree, f.id):
                raise CapError(f"call not allowed: {f.id}")
    return tree


def _is_local_def(tree, name):
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def generate(goal: str, context: dict) -> str | None:
    prompt = (
        "You control a robot arm playing Gobblet Gobblers through a tiny Python API. "
        "Write ONLY a Python program (in one ```python block) that achieves the goal.\n\n"
        f"{API_DOC}\nCurrent state:\n{json.dumps(context, ensure_ascii=False)}\n\nGoal: {goal}\n"
    )
    out = ask(prompt, timeout=120)
    return extract_code(out) if out else None


def run(code: str, api: dict, timeout: float = 600.0) -> dict:
    """Execute validated code with only `api` functions. Returns {"ok", "moves", "error"}."""
    validate(code)
    moves = []

    def move_piece(src, dst, size):
        if len(moves) >= MAX_MOVES:
            raise CapError("too many moves")
        moves.append((src, dst, size))
        return api["move_piece"](src, dst, size)

    env = {"__builtins__": {k: getattr(builtins, k) for k in SAFE_BUILTINS}}
    env.update({k: v for k, v in api.items() if k != "move_piece"})
    env["move_piece"] = move_piece
    result = {"ok": False, "moves": moves, "error": None}

    def target():
        try:
            exec(compile(code, "<cap>", "exec"), env)
            result["ok"] = True
        except Exception as e:  # report back to the game, never crash it
            result["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        result["error"] = "timeout"
    return result
