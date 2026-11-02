"""Gobblet Gobblers: StackChan + SO-101 vs a human. Game server (phone page + game loop).

  ~/Programing/so-101/lerobot/.venv/bin/python server/game_server.py                 # everything simulated
  ... --vision on                 # read the human's moves with the top camera
  ... --arm on --tap on           # the right arm moves pieces, the left arm taps the phone

Flow: pat StackChan's head -> QR on its face -> phone opens this page -> pick who starts ->
human moves and taps "置いた" -> camera reads the board -> robot thinks (solver) and talks (Codex)
-> right arm moves an orange piece -> left arm's stylus taps "ロボットおわり" on the phone -> ...
"""

from __future__ import annotations

import argparse
import numpy as np
import json
import logging
import random
import secrets
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "server"), str(ROOT / "perception"), str(ROOT / "robot")]

from game.rules import HUMAN, RESERVE, ROBOT, SIZE_NAMES, Move, State, best_move, cell_name  # noqa: E402
import cap  # noqa: E402
from codex_llm import ask  # noqa: E402
from stackchan import StackChan  # noqa: E402

log = logging.getLogger("game")
WEB = ROOT / "server" / "web"
OUT = ROOT / "out"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------- hardware adapters
class Vision:
    """Top camera -> per-cell (player, size probabilities)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.cam = None
        if enabled:
            import topcam
            self.topcam = topcam
            self.cam = topcam.BoardCam.load()

    def read(self):
        f = self.topcam.grab()
        import cv2
        cv2.imwrite(str(OUT / "game_last_frame.jpg"), f)
        cells = self.topcam.read_cells(f, self.cam, OUT / "game_last_read.jpg")
        # camera color 0 = orange = ROBOT, 1 = blue = HUMAN
        return [None if c.color is None else (c.color, c.size_p) for c in cells]


def rank_moves(st: State, obs):
    """Score each legal move by how well its predicted top view explains the camera observation."""
    scored = []
    for mv in st.legal_moves():
        top = st.apply(mv).top_view()
        p = 1.0
        for pred, o in zip(top, obs):
            if pred is None and o is None:
                continue
            if pred is None or o is None or pred[0] != o[0]:
                p *= 0.002
            else:
                p *= max(0.02, o[1][pred[1]])
        scored.append((p, mv))
    scored.sort(key=lambda t: -t[0])
    return scored


class Arm:
    """Right arm pick & place + left arm stylus tap. Disabled -> the human is asked to move for the robot."""

    def __init__(self, enabled: bool, tap: bool):
        self.enabled, self.tap_enabled = enabled, tap
        self.slots = {}          # robot reserve slot -> size (None = empty)
        if enabled or tap:
            import gobblet_arm as G
            self.G = G
            self.gp = G.GobbletPlanner() if enabled else None
        self.reset_slots()

    def reset_slots(self):
        from_slots = {0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0}
        if self.enabled:
            from_slots = {i: s for i, (s, _, _) in self.G.RESERVE_SLOTS.items()}
        self.slots = dict(from_slots)

    def slot_for(self, size):
        return next((i for i, s in self.slots.items() if s == size), None)

    def free_slot_for(self, size):
        """Empty reserve slot whose home size matches (for tidying up)."""
        homes = {i: s for i, (s, _, _) in self.G.RESERVE_SLOTS.items()} if self.enabled else \
            {0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0}
        return next((i for i, s in homes.items() if s == size and self.slots.get(i) is None), None)

    BAD_CELLS = {0, 1}   # right arm cannot carry into / out of these (left-near and centre-near cells)

    def ensure_look(self):
        """Right arm to the camera-clear pose (swung out to the right) if it is not there yet."""
        G = self.G
        q = G.current_q("right")
        if abs(np.degrees(q[0]) - G.LOOK_PAN_DEG) < 5:
            return True
        segs, _ = G.plan_to_look(self.gp, q, G.current_q("left"))
        return segs is not None and G.run_segments("right", segs)

    def refresh_reserve(self, st: State, vision) -> bool:
        """Find the robot's off-board pieces with the top camera and use them as the reserve slots."""
        import topcam
        frame = topcam.grab()
        found = topcam.off_board_pieces(frame, vision.cam, color=0)
        counts = st.reserve(ROBOT)
        sized = topcam.assign_sizes(found, counts)
        if sized is None:
            log.warning("reserve: saw %d orange pieces off the board, expected %s", len(found), counts)
            # fall back to the most likely size per blob by rank among what we saw
            if not found:
                return False
            sized = [{**p, "size": min(2, i * 3 // max(1, len(found)))} for i, p in enumerate(found)]
        self.G.RESERVE_SLOTS = {i: (p["size"], p["x"], p["y"]) for i, p in enumerate(sized)}
        self.slots = {i: p["size"] for i, p in enumerate(sized)}
        log.info("reserve from camera: %s", [(p["size"], p["x"], p["y"]) for p in sized])
        return True

    def tuck_left(self):
        segs = self.G.tuck_left_segments()
        if segs:
            self.G.run_segments("left", segs)

    def transfer(self, st: State, src, dst, size, log_fn=log.info) -> bool:
        segs = self.plan(st, src, dst, size, log_fn)
        return segs is not None and self.G.run_segments("right", segs, log=log_fn)

    def plan(self, st: State, src, dst, size, log_fn=log.info):
        """src/dst: cell index or ("slot", i). Plans against the pieces in `st`. Returns segments or None."""
        G = self.G
        s_spot = G.Spot("slot", src[1]) if isinstance(src, tuple) else G.Spot("cell", src)
        d_spot = G.Spot("slot", dst[1]) if isinstance(dst, tuple) else G.Spot("cell", dst)
        used = {i for i, s in self.slots.items() if s is not None}
        stacks = [st.stack(c) for c in range(9)]
        obst = G.obstacles_from(stacks, used, s_spot, d_spot)
        segs, info = self.gp.plan_transfer(s_spot, d_spot, size, obst, G.home_q("right"), G.home_q("left"))
        if segs is None:
            log_fn(f"plan failed: {info}")
            return None
        log_fn(f"plan {s_spot}->{d_spot}: {info}")
        return segs

    def tap(self) -> bool:
        if not self.tap_enabled:
            return False
        G = self.G
        if not G.TAP_FILE.exists():
            log.warning("no tap pose taught (robot/gobblet_arm.py teach-tap)")
            return False
        return G.run_segments("left", G.tap_segments(G.home_q("left")))


# ---------------------------------------------------------------- talk
PERSONA = ("あなたはStackChan（スタックチャン）という小さな卓上ロボット。いま人間とゴブレットゴブラーズで対戦中。"
           "一人称は「ぼく」。明るく、ちょっと生意気で、負けず嫌いだけど優しい。子どもにも分かる言葉で話す。")

TEMPLATES = {
    "greet": ["よろしくね！ぼく、けっこう強いよ。", "勝負だ！手加減しないからね。"],
    "human_moved": ["なるほど、そうきたか。", "ふむふむ。", "いい手だね。でも…"],
    "robot_move": ["ここにしよう！", "これでどうだ！", "えいっ！"],
    "winning": ["ふふふ、見えてきたよ。", "もう勝ち筋が見えた！"],
    "win": ["やったー！ぼくの勝ち！", "勝った！もう一回やる？"],
    "lose": ["うわー、負けちゃった！強いね！", "くやしい！もう一回！"],
    "your_turn": ["きみの番だよ。", "どうぞ！"],
}


class Talker:
    def __init__(self, sc: StackChan, use_codex: bool):
        self.sc, self.use_codex = sc, use_codex
        self.lines = []          # recent things StackChan said (shown on the phone)

    def _emit(self, text, expression=None):
        self.lines = (self.lines + [text])[-6:]
        self.sc.say(text, expression=expression)

    def template(self, kind, expression=None):
        self._emit(random.choice(TEMPLATES[kind]), expression)

    def comment(self, kind, situation: str, expression=None, timeout=25.0):
        """Codex line for the situation, falling back to a template when slow or offline. Non-blocking."""
        if not self.use_codex:
            return self.template(kind, expression)

        def run():
            text = ask(f"{PERSONA}\n状況: {situation}\n"
                       "StackChanとして、声に出して言う一言だけを日本語で返して（30文字以内、記号や絵文字なし）。",
                       timeout=timeout)
            text = (text or "").strip().strip("「」\"")
            if not text or len(text) > 60:
                return self.template(kind, expression)
            self._emit(text, expression)

        threading.Thread(target=run, daemon=True).start()

    def chat(self, message: str, situation: str) -> str:
        text = ask(f"{PERSONA}\n対局の状況: {situation}\n相手がこう話しかけてきた: 「{message}」\n"
                   "StackChanとして返事を一言（40文字以内、記号や絵文字なし）。", timeout=45) if self.use_codex else None
        text = (text or "えへへ、なになに？").strip().strip("「」\"")
        self._emit(text)
        return text


# ---------------------------------------------------------------- game
class Game:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.sc = StackChan()
        self.talk = Talker(self.sc, args.codex == "on")
        self.vision = Vision(args.vision == "on")
        self.arm = Arm(args.arm == "on", args.tap == "on")
        self.token = secrets.token_urlsafe(6)
        self.url = f"http://{lan_ip()}:{args.port}/?t={self.token}"
        self.phase = "idle"          # idle qr lobby human reading confirm manual robot assist robot_tap over
        self.state = State()
        self.message = "スタックチャンの頭をなでてね"
        self.candidates = []         # for the confirm phase
        self.robot_move = None
        self.result = None
        self.mercy = False
        self.qr_until = 0.0
        self.touch_after = self.sc.touch_log(0).get("last", 0)
        self.version = 0
        self.busy = False

    # ---- helpers
    # LED per phase: blue = your turn, orange = robot's turn (the robot's pieces are orange)
    PHASE_LED = {
        "idle": ("pulse", (40, 40, 40)), "qr": ("pulse", (255, 255, 255)), "lobby": ("pulse", (0, 200, 60)),
        "human": ("pulse", (0, 60, 255)), "reading": ("chase", (0, 120, 255)), "confirm": ("blink", (0, 60, 255)),
        "manual": ("blink", (0, 60, 255)), "robot": ("chase", (255, 90, 0)), "robot_tap": ("chase", (255, 90, 0)),
        "assist": ("blink", (255, 200, 0)),
    }

    def set(self, phase, message=None):
        if phase != getattr(self, "phase", None) and phase in self.PHASE_LED:
            self.sc.led(*self.PHASE_LED[phase])
        with self.lock:
            self.phase = phase
            if message is not None:
                self.message = message
            self.version += 1
        log.info("phase=%s %s", phase, message or "")

    def situation(self) -> str:
        st = self.state
        cells = []
        for c in range(9):
            t = st.top(c)
            cells.append("・" if t is None else ("橙" if t[0] == ROBOT else "青") + SIZE_NAMES[t[1]])
        rows = [" ".join(cells[r * 3:r * 3 + 3]) for r in (2, 1, 0)]
        return (f"盤面（相手から見て奥→手前ではなく、相手側の行が先頭）: {' / '.join(rows)}。"
                f"ぼく(橙)の手持ち 大{st.reserve(ROBOT)[2]}中{st.reserve(ROBOT)[1]}小{st.reserve(ROBOT)[0]}、"
                f"相手(青)の手持ち 大{st.reserve(HUMAN)[2]}中{st.reserve(HUMAN)[1]}小{st.reserve(HUMAN)[0]}。"
                f"{len(st.history)}手目。")

    def snapshot(self):
        with self.lock:
            return {
                "phase": self.phase, "message": self.message, "board": self.state.to_dict(),
                "candidates": [{"move": m.to_dict(), "text": m.describe()} for m in self.candidates],
                "robotMove": None if self.robot_move is None else {**self.robot_move.to_dict(),
                                                                   "text": self.robot_move.describe()},
                "result": self.result, "lines": self.talk.lines[-3:], "version": self.version,
                "legal": [m.to_dict() for m in self.state.legal_moves(HUMAN)] if self.phase == "manual" else [],
                "config": {"arm": self.args.arm, "vision": self.args.vision, "tap": self.args.tap},
                "stackchanOnline": self._online_cached(),
            }

    _online = (0.0, False)

    def _online_cached(self):
        t, v = self._online
        if time.time() - t > 5:
            v = self.sc.online()
            self._online = (time.time(), v)
        return v

    def spawn(self, fn, *a):
        def run():
            try:
                fn(*a)
            except Exception:
                log.error("worker crashed:\n%s", traceback.format_exc())
                self.set("lobby", "ごめん、ちょっと調子が悪いみたい。もう一回はじめてね")
            finally:
                self.busy = False
        self.busy = True
        threading.Thread(target=run, daemon=True).start()

    # ---- head pat -> QR
    def poll_touch(self):
        while True:
            time.sleep(0.7)
            try:
                d = self.sc.touch_log(self.touch_after)
                evs = d.get("events", [])
                if evs:
                    self.touch_after = evs[-1]["id"]
                    self.on_pat()
                if self.phase == "qr" and time.time() > self.qr_until:
                    self.sc.hide_qr()
                    self.set("idle", "スタックチャンの頭をなでてね")
            except Exception:
                log.exception("touch poll")

    def on_pat(self):
        if self.phase in ("idle", "qr"):
            self.sc.show_qr(self.url, "スマホで\nよみとってね")
            self.qr_until = time.time() + 120
            self.set("qr", "QRコードを読みとってね")
            self.sc.say("ゴブレットで勝負しよう！スマホでQRコードを読みとってね。")
        else:
            self.sc.say(random.choice(["えへへ。", "くすぐったいよ。", "集中してるんだから！"]))

    # ---- phone actions
    def join(self):
        if self.phase in ("idle", "qr", "over"):
            self.sc.hide_qr()
            self.set("lobby", "先手と後手をえらんでね。スマホは机の決まった場所に置いてね")
            self.talk.template("greet", "happy")

    def start(self, human_first: bool, mercy: bool):
        self.state = State(turn=HUMAN if human_first else ROBOT)
        self.arm.reset_slots()
        self.result, self.robot_move, self.candidates = None, None, []
        self.mercy = mercy
        if human_first:
            self.human_turn()
        else:
            self.spawn(self.robot_turn)

    def human_turn(self):
        self.set("human", "きみの番！駒を置いたら「置いた！」をタッチ")
        self.sc.look(0, 5)

    def human_done(self):
        if self.phase != "human":
            return
        if not self.vision.enabled:
            self.set("manual", "どこに置いたか教えてね")
            return
        self.set("reading", "盤面を見てるよ…")
        self.spawn(self._read_human)

    def _read_human(self):
        self.sc.look(0, 25)
        obs = self.vision.read()
        ranked = rank_moves(self.state, obs)
        if not ranked or ranked[0][0] < 1e-4:
            self.set("manual", "うまく見えなかった…どこに置いたか教えてね")
            return
        best_p, best = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if not (best_p > 0.3 and best_p > 5 * second):     # no more "which one?" on the phone: take the best guess
            log.info("human move not certain (%.2f vs %.2f), taking %s", best_p, second, best)
        self.apply_human(best)

    def apply_human(self, mv: Move):
        if mv not in self.state.legal_moves(HUMAN) or self.state.turn != HUMAN:
            return
        self.candidates = []
        self.state = self.state.apply(mv)
        log.info("human: %s", mv)
        if self.finish_if_over():
            return
        self.talk.comment("human_moved", f"{self.situation()} 相手は今「{mv.describe()}」をした。それへの一言。")
        self.spawn(self.robot_turn)

    # (cell, size) the right arm can put down on / pick up from, checked with the board-touch map (2026-11-01)
    # cells each arm can put down on / pick up from. The right arm reaches every cell (checked 2026-11-01);
    # the left arm cannot get to the right column (cell 2 failed in a game).
    REACH = {"right": frozenset(range(9)), "left": frozenset(range(9))}     # both arms reach every cell
    desk_count = None            # {"right": [S, M, L], "left": [S, M, L]} orange pieces beside the board

    def count_desk(self):
        import move_test as MT
        import sam_pieces
        import topcam
        try:
            frame = topcam.grab(warmup=30)
            cam = topcam.BoardCam.load()
            self.desk_count = {}
            for a in MT.available_arms():
                ps = sam_pieces.detect(frame, cam, 0, region=MT.DESK_REGION[a])
                self.desk_count[a] = [sum(1 for p in ps if p["size"] == s_) for s_ in range(3)]
            log.info("orange pieces beside the board: %s", self.desk_count)
        except Exception:
            log.exception("desk count failed")
            self.desk_count = None

    def arm_for(self, mv: Move):
        """The arm that can do this move (None if neither): it must reach the cells and, for a reserve piece,
        have one of that size on its side. Prefers the arm on the side of the column."""
        import move_test as MT
        arms = MT.available_arms()
        col = (mv.frm if mv.frm != RESERVE and mv.frm % 3 != 1 else mv.to) % 3
        arms.sort(key=lambda a: 0 if (a == "left") == (col == 0) else 1)
        for a in arms:
            if mv.to not in self.REACH[a] or (mv.frm != RESERVE and mv.frm not in self.REACH[a]):
                continue
            if mv.frm == RESERVE and self.desk_count is not None and self.desk_count.get(a, [0, 0, 0])[mv.size] == 0:
                continue
            return a
        return None

    def arm_can(self, mv: Move) -> bool:
        return self.arm_for(mv) is not None

    def sync_board_with_camera(self):
        """Before the robot thinks: make the game state match what the top camera sees. Cells that agree keep
        their full stacks (hidden pieces included); cells that disagree are rebuilt from the camera."""
        try:
            obs = self.vision.read()
            obs2 = None
        except Exception:
            log.exception("board check failed")
            return
        st = self.state
        top = st.top_view()
        fixed = []
        m = list(st.m)
        for c in range(9):
            o = obs[c]
            seen = None if o is None else (o[0], int(max(range(3), key=lambda s_: o[1][s_])))
            same = (top[c] is None and seen is None) or (top[c] is not None and seen is not None and top[c][0] == seen[0])
            if same:
                continue
            # one bad frame (a hand, an arm, glare) must not delete a piece: the second photo has to agree
            if obs2 is None:
                time.sleep(0.8)
                try:
                    obs2 = self.vision.read()
                except Exception:
                    obs2 = [None] * 9
            o2 = obs2[c]
            seen2 = None if o2 is None else (o2[0], int(max(range(3), key=lambda s_: o2[1][s_])))
            if (seen is None) != (seen2 is None) or (seen is not None and seen2[0] != seen[0]):
                log.info("board check: cell %d differs only in one photo (%s / %s) - kept", c, seen, seen2)
                continue
            for i in range(6):
                m[i] &= ~(1 << c)
            if seen is not None and bin(m[seen[0] * 3 + seen[1]]).count("1") < 2:
                m[seen[0] * 3 + seen[1]] |= 1 << c
            fixed.append((c, top[c], seen))
        if fixed:
            # the game record wins (user, 2026-11-01): a missed or misread piece in the photo must not change it
            log.warning("board differs from the camera (record kept, not changed): %s", fixed)

    def robot_turn(self):
        if self.vision.enabled:
            self.set("robot", "盤面を確認中…")
            self.sync_board_with_camera()
            if self.finish_if_over():
                return
        self.set("robot", "ぼくの番…考え中")
        self.sc.expression("neutral")
        r = best_move(self.state, time_ms=self.args.think_ms)
        order = [r.move]
        if self.mercy and r.forced_win_in is None and random.random() < 0.35:
            order = [random.choice(self.state.legal_moves(ROBOT))]
        if self.arm.enabled:
            from game.rules import score_moves
            self.count_desk()
            ranked = [m for m, _ in score_moves(self.state, depth=4)]
            order = [m for m in order + ranked if self.arm_can(m)] or order
        mv = order[0]
        self.robot_move = mv
        log.info("robot best: %s score=%s depth=%s win_in=%s -> playing %s", r.move, r.score, r.depth,
                 r.forced_win_in, mv)
        hint = f"あと{(r.forced_win_in + 1) // 2}手で勝てると読んでいる。" if (r.forced_win_in and mv == r.move) else ""
        self.talk.comment("winning" if hint else "robot_move",
                          f"{self.situation()} ぼくはこれから「{mv.describe()}」をする。{hint}",
                          "happy" if hint else None)
        if not self.arm.enabled:
            self.set("assist", f"ぼくの代わりに、オレンジの{mv.describe()}してね")
            return
        self.set("robot", f"ぼくの番：{mv.describe()}")
        import move_test as MT
        src = ("reserve", mv.size) if mv.frm == RESERVE else ("cell", mv.frm)
        try:
            arm = self.arm_for(mv) or MT.choose_arm(src, mv.to, mv.size, log=log.info)
            log.info("arm for %s: %s", mv, arm)
            self.set("robot", f"ぼくの番：{mv.describe()}（{'左手' if arm == 'left' else '右手'}）")
            ok = MT.move(src, mv.to, mv.size, log=log.info, arm=arm)
        except Exception:
            log.error("arm move crashed:\n%s", traceback.format_exc())
            ok = False
        if ok and self.vision.enabled:
            ok = self.verify_robot_move(mv)
        if not ok:
            self.set("assist", f"ごめん、うまくできなかった。オレンジの{mv.describe()}にしてくれる？")
            return
        self.after_robot_move()

    def plan_robot_move(self, mv: Move):
        if mv.frm == RESERVE and self.vision.enabled and not getattr(self, "_reserve_fresh", False):
            self.arm.refresh_reserve(self.state, self.vision)
            self._reserve_fresh = True
        if mv.frm == RESERVE:
            slot = self.arm.slot_for(mv.size)
            if slot is None:
                return None
            self._planned_slot = slot
            return self.arm.plan(self.state, ("slot", slot), mv.to, mv.size)
        return self.arm.plan(self.state, mv.frm, mv.to, mv.size)

    def G_run(self, segs) -> bool:
        return self.arm.G.run_segments("right", segs, log=log.info)

    def execute_robot_move(self, mv: Move) -> bool:
        if mv.frm == RESERVE:
            slot = self.arm.slot_for(mv.size)
            if slot is None:
                return False
            ok = self.arm.transfer(self.state, ("slot", slot), mv.to, mv.size)
            if ok:
                self.arm.slots[slot] = None
        else:
            ok = self.arm.transfer(self.state, mv.frm, mv.to, mv.size)
        if ok and self.vision.enabled:
            ok = self.verify_robot_move(mv)
        return ok

    def verify_robot_move(self, mv: Move) -> bool:
        expect = self.state.apply(mv).top_view()
        obs = self.vision.read()
        bad = [c for c in range(9) if (expect[c] is None) != (obs[c] is None) or
               (expect[c] and obs[c] and expect[c][0] != obs[c][0])]
        if not bad:
            return True
        # only the cells the robot touched decide success; a misread elsewhere (e.g. the human's piece at
        # cell 8 on 2026-11-02 after a good move 4 -> 2) must not turn a good move into "うまくできなかった"
        touched = {mv.to} | ({mv.frm} if mv.frm != RESERVE else set())
        desc = lambda v: None if v is None else ("orange" if v[0] == 0 else "blue")
        log.info("verify: expect %s / camera %s", [desc(e) for e in expect], [desc(o) for o in obs])
        if not touched & set(bad):
            log.warning("verify: cells %s look different, but the moved cells %s are right - move OK",
                        bad, sorted(touched))
            return True
        log.warning("robot move verification failed at cells %s", bad)
        return False

    def recover(self, mv, bad, obs) -> bool:
        """Code-as-Policies recovery: Codex writes a short fix-up program against the arm primitives."""
        self.sc.say("あれ、ちょっとずれちゃったかも。直すね。", expression="surprised")
        ctx = {"intended_move": mv.to_dict(), "cells_that_look_wrong": bad,
               "camera": [None if o is None else {"color": "orange" if o[0] == 0 else "blue",
                                                  "size": int(max(range(3), key=lambda s: o[1][s]))} for o in obs],
               "board_before_move": self.state.to_dict(), "robot_reserve_slots": self.arm.slots}
        code = cap.generate("Make the board match board_before_move with the intended move applied, "
                            "using as few move_piece calls as possible.", ctx)
        if not code:
            return False
        log.info("CaP recovery program:\n%s", code)
        res = cap.run(code, self.cap_api())
        log.info("CaP result: %s", res)
        return res["ok"] and self.vision.read() is not None

    def cap_api(self):
        def move_piece(src, dst, size):
            s = ("slot", int(str(src)[1:])) if str(src).upper().startswith("R") else int(src)
            d = ("slot", int(str(dst)[1:])) if str(dst).upper().startswith("R") else int(dst)
            return self.arm.transfer(self.state, s, d, int(size))
        return {
            "board": lambda: {**self.state.to_dict(), "robot_reserve_slots": self.arm.slots},
            "read_board": lambda: [None if o is None else {"color": "orange" if o[0] == 0 else "blue",
                                                           "size": int(max(range(3), key=lambda s: o[1][s]))}
                                   for o in self.vision.read()],
            "move_piece": move_piece,
            "say": lambda t: self.sc.say(str(t)[:80], wait=True),
            "wait": lambda s: time.sleep(min(10.0, float(s))),
        }

    def assist_done(self):
        if self.phase == "assist":
            if self.robot_move.frm == RESERVE:
                slot = self.arm.slot_for(self.robot_move.size)
                if slot is not None:
                    self.arm.slots[slot] = None
            self.spawn(self.after_robot_move)

    def after_robot_move(self):
        self.state = self.state.apply(self.robot_move)
        if self.finish_if_over():
            return
        self.set("robot_tap", "ぼくの番おわり！")
        if self.arm.tap_enabled:
            self.arm.tap()
            deadline = time.time() + 6
            while time.time() < deadline and self.phase == "robot_tap":
                time.sleep(0.2)
        if self.phase == "robot_tap":   # nobody tapped (stylus missed or tap disabled): move on anyway
            self.robot_tapped()

    def robot_tapped(self):
        if self.phase == "robot_tap":
            self.talk.template("your_turn")
            self.human_turn()

    def finish_if_over(self) -> bool:
        w = self.state.winner()
        if w is None and len(self.state.history) >= 80:
            self.result = "draw"
            self.set("over", "引き分け！")
            self.sc.led("pulse", (160, 0, 255))
            self.talk.comment("lose", f"{self.situation()} 長い勝負で引き分けになった。")
            return True
        if w is None:
            return False
        self.result = "robot" if w == ROBOT else "human"
        if w == ROBOT:
            self.set("over", "ぼくの勝ち！")
            self.sc.led("rainbow", ms=15000)
            self.sc.dance(6000)
            self.talk.comment("win", f"{self.situation()} ぼくが勝った。", "happy")
        else:
            self.set("over", "きみの勝ち！")
            self.sc.led("blink", (0, 60, 255), ms=4000)
            self.talk.comment("lose", f"{self.situation()} ぼくが負けた。", "sad")
        return True

    def tidy(self):
        """After a game: put the robot's orange pieces back into their reserve slots (CaP, arm only)."""
        if not self.arm.enabled:
            self.set("lobby", "駒を片付けて、もう一回えらんでね")
            return
        self.set("robot", "おかたづけ中…")
        ctx = {"board": self.state.to_dict(), "robot_reserve_slots": self.arm.slots,
               "slot_sizes": {i: s for i, (s, _, _) in self.arm.G.RESERVE_SLOTS.items()}}
        code = cap.generate("Put every orange piece that is visible on the board back into an empty reserve slot "
                            "whose slot_sizes entry equals the piece size. Move top pieces first. "
                            "Blue pieces are the human's: leave them.", ctx)
        res = cap.run(code, self.cap_api()) if code else {"ok": False, "error": "no program"}
        log.info("tidy: %s", res)
        self.set("lobby", "片付けたよ！青い駒をもどして、もう一回えらんでね" if res["ok"] else
                 "ごめん、全部は片付けられなかった。手で戻してね")


# ---------------------------------------------------------------- HTTP
def make_handler(game: Game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authed(self):
            q = parse_qs(urlparse(self.path).query)
            return q.get("t", [""])[0] == game.token or self.client_address[0] == "127.0.0.1"

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                if not self._authed():
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write("StackChanの頭をなでて、QRコードから開いてね".encode())
                    return
                b = (WEB / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/state":
                self._json(game.snapshot() if self._authed() else {"error": "token"}, 200 if self._authed() else 403)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self._authed():
                return self._json({"error": "token"}, 403)
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            p = urlparse(self.path).path
            g = game
            if p == "/api/join":
                g.join()
            elif p == "/api/start":
                g.start(bool(body.get("humanFirst", True)), bool(body.get("mercy", False)))
            elif p == "/api/human-done":
                g.human_done()
            elif p == "/api/human-move":       # confirm / manual entry
                mv = Move(int(body["from"]), int(body["to"]), int(body["size"]))
                if g.phase in ("confirm", "manual"):
                    g.apply_human(mv)
            elif p == "/api/manual":
                if g.phase == "confirm":
                    g.set("manual", "どこに置いたか教えてね")
            elif p == "/api/assist-done":
                g.assist_done()
            elif p == "/api/robot-tap":
                g.robot_tapped()
            elif p == "/api/tidy":
                if g.phase == "over":
                    g.spawn(g.tidy)
            elif p == "/api/again":
                g.set("lobby", "先手と後手をえらんでね")
            elif p == "/api/chat":
                msg = str(body.get("text", ""))[:200]
                threading.Thread(target=g.talk.chat, args=(msg, g.situation()), daemon=True).start()
            elif p == "/api/pat":              # testing without StackChan
                g.on_pat()
            else:
                return self._json({"error": "unknown"}, 404)
            self._json(g.snapshot())

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--arm", choices=["on", "off"], default="off")
    ap.add_argument("--tap", choices=["on", "off"], default="off")
    ap.add_argument("--vision", choices=["on", "off"], default="off")
    ap.add_argument("--codex", choices=["on", "off"], default="on")
    ap.add_argument("--cap", choices=["on", "off"], default="on")
    ap.add_argument("--think-ms", type=int, default=3000)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "game.log")])
    game = Game(a)
    threading.Thread(target=game.poll_touch, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(game))
    log.info("Gobblet server on %s  (arm=%s tap=%s vision=%s codex=%s)", game.url, a.arm, a.tap, a.vision, a.codex)
    print(f"PHONE URL: {game.url}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
