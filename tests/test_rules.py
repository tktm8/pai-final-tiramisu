import random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game.rules import State, best_move, score_moves, infer_move, ROBOT, HUMAN, Move, RESERVE


def test_c_matches_python_moves():
    rng = random.Random(1)
    for _ in range(200):
        st = State(turn=rng.randint(0, 1))
        for _ in range(rng.randint(0, 12)):
            mv = st.legal_moves()
            if not mv: break
            st = st.apply(rng.choice(mv))
            if st.winner() is not None: break
        if st.winner() is not None: continue
        py = {(m.frm, m.to, m.size) for m in st.legal_moves()}
        c = {(m.frm, m.to, m.size) for m, _ in score_moves(st, 1)}
        assert py == c, (st.m, py ^ c)


def test_reveal_rule():
    # human line on top row under a robot large; robot lifts its large away -> reveals human line -> robot loses
    st = State(turn=ROBOT)
    st.m[HUMAN * 3 + 0] = 0b000000111            # human smalls... only 2 allowed, use medium too
    st.m[HUMAN * 3 + 0] = 0b000000011
    st.m[HUMAN * 3 + 1] = 0b000000100
    st.m[ROBOT * 3 + 2] = 0b000000001            # robot large covers cell 0
    after = st.apply(Move(0, 8, 2))
    assert after.winner() == HUMAN


def play(robot_first, rng, ms=300):
    st = State(turn=ROBOT if robot_first else HUMAN)
    for ply in range(60):
        if st.turn == ROBOT:
            mv = best_move(st, ms).move
        else:
            mv = rng.choice(st.legal_moves())
        st = st.apply(mv)
        w = st.winner()
        if w is not None:
            return w, ply
    return None, 60


def test_beats_random():
    rng = random.Random(0)
    res = [play(i % 2 == 0, rng) for i in range(20)]
    assert all(w == ROBOT for w, _ in res), res


def test_infer():
    st = State(turn=HUMAN)
    st = st.apply(Move(RESERVE, 4, 1))           # human medium centre
    st = st.apply(Move(RESERVE, 0, 0))           # robot small corner
    obs = st.top_view(); obs[0] = (HUMAN, 2)     # human gobbles corner with a large from reserve
    hits, changed = infer_move(st, obs)
    assert hits == [Move(RESERVE, 0, 2)] and changed == [0]
    obs2 = st.top_view(); obs2[4] = None; obs2[0] = (HUMAN, 1)   # human moves centre medium onto the corner small
    hits, _ = infer_move(st, obs2)
    assert hits == [Move(4, 0, 1)], hits


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"ok {name} {time.time()-t:.1f}s")
    st = State(turn=ROBOT)
    for ms in (500, 2000, 5000):
        r = best_move(st, ms)
        print(f"opening {ms}ms: {r.move} score={r.score} depth={r.depth} nodes={r.nodes} win_in={r.forced_win_in}")
