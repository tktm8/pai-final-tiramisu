// Gobblet Gobblers search: negamax + alpha-beta + transposition table, iterative deepening.
// Build: clang -O3 -shared -fPIC -o libsolver.dylib solver.c
//
// Position = 6 nine-bit masks: m[p*3+s] = cells holding player p's piece of size s (0 small, 1 medium, 2 large).
// A cell holds at most one piece per size, stacked by size, so the masks fully describe hidden pieces too.
// Move = (from, to, size): from 0..8 is a board cell, from 9 means "from the reserve".
// Rule: after a move, if the opponent shows three in a row the mover loses (covers the "revealed line" rule,
// including blocking it with the lifted piece). Otherwise the mover wins with its own line.

#include <stdint.h>
#include <string.h>
#include <time.h>

#define WIN 10000
#define INF 30000
#define RESERVE 9

static const uint16_t LINES[8] = {0007, 0070, 0700, 0111, 0222, 0444, 0421, 0124};

typedef struct { uint16_t m[6]; } Pos;
typedef struct { int8_t from, to, size; } Move;

static inline int pc(unsigned x) { return __builtin_popcount(x); }

static inline uint16_t occ_size(const Pos *p, int s) { return p->m[s] | p->m[3 + s]; }

// cells whose top piece belongs to player pl
static inline uint16_t visible(const Pos *p, int pl) {
    uint16_t l = occ_size(p, 2), m = occ_size(p, 1);
    return (p->m[pl * 3 + 2]) | (p->m[pl * 3 + 1] & ~l) | (p->m[pl * 3 + 0] & ~l & ~m);
}

static inline int has_line(uint16_t v) {
    for (int i = 0; i < 8; i++) if ((v & LINES[i]) == LINES[i]) return 1;
    return 0;
}

// cells where a piece of size s can go (empty or top piece smaller)
static inline uint16_t can_land(const Pos *p, int s) {
    uint16_t bigger_or_equal = 0;
    for (int t = s; t < 3; t++) bigger_or_equal |= occ_size(p, t);
    return (uint16_t)(~bigger_or_equal & 0777);
}

static int gen_moves(const Pos *p, int pl, Move *out) {
    int n = 0;
    uint16_t vis = visible(p, pl);
    for (int s = 2; s >= 0; s--) {
        uint16_t land = can_land(p, s);
        if (pc(p->m[pl * 3 + s]) < 2) {
            for (int c = 0; c < 9; c++) if (land >> c & 1) out[n++] = (Move){RESERVE, c, s};
        }
        uint16_t mine = p->m[pl * 3 + s] & vis;   // my visible pieces of size s
        for (int f = 0; f < 9; f++) {
            if (!(mine >> f & 1)) continue;
            for (int c = 0; c < 9; c++) if (c != f && (land >> c & 1)) out[n++] = (Move){f, c, s};
        }
    }
    return n;
}

static inline void apply(Pos *p, int pl, Move mv) {
    if (mv.from != RESERVE) p->m[pl * 3 + mv.size] &= ~(1u << mv.from);
    p->m[pl * 3 + mv.size] |= 1u << mv.to;
}

// result of the move for the mover: +1 win, -1 loss, 0 nothing
static inline int outcome(const Pos *p, int pl) {
    if (has_line(visible(p, 1 - pl))) return -1;
    if (has_line(visible(p, pl))) return 1;
    return 0;
}

// ---------------- evaluation ----------------
static int eval_side(const Pos *p, int pl) {
    uint16_t me = visible(p, pl), op = visible(p, 1 - pl);
    int sc = 0;
    for (int i = 0; i < 8; i++) {
        int a = pc(me & LINES[i]), b = pc(op & LINES[i]);
        if (b == 0) sc += a == 2 ? 30 : a == 1 ? 4 : 0;
    }
    // large pieces in reserve are flexible (they can gobble anything)
    sc += (2 - pc(p->m[pl * 3 + 2])) * 6;
    if (me & 0020) sc += 6;   // centre
    return sc;
}

static int evaluate(const Pos *p, int pl) { return eval_side(p, pl) - eval_side(p, 1 - pl); }

// ---------------- transposition table ----------------
#define TT_BITS 22
#define TT_SIZE (1u << TT_BITS)
typedef struct { uint64_t key; int16_t score; int8_t depth, flag; Move best; } TTE;
static TTE tt[TT_SIZE];

static inline uint64_t key_of(const Pos *p, int pl) {
    uint64_t k = pl;
    for (int i = 0; i < 6; i++) k = (k << 9) | p->m[i];
    return k * 0x9E3779B97F4A7C15ull ^ (k >> 17);
}

static long nodes;
static clock_t deadline;
static int timed_out;

static int negamax(Pos *p, int pl, int depth, int ply, int alpha, int beta, Move *best_out) {
    if ((++nodes & 4095) == 0 && clock() > deadline) timed_out = 1;
    if (timed_out) return 0;
    uint64_t key = key_of(p, pl);
    TTE *e = &tt[key & (TT_SIZE - 1)];
    Move tt_move = {-1, -1, -1};
    if (e->key == key) {
        tt_move = e->best;
        if (e->depth >= depth && ply > 0) {
            int s = e->score;
            if (s > WIN - 200) s -= ply; else if (s < -WIN + 200) s += ply;
            if (e->flag == 0) return s;
            if (e->flag == 1 && s >= beta) return s;
            if (e->flag == 2 && s <= alpha) return s;
        }
    }
    if (depth == 0) return evaluate(p, pl);

    Move moves[160];
    int n = gen_moves(p, pl, moves);
    if (n == 0) return 0;
    // ordering: TT move, immediate wins, then large pieces first (generation order)
    int scores[160];
    for (int i = 0; i < n; i++) {
        Pos q = *p; apply(&q, pl, moves[i]);
        int o = outcome(&q, pl);
        scores[i] = o * 100000 + (moves[i].from == RESERVE ? 0 : -1) + moves[i].size * 3;
        if (tt_move.from == moves[i].from && tt_move.to == moves[i].to && tt_move.size == moves[i].size) scores[i] += 1000000;
    }
    int a0 = alpha, best = -INF;
    Move bm = moves[0];
    for (int k = 0; k < n; k++) {
        int bi = k;
        for (int j = k + 1; j < n; j++) if (scores[j] > scores[bi]) bi = j;
        Move mv = moves[bi]; moves[bi] = moves[k]; moves[k] = mv;
        int ts = scores[bi]; scores[bi] = scores[k]; scores[k] = ts;
        Pos q = *p; apply(&q, pl, mv);
        int o = outcome(&q, pl), v;
        if (o == 1) v = WIN - ply - 1;
        else if (o == -1) v = -(WIN - ply - 1);
        else v = -negamax(&q, 1 - pl, depth - 1, ply + 1, -beta, -alpha, NULL);
        if (timed_out) return 0;
        if (v > best) { best = v; bm = mv; }
        if (v > alpha) alpha = v;
        if (alpha >= beta) break;
    }
    int st = best;
    if (st > WIN - 200) st += ply; else if (st < -WIN + 200) st -= ply;
    if (e->key != key || e->depth <= depth) {
        e->key = key; e->depth = depth; e->score = (int16_t)st; e->best = bm;
        e->flag = best <= a0 ? 2 : best >= beta ? 1 : 0;
    }
    if (best_out) *best_out = bm;
    return best;
}

// Public API. masks: 6 x uint16. Returns 0 if a move was found.
// out: from, to, size, score, completed depth, nodes(lower 31 bits)
int gg_search(const uint16_t *masks, int side, int time_ms, int max_depth, int *out) {
    Pos p; memcpy(p.m, masks, sizeof p.m);
    Move moves[160];
    if (gen_moves(&p, side, moves) == 0) return 1;
    nodes = 0; timed_out = 0;
    deadline = clock() + (clock_t)((double)time_ms * CLOCKS_PER_SEC / 1000.0);
    Move best = moves[0];
    int best_score = 0, done = 0;
    for (int d = 1; d <= max_depth; d++) {
        Move m;
        int s = negamax(&p, side, d, 0, -INF, INF, &m);
        if (timed_out) break;
        best = m; best_score = s; done = d;
        if (s > WIN - 200 || s < -WIN + 200) break;   // proven result
    }
    out[0] = best.from; out[1] = best.to; out[2] = best.size;
    out[3] = best_score; out[4] = done; out[5] = (int)(nodes & 0x7fffffff);
    return 0;
}

// Score every legal move at a fixed depth (used for hints / commentary). Returns the move count.
int gg_score_moves(const uint16_t *masks, int side, int depth, int *out, int max_out) {
    Pos p; memcpy(p.m, masks, sizeof p.m);
    Move moves[160];
    int n = gen_moves(&p, side, moves);
    timed_out = 0; deadline = clock() + 60 * CLOCKS_PER_SEC;
    int k = 0;
    for (int i = 0; i < n && k < max_out; i++) {
        Pos q = p; apply(&q, side, moves[i]);
        int o = outcome(&q, side), v;
        if (o == 1) v = WIN - 1;
        else if (o == -1) v = -(WIN - 1);
        else v = depth > 1 ? -negamax(&q, 1 - side, depth - 1, 1, -INF, INF, NULL) : -evaluate(&q, 1 - side);
        out[k * 4 + 0] = moves[i].from; out[k * 4 + 1] = moves[i].to; out[k * 4 + 2] = moves[i].size; out[k * 4 + 3] = v;
        k++;
    }
    return k;
}

void gg_clear(void) { memset(tt, 0, sizeof tt); }
