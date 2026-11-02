"""Top (pole) camera: board calibration and cell reading for Gobblet Gobblers.

Calibration (`calibrate`, board empty): finds the two orange bars (world X 243.4 / 173.4) and the two blue
bars (world Y +35 / -35) and stores the plane homography world(X, Y) -> pixel, plus a mask of the bars
themselves so they are never mistaken for pieces.

Reading: orange / blue blobs are mapped onto the board plane. Each blob goes to the nearest cell centre
(the camera looks almost straight down, so a piece top lands well inside its cell), and its size is judged
from the blob's equivalent diameter on the board plane. Sizes come with a probability so the game can
ask the player when the picture is ambiguous instead of guessing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CALIB = OUT / "topcam_calib.json"
BAR_MASK = OUT / "topcam_bars.png"
SCRIPTS = Path.home() / "Programing/so-101/scripts"
TOP_CAMERA_ID = "0x31300000bda025a"     # GOPPA FHD Webcam on the pole

# Board (world mm). Cells: row 0..2 = X 138.4 / 208.4 / 278.4 (robot -> human), col 0..2 = Y +70 / 0 / -70.
CELL_X = [138.4, 208.4, 278.4]
CELL_Y = [70.0, 0.0, -70.0]
BAR_X = [243.4, 173.4]          # orange bars, upper one in the image first
BAR_Y = [35.0, -35.0]           # blue bars, left one in the image first
CELL_RADIUS = 34.0              # a blob centre farther than this from every cell centre is off-board

# Equivalent blob diameter (mm on the board plane) per size, measured 2026-10-31 from the top camera.
# Glossy highlights eat into the blobs, so these are smaller than the real pieces. Refine with `learn`.
SIZE_D = {0: 17.0, 1: 25.0, 2: 44.0}
SIZE_SIGMA = {0: 3.5, 1: 4.0, 2: 7.0}

ORANGE = ((3, 110, 120), (22, 255, 255))
BLUE = ((88, 60, 110), (112, 255, 255))
COLOR_NAMES = {0: "orange", 1: "blue"}


def camera_index(unique_id: str = TOP_CAMERA_ID) -> int:
    r = subprocess.run([str(SCRIPTS / "camera_index.sh"), unique_id], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return int(r.stdout.strip())


def grab(width=1920, height=1080, warmup=15) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index(), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("top camera gave no frame")
    return frame


# ---------------------------------------------------------------- calibration
def _board_roi(hsv):
    H, S, V = cv2.split(hsv)
    bright = ((V > 225) & (S < 100)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    i = 1 + int(np.argmax(st[1:, 4]))
    roi = (lab == i).astype(np.uint8) * 255
    hull = cv2.convexHull(cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    roi = np.zeros_like(roi)
    cv2.fillPoly(roi, [hull], 255)
    return roi


def _bar_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    roi = _board_roi(hsv)
    org = (((H >= 4) & (H <= 22) & (S > 55) & (V > 200)).astype(np.uint8) * 255) & roi
    blu = (((H >= 88) & (H <= 112) & (S > 35) & (V > 180)).astype(np.uint8) * 255) & roi
    return org, blu


def _two_lines(mask, axis):
    """Two roughly parallel lines. axis=0: horizontal lines (split by y), 1: vertical (split by x)."""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(np.float32)
    key = pts[:, 1] if axis == 0 else pts[:, 0]
    _, lab, centres = cv2.kmeans(key.reshape(-1, 1), 2, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    lines = []
    for k in np.argsort(centres.ravel()):
        p = pts[lab.ravel() == k]
        for _ in range(3):
            vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            d = np.abs((p[:, 0] - x0) * vy - (p[:, 1] - y0) * vx)
            p = p[d < max(4.0, np.percentile(d, 90))]
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    return lines


def _intersect(a, b):
    t = np.linalg.solve(np.c_[a[1], -b[1]], b[0] - a[0])
    return a[0] + a[1] * t[0]


@dataclass
class BoardCam:
    H: np.ndarray                  # world (X, Y) on the bar plane -> pixel
    size: tuple

    def to_px(self, xy):
        p = cv2.perspectiveTransform(np.asarray(xy, np.float64).reshape(-1, 1, 2), self.H)
        return p.reshape(-1, 2)

    def to_world(self, px):
        p = cv2.perspectiveTransform(np.asarray(px, np.float64).reshape(-1, 1, 2), np.linalg.inv(self.H))
        return p.reshape(-1, 2)

    def save(self, extra=None):
        d = dict(H=self.H.tolist(), size=list(self.size), saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        d.update(extra or {})
        CALIB.write_text(json.dumps(d, indent=1))

    @staticmethod
    def load() -> "BoardCam":
        d = json.loads(CALIB.read_text())
        return BoardCam(np.array(d["H"]), tuple(d["size"]))


def calibrate(frame, log=print) -> BoardCam:
    """Board must be empty (pieces may stand around it)."""
    org, blu = _bar_masks(frame)
    oh = _two_lines(org, 0)
    bv = _two_lines(blu, 1)
    img, wld = [], []
    for i, o in enumerate(oh):
        for j, b in enumerate(bv):
            img.append(_intersect(o, b))
            wld.append((BAR_X[i], BAR_Y[j]))
    H, _ = cv2.findHomography(np.array(wld), np.array(img), 0)
    cam = BoardCam(H, frame.shape[1::-1])
    # spacing sanity: one cell should be 60..200 px wide
    cell_px = np.linalg.norm(cam.to_px([(208.4, 35)]) - cam.to_px([(208.4, -35)]))
    if not 60 < cell_px < 200:
        raise RuntimeError(f"calibration looks wrong: one cell = {cell_px:.0f}px")
    cv2.imwrite(str(BAR_MASK), draw_bar_mask(cam, frame.shape[:2]))
    cam.save({"cell_px": float(cell_px)})
    log(f"[calib] ok, one cell = {cell_px:.0f}px, saved {CALIB.name}")
    return cam


def draw_bar_mask(cam: BoardCam, shape, width_px=16):
    """Mask of the four frame bars, drawn from their known world positions (not from colour, so pieces standing
    next to the board can never leak into it)."""
    m = np.zeros(shape, np.uint8)
    for x in (173.4, 243.4):
        a, b = cam.to_px([(x, 110.0), (x, -110.0)]).astype(int)
        cv2.line(m, tuple(a), tuple(b), 255, width_px)
    for y in BAR_Y:
        a, b = cam.to_px([(100.0, y), (318.0, y)]).astype(int)
        cv2.line(m, tuple(a), tuple(b), 255, width_px)
    return m


# ---------------------------------------------------------------- reading
def color_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    o = cv2.inRange(hsv, *ORANGE)
    b = cv2.inRange(hsv, *BLUE)
    if BAR_MASK.exists():
        bars = cv2.imread(str(BAR_MASK), cv2.IMREAD_GRAYSCALE)
        if bars is not None and bars.shape == o.shape:
            b &= ~bars
            o &= ~bars
    k = np.ones((5, 5), np.uint8)
    o = cv2.morphologyEx(cv2.morphologyEx(o, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return o, b


@dataclass
class Blob:
    color: int            # 0 orange, 1 blue
    x: float              # world mm on the board plane
    y: float
    d: float              # equivalent diameter mm
    px: tuple


@dataclass
class CellReading:
    color: int | None             # 0 orange, 1 blue, None empty
    size_p: list | None           # probability of size 0/1/2
    d: float | None

    @property
    def size(self):
        return None if self.size_p is None else int(np.argmax(self.size_p))

    @property
    def sure(self):
        return self.size_p is None or max(self.size_p) > 0.8


def blobs(frame, cam: BoardCam, min_px=250) -> list[Blob]:
    out = []
    for color, m in enumerate(color_masks(frame)):
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            if st[i, 4] < min_px:
                continue
            w, h = st[i, 2], st[i, 3]
            if max(w, h) > 4 * min(w, h):          # thin strips: bar remnants, cables
                continue
            cnt = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
            wc = cam.to_world(cnt.reshape(-1, 2).astype(np.float64))
            area = abs(cv2.contourArea(wc.astype(np.float32)))
            cx, cy = cam.to_world([cen[i]])[0]
            out.append(Blob(color, float(cx), float(cy), float(2 * np.sqrt(area / np.pi)), (int(cen[i][0]), int(cen[i][1]))))
    return out


def size_probs(d):
    lik = np.array([np.exp(-0.5 * ((d - SIZE_D[s]) / SIZE_SIGMA[s]) ** 2) for s in range(3)])
    # beyond the large mean everything is large; below the small mean everything is small
    if d > SIZE_D[2]:
        lik[2] = 1.0
    if d < SIZE_D[0]:
        lik[0] = 1.0
    lik += 1e-6
    return (lik / lik.sum()).round(3).tolist()


def read_cells(frame, cam: BoardCam, debug_path: Path | None = None) -> list[CellReading]:
    bl = blobs(frame, cam)
    cells: list[CellReading] = []
    for c in range(9):
        r, k = divmod(c, 3)
        cand = [b for b in bl if np.hypot(b.x - CELL_X[r], b.y - CELL_Y[k]) < CELL_RADIUS and b.d > 10]
        if not cand:
            cells.append(CellReading(None, None, None))
            continue
        b = max(cand, key=lambda b: b.d)
        cells.append(CellReading(b.color, size_probs(b.d), round(b.d, 1)))
    if debug_path is not None:
        vis = frame.copy()
        for b in bl:
            cv2.putText(vis, f"{'OB'[b.color]}{b.d:.0f}", b.px, 0, 0.7, (0, 0, 0), 3)
            cv2.putText(vis, f"{'OB'[b.color]}{b.d:.0f}", b.px, 0, 0.7, (255, 255, 255), 1)
        for c, rd in enumerate(cells):
            r, k = divmod(c, 3)
            p = cam.to_px([(CELL_X[r], CELL_Y[k])])[0].astype(int)
            cv2.circle(vis, tuple(p), 6, (0, 255, 0), -1)
            lab = "-" if rd.color is None else f"{'OB'[rd.color]}{'SML'[rd.size]}"
            cv2.putText(vis, f"{c}:{lab}", tuple(p + [8, 20]), 0, 0.8, (0, 120, 0), 2)
        cv2.imwrite(str(debug_path), vis)
    return cells


def learn(frame, cam: BoardCam, layout: dict):
    """layout {cell: size} of pieces standing alone on the board -> per-size mean diameter."""
    cells = read_cells(frame, cam)
    per = {0: [], 1: [], 2: []}
    for c, s in layout.items():
        if cells[c].d is not None:
            per[s].append(cells[c].d)
    return {s: (round(float(np.mean(v)), 1), round(float(np.std(v)), 1)) for s, v in per.items() if v}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "read"
    OUT.mkdir(exist_ok=True)
    src = next((a for a in sys.argv[2:] if a.endswith((".jpg", ".png"))), None)
    frame = cv2.imread(src) if src else grab()
    cv2.imwrite(str(OUT / "topcam_last.jpg"), frame)
    if cmd == "calibrate":
        cam = calibrate(frame)
        read_cells(frame, cam, OUT / "topcam_calib_check.jpg")
    elif cmd == "learn":   # python topcam.py learn 0=2 4=1 8=0  (cell=size)
        cam = BoardCam.load()
        layout = {int(a.split("=")[0]): int(a.split("=")[1]) for a in sys.argv[2:] if "=" in a}
        print(learn(frame, cam, layout))
    else:
        cam = BoardCam.load()
        for c, rd in enumerate(read_cells(frame, cam, OUT / "topcam_read.jpg")):
            print(c, asdict(rd))


def off_board_pieces(frame, cam: BoardCam, color: int = 0, region=((80.0, 345.0), (-300.0, -100.0))):
    """Pieces of `color` standing on the desk in `region` ((x0, x1), (y0, y1) world mm), outside the board.

    Returns [{"x", "y", "d", "px"}] sorted by apparent size d (hull-equivalent diameter on the board plane).
    Sizes are NOT decided here: the caller knows how many of each size are left and assigns them by rank.
    Touching pieces are split with the distance transform."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = ORANGE if color == 0 else BLUE
    m = cv2.inRange(hsv, (lo[0], 112, 175), hi)   # desk wood: S~120 V~160 (daylight), S~90 V~190 (evening); pieces S 125+, V 195+
    if BAR_MASK.exists():
        bars = cv2.imread(str(BAR_MASK), cv2.IMREAD_GRAYSCALE)
        if bars is not None and bars.shape == m.shape:
            m &= ~cv2.dilate(bars, np.ones((15, 15), np.uint8))
    board = cam.to_px([(87, 108), (317, 108), (317, -108), (87, -108)]).astype(np.int32)
    cv2.fillPoly(m, [board], 0)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if st[i, 4] < 250:
            continue
        blob = (lab == i).astype(np.uint8)
        dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        # touching pieces: several distance-transform peaks far apart -> split around them
        peaks = []
        thr = dt.max() * 0.6
        cand = np.argwhere((dt >= thr) & (dt == cv2.dilate(dt, np.ones((21, 21), np.uint8))))
        for yx in sorted(cand.tolist(), key=lambda p: -dt[p[0], p[1]]):
            if all(np.hypot(yx[0] - q[0], yx[1] - q[1]) > dt.max() * 1.2 for q in peaks):
                peaks.append(yx)
        parts = [blob]
        area_px = int(blob.sum())
        cnt0 = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        hull0 = cv2.convexHull(max(cnt0, key=cv2.contourArea))
        d0 = 2 * np.sqrt(abs(cv2.contourArea(cam.to_world(hull0.reshape(-1, 2).astype(np.float64)).astype(np.float32))) / np.pi)
        if d0 > 58:                             # bigger than one large piece: k-means into touching pieces
            k = 2 if d0 < 85 else 3
            ys, xs = np.nonzero(blob)
            _, lab_k, _ = cv2.kmeans(np.c_[xs, ys].astype(np.float32), k, None,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
            parts = []
            for j in range(k):
                pm = np.zeros_like(blob)
                sel = lab_k.ravel() == j
                pm[ys[sel], xs[sel]] = 1
                parts.append(pm)
        elif len(peaks) > 1:
            ys, xs = np.nonzero(blob)
            d2 = [(ys - py) ** 2 + (xs - px) ** 2 for py, px in peaks]
            owner = np.argmin(np.stack(d2), axis=0)
            parts = []
            for k in range(len(peaks)):
                pm = np.zeros_like(blob)
                pm[ys[owner == k], xs[owner == k]] = 1
                parts.append(pm)
        for pm in parts:
            cnt = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not cnt:
                continue
            hull = cv2.convexHull(max(cnt, key=cv2.contourArea))
            w = cam.to_world(hull.reshape(-1, 2).astype(np.float64))
            d = 2 * np.sqrt(abs(cv2.contourArea(w.astype(np.float32))) / np.pi)
            mo = cv2.moments(pm, True)
            c = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
            x, y = cam.to_world([c])[0]
            (x0, x1), (y0, y1) = region
            if x0 <= x <= x1 and y0 <= y <= y1 and d > 12:
                out.append({"x": round(float(x), 1), "y": round(float(y), 1), "d": round(float(d), 1),
                            "px": (int(c[0]), int(c[1]))})
    return sorted(out, key=lambda p: p["d"])


# Rough top-camera position (world mm) from the pole-mounted PnP fit. Piece tops stand above the bar plane the
# homography was fitted on, so their blob centres are pushed away from the camera nadir; undo that per size.
CAMERA_POS = np.array([-90.0, 33.0, 650.0])
BLOB_Z = {0: 28.0, 1: 38.0, 2: 48.0}      # height of the visible blob centre per size (mm)
PLANE_Z = 17.0


def parallax_fix(x, y, size):
    f = (CAMERA_POS[2] - BLOB_Z[size]) / (CAMERA_POS[2] - PLANE_Z)
    return (float(CAMERA_POS[0] + (x - CAMERA_POS[0]) * f), float(CAMERA_POS[1] + (y - CAMERA_POS[1]) * f))


GRASP_OFFSETS = OUT / "grasp_offsets.json"   # learned by robot/grasp_probe.py: true - parallax_fix(camera), per size


def assign_sizes(pieces, counts, use_offsets=True):
    """counts[s] = how many of size s should be there. Smallest blobs get the smallest sizes."""
    sizes = [s for s in (0, 1, 2) for _ in range(counts[s])]
    if len(sizes) != len(pieces):
        return None
    offs = json.loads(GRASP_OFFSETS.read_text()) if (use_offsets and GRASP_OFFSETS.exists()) else {}
    out = []
    for p, s in zip(pieces, sizes):
        x, y = parallax_fix(p["x"], p["y"], s)
        o = offs.get(str(s)) or (offs.get("1") if offs else None)   # unknown size: borrow the medium offset
        if o:
            x, y = x + o["dx"], y + o["dy"]
        out.append({**p, "size": s, "x": round(x, 1), "y": round(y, 1), "x_raw": p["x"], "y_raw": p["y"]})
    return out
