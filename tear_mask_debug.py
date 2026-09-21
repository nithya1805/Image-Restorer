"""
Tear-mask debugger
==================
Builds ONLY the tear / missing-paper mask for one image and saves every
intermediate stage, so you can see exactly how the mask is formed.

Assumes the document was photographed on a dark (black) background.
Prints a verdict line "TEAR DETECTED: YES / NO". Images without a dark
backdrop can't be checked: they get NO, an empty mask, and a note saying why.

By default only the final tear mask (07_tear_mask.png) is saved.
With --debug every stage is saved as a numbered PNG + one side-by-side panel:
  01 background  : dark pixels connected to the image border (the backdrop)
  02 paper       : everything that is not backdrop (enclosed holes included)
  03 outline     : the intact page shape (rotated rectangle or convex hull)
  04 edge tears  : backdrop pixels that lie INSIDE the outline
                   -> missing corners, torn edges, cracks open to the edge
  05 holes       : near-black regions enclosed by paper (not thin ink strokes)
  06 ink         : strokes darker than their local paper (protected from growth)
  07 tear mask   : edge tears + holes, grown a few px into the ragged paper
                   edge, but never over ink
  08 overlay     : colour-coded view on the original
  09 inpainted   : OpenCV Telea fill using the tear mask (quick preview only)

Usage
  python tear_mask_debug.py                            # default test image
  python tear_mask_debug.py path/to/image.png
  python tear_mask_debug.py image.png --outline hull   # follow page shape instead of rectangle
  python tear_mask_debug.py image.png --debug          # also save all stages + panel
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR   = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_DIR / "Inpainiting_test_image.png"
DEFAULT_OUT   = PROJECT_DIR / "tear_debug"

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
# Pixel sizes are tuned for an image whose long side is ~1536 px and are scaled
# proportionally for other sizes.

BG_MARGIN        = 25     # backdrop = gray < (border median + BG_MARGIN)
HOLE_MARGIN      = 15     # holes must be almost as dark as the backdrop
MIN_PIECE_FRAC   = 0.01   # paper pieces smaller than this fraction are lint/noise
MIN_REGION_FRAC  = 1e-4   # tear/hole regions smaller than this are ignored
MIN_HOLE_RADIUS  = 4      # px; a hole must fit a circle this big (rejects ink strokes)
SLIVER_OPEN      = 7      # px; removes thin slivers between wavy page edge and outline
GROW_PX          = 4      # px; grow the mask into the ragged paper edge
INK_WINDOW       = 31     # px; window for the local paper estimate
INK_CONTRAST     = 25     # ink = darker than the local paper by this many gray levels
DARK_BG_MAX      = 100    # border median above this = no dark backdrop, can't check
TEAR_MIN_FRAC    = 0.001  # verdict YES if tears + holes cover at least this fraction


# ─── IO (np.fromfile handles non-ASCII Windows paths) ─────────────────────────

def imread(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def imwrite(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buf.tofile(str(path))


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def components(mask: np.ndarray):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def keep_components(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Vectorised 'keep these labels' -> uint8 mask. keep[0] must be False."""
    return keep[labels].astype(np.uint8) * 255


# ─── STAGES ───────────────────────────────────────────────────────────────────

def find_background(gray: np.ndarray):
    """Stage 1: the dark backdrop = dark pixels connected to the image border."""
    h, w  = gray.shape
    blur  = cv2.medianBlur(gray, 5)

    # The image border is (almost) always backdrop, so it gives the backdrop level.
    b = max(2, min(h, w) // 100)
    border = np.concatenate([blur[:b].ravel(), blur[-b:].ravel(),
                             blur[:, :b].ravel(), blur[:, -b:].ravel()])
    bg_level = float(np.median(border))
    thresh = bg_level + BG_MARGIN

    dark = (blur < thresh).astype(np.uint8) * 255
    n, labels, stats, _ = components(dark)
    x, y, ww, hh = (stats[:, i] for i in range(4))
    touches = (x == 0) | (y == 0) | (x + ww >= w) | (y + hh >= h)
    touches[0] = False                      # label 0 = non-dark pixels
    background = keep_components(labels, touches)
    return blur, bg_level, thresh, background


def find_paper(background: np.ndarray) -> np.ndarray:
    """Stage 2: everything that isn't backdrop, minus small specks."""
    not_bg = cv2.bitwise_not(background)
    n, labels, stats, _ = components(not_bg)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_PIECE_FRAC * background.size
    keep[0] = False
    return keep_components(labels, keep)


def find_outline(paper: np.ndarray, mode: str):
    """Stage 3: the shape the page had before it was torn."""
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack(contours)
    if mode == "rect":
        poly = cv2.boxPoints(cv2.minAreaRect(pts))
    else:
        poly = cv2.convexHull(pts).reshape(-1, 2)
    poly = np.round(poly).astype(np.int32)
    outline = np.zeros_like(paper)
    cv2.fillPoly(outline, [poly], 255)
    return outline, poly


def find_edge_tears(outline, background, min_area, sliver):
    """Stage 4: backdrop showing inside the page outline."""
    edge = cv2.bitwise_and(outline, background)
    # Wavy but intact edges leave thin slivers against the outline: drop them.
    edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, ellipse(sliver))
    n, labels, stats, _ = components(edge)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep_components(labels, keep), labels, stats, keep


def find_holes(blur, background, bg_level, min_area, min_radius):
    """Stage 5: near-black regions enclosed by paper (internal holes)."""
    cand = ((blur < bg_level + HOLE_MARGIN) & (background == 0)).astype(np.uint8) * 255
    n, labels, stats, _ = components(cand)

    # Largest inscribed radius per component: holes are blobs, ink is thin.
    dist  = cv2.distanceTransform(cand, cv2.DIST_L2, 5)
    max_r = np.zeros(n, np.float32)
    idx   = cand > 0
    np.maximum.at(max_r, labels[idx], dist[idx])

    keep = (stats[:, cv2.CC_STAT_AREA] >= min_area) & (max_r >= min_radius)
    keep[0] = False
    return keep_components(labels, keep), stats, keep, max_r


def find_ink(gray, paper, window):
    """Stage 6: pixels clearly darker than the paper around them (any colour)."""
    window = window | 1                     # medianBlur needs an odd size
    local_paper = cv2.medianBlur(gray, window)
    contrast = local_paper.astype(np.int16) - gray.astype(np.int16)
    return ((contrast > INK_CONTRAST) & (paper > 0)).astype(np.uint8) * 255


def build_tear_mask(edge, holes, ink, grow):
    """Stage 7: core damage + a thin ring into the paper edge, skipping ink."""
    core  = cv2.bitwise_or(edge, holes)
    grown = cv2.dilate(core, ellipse(2 * grow + 1))
    ring  = grown & ~core & ~ink
    return core | ring, core, ring


# ─── VISUALISATION ────────────────────────────────────────────────────────────

def tint(img, mask, color, alpha=0.6):
    out = img.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def labelled(img, text, width):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_panel(tiles, cols=3, width=640):
    tiles = [labelled(img, name, width) for name, img in tiles]
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(img_path: Path, out_root: Path, outline_mode: str, debug: bool):
    bgr  = imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total = h * w
    s = max(h, w) / 1536                    # scale pixel parameters to image size
    px = lambda v: max(1, int(round(v * s)))
    min_area = MIN_REGION_FRAC * total
    pct = lambda m: np.count_nonzero(m) / total * 100

    out = out_root / img_path.stem
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nImage   : {img_path.name}  {w}x{h}")

    blur, bg_level, thresh, background = find_background(gray)
    print(f"01 background : border level {bg_level:.0f}, threshold gray < {thresh:.0f}"
          f"  -> {pct(background):.1f}% of image")

    mask_path = out / "07_tear_mask.png"
    skip = None
    if bg_level > DARK_BG_MAX:
        skip = f"border is bright (median {bg_level:.0f}), no dark backdrop to find tears against"
    else:
        paper = find_paper(background)
        print(f"02 paper      : {pct(paper):.1f}% of image")
        if not paper.any():
            skip = "no paper found"
    if skip:
        imwrite(mask_path, np.zeros_like(gray))
        print(f"\nTEAR DETECTED: NO  (check skipped: {skip})")
        print(f"Saved empty tear mask to {mask_path}")
        return np.zeros_like(gray), False

    outline, poly = find_outline(paper, outline_mode)
    print(f"03 outline    : {outline_mode}, {pct(outline):.1f}% of image "
          f"(paper fills {np.count_nonzero(paper) / np.count_nonzero(outline) * 100:.1f}% of it)")

    edge, e_labels, e_stats, e_keep = find_edge_tears(
        outline, background, min_area, px(SLIVER_OPEN))
    print(f"04 edge tears : {int(e_keep.sum())} regions, {pct(edge):.2f}% of image")
    for i in np.argsort(-e_stats[:, cv2.CC_STAT_AREA]):
        if e_keep[i]:
            x, y, ww, hh, a = e_stats[i]
            print(f"      area {a:7d}  bbox x={x} y={y} {ww}x{hh}")

    holes, h_stats, h_keep, h_r = find_holes(
        blur, background, bg_level, min_area, px(MIN_HOLE_RADIUS))
    print(f"05 holes      : {int(h_keep.sum())} regions, {pct(holes):.2f}% of image")
    for i in np.argsort(-h_stats[:, cv2.CC_STAT_AREA]):
        if h_keep[i]:
            x, y, ww, hh, a = h_stats[i]
            print(f"      area {a:7d}  bbox x={x} y={y} {ww}x{hh}  radius {h_r[i]:.1f}")

    ink = find_ink(gray, paper, px(INK_WINDOW))
    print(f"06 ink        : {pct(ink):.1f}% of image")

    tear_mask, core, ring = build_tear_mask(edge, holes, ink, px(GROW_PX))
    print(f"07 tear mask  : {pct(tear_mask):.2f}% of image "
          f"(core {pct(core):.2f}% + edge ring {pct(ring):.2f}%)")

    has_tear = np.count_nonzero(core) >= TEAR_MIN_FRAC * total
    print(f"\nTEAR DETECTED: {'YES' if has_tear else 'NO'}  "
          f"(damaged area {pct(core):.2f}% of image, threshold {TEAR_MIN_FRAC * 100:.2f}%)")

    imwrite(mask_path, tear_mask)
    print(f"Saved tear mask to {mask_path}")
    if not debug:
        return tear_mask, has_tear

    # Colour-coded overlay: red = edge tears, magenta = holes,
    # yellow = growth ring, green line = outline.
    overlay = tint(bgr, edge, (0, 0, 255))
    overlay = tint(overlay, holes, (255, 0, 255))
    overlay = tint(overlay, ring, (0, 255, 255), 0.7)
    cv2.polylines(overlay, [poly], True, (0, 255, 0), max(1, px(2)))

    outline_view = bgr.copy()
    cv2.polylines(outline_view, [poly], True, (0, 255, 0), max(1, px(3)))
    ink_view = tint(bgr, ink, (255, 128, 0), 0.8)

    stages = [
        ("01_background", background),
        ("02_paper",      paper),
        ("03_outline",    outline_view),
        ("04_edge_tears", edge),
        ("05_holes",      holes),
        ("06_ink",        ink_view),
        ("07_tear_mask",  tear_mask),
        ("08_overlay",    overlay),
        ("09_inpainted_telea", cv2.inpaint(bgr, tear_mask, px(5), cv2.INPAINT_TELEA)),
    ]

    for name, img in stages:
        imwrite(out / f"{name}.png", img)
    imwrite(out / "00_panel.png", make_panel([("00 original", bgr)] + stages))

    print(f"Saved debug stages to {out}")
    return tear_mask, has_tear


def main():
    ap = argparse.ArgumentParser(description="Visualise how the tear mask is built.")
    ap.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--outline", choices=["rect", "hull"], default="rect",
                    help="rect: page was rectangular (restores missing corners); "
                         "hull: follow the convex page shape")
    ap.add_argument("--debug", action="store_true",
                    help="also save every intermediate stage, the panel and a Telea preview")
    args = ap.parse_args()
    run(args.image, args.out, args.outline, args.debug)


if __name__ == "__main__":
    main()
