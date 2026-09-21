"""
LaMa tear inpainting
====================
Builds the tear mask with tear_mask_debug.py (same code, same parameters) and
fills it with big-LaMa. Images whose verdict is "TEAR DETECTED: NO" are skipped.

What goes into LaMa
  - the backdrop outside the page outline is painted in the median paper colour,
    so tears are filled with paper instead of black
  - the tear mask is grown EXTRA_GROW_PX further (never over ink), so the
    browned, ragged tear fringe is replaced too instead of being copied inward
  - LaMa runs at --max-side (default 768 px); big holes come out much cleaner
    and ~4x faster than at full resolution. Only masked pixels are replaced,
    so the rest of the page keeps its full resolution.

Outputs in <out>/<image stem>/
  07_tear_mask.png     the tear mask, identical to tear_mask_debug.py
  lama_mask.png        the pixels actually filled (tear mask + extra growth)
  lama_inpainted.png   the result

Needs: torch, opencv-python, numpy, and the big-lama.pt TorchScript model
(default: models/big-lama.pt in this folder).

Usage
  python lama_inpaint.py                               # default test image
  python lama_inpaint.py path/to/image.png
  python lama_inpaint.py path/to/folder                # every image in the folder
  python lama_inpaint.py image.png --outline hull      # follow page shape instead of rectangle
  python lama_inpaint.py image.png --max-side 0        # run LaMa at full resolution
  python lama_inpaint.py image.png --model path/to/big-lama.pt
"""

import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from tear_mask_debug import (DEFAULT_IMAGE, INK_WINDOW, ellipse, find_background, find_ink,
                             find_outline, find_paper, imread, imwrite,
                             run as build_tear_mask)


PROJECT_DIR   = Path(__file__).resolve().parent
DEFAULT_OUT   = PROJECT_DIR / "lama_output"
DEFAULT_MODEL = PROJECT_DIR / "models" / "big-lama.pt"
MODEL_URL     = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAD_MULTIPLE  = 8      # LaMa needs H and W divisible by 8
MAX_SIDE      = 768    # LaMa working resolution (long side); 0 = full resolution
EXTRA_GROW_PX = 8      # px at 1536 long side; extra mask growth for LaMa only


# ─── MODEL ────────────────────────────────────────────────────────────────────

def load_lama(path: Path, device: torch.device):
    if not path.exists():
        raise SystemExit(f"LaMa model not found: {path}\n"
                         f"Download big-lama.pt from {MODEL_URL} or pass --model.")
    with warnings.catch_warnings():
        # torch.jit.load warns on Python 3.14+, but the model loads and runs fine.
        warnings.simplefilter("ignore", FutureWarning)
        model = torch.jit.load(str(path), map_location=device)
    return model.eval()


def lama_inpaint(model, bgr: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    """bgr uint8 HxWx3, mask uint8 HxW (>0 = fill) -> raw LaMa output, bgr uint8 HxWx3.
    LaMa ignores whatever is under the mask."""
    h, w = mask.shape
    ph, pw = -h % PAD_MULTIPLE, -w % PAD_MULTIPLE
    rgb = cv2.copyMakeBorder(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0, ph, 0, pw,
                             cv2.BORDER_REFLECT)
    m   = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_REFLECT)

    # Model input: RGB [1,3,H,W] and mask [1,1,H,W], both float in [0, 1].
    image  = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(device)
    mask_t = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(device)
    with torch.inference_mode():
        out = model(image, mask_t)[0].permute(1, 2, 0).cpu().numpy()

    out = np.clip(out[:h, :w] * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def lama_fill(model, bgr, mask, device, max_side: int) -> np.ndarray:
    """Run LaMa at max_side (long side) and scale the result back to full size."""
    h, w = mask.shape
    if not max_side or max(h, w) <= max_side:
        return lama_inpaint(model, bgr, mask, device)
    f = max_side / max(h, w)
    size = (round(w * f), round(h * f))
    small_mask = (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
    small = lama_inpaint(model, cv2.resize(bgr, size, interpolation=cv2.INTER_AREA),
                         small_mask, device)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


# ─── PREPARATION ──────────────────────────────────────────────────────────────

def prepare(bgr: np.ndarray, tear_mask: np.ndarray, outline_mode: str):
    """Returns (context, lama_mask):
    context   : the image with the backdrop outside the page outline painted in
                the median paper colour
    lama_mask : tear mask grown EXTRA_GROW_PX more (skipping ink), clipped to the
                page outline; these are the only pixels that will change"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = max(gray.shape) / 1536              # same pixel scaling as tear_mask_debug
    px = lambda v: max(1, int(round(v * s)))

    paper = find_paper(find_background(gray)[3])
    outline, _ = find_outline(paper, outline_mode)
    context = bgr.copy()
    context[outline == 0] = np.median(bgr[paper > 0], axis=0)

    ink = find_ink(gray, paper, px(INK_WINDOW))
    grown = cv2.dilate(tear_mask, ellipse(2 * px(EXTRA_GROW_PX) + 1))
    lama_mask = (tear_mask | (grown & ~ink)) & outline
    return context, lama_mask


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def process(model, device, img_path: Path, out_root: Path, outline_mode: str, max_side: int):
    tear_mask, has_tear = build_tear_mask(img_path, out_root, outline_mode, debug=False)
    if not has_tear:
        print("LaMa    : skipped (no tear)")
        return

    t = time.time()
    bgr = imread(img_path)
    context, lama_mask = prepare(bgr, tear_mask, outline_mode)
    filled = lama_fill(model, context, lama_mask, device, max_side)
    result = np.where(lama_mask[..., None] > 0, filled, bgr)

    out = out_root / img_path.stem
    imwrite(out / "lama_mask.png", lama_mask)
    imwrite(out / "lama_inpainted.png", result)
    res = f"{max_side}px" if max_side and max(bgr.shape[:2]) > max_side else "full-res"
    print(f"LaMa    : {time.time() - t:.1f}s on {device} ({res})  -> {out / 'lama_inpainted.png'}")


def main():
    ap = argparse.ArgumentParser(description="Detect tears and inpaint them with LaMa.")
    ap.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE,
                    help="image file or folder of images")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to big-lama.pt")
    ap.add_argument("--outline", choices=["rect", "hull"], default="rect",
                    help="rect: page was rectangular (restores missing corners); "
                         "hull: follow the convex page shape")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help="LaMa working resolution, long side in px (0 = full resolution)")
    args = ap.parse_args()

    if args.image.is_dir():
        paths = [p for p in sorted(args.image.iterdir()) if p.suffix.lower() in VALID_EXTS]
    else:
        paths = [args.image]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_lama(args.model, device)
    print(f"Model   : {args.model}  ({device})")

    for p in paths:
        process(model, device, p, args.out, args.outline, args.max_side)


if __name__ == "__main__":
    main()
