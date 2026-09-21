#!/usr/bin/env python3
"""
shadow_webapp/app.py
============================================================
Simple Flask web app that checks uploaded document images and fixes them:

    INPUT
      -> dewarp DETECTION (UVDoc deformation score >= CONFIG["dewarp_threshold"]?)
           yes -> dewarpING         (UVDoc, same prediction)
      -> SHADOW DETECTION (ONNX classifier, models/shadow_model.onnx), on the dewarped image
           yes -> SHADOW CORRECTION (DocRes, task "deshadowing")
      -> BLUR DETECTION   (MobileNetV3, models/blur_model.pt), on the dewarped/deshadowed image
           yes -> BLUR CORRECTION   (DocRes, task "deblurring")
      -> RESOLUTION CHECK (width or height of the corrected image below CONFIG["upscale_below_px"] = 1200 px?)
           yes -> UPSCALING 4x      (Real-ESRGAN, models/RealESRGAN_x4plus.pth), once only
      -> ENHANCEMENT      (DocRes, task "appearance") - always, on every image
      -> DAMAGE DETECTION (tear / missing-paper mask, tear_mask_debug.py) - the last check
           yes -> INPAINTING        (big-LaMa, lama_inpaint.py)
      -> FINAL OUTPUT

  * Blur detection: grid of crops at training resolution, no resizing, blur
    probabilities averaged (same method as BLUR/infer_blur.py).
  * Shadow detection: letterbox to a white square, BGR->RGB, scale to [0,1],
    optional CLAHE (same preprocessing as infer_shadow_onnx.py).
  * Corrections: DocRes (models/docres.pkl), same pre/post-processing as
    DocRes/inference.py. Only the Restormer network definition is loaded from
    the DocRes folder (CONFIG["docres_code_dir"]).

Images or a whole folder can be uploaded. Uploads are processed in the
background (one run at a time) and the run page (/run/<run id>) shows, live,
which step each image is at. Every run is saved in
results/<run id>/:  output/ (the images, same names and sub-folders: corrected
ones fixed, good ones copied unchanged), report.csv and a ZIP of both, which
the results page offers for download. Runs older than
CONFIG["keep_results_days"] are deleted automatically.

------------------------------------------------------------
This folder is self-contained (see README.txt), apart from the DocRes code.
INSTALL (once):  double-click setup.bat
RUN:             double-click start_server.bat
then open http://127.0.0.1:5000
------------------------------------------------------------
"""

import copy
import csv
import importlib.util
import io
import math
import os
import queue
import re
import secrets
import shutil
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from flask import Flask, abort, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms

import lama_inpaint as lamalib      # tear inpainting with big-LaMa (this folder)
import tear_mask_debug as tearlib   # tear / missing-paper mask (this folder)

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Hardware
    "device": "auto",           # "auto" = NVIDIA GPU (CUDA) if available, else CPU; or force "cuda" / "cpu"

    # Upscaling (Real-ESRGAN) — the last step. x4plus = 4x; RealESRGAN_x2plus.pth also works (2x)
    "esrgan_model_path": BASE_DIR / "models" / "RealESRGAN_x2plus.pth",
    "upscale_below_px": 1200,   # images whose width or height is below this are upscaled (once only)
    "esrgan_tile": 512,         # processed in tiles of this many input px to limit memory (0 = whole image)

    # Shadow (ONNX)
    "shadow_model_path": BASE_DIR / "models" / "shadow_model.onnx",
    "shadow_threshold": 0.5,    # shadow_probability >= threshold -> SHADOW
    "enhance_contrast": False,  # must match the value used in training
    "use_gpu": False,           # shadow model only (small, fast on CPU); True needs onnxruntime-gpu/-directml

    # Blur (PyTorch)
    "blur_model_path": BASE_DIR / "models" / "blur_model.pt",
    "blur_threshold": 0.5,      # p_blur >= threshold -> BLURRED
    "blur_max_crops": 16,       # max crops scored per image

    # Correction (DocRes)
    "docres_model_path": BASE_DIR / "models" / "docres.pkl",
    "docres_code_dir": Path(r"C:\DocRes"),  # folder containing m odels/restormer_arch.py
    "deshadow_max_side": 1600,  # as in DocRes: larger images are processed at this size, then the
                                # correction is applied to the full-resolution image
    "deblur_max_side": 1600,    # deblurring runs at full resolution in DocRes; larger images are
                                # downscaled to this first (and scaled back) to limit memory use
    "docres_tile_gpu": 768,     # on the GPU, DocRes runs in overlapping tiles of this size (multiple of 8):
                                # 768 px needs ~5 GB, a whole 1600 px page would need far more than 8 GB
    "docres_tile_overlap": 64,  # overlap between tiles (multiple of 8), blended smoothly
    "appearance_max_side": 1600,  # DocRes "appearance" (final enhancement): as for deshadowing, larger
                                  # images are processed at this size and the result applied full-size

    # dewarping (UVDoc) — the first step
    "uvdoc_code_dir": Path(r"C:\UVDoc"),                         # folder containing UVDoc's model.py
    "uvdoc_model_path": Path(r"C:\UVDoc\model\best_model.pkl"),
    "dewarp_threshold": 0.02,   # bending left in UVDoc's predicted grid after removing shift/zoom/rotation
                                # (coordinates -1..1); score >= threshold -> WARPED -> dewarp. Samples:
                                # curved pages 0.024-0.035, flat pages / crops / mild warps 0.005-0.017

    # Inpainting (tear mask + big-LaMa) — the last step
    "lama_model_path": BASE_DIR / "models" / "big-lama.pt",
    "tear_outline_mode": "rect",  # "rect": page was rectangular (restores missing corners);
                                  # "hull": follow the page shape
    "lama_max_side": 768,         # LaMa working resolution, long side in px (0 = full resolution)

    # Web
    "results_dir": BASE_DIR / "results",  # one sub-folder per run: output/, report.csv, ZIP
    "keep_results_days": 7,     # older runs are deleted when a new run starts
    "max_upload_mb": 2048,      # per request (a whole folder is one request)
    "display_max_side": 1200,   # preview images on the results page are downscaled to this
    "host": "0.0.0.0",          # 0.0.0.0 = reachable from other devices on the network; 127.0.0.1 = this PC only
    "port": 5004,
}
# =============================================================================

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".j2k", ".jp2"}
JPEG2000_EXTS = {".j2k", ".jp2"}  # OpenCV here can read these but not write them: saved with Pillow
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REPORT_FIELDS = ["file", "status", "resolution", "blur_prob", "shadow_prob", "dewarp_score", "damaged_pct",
                 "corrections", "seconds"]
ENCODE_PARAMS = {".jpg": [cv2.IMWRITE_JPEG_QUALITY, 95], ".jpeg": [cv2.IMWRITE_JPEG_QUALITY, 95],
                 ".webp": [cv2.IMWRITE_WEBP_QUALITY, 95]}  # other formats are saved lossless
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


# ---------------------------------------------------------------------------
# DEVICE — the PyTorch models (blur, DocRes, Real-ESRGAN) run on the GPU when there is one
# ---------------------------------------------------------------------------
def pick_device(setting):
    if setting != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if setting == "cuda":
        print("WARNING: CONFIG['device'] is 'cuda' but PyTorch sees no CUDA GPU - using the CPU "
              "(is the CUDA build of PyTorch installed? see requirements.txt)")
    return torch.device("cpu")


DEVICE = pick_device(CONFIG["device"])
if DEVICE.type == "cuda":
    _props = torch.cuda.get_device_properties(0)
    print(f"Device:       GPU {_props.name} ({_props.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda})")
else:
    print("Device:       CPU")


def to_cpu(out):
    return tuple(o.cpu() for o in out) if isinstance(out, (tuple, list)) else out.cpu()


def run_on_device(model, x):
    """model(x) on DEVICE, result on the CPU. If the GPU runs out of memory (very large image),
    this one call runs on the CPU instead of failing."""
    if DEVICE.type != "cuda":
        return model(x)
    try:
        return to_cpu(model(x.to(DEVICE)))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"GPU out of memory for input {tuple(x.shape)} - running this step on the CPU", flush=True)
        model.cpu()
        try:
            return model(x.cpu())
        finally:
            model.to(DEVICE)


# ---------------------------------------------------------------------------
# SHADOW — preprocessing identical to infer_shadow_onnx.py / train_shadow_classifier.py
# ---------------------------------------------------------------------------
def letterbox_resize(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def preprocess_image(img, img_size, enhance_contrast=False):
    """BGR uint8 image -> letterboxed RGB float32 in [0,1]."""
    if enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    img = letterbox_resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def create_shadow_session(model_path, use_gpu):
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if p in available:
                providers.insert(0, p)
                break

    t0 = time.time()
    sess = ort.InferenceSession(str(model_path), providers=providers)
    inp = sess.get_inputs()[0]
    meta = sess.get_modelmeta().custom_metadata_map
    img_size = int(meta.get("img_size", inp.shape[1]))
    print(f"Shadow model: {model_path}  ({time.time() - t0:.2f}s, providers={sess.get_providers()}, "
          f"img_size={img_size})")
    return sess, inp.name, img_size


def predict_shadow(img):
    """BGR uint8 image -> shadow probability."""
    batch = preprocess_image(img, IMG_SIZE, CONFIG["enhance_contrast"])[None]
    return float(SESSION.run(None, {INPUT_NAME: batch})[0].ravel()[0])


# ---------------------------------------------------------------------------
# BLUR — same model and crop scoring as BLUR/infer_blur.py
# ---------------------------------------------------------------------------
def load_blur_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["arch"] != "v3_small":
        raise SystemExit(f"ERROR: unsupported blur model arch '{ckpt['arch']}' (expected v3_small)")
    # Same layout as train_mobilenet.build_model: torchvision MobileNetV3-Small with a new last layer
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    print(f"Blur model:   {path}  (arch {ckpt['arch']}, epoch {ckpt.get('epoch')}, "
          f"val acc {ckpt.get('val_acc', float('nan')):.4f}, img_size={ckpt['img_size']})")
    return model, ckpt["img_size"], ckpt["classes"].index("blur")


def grid_crops(img, size, max_crops):
    """Same as infer_blur.grid_crops, but also returns the (x, y) of each crop."""
    w, h = img.size
    if w < size or h < size:
        img = TF.pad(img, [0, 0, max(0, size - w), max(0, size - h)], fill=255, padding_mode="constant")
        w, h = img.size
    xs = [round(v) for v in torch.linspace(0, w - size, math.ceil(w / size)).tolist()] if w > size else [0]
    ys = [round(v) for v in torch.linspace(0, h - size, math.ceil(h / size)).tolist()] if h > size else [0]
    boxes = [(x, y) for y in ys for x in xs]
    if len(boxes) > max_crops:  # keep an evenly spaced subset
        step = len(boxes) / max_crops
        boxes = [boxes[int(i * step)] for i in range(max_crops)]
    return [img.crop((x, y, x + size, y + size)) for x, y in boxes], boxes


@torch.no_grad()
def predict_blur(pil_img):
    """Return (mean p_blur, [((x, y), p_blur_of_crop), ...])."""
    crops, boxes = grid_crops(pil_img, BLUR_SIZE, CONFIG["blur_max_crops"])
    batch = torch.stack([BLUR_TO_TENSOR(c) for c in crops]).to(DEVICE)
    probs = torch.softmax(BLUR_MODEL(batch), dim=1)[:, BLUR_IDX].cpu()
    return probs.mean().item(), list(zip(boxes, probs.tolist()))


def draw_blur_map(img, tiles, size, threshold):
    """Draw every scored crop on the image: orange = blurred, green = sharp."""
    out = img.copy()
    fill = out.copy()
    h, w = img.shape[:2]
    thickness = max(2, round(max(h, w) / 400))
    font_scale = max(0.5, size / 300)
    for (x, y), p in tiles:
        x2, y2 = min(w, x + size), min(h, y + size)
        color = (0, 140, 255) if p >= threshold else (60, 170, 60)  # BGR
        cv2.rectangle(fill, (x, y), (x2, y2), color, -1)
        cv2.rectangle(out, (x, y), (x2, y2), color, thickness)
    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    for (x, y), p in tiles:
        color = (0, 140, 255) if p >= threshold else (60, 170, 60)
        label = f"{p:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(out, (x, y), (x + tw + 8, y + th + 10), color, -1)
        cv2.putText(out, label, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# UPSCALING — Real-ESRGAN x4plus (or x2plus): RRDBNet as in basicsr, processing as in RealESRGANer
# (reflect pre-padding, overlapping tiles), without needing the basicsr/realesrgan packages
# ---------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN generator; the network always upsamples 4x. scale=4 (x4plus) takes the RGB
    image as is; scale=2 (x2plus) pixel-unshuffles it first (3 -> 12 channels, half size)."""

    def __init__(self, scale=4, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch * (4 if scale == 2 else 1), num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(F.pixel_unshuffle(x, 2) if self.scale == 2 else x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_esrgan(path):
    """Load a Real-ESRGAN RRDBNet checkpoint; x4plus or x2plus is detected from its first layer."""
    t0 = time.time()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    scale = {3: 4, 12: 2}.get(state["conv_first.weight"].shape[1])
    if scale is None:
        raise SystemExit(f"ERROR: {path} is not a Real-ESRGAN x4plus/x2plus (RRDBNet) model")
    model = RRDBNet(scale=scale)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    print(f"Real-ESRGAN:  {path}  (x{scale}, {time.time() - t0:.2f}s)")
    return model, scale


@torch.no_grad()
def esrgan_upscale(img):
    """BGR uint8 image -> BGR uint8 image ESRGAN_SCALE (4 or 2) times larger."""
    s = ESRGAN_SCALE
    h, w = img.shape[:2]
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    # Pad right/bottom by 10 px (reduces border artefacts); the x2 model also needs even sizes
    pre_pad, mod = 10, (2 if s == 2 else 1)
    pad_w, pad_h = pre_pad + (-(w + pre_pad)) % mod, pre_pad + (-(h + pre_pad)) % mod
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > pre_pad + 1 else "replicate")
    H, W = x.shape[2:]

    # Tiles of `tile` px with a 10 px overlap on each side (even offsets, for the x2 model's pixel_unshuffle)
    tile = CONFIG["esrgan_tile"] or max(H, W)
    tile, overlap = max(2, tile - tile % 2), 10
    out = torch.empty(1, 3, s * H, s * W)
    with HEAVY_LOCK:
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                ys, xs = max(y0 - overlap, 0), max(x0 - overlap, 0)
                ye, xe = min(y1 + overlap, H), min(x1 + overlap, W)
                o = run_on_device(ESRGAN_MODEL, x[:, :, ys:ye, xs:xe])
                oy, ox = s * (y0 - ys), s * (x0 - xs)
                out[:, :, s * y0:s * y1, s * x0:s * x1] = o[:, :, oy:oy + s * (y1 - y0), ox:ox + s * (x1 - x0)]

    out = out[0, :, :s * h, :s * w].clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# CORRECTION — DocRes, same pre/post-processing as DocRes/inference.py
# ---------------------------------------------------------------------------
def load_docres(code_dir, model_path):
    # Load restormer_arch.py by file path: DocRes's "models"/"utils" folder names clash with ours
    arch_file = Path(code_dir) / "models" / "restormer_arch.py"
    if not arch_file.is_file():
        raise SystemExit(f"ERROR: DocRes code not found: {arch_file} (set CONFIG['docres_code_dir'])")
    spec = importlib.util.spec_from_file_location("docres_restormer_arch", arch_file)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)

    t0 = time.time()
    model = arch.Restormer(inp_channels=6, out_channels=3, dim=48, num_blocks=[2, 3, 3, 4],
                           num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                           bias=False, LayerNorm_type="WithBias", dual_pixel_task=True)
    state = torch.load(model_path, map_location="cpu", weights_only=False)["model_state"]
    # Checkpoint was saved from a DataParallel model: strip the "module." prefix
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.eval().to(DEVICE)
    print(f"DocRes model: {model_path}  ({time.time() - t0:.2f}s)")
    return model


def stride_integral(img, stride):
    """Pad top/left (replicate) so height and width are multiples of stride."""
    h, w = img.shape[:2]
    padding_h, padding_w = (-h) % stride, (-w) % stride
    img = cv2.copyMakeBorder(img, padding_h, 0, padding_w, 0, borderType=cv2.BORDER_REPLICATE)
    return img, padding_h, padding_w


def deshadow_prompt(img):
    """DocRes deshadowing prompt: estimated paper background (text removed per channel)."""
    h, w = img.shape[:2]
    planes = []
    for plane in cv2.split(cv2.resize(img, (1024, 1024))):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        planes.append(cv2.medianBlur(dilated, 21))
    return cv2.resize(cv2.merge(planes), (w, h))


def appearance_prompt(img):
    """DocRes appearance prompt: background removed and normalised, per channel."""
    h, w = img.shape[:2]
    planes = []
    for plane in cv2.split(cv2.resize(img, (1024, 1024))):
        background = cv2.medianBlur(cv2.dilate(plane, np.ones((7, 7), np.uint8)), 21)
        diff = 255 - cv2.absdiff(plane, background)
        planes.append(cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
    return cv2.resize(cv2.merge(planes), (w, h))


def deblur_prompt(img):
    """DocRes deblurring prompt: grey Sobel edge magnitude."""
    abs_x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0))
    abs_y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1))
    high_frequency = cv2.cvtColor(cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(high_frequency, cv2.COLOR_GRAY2BGR)


def tile_starts(n, tile, overlap):
    """Start offsets of tiles of size `tile` covering 0..n with at least `overlap` px overlap."""
    if n <= tile:
        return [0]
    return list(range(0, n - tile, tile - overlap)) + [n - tile]


def blend_ramp(n, overlap, first, last):
    """1-D blending weights for a tile: fade in/out over `overlap` px, except at the image border."""
    r = torch.ones(n)
    if overlap and not first:
        r[:overlap] = torch.linspace(0, 1, overlap + 2)[1:-1]
    if overlap and not last:
        r[-overlap:] = torch.minimum(r[-overlap:], torch.linspace(1, 0, overlap + 2)[1:-1])
    return r


@torch.no_grad()
def docres_forward(in_im):
    """6-channel uint8 image (BGR + prompt) -> restored BGR uint8 image of the same size.
    Height and width must be multiples of 8 (see stride_integral)."""
    x = torch.from_numpy(in_im.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    H, W = x.shape[2:]
    tile = CONFIG["docres_tile_gpu"] if DEVICE.type == "cuda" else 0
    with HEAVY_LOCK:  # one DocRes / Real-ESRGAN run at a time: they are memory-hungry (GPU and CPU)
        if not tile or (H <= tile and W <= tile):
            pred = run_on_device(DOCRES_MODEL, x)
        else:  # overlapping tiles, blended with linear ramps so there are no seams
            overlap = CONFIG["docres_tile_overlap"]
            th, tw = min(tile, H), min(tile, W)
            out, weight = torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W)
            for y0 in tile_starts(H, th, overlap):
                for x0 in tile_starts(W, tw, overlap):
                    p = run_on_device(DOCRES_MODEL, x[:, :, y0:y0 + th, x0:x0 + tw])
                    wgt = (blend_ramp(th, overlap, y0 == 0, y0 + th == H)[:, None]
                           * blend_ramp(tw, overlap, x0 == 0, x0 + tw == W)[None, :])
                    out[:, :, y0:y0 + th, x0:x0 + tw] += p * wgt
                    weight[:, :, y0:y0 + th, x0:x0 + tw] += wgt
            pred = out / weight
    pred = torch.clamp(pred, 0, 1)
    return (pred[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def docres_shading_task(img, prompt, max_size):
    """DocRes tasks that work on the paper shading (deshadowing and appearance), exactly as
    DocRes/inference.py: run the model on image + prompt; for large images work at max_size and
    apply the estimated shading to the full-resolution image."""
    h, w = img.shape[:2]
    in_im = np.concatenate((img, prompt), -1)
    if max(w, h) < max_size:
        in_im, padding_h, padding_w = stride_integral(in_im, 8)
        return docres_forward(in_im)[padding_h:, padding_w:]

    pred = docres_forward(cv2.resize(in_im, (max_size, max_size)))
    pred[pred == 0] = 1
    shadow_map = cv2.resize(img, (max_size, max_size)).astype(float) / pred.astype(float)
    shadow_map = cv2.resize(shadow_map, (w, h))
    shadow_map[shadow_map == 0] = 0.00001
    return np.clip(img.astype(float) / shadow_map, 0, 255).astype(np.uint8)


def docres_deshadow(img):
    return docres_shading_task(img, deshadow_prompt(img), CONFIG["deshadow_max_side"])


def docres_appearance(img):
    """Final enhancement: DocRes task "appearance" (lighting, background and colour clean-up)."""
    return docres_shading_task(img, appearance_prompt(img), CONFIG["appearance_max_side"])


def docres_deblur(img):
    h, w = img.shape[:2]
    scale = CONFIG["deblur_max_side"] / max(h, w)
    work = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img

    padded, padding_h, padding_w = stride_integral(work, 8)
    out = docres_forward(np.concatenate((padded, deblur_prompt(padded)), -1))[padding_h:, padding_w:]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC) if scale < 1 else out


# ---------------------------------------------------------------------------
# dewarpING — UVDoc, same processing as UVDoc/demo.py (network from UVDoc/model.py)
# ---------------------------------------------------------------------------
UVDOC_IMG_SIZE = (488, 712)  # (w, h) network input, as IMG_SIZE in UVDoc/utils.py


def load_uvdoc(code_dir, model_path):
    # Load UVDoc's model.py by file path: its "utils"/"model" module names would clash
    arch_file = Path(code_dir) / "model.py"
    if not arch_file.is_file():
        raise SystemExit(f"ERROR: UVDoc code not found: {arch_file} (set CONFIG['uvdoc_code_dir'])")
    spec = importlib.util.spec_from_file_location("uvdoc_model", arch_file)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)

    t0 = time.time()
    model = arch.UVDocnet(num_filter=32, kernel_size=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False)["model_state"])
    model.eval().to(DEVICE)
    print(f"UVDoc model:  {model_path}  ({time.time() - t0:.2f}s)")
    return model


@torch.no_grad()
def uvdoc_grid(img):
    """BGR image -> UVDoc's predicted 2D grid, tensor (1, 2, Gh, Gw) with x, y in -1..1."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(cv2.resize(rgb, UVDOC_IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0)
    with HEAVY_LOCK:
        points_2d, _ = run_on_device(UVDOC_MODEL, inp)
    return points_2d[:1]


def deformation_score(grid):
    """How much the page is actually bent: mean distance of UVDoc's predicted grid points from the
    best affine fit (shift / zoom / rotation / shear) of a flat grid. Shifting or zooming the whole
    grid - which UVDoc also predicts for flat close-up crops with no page edges - is not warping and
    scores near 0; only curl that no straight-line transform explains is left."""
    g = grid[0].permute(1, 2, 0).numpy().reshape(-1, 2).astype(np.float64)  # predicted points (x, y)
    h, w = grid.shape[2:]
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    flat = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    coef, *_ = np.linalg.lstsq(flat, g, rcond=None)
    return float(np.linalg.norm(g - flat @ coef, axis=1).mean())


@torch.no_grad()
def uvdoc_unwarp(img, grid):
    """Unwarp the full-resolution BGR image with the predicted grid (bilinear_unwarping in UVDoc)."""
    h, w = img.shape[:2]
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    full_grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=True)
    out = F.grid_sample(x, full_grid.permute(0, 2, 3, 1), align_corners=True)
    out = (out[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# INPAINTING — tear mask from tear_mask_debug.py, filled with big-LaMa as in lama_inpaint.py
# ---------------------------------------------------------------------------
def load_lama(path):
    if not Path(path).is_file():
        print(f"WARNING: LaMa model not found: {path}\n         Inpainting is disabled. Download big-lama.pt "
              f"from {lamalib.MODEL_URL} or set CONFIG['lama_model_path'].")
        return None
    t0 = time.time()
    model = lamalib.load_lama(Path(path), DEVICE)
    print(f"LaMa model:   {path}  ({time.time() - t0:.2f}s)")
    return model


def tear_mask(bgr, outline_mode):
    """Tear / missing-paper mask: the same stages and parameters as tear_mask_debug.run(), but in
    memory (no files). Returns (mask, damaged fraction of the image, note or None)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(h, w) / 1536  # the parameters are tuned for a 1536 px long side
    px = lambda v: max(1, int(round(v * scale)))
    min_area = tearlib.MIN_REGION_FRAC * gray.size
    empty = np.zeros_like(gray)

    blur, bg_level, _, background = tearlib.find_background(gray)
    if bg_level > tearlib.DARK_BG_MAX:  # tears are found against a dark backdrop only
        return empty, 0.0, f"no dark backdrop (border level {bg_level:.0f})"
    paper = tearlib.find_paper(background)
    if not paper.any():
        return empty, 0.0, "no paper found"

    outline, _ = tearlib.find_outline(paper, outline_mode)
    edge = tearlib.find_edge_tears(outline, background, min_area, px(tearlib.SLIVER_OPEN))[0]
    holes = tearlib.find_holes(blur, background, bg_level, min_area, px(tearlib.MIN_HOLE_RADIUS))[0]
    ink = tearlib.find_ink(gray, paper, px(tearlib.INK_WINDOW))
    mask, core, _ = tearlib.build_tear_mask(edge, holes, ink, px(tearlib.GROW_PX))
    return mask, float(np.count_nonzero(core)) / gray.size, None  # plain float: the page sends it as JSON


def lama_inpaint(bgr, mask):
    """Fill the tear mask with big-LaMa, exactly as lama_inpaint.process() does."""
    context, fill_mask = lamalib.prepare(bgr, mask, CONFIG["tear_outline_mode"])
    with HEAVY_LOCK:  # one heavy model at a time
        filled = lamalib.lama_fill(LAMA_MODEL, context, fill_mask, DEVICE, CONFIG["lama_max_side"])
    return np.where(fill_mask[..., None] > 0, filled, bgr)


# ---------------------------------------------------------------------------
# WORKFLOW — shared by the web app and process_folder.py
# ---------------------------------------------------------------------------
def run_workflow(img, data, checks, progress=None):
    """dewarp detection -> dewarping -> shadow detection -> shadow correction
    -> blur detection -> blur correction -> resolution check -> upscaling (Real-ESRGAN)
    -> enhancement (DocRes "appearance", on every image)
    -> damage detection -> inpainting (big-LaMa), the last step.
    Each check runs on the image as corrected by the steps before it.

    img: decoded BGR image, data: the raw file bytes (the blur model uses PIL's loader).
    checks: which of "dewarp", "shadow", "blur", "upscale", "inpaint", "appearance" to run.
    progress(stage, state, text="", hit=False), if given, is called as each step starts and ends:
    stage is one of STAGES' keys, state is "running", "done", "skipped" or "error", hit=True means
    a check found a problem. Returns a dict with "size" (w, h) and "upscaled_to" (w, h or None),
    "dewarp" / "shadow" / "blur" results (None if not checked), "steps" (corrections applied, in
    order), "image" (final BGR image) and, for the blur map, "blur_bgr" + "tiles".
    """
    step = progress or (lambda *args, **kwargs: None)
    ih, iw = img.shape[:2]
    w = {"size": (iw, ih), "upscaled_to": None, "still_below": False, "blur": None, "shadow": None,
         "dewrap": None, "tear": None, "steps": [],
         "blur_bgr": None, "tiles": None}

    # ---- 1. dewarp detection -> dewarping (UVDoc) ----
    if "dewrap" in checks:
        step("dewarp_detect", "running")
        grid = uvdoc_grid(img)
        score = deformation_score(grid)
        d = {"score": score, "detected": score >= float(CONFIG["dewarp_threshold"])}
        w["dewrap"] = d
        step("dewarp_detect", "done", f"{'Warped' if d['detected'] else 'Flat'} (score {score:.3f})",
             hit=d["detected"])
        if d["detected"]:
            step("dewarp_fix", "running")
            img = uvdoc_unwarp(img, grid)
            w["steps"].append("dewarped")
            step("dewarp_fix", "done", "dewarped")
        else:
            step("dewarp_fix", "skipped", "Not needed")
    else:
        step("dewarp_detect", "skipped", "Not selected")
        step("dewarp_fix", "skipped", "Not selected")

    # ---- 2. Shadow detection (on the dewarped image) -> shadow correction ----
    if "shadow" in checks:
        step("shadow_detect", "running")
        prob = predict_shadow(img)
        s = {"prob": prob, "detected": prob >= float(CONFIG["shadow_threshold"])}
        s["confidence"] = prob if s["detected"] else 1.0 - prob
        w["shadow"] = s
        step("shadow_detect", "done", f"{'Shadow found' if s['detected'] else 'No shadow'} ({prob:.2f})",
             hit=s["detected"])
        if s["detected"]:
            step("shadow_fix", "running")
            img = docres_deshadow(img)
            w["steps"].append("Shadow removed")
            step("shadow_fix", "done", "Shadow removed")
        else:
            step("shadow_fix", "skipped", "Not needed")
    else:
        step("shadow_detect", "skipped", "Not selected")
        step("shadow_fix", "skipped", "Not selected")

    # ---- 3. Blur detection (on the dewarped / deshadowed image) -> blur correction ----
    if "blur" in checks:
        threshold = float(CONFIG["blur_threshold"])
        step("blur_detect", "running")
        try:
            if w["steps"]:  # an earlier step changed the image: check the current one
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                pil = Image.open(io.BytesIO(data)).convert("RGB")  # same loader as infer_blur.py
        except Exception as e:
            w["blur"] = {"error": f"Could not read for blur check: {e}"}
            step("blur_detect", "error", "Could not read image")
            step("blur_fix", "skipped", "Not possible")
        else:
            prob, tiles = predict_blur(pil)
            b = {"prob": prob, "detected": prob >= threshold,
                 "tiles": len(tiles), "blurred_tiles": sum(p >= threshold for _, p in tiles)}
            b["confidence"] = prob if b["detected"] else 1.0 - prob
            w["blur"], w["tiles"] = b, tiles
            w["blur_bgr"] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            step("blur_detect", "done", f"{'Blurred' if b['detected'] else 'Sharp'} ({prob:.2f})", hit=b["detected"])
            if b["detected"]:
                step("blur_fix", "running")
                img = docres_deblur(img)
                w["steps"].append("Blur corrected")
                step("blur_fix", "done", "Blur removed")
            else:
                step("blur_fix", "skipped", "Not needed")
    else:
        step("blur_detect", "skipped", "Not selected")
        step("blur_fix", "skipped", "Not selected")

    # ---- 4. Resolution check (on the corrected image) -> upscaling (Real-ESRGAN, ESRGAN_SCALE x) ----
    if "upscale" in checks:
        limit = CONFIG["upscale_below_px"]
        ch, cw = img.shape[:2]
        small = min(cw, ch) < limit  # upscale if either side (width or height) is below the limit
        step("res_check", "done", f"{cw}×{ch} px — " + (f"{'width' if cw < ch else 'height'} {min(cw, ch)} "
             f"below {limit}" if small else f"both sides at least {limit}"), hit=small)
        if small:  # upscale once only; if still below the limit, finish anyway
            step("upscale", "running")
            img = esrgan_upscale(img)
            w["upscaled_to"] = (img.shape[1], img.shape[0])
            w["steps"].append(f"Upscaled {ESRGAN_SCALE}x")
            w["still_below"] = min(w["upscaled_to"]) < limit
            step("upscale", "done", "Upscaled to {}×{} px".format(*w["upscaled_to"])
                 + (f" (still below {limit}, applied once only)" if w["still_below"] else ""))
        else:
            step("upscale", "skipped", "Not needed")
    else:
        step("res_check", "skipped", "Not selected")
        step("upscale", "skipped", "Not selected")

    # ---- 5. Enhancement (DocRes "appearance") - no check, every image gets it ----
    if "appearance" in checks:
        step("appearance", "running")
        img = docres_appearance(img)
        w["steps"].append("Enhanced")
        step("appearance", "done", "Appearance enhanced")
    else:
        step("appearance", "skipped", "Not selected")

    # ---- 6. Damage detection (tear / missing paper) -> inpainting (big-LaMa) - the last step ----
    #        Tears are found against a dark backdrop; if earlier steps (e.g. dewarping) cropped it
    #        away, the check reports "no dark backdrop" and nothing is inpainted.
    if "inpaint" in checks and LAMA_MODEL is not None:
        step("tear_detect", "running")
        mask, damaged, note = tear_mask(img, CONFIG["tear_outline_mode"])
        detected = bool(damaged >= tearlib.TEAR_MIN_FRAC)
        w["tear"] = {"damaged": damaged, "detected": detected, "note": note}
        step("tear_detect", "done",
             note or f"{'Damage found' if detected else 'No damage'} ({damaged * 100:.2f}% of image)",
             hit=detected)
        if detected:
            step("tear_fix", "running")
            img = lama_inpaint(img, mask)
            w["steps"].append("Inpainted")
            step("tear_fix", "done", "Damage inpainted")
        else:
            step("tear_fix", "skipped", "Not needed")
    else:
        reason = "Model missing" if "inpaint" in checks else "Not selected"
        step("tear_detect", "skipped", reason)
        step("tear_fix", "skipped", reason)

    w["image"] = img
    return w


def encode_image(img, ext):
    """BGR image -> file bytes in the format of ext (imencode, so non-ASCII paths are no problem)."""
    if ext.lower() in JPEG2000_EXTS:  # lossless JPEG 2000; .j2k = raw codestream, .jp2 = JP2 container
        out = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
            out, "JPEG2000", no_jp2=ext.lower() == ".j2k")
        return out.getvalue()
    ok, buf = cv2.imencode(ext, img, ENCODE_PARAMS.get(ext.lower(), []))
    if not ok:
        raise ValueError(f"could not encode {ext}")
    return buf.tobytes()


def fmt_prob(result):
    return f"{result['prob']:.3f}" if result and "prob" in result else ""


def report_row(name, error, w, seconds):
    """One report.csv row. w = run_workflow result (None if the image could not be processed)."""
    if error:
        status = f"error: {error}"
    else:
        status = "corrected" if w["steps"] else "no problems (copied)"
        if w["blur"] and "error" in w["blur"]:
            status += f"; blur check failed: {w['blur']['error']}"
    w = w or {"blur": None, "shadow": None, "steps": []}
    resolution = "{}x{}".format(*w["size"]) if w.get("size") else ""
    if w.get("upscaled_to"):
        resolution += " -> {}x{}".format(*w["upscaled_to"])
        if w.get("still_below"):
            resolution += f" (still below {CONFIG['upscale_below_px']}, upscaled once only)"
    return {"file": name, "status": status, "resolution": resolution, "blur_prob": fmt_prob(w["blur"]),
            "shadow_prob": fmt_prob(w["shadow"]),
            "dewarp_score": f"{w['dewarp']['score']:.4f}" if w.get("dewarp") else "",
            "damaged_pct": f"{w['tear']['damaged'] * 100:.2f}" if w.get("tear") else "", "corrections": ", then ".join(w["steps"]),
            "seconds": f"{seconds:.1f}"}


# ---------------------------------------------------------------------------
# Load models once at start-up
# ---------------------------------------------------------------------------
for _key in ("esrgan_model_path", "shadow_model_path", "blur_model_path", "docres_model_path", "uvdoc_model_path"):
    if not os.path.isfile(CONFIG[_key]):
        raise SystemExit(f"ERROR: model not found: {CONFIG[_key]}")

SESSION, INPUT_NAME, IMG_SIZE = create_shadow_session(CONFIG["shadow_model_path"], CONFIG["use_gpu"])
BLUR_MODEL, BLUR_SIZE, BLUR_IDX = load_blur_model(CONFIG["blur_model_path"])
BLUR_TO_TENSOR = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
DOCRES_MODEL = load_docres(CONFIG["docres_code_dir"], CONFIG["docres_model_path"])
ESRGAN_MODEL, ESRGAN_SCALE = load_esrgan(CONFIG["esrgan_model_path"])
UVDOC_MODEL = load_uvdoc(CONFIG["uvdoc_code_dir"], CONFIG["uvdoc_model_path"])
LAMA_MODEL = load_lama(CONFIG["lama_model_path"])
HEAVY_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CONFIG["max_upload_mb"] * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True  # re-read templates/index.html when it changes (no restart needed)


# ---------------------------------------------------------------------------
# Web: every upload is a run (a folder in results/), processed in the background one run
# at a time. The run page polls /status/<run id> to show each image's progress.
# ---------------------------------------------------------------------------
STAGES = [("dewarp_detect", "Dewrap detection"), ("dewarp_fix", "Dewraping"),
          ("shadow_detect", "Shadow detection"), ("shadow_fix", "Shadow removal"),
          ("blur_detect", "Blur detection"), ("blur_fix", "Blur removal"),
          ("res_check", "Resolution check"), ("upscale", f"Upscaling (ESRGAN {ESRGAN_SCALE}×)"),
          ("appearance", "Enhancement (DocRes)"),
          ("tear_detect", "Damage detection"), ("tear_fix", "Inpainting (LaMa)")]
ALL_CHECKS = {"upscale", "blur", "shadow", "dewrap", "inpaint", "appearance"}
JOBS = {}                  # run id -> job; in memory, so live progress is lost if the server restarts
JOBS_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()  # run ids waiting for the worker thread
_worker = None


def safe_relpath(name):
    """Uploaded file name (may contain the folder path) -> safe relative path with '/' separators."""
    parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", p) for p in re.split(r"[\\/]+", name)
             if p not in ("", ".", "..")]
    return "/".join(parts) or "image"


def unique_path(rel, used):
    """Add _2, _3, ... if the same name was uploaded twice."""
    stem, ext, n = rel[:len(rel) - len(Path(rel).suffix)], Path(rel).suffix, 2
    while rel.lower() in used:
        rel, n = f"{stem}_{n}{ext}", n + 1
    used.add(rel.lower())
    return rel


def new_run_dir():
    """Create results/<run id>/output and delete runs older than keep_results_days."""
    root = CONFIG["results_dir"]
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - CONFIG["keep_results_days"] * 86400
    for d in root.iterdir():
        if d.is_dir() and RUN_ID_RE.match(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
    with JOBS_LOCK:  # forget runs whose folder is gone
        for rid in [rid for rid in JOBS if not (root / rid).exists()]:
            del JOBS[rid]
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    (root / run_id / "output").mkdir(parents=True)
    return run_id


def result_url(run_id, name):
    return quote(f"/results/{run_id}/{name}")


def new_item(name, error=None):
    """Progress record of one image, as sent to the page by /status."""
    stage_state = {"state": "skipped", "text": "—"} if error else {"state": "pending", "text": ""}
    return {"name": name, "state": "error" if error else "waiting", "error": error, "outcome": "", "seconds": None,
            "stages": {key: dict(stage_state, hit=False) for key, _ in STAGES}}


def set_stage(item, stage, state, text="", hit=False):
    with JOBS_LOCK:
        item["stages"][stage].update(state=state, text=text, hit=hit)


def process_file(res, src, checks, run_id, index, progress):
    """Process one uploaded image: fill res for the page, write the output file. Returns the workflow result."""
    rel = res["name"]
    data = src.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        res["error"] = "Unreadable image"
        return None

    run_dir = CONFIG["results_dir"] / run_id

    def preview(kind, im):
        """Save a downscaled JPEG for the results page, return its URL."""
        h, w = im.shape[:2]
        scale = CONFIG["display_max_side"] / max(h, w)
        if scale < 1:
            im = cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        name = f"previews/{index}_{kind}.jpg"
        (run_dir / "previews").mkdir(exist_ok=True)
        _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (run_dir / name).write_bytes(buf.tobytes())
        return result_url(run_id, name)

    res["height"], res["width"] = img.shape[:2]
    res["original"] = preview("original", img)

    w = run_workflow(img, data, checks, progress)
    res["blur"], res["shadow"], res["steps"] = w["blur"], w["shadow"], w["steps"]
    res["upscaled_to"], res["still_below"] = w["upscaled_to"], w["still_below"]
    res["dewrap"], res["tear"] = w["dewrap"], w["tear"]
    # ---- Final output: corrected image, or the original file unchanged ----
    dst = run_dir / "output" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if w["steps"]:
        dst.write_bytes(encode_image(w["image"], dst.suffix))
        res["final"] = preview("final", w["image"])
    else:
        dst.write_bytes(data)
    res["download"] = result_url(run_id, f"output/{rel}")
    return w


def summarize(results):
    ok = [r for r in results if "error" not in r]
    return {
        "total": len(results),
        "upscaled": sum(1 for r in ok if "upscale" in r["issues"]),
        "shadow": sum(1 for r in ok if "shadow" in r["issues"]),
        "dewrap": sum(1 for r in ok if "dewrap" in r["issues"]),
        "tear": sum(1 for r in ok if "tear" in r["issues"]),
        "enhanced": sum(1 for r in ok if "Enhanced" in r["steps"]),
        "blur": sum(1 for r in ok if "blur" in r["issues"]),
        "corrected": sum(1 for r in ok if r["steps"]),
        "good": sum(1 for r in ok if not r["issues"]),
        "errors": len(results) - len(ok),
    }


def run_job(job):
    """Process all images of a run (in the worker thread), updating the progress records as it goes."""
    run_id, checks = job["id"], job["checks"]
    run_dir = CONFIG["results_dir"] / run_id
    with JOBS_LOCK:
        job["state"], job["started"] = "running", time.time()

    with open(run_dir / "report.csv", "w", newline="", encoding="utf-8") as fh:
        report = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        report.writeheader()
        for i, item in enumerate(job["items"]):
            res = {"name": item["name"], "upscaled_to": None, "shadow": None, "blur": None, "dewrap": None,
                   "tear": None, "steps": []}
            job["results"].append(res)
            t0, w = time.time(), None
            if item["error"]:  # e.g. unsupported file type, found at upload
                res["error"] = item["error"]
            else:
                with JOBS_LOCK:
                    item["state"] = "processing"
                try:
                    w = process_file(res, run_dir / "input" / item["name"], checks, run_id, i,
                                     lambda *args, **kwargs: set_stage(item, *args, **kwargs))
                except Exception as e:  # e.g. out of memory in DocRes: report it, keep going with the other images
                    res["error"] = f"Processing failed: {e}"
            with JOBS_LOCK:
                item["seconds"] = round(time.time() - t0, 1)
                if "error" in res:
                    item["state"], item["error"] = "error", res["error"]
                    for s in item["stages"].values():
                        if s["state"] in ("pending", "running"):
                            s.update(state="skipped", text="—")
                else:
                    res["issues"] = (["upscale"] if res["upscaled_to"] else []) + \
                        [k for k in ("blur", "shadow", "dewrap", "tear") if res[k] and res[k].get("detected")]
                    item["state"] = "done"
                    item["outcome"] = ", then ".join(res["steps"]) or "No problems"
            report.writerow(report_row(res["name"], res.get("error"), w, time.time() - t0))
            fh.flush()

    # ZIP of the output folder + report. Folder uploads keep their folder name as the ZIP's top level.
    results = job["results"]
    tops = {r["name"].split("/")[0] for r in results}
    folder = tops.pop() if len(tops) == 1 and all("/" in r["name"] for r in results) else None
    zip_name = f"{folder or 'images'}_restored.zip"
    with zipfile.ZipFile(run_dir / zip_name, "w") as zf:  # images are already compressed: store them
        for p in sorted((run_dir / "output").rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir / "output").as_posix())
        zf.write(run_dir / "report.csv", "report.csv", compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(run_dir / "input", ignore_errors=True)  # the output folder and ZIP have everything

    with JOBS_LOCK:
        job["run"] = {"id": run_id, "dir": str(run_dir), "folder": folder,
                      "zip": result_url(run_id, zip_name), "csv": result_url(run_id, "report.csv")}
        job["summary"] = summarize(results)
        job["elapsed"] = time.time() - job["started"]
        job["state"] = "done"


def worker():
    """Runs are processed one after another (DocRes uses all CPU cores)."""
    while True:
        job = JOBS.get(JOB_QUEUE.get())
        if job is None:
            continue
        try:
            run_job(job)
        except Exception as e:
            with JOBS_LOCK:
                job["state"], job["error"] = "failed", f"Run failed: {e}"


def ensure_worker():
    global _worker
    with JOBS_LOCK:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=worker, name="restoration-worker", daemon=True)
            _worker.start()


def job_status(job):
    """Snapshot of a run's progress (the JSON behind the progress table)."""
    with JOBS_LOCK:
        items = copy.deepcopy(job["items"])
        state, error, started = job["state"], job.get("error"), job.get("started")
        elapsed = job.get("elapsed") or (time.time() - started if started else 0)
    return {"id": job["id"], "state": state, "error": error, "total": len(items),
            "done": sum(it["state"] in ("done", "error") for it in items),
            "elapsed": round(elapsed, 1), "items": items}


def render_page(job=None):
    done = job is not None and job["state"] == "done"
    return render_template("index.html", job=job, status=job_status(job) if job else None,
                           stages=STAGES, stage_keys=[k for k, _ in STAGES],
                           results=job["results"] if done else [], summary=job.get("summary") if done else None,
                           run=job.get("run") if done else None, elapsed=job.get("elapsed") if done else None,
                           checks=job["checks"] if job else ALL_CHECKS, upscale_below_px=CONFIG["upscale_below_px"],
                           esrgan_scale=ESRGAN_SCALE, dewarp_threshold=CONFIG["dewarp_threshold"], dewrap_threshold=CONFIG["dewarp_threshold"],
                           shadow_threshold=CONFIG["shadow_threshold"], blur_threshold=CONFIG["blur_threshold"],
                           exts=", ".join(sorted(VALID_EXTS)), exts_list=sorted(VALID_EXTS))


@app.route("/")
def index():
    return render_page()


@app.route("/thumb", methods=["POST"])
def thumb():
    """Small JPEG preview of one uploaded image, for the upload thumbnails of formats browsers
    cannot display themselves (TIFF, JPEG 2000)."""
    f = request.files.get("image")
    if f is None:
        abort(400)
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        abort(415)
    h, w = img.shape[:2]
    scale = 256 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return app.response_class(buf.tobytes(), mimetype="image/jpeg")


@app.route("/start", methods=["POST"])
def start():
    """Save the upload in results/<run id>/input and queue it; the page then opens /run/<run id>."""
    checks = set(request.form.getlist("checks")) & ALL_CHECKS or set(ALL_CHECKS)
    files = [f for f in request.files.getlist("images") if f.filename]
    if not files:
        return {"error": "No images were uploaded."}, 400

    run_id = new_run_dir()
    input_dir = CONFIG["results_dir"] / run_id / "input"
    items, used = [], set()
    for f in files:
        rel = unique_path(safe_relpath(f.filename), used)
        if Path(rel).suffix.lower() not in VALID_EXTS:
            items.append(new_item(rel, "Unsupported file type"))
            continue
        (input_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        f.save(input_dir / rel)
        items.append(new_item(rel))

    job = {"id": run_id, "checks": checks, "state": "queued", "items": items, "results": []}
    with JOBS_LOCK:
        JOBS[run_id] = job
    ensure_worker()
    JOB_QUEUE.put(run_id)
    return {"run_id": run_id, "url": f"/run/{run_id}"}


@app.route("/run/<run_id>")
def run_page(run_id):
    job = JOBS.get(run_id)
    if job is None:
        return (f"Run {run_id} is not known (the server may have been restarted). "
                f"Its files, if any, are in {CONFIG['results_dir'] / run_id}"), 404
    return render_page(job)


@app.route("/status/<run_id>")
def status(run_id):
    job = JOBS.get(run_id)
    if job is None:
        return {"error": "unknown run"}, 404
    return job_status(job)


@app.route("/results/<run_id>/<path:name>")
def result_file(run_id, name):
    if not RUN_ID_RE.match(run_id):
        abort(404)
    # Previews are shown in the page; everything else (output images, report, ZIP) is downloaded
    return send_from_directory(CONFIG["results_dir"] / run_id, name,
                               as_attachment=not name.startswith("previews/"))


@app.errorhandler(413)
def too_large(_):
    return f"Upload too large (limit {CONFIG['max_upload_mb']} MB per request).", 413


def lan_ip():
    """Best guess at this PC's address on the local network."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


if __name__ == "__main__":
    host, port = CONFIG["host"], CONFIG["port"]
    print(f"\nOpen on this PC:       http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"Open from the network: http://{lan_ip()}:{port}")
    print(f"Results are saved in:  {CONFIG['results_dir']}")
    print("Press Ctrl+C to stop.\n")
    try:
        from waitress import serve
    except ImportError:  # fall back to Flask's development server
        app.run(host=host, port=port, debug=False, threaded=True)
    else:
        serve(app, host=host, port=port, threads=8, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
app.config["TEMPLATES_AUTO_RELOAD"] = True  # re-read templates/index.html when it changes (no restart needed)
