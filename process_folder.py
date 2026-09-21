#!/usr/bin/env python3
"""
process_folder.py
============================================================
Run the same workflow as the web app on every image in a folder:

    dewarp detection -> dewarping (UVDoc) -> shadow detection -> shadow correction (DocRes)
    -> blur detection -> blur correction (DocRes)
    -> resolution check -> 4x upscaling (Real-ESRGAN x4plus, once, if width or height is below 1200 px)
    -> enhancement (DocRes "appearance", every image)
    -> damage detection (tear mask) -> inpainting (big-LaMa)   [last]

    python process_folder.py INPUT_FOLDER OUTPUT_FOLDER [--checks dewrap shadow blur upscale inpaint appearance]
                             [--overwrite]

  * Every image is written to OUTPUT_FOLDER with the same name (sub-folders are kept):
    corrected images are saved re-encoded, images without problems are copied unchanged.
  * Images that already have an output are skipped, so an interrupted run can simply be
    started again (use --overwrite to redo them).
  * OUTPUT_FOLDER/report.csv lists, per image, the blur/shadow probabilities and the
    corrections applied. It is written as the run goes.

Thresholds and other settings come from the CONFIG block in app.py.
------------------------------------------------------------
"""

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import app  # loads the blur, shadow and DocRes models


def find_images(input_dir, output_dir):
    out = output_dir.resolve()
    return sorted(p for p in input_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in app.VALID_EXTS
                  and out not in p.resolve().parents)  # output folder may be inside the input folder


def process_one(src, dst, checks):
    """Run the workflow on src and write dst. Returns (error or None, workflow result or None)."""
    data = src.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable image", None

    w = app.run_workflow(img, data, checks)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if w["steps"]:
        dst.write_bytes(app.encode_image(w["image"], dst.suffix))
    else:
        shutil.copy2(src, dst)
    return None, w


def main():
    ap = argparse.ArgumentParser(description="Fix blur and shadows in every image of a folder.")
    ap.add_argument("input_dir", type=Path, help="folder with the images to process")
    ap.add_argument("output_dir", type=Path, help="folder for the results (created if needed)")
    ap.add_argument("--checks", nargs="+", choices=["upscale", "blur", "shadow", "dewrap", "inpaint", "appearance"],
                    default=["upscale", "blur", "shadow", "dewrap", "inpaint", "appearance"],
                    help="which checks/corrections to run (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="re-process images that already have an output")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"ERROR: input folder not found: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        sys.exit("ERROR: the output folder must be different from the input folder")

    files = find_images(args.input_dir, args.output_dir)
    if not files:
        sys.exit(f"No images ({', '.join(sorted(app.VALID_EXTS))}) found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = set(args.checks)
    print(f"\n{len(files)} image(s) in {args.input_dir} -> {args.output_dir}  (checks: {', '.join(sorted(checks))})\n")

    report_path = args.output_dir / "report.csv"
    previous = {}
    if report_path.exists():  # rows from earlier runs, kept for the images skipped this time
        with open(report_path, newline="", encoding="utf-8") as fh:
            previous = {row["file"]: row for row in csv.DictReader(fh)}

    counts = {}
    t_start = time.time()
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        report = csv.DictWriter(fh, fieldnames=app.REPORT_FIELDS, extrasaction="ignore")
        report.writeheader()
        for i, src in enumerate(files, 1):
            rel = str(src.relative_to(args.input_dir))
            dst = args.output_dir / rel
            if dst.exists() and not args.overwrite:
                row = previous.get(rel) or {"status": "skipped (output exists)"}
                kind, detail = "skipped", "already done (output exists)"
            else:
                t0 = time.time()
                try:
                    error, w = process_one(src, dst, checks)
                except Exception as e:  # e.g. out of memory in DocRes: note it and carry on
                    error, w = str(e), None
                row = app.report_row(rel, error, w, time.time() - t0)
                kind = next((k for k in ("error", "corrected") if row["status"].startswith(k)), "unchanged")
                detail = f"{row.get('corrections') or row['status']}  ({row['seconds']} s)"
            row["file"] = rel
            report.writerow(row)
            fh.flush()
            counts[kind] = counts.get(kind, 0) + 1
            print(f"[{i}/{len(files)}] {rel}: {detail}")

    print(f"\nDone in {time.time() - t_start:.0f} s: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    print(f"Report: {args.output_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
