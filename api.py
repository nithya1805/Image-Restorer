#!/usr/bin/env python3
"""
api.py
============================================================
The Image Restorer services as a FastAPI app.

    POST /api/v1/images                         one image in, restored image out
    POST /api/v1/batches                        process a folder (zip upload, or a folder on this PC)
    GET  /api/v1/batches/{batch_id}             progress: processed / remaining / per image
    GET  /api/v1/batches/{batch_id}/download    the processed folder as a ZIP
    GET  /api/v1/batches/{batch_id}/report      report.csv

The restoration pipeline, the models and the background worker all come from app.py; this file only
exposes them over HTTP. The Flask web page is mounted at / so one process serves both, and the
models are loaded once (they share the GPU).

    /docs        Swagger UI, generated from the code
    /redoc       the same API as ReDoc
    /openapi.json

RUN:
    .venv\\Scripts\\python -m uvicorn api:api --host 0.0.0.0 --port 8000
(then the page is at http://127.0.0.1:8000/ and the API docs at http://127.0.0.1:8000/docs)
------------------------------------------------------------
"""

import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from a2wsgi import WSGIMiddleware
from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, Query, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import app as core  # the pipeline, the models and the run queue (loads the models)
import batch_db    # batch / image records (SQLite, next to the results)

batch_db.init(core.CONFIG["results_dir"] / "batches.db")
for _recovered in batch_db.recover_interrupted():   # a restart leaves nothing running
    print(f"batch {_recovered}: was interrupted by a restart, closed off from what finished")

api = FastAPI(
    title="Image Restorer API",
    version="1.0.0",
    description=(
        "Restores document images: warping, shadows, blur, low resolution, appearance and tears. "
        "Each step has its own detector, and a correction only runs when its detector finds the "
        "problem.\n\n"
        "**One image:** `POST /api/v1/images` returns the restored file in the same response.\n\n"
        "**A folder:** `POST /api/v1/batches` returns a `batch_id`; poll "
        "`GET /api/v1/batches/{batch_id}` and download the result when it is done. Batches run one "
        "at a time, image by image, on the GPU.\n\n"
        "Every image goes through every check, and a correction runs only when its detector finds "
        "the problem - the same behaviour as the web page."
    ),
    openapi_tags=[
        {"name": "single image", "description": "Process one image and get it back"},
        {"name": "batch", "description": "Process a folder of images"},
    ],
)


# ─── Swagger: show file fields as file pickers ────────────────────────────────
# FastAPI writes OpenAPI 3.1, where an upload is `contentMediaType`. The Swagger UI bundled here
# only knows the older `format: binary`, and without it draws a text box instead of "Choose Files"
# (and then posts an empty string). Rewriting that one keyword in the generated spec fixes it.

def as_binary(node):
    if isinstance(node, dict):
        if node.pop("contentMediaType", None) is not None and node.get("type") == "string":
            node["format"] = "binary"
        for value in node.values():
            as_binary(value)
    elif isinstance(node, list):
        for value in node:
            as_binary(value)
    return node


def openapi_with_file_pickers():
    if not api.openapi_schema:
        api.openapi_schema = as_binary(get_openapi(
            title=api.title, version=api.version, description=api.description,
            tags=api.openapi_tags, routes=api.routes))
    return api.openapi_schema


api.openapi = openapi_with_file_pickers


# ─── models for the responses (these become the Swagger schemas) ──────────────

class BatchCreated(BaseModel):
    """What POST /api/v1/batches returns, straight after the batch is queued."""
    batch_id: str = Field(examples=["BATCH-20260922-001"])
    batch_name: str | None = Field(None, examples=["Grave Images"])
    status: str = Field(examples=["PROCESSING"])
    total_images: int = Field(examples=[100])
    completed_images: int = 0
    failed_images: int = 0
    pending_images: int = Field(examples=[100])
    skipped_files: int = Field(0, description="files in the folder that were not images")
    status_url: str
    download_url: str
    message: str = Field(examples=["Batch created and processing started"])


class ImageStatus(BaseModel):
    """One image of a batch."""
    image_id: int
    filename: str = Field(examples=["old_records/image002.jpg"])
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    output_available: bool
    error_message: str | None = None


class BatchStatus(BaseModel):
    """What GET /api/v1/batches/{batch_id} returns."""
    batch_id: str = Field(examples=["BATCH-20260922-001"])
    batch_name: str | None = None
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "PARTIALLY_COMPLETED", "FAILED"]
    total_images: int
    completed_images: int
    failed_images: int
    pending_images: int
    processing_images: int
    progress_percentage: int = Field(examples=[67], description="finished (completed + failed) of total")
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    images: list[ImageStatus]


class BatchSummary(BaseModel):
    """One row of the batch list."""
    batch_id: str
    batch_name: str | None = None
    status: str
    total_images: int
    completed_images: int
    failed_images: int
    pending_images: int
    created_at: str | None = None
    completed_at: str | None = None


# ─── helpers ─────────────────────────────────────────────────────────────────

def ascii_header(value: str, limit: int = 800) -> str:
    """HTTP headers must be plain ASCII; the step texts contain - and x."""
    value = value.replace("\u2014", "-").replace("\u2013", "-").replace("\u00d7", "x")
    value = value.encode("ascii", "replace").decode("ascii")
    return value[:limit - 3] + "..." if len(value) > limit else value


def job_or_404(batch_id: str):
    job = core.JOBS.get(batch_id)
    if job is None:
        raise HTTPException(404, f"unknown batch id '{batch_id}' (the app may have been restarted)")
    return job


# ─── 1. one image ────────────────────────────────────────────────────────────

@api.post(
    "/api/v1/images",
    tags=["single image"],
    summary="Process one image and download the result",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}},
                     "description": "The restored image, in the format it was uploaded in"}},
)
def process_image(
    image: Annotated[UploadFile, File(description="One image (.jpg .jpeg .png .bmp .tif .tiff .webp .j2k .jp2)")],
):
    """Processes ONE image and returns the restored file in the same response.

    Every check runs - warping, shadow, blur, resolution and damage - and each model decides for
    itself whether its correction is needed, exactly as the web page does.

    The call blocks until the image is finished (about 5-75 s, depending on the image and the
    corrections needed), so allow a long client timeout; if a batch is running it waits its turn
    for the GPU. What was done comes back in the headers **X-Corrections**, **X-Steps-Detail**,
    **X-Resolution** and **X-Seconds**.
    """
    ext = Path(image.filename or "").suffix.lower()
    if ext not in core.VALID_EXTS:
        raise HTTPException(415, f"unsupported file type '{ext}'; use one of {sorted(core.VALID_EXTS)}")

    data = image.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "could not read the image")

    detail: list[str] = []
    t0 = time.time()
    try:
        w = core.run_workflow(
            img, data, set(core.ALL_CHECKS),   # every detector runs; the models decide what to correct
            lambda stage, state, text="", hit=False:
            state != "running" and detail.append(f"{stage}={state}" + (f":{text}" if text else "")))
        out = core.encode_image(w["image"], ext)
    except Exception as e:
        raise HTTPException(500, f"processing failed: {e}")

    resolution = "{}x{}".format(*w["size"]) + (" -> {}x{}".format(*w["upscaled_to"]) if w["upscaled_to"] else "")
    stem = Path(image.filename or "image").stem
    return Response(
        out,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{ascii_header(stem, 100)}_restored{ext}"',
            "X-Corrections": ascii_header(", ".join(w["steps"]) or "none"),
            "X-Steps-Detail": ascii_header(" | ".join(detail)),
            "X-Resolution": ascii_header(resolution),
            "X-Seconds": f"{time.time() - t0:.1f}",
        },
    )


# ─── 2. batch: upload a folder, process it in the background ─────────────────

CHUNK = 1024 * 1024          # uploads are streamed to disk a megabyte at a time


def safe_relative(name: str) -> str:
    """The path a browser sent for a file in a folder -> a safe relative path.
    Drops drive letters, leading slashes and any '..', so nothing can escape the batch folder."""
    parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)
             for part in re.split(r"[\\/]+", name) if part not in ("", ".", "..")]
    return "/".join(parts) or "image"


def save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream one uploaded file to disk (never holds a whole image in memory)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while chunk := upload.file.read(CHUNK):
            out.write(chunk)


def watch_batch(batch_id: str, run_id: str) -> None:
    """Mirror the worker's progress into the database until the run is finished.
    The processing itself is untouched - this only records what the worker is doing."""
    seen: dict[str, str] = {}
    out_dir = core.CONFIG["results_dir"] / run_id / "output"
    while True:
        job = core.JOBS.get(run_id)
        if job is None:
            batch_db.finish_batch(batch_id, "run disappeared")
            return
        state = core.job_status(job)
        for item in state["items"]:
            status = {"waiting": batch_db.PENDING, "processing": batch_db.PROCESSING,
                      "done": batch_db.COMPLETED, "error": batch_db.FAILED}[item["state"]]
            if seen.get(item["name"]) == status:
                continue
            seen[item["name"]] = status
            out = out_dir / item["name"]
            batch_db.set_image_status(
                batch_id, item["name"], status,
                output_path=str(out) if status == batch_db.COMPLETED and out.is_file() else None,
                error_message=item["error"] if status == batch_db.FAILED else None)
        if state["state"] in ("done", "failed"):
            batch_db.finish_batch(batch_id, state.get("error"))
            return
        time.sleep(0.4)


@api.post("/api/v1/batches", tags=["batch"], summary="Upload a folder and start processing",
          response_model=BatchCreated, status_code=202)
def create_batch(
    files: Annotated[list[UploadFile],
                     File(description="The images of the selected folder, as a browser folder picker "
                                      "sends them (<input type=\"file\" webkitdirectory multiple>). "
                                      "Each file keeps its path inside the folder, e.g. "
                                      "old_records/page2.jpg")],
    batch_name: Annotated[str | None, Form(description="A name for this batch, e.g. Grave Images")] = None,
):
    """Takes the images of one folder, creates a batch and **starts processing in the background**.

    Returns straight away with the `batch_id`; nothing is processed inside this request. Every image
    gets a record with status `PENDING`, and the worker moves each one to `PROCESSING` and then
    `COMPLETED` or `FAILED`. Sub-folders are kept, so `old_records/page2.jpg` stays where it was and
    files with the same name in different folders do not clash. Unsupported files (.txt, .pdf,
    .docx, .json, ...) are ignored and never fail the batch.

    Follow the batch with `GET /api/v1/batches/{batch_id}` and fetch the result from
    `GET /api/v1/batches/{batch_id}/download`.

    In Swagger, "Choose Files" can only select files, not a folder - browsers do not allow a folder
    picker there. Select the folder's images, or use the app's own page, which has a real
    "Choose a folder" button.
    """
    uploads = [f for f in (files or []) if f.filename]
    if not uploads:
        raise HTTPException(400, "no images given: choose a folder in the frontend (field 'files')")

    batch_id = batch_db.new_batch_id()
    run_dir = core.CONFIG["results_dir"] / batch_id
    input_dir = run_dir / "input"
    (run_dir / "output").mkdir(parents=True, exist_ok=True)

    saved: list[tuple[str, str]] = []            # (name inside the folder, path on disk)
    used: set[str] = set()
    skipped: list[str] = []

    for upload in uploads:
        rel = safe_relative(upload.filename)
        if Path(rel).suffix.lower() not in core.VALID_EXTS:
            skipped.append(rel)                   # .txt, .pdf, .json, ... are ignored, never an error
            continue
        rel = core.unique_path(rel, used)         # the same name twice -> page.jpg, page_2.jpg
        save_upload(upload, input_dir / rel)
        saved.append((rel, str(input_dir / rel)))
    source = next((Path(u.filename).parts[0] for u in uploads if len(Path(u.filename).parts) > 1), "")

    if not saved:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, f"no supported images in the folder (supported: "
                                 f"{', '.join(sorted(core.VALID_EXTS))}; ignored {len(skipped)} other files)")

    batch_db.create_batch(batch_id, batch_name or source or batch_id, saved)

    # hand the images to the worker that already exists in app.py, then follow it in the database
    job = {"id": batch_id, "checks": set(core.ALL_CHECKS), "state": "queued",
           "items": [core.new_item(name) for name, _ in saved], "results": []}
    with core.JOBS_LOCK:
        core.JOBS[batch_id] = job
    core.ensure_worker()
    core.JOB_QUEUE.put(batch_id)
    batch_db.start_batch(batch_id)
    threading.Thread(target=watch_batch, args=(batch_id, batch_id), daemon=True,
                     name=f"watch-{batch_id}").start()

    row = batch_db.get_batch(batch_id)
    return BatchCreated(
        batch_id=batch_id, batch_name=row["batch_name"], status=batch_db.PROCESSING,
        total_images=row["total_images"], completed_images=0, failed_images=0,
        pending_images=row["total_images"], skipped_files=len(skipped),
        status_url=f"/api/v1/batches/{batch_id}", download_url=f"/api/v1/batches/{batch_id}/download",
        message="Batch created and processing started")


# ─── 3. batch progress ───────────────────────────────────────────────────────

@api.get("/api/v1/batches/{batch_id}", tags=["batch"], summary="Batch and per-image status",
         response_model=BatchStatus)
def batch_status(batch_id: Annotated[str, PathParam(examples=["BATCH-20260922-001"])]):
    """The batch's progress and the status of every image in it, straight from the database.

    Statuses are `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED` per image, and `PENDING`,
    `PROCESSING`, `COMPLETED`, `PARTIALLY_COMPLETED`, `FAILED` for the batch. Poll this while the
    batch runs to show progress; `output_available` says whether that image has a restored file.
    """
    row = batch_db.get_batch(batch_id)
    if row is None:
        raise HTTPException(404, f"unknown batch_id '{batch_id}'")
    images = batch_db.get_images(batch_id)
    processing = sum(1 for i in images if i["status"] == batch_db.PROCESSING)
    total = row["total_images"] or 1
    return BatchStatus(
        batch_id=batch_id, batch_name=row["batch_name"], status=row["status"],
        total_images=row["total_images"], completed_images=row["completed_images"],
        failed_images=row["failed_images"], pending_images=row["pending_images"],
        processing_images=processing,
        progress_percentage=round(100 * (row["completed_images"] + row["failed_images"]) / total),
        created_at=row["created_at"], started_at=row["started_at"], completed_at=row["completed_at"],
        error_message=row["error_message"],
        images=[ImageStatus(image_id=i["id"], filename=i["filename"], status=i["status"],
                            output_available=bool(i["output_path"]),
                            error_message=i["error_message"]) for i in images])


# ─── 4. download the batch output ────────────────────────────────────────────

@api.get("/api/v1/batches/{batch_id}/download", tags=["batch"],
         summary="Download the restored images as a ZIP",
         response_class=FileResponse,
         responses={200: {"content": {"application/zip": {}}, "description": "ZIP of the restored images"}})
def download_batch(batch_id: str):
    """The restored images of a finished batch, as `<batch_id>.zip`.

    Failed images are not in the ZIP; `failed_images.txt` inside it lists them with their error, so
    nothing is lost silently. Available once the batch has finished (409 while it is still running).
    """
    row = batch_db.get_batch(batch_id)
    if row is None:
        raise HTTPException(404, f"unknown batch_id '{batch_id}'")
    if row["status"] in (batch_db.PENDING, batch_db.PROCESSING):
        raise HTTPException(409, f"batch is {row['status']}; {row['completed_images']} of "
                                 f"{row['total_images']} images are done")
    if not row["completed_images"]:
        raise HTTPException(404, "no images in this batch were processed successfully")

    images = batch_db.get_images(batch_id)
    zip_path = core.CONFIG["results_dir"] / batch_id / f"{batch_id}.zip"
    if not zip_path.is_file():                   # built once, then served from disk
        with zipfile.ZipFile(zip_path, "w") as zf:
            for image in images:
                if image["status"] != batch_db.COMPLETED or not image["output_path"]:
                    continue
                out = Path(image["output_path"])
                if out.is_file():
                    rel = Path(image["filename"])
                    zf.write(out, f"{batch_id}/{rel.with_name(rel.stem + '_restored' + rel.suffix).as_posix()}")
            failed = [i for i in images if i["status"] == batch_db.FAILED]
            if failed:
                zf.writestr(f"{batch_id}/failed_images.txt",
                            "\n".join(f"{i['filename']}: {i['error_message'] or 'failed'}" for i in failed))
    return FileResponse(zip_path, media_type="application/zip", filename=f"{batch_id}.zip")


@api.get("/api/v1/batches", tags=["batch"], summary="Recent batches",
         response_model=list[BatchSummary])
def list_batches(limit: Annotated[int, Query(ge=1, le=200)] = 50):
    """The most recent batches, newest first - for a batch list in the frontend."""
    return [BatchSummary(**{k: r[k] for k in
                           ("batch_id", "batch_name", "status", "total_images", "completed_images",
                            "failed_images", "pending_images", "created_at", "completed_at")})
            for r in batch_db.list_batches(limit)]


# ─── the existing web page, served by the same process ───────────────────────
# Mounted last so the API routes above win; everything else goes to Flask.
api.mount("/", WSGIMiddleware(core.app.wsgi_app))


if __name__ == "__main__":
    import uvicorn

    host, port = core.CONFIG["host"], 8000
    print(f"\nWeb page:  http://127.0.0.1:{port}/")
    print(f"API docs:  http://127.0.0.1:{port}/docs\n")
    uvicorn.run(api, host=host, port=port)
