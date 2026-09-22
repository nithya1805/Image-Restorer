#!/usr/bin/env python3
"""
api_docs.py
============================================================
Swagger UI for the endpoints the app already has, so they can be tried out in
a browser:

    /docs          Swagger UI (try the endpoints here)
    /openapi.json  the OpenAPI 3 description

It is registered from app.py:

    from api_docs import docs_bp
    app.register_blueprint(docs_bp)

Nothing else in the app is changed. Swagger UI itself is loaded from a CDN, so
the browser opening /docs needs internet access (the app does not).
------------------------------------------------------------
"""

from flask import Blueprint, jsonify

docs_bp = Blueprint("docs", __name__)

SWAGGER_UI_VERSION = "5.17.14"

OPENAPI = {
    "openapi": "3.0.3",
    "info": {
        "title": "Image Restorer API",
        "version": "1.0.0",
        "description": (
            "Two ways to use the restorer.\n\n"
            "**One image:** `POST /api/v1/images` processes it and returns the restored file "
            "in the same response.\n\n"
            "**A folder:** zip it, `POST /api/v1/batches`, then poll "
            "`GET /api/v1/batches/{batch_id}` and download the result when it is done. "
            "Batches run one at a time, image by image.\n\n"
            "Which corrections are applied is decided by the models; `checks` only limits "
            "which detectors may run (omit it for all of them)."
        ),
    },
    "tags": [
        {"name": "single image", "description": "Process one image and get it back"},
        {"name": "batch", "description": "Process a folder of images"},
    ],
    "paths": {
        "/api/v1/images": {
            "post": {
                "tags": ["single image"],
                "summary": "Process one image and download the result",
                "description": (
                    "Processes ONE image and returns the restored file in the same response, so it can "
                    "be downloaded straight away. Same formats and steps as a batch run; the result "
                    "keeps the uploaded file format.\n\n"
                    "**This call blocks until the image is finished** (roughly 5-75 s, depending on the "
                    "image and which corrections are needed), so allow a long client timeout. If a batch "
                    "is running, this waits its turn for the GPU.\n\n"
                    "What was done comes back in the response headers:\n"
                    "* `X-Corrections` - corrections applied, e.g. `Shadow removed, Enhanced` (or `none`)\n"
                    "* `X-Steps-Detail` - every step and its result\n"
                    "* `X-Resolution` - size, and the new size if it was upscaled\n"
                    "* `X-Seconds` - processing time"
                ),
                "requestBody": {
                    "required": True,
                    "content": {"multipart/form-data": {"schema": {
                        "type": "object",
                        "properties": {
                            "image": {
                                "type": "string", "format": "binary",
                                "description": "One image (.jpg .jpeg .png .bmp .tif .tiff .webp .j2k .jp2)",
                            },
                            "checks": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["dewrap", "shadow", "blur", "upscale", "inpaint", "appearance"],
                                },
                                "description": "Steps to run. Omit for all of them.",
                            },
                        },
                        "required": ["image"],
                    }}},
                },
                "responses": {
                    "200": {
                        "description": "The restored image (same format as the upload)",
                        "headers": {
                            "Content-Disposition": {"schema": {"type": "string"},
                                                    "description": 'attachment; filename="<name>_restored.<ext>"'},
                            "X-Corrections": {"schema": {"type": "string"}, "description": "Shadow removed, Enhanced"},
                            "X-Steps-Detail": {"schema": {"type": "string"}},
                            "X-Resolution": {"schema": {"type": "string"}, "description": "1000x1329 -> 2000x2658"},
                            "X-Seconds": {"schema": {"type": "string"}},
                        },
                        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
                    },
                    "400": {"description": "No image uploaded"},
                    "415": {"description": "Unsupported file type, or the image could not be read"},
                    "500": {"description": "Processing failed"},
                },
            }
        },
        "/api/v1/batches": {
            "post": {
                "tags": ["batch"],
                "summary": "Process a folder of images",
                "description": (
                    "Processes a whole FOLDER. Every image in it, sub-folders included, is processed one "
                    "by one in the background; you do not list the images yourself. For a single image use "
                    "`POST /api/v1/images`.\n\n"
                    "Give the folder either way:\n"
                    "* **folder** - upload the files of an unzipped folder, or a `.zip` archive of the folder.\n"
                    "* **folder_path** - a folder on the machine running the app, e.g. "
                    "`C:\\scans\\batch1`. Nothing is uploaded; the app reads the folder itself. Handy "
                    "for large batches and for scheduled jobs.\n\n"
                    "Returns at once with a `batch_id`. Follow it with `GET /api/v1/batches/{batch_id}`, "
                    "then download from `GET /api/v1/batches/{batch_id}/download`. Non-image files are "
                    "ignored. Roughly 30-75 s per image."
                ),
                "requestBody": {
                    "required": True,
                    "content": {"multipart/form-data": {"schema": {
                        "type": "object",
                        "properties": {
                            "folder": {"type": "string", "format": "binary",
                                       "description": "The folder as a .zip file"},
                            "folder_path": {"type": "string",
                                            "example": "C:\\scans\\batch1",
                                            "description": "or a folder on the machine running the app"},
                            "checks": {"type": "array",
                                       "items": {"type": "string",
                                                 "enum": ["dewrap", "shadow", "blur", "upscale", "inpaint",
                                                          "appearance"]},
                                       "description": "Steps to run. Omit for all of them."},
                        },
                    }}},
                },
                "responses": {
                    "200": {"description": "Batch queued", "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "batch_id": {"type": "string", "example": "20260921-161207-ff1529"},
                            "source": {"type": "string", "example": "C:\\scans\\batch1"},
                            "total": {"type": "integer", "example": 100},
                            "checks": {"type": "array", "items": {"type": "string"}},
                            "status_url": {"type": "string"},
                            "download_url": {"type": "string"},
                            "report_url": {"type": "string"},
                        }}}}},
                    "400": {"description": "No folder given, or no images in it"},
                    "404": {"description": "folder_path does not exist on the server"},
                    "415": {"description": "The uploaded file is not a .zip"},
                    "413": {"description": "Upload larger than max_upload_mb"},
                },
            }
        },

        "/api/v1/batches/{batch_id}": {
            "get": {
                "tags": ["batch"],
                "summary": "Progress of a batch",
                "description": (
                    "How many images are processed and how many are left, plus the result of each image.\n\n"
                    "`state` is `queued`, `running`, `done` or `failed`. Add `?detail=full` for every step "
                    "of every image."
                ),
                "parameters": [
                    {"name": "batch_id", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "detail", "in": "query", "required": False,
                     "schema": {"type": "string", "enum": ["full"]},
                     "description": "full = include every step of every image"},
                ],
                "responses": {
                    "200": {"description": "Progress", "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "batch_id": {"type": "string"},
                            "state": {"type": "string", "enum": ["queued", "running", "done", "failed"]},
                            "total": {"type": "integer", "example": 100},
                            "processed": {"type": "integer", "example": 42},
                            "remaining": {"type": "integer", "example": 58},
                            "failed": {"type": "integer", "example": 1},
                            "corrected": {"type": "integer", "example": 33},
                            "elapsed_seconds": {"type": "number", "example": 1274.5},
                            "current": {"type": "string", "nullable": True, "example": "scans/page43.jpg"},
                            "download_url": {"type": "string"},
                            "images": {"type": "array", "items": {"type": "object", "properties": {
                                "name": {"type": "string", "example": "scans/page1.jpg"},
                                "state": {"type": "string", "enum": ["waiting", "processing", "done", "error"]},
                                "result": {"type": "string", "example": "Shadow removed, then Enhanced"},
                                "seconds": {"type": "number", "nullable": True},
                            }}},
                        }}}}},
                    "404": {"description": "Unknown batch id (or the app restarted)"},
                },
            }
        },
        "/api/v1/batches/{batch_id}/download": {
            "get": {
                "tags": ["batch"],
                "summary": "Download the processed folder",
                "description": ("The processed images as a ZIP, with the same folder structure and "
                                "report.csv inside. Available once the batch is finished."),
                "parameters": [{"name": "batch_id", "in": "path", "required": True,
                                "schema": {"type": "string"}}],
                "responses": {
                    "200": {"description": "ZIP of the processed images",
                            "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}},
                    "404": {"description": "Unknown batch id, or no result file"},
                    "409": {"description": "The batch has not finished yet"},
                },
            }
        },
        "/api/v1/batches/{batch_id}/report": {
            "get": {
                "tags": ["batch"],
                "summary": "Download the report (CSV)",
                "description": ("One row per image: resolution, blur and shadow probabilities, dewarp score, "
                                "damaged percentage, corrections applied and seconds taken."),
                "parameters": [{"name": "batch_id", "in": "path", "required": True,
                                "schema": {"type": "string"}}],
                "responses": {
                    "200": {"description": "report.csv", "content": {"text/csv": {"schema": {"type": "string"}}}},
                    "404": {"description": "Unknown batch id, or the report is not written yet"},
                },
            }
        },
    },
}


SWAGGER_PAGE = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Restorer API</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui.css">
  <style>body {{ margin: 0; }} .topbar {{ display: none; }}</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      deepLinking: true,
      tryItOutEnabled: true,
      displayRequestDuration: true,
      defaultModelsExpandDepth: -1,
    }});
  </script>
</body>
</html>
"""


@docs_bp.route("/openapi.json")
def openapi_json():
    return jsonify(OPENAPI)


@docs_bp.route("/docs")
def swagger_ui():
    return SWAGGER_PAGE
