#!/usr/bin/env python3
"""
batch_db.py
============================================================
The batch and per-image records behind the batch API, kept in SQLite (standard library, one file,
no server to run). The project had no database before this; image processing itself is unchanged -
the pipeline, the models and the worker thread all stay in app.py.

    batches        one row per uploaded folder     BATCH-20260922-001
    batch_images   one row per image in a batch    PENDING -> PROCESSING -> COMPLETED / FAILED

The file lives next to the results, so the records and the images are kept together:
    results/batches.db

Every function opens its own short-lived connection, which keeps it safe to call from the web
thread and from the background worker at the same time.
------------------------------------------------------------
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

# batch statuses
PENDING, PROCESSING, COMPLETED, PARTIALLY_COMPLETED, FAILED = (
    "PENDING", "PROCESSING", "COMPLETED", "PARTIALLY_COMPLETED", "FAILED")

_DB_PATH: Path | None = None
_ID_LOCK = threading.Lock()          # so two uploads at once cannot take the same batch number

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id       TEXT    NOT NULL UNIQUE,
    batch_name     TEXT,
    status         TEXT    NOT NULL,
    total_images   INTEGER NOT NULL DEFAULT 0,
    completed_images INTEGER NOT NULL DEFAULT 0,
    failed_images  INTEGER NOT NULL DEFAULT 0,
    pending_images INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT    NOT NULL,
    started_at     TEXT,
    completed_at   TEXT,
    error_message  TEXT
);
CREATE TABLE IF NOT EXISTS batch_images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT    NOT NULL REFERENCES batches(batch_id),
    filename      TEXT    NOT NULL,
    input_path    TEXT    NOT NULL,
    output_path   TEXT,
    status        TEXT    NOT NULL,
    error_message TEXT,
    created_at    TEXT    NOT NULL,
    started_at    TEXT,
    completed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_batch_images_batch ON batch_images(batch_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init(db_path: Path) -> None:
    """Create the database file and tables if they are not there yet."""
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript(SCHEMA)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")      # readers do not block the worker's writes
    con.execute("PRAGMA foreign_keys=ON")
    return con


# ─── writing ─────────────────────────────────────────────────────────────────

def new_batch_id() -> str:
    """BATCH-<date>-<number>, the number counting that day's batches: BATCH-20260922-001."""
    with _ID_LOCK, connect() as con:
        day = datetime.now().strftime("%Y%m%d")
        used = con.execute("SELECT COUNT(*) FROM batches WHERE batch_id LIKE ?", (f"BATCH-{day}-%",)).fetchone()[0]
        while True:
            candidate = f"BATCH-{day}-{used + 1:03d}"
            if con.execute("SELECT 1 FROM batches WHERE batch_id = ?", (candidate,)).fetchone() is None:
                return candidate
            used += 1


def create_batch(batch_id: str, batch_name: str | None, images: list[tuple[str, str]]) -> None:
    """One transaction: the batch row and a PENDING row per image.
    images: (filename with its folder path, input_path on disk) pairs."""
    stamp = now()
    with connect() as con, con:                 # `with con` = commit, or roll back on error
        con.execute(
            "INSERT INTO batches (batch_id, batch_name, status, total_images, completed_images,"
            " failed_images, pending_images, created_at) VALUES (?, ?, ?, ?, 0, 0, ?, ?)",
            (batch_id, batch_name, PENDING, len(images), len(images), stamp))
        con.executemany(
            "INSERT INTO batch_images (batch_id, filename, input_path, status, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [(batch_id, name, path, PENDING, stamp) for name, path in images])


def start_batch(batch_id: str) -> None:
    with connect() as con, con:
        con.execute("UPDATE batches SET status = ?, started_at = COALESCE(started_at, ?) WHERE batch_id = ?",
                    (PROCESSING, now(), batch_id))


def set_image_status(batch_id: str, filename: str, status: str,
                     output_path: str | None = None, error_message: str | None = None) -> None:
    """Move one image on, and keep the batch's counters in step - all in one transaction."""
    stamp = now()
    with connect() as con, con:
        row = con.execute("SELECT status FROM batch_images WHERE batch_id = ? AND filename = ?",
                          (batch_id, filename)).fetchone()
        if row is None or row["status"] == status:
            return
        con.execute(
            "UPDATE batch_images SET status = ?,"
            " started_at = CASE WHEN ? = 'PROCESSING' THEN ? ELSE started_at END,"
            " completed_at = CASE WHEN ? IN ('COMPLETED', 'FAILED') THEN ? ELSE completed_at END,"
            " output_path = COALESCE(?, output_path), error_message = COALESCE(?, error_message)"
            " WHERE batch_id = ? AND filename = ?",
            (status, status, stamp, status, stamp, output_path, error_message, batch_id, filename))
        _recount(con, batch_id)


def finish_batch(batch_id: str, error_message: str | None = None) -> str:
    """Set the final batch status from how its images ended up. Returns that status."""
    with connect() as con, con:
        _recount(con, batch_id)
        r = con.execute("SELECT total_images, completed_images, failed_images FROM batches WHERE batch_id = ?",
                        (batch_id,)).fetchone()
        if r is None:
            return FAILED
        done, failed, total = r["completed_images"], r["failed_images"], r["total_images"]
        status = (FAILED if done == 0 and failed else
                  COMPLETED if failed == 0 and done == total else
                  PARTIALLY_COMPLETED if done else FAILED)
        con.execute("UPDATE batches SET status = ?, completed_at = ?, error_message = COALESCE(?, error_message)"
                    " WHERE batch_id = ?", (status, now(), error_message, batch_id))
        return status


def _recount(con: sqlite3.Connection, batch_id: str) -> None:
    con.execute(
        "UPDATE batches SET"
        " completed_images = (SELECT COUNT(*) FROM batch_images WHERE batch_id = :b AND status = 'COMPLETED'),"
        " failed_images    = (SELECT COUNT(*) FROM batch_images WHERE batch_id = :b AND status = 'FAILED'),"
        " pending_images   = (SELECT COUNT(*) FROM batch_images WHERE batch_id = :b AND status = 'PENDING')"
        " WHERE batch_id = :b", {"b": batch_id})


def recover_interrupted() -> list[str]:
    """After a restart: nothing is processing any more. Images left mid-flight go back to PENDING and
    their batch is closed off from what actually finished. Returns the batch ids that were touched."""
    with connect() as con, con:
        rows = con.execute("SELECT batch_id FROM batches WHERE status IN (?, ?)", (PENDING, PROCESSING)).fetchall()
        ids = [r["batch_id"] for r in rows]
        for batch_id in ids:
            con.execute("UPDATE batch_images SET status = ?, error_message = COALESCE(error_message,"
                        " 'interrupted by a restart') WHERE batch_id = ? AND status = ?",
                        (PENDING, batch_id, PROCESSING))
            _recount(con, batch_id)
            r = con.execute("SELECT completed_images, failed_images, total_images FROM batches"
                            " WHERE batch_id = ?", (batch_id,)).fetchone()
            status = (COMPLETED if r["completed_images"] == r["total_images"] else
                      PARTIALLY_COMPLETED if r["completed_images"] else FAILED)
            con.execute("UPDATE batches SET status = ?, completed_at = COALESCE(completed_at, ?),"
                        " error_message = COALESCE(error_message, 'interrupted by a restart')"
                        " WHERE batch_id = ?", (status, now(), batch_id))
        return ids


# ─── reading ─────────────────────────────────────────────────────────────────

def get_batch(batch_id: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return dict(row) if row else None


def get_images(batch_id: str) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM batch_images WHERE batch_id = ? ORDER BY id", (batch_id,))]


def list_batches(limit: int = 50) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,))]
