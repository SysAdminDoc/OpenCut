"""A v1.25.1-shaped ``~/.opencut`` tree, built from that tag's own schemas.

The latest public installer is v1.25.1 while source has moved many schema
versions past it. Nothing proved that a user upgrading across that gap keeps
their queue, journal, job history, footage index, review artifacts, and
settings — so this module materializes the *old* shape and
``tests/test_upgrade_conformance.py`` drives the current code over it.

The DDL and document shapes below are transcribed from ``git show v1.25.1``
and must not be "modernized". They are the historical side of the upgrade;
rewriting them to match today's schema would make the conformance test pass
by construction and prove nothing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

LEGACY_VERSION = "1.25.1"

# --- v1.25.1 opencut/job_store.py::init_db ---------------------------------
JOBS_DDL_V1_25_1 = """
    CREATE TABLE IF NOT EXISTS jobs (
        id           TEXT PRIMARY KEY,
        type         TEXT NOT NULL,
        filepath     TEXT DEFAULT '',
        status       TEXT NOT NULL DEFAULT 'running',
        progress     INTEGER DEFAULT 0,
        message      TEXT DEFAULT '',
        result_json  TEXT DEFAULT NULL,
        error        TEXT DEFAULT NULL,
        endpoint     TEXT DEFAULT '',
        payload_json TEXT DEFAULT NULL,
        created_at   REAL NOT NULL,
        started_at   REAL DEFAULT NULL,
        completed_at REAL DEFAULT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at);
"""

# --- v1.25.1 opencut/journal.py::init_db -----------------------------------
# ``forward_json`` was added by a bare ALTER outside any versioning scheme,
# so a v1.25.1 database already carries it while ``user_version`` is still 0.
JOURNAL_DDL_V1_25_1 = """
    CREATE TABLE IF NOT EXISTS journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL NOT NULL,
        action TEXT NOT NULL,
        clip_path TEXT NOT NULL DEFAULT '',
        label TEXT NOT NULL DEFAULT '',
        inverse_json TEXT NOT NULL DEFAULT '{}',
        reverted INTEGER NOT NULL DEFAULT 0,
        reverted_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_journal_created ON journal(created_at DESC);
"""

# --- v1.25.1 opencut/core/footage_index_db.py::init_db ---------------------
FOOTAGE_DDL_V1_25_1 = """
    CREATE TABLE IF NOT EXISTS footage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        transcript TEXT NOT NULL DEFAULT '',
        indexed_at REAL NOT NULL,
        file_mtime REAL NOT NULL DEFAULT 0,
        duration REAL NOT NULL DEFAULT 0,
        file_size INTEGER NOT NULL DEFAULT 0
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS footage_fts USING fts5(
        file_path,
        transcript,
        content=footage,
        content_rowid=id
    );

    CREATE TRIGGER IF NOT EXISTS footage_ai AFTER INSERT ON footage BEGIN
        INSERT INTO footage_fts(rowid, file_path, transcript)
        VALUES (new.id, new.file_path, new.transcript);
    END;

    CREATE TRIGGER IF NOT EXISTS footage_ad AFTER DELETE ON footage BEGIN
        INSERT INTO footage_fts(footage_fts, rowid, file_path, transcript)
        VALUES ('delete', old.id, old.file_path, old.transcript);
    END;

    CREATE TRIGGER IF NOT EXISTS footage_au AFTER UPDATE ON footage BEGIN
        INSERT INTO footage_fts(footage_fts, rowid, file_path, transcript)
        VALUES ('delete', old.id, old.file_path, old.transcript);
        INSERT INTO footage_fts(rowid, file_path, transcript)
        VALUES (new.id, new.file_path, new.transcript);
    END;
"""

# Stable clock so every generated artifact is reproducible.
LEGACY_EPOCH = 1_735_689_600.0  # 2025-01-01T00:00:00Z

LEGACY_JOB_IDS = ("job-legacy-complete", "job-legacy-running")
LEGACY_JOURNAL_LABELS = ("Silence cut", "Caption burn-in")
LEGACY_FOOTAGE_PATH = "/media/legacy/interview-take-3.mp4"
LEGACY_QUEUE_ENDPOINT = "/silence"
LEGACY_REVIEW_ID = "review-legacy-001"
LEGACY_PLUGIN_NAME = "timecode-watermark"
LEGACY_API_KEY = "sk-legacy-plaintext-key-0001"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_jobs_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(JOBS_DDL_V1_25_1)
        conn.execute(
            "INSERT INTO jobs (id, type, filepath, status, progress, message, "
            "result_json, endpoint, payload_json, created_at, started_at, completed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                LEGACY_JOB_IDS[0], "silence", "/media/legacy/a.mp4", "complete", 100,
                "Done", json.dumps({"output": "/media/legacy/a-cut.mp4"}),
                LEGACY_QUEUE_ENDPOINT, json.dumps({"filepath": "/media/legacy/a.mp4"}),
                LEGACY_EPOCH, LEGACY_EPOCH, LEGACY_EPOCH + 12.0,
            ),
        )
        conn.execute(
            "INSERT INTO jobs (id, type, filepath, status, progress, message, "
            "endpoint, created_at, started_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                LEGACY_JOB_IDS[1], "captions", "/media/legacy/b.mp4", "running", 40,
                "Transcribing", "/captions", LEGACY_EPOCH + 60.0, LEGACY_EPOCH + 60.0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _build_journal_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(JOURNAL_DDL_V1_25_1)
        # The out-of-band ALTER that shipped in v1.10.3, replayed verbatim.
        conn.execute("ALTER TABLE journal ADD COLUMN forward_json TEXT")
        for offset, label in enumerate(LEGACY_JOURNAL_LABELS):
            conn.execute(
                "INSERT INTO journal (created_at, action, clip_path, label, "
                "inverse_json, reverted, forward_json) VALUES (?,?,?,?,?,?,?)",
                (
                    LEGACY_EPOCH + offset,
                    "apply_cuts" if offset == 0 else "burn_captions",
                    f"/media/legacy/{'a' if offset == 0 else 'b'}.mp4",
                    label,
                    json.dumps({"undo": "restore"}),
                    0,
                    json.dumps({"redo": "reapply"}) if offset == 0 else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _build_footage_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(FOOTAGE_DDL_V1_25_1)
        conn.execute(
            "INSERT INTO footage (file_path, transcript, indexed_at, file_mtime, "
            "duration, file_size) VALUES (?,?,?,?,?,?)",
            (
                LEGACY_FOOTAGE_PATH,
                "welcome back to the channel today we are talking about edge cases",
                LEGACY_EPOCH, LEGACY_EPOCH, 305.5, 1_048_576,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def build_v1_25_1_user_data(root: Path) -> Path:
    """Materialize a v1.25.1 ``~/.opencut`` tree under ``root`` and return it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    _build_jobs_db(root / "jobs.db")
    _build_journal_db(root / "journal.db")
    _build_footage_db(root / "footage_index.db")

    # v1.25.1 wrote the queue as a bare list — no envelope, no schema version.
    _write_json(root / "job_queue.json", [
        {
            "id": "queue-legacy-001",
            "endpoint": LEGACY_QUEUE_ENDPOINT,
            "payload": {"filepath": "/media/legacy/c.mp4"},
            "status": "pending",
            "created_at": LEGACY_EPOCH,
        }
    ])

    # Plain JSON settings with no `_schema_version` key anywhere.
    _write_json(root / "user_presets.json", {
        "Quiet room": {"threshold_db": -38.0, "min_duration": 0.4}
    })
    _write_json(root / "favorites.json", ["silence", "captions"])
    _write_json(root / "workflows.json", {
        "Legacy podcast": [
            {"endpoint": "/audio/denoise", "label": "Denoise", "payload": {}},
            {"endpoint": "/captions", "label": "Captions", "payload": {}},
        ]
    })
    _write_json(root / "whisper_settings.json", {"model": "base", "device": "auto"})
    _write_json(root / "loudness_settings.json", {"target_lufs": -14.0})
    _write_json(root / "chapter_defaults.json", {"max_chapters": 8})
    _write_json(root / "footage_index_config.json", {"auto_index": True})

    # v1.25.1 stored the LLM key in plaintext; the current build migrates it
    # into the OS credential vault (or refuses to keep it in the clear).
    _write_json(root / "llm_settings.json", {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": LEGACY_API_KEY,
        "base_url": "",
    })

    # Pre-versioning review record: no `versions` list, no schema_version.
    _write_json(root / "reviews" / "reviews.json", {
        LEGACY_REVIEW_ID: {
            "review_id": LEGACY_REVIEW_ID,
            "video_path": "/media/legacy/a-cut.mp4",
            "status": "approved",
            "status_updated_at": LEGACY_EPOCH,
            "created_at": LEGACY_EPOCH,
            "comments": [
                {"comment_id": "c1", "author": "editor", "text": "ship it",
                 "created_at": LEGACY_EPOCH}
            ],
        }
    })

    # An installed plugin. v1.25.1 predates the publisher trust store, so the
    # upgrade must tolerate its absence rather than treat it as tampering.
    _write_json(root / "plugins" / LEGACY_PLUGIN_NAME / "plugin.json", {
        "name": LEGACY_PLUGIN_NAME,
        "version": "1.0.0",
        "description": "Legacy timecode watermark plugin",
        "api_version": 1,
    })
    (root / "plugins" / LEGACY_PLUGIN_NAME / "routes.py").write_text(
        "from flask import Blueprint\n\nplugin_bp = Blueprint('legacy', __name__)\n",
        encoding="utf-8",
    )

    # The installer's own stamp — the only record of which build wrote this tree.
    _write_json(root / "installer.json", {
        "app_name": "OpenCut",
        "app_version": LEGACY_VERSION,
        "installer_kind": "inno",
        "install_path": "C:\\Program Files\\OpenCut",
        "installed_at_utc": "2026-01-01T00:00:00Z",
    })

    (root / "crash.log").write_text("legacy crash log line\n", encoding="utf-8")
    os.makedirs(root / "models", exist_ok=True)
    return root


def snapshot_tree(root: Path) -> dict[str, int]:
    """Return ``{relative path: size}`` for every file under ``root``."""
    root = Path(root)
    return {
        str(path.relative_to(root)).replace("\\", "/"): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def legacy_mtime_stamp() -> float:
    return LEGACY_EPOCH + time.timezone * 0
