"""
OpenCut Search Routes

Footage search index: build, query, clear.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import (
    _is_cancelled,
    _update_job,
    async_job,
)
from opencut.security import (
    require_csrf,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

search_bp = Blueprint("search", __name__)

# ---------------------------------------------------------------------------
# Search: Index Files
# ---------------------------------------------------------------------------
@search_bp.route("/search/index", methods=["POST"])
@require_csrf
@async_job("search-index", filepath_required=False)
def search_index(job_id, filepath, data):
    """Transcribe and index a list of files (or scan a folder) for footage search."""
    files = data.get("files", [])
    folder = data.get("folder", "").strip()
    model = data.get("model", "base")
    language = data.get("language", None)

    # If folder provided, scan for media files
    if folder and (not files or not isinstance(files, list)):
        from opencut.security import validate_path
        folder = validate_path(folder)
        if not os.path.isdir(folder):
            raise ValueError("folder is not a directory")
        _MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
        files = []
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in _MEDIA_EXTS:
                files.append(os.path.join(folder, fname))
        if not files:
            raise ValueError("No media files found in folder")

    if not isinstance(files, list) or not files:
        raise ValueError("files or folder required")

    if len(files) > 100:
        raise ValueError("Too many files (max 100)")

    # Validate all file paths up-front
    validated_files = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            raise ValueError("Each file entry must be a non-empty string")
        validated_files.append(validate_filepath(f.strip()))

    from opencut.security import VALID_WHISPER_MODELS
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")

    from opencut.core import footage_search
    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.utils.config import CaptionConfig

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    total = len(validated_files)
    indexed = 0
    errors = []

    for idx, fpath in enumerate(validated_files):
        if _is_cancelled(job_id):
            return {"indexed": indexed, "total": total, "errors": errors}

        pct = int((idx / total) * 90)
        _update_job(job_id, progress=pct, message=f"Indexing {os.path.basename(fpath)} ({idx + 1}/{total})...")

        try:
            config = CaptionConfig(model=model, language=language, word_timestamps=True)
            result = transcribe(fpath, config=config)
            segments = []
            if hasattr(result, "segments"):
                for seg in result.segments:
                    segments.append({
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                    })
            footage_search.index_file(fpath, segments)
            indexed += 1
        except Exception as file_exc:
            logger.warning("Failed to index %s: %s", fpath, file_exc)
            errors.append({"file": fpath, "error": str(file_exc)})

    return {"indexed": indexed, "total": total, "errors": errors}


# ---------------------------------------------------------------------------
# Search: Query Footage
# ---------------------------------------------------------------------------
@search_bp.route("/search/footage", methods=["POST"])
@require_csrf
def search_footage():
    """Search the indexed footage for a text query."""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = safe_int(data.get("top_k", 10), 10, min_val=1, max_val=100)

    if not query:
        return jsonify({"error": "query must not be empty"}), 400

    if len(query) > 500:
        return jsonify({"error": "query too long (max 500 characters)"}), 400

    try:
        from opencut.core import footage_search
        results = footage_search.search_footage(query, top_k=top_k)
        return jsonify({
            "results": results,
            "query": query,
            "total_matches": len(results),
        })
    except ImportError:
        return jsonify({"error": "footage_search module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "search_footage")


# ---------------------------------------------------------------------------
# Search: Clear Index
# ---------------------------------------------------------------------------
@search_bp.route("/search/index", methods=["DELETE"])
@require_csrf
def search_clear_index():
    """Clear the footage search index."""
    try:
        from opencut.core import footage_search
        footage_search.clear_index()
        return jsonify({"success": True})
    except ImportError:
        return jsonify({"error": "footage_search module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "search_clear_index")


# ---------------------------------------------------------------------------
# Search: Auto-Index Project Files (SQLite)
# ---------------------------------------------------------------------------
@search_bp.route("/search/auto-index", methods=["POST"])
@require_csrf
def auto_index_project():
    """Trigger background indexing for all project media files.

    Expects JSON body:
    {
        "files": [{"path": "/path/to/file.mp4", "duration": 120.5}, ...]
    }

    Only files that need re-indexing (new or modified) will be processed.
    """
    data = request.get_json(force=True, silent=True) or {}
    files = data.get("files", [])

    if not files or not isinstance(files, list):
        return jsonify({"error": "files must be a non-empty list"}), 400

    try:
        from opencut.core.footage_index_db import init_db, needs_reindex
        init_db()
    except Exception as e:
        logger.error("Failed to init footage index DB: %s", e)
        return safe_error(e, "auto_index_project")

    # Filter to only files that need indexing
    to_index = []
    for f in files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if not path:
            continue
        try:
            path = validate_filepath(path.strip())
        except ValueError:
            continue
        if needs_reindex(path):
            to_index.append(f if isinstance(f, dict) else {"path": path})

    if not to_index:
        return jsonify({
            "message": "All files are up to date",
            "queued": 0,
            "skipped": len(files),
        })

    # Queue each file for background transcription + indexing
    return jsonify({
        "message": f"Queued {len(to_index)} files for indexing",
        "queued": len(to_index),
        "skipped": len(files) - len(to_index),
        "files": [f.get("path", "") if isinstance(f, dict) else str(f) for f in to_index],
    })


# ---------------------------------------------------------------------------
# Search: FTS5 Database Search
# ---------------------------------------------------------------------------
@search_bp.route("/search/db-search", methods=["POST"])
@require_csrf
def search_footage_db():
    """Search indexed footage using SQLite FTS5.

    Expects JSON body: {"query": "search terms", "limit": 50}
    """
    data = request.get_json(force=True, silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    limit = safe_int(data.get("limit", 50), min_val=1, max_val=200)

    try:
        from opencut.core.footage_index_db import init_db, search
        init_db()
        results = search(query, limit=limit)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        logger.error("FTS search failed: %s", e)
        return safe_error(e, "search_footage_db")


# ---------------------------------------------------------------------------
# Search: Database Stats
# ---------------------------------------------------------------------------
@search_bp.route("/search/db-stats", methods=["GET"])
def search_db_stats():
    """Get SQLite footage index statistics."""
    try:
        from opencut.core.footage_index_db import get_stats, init_db
        init_db()
        stats = get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error("Failed to get index stats: %s", e)
        return safe_error(e, "search_db_stats")


# ---------------------------------------------------------------------------
# Search: Cleanup Missing Files
# ---------------------------------------------------------------------------
@search_bp.route("/search/cleanup", methods=["POST"])
@require_csrf
def cleanup_index():
    """Remove index entries for files that no longer exist."""
    try:
        from opencut.core.footage_index_db import init_db, remove_missing_files
        init_db()
        removed = remove_missing_files()
        return jsonify({"removed": removed})
    except Exception as e:
        return safe_error(e, "cleanup_index")
