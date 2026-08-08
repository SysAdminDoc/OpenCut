"""
OpenCut Search Routes

Footage search index: build, query, clear.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import (
    MAX_BATCH_FILES,
    _is_cancelled,
    _update_job,
    async_job,
)
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    require_csrf,
    safe_bool,
    safe_int,
    validate_filepath,
    verify_destructive_confirm_token,
)

logger = logging.getLogger("opencut")

search_bp = Blueprint("search", __name__)


def _validate_auto_index_payload(data):
    """Validate the synchronous portion of an auto-index request."""
    files = data.get("files", [])
    if not files or not isinstance(files, list):
        return "files must be a non-empty list"
    if len(files) > MAX_BATCH_FILES:
        return f"Too many files (max {MAX_BATCH_FILES})"

    from opencut.security import VALID_WHISPER_MODELS

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        return f"Invalid model: {model}"
    return None

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
    data = get_json_dict()
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
@async_job(
    "auto-index",
    filepath_required=False,
    pre_validate=_validate_auto_index_payload,
    resumable=True,
)
def auto_index_project(job_id, filepath, data):
    """Index changed project media through the durable job system.

    Expects JSON body:
    {
        "files": [{"path": "/path/to/file.mp4", "duration": 120.5}, ...]
    }

    Only files that need re-indexing (new or modified) are processed. Invalid,
    missing, and up-to-date entries are reported in the completed job result.
    """
    files = data.get("files", [])
    model = data.get("model", "base")
    language = data.get("language")

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.core.footage_index_db import (
        index_file as db_index_file,
    )
    from opencut.core.footage_index_db import (
        init_db,
        needs_reindex,
    )
    from opencut.helpers import get_video_info
    from opencut.utils.config import CaptionConfig

    init_db()
    to_index = []
    skipped_files = []
    for f in files:
        raw_path = f.get("path", "") if isinstance(f, dict) else str(f)
        path = str(raw_path).strip()
        if not path:
            skipped_files.append({"path": "", "reason": "missing_path"})
            continue
        try:
            path = validate_filepath(path)
        except ValueError as exc:
            skipped_files.append({"path": path, "reason": "invalid_path", "error": str(exc)})
            continue
        if needs_reindex(path):
            to_index.append((path, f if isinstance(f, dict) else {"path": path}))
        else:
            skipped_files.append({"path": path, "reason": "up_to_date"})

    if not to_index:
        return {
            "message": "All files are up to date",
            "queued": 0,
            "skipped": len(skipped_files),
            "skipped_files": skipped_files,
            "indexed": 0,
            "total": 0,
            "errors": [],
        }

    available, _backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    total = len(to_index)
    indexed = 0
    errors = []
    for idx, (fpath, metadata) in enumerate(to_index):
        if _is_cancelled(job_id):
            return {
                "message": "Indexing cancelled",
                "queued": total,
                "skipped": len(skipped_files),
                "skipped_files": skipped_files,
                "indexed": indexed,
                "total": total,
                "errors": errors,
            }

        _update_job(
            job_id,
            progress=int((idx / total) * 90),
            message=f"Indexing {os.path.basename(fpath)} ({idx + 1}/{total})...",
        )
        try:
            config = CaptionConfig(model=model, language=language, word_timestamps=True)
            result = transcribe(fpath, config=config)
            transcript = ""
            if hasattr(result, "segments"):
                transcript = " ".join(
                    str(getattr(segment, "text", "") or "").strip()
                    for segment in result.segments
                ).strip()

            duration = metadata.get("duration") if isinstance(metadata, dict) else None
            try:
                duration = float(duration) if duration is not None else None
            except (TypeError, ValueError):
                duration = None
            if duration is None:
                info = get_video_info(fpath)
                duration = float(info.get("duration", 0) or 0)

            db_index_file(fpath, transcript, duration=duration)
            indexed += 1
        except Exception as file_exc:
            logger.warning("Failed to auto-index %s: %s", fpath, file_exc)
            errors.append({"file": fpath, "error": str(file_exc)})

    return {
        "message": f"Indexed {indexed} of {total} changed files",
        "queued": total,
        "skipped": len(skipped_files),
        "skipped_files": skipped_files,
        "files": [path for path, _metadata in to_index],
        "indexed": indexed,
        "total": total,
        "errors": errors,
    }


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

    if len(query) > 500:
        return jsonify({"error": "query too long (max 500 characters)"}), 400

    limit = safe_int(data.get("limit", 50), 50, min_val=1, max_val=200)

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


@search_bp.route("/search/db-index", methods=["DELETE"])
@require_csrf
def search_clear_db_index():
    """Clear the SQLite footage index with optional dry-run/backup metadata."""
    payload = request.get_json(silent=True) or {}
    dry_run = safe_bool(request.args.get("dry_run", payload.get("dry_run", False)), False)
    backup = safe_bool(request.args.get("backup", payload.get("backup", False)), False)
    try:
        from opencut.core.footage_index_db import clear_index
        result = clear_index(dry_run=dry_run, backup=backup)
        if isinstance(result, dict):
            return jsonify(result)
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Failed to clear DB index: %s", e)
        return safe_error(e, "search_clear_db_index")


# ---------------------------------------------------------------------------
# Multimodal index: transcript + OCR text + audio event tags
# ---------------------------------------------------------------------------
@search_bp.route("/search/multimodal-index", methods=["POST"])
@require_csrf
@async_job("multimodal-index", filepath_required=False)
def search_multimodal_index(job_id, filepath, data):
    """Index files with transcript, OCR text, and audio event classification."""
    files = data.get("files", [])
    folder = data.get("folder", "").strip()
    model = data.get("model", "base")
    language = data.get("language", None)
    enable_ocr = safe_bool(data.get("ocr"), default=True)
    enable_audio_tags = safe_bool(data.get("audio_tags"), default=True)

    if folder and (not files or not isinstance(files, list)):
        from opencut.security import validate_path
        folder = validate_path(folder)
        if not os.path.isdir(folder):
            raise ValueError("folder is not a directory")
        _MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
                       ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in _MEDIA_EXTS]
        if not files:
            raise ValueError("No media files found in folder")

    if not isinstance(files, list) or not files:
        raise ValueError("files or folder required")
    if len(files) > 100:
        raise ValueError("Too many files (max 100)")

    validated_files = [validate_filepath(str(f).strip()) for f in files if isinstance(f, str) and f.strip()]

    from opencut.security import VALID_WHISPER_MODELS
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.core.footage_index_db import index_file as db_index_file
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

        base_pct = int((idx / total) * 90)
        _update_job(job_id, progress=base_pct,
                    message=f"Indexing {os.path.basename(fpath)} ({idx+1}/{total})...")

        try:
            config = CaptionConfig(model=model, language=language, word_timestamps=True)
            result = transcribe(fpath, config=config)
            transcript = ""
            if hasattr(result, "segments"):
                transcript = " ".join(getattr(seg, "text", "") for seg in result.segments)

            ocr_text = ""
            if enable_ocr:
                try:
                    from opencut.core.multimodal_index import extract_ocr_text
                    ocr_text = extract_ocr_text(fpath)
                except Exception as ocr_exc:
                    logger.debug("OCR failed for %s: %s", fpath, ocr_exc)

            audio_tags = ""
            if enable_audio_tags:
                try:
                    from opencut.core.multimodal_index import classify_audio_events
                    audio_tags = classify_audio_events(fpath)
                except Exception as at_exc:
                    logger.debug("Audio classify failed for %s: %s", fpath, at_exc)

            from opencut.helpers import get_video_info
            info = get_video_info(fpath)
            duration = info.get("duration", 0)

            db_index_file(fpath, transcript, duration=duration,
                          ocr_text=ocr_text, audio_tags=audio_tags)
            indexed += 1
        except Exception as file_exc:
            logger.warning("Failed to index %s: %s", fpath, file_exc)
            errors.append({"file": fpath, "error": str(file_exc)})

    return {"indexed": indexed, "total": total, "errors": errors,
            "modalities": ["transcript"] +
            (["ocr"] if enable_ocr else []) +
            (["audio_tags"] if enable_audio_tags else [])}


# ---------------------------------------------------------------------------
# Import-by-link: URL → local media cache
# ---------------------------------------------------------------------------
@search_bp.route("/search/ingest", methods=["POST"])
@require_csrf
@async_job("url_ingest", filepath_required=False)
def search_ingest_url(job_id, filepath, data):
    """Fetch a video URL to the local media cache."""
    from opencut.core import url_ingest

    def _prog(p, m=""):
        _update_job(job_id, progress=int(p), message=str(m))

    url = str(data.get("url") or "")
    result = url_ingest.ingest_url(url, on_progress=_prog)
    return {
        "filepath": result.filepath,
        "url": result.url,
        "title": result.title,
        "duration": result.duration,
        "filesize_mb": result.filesize_mb,
        "source": result.source,
        "cached": result.cached,
        "notes": result.notes,
    }


@search_bp.route("/search/ingest/cache", methods=["GET"])
def search_ingest_cache():
    """List cached ingest files."""
    try:
        from opencut.core import url_ingest
        entries = url_ingest.list_cached()
        return jsonify({"entries": entries, "count": len(entries)})
    except Exception as e:
        return safe_error(e, "ingest_cache_list")


@search_bp.route("/search/ingest/cache", methods=["DELETE"])
@require_csrf
def search_ingest_cache_clear():
    """Clear the ingest cache."""
    try:
        from opencut.core import url_ingest
        count = url_ingest.clear_cache()
        return jsonify({"cleared": count})
    except Exception as e:
        return safe_error(e, "ingest_cache_clear")


@search_bp.route("/search/db-diagnostics", methods=["GET"])
def search_db_diagnostics():
    """Get read-only diagnostics for the SQLite footage index."""
    try:
        from opencut.core.footage_index_db import get_db_diagnostics
        return jsonify(get_db_diagnostics())
    except Exception as e:
        logger.error("Failed to get index diagnostics: %s", e)
        return safe_error(e, "search_db_diagnostics")


# ---------------------------------------------------------------------------
# Search: Cleanup Missing Files
# ---------------------------------------------------------------------------
@search_bp.route("/search/cleanup", methods=["POST"])
@require_csrf
def cleanup_index():
    """Remove index entries for files that no longer exist."""
    try:
        data = get_json_dict() if request.data else {}
        dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
        from opencut.core.footage_index_db import init_db, missing_files_plan, remove_missing_files
        init_db()
        cleanup_plan = missing_files_plan()
        plan = build_destructive_plan(
            "search.cleanup_missing_files",
            records=cleanup_plan["entries"],
            metadata={
                "route": "/search/cleanup",
                "missing_count": cleanup_plan["missing_count"],
            },
            reversible=False,
        )
        if dry_run:
            return jsonify({
                "success": True,
                "dry_run": True,
                "removed": 0,
                "would_remove": cleanup_plan["missing_count"],
                "destructive_plan": plan,
                "confirm_token": plan["confirm_token"],
            })
        if plan["records"] and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409
        removed = remove_missing_files()
        return jsonify({"success": True, "removed": removed, "destructive_plan": plan})
    except Exception as e:
        return safe_error(e, "cleanup_index")
