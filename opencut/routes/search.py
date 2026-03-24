"""
OpenCut Search Routes

Footage search index: build, query, clear.
"""

import logging
import os
import threading

from flask import Blueprint, jsonify, request

from opencut.jobs import (
    _is_cancelled,
    _new_job,
    _safe_error,
    _update_job,
    job_lock,
    jobs,
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
def search_index():
    """Transcribe and index a list of files for footage search."""
    data = request.get_json(force=True)
    files = data.get("files", [])
    model = data.get("model", "base")
    language = data.get("language", None)

    if not isinstance(files, list) or not files:
        return jsonify({"error": "files must be a non-empty list"}), 400

    if len(files) > 100:
        return jsonify({"error": "Too many files (max 100)"}), 400

    # Validate all file paths up-front
    validated_files = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            return jsonify({"error": "Each file entry must be a non-empty string"}), 400
        try:
            validated_files.append(validate_filepath(f.strip()))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    from opencut.security import VALID_WHISPER_MODELS
    if model not in VALID_WHISPER_MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400

    job_id = _new_job("search-index", validated_files[0] if validated_files else "")

    def _process():
        try:
            from opencut.core import footage_search
            from opencut.core.captions import check_whisper_available, transcribe
            from opencut.utils.config import CaptionConfig

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="No Whisper backend installed. Run: pip install faster-whisper",
                    message="Whisper not installed",
                )
                return

            total = len(validated_files)
            indexed = 0
            errors = []

            for idx, filepath in enumerate(validated_files):
                if _is_cancelled(job_id):
                    return

                pct = int((idx / total) * 90)
                _update_job(job_id, progress=pct, message=f"Indexing {os.path.basename(filepath)} ({idx + 1}/{total})...")

                try:
                    config = CaptionConfig(model=model, language=language, word_timestamps=True)
                    result = transcribe(filepath, config=config)
                    segments = []
                    if hasattr(result, "segments"):
                        for seg in result.segments:
                            segments.append({
                                "start": getattr(seg, "start", 0),
                                "end": getattr(seg, "end", 0),
                                "text": getattr(seg, "text", ""),
                            })
                    footage_search.index_file(filepath, segments)
                    indexed += 1
                except Exception as file_exc:
                    logger.warning("Failed to index %s: %s", filepath, file_exc)
                    errors.append({"file": filepath, "error": str(file_exc)})

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Indexed {indexed}/{total} files",
                result={"indexed": indexed, "total": total, "errors": errors},
            )
        except Exception as exc:
            _update_job(job_id, status="error", error=str(exc), message=f"Error: {exc}")
            logger.exception("search-index error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(exc)


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
        return _safe_error(exc)
