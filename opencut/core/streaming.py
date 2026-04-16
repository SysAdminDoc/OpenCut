"""
OpenCut Response Streaming

NDJSON (Newline-Delimited JSON) streaming utilities for progressively
delivering large result sets to the frontend.
"""

import json
import logging

logger = logging.getLogger("opencut")


def _safe_dumps(obj):
    """``json.dumps`` that won't crash mid-stream on non-serializable values.

    A single bad item with ``bytes``/``datetime``/``set`` would otherwise
    raise ``TypeError`` and abort the entire NDJSON response. Falling back
    to ``default=str`` keeps the stream alive and gives the frontend a
    string approximation of the value.
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        try:
            return json.dumps(obj, default=str)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"_error": "serialization_failed", "_reason": str(exc)})


def ndjson_generator(items, chunk_size=50):
    """Yield items as NDJSON lines, with optional batching metadata.

    Args:
        items: iterable of dicts to stream
        chunk_size: how many items per logical batch (for progress tracking)

    Yields:
        str: JSON-encoded lines terminated by newline
    """
    total = len(items) if hasattr(items, "__len__") else None
    sent = 0

    # Send header with total count if known
    header = {"type": "header", "total": total, "chunk_size": chunk_size}
    yield _safe_dumps(header) + "\n"

    batch = []
    for item in items:
        batch.append(item)
        sent += 1

        if len(batch) >= chunk_size:
            yield _safe_dumps({"type": "batch", "items": batch, "sent": sent}) + "\n"
            batch = []

    # Flush remaining items
    if batch:
        yield _safe_dumps({"type": "batch", "items": batch, "sent": sent}) + "\n"

    # Send footer
    yield _safe_dumps({"type": "done", "total_sent": sent}) + "\n"


def ndjson_item_generator(items):
    """Yield each item as its own NDJSON line (one item per line).

    Simpler than batched streaming — each line is one result.
    Useful when the frontend wants to render items as they arrive.

    Args:
        items: iterable of dicts

    Yields:
        str: JSON-encoded lines terminated by newline
    """
    total = len(items) if hasattr(items, "__len__") else None
    yield _safe_dumps({"type": "header", "total": total}) + "\n"

    sent = 0
    for item in items:
        sent += 1
        yield _safe_dumps(item) + "\n"

    yield _safe_dumps({"type": "done", "total_sent": sent}) + "\n"


def ndjson_progress_generator(generator_fn, total_hint=None):
    """Wrap a generator function that yields (item, progress_pct) tuples.

    Useful for streaming results from long-running operations where each
    item is produced incrementally (e.g., thumbnail generation).

    Args:
        generator_fn: callable that yields (item_dict, progress_int) tuples
        total_hint: optional total count for the header

    Yields:
        str: NDJSON lines with progress info
    """
    yield _safe_dumps({"type": "header", "total": total_hint}) + "\n"

    sent = 0
    for item, progress in generator_fn():
        sent += 1
        # Build a copy so we don't mutate the caller's dict — the previous
        # implementation set ``item["_progress"]`` in place, which leaked
        # the streaming-only key back into the caller's data structure.
        if isinstance(item, dict):
            payload = dict(item)
            payload["_progress"] = progress
        else:
            payload = {"value": item, "_progress": progress}
        yield _safe_dumps(payload) + "\n"

    yield _safe_dumps({"type": "done", "total_sent": sent}) + "\n"


def make_ndjson_response(generator, flask_response_class):
    """Create a Flask streaming Response from an NDJSON generator.

    Args:
        generator: NDJSON line generator (from ndjson_generator etc.)
        flask_response_class: Flask Response class

    Returns:
        Flask Response with correct headers for NDJSON streaming
    """
    # Don't manually set ``Transfer-Encoding: chunked`` — WSGI servers add
    # it automatically when the response body is a generator. Setting it
    # ourselves caused the body to be double-chunked (chunk size + content)
    # on some servers, breaking strict NDJSON line parsers on the client.
    resp = flask_response_class(generator, mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp
