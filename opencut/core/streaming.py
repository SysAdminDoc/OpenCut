"""
OpenCut Response Streaming

NDJSON (Newline-Delimited JSON) streaming utilities for progressively
delivering large result sets to the frontend.
"""

import json
import logging

logger = logging.getLogger("opencut")


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
    yield json.dumps(header) + "\n"

    batch = []
    for item in items:
        batch.append(item)
        sent += 1

        if len(batch) >= chunk_size:
            yield json.dumps({"type": "batch", "items": batch, "sent": sent}) + "\n"
            batch = []

    # Flush remaining items
    if batch:
        yield json.dumps({"type": "batch", "items": batch, "sent": sent}) + "\n"

    # Send footer
    yield json.dumps({"type": "done", "total_sent": sent}) + "\n"


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
    yield json.dumps({"type": "header", "total": total}) + "\n"

    sent = 0
    for item in items:
        sent += 1
        yield json.dumps(item) + "\n"

    yield json.dumps({"type": "done", "total_sent": sent}) + "\n"


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
    yield json.dumps({"type": "header", "total": total_hint}) + "\n"

    sent = 0
    for item, progress in generator_fn():
        sent += 1
        item["_progress"] = progress
        yield json.dumps(item) + "\n"

    yield json.dumps({"type": "done", "total_sent": sent}) + "\n"


def make_ndjson_response(generator, flask_response_class):
    """Create a Flask streaming Response from an NDJSON generator.

    Args:
        generator: NDJSON line generator (from ndjson_generator etc.)
        flask_response_class: Flask Response class

    Returns:
        Flask Response with correct headers for NDJSON streaming
    """
    resp = flask_response_class(generator, mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Transfer-Encoding"] = "chunked"
    return resp
