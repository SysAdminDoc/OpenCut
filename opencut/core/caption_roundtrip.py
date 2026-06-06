"""Caption sidecars for timeline-native round-trip workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from opencut.core.captions import TranscriptionResult, caption_segment_to_dict
from opencut.core.transcript_cache import source_digest

SIDECAR_SCHEMA = "opencut.caption_sidecar"
SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".opencut-captions.json"


def caption_sidecar_path(export_path: str) -> str:
    """Return the canonical sidecar path for a subtitle export."""
    path = Path(export_path)
    if path.suffix:
        return str(path.with_suffix(path.suffix + SIDECAR_SUFFIX))
    return str(path.with_suffix(SIDECAR_SUFFIX))


def _safe_realpath(path: str | None) -> str:
    if not path:
        return ""
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except (OSError, ValueError):
        return str(path)


def _source_identity(source_path: str | None) -> dict[str, Any]:
    resolved = _safe_realpath(source_path)
    if not resolved or not os.path.isfile(resolved):
        return {"path": resolved, "hash_algorithm": "sha256", "source_file_hash": None}
    try:
        digest = source_digest(resolved)
    except OSError:
        return {"path": resolved, "hash_algorithm": "sha256", "source_file_hash": None}
    return {
        "path": resolved,
        "hash_algorithm": digest.get("hash_algorithm", "sha256"),
        "source_file_hash": digest.get("source_sha256"),
        "source_size_bytes": digest.get("source_size_bytes"),
        "source_mtime_ns": digest.get("source_mtime_ns"),
    }


def _segment_source_id(segment: dict[str, Any], index: int) -> str:
    for key in ("source_segment_id", "segment_id", "id"):
        value = str(segment.get(key, "")).strip()
        if value:
            return value[:128]
    return f"segment_{index + 1:06d}"


def _word_payload(words: list[dict[str, Any]], caption_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    word_items: list[dict[str, Any]] = []
    word_ids: list[str] = []
    for index, word in enumerate(words):
        word_id = str(word.get("word_id", "")).strip() or f"{caption_id}_word_{index + 1:04d}"
        word_ids.append(word_id)
        word_items.append({
            "word_id": word_id,
            "text": str(word.get("text", "")).strip(),
            "start": word.get("start", 0.0),
            "end": word.get("end", 0.0),
            "confidence": word.get("confidence", 1.0),
        })
    return word_items, word_ids


def build_caption_sidecar(
    result: TranscriptionResult,
    export_path: str,
    *,
    export_format: str,
    source_path: str | None = None,
    transcript_cache_key: str | None = None,
    display_settings: dict[str, Any] | None = None,
    host_locators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned caption metadata sidecar payload."""
    source = _source_identity(source_path)
    cache_key = transcript_cache_key or getattr(result, "cache_key", None)
    cues = []
    for index, segment_obj in enumerate(getattr(result, "segments", []) or []):
        segment = caption_segment_to_dict(segment_obj, include_words=True)
        caption_id = str(segment.get("caption_id", "")).strip() or f"cap_{index + 1:06d}"
        words, word_ids = _word_payload(segment.get("words") or [], caption_id)
        cues.append({
            **segment,
            "caption_id": caption_id,
            "source_segment_id": _segment_source_id(segment, index),
            "transcript_cache_key": cache_key,
            "source_file_hash": source.get("source_file_hash"),
            "word_ids": word_ids,
            "words": words,
            "display_setting_token_ids": list((display_settings or {}).get("token_ids") or []),
            "display_settings": display_settings or {},
            "export_format": export_format,
            "host_locators": host_locators or {},
        })

    return {
        "schema": SIDECAR_SCHEMA,
        "schema_version": SIDECAR_VERSION,
        "export": {
            "format": export_format,
            "path": _safe_realpath(export_path),
        },
        "source": {
            **source,
            "transcript_cache_key": cache_key,
        },
        "result": {
            "language": getattr(result, "language", "en"),
            "duration": float(getattr(result, "duration", 0.0) or 0.0),
            "language_confidence": float(getattr(result, "language_confidence", 1.0) or 1.0),
            "text": getattr(result, "text", ""),
            "word_count": int(getattr(result, "word_count", 0) or 0),
            "human_review_recommended": bool(getattr(result, "human_review_recommended", False)),
            "review_segment_count": int(getattr(result, "review_segment_count", 0) or 0),
        },
        "cues": cues,
        "warnings": [],
    }


def write_caption_sidecar(
    result: TranscriptionResult,
    export_path: str,
    *,
    export_format: str,
    source_path: str | None = None,
    transcript_cache_key: str | None = None,
    display_settings: dict[str, Any] | None = None,
    host_locators: dict[str, Any] | None = None,
    sidecar_path: str | None = None,
) -> str:
    """Write a caption sidecar next to an exported subtitle file."""
    path = sidecar_path or caption_sidecar_path(export_path)
    payload = build_caption_sidecar(
        result,
        export_path,
        export_format=export_format,
        source_path=source_path,
        transcript_cache_key=transcript_cache_key,
        display_settings=display_settings,
        host_locators=host_locators,
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def read_caption_sidecar(
    sidecar_path: str,
    *,
    expected_export_path: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate a caption sidecar, returning warnings instead of raising."""
    warnings: list[str] = []
    if not sidecar_path:
        return None, ["sidecar_missing"]
    try:
        exists = os.path.isfile(sidecar_path)
    except (OSError, ValueError):
        return None, ["sidecar_missing"]
    if not exists:
        return None, ["sidecar_missing"]
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, ["sidecar_invalid"]
    if not isinstance(payload, dict):
        return None, ["sidecar_invalid"]
    if payload.get("schema") != SIDECAR_SCHEMA or payload.get("schema_version") != SIDECAR_VERSION:
        warnings.append("sidecar_schema_mismatch")
    cues = payload.get("cues")
    if not isinstance(cues, list):
        warnings.append("sidecar_cues_missing")
    if expected_export_path:
        stored = _safe_realpath((payload.get("export") or {}).get("path"))
        expected = _safe_realpath(expected_export_path)
        if stored and expected and stored != expected:
            warnings.append("sidecar_export_path_mismatch")
    return payload, warnings


def merge_segments_with_sidecar(
    segments: list[dict[str, Any]],
    sidecar: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Enrich parsed SRT/VTT segments with sidecar metadata by cue order."""
    if not sidecar:
        return segments, ["metadata_unavailable"]
    cues = sidecar.get("cues")
    if not isinstance(cues, list):
        return segments, ["metadata_unavailable"]
    warnings: list[str] = []
    if len(cues) != len(segments):
        warnings.append("sidecar_cue_count_mismatch")
    enriched: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if index >= len(cues) or not isinstance(cues[index], dict):
            enriched.append(segment)
            continue
        cue = dict(cues[index])
        cue["start"] = segment.get("start", 0.0)
        cue["end"] = segment.get("end", 0.0)
        cue["text"] = segment.get("text", "")
        cue["sidecar_index"] = index
        enriched.append(cue)
    return enriched, warnings
