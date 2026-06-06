"""Caption sidecars for timeline-native round-trip workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from opencut.core.captions import TranscriptionResult, caption_segment_to_dict
from opencut.core.transcript_cache import source_digest
from opencut.user_data import OPENCUT_DIR

SIDECAR_SCHEMA = "opencut.caption_sidecar"
SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".opencut-captions.json"
REVISION_SCHEMA = "opencut.caption_revision"
REVISION_VERSION = 1
REVISION_DIR_ENV = "OPENCUT_CAPTION_REVISION_DIR"
_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")


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


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _tc_to_seconds(timecode: str) -> float:
    clean = str(timecode).replace(",", ".")
    hours, minutes, seconds = clean.split(":")
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


def parse_srt_text(content: str, *, max_segments: int = 10000) -> list[dict[str, Any]]:
    """Parse SRT text into lossy ``start``/``end``/``text`` segments."""
    segments: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", str(content or "").strip())
    for block in blocks:
        if len(segments) >= max_segments:
            break
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        time_index = None
        for index, line in enumerate(lines):
            if _TIME_RE.match(line.strip()):
                time_index = index
                break
        if time_index is None:
            continue
        match = _TIME_RE.match(lines[time_index].strip())
        if not match:
            continue
        text = " ".join(lines[time_index + 1:]).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            segments.append({
                "start": _tc_to_seconds(match.group(1)),
                "end": _tc_to_seconds(match.group(2)),
                "text": text,
            })
    return segments


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


_SUMMARY_METADATA_FIELDS = (
    "source_segment_id",
    "source_caption_id",
    "source_caption_ids",
    "transcript_cache_key",
    "source_file_hash",
    "word_ids",
    "words",
    "speaker",
    "language",
    "language_confidence",
    "confidence",
    "human_review_recommended",
    "review_reasons",
    "display_setting_token_ids",
    "display_settings",
    "export_format",
    "host_locators",
    "sidecar_index",
)


def _cue_summary(cue: dict[str, Any], fallback_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "caption_id": cue.get("caption_id"),
        "start": cue.get("start", 0.0),
        "end": cue.get("end", 0.0),
        "text": cue.get("text", ""),
    }
    for field in _SUMMARY_METADATA_FIELDS:
        if field in cue:
            summary[field] = cue[field]
        elif fallback_metadata and field in fallback_metadata:
            summary[field] = fallback_metadata[field]
    return summary


def _normal_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _timing_changed(before: dict[str, Any], after: dict[str, Any], tolerance: float = 0.001) -> bool:
    return (
        abs(float(before.get("start", 0.0) or 0.0) - float(after.get("start", 0.0) or 0.0)) > tolerance
        or abs(float(before.get("end", 0.0) or 0.0) - float(after.get("end", 0.0) or 0.0)) > tolerance
    )


def _style_payload(cue: dict[str, Any]) -> dict[str, Any]:
    return {
        "display_setting_token_ids": list(cue.get("display_setting_token_ids") or []),
        "display_settings": dict(cue.get("display_settings") or {}),
    }


def _source_caption_ids(cue: dict[str, Any]) -> list[str]:
    raw = cue.get("source_caption_ids")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    for key in ("source_caption_id", "caption_id"):
        value = str(cue.get(key, "")).strip()
        if value:
            return [value]
    return []


def diff_caption_roundtrip(
    *,
    sidecar: dict[str, Any] | None = None,
    edited_segments: list[dict[str, Any]] | None = None,
    original_segments: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic caption round-trip diff payload."""
    diff_warnings = list(warnings or [])
    sidecar_cues = (sidecar or {}).get("cues")
    has_sidecar_cues = isinstance(sidecar_cues, list)
    metadata_preserved = bool(sidecar) and has_sidecar_cues
    originals = list(sidecar_cues if has_sidecar_cues else original_segments or [])
    edited = list(edited_segments or [])
    if sidecar and not has_sidecar_cues and "sidecar_cues_missing" not in diff_warnings:
        diff_warnings.append("sidecar_cues_missing")
    if not metadata_preserved:
        diff_warnings.append("metadata_unavailable")
    if len(originals) != len(edited):
        diff_warnings.append("cue_count_mismatch")

    source_counts: dict[str, int] = {}
    for cue in edited:
        for source_id in _source_caption_ids(cue):
            source_counts[source_id] = source_counts.get(source_id, 0) + 1

    counts = {
        "unchanged": 0,
        "changed": 0,
        "text_changed": 0,
        "timing_changed": 0,
        "style_changed": 0,
        "split": 0,
        "merge": 0,
        "deleted": 0,
        "inserted": 0,
    }
    changes: list[dict[str, Any]] = []
    max_len = max(len(originals), len(edited))
    for index in range(max_len):
        if index >= len(originals):
            counts["changed"] += 1
            counts["inserted"] += 1
            changes.append({
                "caption_id": edited[index].get("caption_id"),
                "change_type": "inserted",
                "changes": ["inserted"],
                "before": None,
                "after": _cue_summary(edited[index]),
            })
            continue
        if index >= len(edited):
            counts["changed"] += 1
            counts["deleted"] += 1
            changes.append({
                "caption_id": originals[index].get("caption_id"),
                "change_type": "deleted",
                "changes": ["deleted"],
                "before": _cue_summary(originals[index]),
                "after": None,
            })
            continue

        before = originals[index]
        after = edited[index]
        cue_changes: list[str] = []
        if _normal_text(before.get("text")) != _normal_text(after.get("text")):
            cue_changes.append("text_changed")
        if _timing_changed(before, after):
            cue_changes.append("timing_changed")
        if _style_payload(before) != _style_payload(after):
            cue_changes.append("style_changed")
        source_ids = _source_caption_ids(after)
        if any(source_counts.get(source_id, 0) > 1 for source_id in source_ids):
            cue_changes.append("split")
        if len(source_ids) > 1:
            cue_changes.append("merge")

        if cue_changes:
            counts["changed"] += 1
            for change in sorted(set(cue_changes), key=cue_changes.index):
                counts[change] += 1
        else:
            counts["unchanged"] += 1
        changes.append({
            "caption_id": before.get("caption_id") or after.get("caption_id"),
            "change_type": cue_changes[0] if cue_changes else "unchanged",
            "changes": cue_changes,
            "before": _cue_summary(before),
            "after": _cue_summary(after, before if metadata_preserved else None),
        })

    confidence = 0.9 if metadata_preserved else 0.45
    if "cue_count_mismatch" in diff_warnings:
        confidence = min(confidence, 0.65)
    if any(warning.endswith("mismatch") for warning in diff_warnings):
        confidence = min(confidence, 0.6)

    return {
        "metadata_preserved": metadata_preserved,
        "confidence": round(confidence, 2),
        "confidence_label": "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low",
        "warnings": sorted(set(diff_warnings)),
        "counts": counts,
        "changes": changes,
        "summary": {
            "changed": counts["changed"] > 0,
            "changed_cues": counts["changed"],
            "unchanged_cues": counts["unchanged"],
            "total_before": len(originals),
            "total_after": len(edited),
            "metadata_preserved": metadata_preserved,
        },
        "source": {
            "transcript_cache_key": (sidecar or {}).get("source", {}).get("transcript_cache_key"),
            "source_file_hash": (sidecar or {}).get("source", {}).get("source_file_hash"),
        },
    }


def caption_revision_dir() -> str:
    override = os.environ.get(REVISION_DIR_ENV)
    if override:
        return _safe_realpath(override)
    return os.path.join(OPENCUT_DIR, "caption_revisions")


def store_caption_revision(diff_payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a content-addressed caption round-trip revision."""
    source = diff_payload.get("source") or {}
    revision_body = {
        "schema": REVISION_SCHEMA,
        "schema_version": REVISION_VERSION,
        "source": source,
        "diff": diff_payload,
    }
    revision_id = hashlib.sha256(_stable_json(revision_body).encode("utf-8")).hexdigest()
    revision_body["revision_id"] = revision_id
    root = caption_revision_dir()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{revision_id}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(revision_body, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {
        "revision_id": revision_id,
        "revision_path": path,
        "transcript_cache_key": source.get("transcript_cache_key"),
        "source_file_hash": source.get("source_file_hash"),
    }
