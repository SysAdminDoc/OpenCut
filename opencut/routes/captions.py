"""
OpenCut Captions Routes

Transcription, caption styles, burn-in, animated captions, whisperx,
translation, karaoke, format conversion.
"""

# ruff: noqa: F401 -- compatibility facade exports shared contracts and routes.

import copy
import logging
import os
import tempfile
from types import SimpleNamespace

from flask import Blueprint, jsonify, request

from opencut.core.caption_roundtrip import (
    diff_caption_roundtrip,
    parse_srt_text,
    read_caption_sidecar,
    store_caption_revision,
    write_caption_sidecar,
)
from opencut.core.workflow import workflow_step
from opencut.errors import safe_error
from opencut.helpers import _make_sequence_name, _resolve_output_dir
from opencut.jobs import _is_cancelled, _update_job, async_job
from opencut.security import (
    VALID_WHISPER_MODELS,
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
    validate_path,
    verify_destructive_confirm_token,
)

logger = logging.getLogger("opencut")

# Core imports used by multiple routes (try relative/absolute, tolerate missing)
detect_speech = get_edit_summary = generate_zoom_events = None
export_premiere_xml = export_ass = export_json = export_vtt = None
CaptionConfig = ExportConfig = get_preset = _probe_media = LLMConfig = None
try:
    from ..core.llm import LLMConfig  # noqa: F811
    from ..core.silence import detect_speech, get_edit_summary  # noqa: F811
    from ..core.zoom import generate_zoom_events  # noqa: F811
    from ..export.premiere import export_premiere_xml  # noqa: F811
    from ..export.srt import export_ass, export_json, export_vtt  # noqa: F811
    from ..utils.config import CaptionConfig, ExportConfig, get_preset  # noqa: F811
    from ..utils.media import probe as _probe_media  # noqa: F811
except ImportError:
    try:
        from opencut.core.llm import LLMConfig  # noqa: F811
        from opencut.core.silence import detect_speech, get_edit_summary  # noqa: F811
        from opencut.core.zoom import generate_zoom_events  # noqa: F811
        from opencut.export.premiere import export_premiere_xml  # noqa: F811
        from opencut.export.srt import export_ass, export_json, export_vtt  # noqa: F811
        from opencut.utils.config import CaptionConfig, ExportConfig, get_preset  # noqa: F811
        from opencut.utils.media import probe as _probe_media  # noqa: F811
    except ImportError:
        LLMConfig = None
        logger.warning("Some caption dependencies could not be imported")

captions_bp = Blueprint("captions", __name__)

import re as _re  # noqa: E402


def _safe_probe(filepath):
    """Probe media or return None — never raise.

    ``_probe_media`` may itself be ``None`` if the optional ``utils.media``
    import failed at startup (e.g. missing ffprobe), and even when present it
    raises on truncated/missing files.  Routes that only use the result for
    optional metadata (resolution, duration) treat ``None`` as a fallback
    sentinel and substitute defaults instead of crashing the worker.
    """
    if _probe_media is None:
        return None
    try:
        if not filepath or not os.path.isfile(filepath):
            return None
        return _probe_media(filepath)
    except Exception as exc:  # noqa: BLE001 — probing must never break a job
        logger.debug("Media probe failed for %s: %s", filepath, exc)
        return None

_VALID_SUBTITLE_FORMATS = {"srt", "ass", "vtt", "sub", "json"}
_VALID_ANIMATIONS = {"pop", "slide_up", "fade", "bounce", "typewriter", "highlight_box", "glow"}
_VALID_BURNIN_STYLES = {"default", "bold_yellow", "boxed_dark", "neon_cyan", "cinematic_serif", "top_center"}


def _sanitize_force_style(s: str) -> str:
    """Allow only safe ASS style directives (alphanumeric, commas, equals, parens, dots, spaces)."""
    if s and _re.match(r'^[a-zA-Z0-9,=.() -]+$', s):
        return s
    return ""


def _sanitize_font_name(s: str) -> str:
    """Allow only safe font names (alphanumeric, spaces, hyphens, underscores)."""
    if s and _re.match(r'^[a-zA-Z0-9 _-]+$', s):
        return s
    return "Arial"


def _legacy_srt_bom_requested(data: dict) -> bool:
    """Return whether the request opted into a legacy UTF-8 BOM for SRT output."""
    for key in ("srt_legacy_bom", "windows_legacy_bom", "legacy_bom"):
        if key in data:
            return safe_bool(data.get(key), False)
    return False


def _caption_display_settings(data: dict) -> dict | None:
    settings = data.get("display_settings")
    return settings if isinstance(settings, dict) else None


def _write_caption_roundtrip_sidecar(result, out_path: str, sub_format: str, filepath: str, data: dict) -> tuple[str, list[str]]:
    try:
        sidecar_path = write_caption_sidecar(
            result,
            out_path,
            export_format=sub_format,
            source_path=filepath,
            transcript_cache_key=data.get("transcript_cache_key") or getattr(result, "cache_key", None),
            display_settings=_caption_display_settings(data),
        )
    except Exception as exc:  # noqa: BLE001 - metadata sidecars must not block subtitle export.
        logger.warning("Caption sidecar write failed for %s: %s", out_path, exc)
        return "", ["sidecar_write_failed"]
    return sidecar_path, []


def _clean_roundtrip_segments(raw_segments) -> list[dict]:
    if not isinstance(raw_segments, list):
        return []
    cleaned = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        segment = dict(item)
        segment["start"] = safe_float(segment.get("start", 0.0), default=0.0, min_val=0.0)
        segment["end"] = safe_float(segment.get("end", 0.0), default=0.0, min_val=0.0)
        segment["text"] = str(segment.get("text", "")).strip()
        cleaned.append(segment)
    return cleaned


def _snapshot_segments(snapshot) -> list[dict]:
    if not isinstance(snapshot, dict):
        return []
    if isinstance(snapshot.get("segments"), list):
        return _clean_roundtrip_segments(snapshot.get("segments"))
    segments = []
    for track in snapshot.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        for key in ("segments", "cues", "items", "track_items"):
            if isinstance(track.get(key), list):
                segments.extend(track[key])
                break
    return _clean_roundtrip_segments(segments)


def _roundtrip_edited_segments(data: dict) -> tuple[list[dict], list[str]]:
    warnings = []
    for key in ("edited_segments", "segments"):
        if isinstance(data.get(key), list):
            return _clean_roundtrip_segments(data.get(key)), warnings
    snapshot = data.get("caption_track_snapshot")
    if snapshot is not None:
        return _snapshot_segments(snapshot), warnings
    srt_text = data.get("srt_text")
    if isinstance(srt_text, str) and srt_text.strip():
        return parse_srt_text(srt_text), warnings
    srt_path = str(data.get("srt_path", "")).strip()
    if srt_path:
        try:
            srt_path = validate_filepath(srt_path)
            with open(srt_path, encoding="utf-8-sig", errors="replace") as handle:
                return parse_srt_text(handle.read()), warnings
        except (OSError, ValueError):
            warnings.append("srt_unavailable")
    return [], warnings


def _roundtrip_sidecar(data: dict) -> tuple[dict | None, list[str]]:
    sidecar = data.get("sidecar")
    if isinstance(sidecar, dict):
        return sidecar, []
    sidecar_path = str(data.get("sidecar_path", "")).strip()
    if not sidecar_path:
        return None, ["sidecar_missing"]
    try:
        sidecar_path = validate_filepath(sidecar_path)
    except ValueError:
        return None, ["sidecar_missing"]
    expected_export_path = str(data.get("srt_path") or data.get("export_path") or "").strip() or None
    return read_caption_sidecar(sidecar_path, expected_export_path=expected_export_path)


def _roundtrip_diff_from_data(data: dict) -> dict:
    sidecar, sidecar_warnings = _roundtrip_sidecar(data)
    edited_segments, input_warnings = _roundtrip_edited_segments(data)
    if not edited_segments:
        raise ValueError("Provide edited_segments, segments, caption_track_snapshot, srt_text, or srt_path.")
    original_segments = _clean_roundtrip_segments(data.get("original_segments"))
    return diff_caption_roundtrip(
        sidecar=sidecar,
        edited_segments=edited_segments,
        original_segments=original_segments,
        warnings=sidecar_warnings + input_warnings,
    )


def _export_srt_with_policy(result, out_path: str, *, legacy_windows_bom: bool = False) -> str:
    from opencut.export import srt as srt_export

    if legacy_windows_bom:
        return srt_export.export_srt(result, out_path, legacy_windows_bom=True)
    return srt_export.export_srt(result, out_path)


def _word_namespace_from_dict(word_data):
    """Build a word-like object from cached/editable transcript JSON."""
    if not isinstance(word_data, dict):
        return word_data
    text = word_data.get("text", word_data.get("word", ""))
    return SimpleNamespace(
        text=text,
        word=text,
        start=word_data.get("start", 0),
        end=word_data.get("end", 0),
        confidence=word_data.get("confidence", 1.0),
        boundary_confidence=word_data.get("boundary_confidence"),
    )


def _segment_namespace_from_dict(seg_data):
    """Build a segment-like object while preserving ASR review metadata."""
    if not isinstance(seg_data, dict):
        return seg_data
    return SimpleNamespace(
        start=seg_data.get("start", 0),
        end=seg_data.get("end", 0),
        text=seg_data.get("text", ""),
        words=[_word_namespace_from_dict(w) for w in (seg_data.get("words") or [])],
        speaker=seg_data.get("speaker"),
        language=seg_data.get("language"),
        language_confidence=seg_data.get("language_confidence", 1.0),
        confidence=seg_data.get("confidence", 1.0),
        boundary_confidence=seg_data.get("boundary_confidence"),
        human_review_recommended=safe_bool(seg_data.get("human_review_recommended", False), False),
        review_reasons=list(seg_data.get("review_reasons") or []),
    )


def _segment_payload(seg, *, include_words=True, precision=None):
    from opencut.core.captions import caption_segment_to_dict

    return caption_segment_to_dict(
        seg,
        include_words=include_words,
        precision=precision,
    )


def _segment_payloads(segments, *, include_words=True, precision=None):
    return [
        _segment_payload(seg, include_words=include_words, precision=precision)
        for seg in (segments or [])
    ]


def _caption_review_summary(result_or_segments):
    if isinstance(result_or_segments, dict):
        segments = result_or_segments.get("segments", []) or []
    else:
        segments = getattr(result_or_segments, "segments", result_or_segments) or []
    review_count = 0
    low_confidence_count = 0
    language_review_count = 0
    boundary_review_count = 0
    languages = set()
    for seg in segments:
        if isinstance(seg, dict):
            reasons = set(seg.get("review_reasons", []) or [])
            human_review = safe_bool(seg.get("human_review_recommended", False), False)
            language = seg.get("language")
        else:
            reasons = set(getattr(seg, "review_reasons", []) or [])
            human_review = bool(getattr(seg, "human_review_recommended", False))
            language = getattr(seg, "language", None)
        if human_review:
            review_count += 1
        if reasons.intersection({"low_asr_confidence", "low_language_confidence"}):
            low_confidence_count += 1
        if "low_boundary_confidence" in reasons:
            boundary_review_count += 1
        if "language_requires_human_review" in reasons:
            language_review_count += 1
            if language:
                languages.add(str(language))
    return {
        "human_review_recommended": review_count > 0,
        "human_review_segments": review_count,
        "low_confidence_segments": low_confidence_count,
        "language_review_segments": language_review_count,
        "low_boundary_confidence_segments": boundary_review_count,
        "human_review_languages": sorted(languages),
    }


def _asr_provenance_payload(result) -> dict:
    from opencut.core.asr_provenance import provenance_to_dict

    return provenance_to_dict(getattr(result, "provenance", None))

from .caption_analysis_routes import *  # noqa: F401,F403,E402
from .caption_enhancement_routes import (  # noqa: E402
    _segments_to_srt_text,
    _srt_to_translate_segments,
    _validate_enhanced_install_input,
    _validate_segments,
    _validate_translate_input,
)
from .caption_enhancement_routes import *  # noqa: F401,F403,E402
from .caption_generation_routes import *  # noqa: F401,F403,E402
from .caption_interview_routes import *  # noqa: F401,F403,E402
from .caption_pipeline_routes import *  # noqa: F401,F403,E402
from .caption_render_routes import *  # noqa: F401,F403,E402
from .caption_transcript_routes import *  # noqa: F401,F403,E402
