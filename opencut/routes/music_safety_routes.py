"""
OpenCut Music & Safety Routes

Blueprint for lip sync verification, rhythm-driven effects,
C2PA content credentials, invisible watermarking, evidence
chain-of-custody, LTC/VITC timecode, and timecode-based sync.
"""

import logging
import os

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
)

logger = logging.getLogger("opencut")

music_safety_bp = Blueprint("music_safety", __name__)


# ===========================================================================
# 16.2 — Lip Sync Verification
# ===========================================================================

@music_safety_bp.route("/video/lip-sync-verify", methods=["POST"])
@require_csrf
@async_job("lip_sync_verify")
def route_lip_sync_verify(job_id, filepath, data):
    """Verify lip sync by comparing audio and mouth movement energy."""
    from opencut.core.lip_sync_verify import verify_lip_sync

    threshold = safe_float(data.get("threshold", 0.3), 0.3, min_val=0.01, max_val=1.0)
    window_ms = safe_float(data.get("window_ms", 40.0), 40.0, min_val=10.0, max_val=200.0)
    min_drift_windows = safe_int(data.get("min_drift_windows", 5), 5, min_val=1, max_val=100)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = verify_lip_sync(
        video_path=filepath,
        threshold=threshold,
        window_ms=window_ms,
        min_drift_windows=min_drift_windows,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/mouth-energy", methods=["POST"])
@require_csrf
@async_job("mouth_energy")
def route_mouth_energy(job_id, filepath, data):
    """Compute mouth movement energy from video frames."""
    from opencut.core.lip_sync_verify import compute_mouth_energy

    window_ms = safe_float(data.get("window_ms", 40.0), 40.0, min_val=10.0, max_val=200.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    energy = compute_mouth_energy(
        video_path=filepath,
        window_ms=window_ms,
        on_progress=_on_progress,
    )
    return {"energy": energy, "samples": len(energy), "window_ms": window_ms}


@music_safety_bp.route("/audio/energy-envelope", methods=["POST"])
@require_csrf
@async_job("audio_energy")
def route_audio_energy(job_id, filepath, data):
    """Compute audio energy envelope."""
    from opencut.core.lip_sync_verify import compute_audio_energy

    window_ms = safe_float(data.get("window_ms", 40.0), 40.0, min_val=10.0, max_val=200.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    energy = compute_audio_energy(
        audio_path=filepath,
        window_ms=window_ms,
        on_progress=_on_progress,
    )
    return {"energy": energy, "samples": len(energy), "window_ms": window_ms}


# ===========================================================================
# 16.3 — Rhythm-Driven Effects
# ===========================================================================

@music_safety_bp.route("/video/rhythm-effects", methods=["POST"])
@require_csrf
@async_job("rhythm_effects")
def route_rhythm_effects(job_id, filepath, data):
    """Apply rhythm-driven visual effects to a video."""
    from opencut.core.rhythm_effects import apply_rhythm_effects, create_effect_map

    audio_path = data.get("audio_path", "").strip() or None
    if audio_path:
        audio_path = validate_filepath(audio_path)

    window_ms = safe_float(data.get("window_ms", 50.0), 50.0, min_val=10.0, max_val=200.0)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    # Parse effect map
    raw_map = data.get("effect_map", None)
    effect_map = None
    if raw_map and isinstance(raw_map, list):
        effect_map = []
        for m in raw_map:
            effect_map.append(create_effect_map(
                audio_feature=m.get("audio_feature", "amplitude"),
                visual_effect=m.get("visual_effect", "zoom"),
                intensity=safe_float(m.get("intensity", 1.0), 1.0, min_val=0.0, max_val=5.0),
                min_value=safe_float(m.get("min_value", 0.0), 0.0),
                max_value=safe_float(m.get("max_value", 1.0), 1.0),
            ))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_rhythm_effects(
        video_path=filepath,
        audio_path=audio_path,
        effect_map=effect_map,
        output=out_path,
        window_ms=window_ms,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/audio/analyze-features", methods=["POST"])
@require_csrf
@async_job("audio_features")
def route_analyze_audio_features(job_id, filepath, data):
    """Analyze audio features for rhythm effect mapping."""
    from opencut.core.rhythm_effects import analyze_audio_features

    features = data.get("features", None)
    window_ms = safe_float(data.get("window_ms", 50.0), 50.0, min_val=10.0, max_val=200.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = analyze_audio_features(
        audio_path=filepath,
        features=features,
        window_ms=window_ms,
        on_progress=_on_progress,
    )
    return {"features": result, "window_ms": window_ms}


@music_safety_bp.route("/video/rhythm-effects/presets", methods=["GET"])
def route_rhythm_presets():
    """Return available audio features and visual effects."""
    from opencut.core.rhythm_effects import AUDIO_FEATURES, VISUAL_EFFECTS

    return jsonify({
        "audio_features": list(AUDIO_FEATURES),
        "visual_effects": list(VISUAL_EFFECTS),
    })


# ===========================================================================
# 27.1 — C2PA Content Credentials
# ===========================================================================

@music_safety_bp.route("/video/c2pa/create", methods=["POST"])
@require_csrf
@async_job("c2pa_create", filepath_required=False)
def route_c2pa_create(job_id, filepath, data):
    """Create a C2PA manifest for content credentials."""
    from opencut.core.c2pa_embed import create_c2pa_manifest

    operations = data.get("operations", [])
    source_hash = data.get("source_hash", "")
    title = data.get("title", "")
    fmt = data.get("format", "video/mp4")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    manifest = create_c2pa_manifest(
        operations=operations,
        source_hash=source_hash,
        title=title,
        format=fmt,
        on_progress=_on_progress,
    )
    return manifest


@music_safety_bp.route("/video/c2pa/embed", methods=["POST"])
@require_csrf
@async_job("c2pa_embed")
def route_c2pa_embed(job_id, filepath, data):
    """Embed a C2PA manifest into a video file."""
    from opencut.core.c2pa_embed import create_c2pa_manifest, embed_c2pa

    # Accept pre-built manifest or create from operations
    manifest = data.get("manifest", None)
    if not manifest:
        operations = data.get("operations", [])
        source_hash = data.get("source_hash", "")
        title = data.get("title", os.path.basename(filepath))
        manifest = create_c2pa_manifest(
            operations=operations,
            source_hash=source_hash,
            title=title,
        )

    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = embed_c2pa(
        video_path=filepath,
        manifest=manifest,
        output=out_path,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/c2pa/read", methods=["POST"])
@require_csrf
@async_job("c2pa_read")
def route_c2pa_read(job_id, filepath, data):
    """Read a C2PA manifest from a video file."""
    from opencut.core.c2pa_embed import read_c2pa

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    manifest = read_c2pa(
        video_path=filepath,
        on_progress=_on_progress,
    )
    return {"manifest": manifest, "has_c2pa": bool(manifest)}


# ===========================================================================
# 27.2 — Invisible AI Watermarking
# ===========================================================================

@music_safety_bp.route("/video/watermark/embed", methods=["POST"])
@require_csrf
@async_job("watermark_embed")
def route_watermark_embed(job_id, filepath, data):
    """Embed an invisible watermark in a video."""
    from opencut.core.invisible_watermark import embed_watermark

    message = data.get("message", "").strip()
    if not message:
        raise ValueError("Watermark message is required")

    strength = safe_float(data.get("strength", 25.0), 25.0, min_val=1.0, max_val=100.0)
    key_frames_only = safe_bool(data.get("key_frames_only", False))
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = embed_watermark(
        video_path=filepath,
        message=message,
        output=out_path,
        strength=strength,
        key_frames_only=key_frames_only,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/watermark/extract", methods=["POST"])
@require_csrf
@async_job("watermark_extract")
def route_watermark_extract(job_id, filepath, data):
    """Extract an invisible watermark from a video."""
    from opencut.core.invisible_watermark import extract_watermark

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_watermark(
        video_path=filepath,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/watermark/verify", methods=["POST"])
@require_csrf
@async_job("watermark_verify")
def route_watermark_verify(job_id, filepath, data):
    """Verify that a video contains the expected watermark."""
    from opencut.core.invisible_watermark import verify_watermark

    expected = data.get("expected_message", "").strip()
    if not expected:
        raise ValueError("Expected watermark message is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = verify_watermark(
        video_path=filepath,
        expected_message=expected,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# 35.3 — Evidence Chain-of-Custody
# ===========================================================================

@music_safety_bp.route("/video/custody/create", methods=["POST"])
@require_csrf
@async_job("custody_create")
def route_custody_create(job_id, filepath, data):
    """Create a new chain-of-custody record for a video."""
    from opencut.core.evidence_chain import create_custody_chain

    operator = data.get("operator", "").strip()
    case_id = data.get("case_id", "").strip()
    notes = data.get("notes", "").strip()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    chain = create_custody_chain(
        video_path=filepath,
        operator=operator,
        case_id=case_id,
        notes=notes,
        on_progress=_on_progress,
    )
    return chain


@music_safety_bp.route("/video/custody/log", methods=["POST"])
@require_csrf
@async_job("custody_log", filepath_required=False)
def route_custody_log(job_id, filepath, data):
    """Log an operation to an existing chain of custody."""
    from opencut.core.evidence_chain import log_operation

    chain = data.get("chain")
    if not chain:
        raise ValueError("Chain data is required")

    operation_data = data.get("operation_data", {})
    if not operation_data.get("operation"):
        raise ValueError("Operation name is required in operation_data")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    updated = log_operation(
        chain=chain,
        operation_data=operation_data,
        on_progress=_on_progress,
    )
    return updated


@music_safety_bp.route("/video/custody/finalize", methods=["POST"])
@require_csrf
@async_job("custody_finalize", filepath_required=False)
def route_custody_finalize(job_id, filepath, data):
    """Finalize a chain of custody."""
    from opencut.core.evidence_chain import finalize_chain

    chain = data.get("chain")
    if not chain:
        raise ValueError("Chain data is required")

    output_hash = data.get("output_hash", "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    finalized = finalize_chain(
        chain=chain,
        output_hash=output_hash,
        on_progress=_on_progress,
    )
    return finalized


@music_safety_bp.route("/video/custody/export", methods=["POST"])
@require_csrf
@async_job("custody_export", filepath_required=False)
def route_custody_export(job_id, filepath, data):
    """Export chain of custody as a report."""
    from opencut.core.evidence_chain import export_custody_report

    chain = data.get("chain")
    if not chain:
        raise ValueError("Chain data is required")

    out_path = data.get("output_path", "").strip()
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path:
        raise ValueError("Output path is required for report export")

    fmt = data.get("format", "json").strip().lower()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_custody_report(
        chain=chain,
        output_path=out_path,
        format=fmt,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# 44.2 — LTC/VITC Timecode Extraction
# ===========================================================================

@music_safety_bp.route("/audio/ltc-extract", methods=["POST"])
@require_csrf
@async_job("ltc_extract")
def route_ltc_extract(job_id, filepath, data):
    """Extract LTC timecode from audio track."""
    from opencut.core.ltc_vitc import extract_ltc

    fps = safe_float(data.get("fps", 0.0), 0.0, min_val=0.0, max_val=120.0)
    channel = safe_int(data.get("channel", -1), -1, min_val=-1, max_val=32)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_ltc(
        audio_path=filepath,
        fps=fps,
        channel=channel,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/vitc-extract", methods=["POST"])
@require_csrf
@async_job("vitc_extract")
def route_vitc_extract(job_id, filepath, data):
    """Extract VITC timecode from video scan lines."""
    from opencut.core.ltc_vitc import extract_vitc

    fps = safe_float(data.get("fps", 0.0), 0.0, min_val=0.0, max_val=120.0)
    scan_lines = data.get("scan_lines", None)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_vitc(
        video_path=filepath,
        scan_lines=scan_lines,
        fps=fps,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# 44.3 — Timecode-Based Sync
# ===========================================================================

@music_safety_bp.route("/video/tc-sync", methods=["POST"])
@require_csrf
@async_job("tc_sync", filepath_required=False)
def route_tc_sync(job_id, filepath, data):
    """Sync multiple sources by matching timecodes."""
    from opencut.core.tc_sync import sync_by_timecode

    sources = data.get("sources", [])
    if not sources:
        raise ValueError("At least one source file path is required")

    # Validate each source path
    validated = []
    for src in sources:
        if isinstance(src, str):
            validated.append(validate_filepath(src.strip()))
        elif isinstance(src, dict):
            validated.append(validate_filepath(src.get("filepath", "").strip()))

    fps = safe_float(data.get("fps", 0.0), 0.0, min_val=0.0, max_val=120.0)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)
    timeline_fmt = data.get("timeline_format", "json").strip().lower()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = sync_by_timecode(
        sources=validated,
        fps=fps,
        output=out_path,
        timeline_format=timeline_fmt,
        on_progress=_on_progress,
    )
    return result


@music_safety_bp.route("/video/tc-offsets", methods=["POST"])
@require_csrf
@async_job("tc_offsets", filepath_required=False)
def route_tc_offsets(job_id, filepath, data):
    """Compute timecode offsets for multiple sources."""
    from opencut.core.tc_sync import compute_tc_offsets

    sources = data.get("sources", [])
    if not sources:
        raise ValueError("Source list is required")

    offsets = compute_tc_offsets(sources)
    return {"offsets": offsets, "source_count": len(sources)}


@music_safety_bp.route("/video/tc-common-range", methods=["POST"])
@require_csrf
@async_job("tc_common_range", filepath_required=False)
def route_tc_common_range(job_id, filepath, data):
    """Find common timecode range across sources."""
    from opencut.core.tc_sync import find_common_timecode_range

    timecodes = data.get("timecodes", [])
    if not timecodes:
        raise ValueError("Timecodes list is required")

    result = find_common_timecode_range(timecodes)
    return result
