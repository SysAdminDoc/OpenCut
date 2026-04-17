"""
OpenCut Wave A Routes (v1.18.0)

Feature surface backed by new core modules added after the v1.17.0 OSS
research pass:

- ``/audio/tts/f5``                — F5-TTS voice cloning
- ``/audio/beats/beatnet``         — BeatNet downbeat detection backend
- ``/video/scenes/auto``           — TransNetV2 → PySceneDetect → FFmpeg dispatcher
- ``/video/quality/score``         — CLIP-IQA+ per-clip quality
- ``/video/quality/rank``          — rank multiple clips by CLIP-IQA+
- ``/video/emotion/arc``           — HSEmotion emotion timeline
- ``/video/encode/vmaf-target``    — ab-av1 VMAF-target encode
- ``/timeline/export/aaf``         — Avid AAF export
- ``/timeline/export/otioz``       — OTIOZ portable bundle
- ``/events/moments``              — wedding/event moment finder
- ``/captions/compliance/standards`` — list compliance profiles
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_filepath, validate_path

logger = logging.getLogger("opencut")

wave_a_bp = Blueprint("wave_a", __name__)


# ---------------------------------------------------------------------------
# F5-TTS
# ---------------------------------------------------------------------------

@wave_a_bp.route("/audio/tts/f5", methods=["POST"])
@require_csrf
@async_job("tts_f5", filepath_required=False)
def route_f5_tts(job_id, filepath, data):
    """Synthesise text in a cloned voice using F5-TTS."""
    from opencut.core import tts_f5

    text = (data.get("text") or "").strip()
    voice_ref = (data.get("voice_ref") or data.get("ref_file") or "").strip()
    ref_text = data.get("ref_text")
    model = (data.get("model") or tts_f5.DEFAULT_MODEL).strip()
    output = (data.get("output") or data.get("output_path") or "").strip()
    speed = data.get("speed", 1.0)
    if output:
        output = validate_path(output)
    if voice_ref:
        voice_ref = validate_filepath(voice_ref)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = tts_f5.f5_generate(
        text=text,
        voice_ref=voice_ref,
        ref_text=ref_text,
        output_path=output or None,
        model=model,
        speed=speed,
        on_progress=_on_progress,
    )
    return dict(result)


@wave_a_bp.route("/audio/tts/f5/models", methods=["GET"])
def route_f5_models():
    from opencut.core import tts_f5
    return jsonify({
        "models": tts_f5.list_models(),
        "installed": tts_f5.check_f5_available(),
    })


# ---------------------------------------------------------------------------
# BeatNet
# ---------------------------------------------------------------------------

@wave_a_bp.route("/audio/beats/beatnet", methods=["POST"])
@require_csrf
@async_job("beatnet")
def route_beatnet(job_id, filepath, data):
    from opencut.core import beats_beatnet

    mode = str(data.get("mode") or "offline").lower()
    meter = int(data.get("meter", 4) or 4)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    info = beats_beatnet.detect_beats_beatnet(
        filepath, mode=mode, meter=meter, on_progress=_on_progress,
    )
    return {
        "bpm": info.bpm,
        "beat_times": info.beat_times,
        "downbeat_times": info.downbeat_times,
        "total_beats": len(info.beat_times),
        "confidence": info.confidence,
        "mode": mode,
        "meter": meter,
    }


# ---------------------------------------------------------------------------
# Scene detection auto-dispatcher
# ---------------------------------------------------------------------------

@wave_a_bp.route("/video/scenes/auto", methods=["POST"])
@require_csrf
@async_job("scenes_auto")
def route_scenes_auto(job_id, filepath, data):
    from opencut.core import scene_detect
    from opencut.security import safe_float

    threshold = safe_float(data.get("threshold", 0.4), 0.4, min_val=0.0, max_val=1.0)
    min_scene = safe_float(
        data.get("min_scene_length", 1.0), 1.0, min_val=0.1, max_val=60.0,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    info = scene_detect.detect_scenes_auto(
        filepath,
        threshold=threshold,
        min_scene_length=min_scene,
        on_progress=_on_progress,
    )
    return {
        "boundaries": [
            {"time": b.time, "frame": b.frame, "score": b.score, "label": b.label}
            for b in info.boundaries
        ],
        "total_scenes": info.total_scenes,
        "duration": info.duration,
        "avg_scene_length": info.avg_scene_length,
    }


# ---------------------------------------------------------------------------
# Clip quality scoring (CLIP-IQA+)
# ---------------------------------------------------------------------------

@wave_a_bp.route("/video/quality/score", methods=["POST"])
@require_csrf
@async_job("clip_quality")
def route_quality_score(job_id, filepath, data):
    from opencut.core import clip_quality
    from opencut.security import safe_float, safe_int

    axes = data.get("axes") or list(clip_quality.DEFAULT_AXES)
    axes = [a for a in axes if isinstance(a, str)][:8] or list(clip_quality.DEFAULT_AXES)
    fps_sample = safe_float(data.get("fps_sample", 1.0), 1.0, min_val=0.1, max_val=30.0)
    max_frames = safe_int(data.get("max_frames", 60), 60, min_val=5, max_val=600)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = clip_quality.score_clip(
        filepath,
        axes=axes,
        fps_sample=fps_sample,
        max_frames=max_frames,
        on_progress=_on_progress,
    )
    return dict(result)


@wave_a_bp.route("/video/quality/rank", methods=["POST"])
@require_csrf
@async_job("clip_quality_rank", filepath_required=False)
def route_quality_rank(job_id, filepath, data):
    from opencut.core import clip_quality

    paths = data.get("filepaths") or []
    if not isinstance(paths, list) or not paths:
        raise ValueError("filepaths: list of validated file paths required")
    cleaned = []
    for p in paths[:40]:
        if not isinstance(p, str):
            continue
        try:
            cleaned.append(validate_filepath(p))
        except ValueError:
            continue
    if not cleaned:
        raise ValueError("no valid filepaths after validation")

    axes = data.get("axes") or list(clip_quality.DEFAULT_AXES)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    results = clip_quality.rank_clips(
        cleaned, axes=axes, on_progress=_on_progress,
    )
    return {
        "ranked": [dict(r) for r in results],
        "count": len(results),
    }


# ---------------------------------------------------------------------------
# Emotion arc (HSEmotion)
# ---------------------------------------------------------------------------

@wave_a_bp.route("/video/emotion/arc", methods=["POST"])
@require_csrf
@async_job("emotion_arc")
def route_emotion_arc(job_id, filepath, data):
    from opencut.core import emotion_arc
    from opencut.security import safe_float, safe_int

    fps_sample = safe_float(data.get("fps_sample", 1.0), 1.0, min_val=0.1, max_val=30.0)
    max_frames = safe_int(data.get("max_frames", 120), 120, min_val=10, max_val=1200)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    arc = emotion_arc.analyse_emotion_arc(
        filepath,
        fps_sample=fps_sample,
        max_frames=max_frames,
        on_progress=_on_progress,
    )
    return {
        "filepath": arc.filepath,
        "samples": [
            {
                "t": s.t, "dominant": s.dominant,
                "probabilities": s.probabilities,
                "face_found": s.face_found,
            }
            for s in arc.samples
        ],
        "dominant_overall": arc.dominant_overall,
        "mean_probabilities": arc.mean_probabilities,
        "transitions": arc.transitions,
        "emotional_range": arc.emotional_range,
        "duration": arc.duration,
        "notes": arc.notes,
    }


# ---------------------------------------------------------------------------
# ab-av1 VMAF-target encode
# ---------------------------------------------------------------------------

@wave_a_bp.route("/video/encode/vmaf-target", methods=["POST"])
@require_csrf
@async_job("ab_av1")
def route_vmaf_encode(job_id, filepath, data):
    from opencut.core import ab_av1
    from opencut.security import safe_float, safe_int

    target_vmaf = safe_float(
        data.get("target_vmaf", 95.0), 95.0, min_val=20.0, max_val=99.5,
    )
    encoder = str(data.get("encoder") or "libsvtav1").strip().lower()
    preset = data.get("preset")
    if preset is not None:
        preset = safe_int(preset, 8, min_val=0, max_val=13)
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = ab_av1.encode_to_vmaf(
        filepath,
        target_vmaf=target_vmaf,
        encoder=encoder,
        preset=preset,
        output=output or None,
        on_progress=_on_progress,
    )
    return dict(result)


@wave_a_bp.route("/video/encode/vmaf-target/info", methods=["GET"])
def route_vmaf_info():
    from opencut.core import ab_av1
    return jsonify({
        "installed": ab_av1.check_ab_av1_available(),
        "version": ab_av1.version(),
        "encoders": sorted(ab_av1.SUPPORTED_ENCODERS),
    })


# ---------------------------------------------------------------------------
# Timeline export (AAF / OTIOZ)
# ---------------------------------------------------------------------------

def _parse_segments(raw_segments):
    from opencut.core.silence import TimeSegment
    parsed = []
    for i, seg in enumerate(raw_segments or []):
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        label = str(seg.get("label") or "speech")[:64]
        parsed.append(TimeSegment(start=start, end=end, label=label))
    return parsed


@wave_a_bp.route("/timeline/export/aaf", methods=["POST"])
@require_csrf
def route_export_aaf():
    try:
        from opencut.export import otio_export

        data = request.get_json(force=True) or {}
        filepath = validate_filepath(str(data.get("filepath") or ""))
        segments = _parse_segments(data.get("segments"))
        if not segments:
            return jsonify({
                "error": "segments: non-empty list of {start,end[,label]} required",
                "code": "INVALID_INPUT",
            }), 400
        output = str(data.get("output") or os.path.splitext(filepath)[0] + ".aaf").strip()
        output = validate_path(output)
        framerate = float(data.get("framerate") or 24.0)
        seq = str(data.get("sequence_name") or "OpenCut Edit")[:120]

        path = otio_export.export_aaf(
            filepath, segments, output,
            sequence_name=seq, framerate=framerate,
        )
        return jsonify({
            "output": path,
            "clip_count": len(segments),
            "adapter": "aaf",
        })
    except ImportError as exc:
        return jsonify({
            "error": str(exc),
            "code": "MISSING_DEPENDENCY",
            "suggestion": "pip install otio-aaf-adapter",
        }), 503
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "export_aaf")


@wave_a_bp.route("/timeline/export/otioz", methods=["POST"])
@require_csrf
def route_export_otioz():
    try:
        from opencut.export import otio_export

        data = request.get_json(force=True) or {}
        filepath = validate_filepath(str(data.get("filepath") or ""))
        segments = _parse_segments(data.get("segments"))
        if not segments:
            return jsonify({
                "error": "segments: non-empty list of {start,end[,label]} required",
                "code": "INVALID_INPUT",
            }), 400
        output = str(data.get("output") or os.path.splitext(filepath)[0] + ".otioz").strip()
        output = validate_path(output)
        framerate = float(data.get("framerate") or 24.0)
        seq = str(data.get("sequence_name") or "OpenCut Edit")[:120]
        bundle_media = bool(data.get("bundle_media"))

        path = otio_export.export_otioz(
            filepath, segments, output,
            sequence_name=seq, framerate=framerate,
            bundle_media=bundle_media,
        )
        return jsonify({
            "output": path,
            "clip_count": len(segments),
            "adapter": "otioz",
            "media_bundled": bundle_media,
        })
    except ImportError as exc:
        return jsonify({
            "error": str(exc),
            "code": "MISSING_DEPENDENCY",
            "suggestion": "pip install opentimelineio",
        }), 503
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "export_otioz")


# ---------------------------------------------------------------------------
# Event moments
# ---------------------------------------------------------------------------

@wave_a_bp.route("/events/moments", methods=["POST"])
@require_csrf
@async_job("event_moments")
def route_event_moments(job_id, filepath, data):
    from opencut.core import event_moments
    from opencut.security import safe_float, safe_int

    mode = str(data.get("mode") or "heuristic").lower()
    k_sigma = safe_float(data.get("k_sigma", 2.0), 2.0, min_val=0.5, max_val=6.0)
    min_spacing = safe_float(
        data.get("min_spacing", 8.0), 8.0, min_val=0.5, max_val=600.0,
    )
    max_moments = safe_int(data.get("max_moments", 20), 20, min_val=1, max_val=200)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = event_moments.find_event_moments(
        filepath,
        mode=mode,
        k_sigma=k_sigma,
        min_spacing=min_spacing,
        max_moments=max_moments,
        on_progress=_on_progress,
    )
    return {
        "filepath": result.filepath,
        "moments": [
            {
                "t": m.t, "label": m.label, "score": m.score,
                "surrounding_energy": m.surrounding_energy,
                "notes": m.notes,
            }
            for m in result.moments
        ],
        "duration": result.duration,
        "mode": result.mode,
        "notes": result.notes,
    }


# ---------------------------------------------------------------------------
# Caption compliance standards list
# ---------------------------------------------------------------------------

@wave_a_bp.route("/captions/compliance/standards", methods=["GET"])
def route_compliance_standards():
    """Return the available compliance profiles and their rule sets."""
    from opencut.core.caption_compliance import STANDARDS
    return jsonify({
        "standards": [
            {"name": name, **{k: v for k, v in rules.items()}}
            for name, rules in STANDARDS.items()
        ],
        "default": "netflix",
    })
