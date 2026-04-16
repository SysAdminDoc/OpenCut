"""
OpenCut AI Editing Routes

Endpoints for AI-powered editing features:
- Eye contact correction
- AI overdub / voice correction
- Lip sync generation
- Voice conversion (RVC)
- B-roll suggestion
- Morph cut / smooth jump cut
- Frame extension / outpainting
- AI storyboard generator
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
    validate_path,
)

logger = logging.getLogger("opencut")

ai_editing_bp = Blueprint("ai_editing", __name__)


# ---------------------------------------------------------------------------
# POST /ai/eye-contact
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/eye-contact", methods=["POST"])
@require_csrf
@async_job("eye_contact")
def ai_eye_contact(job_id, filepath, data):
    """Correct eye contact in video to make subject look at camera."""
    from opencut.core.eye_contact import EyeContactConfig, correct_eye_contact

    config = EyeContactConfig(
        strength=safe_float(data.get("strength"), 0.7, 0.0, 1.0),
        smoothing_window=safe_int(data.get("smoothing_window"), 5, 1, 30),
        max_yaw_correction=safe_float(data.get("max_yaw"), 25.0, 1.0, 60.0),
        max_pitch_correction=safe_float(data.get("max_pitch"), 15.0, 1.0, 40.0),
        face_confidence=safe_float(data.get("face_confidence"), 0.5, 0.1, 1.0),
        only_center_face=safe_bool(data.get("only_center_face"), True),
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = correct_eye_contact(
        video_path=filepath,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "frames_processed": result.frames_processed,
        "frames_corrected": result.frames_corrected,
        "avg_gaze_offset": round(result.avg_gaze_offset, 2),
    }


# ---------------------------------------------------------------------------
# POST /ai/overdub
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/overdub", methods=["POST"])
@require_csrf
@async_job("overdub")
def ai_overdub(job_id, filepath, data):
    """Replace audio segment with AI-generated speech."""
    from opencut.core.overdub import OverdubConfig, overdub_segment

    start = safe_float(data.get("start"), None)
    end = safe_float(data.get("end"), None)
    new_text = data.get("new_text", "").strip()

    if start is None or end is None:
        raise ValueError("start and end times are required")
    if not new_text:
        raise ValueError("new_text is required")

    config = OverdubConfig(
        crossfade_ms=safe_int(data.get("crossfade_ms"), 150, 10, 1000),
        voice_clone_seconds=safe_float(data.get("voice_clone_seconds"), 10.0, 2.0, 30.0),
        tts_backend=data.get("tts_backend", "edge").strip() or "edge",
        language=data.get("language", "en").strip() or "en",
        speed=safe_float(data.get("speed"), 1.0, 0.5, 2.0),
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = overdub_segment(
        video_path=filepath,
        start=start,
        end=end,
        new_text=new_text,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "segment_start": result.segment_start,
        "segment_end": result.segment_end,
        "new_text": result.new_text,
        "tts_backend_used": result.tts_backend_used,
        "duration_generated": round(result.duration_generated, 2),
    }


# ---------------------------------------------------------------------------
# POST /ai/lip-sync
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/lip-sync", methods=["POST"])
@require_csrf
@async_job("lip_sync")
def ai_lip_sync(job_id, filepath, data):
    """Apply lip sync to video with replacement audio."""
    from opencut.core.lip_sync_gen import LipSyncConfig, apply_lip_sync

    audio_path = data.get("audio_path", "").strip()
    if not audio_path:
        raise ValueError("audio_path is required")
    audio_path = validate_filepath(audio_path)

    config = LipSyncConfig(
        face_confidence=safe_float(data.get("face_confidence"), 0.5, 0.1, 1.0),
        blend_radius=safe_int(data.get("blend_radius"), 15, 5, 50),
        jaw_sensitivity=safe_float(data.get("jaw_sensitivity"), 1.0, 0.1, 3.0),
        smooth_frames=safe_int(data.get("smooth_frames"), 3, 1, 15),
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_lip_sync(
        video_path=filepath,
        audio_path=audio_path,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "frames_processed": result.frames_processed,
        "frames_synced": result.frames_synced,
        "audio_duration": round(result.audio_duration, 2),
    }


# ---------------------------------------------------------------------------
# POST /ai/voice-convert
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/voice-convert", methods=["POST"])
@require_csrf
@async_job("voice_convert")
def ai_voice_convert(job_id, filepath, data):
    """Convert speaker voice using a target voice model."""
    from opencut.core.voice_conversion import VoiceConversionConfig, convert_voice

    model_path = data.get("model_path", "").strip()
    if not model_path:
        raise ValueError("model_path is required (path or model name)")

    config = VoiceConversionConfig(
        pitch_shift=safe_int(data.get("pitch_shift"), 0, -12, 12),
        index_rate=safe_float(data.get("index_rate"), 0.75, 0.0, 1.0),
        filter_radius=safe_int(data.get("filter_radius"), 3, 1, 7),
        rms_mix_rate=safe_float(data.get("rms_mix_rate"), 0.25, 0.0, 1.0),
        protect=safe_float(data.get("protect"), 0.33, 0.0, 0.5),
        f0_method=data.get("f0_method", "harvest").strip() or "harvest",
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = convert_voice(
        audio_path=filepath,
        model_path=model_path,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "model_used": result.model_used,
        "duration": round(result.duration, 2),
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# GET /ai/voice-models
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/voice-models", methods=["GET"])
def ai_voice_models():
    """List available RVC voice models."""
    from opencut.core.voice_conversion import list_voice_models

    models = list_voice_models()
    return jsonify({"models": models, "count": len(models)})


# ---------------------------------------------------------------------------
# POST /ai/broll-suggest
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/broll-suggest", methods=["POST"])
@require_csrf
@async_job("broll_suggest", filepath_required=False)
def ai_broll_suggest(job_id, filepath, data):
    """Analyze transcript and suggest B-roll clips."""
    from opencut.core.broll_suggest import suggest_broll

    transcript = data.get("transcript")
    if not transcript or not isinstance(transcript, list):
        raise ValueError("transcript is required (list of {start, end, text} segments)")

    footage_index = data.get("footage_index")
    if footage_index is not None and not isinstance(footage_index, list):
        raise ValueError("footage_index must be a list of clip metadata dicts")

    min_gap = safe_float(data.get("min_gap"), 1.0, 0.3, 10.0)
    max_results = safe_int(data.get("max_results"), 20, 1, 100)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = suggest_broll(
        transcript=transcript,
        footage_index=footage_index,
        min_gap=min_gap,
        max_results=max_results,
        on_progress=_on_progress,
    )

    cues = []
    for cue in result.cues:
        cues.append({
            "start": cue.start,
            "end": cue.end,
            "cue_type": cue.cue_type,
            "keywords": cue.keywords,
            "category": cue.category,
            "text": cue.text,
            "score": round(cue.score, 3),
        })

    matches = []
    for m in result.matches:
        matches.append({
            "cue_index": m.cue_index,
            "clip_path": m.clip_path,
            "clip_name": m.clip_name,
            "relevance": round(m.relevance, 3),
            "in_point": m.in_point,
            "out_point": m.out_point,
            "match_keywords": m.match_keywords,
        })

    return {
        "cues": cues,
        "matches": matches,
        "unmatched_cues": result.unmatched_cues,
        "total_broll_duration": result.total_broll_duration,
        "coverage_ratio": result.coverage_ratio,
    }


# ---------------------------------------------------------------------------
# POST /ai/morph-cut
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/morph-cut", methods=["POST"])
@require_csrf
@async_job("morph_cut")
def ai_morph_cut(job_id, filepath, data):
    """Apply morph cut at a jump cut point in video."""
    from opencut.core.morph_cut import MorphCutConfig, apply_morph_cut

    cut_point = safe_float(data.get("cut_point"), None)
    if cut_point is None:
        raise ValueError("cut_point (seconds) is required")

    config = MorphCutConfig(
        transition_frames=safe_int(data.get("transition_frames"), 8, 2, 30),
        blend_mode=data.get("blend_mode", "optical_flow").strip() or "optical_flow",
        face_weight=safe_float(data.get("face_weight"), 0.7, 0.0, 1.0),
        background_weight=safe_float(data.get("background_weight"), 0.3, 0.0, 1.0),
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_morph_cut(
        video_path=filepath,
        cut_point=cut_point,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "cut_point_frame": result.cut_point_frame,
        "frames_interpolated": result.frames_interpolated,
        "face_detected": result.face_detected,
        "method_used": result.method_used,
    }


# ---------------------------------------------------------------------------
# POST /ai/extend-spatial
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/extend-spatial", methods=["POST"])
@require_csrf
@async_job("extend_spatial")
def ai_extend_spatial(job_id, filepath, data):
    """Extend video frame to a new aspect ratio (outpainting)."""
    from opencut.core.frame_extension import FrameExtensionConfig, extend_frame_spatial

    target_aspect = data.get("target_aspect", "16:9").strip()
    fill_method = data.get("fill_method", "reflect").strip()
    if fill_method not in ("reflect", "blur", "replicate", "inpaint"):
        fill_method = "reflect"

    config = FrameExtensionConfig(
        fill_method=fill_method,
        blur_strength=safe_int(data.get("blur_strength"), 51, 3, 151),
        inpaint_radius=safe_int(data.get("inpaint_radius"), 5, 1, 20),
    )
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extend_frame_spatial(
        video_path=filepath,
        target_aspect=target_aspect,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "original_aspect": result.original_aspect,
        "target_aspect": result.target_aspect,
        "original_size": list(result.original_size),
        "output_size": list(result.output_size),
        "frames_processed": result.frames_processed,
        "fill_method": result.fill_method,
    }


# ---------------------------------------------------------------------------
# POST /ai/extend-temporal
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/extend-temporal", methods=["POST"])
@require_csrf
@async_job("extend_temporal")
def ai_extend_temporal(job_id, filepath, data):
    """Extend video duration by generating extra frames."""
    from opencut.core.frame_extension import FrameExtensionConfig, extend_frame_temporal

    extra_seconds = safe_float(data.get("extra_seconds"), 2.0, 0.1, 30.0)
    position = data.get("position", "end").strip()
    if position not in ("start", "end", "both"):
        position = "end"

    temporal_method = data.get("method", "hold").strip()
    if temporal_method not in ("hold", "reverse", "flow"):
        temporal_method = "hold"

    config = FrameExtensionConfig(temporal_method=temporal_method)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extend_frame_temporal(
        video_path=filepath,
        extra_seconds=extra_seconds,
        position=position,
        output_path=out_path,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "extra_seconds": result.extra_seconds,
        "frames_added": result.frames_added,
        "method": result.method,
        "total_duration": round(result.total_duration, 2),
    }


# ---------------------------------------------------------------------------
# POST /ai/storyboard
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/storyboard", methods=["POST"])
@require_csrf
@async_job("storyboard", filepath_required=False)
def ai_storyboard(job_id, filepath, data):
    """Generate storyboard from script text."""
    from opencut.core.ai_storyboard import generate_storyboard

    script_text = data.get("script_text", "").strip()
    if not script_text:
        raise ValueError("script_text is required")

    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    if not output_dir:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="opencut_storyboard_")

    columns = safe_int(data.get("columns"), 3, 1, 6)
    panel_width = safe_int(data.get("panel_width"), 640, 320, 1920)
    panel_height = safe_int(data.get("panel_height"), 360, 180, 1080)
    export_pdf = safe_bool(data.get("export_pdf"), True)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_storyboard(
        script_text=script_text,
        output_dir=output_dir,
        columns=columns,
        panel_width=panel_width,
        panel_height=panel_height,
        export_pdf=export_pdf,
        on_progress=_on_progress,
    )

    panels = []
    for p in result.panels:
        panels.append({
            "shot_number": p.shot_number,
            "image_path": p.image_path,
            "shot_type": p.shot_type,
            "camera_direction": p.camera_direction,
            "description": p.description,
        })

    return {
        "total_shots": result.total_shots,
        "panels": panels,
        "grid_path": result.grid_path,
        "pdf_path": result.pdf_path,
    }


# ---------------------------------------------------------------------------
# POST /ai/detect-edges (utility for frame extension)
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/detect-edges", methods=["POST"])
@require_csrf
def ai_detect_edges():
    """Analyze frame edge regions for extension suitability."""
    from opencut.helpers import ensure_package

    data = request.get_json(force=True) or {}
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "filepath is required"}), 400
    filepath = validate_filepath(filepath)

    frame_time = safe_float(data.get("frame_time"), 0.0, 0.0)

    if not ensure_package("cv2", "opencv-python-headless"):
        return jsonify({"error": "opencv-python-headless is required"}), 500

    import cv2

    from opencut.core.frame_extension import detect_edge_regions

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({"error": f"Cannot open video: {filepath}"}), 400

    if frame_time > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps))

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Cannot read frame from video"}), 400

    result = detect_edge_regions(frame)
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /ai/parse-script (utility for storyboard)
# ---------------------------------------------------------------------------
@ai_editing_bp.route("/ai/parse-script", methods=["POST"])
@require_csrf
def ai_parse_script():
    """Parse script text into shot descriptions (preview before generating)."""
    from opencut.core.ai_storyboard import parse_shot_descriptions

    data = request.get_json(force=True) or {}
    script_text = data.get("script_text", "").strip()
    if not script_text:
        return jsonify({"error": "script_text is required"}), 400

    shots = parse_shot_descriptions(script_text)

    result = []
    for s in shots:
        result.append({
            "shot_number": s.shot_number,
            "shot_type": s.shot_type,
            "description": s.description,
            "action": s.action,
            "camera_direction": s.camera_direction,
            "dialogue": s.dialogue,
        })

    return jsonify({"shots": result, "count": len(result)})
