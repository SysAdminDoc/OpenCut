"""
OpenCut Video Specialty Routes

Depth effects, B-roll, titles, object removal, shorts pipeline,
install endpoints.
"""

import logging
import tempfile

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.helpers import (
    _resolve_output_dir,
)
from opencut.jobs import (
    _is_cancelled,
    _update_job,
    async_job,
    make_install_route,
)
from opencut.security import (
    VALID_WHISPER_MODELS,
    rate_limit,
    rate_limit_release,
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_path,
)

logger = logging.getLogger("opencut")

video_specialty_bp = Blueprint("video_specialty", __name__)


# ---------------------------------------------------------------------------
# Object Removal
# ---------------------------------------------------------------------------
@video_specialty_bp.route("/video/remove/capabilities", methods=["GET"])
def removal_capabilities():
    try:
        from opencut.core.object_removal import get_removal_capabilities
        return jsonify(get_removal_capabilities())
    except Exception as e:
        return safe_error(e, "removal_capabilities")


@video_specialty_bp.route("/video/remove/watermark", methods=["POST"])
@require_csrf
@async_job("remove_watermark")
def remove_watermark_route(job_id, filepath, data):
    """Remove watermark using delogo or LaMA."""
    region = data.get("region", {})
    method = data.get("method", "delogo")
    if method not in ("delogo", "lama"):
        method = "delogo"
    if not region:
        raise ValueError("No region specified")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    if method == "lama":
        from opencut.core.object_removal import remove_watermark_lama
        out = remove_watermark_lama(filepath, region, output_dir=d, on_progress=_p)
    else:
        from opencut.core.object_removal import remove_watermark_delogo
        out = remove_watermark_delogo(filepath, region, output_dir=d, on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Motion Graphics / Titles
# ---------------------------------------------------------------------------
@video_specialty_bp.route("/video/title/presets", methods=["GET"])
def title_presets():
    try:
        from opencut.core.motion_graphics import get_title_presets
        return jsonify({"presets": get_title_presets()})
    except Exception as e:
        return safe_error(e, "title_presets")


@video_specialty_bp.route("/video/title/render", methods=["POST"])
@require_csrf
@async_job("title_render", filepath_required=False)
def title_render(job_id, filepath, data):
    """Render a standalone title card video."""
    text = data.get("text", "").strip()
    if not text:
        raise ValueError("No text")
    if len(text) > 500:
        raise ValueError("Title text too long (max 500 chars)")
    subtitle = data.get("subtitle", "")
    if len(subtitle) > 500:
        raise ValueError("Subtitle too long (max 500 chars)")

    from opencut.core.motion_graphics import render_title_card

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = data.get("output_dir", "")
    if d:
        try:
            d = validate_path(d)
        except ValueError as e:
            raise ValueError(str(e))
    else:
        d = tempfile.gettempdir()
    _title_preset = data.get("preset", "fade_center")
    if _title_preset not in ("fade_center", "slide_left", "typewriter", "lower_third", "countdown", "kinetic_bounce"):
        _title_preset = "fade_center"
    out = render_title_card(text, output_dir=d,
                             preset=_title_preset,
                             duration=safe_float(data.get("duration", 5.0), 5.0, min_val=0.5, max_val=60.0),
                             font_size=safe_int(data.get("font_size", 72), 72, min_val=8, max_val=500),
                             subtitle=subtitle,
                             on_progress=_p)
    return {"output_path": out}


@video_specialty_bp.route("/video/title/overlay", methods=["POST"])
@require_csrf
@async_job("title_overlay")
def title_overlay(job_id, filepath, data):
    """Overlay animated title onto existing video."""
    text = data.get("text", "").strip()
    if not text:
        raise ValueError("No text")
    if len(text) > 500:
        raise ValueError("Title text too long (max 500 chars)")

    from opencut.core.motion_graphics import overlay_title

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    _title_preset2 = data.get("preset", "fade_center")
    if _title_preset2 not in ("fade_center", "slide_left", "typewriter", "lower_third", "countdown", "kinetic_bounce"):
        _title_preset2 = "fade_center"
    out = overlay_title(filepath, text, output_dir=d,
                         preset=_title_preset2,
                         font_size=safe_int(data.get("font_size", 72), 72, min_val=8, max_val=500),
                         start_time=safe_float(data.get("start_time", 0), 0.0, min_val=0.0, max_val=86400.0),
                         duration=safe_float(data.get("duration", 5.0), 5.0, min_val=0.5, max_val=60.0),
                         subtitle=data.get("subtitle", "")[:500],
                         on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# One-Click Shorts Pipeline
# ---------------------------------------------------------------------------
@video_specialty_bp.route("/video/shorts-pipeline", methods=["POST"])
@require_csrf
@async_job("shorts_pipeline")
def video_shorts_pipeline(job_id, filepath, data):
    """Generate short-form clips from a long video (transcribe + highlight + reframe + captions)."""
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    acquired = rate_limit("ai_gpu")
    if not acquired:
        raise ValueError("A ai_gpu operation is already running. Please wait.")
    try:
        from opencut.core.llm import LLMConfig
        from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

        def _on_progress(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        _shorts_provider = data.get("llm_provider", "ollama")
        if _shorts_provider not in ("ollama", "openai", "anthropic", "gemini"):
            _shorts_provider = "ollama"
        llm_config = LLMConfig(
            provider=_shorts_provider,
            model=data.get("llm_model", ""),
            api_key=data.get("llm_api_key", ""),
            base_url=data.get("llm_base_url", ""),
        )

        _shorts_whisper = data.get("whisper_model", "base")
        if _shorts_whisper not in VALID_WHISPER_MODELS:
            _shorts_whisper = "base"
        config = ShortsPipelineConfig(
            whisper_model=_shorts_whisper,
            max_shorts=safe_int(data.get("max_shorts", 5), 5, min_val=1, max_val=20),
            min_duration=safe_float(data.get("min_duration", 15.0), 15.0, min_val=5.0, max_val=300.0),
            max_duration=safe_float(data.get("max_duration", 60.0), 60.0, min_val=10.0, max_val=600.0),
            target_w=safe_int(data.get("width", 1080), 1080, min_val=100, max_val=7680),
            target_h=safe_int(data.get("height", 1920), 1920, min_val=100, max_val=7680),
            face_track=safe_bool(data.get("face_track", True), True),
            burn_captions=safe_bool(data.get("burn_captions", True), True),
            caption_style=data.get("caption_style", "default") if data.get("caption_style", "default") in ("default", "bold_yellow", "boxed_dark", "neon_cyan", "cinematic_serif", "top_center") else "default",
            llm_provider=llm_config.provider,
            llm_model=llm_config.model,
            llm_api_key=llm_config.api_key,
            llm_base_url=llm_config.base_url,
        )

        clips = generate_shorts(
            filepath,
            config=config,
            output_dir=output_dir,
            on_progress=_on_progress,
        )

        return {
            "clips": [
                {
                    "index": c.index,
                    "output_path": c.output_path,
                    "start": c.start,
                    "end": c.end,
                    "duration": c.duration,
                    "title": c.title,
                    "score": round(c.score, 3) if c.score else 0,
                    "engagement": {
                        "hook_strength": getattr(c.engagement, "hook_strength", 0),
                        "emotional_peak": getattr(c.engagement, "emotional_peak", 0),
                        "pacing": getattr(c.engagement, "pacing", 0),
                        "quotability": getattr(c.engagement, "quotability", 0),
                        "overall": getattr(c.engagement, "overall", 0),
                    } if getattr(c, "engagement", None) else None,
                }
                for c in clips
            ],
            "total_clips": len(clips),
        }
    finally:
        if acquired:
            rate_limit_release("ai_gpu")


# ---------------------------------------------------------------------------
# Video: Depth Effects (Depth Anything V2)
# ---------------------------------------------------------------------------
@video_specialty_bp.route("/video/depth/map", methods=["POST"])
@require_csrf
@async_job("depth_map")
def video_depth_map(job_id, filepath, data):
    """Generate a depth map video using Depth Anything V2."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    model_size = data.get("model_size", "small")
    if model_size not in ("small", "base", "large"):
        model_size = "small"

    acquired = rate_limit("gpu_job")
    if not acquired:
        raise ValueError("A gpu_job operation is already running. Please wait.")
    try:
        from opencut.core.depth_effects import estimate_depth_map

        def _on_progress(pct, msg=""):
            if _is_cancelled(job_id):
                raise InterruptedError("Job cancelled")
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = estimate_depth_map(filepath, output_dir=effective_dir, model_size=model_size, on_progress=_on_progress)
        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("gpu_job")


@video_specialty_bp.route("/video/depth/bokeh", methods=["POST"])
@require_csrf
@async_job("depth_bokeh")
def video_depth_bokeh(job_id, filepath, data):
    """Apply depth-of-field (bokeh) simulation using depth estimation."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    focus_point = safe_float(data.get("focus_point", 0.5), 0.5, min_val=0.0, max_val=1.0)
    blur_strength = safe_int(data.get("blur_strength", 25), 25, min_val=3, max_val=99)
    model_size = data.get("model_size", "small")
    if model_size not in ("small", "base", "large"):
        model_size = "small"

    acquired = rate_limit("gpu_job")
    if not acquired:
        raise ValueError("A gpu_job operation is already running. Please wait.")
    try:
        from opencut.core.depth_effects import apply_bokeh_effect

        def _on_progress(pct, msg=""):
            if _is_cancelled(job_id):
                raise InterruptedError("Job cancelled")
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = apply_bokeh_effect(
            filepath, output_dir=effective_dir,
            focus_point=focus_point, blur_strength=blur_strength,
            model_size=model_size, on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("gpu_job")


@video_specialty_bp.route("/video/depth/parallax", methods=["POST"])
@require_csrf
@async_job("depth_parallax")
def video_depth_parallax(job_id, filepath, data):
    """Apply 3D parallax zoom (Ken Burns) effect using depth estimation."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    zoom_amount = safe_float(data.get("zoom_amount", 1.15), 1.15, min_val=1.01, max_val=2.0)
    model_size = data.get("model_size", "small")
    if model_size not in ("small", "base", "large"):
        model_size = "small"

    acquired = rate_limit("gpu_job")
    if not acquired:
        raise ValueError("A gpu_job operation is already running. Please wait.")
    try:
        from opencut.core.depth_effects import apply_parallax_zoom

        def _on_progress(pct, msg=""):
            if _is_cancelled(job_id):
                raise InterruptedError("Job cancelled")
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = apply_parallax_zoom(
            filepath, output_dir=effective_dir,
            zoom_amount=zoom_amount, model_size=model_size,
            on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("gpu_job")


# ---------------------------------------------------------------------------
# Video: Auto B-Roll Insertion Analysis
# ---------------------------------------------------------------------------
@video_specialty_bp.route("/video/broll-plan", methods=["POST"])
@require_csrf
@async_job("broll_plan")
def video_broll_plan(job_id, filepath, data):
    """Analyze transcript to identify B-roll insertion opportunities.

    Requires a transcribed clip. Returns a list of time windows where B-roll
    would naturally fit, with suggested search keywords for matching.
    """
    min_gap = safe_float(data.get("min_gap", 1.0), 1.0, min_val=0.3, max_val=10.0)
    max_results = safe_int(data.get("max_results", 15), 15, min_val=1, max_val=50)

    _update_job(job_id, progress=5, message="Transcribing for B-roll analysis...")

    from opencut.core.captions import check_whisper_available, transcribe
    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper required for B-roll analysis. Install from Settings.")

    from opencut.utils.config import CaptionConfig
    transcript = transcribe(filepath, config=CaptionConfig(model="base"))

    segments = []
    if hasattr(transcript, "segments"):
        for seg in transcript.segments:
            if isinstance(seg, dict):
                segments.append(seg)
            elif hasattr(seg, "start"):
                segments.append({"start": seg.start, "end": seg.end, "text": getattr(seg, "text", "")})
    elif isinstance(transcript, dict):
        segments = transcript.get("segments", [])

    if not segments:
        return {"windows": [], "total_windows": 0, "total_broll_time": 0, "keywords_used": []}

    _update_job(job_id, progress=60, message=f"Analyzing {len(segments)} segments for B-roll opportunities...")

    # Step 2: Analyze for B-roll insertion points
    from opencut.core.broll_insert import analyze_broll_opportunities

    plan = analyze_broll_opportunities(
        segments,
        min_gap=min_gap,
        max_results=max_results,
    )

    if plan is None:
        return {"windows": [], "total_windows": 0, "total_broll_time": 0, "keywords_used": []}

    return {
            "windows": [
                {
                    "start": getattr(w, "start", 0),
                    "end": getattr(w, "end", 0),
                    "duration": getattr(w, "duration", 0),
                    "reason": getattr(w, "reason", ""),
                    "keywords": getattr(w, "keywords", []),
                    "score": getattr(w, "score", 0),
                    "context": getattr(w, "context", ""),
                }
                for w in getattr(plan, "windows", [])
            ],
            "total_windows": getattr(plan, "total_windows", 0),
            "total_broll_time": getattr(plan, "total_broll_time", 0),
            "keywords_used": getattr(plan, "keywords_used", []),
        }


# ---------------------------------------------------------------------------
# Install Endpoints for New AI Features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Install Endpoints (generated via factory)
# ---------------------------------------------------------------------------
make_install_route(video_specialty_bp, "/video/depth/install", "depth_effects",
                   ["torch", "torchvision", "transformers", "opencv-python-headless", "Pillow"],
                   doc="Install Depth Anything V2 dependencies.")

make_install_route(video_specialty_bp, "/video/emotion/install", "emotion_highlights",
                   ["deepface", "opencv-python-headless"],
                   doc="Install emotion analysis dependencies.")

make_install_route(video_specialty_bp, "/video/multimodal-diarize/install", "multimodal_diarize",
                   ["opencv-python-headless", "insightface", "onnxruntime"],
                   doc="Install multimodal diarization dependencies.")

make_install_route(video_specialty_bp, "/video/broll-generate/install", "broll_generate",
                   ["torch", "torchvision", "diffusers", "transformers", "accelerate"],
                   doc="Install AI B-roll generation dependencies.")
