"""
OpenCut Content Generation Routes

Endpoints for AI-powered content generation:
- Voice avatar generation (audio + face -> talking head video)
- Thumbnail CTR prediction and comparison
- AI B-roll generation from text prompts
- Chapter artwork / title card generation
- Animated intro generation from brand kit
"""

import logging

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath, validate_path

logger = logging.getLogger("opencut")

content_gen_bp = Blueprint("content_gen", __name__)


# ===========================================================================
# Voice Avatar Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /ai/voice-avatar — generate talking avatar (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/ai/voice-avatar", methods=["POST"])
@require_csrf
@async_job("voice_avatar", filepath_param="audio_path")
def voice_avatar_generate(job_id, filepath, data):
    """Generate a talking avatar video from audio and face image."""
    from opencut.core.voice_avatar import AvatarConfig, generate_avatar

    face_image = data.get("face_image", "").strip()
    if not face_image:
        raise ValueError("No face_image path provided")
    face_image = validate_filepath(face_image)

    style = data.get("style", "cartoon").strip()
    width = safe_int(data.get("width", 720), 720, min_val=64, max_val=4096)
    height = safe_int(data.get("height", 720), 720, min_val=64, max_val=4096)
    fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=120)
    bg_mode = data.get("background_mode", "solid").strip()
    bg_image = data.get("background_image", "").strip()
    bg_blur = safe_int(data.get("background_blur_radius", 25), 25, min_val=0, max_val=100)
    mouth_threshold = safe_float(data.get("mouth_open_threshold", 0.02), 0.02,
                                  min_val=0.0, max_val=1.0)
    mouth_scale = safe_float(data.get("mouth_amplitude_scale", 1.5), 1.5,
                              min_val=0.1, max_val=5.0)
    face_scale = safe_float(data.get("face_scale", 0.6), 0.6,
                             min_val=0.1, max_val=1.0)
    max_duration = safe_float(data.get("max_duration", 0.0), 0.0,
                               min_val=0.0, max_val=600.0)

    bg_color = tuple(data.get("background_color", [18, 18, 24])[:3])
    face_pos_raw = data.get("face_position", [0.5, 0.45])
    face_pos = (
        safe_float(face_pos_raw[0] if isinstance(face_pos_raw, list) and len(face_pos_raw) > 0 else 0.5, 0.5,
                   min_val=0.0, max_val=1.0),
        safe_float(face_pos_raw[1] if isinstance(face_pos_raw, list) and len(face_pos_raw) > 1 else 0.45, 0.45,
                   min_val=0.0, max_val=1.0),
    )

    if bg_image:
        bg_image = validate_filepath(bg_image)

    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    config = AvatarConfig(
        style=style,
        width=width,
        height=height,
        fps=fps,
        background_mode=bg_mode,
        background_color=bg_color,
        background_image=bg_image,
        background_blur_radius=bg_blur,
        mouth_open_threshold=mouth_threshold,
        mouth_amplitude_scale=mouth_scale,
        face_scale=face_scale,
        face_position=face_pos,
        max_duration=max_duration,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_avatar(
        audio_path=filepath,
        face_image=face_image,
        config=config,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /ai/voice-avatar/styles — list available avatar styles (sync)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/ai/voice-avatar/styles", methods=["GET"])
def voice_avatar_styles():
    """Return list of available avatar styles."""
    from opencut.core.voice_avatar import AVATAR_STYLES
    return jsonify({"styles": AVATAR_STYLES})


# ===========================================================================
# CTR Prediction Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /content/predict-ctr — predict thumbnail CTR (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/content/predict-ctr", methods=["POST"])
@require_csrf
@async_job("predict_ctr", filepath_param="image_path")
def predict_ctr_route(job_id, filepath, data):
    """Predict thumbnail click-through rate."""
    from opencut.core.ctr_predict import predict_ctr

    platform = data.get("platform", "youtube").strip()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = predict_ctr(
        image_path=filepath,
        platform=platform,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /content/compare-thumbnails — compare multiple thumbnails (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/content/compare-thumbnails", methods=["POST"])
@require_csrf
@async_job("compare_thumbnails")
def compare_thumbnails_route(job_id, filepath, data):
    """Compare multiple thumbnails for CTR ranking."""
    from opencut.core.ctr_predict import compare_thumbnails

    image_paths = data.get("image_paths", [])
    if not image_paths or not isinstance(image_paths, list):
        raise ValueError("image_paths must be a non-empty list")

    validated_paths = []
    for p in image_paths:
        if isinstance(p, str) and p.strip():
            validated_paths.append(validate_filepath(p.strip()))

    if len(validated_paths) < 2:
        raise ValueError("Need at least 2 valid image paths to compare")

    platform = data.get("platform", "youtube").strip()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = compare_thumbnails(
        image_paths=validated_paths,
        platform=platform,
        on_progress=_on_progress,
    )

    return result


# ===========================================================================
# B-Roll AI Generation Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /ai/generate-broll — generate single B-roll clip (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/ai/generate-broll", methods=["POST"])
@require_csrf
@async_job("generate_broll")
def generate_broll_route(job_id, filepath, data):
    """Generate a B-roll video clip from a text prompt."""
    from opencut.core.broll_ai_gen import BRollGenConfig, generate_broll

    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("No prompt provided")

    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=2.0, max_val=10.0)
    backend = data.get("backend", "image_kenburns").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=64, max_val=4096)
    height = safe_int(data.get("height", 1080), 1080, min_val=64, max_val=4096)
    fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=120)
    kb_preset = data.get("ken_burns_preset", "zoom_in").strip()
    style_suffix = data.get("style_prompt_suffix", "").strip()
    negative_prompt = data.get("negative_prompt",
                                "blurry, low quality, distorted, watermark, text").strip()
    seed = safe_int(data.get("seed", -1), -1)
    guidance = safe_float(data.get("guidance_scale", 7.5), 7.5, min_val=1.0, max_val=30.0)
    source_video = data.get("source_video_path", "").strip()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if source_video:
        source_video = validate_filepath(source_video)

    config = BRollGenConfig(
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        ken_burns_preset=kb_preset,
        style_prompt_suffix=style_suffix,
        negative_prompt=negative_prompt,
        seed=seed,
        guidance_scale=guidance,
        match_source=bool(source_video),
        source_video_path=source_video,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_broll(
        prompt=prompt,
        duration=duration,
        config=config,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /ai/generate-broll/batch — batch generate B-roll clips (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/ai/generate-broll/batch", methods=["POST"])
@require_csrf
@async_job("batch_generate_broll")
def batch_generate_broll_route(job_id, filepath, data):
    """Batch generate B-roll clips from multiple prompts."""
    from opencut.core.broll_ai_gen import BRollGenConfig, batch_generate_broll

    prompts = data.get("prompts", [])
    if not prompts or not isinstance(prompts, list):
        raise ValueError("prompts must be a non-empty list of strings")

    validated_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if not validated_prompts:
        raise ValueError("No valid prompts provided")

    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=2.0, max_val=10.0)
    backend = data.get("backend", "image_kenburns").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=64, max_val=4096)
    height = safe_int(data.get("height", 1080), 1080, min_val=64, max_val=4096)
    fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=120)
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    config = BRollGenConfig(
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_generate_broll(
        prompts=validated_prompts,
        duration=duration,
        config=config,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ===========================================================================
# Chapter Art Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /content/chapter-art — generate chapter artwork (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/content/chapter-art", methods=["POST"])
@require_csrf
@async_job("chapter_art")
def chapter_art_route(job_id, filepath, data):
    """Generate chapter artwork cards for a video."""
    from opencut.core.auto_chapter_art import ChapterArtConfig, generate_chapter_art

    chapters = data.get("chapters", [])
    if not chapters or not isinstance(chapters, list):
        raise ValueError("chapters must be a non-empty list")

    style = data.get("style", "minimal").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=64, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=64, max_val=4320)
    card_duration = safe_float(data.get("card_duration", 3.0), 3.0,
                                min_val=0.5, max_val=30.0)
    export_images = data.get("export_images", True)
    export_video = data.get("export_video", False)
    title_prefix = data.get("title_prefix", "Chapter").strip()
    auto_title = data.get("auto_title_from_transcript", True)
    image_format = data.get("image_format", "png").strip()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    brand_kit_data = data.get("brand_kit")
    brand_kit = None
    if brand_kit_data and isinstance(brand_kit_data, dict):
        from opencut.core.auto_chapter_art import BrandKit
        logo_path = brand_kit_data.get("logo_path", "")
        if logo_path:
            logo_path = validate_filepath(logo_path)
        brand_kit = BrandKit(
            font_path=brand_kit_data.get("font_path", ""),
            font_name=brand_kit_data.get("font_name", "Arial"),
            primary_color=tuple(brand_kit_data.get("primary_color", [80, 140, 255])[:3]),
            secondary_color=tuple(brand_kit_data.get("secondary_color", [255, 200, 80])[:3]),
            text_color=tuple(brand_kit_data.get("text_color", [255, 255, 255])[:3]),
            logo_path=logo_path,
            logo_position=brand_kit_data.get("logo_position", "top_right"),
            logo_scale=safe_float(brand_kit_data.get("logo_scale", 0.08), 0.08,
                                   min_val=0.01, max_val=0.3),
        )

    config = ChapterArtConfig(
        style=style,
        width=width,
        height=height,
        card_duration=card_duration,
        export_images=export_images,
        export_video=export_video,
        brand_kit=brand_kit,
        title_prefix=title_prefix,
        auto_title_from_transcript=auto_title,
        image_format=image_format,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_chapter_art(
        video_path=filepath,
        chapters=chapters,
        config=config,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /content/chapter-art/styles — list card styles (sync)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/content/chapter-art/styles", methods=["GET"])
def chapter_art_styles():
    """Return available chapter card styles."""
    from opencut.core.auto_chapter_art import list_card_styles
    return jsonify({"styles": list_card_styles()})


# ===========================================================================
# Intro Generation Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /video/generate-intro — generate animated intro (async)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/video/generate-intro", methods=["POST"])
@require_csrf
@async_job("generate_intro")
def generate_intro_route(job_id, filepath, data):
    """Generate an animated video intro from a brand kit."""
    from opencut.core.ai_intro_gen import IntroConfig, generate_intro

    brand_kit = data.get("brand_kit", {})
    if not isinstance(brand_kit, dict):
        raise ValueError("brand_kit must be a dict")

    if not brand_kit.get("name") and not brand_kit.get("logo_path"):
        raise ValueError("brand_kit must have at least a name or logo_path")

    if brand_kit.get("logo_path"):
        brand_kit["logo_path"] = validate_filepath(brand_kit["logo_path"])

    style = data.get("style", "logo_reveal").strip()
    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=2.0, max_val=10.0)
    width = safe_int(data.get("width", 1920), 1920, min_val=64, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=64, max_val=4320)
    fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=120)
    bg_mode = data.get("background_mode", "solid").strip()
    bg_image = data.get("background_image", "").strip()
    music_path = data.get("music_path", "").strip()
    music_volume = safe_float(data.get("music_volume", 0.8), 0.8,
                               min_val=0.0, max_val=1.0)
    prepend_to = data.get("prepend_to", "").strip()
    glow = safe_float(data.get("glow_intensity", 1.0), 1.0, min_val=0.0, max_val=3.0)
    anim_speed = safe_float(data.get("animation_speed", 1.0), 1.0,
                             min_val=0.3, max_val=3.0)
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if bg_image:
        bg_image = validate_filepath(bg_image)
    if music_path:
        music_path = validate_filepath(music_path)
    if prepend_to:
        prepend_to = validate_filepath(prepend_to)

    config = IntroConfig(
        style=style,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        background_mode=bg_mode,
        background_image=bg_image,
        music_path=music_path,
        music_volume=music_volume,
        prepend_to=prepend_to,
        glow_intensity=glow,
        animation_speed=anim_speed,
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_intro(
        brand_kit=brand_kit,
        style=style,
        duration=duration,
        config=config,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /video/intro-styles — list intro styles (sync)
# ---------------------------------------------------------------------------
@content_gen_bp.route("/video/intro-styles", methods=["GET"])
def intro_styles():
    """Return available intro animation styles."""
    from opencut.core.ai_intro_gen import list_intro_styles
    return jsonify({"styles": list_intro_styles()})
