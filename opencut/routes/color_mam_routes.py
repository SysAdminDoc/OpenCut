"""
OpenCut Color & MAM Routes

Routes for Color Scopes (13.1), Three-Way Color Wheels (13.2),
HSL Qualifier (13.3), Power Windows (13.6), ACES Pipeline (43.1),
Proxy Generation (23.1), AI Metadata (23.2), Kinetic Typography (26.1),
Data Animation (26.2), and Shape Animation (26.3).
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
    validate_output_path,
)

logger = logging.getLogger("opencut")

color_mam_bp = Blueprint("color_mam", __name__)


# ===================================================================
# 1. Color Scopes (13.1)
# ===================================================================

@color_mam_bp.route("/video/color-scopes/waveform", methods=["POST"])
@require_csrf
def color_scope_waveform():
    """Generate a waveform monitor from a video frame (sync)."""
    try:
        from opencut.core.color_scopes import generate_waveform

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
        output_dir = data.get("output_dir", "")
        if output_dir:
            output_dir = validate_path(output_dir)
        output = data.get("output_path") or None
        if output:
            output = validate_output_path(output)

        result = generate_waveform(
            video_path=filepath,
            timestamp=timestamp,
            output_path=output,
            output_dir=output_dir,
            mode=data.get("mode", "lowpass"),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_scope_waveform")


@color_mam_bp.route("/video/color-scopes/vectorscope", methods=["POST"])
@require_csrf
def color_scope_vectorscope():
    """Generate a vectorscope from a video frame (sync)."""
    try:
        from opencut.core.color_scopes import generate_vectorscope

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)

        result = generate_vectorscope(
            video_path=filepath,
            timestamp=timestamp,
            output_path=data.get("output_path") or None,
            output_dir=data.get("output_dir", ""),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_scope_vectorscope")


@color_mam_bp.route("/video/color-scopes/rgb-parade", methods=["POST"])
@require_csrf
def color_scope_rgb_parade():
    """Generate an RGB parade from a video frame (sync)."""
    try:
        from opencut.core.color_scopes import generate_rgb_parade

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)

        result = generate_rgb_parade(
            video_path=filepath,
            timestamp=timestamp,
            output_path=data.get("output_path") or None,
            output_dir=data.get("output_dir", ""),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_scope_rgb_parade")


@color_mam_bp.route("/video/color-scopes/histogram", methods=["POST"])
@require_csrf
def color_scope_histogram():
    """Generate a histogram from a video frame (sync)."""
    try:
        from opencut.core.color_scopes import generate_histogram

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
        display_mode = data.get("display_mode", "stack")

        result = generate_histogram(
            video_path=filepath,
            timestamp=timestamp,
            output_path=data.get("output_path") or None,
            output_dir=data.get("output_dir", ""),
            display_mode=display_mode,
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_scope_histogram")


@color_mam_bp.route("/video/color-scopes/all", methods=["POST"])
@require_csrf
def color_scopes_all():
    """Generate all four scopes at once (sync)."""
    try:
        from opencut.core.color_scopes import generate_all_scopes

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)

        result = generate_all_scopes(
            video_path=filepath,
            timestamp=timestamp,
            output_dir=data.get("output_dir", ""),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_scopes_all")


# ===================================================================
# 2. Three-Way Color Wheels (13.2)
# ===================================================================

@color_mam_bp.route("/video/color-wheels/apply", methods=["POST"])
@require_csrf
@async_job("color_wheels")
def color_wheels_apply(job_id, filepath, data):
    """Apply three-way color wheel grading (async)."""
    from opencut.core.color_wheels import apply_color_wheels

    settings = data.get("settings", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_color_wheels(
        video_path=filepath,
        settings=settings,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/color-wheels/preview", methods=["POST"])
@require_csrf
def color_wheels_preview():
    """Generate a single-frame color wheel preview (sync)."""
    try:
        from opencut.core.color_wheels import preview_color_wheels

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
        settings = data.get("settings", {})

        result = preview_color_wheels(
            video_path=filepath,
            timestamp=timestamp,
            settings=settings,
            output_path=data.get("output_path") or None,
            output_dir=data.get("output_dir", ""),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "color_wheels_preview")


# ===================================================================
# 3. HSL Qualifier (13.3)
# ===================================================================

@color_mam_bp.route("/video/hsl-qualifier/qualify", methods=["POST"])
@require_csrf
@async_job("hsl_qualify")
def hsl_qualify(job_id, filepath, data):
    """Apply HSL qualification to isolate a color range (async)."""
    from opencut.core.hsl_qualifier import qualify_hsl

    hsl_range = data.get("hsl_range", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = qualify_hsl(
        video_path=filepath,
        hsl_range=hsl_range,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/hsl-qualifier/matte-preview", methods=["POST"])
@require_csrf
def hsl_matte_preview():
    """Generate a matte preview for HSL qualification (sync)."""
    try:
        from opencut.core.hsl_qualifier import preview_matte

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
        hsl_range = data.get("hsl_range", {})

        result = preview_matte(
            video_path=filepath,
            timestamp=timestamp,
            hsl_range=hsl_range,
            output_path=data.get("output_path") or None,
            output_dir=data.get("output_dir", ""),
        )
        return jsonify(result.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "hsl_matte_preview")


@color_mam_bp.route("/video/hsl-qualifier/secondary", methods=["POST"])
@require_csrf
@async_job("hsl_secondary")
def hsl_secondary(job_id, filepath, data):
    """Apply secondary color correction via HSL qualification (async)."""
    from opencut.core.hsl_qualifier import apply_secondary_correction

    qualification = data.get("qualification", data.get("hsl_range", {}))
    correction = data.get("correction", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_secondary_correction(
        video_path=filepath,
        qualification=qualification,
        correction=correction,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


# ===================================================================
# 4. Power Windows (13.6)
# ===================================================================

@color_mam_bp.route("/video/power-windows/create", methods=["POST"])
@require_csrf
def power_window_create():
    """Create a power window definition (sync, no processing)."""
    try:
        from opencut.core.power_windows import create_power_window

        data = request.get_json(force=True) or {}
        shape = data.get("shape", "circle")
        position = data.get("position", (0.5, 0.5))
        if isinstance(position, list):
            position = tuple(position)

        window = create_power_window(
            shape=shape,
            position=position,
            feather=safe_float(data.get("feather", 0.05), 0.05),
            width=safe_float(data.get("width", 0.3), 0.3),
            height=safe_float(data.get("height", 0.3), 0.3),
            rotation=safe_float(data.get("rotation", 0), 0),
            invert=data.get("invert", False),
        )
        return jsonify(window.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "power_window_create")


@color_mam_bp.route("/video/power-windows/track", methods=["POST"])
@require_csrf
@async_job("power_window_track")
def power_window_track(job_id, filepath, data):
    """Track a power window across video frames (async)."""
    from opencut.core.power_windows import track_window

    window = data.get("window", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = track_window(
        video_path=filepath,
        window=window,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/power-windows/apply", methods=["POST"])
@require_csrf
@async_job("power_window_apply")
def power_window_apply(job_id, filepath, data):
    """Apply windowed correction to a video (async)."""
    from opencut.core.power_windows import apply_windowed_correction

    window = data.get("window", {})
    correction = data.get("correction", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_windowed_correction(
        video_path=filepath,
        window_data=window,
        correction=correction,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


# ===================================================================
# 5. ACES Color Pipeline (43.1)
# ===================================================================

@color_mam_bp.route("/video/aces/detect-idt", methods=["POST"])
@require_csrf
def aces_detect_idt():
    """Auto-detect the camera IDT for a video (sync)."""
    try:
        from opencut.core.aces_pipeline import detect_camera_idt

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        idt = detect_camera_idt(filepath)
        return jsonify({"idt": idt})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "aces_detect_idt")


@color_mam_bp.route("/video/aces/apply", methods=["POST"])
@require_csrf
@async_job("aces_pipeline")
def aces_apply(job_id, filepath, data):
    """Apply full ACES color pipeline (async)."""
    from opencut.core.aces_pipeline import apply_aces_pipeline

    idt = data.get("idt", "srgb")
    odt = data.get("odt", "rec709")
    config = data.get("config", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_aces_pipeline(
        video_path=filepath,
        idt=idt,
        odt=odt,
        config=config if config else None,
        output_path=output,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/aces/idts", methods=["GET"])
def aces_list_idts():
    """List available Input Device Transforms."""
    from opencut.core.aces_pipeline import list_available_idts
    return jsonify({"idts": list_available_idts()})


@color_mam_bp.route("/video/aces/odts", methods=["GET"])
def aces_list_odts():
    """List available Output Device Transforms."""
    from opencut.core.aces_pipeline import list_available_odts
    return jsonify({"odts": list_available_odts()})


# ===================================================================
# 6. Proxy Generation (23.1)
# ===================================================================

@color_mam_bp.route("/video/proxy/generate", methods=["POST"])
@require_csrf
@async_job("proxy_generate")
def proxy_generate(job_id, filepath, data):
    """Generate a proxy for a single video (async)."""
    from opencut.core.proxy_gen import generate_proxy

    config = data.get("config", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path") or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_proxy(
        video_path=filepath,
        output_path=output,
        output_dir=output_dir,
        config=config if config else None,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/proxy/batch", methods=["POST"])
@require_csrf
@async_job("proxy_batch", filepath_required=False)
def proxy_batch(job_id, filepath, data):
    """Batch-generate proxies for multiple videos (async)."""
    from opencut.core.proxy_gen import batch_generate_proxies

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("No file paths provided")
    file_paths = [validate_filepath(fp) for fp in file_paths]

    config = data.get("config", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_generate_proxies(
        file_paths=file_paths,
        output_dir=output_dir,
        config=config if config else None,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/proxy/relink", methods=["POST"])
@require_csrf
def proxy_relink():
    """Relink a proxy to its original file (sync)."""
    try:
        from opencut.core.proxy_gen import relink_proxy_to_original

        data = request.get_json(force=True) or {}
        proxy_path = data.get("proxy_path", "").strip()
        if not proxy_path:
            return jsonify({"error": "No proxy path provided"}), 400
        proxy_path = validate_filepath(proxy_path)

        original = relink_proxy_to_original(
            proxy_path=proxy_path,
            proxy_dir=data.get("proxy_dir", ""),
        )
        return jsonify({"original_path": original})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "proxy_relink")


# ===================================================================
# 7. AI Metadata Enrichment (23.2)
# ===================================================================

@color_mam_bp.route("/video/ai-metadata/enrich", methods=["POST"])
@require_csrf
@async_job("ai_metadata")
def ai_metadata_enrich(job_id, filepath, data):
    """Enrich a video with AI-detected metadata (async)."""
    from opencut.core.ai_metadata import enrich_metadata

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = enrich_metadata(
        video_path=filepath,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/ai-metadata/batch", methods=["POST"])
@require_csrf
@async_job("ai_metadata_batch", filepath_required=False)
def ai_metadata_batch(job_id, filepath, data):
    """Batch-enrich metadata for multiple videos (async)."""
    from opencut.core.ai_metadata import batch_enrich

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("No file paths provided")
    file_paths = [validate_filepath(fp) for fp in file_paths]

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_enrich(
        file_paths=file_paths,
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/ai-metadata/detect-objects", methods=["POST"])
@require_csrf
def ai_metadata_detect_objects():
    """Detect objects in a single frame image (sync)."""
    try:
        from opencut.core.ai_metadata import detect_objects

        data = request.get_json(force=True) or {}
        frame = data.get("frame", data.get("filepath", "")).strip()
        if not frame:
            return jsonify({"error": "No frame path provided"}), 400
        frame = validate_filepath(frame)

        objects = detect_objects(frame)
        return jsonify({"objects": objects})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "ai_metadata_detect_objects")


@color_mam_bp.route("/video/ai-metadata/classify-scene", methods=["POST"])
@require_csrf
def ai_metadata_classify_scene():
    """Classify the scene type of a frame (sync)."""
    try:
        from opencut.core.ai_metadata import classify_scene

        data = request.get_json(force=True) or {}
        frame = data.get("frame", data.get("filepath", "")).strip()
        if not frame:
            return jsonify({"error": "No frame path provided"}), 400
        frame = validate_filepath(frame)

        result = classify_scene(frame)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "ai_metadata_classify_scene")


# ===================================================================
# 8. Kinetic Typography (26.1)
# ===================================================================

@color_mam_bp.route("/video/kinetic-text/animate", methods=["POST"])
@require_csrf
@async_job("kinetic_text", filepath_required=False)
def kinetic_text_animate(job_id, filepath, data):
    """Render a kinetic typography animation (async)."""
    from opencut.core.kinetic_type import animate_text

    text = data.get("text", "")
    if not text:
        raise ValueError("No text provided")

    preset = data.get("preset", data.get("animation_preset", "fade_in"))
    duration = safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60)
    width = safe_int(data.get("width", 1920), 1920, min_val=100, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=100, max_val=4320)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = animate_text(
        text=text,
        animation_preset=preset,
        duration=duration,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=width,
        height=height,
        font_size=safe_int(data.get("font_size", 64), 64, min_val=8, max_val=500),
        font_color=data.get("font_color", "white"),
        background_color=data.get("background_color", "black"),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/kinetic-text/presets", methods=["GET"])
def kinetic_text_presets():
    """List available kinetic text animation presets."""
    from opencut.core.kinetic_type import list_animation_presets
    return jsonify({"presets": list_animation_presets()})


@color_mam_bp.route("/video/kinetic-text/custom", methods=["POST"])
@require_csrf
@async_job("kinetic_text_custom", filepath_required=False)
def kinetic_text_custom(job_id, filepath, data):
    """Render a custom keyframe-based text animation (async)."""
    from opencut.core.kinetic_type import create_custom_animation

    text = data.get("text", "")
    if not text:
        raise ValueError("No text provided")
    keyframes = data.get("keyframes", [])
    easing = data.get("easing", "ease_out")
    duration = safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_custom_animation(
        keyframes=keyframes,
        easing=easing,
        text=text,
        duration=duration,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=safe_int(data.get("width", 1920), 1920),
        height=safe_int(data.get("height", 1080), 1080),
        font_size=safe_int(data.get("font_size", 64), 64),
        font_color=data.get("font_color", "white"),
        background_color=data.get("background_color", "black"),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/kinetic-text/render", methods=["POST"])
@require_csrf
@async_job("kinetic_text_render", filepath_required=False)
def kinetic_text_render(job_id, filepath, data):
    """Render kinetic text from full animation data spec (async)."""
    from opencut.core.kinetic_type import render_kinetic_text

    animation_data = data.get("animation_data", data)
    resolution = data.get("resolution", [1920, 1080])
    if isinstance(resolution, list):
        resolution = tuple(resolution)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_kinetic_text(
        animation_data=animation_data,
        resolution=resolution,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        on_progress=_progress,
    )
    return result.to_dict()


# ===================================================================
# 9. Data-Driven Animation (26.2)
# ===================================================================

@color_mam_bp.route("/video/data-animation/create", methods=["POST"])
@require_csrf
@async_job("data_animation", filepath_required=False)
def data_animation_create(job_id, filepath, data):
    """Create a data-driven animation from a template and data source (async)."""
    from opencut.core.data_animation import create_data_animation

    template = data.get("template", {})
    data_source = data.get("data_source", data.get("data", []))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_data_animation(
        template=template,
        data_source=data_source,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/data-animation/bar-chart", methods=["POST"])
@require_csrf
@async_job("data_bar_chart", filepath_required=False)
def data_animation_bar_chart(job_id, filepath, data):
    """Render an animated bar chart (async)."""
    from opencut.core.data_animation import render_bar_chart

    chart_data = data.get("data", [])
    config = data.get("config", {})

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_bar_chart(
        data=chart_data,
        config=config,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/data-animation/counter", methods=["POST"])
@require_csrf
@async_job("data_counter", filepath_required=False)
def data_animation_counter(job_id, filepath, data):
    """Render an animated number counter (async)."""
    from opencut.core.data_animation import render_counter

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_counter(
        start=safe_float(data.get("start", 0), 0),
        end=safe_float(data.get("end", 100), 100),
        duration=safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60),
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=safe_int(data.get("width", 1920), 1920),
        height=safe_int(data.get("height", 1080), 1080),
        font_size=safe_int(data.get("font_size", 96), 96),
        font_color=data.get("font_color", "white"),
        background_color=data.get("background_color", "black"),
        title=data.get("title", ""),
        prefix=data.get("prefix", ""),
        suffix=data.get("suffix", ""),
        on_progress=_progress,
    )
    return result.to_dict()


# ===================================================================
# 10. Shape Layer Animation (26.3)
# ===================================================================

@color_mam_bp.route("/video/shape-animation/morph", methods=["POST"])
@require_csrf
@async_job("shape_morph", filepath_required=False)
def shape_animation_morph(job_id, filepath, data):
    """Render a shape morph animation (async)."""
    from opencut.core.shape_animation import animate_shape_morph

    shape_a = data.get("shape_a", {})
    shape_b = data.get("shape_b", {})
    duration = safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = animate_shape_morph(
        shape_a=shape_a if shape_a else None,
        shape_b=shape_b if shape_b else None,
        duration=duration,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=safe_int(data.get("width", 1920), 1920),
        height=safe_int(data.get("height", 1080), 1080),
        background_color=data.get("background_color", "black"),
        easing=data.get("easing", "ease_in_out"),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/shape-animation/stroke-draw", methods=["POST"])
@require_csrf
@async_job("shape_stroke_draw")
def shape_animation_stroke_draw(job_id, filepath, data):
    """Render a stroke drawing animation on an image/SVG (async)."""
    from opencut.core.shape_animation import animate_stroke_draw

    duration = safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = animate_stroke_draw(
        svg_path=filepath,
        duration=duration,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=safe_int(data.get("width", 1920), 1920),
        height=safe_int(data.get("height", 1080), 1080),
        background_color=data.get("background_color", "black"),
        stroke_color=data.get("stroke_color", "white"),
        stroke_width=safe_int(data.get("stroke_width", 3), 3),
        on_progress=_progress,
    )
    return result.to_dict()


@color_mam_bp.route("/video/shape-animation/fill-transition", methods=["POST"])
@require_csrf
@async_job("shape_fill_transition", filepath_required=False)
def shape_animation_fill_transition(job_id, filepath, data):
    """Render a shape fill color transition (async)."""
    from opencut.core.shape_animation import animate_fill_transition

    shape = data.get("shape", {})
    color_a = data.get("color_a", "0x4285F4")
    color_b = data.get("color_b", "0xEA4335")
    duration = safe_float(data.get("duration", 3), 3, min_val=0.1, max_val=60)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = animate_fill_transition(
        shape=shape if shape else None,
        color_a=color_a,
        color_b=color_b,
        duration=duration,
        output_path=data.get("output_path") or None,
        output_dir=data.get("output_dir", ""),
        width=safe_int(data.get("width", 1920), 1920),
        height=safe_int(data.get("height", 1080), 1080),
        background_color=data.get("background_color", "black"),
        on_progress=_progress,
    )
    return result.to_dict()
