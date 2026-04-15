"""
OpenCut Motion Design & Animation Routes (Category 79)

Blueprint providing endpoints for kinetic typography, data-driven animation,
shape animation, expression evaluation, and particle systems.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath, validate_output_path, validate_path

logger = logging.getLogger("opencut")

motion_design_bp = Blueprint("motion_design", __name__)


# ---------------------------------------------------------------------------
# Kinetic Typography
# ---------------------------------------------------------------------------


@motion_design_bp.route("/motion/kinetic-text", methods=["POST"])
@require_csrf
@async_job("kinetic_text", filepath_required=False)
def kinetic_text_render(job_id, filepath, data):
    """Render kinetic typography animation to video."""
    from opencut.core.kinetic_typography import render_kinetic_text

    text = data.get("text", "").strip()
    if not text:
        raise ValueError("Text is required")

    preset = data.get("preset", "bounce").strip().lower()
    mode = data.get("mode", "char").strip().lower()
    duration = safe_float(data.get("duration"), 3.0, 0.5, 30.0)
    fps = safe_int(data.get("fps"), 30, 10, 60)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    font_name = data.get("font_name")
    font_size = safe_int(data.get("font_size"), 72, 12, 500)
    color = data.get("color", "#FFFFFF")
    outline_color = data.get("outline_color")
    outline_width = safe_int(data.get("outline_width"), 0, 0, 20)
    shadow_color = data.get("shadow_color")
    bg_color = data.get("background_color", "#00000000")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    if output_dir:
        output_dir = _resolve_output_dir("", output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Rendering kinetic text ({pct}%)")

    result = render_kinetic_text(
        text=text,
        preset=preset,
        mode=mode,
        duration=duration,
        fps=fps,
        resolution=(width, height),
        font_name=font_name,
        font_size=font_size,
        color=color,
        outline_color=outline_color,
        outline_width=outline_width,
        shadow_color=shadow_color,
        background_color=bg_color,
        output_path=out_path,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@motion_design_bp.route("/motion/kinetic-text/presets", methods=["GET"])
def kinetic_text_presets():
    """List available kinetic typography animation presets."""
    try:
        from opencut.core.kinetic_typography import list_presets
        return jsonify({"presets": list_presets()})
    except Exception as e:
        return safe_error(e, "kinetic_text_presets")


@motion_design_bp.route("/motion/kinetic-text/preview", methods=["POST"])
@require_csrf
@async_job("kinetic_text_preview", filepath_required=False)
def kinetic_text_preview(job_id, filepath, data):
    """Render a single kinetic typography preview frame."""
    from opencut.core.kinetic_typography import preview_frame

    text = data.get("text", "").strip()
    if not text:
        raise ValueError("Text is required")

    preset = data.get("preset", "bounce").strip().lower()
    mode = data.get("mode", "char").strip().lower()
    time_s = safe_float(data.get("time"), 0.5, 0.0, 30.0)
    duration = safe_float(data.get("duration"), 3.0, 0.5, 30.0)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    font_size = safe_int(data.get("font_size"), 72, 12, 500)
    color = data.get("color", "#FFFFFF")

    result_path = preview_frame(
        text=text,
        preset=preset,
        mode=mode,
        time_s=time_s,
        duration=duration,
        resolution=(width, height),
        font_size=font_size,
        color=color,
    )
    return {"preview_path": result_path}


# ---------------------------------------------------------------------------
# Data Animation
# ---------------------------------------------------------------------------


@motion_design_bp.route("/motion/data-animation", methods=["POST"])
@require_csrf
@async_job("data_animation", filepath_required=False)
def data_animation_render(job_id, filepath, data):
    """Render data-driven animation to video."""
    from opencut.core.data_animation import render_data_animation

    template = data.get("template")
    if not template or not isinstance(template, dict):
        raise ValueError("Template dict is required")

    data_rows = data.get("data")
    data_content = data.get("data_content", "")
    data_file = data.get("data_file", "")
    data_format = data.get("data_format", "auto")

    if data_file:
        data_file = validate_filepath(data_file)

    duration_per_row = safe_float(data.get("duration_per_row"), 2.0, 0.5, 30.0)
    transition_duration = safe_float(data.get("transition_duration"), 0.5, 0.0, 5.0)
    fps = safe_int(data.get("fps"), 30, 10, 60)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    bg_color = data.get("bg_color", "#1A1A2E")
    font_size = safe_int(data.get("font_size"), 24, 8, 200)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    if output_dir:
        output_dir = _resolve_output_dir("", output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Rendering data animation ({pct}%)")

    result = render_data_animation(
        template=template,
        data=data_rows,
        data_content=data_content,
        data_file=data_file,
        data_format=data_format,
        duration_per_row=duration_per_row,
        transition_duration=transition_duration,
        fps=fps,
        resolution=(width, height),
        bg_color=bg_color,
        font_size=font_size,
        output_path=out_path,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@motion_design_bp.route("/motion/data-animation/validate", methods=["POST"])
@require_csrf
def data_animation_validate():
    """Validate a data animation template against data."""
    try:
        from opencut.core.data_animation import load_data, validate_template

        data = request.get_json(force=True) or {}
        template = data.get("template")
        if not template:
            return jsonify({"error": "Template is required"}), 400

        data_rows = data.get("data")
        data_content = data.get("data_content", "")
        data_format = data.get("data_format", "auto")

        if data_rows is None and data_content:
            data_rows = load_data(data_content, data_format)

        result = validate_template(template, data_rows or [])
        return jsonify(result)
    except Exception as e:
        return safe_error(e, "data_animation_validate")


# ---------------------------------------------------------------------------
# Shape Animation
# ---------------------------------------------------------------------------


@motion_design_bp.route("/motion/shape-animate", methods=["POST"])
@require_csrf
@async_job("shape_animation", filepath_required=False)
def shape_animate_render(job_id, filepath, data):
    """Render shape animation to video."""
    from opencut.core.shape_animation import render_shape_animation

    shapes = data.get("shapes", [])
    if not shapes:
        raise ValueError("At least one shape definition is required")

    animation = data.get("animation", "morph").strip().lower()
    duration = safe_float(data.get("duration"), 3.0, 0.5, 30.0)
    fps = safe_int(data.get("fps"), 30, 10, 60)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    easing = data.get("easing", "ease_out").strip()
    stroke_color = data.get("stroke_color", "#FFFFFF")
    stroke_width = safe_int(data.get("stroke_width"), 2, 0, 20)
    fill_color = data.get("fill_color")
    fill_color_end = data.get("fill_color_end")
    num_points = safe_int(data.get("num_points"), 64, 8, 512)
    bg_color = data.get("background_color", "#00000000")
    scale_start = safe_float(data.get("scale_start"), 1.0, 0.01, 10.0)
    scale_end = safe_float(data.get("scale_end"), 1.0, 0.01, 10.0)
    rotation_start = safe_float(data.get("rotation_start"), 0.0, -3600, 3600)
    rotation_end = safe_float(data.get("rotation_end"), 0.0, -3600, 3600)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    if output_dir:
        output_dir = _resolve_output_dir("", output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Rendering shape animation ({pct}%)")

    result = render_shape_animation(
        shapes=shapes,
        animation=animation,
        duration=duration,
        fps=fps,
        resolution=(width, height),
        easing=easing,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_color=fill_color,
        fill_color_end=fill_color_end,
        num_points=num_points,
        bg_color=bg_color,
        scale_start=scale_start,
        scale_end=scale_end,
        rotation_start=rotation_start,
        rotation_end=rotation_end,
        output_path=out_path,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@motion_design_bp.route("/motion/shape-animate/types", methods=["GET"])
def shape_animate_types():
    """List available shape types and animation types."""
    try:
        from opencut.core.shape_animation import (
            list_animation_types,
            list_shape_types,
        )
        return jsonify({
            "shape_types": list_shape_types(),
            "animation_types": list_animation_types(),
        })
    except Exception as e:
        return safe_error(e, "shape_animate_types")


# ---------------------------------------------------------------------------
# Expression Engine
# ---------------------------------------------------------------------------


@motion_design_bp.route("/motion/expression/evaluate", methods=["POST"])
@require_csrf
@async_job("expression_evaluate", filepath_required=False)
def expression_evaluate(job_id, filepath, data):
    """Evaluate an expression for preview."""
    from opencut.core.expression_engine import (
        ExpressionContext,
        evaluate_expression,
    )

    expression = data.get("expression", "").strip()
    if not expression:
        raise ValueError("Expression string is required")

    time_val = safe_float(data.get("time"), 0.0, 0.0, 3600.0)
    frame_val = safe_int(data.get("frame"), 0, 0, 1000000)
    fps_val = safe_float(data.get("fps"), 30.0, 1.0, 120.0)
    amplitude = safe_float(data.get("audio_amplitude"), 0.0, 0.0, 1.0)
    beat = bool(data.get("beat", False))
    custom_vars = data.get("variables", {})

    ctx = ExpressionContext(
        time=time_val,
        frame=frame_val,
        fps=fps_val,
        audio_amplitude=amplitude,
        beat=beat,
        custom_vars=custom_vars if isinstance(custom_vars, dict) else {},
    )

    result = evaluate_expression(expression, ctx)
    return {"value": result, "expression": expression}


@motion_design_bp.route("/motion/expression/timeline", methods=["POST"])
@require_csrf
@async_job("expression_timeline", filepath_required=False)
def expression_timeline(job_id, filepath, data):
    """Generate expression value timeline for visualization."""
    from opencut.core.expression_engine import evaluate_timeline

    expression = data.get("expression", "").strip()
    if not expression:
        raise ValueError("Expression string is required")

    fps = safe_float(data.get("fps"), 30.0, 1.0, 120.0)
    duration = safe_float(data.get("duration"), 1.0, 0.1, 60.0)
    seed = safe_int(data.get("seed"), 42, 0, 999999)
    custom_vars = data.get("variables", {})

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Evaluating expression ({pct}%)")

    kwargs = {}
    if isinstance(custom_vars, dict):
        kwargs = custom_vars

    result = evaluate_timeline(
        expression,
        fps=fps,
        duration=duration,
        seed=seed,
        on_progress=_progress,
        **kwargs,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Particle System
# ---------------------------------------------------------------------------


@motion_design_bp.route("/motion/particles", methods=["POST"])
@require_csrf
@async_job("particles", filepath_required=False)
def particles_render(job_id, filepath, data):
    """Render particle overlay to video."""
    from opencut.core.particle_system import render_particles

    preset = data.get("preset", "").strip() or None
    emitter_config = data.get("emitter_config")
    if not preset and not emitter_config:
        raise ValueError("Either preset or emitter_config is required")

    duration = safe_float(data.get("duration"), 5.0, 0.5, 60.0)
    fps = safe_int(data.get("fps"), 30, 10, 60)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    blend_mode = data.get("blend_mode", "alpha").strip().lower()
    if blend_mode not in ("alpha", "additive"):
        blend_mode = "alpha"
    bg_color = data.get("background_color")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    if output_dir:
        output_dir = _resolve_output_dir("", output_dir)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Rendering particles ({pct}%)")

    result = render_particles(
        preset=preset,
        emitter_config=emitter_config,
        duration=duration,
        fps=fps,
        resolution=(width, height),
        blend_mode=blend_mode,
        bg_color=bg_color,
        output_path=out_path,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result.to_dict()


@motion_design_bp.route("/motion/particles/presets", methods=["GET"])
def particles_presets():
    """List available particle presets."""
    try:
        from opencut.core.particle_system import list_presets
        return jsonify({"presets": list_presets()})
    except Exception as e:
        return safe_error(e, "particle_presets")


@motion_design_bp.route("/motion/particles/preview", methods=["POST"])
@require_csrf
@async_job("particle_preview", filepath_required=False)
def particles_preview(job_id, filepath, data):
    """Render a single particle system preview frame."""
    from opencut.core.particle_system import preview_frame

    preset = data.get("preset", "").strip() or None
    emitter_config = data.get("emitter_config")
    if not preset and not emitter_config:
        raise ValueError("Either preset or emitter_config is required")

    time_s = safe_float(data.get("time"), 1.0, 0.0, 60.0)
    fps = safe_int(data.get("fps"), 30, 10, 60)
    width = safe_int(data.get("width"), 1920, 320, 3840)
    height = safe_int(data.get("height"), 1080, 240, 2160)
    blend_mode = data.get("blend_mode", "alpha").strip().lower()
    bg_color = data.get("background_color")

    result_path = preview_frame(
        preset=preset,
        emitter_config=emitter_config,
        time_s=time_s,
        fps=fps,
        resolution=(width, height),
        blend_mode=blend_mode,
        bg_color=bg_color,
    )
    return {"preview_path": result_path}
