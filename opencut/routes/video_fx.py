"""
OpenCut Video FX Routes

Effects, speed, LUT, transitions, particles, compositing, color grading.
"""

import logging
import re

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import (
    _resolve_output_dir,
)
from opencut.jobs import (
    MAX_BATCH_FILES,
    _update_job,
    async_job,
)
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

video_fx_bp = Blueprint("video_fx", __name__)


# ---------------------------------------------------------------------------
# Video Effects (FFmpeg-based, always available)
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/fx/list", methods=["GET"])
def video_fx_list():
    """Return available FFmpeg-based video effects."""
    try:
        from opencut.core.video_fx import get_available_video_effects
        return jsonify({"effects": get_available_video_effects()})
    except Exception as e:
        return safe_error(e, "video_fx_list")


@video_fx_bp.route("/video/fx/apply", methods=["POST"])
@require_csrf
@async_job("fx")
def video_fx_apply(job_id, filepath, data):
    """Apply a video effect."""
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "").strip()
    params = data.get("params", {})
    if not effect:
        raise ValueError("No effect specified")

    from opencut.core import video_fx

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)

    if effect == "stabilize":
        _fx_crop = params.get("crop", "keep")
        if _fx_crop not in ("keep", "black"):
            _fx_crop = "keep"
        out = video_fx.stabilize_video(
            filepath, output_dir=effective_dir,
            smoothing=safe_int(params.get("smoothing", 10), 10, min_val=1, max_val=100),
            crop=_fx_crop,
            zoom=safe_int(params.get("zoom", 0), 0, min_val=0, max_val=100),
            on_progress=_on_progress,
        )
    elif effect == "chromakey":
        import re as _re_fx
        _fx_color = params.get("color", "0x00FF00")
        if not _re_fx.fullmatch(r"0x[0-9A-Fa-f]{6}", _fx_color):
            _fx_color = "0x00FF00"
        _fx_bg = params.get("background", "")
        if _fx_bg:
            _fx_bg = validate_filepath(_fx_bg)
        out = video_fx.chromakey(
            filepath, output_dir=effective_dir,
            color=_fx_color,
            similarity=safe_float(params.get("similarity", 0.3), 0.3, min_val=0.0, max_val=1.0),
            blend=safe_float(params.get("blend", 0.1), 0.1, min_val=0.0, max_val=1.0),
            background=_fx_bg,
            on_progress=_on_progress,
        )
    elif effect == "lut":
        lut_path = params.get("lut_path", "")
        if not lut_path:
            raise ValueError("No LUT file path provided")
        lut_path = validate_filepath(lut_path)
        out = video_fx.apply_lut(
            filepath, lut_path, output_dir=effective_dir,
            intensity=safe_float(params.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0),
            on_progress=_on_progress,
        )
    elif effect == "vignette":
        out = video_fx.apply_vignette(
            filepath, output_dir=effective_dir,
            intensity=safe_float(params.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=_on_progress,
        )
    elif effect == "film_grain":
        out = video_fx.apply_film_grain(
            filepath, output_dir=effective_dir,
            intensity=safe_float(params.get("intensity", 0.5), 0.5, min_val=0.0, max_val=2.0),
            on_progress=_on_progress,
        )
    elif effect == "letterbox":
        _fx_aspect = params.get("aspect", "2.39:1")
        if _fx_aspect not in ("2.39:1", "2.35:1", "1.85:1", "16:9", "4:3", "1:1", "21:9"):
            _fx_aspect = "2.39:1"
        out = video_fx.apply_letterbox(
            filepath, output_dir=effective_dir,
            aspect=_fx_aspect,
            on_progress=_on_progress,
        )
    elif effect == "color_match":
        ref_path = params.get("reference_path", "")
        if not ref_path:
            raise ValueError("No reference file path provided")
        ref_path = validate_filepath(ref_path)
        out = video_fx.color_match(
            filepath, output_dir=effective_dir,
            reference_path=ref_path,
            on_progress=_on_progress,
        )
    else:
        raise ValueError(f"Unknown video effect: {effect}")

    return {"output_path": out, "effect": effect}


# ---------------------------------------------------------------------------
# Speed Ramp
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/speed/presets", methods=["GET"])
def speed_ramp_presets():
    """Return available speed ramp presets."""
    try:
        from opencut.core.speed_ramp import EASING_FUNCTIONS, get_speed_ramp_presets
        return jsonify({
            "presets": get_speed_ramp_presets(),
            "easings": list(EASING_FUNCTIONS.keys()),
        })
    except Exception as e:
        return safe_error(e, "speed_ramp_presets")


@video_fx_bp.route("/video/speed/change", methods=["POST"])
@require_csrf
@async_job("speed_change")
def speed_change_route(job_id, filepath, data):
    """Apply constant speed change."""
    speed = safe_float(data.get("speed", 2.0), 2.0, min_val=0.1, max_val=8.0)
    output_dir = data.get("output_dir", "")

    from opencut.core.speed_ramp import change_speed

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = change_speed(
        filepath, speed=speed, output_dir=effective_dir,
        maintain_pitch=data.get("maintain_pitch", False),
        on_progress=_on_progress,
    )
    return {"output_path": out, "speed": speed}


@video_fx_bp.route("/video/speed/reverse", methods=["POST"])
@require_csrf
@async_job("speed_reverse")
def speed_reverse_route(job_id, filepath, data):
    """Reverse video playback."""
    output_dir = data.get("output_dir", "")

    from opencut.core.speed_ramp import reverse_video

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = reverse_video(
        filepath, output_dir=effective_dir,
        reverse_audio=data.get("reverse_audio", True),
        on_progress=_on_progress,
    )
    return {"output_path": out}


@video_fx_bp.route("/video/speed/ramp", methods=["POST"])
@require_csrf
@async_job("speed_ramp")
def speed_ramp_route(job_id, filepath, data):
    """Apply keyframe-based speed ramp or preset."""
    preset = data.get("preset", "")
    keyframes = data.get("keyframes", [])
    output_dir = data.get("output_dir", "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)

    if preset:
        from opencut.core.speed_ramp import apply_speed_ramp_preset
        out = apply_speed_ramp_preset(
            filepath, preset, output_dir=effective_dir,
            on_progress=_on_progress,
        )
    else:
        from opencut.core.speed_ramp import speed_ramp
        _VALID_EASING = {"linear", "ease_in", "ease_out", "ease_in_out", "exponential"}
        easing_val = data.get("easing", "ease_in_out")
        if easing_val not in _VALID_EASING:
            easing_val = "ease_in_out"
        out = speed_ramp(
            filepath, keyframes, output_dir=effective_dir,
            easing=easing_val,
            on_progress=_on_progress,
        )

    return {"output_path": out}


# ---------------------------------------------------------------------------
# LUT Library
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/lut/list", methods=["GET"])
def lut_list():
    """Return available LUTs."""
    try:
        from opencut.core.lut_library import get_lut_list
        return jsonify({"luts": get_lut_list()})
    except Exception as e:
        return safe_error(e, "lut_list")


@video_fx_bp.route("/video/lut/apply", methods=["POST"])
@require_csrf
@async_job("lut_apply")
def lut_apply(job_id, filepath, data):
    """Apply a color LUT to video."""
    lut_name = data.get("lut", "teal_orange")
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0)
    output_dir = data.get("output_dir", "")

    from opencut.core.lut_library import apply_lut

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = apply_lut(
        filepath, lut_name, intensity=intensity,
        output_dir=effective_dir,
        on_progress=_on_progress,
    )
    return {"output_path": out, "lut": lut_name}


@video_fx_bp.route("/video/lut/generate-all", methods=["POST"])
@require_csrf
@async_job("lut_generate_all", filepath_required=False)
def lut_generate_all(job_id, filepath, data):
    """Pre-generate all built-in LUT .cube files."""
    from opencut.core.lut_library import generate_all_luts

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    count = generate_all_luts(on_progress=_on_progress)
    return {"count": count}


# ---------------------------------------------------------------------------
# Chromakey & Compositing
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/chromakey", methods=["POST"])
@require_csrf
@async_job("chromakey")
def chromakey_route(job_id, filepath, data):
    """Chromakey (green/blue screen) removal + compositing."""
    fg = data.get("filepath", "").strip()
    bg = data.get("background", "").strip()
    output_dir = data.get("output_dir", "")
    if not fg:
        raise ValueError("Foreground file not found")
    if not bg:
        raise ValueError("Background file not found")
    try:
        fg = validate_filepath(fg)
    except ValueError as e:
        raise ValueError(f"Foreground: {e}")
    try:
        bg = validate_filepath(bg)
    except ValueError as e:
        raise ValueError(f"Background: {e}")

    from opencut.core.chromakey import chromakey_video

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(fg, output_dir)
    _valid_chroma = {"green", "blue", "red", "custom"}
    _chroma_color = data.get("color", "green")
    if _chroma_color not in _valid_chroma:
        _chroma_color = "green"
    out = chromakey_video(fg, bg, output_dir=d, color=_chroma_color,
                          tolerance=safe_float(data.get("tolerance", 0.5), 0.5, min_val=0.0, max_val=1.0),
                          spill_suppress=safe_float(data.get("spill_suppress", 0.5), 0.5, min_val=0.0, max_val=1.0),
                          edge_blur=safe_int(data.get("edge_blur", 3), 3, min_val=0, max_val=20), on_progress=_p)
    return {"output_path": out}


@video_fx_bp.route("/video/pip", methods=["POST"])
@require_csrf
@async_job("pip")
def pip_route(job_id, filepath, data):
    """Picture-in-picture overlay."""
    main = data.get("filepath", "").strip()
    pip = data.get("pip_path", "").strip()
    if not main:
        raise ValueError("Main file not found")
    if not pip:
        raise ValueError("PiP file not found")
    try:
        main = validate_filepath(main)
    except ValueError as e:
        raise ValueError(f"Main file: {e}")
    try:
        pip = validate_filepath(pip)
    except ValueError as e:
        raise ValueError(f"PiP file: {e}")

    from opencut.core.chromakey import picture_in_picture

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(main, data.get("output_dir", ""))
    out = picture_in_picture(main, pip, output_dir=d,
                              position=data.get("position", "bottom_right"),
                              scale=safe_float(data.get("scale", 0.25), 0.25, min_val=0.05, max_val=1.0), on_progress=_p)
    return {"output_path": out}


@video_fx_bp.route("/video/blend", methods=["POST"])
@require_csrf
@async_job("blend")
def blend_route(job_id, filepath, data):
    """Blend two videos with a blend mode."""
    base = data.get("filepath", "").strip()
    overlay = data.get("overlay_path", "").strip()
    if not base:
        raise ValueError("Base file not found")
    if not overlay:
        raise ValueError("Overlay not found")
    try:
        base = validate_filepath(base)
    except ValueError as e:
        raise ValueError(f"Base file: {e}")
    try:
        overlay = validate_filepath(overlay)
    except ValueError as e:
        raise ValueError(f"Overlay: {e}")

    from opencut.core.chromakey import blend_videos

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(base, data.get("output_dir", ""))
    _blend_mode = data.get("mode", "overlay")
    if _blend_mode not in ("normal", "multiply", "screen", "overlay", "darken", "lighten",
                           "softlight", "hardlight", "difference", "exclusion", "addition",
                           "dodge", "burn", "average"):
        _blend_mode = "overlay"
    out = blend_videos(base, overlay, output_dir=d,
                        mode=_blend_mode,
                        opacity=safe_float(data.get("opacity", 0.5), 0.5, min_val=0.0, max_val=1.0), on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/transitions/list", methods=["GET"])
def transitions_list():
    try:
        from opencut.core.transitions_3d import get_transition_list
        return jsonify({"transitions": get_transition_list()})
    except Exception as e:
        return safe_error(e, "transitions_list")


@video_fx_bp.route("/video/transitions/apply", methods=["POST"])
@require_csrf
@async_job("transition", filepath_param="clip_a")
def transitions_apply(job_id, filepath, data):
    """Apply transition between two clips."""
    a = data.get("clip_a", "").strip()
    b = data.get("clip_b", "").strip()
    if not a:
        raise ValueError("Clip A not found")
    if not b:
        raise ValueError("Clip B not found")
    try:
        a = validate_filepath(a)
    except ValueError as e:
        raise ValueError(f"Clip A: {e}")
    try:
        b = validate_filepath(b)
    except ValueError as e:
        raise ValueError(f"Clip B: {e}")

    from opencut.core.transitions_3d import apply_transition

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(a, data.get("output_dir", ""))
    from opencut.core.transitions_3d import XFADE_TRANSITIONS
    _tr = data.get("transition", "fade")
    if _tr not in XFADE_TRANSITIONS:
        _tr = "fade"
    out = apply_transition(a, b, output_dir=d,
                            transition=_tr,
                            duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=10.0),
                            on_progress=_p)
    return {"output_path": out}


@video_fx_bp.route("/video/transitions/join", methods=["POST"])
@require_csrf
@async_job("join", filepath_required=False)
def transitions_join(job_id, filepath, data):
    """Join multiple clips with transitions."""
    clips = data.get("clips", [])
    if len(clips) < 2:
        raise ValueError("Need 2+ clips")
    if len(clips) > MAX_BATCH_FILES:
        raise ValueError(f"Too many clips (max {MAX_BATCH_FILES})")
    validated_clips = []
    for c in clips:
        validated_clips.append(validate_filepath(c))
    clips = validated_clips

    from opencut.core.transitions_3d import join_with_transitions

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(clips[0], data.get("output_dir", ""))
    from opencut.core.transitions_3d import XFADE_TRANSITIONS
    _tr = data.get("transition", "fade")
    if _tr not in XFADE_TRANSITIONS:
        _tr = "fade"
    out = join_with_transitions(clips, output_dir=d,
                                 transition=_tr,
                                 duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=10.0),
                                 on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/particles/presets", methods=["GET"])
def particle_presets():
    try:
        from opencut.core.particles import get_particle_presets
        return jsonify({"presets": get_particle_presets()})
    except Exception as e:
        return safe_error(e, "particle_presets")


@video_fx_bp.route("/video/particles/apply", methods=["POST"])
@require_csrf
@async_job("particles")
def particle_apply(job_id, filepath, data):
    """Overlay particle effects on video."""
    from opencut.core.particles import overlay_particles

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out = overlay_particles(filepath, output_dir=d,
                             preset=data.get("preset", "confetti"),
                             density=safe_float(data.get("density", 1.0), 1.0, min_val=0.1, max_val=5.0),
                             on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Color Management
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/color/capabilities", methods=["GET"])
def color_capabilities():
    try:
        from opencut.core.color_management import get_color_capabilities
        return jsonify(get_color_capabilities())
    except Exception as e:
        return safe_error(e, "color_capabilities")


@video_fx_bp.route("/video/color/correct", methods=["POST"])
@require_csrf
@async_job("color_correct")
def color_correct_route(job_id, filepath, data):
    """Apply color correction (exposure, contrast, saturation, temperature, etc.)."""
    from opencut.core.color_management import color_correct

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out = color_correct(filepath, output_dir=d,
                         exposure=safe_float(data.get("exposure", 0), 0.0, min_val=-5.0, max_val=5.0),
                         contrast=safe_float(data.get("contrast", 1.0), 1.0, min_val=0.0, max_val=3.0),
                         saturation=safe_float(data.get("saturation", 1.0), 1.0, min_val=0.0, max_val=3.0),
                         temperature=safe_float(data.get("temperature", 0), 0.0, min_val=-100.0, max_val=100.0),
                         tint=safe_float(data.get("tint", 0), 0.0, min_val=-100.0, max_val=100.0),
                         shadows=safe_float(data.get("shadows", 0), 0.0, min_val=-100.0, max_val=100.0),
                         midtones=safe_float(data.get("midtones", 0), 0.0, min_val=-100.0, max_val=100.0),
                         highlights=safe_float(data.get("highlights", 0), 0.0, min_val=-100.0, max_val=100.0),
                         on_progress=_p)
    return {"output_path": out}


@video_fx_bp.route("/video/color/convert", methods=["POST"])
@require_csrf
@async_job("color_convert")
def color_convert_route(job_id, filepath, data):
    """Convert video color space."""
    from opencut.core.color_management import convert_colorspace

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    _VALID_COLORSPACES = {"rec709", "rec2020", "srgb", "dci_p3"}
    target_cs = data.get("target", "rec709")
    if target_cs not in _VALID_COLORSPACES:
        target_cs = "rec709"
    out = convert_colorspace(filepath, target=target_cs,
                              output_dir=d, on_progress=_p)
    return {"output_path": out}


@video_fx_bp.route("/video/color/external-lut", methods=["POST"])
@require_csrf
@async_job("color_external_lut")
def color_external_lut_route(job_id, filepath, data):
    """Apply external .cube/.3dl LUT file."""
    lut = data.get("lut_path", "").strip()
    if not lut:
        raise ValueError("LUT file not found")
    lut = validate_filepath(lut)

    from opencut.core.color_management import apply_external_lut

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out = apply_external_lut(filepath, lut, intensity=safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=2.0),
                              output_dir=d, on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# AI LUT Generation from Reference Image
# ---------------------------------------------------------------------------
@video_fx_bp.route("/video/lut/generate-from-ref", methods=["POST"])
@require_csrf
@async_job("lut_from_ref", filepath_param="reference_path")
def video_lut_from_ref(job_id, filepath, data):
    """Generate a .cube LUT from a reference image's color palette."""
    reference_path = data.get("reference_path", "").strip()
    if not reference_path:
        raise ValueError("No reference image path provided")
    try:
        reference_path = validate_filepath(reference_path)
    except ValueError as e:
        raise ValueError(str(e))
    lut_name = data.get("lut_name", "").strip() or "custom_ref_lut"
    method = data.get("method", "histogram").strip()
    if method not in ("histogram", "average"):
        method = "histogram"
    strength = safe_float(data.get("strength", 0.8), 0.8, min_val=0.1, max_val=1.0)

    from opencut.core.lut_library import generate_lut_from_reference

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    cube_path = generate_lut_from_reference(
        reference_path,
        lut_name=lut_name,
        method=method,
        strength=strength,
        on_progress=_on_progress,
    )

    return {"lut_path": cube_path, "lut_name": lut_name}


@video_fx_bp.route("/video/lut/generate-ai", methods=["POST"])
@require_csrf
@async_job("lut_ai", filepath_param="reference_path")
def video_lut_ai(job_id, filepath, data):
    """Generate a .cube LUT using AI perceptual LAB color matching."""
    reference_path = data.get("reference_path", "").strip()
    if not reference_path:
        raise ValueError("No reference image path provided")
    try:
        reference_path = validate_filepath(reference_path)
    except ValueError as e:
        raise ValueError(str(e))
    lut_name = data.get("lut_name", "").strip() or ""
    _RESERVED = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "LPT1", "LPT2", "LPT3"}
    if lut_name and (lut_name.upper().split(".")[0] in _RESERVED or re.search(r'[<>:"/\\|?*]', lut_name)):
        raise ValueError("Invalid LUT name")

    from opencut.core.lut_library import generate_lut_ai

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    cube_path = generate_lut_ai(
        reference_path, lut_name=lut_name, on_progress=_on_progress,
    )
    return {"lut_path": cube_path}


@video_fx_bp.route("/video/lut/blend", methods=["POST"])
@require_csrf
def video_lut_blend():
    """Blend two LUTs into a new LUT with a single slider."""
    data = request.get_json(force=True)
    lut_a = data.get("lut_a", "").strip()
    lut_b = data.get("lut_b", "").strip()
    blend_val = safe_float(data.get("blend", 0.5), 0.5, min_val=0.0, max_val=1.0)

    if not lut_a or not lut_b:
        return jsonify({"error": "Two LUT names required"}), 400

    try:
        from opencut.core.lut_library import blend_luts
        cube_path = blend_luts(lut_a, lut_b, blend=blend_val, output_name=data.get("output_name", ""))
        return jsonify({"success": True, "lut_path": cube_path})
    except Exception as e:
        return safe_error(e, "video_lut_blend")
