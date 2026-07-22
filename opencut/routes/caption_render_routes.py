"""Caption burn-in and animated rendering routes."""

from .captions import (
    _VALID_ANIMATIONS,
    _VALID_BURNIN_STYLES,
    _resolve_output_dir,
    _sanitize_force_style,
    _update_job,
    async_job,
    captions_bp,
    jsonify,
    require_csrf,
    safe_error,
    safe_int,
    validate_filepath,
    validate_path,
    workflow_step,
)

# Caption Burn-in
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/burnin/styles", methods=["GET"])
def burnin_styles():
    """Return available burn-in styles."""
    try:
        from opencut.core.caption_burnin import get_burnin_styles
        return jsonify({"styles": get_burnin_styles()})
    except Exception as e:
        return safe_error(e, "burnin_styles")


@captions_bp.route("/captions/burnin/file", methods=["POST"])
@require_csrf
@workflow_step("Burning in captions")
@async_job("burnin", disk_operation="video_export")
def burnin_from_file(job_id, filepath, data):
    """Burn a subtitle file into video."""
    video_path = filepath
    subtitle_path = data.get("subtitle_path", "").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not subtitle_path:
        raise ValueError("Subtitle file not found")
    subtitle_path = validate_filepath(subtitle_path)

    from opencut.core.caption_burnin import burnin_subtitles
    from opencut.core.caption_display_settings import settings_to_ass_force_style

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(video_path, output_dir)
    display_settings = data.get("display_settings")
    force_style = (
        settings_to_ass_force_style(display_settings)
        if isinstance(display_settings, dict)
        else _sanitize_force_style(data.get("force_style", ""))
    )
    out = burnin_subtitles(
        video_path, subtitle_path,
        output_dir=effective_dir,
        font_size=safe_int(data.get("font_size", 0)),
        margin_bottom=safe_int(data.get("margin_bottom", 0)),
        force_style=force_style,
        on_progress=_on_progress,
    )
    return {"output_path": out}


@captions_bp.route("/captions/burnin/segments", methods=["POST"])
@require_csrf
@async_job("burnin-seg")
def burnin_from_segments(job_id, filepath, data):
    """Burn caption segments directly into video."""
    video_path = filepath
    segments = data.get("segments", [])
    style = data.get("style", "default")
    if style not in _VALID_BURNIN_STYLES:
        style = "default"
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not segments:
        raise ValueError("No segments provided")
    if len(segments) > 10000:
        raise ValueError("Too many segments (max 10000)")

    from opencut.core.caption_burnin import burnin_segments

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(video_path, output_dir)
    out = burnin_segments(
        video_path, segments, output_dir=effective_dir,
        style=style, on_progress=_on_progress,
    )
    return {"output_path": out, "style": style}


# ---------------------------------------------------------------------------
# Animated Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/animated/presets", methods=["GET"])
def animated_caption_presets():
    try:
        from opencut.core.animated_captions import get_animation_presets
        return jsonify({"presets": get_animation_presets()})
    except Exception as e:
        return safe_error(e, "animated_caption_presets")


@captions_bp.route("/captions/animated/render", methods=["POST"])
@require_csrf
@workflow_step("Rendering animated captions")
@async_job("anim-cap")
def animated_caption_render(job_id, filepath, data):
    """Render animated word-by-word captions onto video."""
    words = data.get("word_segments", [])

    if not words:
        raise ValueError("No word segments")
    if len(words) > 50000:
        raise ValueError("Too many word segments (max 50000)")

    from opencut.core.animated_captions import render_animated_captions

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    _anim = data.get("animation", "pop")
    if _anim not in _VALID_ANIMATIONS:
        _anim = "pop"
    out = render_animated_captions(filepath, words, output_dir=d,
                                    animation=_anim,
                                    font_size=safe_int(data.get("font_size", 56), default=56),
                                    max_words_per_line=safe_int(data.get("max_words", 6), default=6),
                                    on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
