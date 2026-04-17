"""
OpenCut Enhancement Routes (v1.17.0)

Two new feature surfaces pulled from the OSS research:

- **Neural frame interpolation** (RIFE-based, FFmpeg fallback) —
  ``/video/interpolate/neural``, ``/video/interpolate/backends``.
  Inspired by rife-ncnn-vulkan (https://github.com/nihui/rife-ncnn-vulkan).
- **Declarative JSON video composition** (editly.dev-inspired) —
  ``/compose/render``, ``/compose/schema``. Assemble a finished video
  from a single JSON spec — no timeline reasoning required by callers.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_path

logger = logging.getLogger("opencut")

enhancement_bp = Blueprint("enhancement", __name__)


# ---------------------------------------------------------------------------
# Neural frame interpolation
# ---------------------------------------------------------------------------

@enhancement_bp.route("/video/interpolate/backends", methods=["GET"])
def interp_backends():
    """List installed and available neural interpolation backends."""
    try:
        from opencut.core import neural_interp
        return jsonify({
            "backends": neural_interp.list_backends(),
            "priority": neural_interp.available_backends(),
        })
    except Exception as exc:
        return safe_error(exc, "interp_backends")


@enhancement_bp.route("/video/interpolate/neural", methods=["POST"])
@require_csrf
@async_job("neural_interp")
def interp_neural(job_id, filepath, data):
    """Upsample a video's frame rate using RIFE (neural) or FFmpeg (fallback).

    Request body::

        {
          "filepath": "/path/to/input.mp4",
          "target_fps": 60.0,          // required if preset omitted
          "backend": "auto",            // auto | rife_cli | rife_torch | ffmpeg
          "output": "optional/out.mp4"
        }

    Response (on success)::

        {
          "output": "...",
          "backend": "rife_cli",
          "input_fps": 30.0, "output_fps": 60.0,
          "duration": 42.1, "frames_added": 1263, "notes": [...]
        }
    """
    from opencut.core import neural_interp

    target_fps = data.get("target_fps")
    backend = str(data.get("backend") or "auto").strip().lower()
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = neural_interp.interpolate(
        video_path=filepath,
        target_fps=target_fps,
        backend=backend,
        output=output or None,
        on_progress=_on_progress,
    )
    # ``InterpResult`` is subscriptable for Flask jsonify.
    return dict(
        output=result.output,
        backend=result.backend,
        input_fps=result.input_fps,
        output_fps=result.output_fps,
        duration=result.duration,
        frames_added=result.frames_added,
        notes=list(result.notes),
    )


# ---------------------------------------------------------------------------
# Declarative composition
# ---------------------------------------------------------------------------

@enhancement_bp.route("/compose/schema", methods=["GET"])
def compose_schema():
    """Return the JSON schema summary for declarative composition.

    Lists supported clip types, transitions, and default dimensions so
    panel dropdowns and doc generators don't need to hard-code them.
    """
    try:
        from opencut.core import declarative_compose as dc
        return jsonify({
            "clip_types": dc.list_clip_types(),
            "transitions": dc.list_transitions(),
            "defaults": {
                "width": dc.DEFAULT_WIDTH,
                "height": dc.DEFAULT_HEIGHT,
                "fps": dc.DEFAULT_FPS,
                "bg_color": dc.DEFAULT_BG_COLOR,
            },
            "text_positions": sorted(dc.TEXT_POSITIONS),
            "max_clips": 200,
            "example": {
                "width": 1920, "height": 1080, "fps": 30,
                "clips": [
                    {"type": "title", "text": "Chapter One", "duration": 3.0, "bg": "#101014"},
                    {"type": "video", "source": "intro.mp4", "duration": 5.0,
                     "transition": {"name": "fade", "duration": 0.5}},
                    {"type": "image", "source": "logo.png", "duration": 2.0,
                     "transition": {"name": "dissolve", "duration": 0.4}},
                ],
                "audio": {"path": "bed.mp3", "volume": 0.25},
                "output": "final.mp4",
            },
        })
    except Exception as exc:
        return safe_error(exc, "compose_schema")


@enhancement_bp.route("/compose/validate", methods=["POST"])
@require_csrf
def compose_validate():
    """Validate a composition spec without rendering.  Returns a cleaned
    spec with defaults filled in, or a 400 with the validation error.
    """
    try:
        from opencut.core import declarative_compose as dc
        data = request.get_json(force=True) or {}
        spec = data.get("spec") or data
        cleaned = dc.validate_spec(spec)
        return jsonify({"valid": True, "spec": cleaned})
    except ValueError as exc:
        return jsonify({"valid": False, "error": str(exc),
                        "code": "INVALID_INPUT"}), 400
    except Exception as exc:
        return safe_error(exc, "compose_validate")


@enhancement_bp.route("/compose/render", methods=["POST"])
@require_csrf
@async_job("compose", filepath_required=False)
def compose_render(job_id, filepath, data):
    """Render a composition spec to a final video.

    Request body accepts either the spec inline or wrapped::

        { "spec": { ... } }   OR   { ... spec fields ... }

    Optional top-level ``output`` overrides ``spec.output``.
    """
    from opencut.core import declarative_compose as dc

    spec = data.get("spec")
    if not spec:
        # Treat the whole body as the spec (ignoring the async_job wrapper keys)
        spec = {k: v for k, v in data.items() if k not in ("output",)}
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    try:
        cleaned = dc.validate_spec(spec)
    except ValueError as exc:
        raise ValueError(f"invalid composition spec: {exc}") from exc

    result = dc.compose(cleaned, output=output or None, on_progress=_on_progress)
    return {
        "output": result.output,
        "duration": result.duration,
        "clip_count": result.clip_count,
        "transition_count": result.transition_count,
        "width": result.width,
        "height": result.height,
        "fps": result.fps,
        "notes": list(result.notes),
    }
