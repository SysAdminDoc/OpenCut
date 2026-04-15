"""
OpenCut Delivery & Mastering Routes (Category 70)

Blueprint providing DCP export, IMF packaging, delivery validation,
delivery spec management, and multi-format rendering endpoints.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")

delivery_master_bp = Blueprint("delivery_master", __name__)


# ===========================================================================
# DCP Export
# ===========================================================================

@delivery_master_bp.route("/export/dcp", methods=["POST"])
@require_csrf
@async_job("dcp_export")
def export_dcp(job_id, filepath, data):
    """Export a video as a Digital Cinema Package (DCP).

    Expects JSON::

        {
            "filepath": "/path/to/video.mp4",
            "output_dir": "/path/to/output",
            "title": "My Film",
            "format_key": "2K_flat",
            "frame_rate": 24,
            "content_kind": "feature",
            "audio_channels": 6,
            "rating": "PG-13"
        }
    """
    from opencut.core.dcp_export import DCPConfig, export_dcp as _export_dcp

    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    if not output_dir:
        output_dir = os.path.dirname(filepath)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    config = DCPConfig(
        title=data.get("title", "Untitled"),
        format_key=data.get("format_key", "2K_flat"),
        frame_rate=safe_int(data.get("frame_rate"), default=24, min_val=24, max_val=48),
        content_kind=data.get("content_kind", "feature"),
        rating=data.get("rating", ""),
        issuer=data.get("issuer", "OpenCut"),
        creator=data.get("creator", "OpenCut DCP Export"),
        annotation=data.get("annotation", ""),
        audio_channels=safe_int(data.get("audio_channels"), default=6, min_val=1, max_val=16),
        audio_sample_rate=safe_int(data.get("audio_sample_rate"), default=48000),
        audio_bit_depth=safe_int(data.get("audio_bit_depth"), default=24),
        jpeg2000_bandwidth=safe_int(data.get("jpeg2000_bandwidth"), default=0),
        encrypt=bool(data.get("encrypt", False)),
        three_d=bool(data.get("three_d", False)),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _export_dcp(filepath, output_dir, config, on_progress=_progress)
    return result


# ===========================================================================
# IMF Export
# ===========================================================================

@delivery_master_bp.route("/export/imf", methods=["POST"])
@require_csrf
@async_job("imf_export")
def export_imf(job_id, filepath, data):
    """Export a video as an IMF package.

    Expects JSON::

        {
            "filepath": "/path/to/video.mp4",
            "output_dir": "/path/to/output",
            "title": "My Film",
            "profile": "application_2",
            "frame_rate": 24,
            "width": 1920,
            "height": 1080,
            "audio_tracks": [
                {"language": "en", "label": "English"},
                {"language": "es", "label": "Spanish", "source_path": "/path/to/es.wav"}
            ]
        }
    """
    from opencut.core.imf_package import IMFAudioTrack, IMFConfig, export_imf as _export_imf

    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    if not output_dir:
        output_dir = os.path.dirname(filepath)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Parse audio tracks
    audio_tracks = []
    for t in data.get("audio_tracks", []):
        track = IMFAudioTrack(
            language=t.get("language", "en"),
            label=t.get("label", ""),
            source_path=t.get("source_path", ""),
            channels=safe_int(t.get("channels"), default=6, min_val=1, max_val=16),
            sample_rate=safe_int(t.get("sample_rate"), default=48000),
        )
        audio_tracks.append(track)

    config = IMFConfig(
        title=data.get("title", "Untitled"),
        profile=data.get("profile", "application_2"),
        frame_rate=safe_int(data.get("frame_rate"), default=24),
        width=safe_int(data.get("width"), default=1920),
        height=safe_int(data.get("height"), default=1080),
        video_codec=data.get("video_codec", "jpeg2000"),
        audio_tracks=audio_tracks,
        issuer=data.get("issuer", "OpenCut"),
        creator=data.get("creator", "OpenCut IMF Export"),
        annotation=data.get("annotation", ""),
        content_kind=data.get("content_kind", "feature"),
        timecode_start=data.get("timecode_start", "01:00:00:00"),
        audio_bit_depth=safe_int(data.get("audio_bit_depth"), default=24),
        supplemental=bool(data.get("supplemental", False)),
        original_cpl_id=data.get("original_cpl_id", ""),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _export_imf(filepath, output_dir, config, on_progress=_progress)
    return result


# ===========================================================================
# Delivery Validation
# ===========================================================================

@delivery_master_bp.route("/delivery/validate", methods=["POST"])
@require_csrf
@async_job("delivery_validate")
def delivery_validate(job_id, filepath, data):
    """Validate a media file against a delivery specification.

    Expects JSON::

        {
            "filepath": "/path/to/video.mp4",
            "spec_name": "netflix",
            "measure_loudness": true
        }
    """
    from opencut.core.delivery_validate import validate_delivery

    spec_name = data.get("spec_name", "netflix").strip()
    measure_loudness = data.get("measure_loudness", True)
    if isinstance(measure_loudness, str):
        measure_loudness = measure_loudness.lower() in ("true", "1", "yes")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = validate_delivery(
        filepath,
        spec_name=spec_name,
        measure_loudness=bool(measure_loudness),
        on_progress=_progress,
    )
    return result


# ===========================================================================
# Delivery Specs (Synchronous)
# ===========================================================================

@delivery_master_bp.route("/delivery/specs", methods=["GET"])
def delivery_specs_list():
    """List all available delivery specifications."""
    from opencut.core.delivery_spec import list_specs

    specs = list_specs()
    return jsonify({"specs": specs, "count": len(specs)})


@delivery_master_bp.route("/delivery/specs/<name>", methods=["GET"])
def delivery_spec_get(name):
    """Get a specific delivery specification by name."""
    from opencut.core.delivery_spec import get_spec

    spec = get_spec(name)
    if spec is None:
        return jsonify({"error": f"Spec not found: {name}"}), 404
    return jsonify({"spec": spec})


@delivery_master_bp.route("/delivery/specs", methods=["POST"])
@require_csrf
def delivery_spec_create():
    """Create a custom delivery specification.

    Expects JSON::

        {
            "name": "my_custom_spec",
            "display_name": "My Custom Spec",
            "description": "Custom delivery spec for internal use",
            "platform": "custom",
            "requirements": [
                {
                    "category": "video",
                    "field_name": "codec",
                    "operator": "in",
                    "value": ["h264", "h265"],
                    "severity": "error",
                    "description": "H.264 or H.265 required"
                }
            ]
        }
    """
    from opencut.core.delivery_spec import create_custom_spec

    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    try:
        result = create_custom_spec(
            name=name,
            requirements=data.get("requirements"),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            platform=data.get("platform", "custom"),
            metadata=data.get("metadata"),
        )
        return jsonify({"success": True, "spec": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# Delivery Spec Comparison
# ===========================================================================

@delivery_master_bp.route("/delivery/specs/compare", methods=["POST"])
@require_csrf
def delivery_spec_compare():
    """Compare two delivery specifications.

    Expects JSON::

        {
            "spec_a": "netflix",
            "spec_b": "amazon"
        }
    """
    from opencut.core.delivery_spec import compare_specs

    data = request.get_json(force=True)
    spec_a = data.get("spec_a", "").strip()
    spec_b = data.get("spec_b", "").strip()

    if not spec_a or not spec_b:
        return jsonify({"error": "spec_a and spec_b are required"}), 400

    try:
        result = compare_specs(spec_a, spec_b)
        return jsonify({"success": True, "comparison": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ===========================================================================
# Multi-Render
# ===========================================================================

@delivery_master_bp.route("/render/multi", methods=["POST"])
@require_csrf
@async_job("multi_render")
def render_multi(job_id, filepath, data):
    """Execute multiple renders of a video file.

    Expects JSON::

        {
            "filepath": "/path/to/video.mp4",
            "parallel": false,
            "max_parallel": 3,
            "configs": [
                {
                    "name": "YouTube 1080p",
                    "format": "mp4",
                    "video_codec": "libx264",
                    "width": 1920,
                    "height": 1080,
                    "crf": 18,
                    "priority": 10
                },
                {
                    "name": "Instagram 720p",
                    "format": "mp4",
                    "video_codec": "libx264",
                    "width": 1280,
                    "height": 720,
                    "crf": 20,
                    "priority": 5
                }
            ]
        }
    """
    from opencut.core.multi_render import RenderConfig, multi_render as _multi_render

    configs_raw = data.get("configs", [])
    if not configs_raw:
        raise ValueError("At least one render config is required in 'configs'")

    configs = []
    for idx, c in enumerate(configs_raw):
        rc = RenderConfig(
            name=c.get("name", f"render_{idx}"),
            format=c.get("format", "mp4"),
            video_codec=c.get("video_codec", "libx264"),
            audio_codec=c.get("audio_codec", "aac"),
            width=safe_int(c.get("width"), default=0),
            height=safe_int(c.get("height"), default=0),
            frame_rate=safe_float(c.get("frame_rate"), default=0.0),
            video_bitrate=c.get("video_bitrate", ""),
            audio_bitrate=c.get("audio_bitrate", "192k"),
            crf=safe_int(c.get("crf"), default=18, min_val=0, max_val=63),
            preset=c.get("preset", "medium"),
            pixel_format=c.get("pixel_format", "yuv420p"),
            audio_sample_rate=safe_int(c.get("audio_sample_rate"), default=0),
            audio_channels=safe_int(c.get("audio_channels"), default=0),
            priority=safe_int(c.get("priority"), default=0),
            output_suffix=c.get("output_suffix", ""),
            output_dir=c.get("output_dir", ""),
            extra_ffmpeg_args=c.get("extra_ffmpeg_args", []),
            two_pass=bool(c.get("two_pass", False)),
            metadata=c.get("metadata", {}),
        )
        configs.append(rc)

    parallel = data.get("parallel", False)
    if isinstance(parallel, str):
        parallel = parallel.lower() in ("true", "1", "yes")
    max_parallel = safe_int(data.get("max_parallel"), default=3, min_val=1, max_val=8)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _multi_render(
        filepath,
        configs=configs,
        parallel=bool(parallel),
        max_parallel=max_parallel,
        on_progress=_progress,
    )
    return result


# ===========================================================================
# Multi-Render Cancel
# ===========================================================================

@delivery_master_bp.route("/render/multi/cancel", methods=["POST"])
@require_csrf
def render_multi_cancel():
    """Cancel a specific render within a multi-render batch.

    Expects JSON::

        {
            "render_id": "uuid-string"
        }
    """
    from opencut.core.multi_render import cancel_render

    data = request.get_json(force=True)
    render_id = data.get("render_id", "").strip()

    if not render_id:
        return jsonify({"error": "render_id is required"}), 400

    cancelled = cancel_render(render_id)
    return jsonify({
        "success": cancelled,
        "render_id": render_id,
        "message": "Render cancelled" if cancelled else "Render not found or already complete",
    })
