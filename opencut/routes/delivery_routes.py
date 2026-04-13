"""
OpenCut Delivery Routes - Broadcast, Streaming, Export & Compliance

Blueprint: delivery_bp
Routes:
  POST /delivery/broadcast-qc       -> async broadcast standards QC check
  POST /delivery/broadcast-qc/audio -> async audio-only broadcast QC
  POST /delivery/broadcast-qc/report -> async generate QC report file
  POST /delivery/thumbnail-ab       -> async thumbnail A/B generation
  POST /delivery/end-screen         -> async end screen rendering
  POST /delivery/end-screen/templates -> sync list end screen templates
  POST /delivery/news-ticker        -> async news ticker overlay
  POST /delivery/news-ticker/standalone -> async standalone ticker video
  POST /delivery/hls                -> async HLS package creation
  POST /delivery/dash               -> async DASH package creation
  POST /delivery/caption/ebu-tt     -> async EBU-TT caption export
  POST /delivery/caption/ttml       -> async TTML/IMSC1 caption export
  POST /delivery/caption/embed-cc   -> async CEA-608 caption embedding
  POST /delivery/fcpxml             -> async FCPXML 1.11 export
"""

import logging
import os

from flask import Blueprint, jsonify

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
)

logger = logging.getLogger("opencut")

delivery_bp = Blueprint("delivery", __name__)


# ---------------------------------------------------------------------------
# POST /delivery/broadcast-qc
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/broadcast-qc", methods=["POST"])
@require_csrf
@async_job("broadcast_qc")
def broadcast_qc_route(job_id, filepath, data):
    """Run full broadcast QC check (audio, video, codecs, resolution, CC)."""
    from opencut.core.broadcast_qc import check_broadcast_standards

    standard = str(data.get("standard", "ebu_r128")).strip().lower()
    if standard not in ("ebu_r128", "atsc_a85", "arib_tr_b32"):
        standard = "ebu_r128"

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    from dataclasses import asdict

    report = check_broadcast_standards(
        video_path=filepath,
        standard=standard,
        check_audio=safe_bool(data.get("check_audio", True), True),
        check_video=safe_bool(data.get("check_video", True), True),
        check_codecs=safe_bool(data.get("check_codecs", True), True),
        check_resolution_flag=safe_bool(data.get("check_resolution", True), True),
        check_captions_flag=safe_bool(data.get("check_captions", True), True),
        on_progress=_progress,
    )

    result = {
        "overall_pass": report.overall_pass,
        "standard": report.standard,
        "total_checks": report.total_checks,
        "passed_checks": report.passed_checks,
        "failed_checks": report.failed_checks,
        "warning_checks": report.warning_checks,
        "duration_seconds": report.duration_seconds,
        "checks": [asdict(c) for c in report.checks],
    }

    if report.audio_levels:
        result["audio_levels"] = {
            "integrated_loudness": report.audio_levels.integrated_loudness,
            "loudness_range": report.audio_levels.loudness_range,
            "true_peak": report.audio_levels.true_peak,
            "passed": report.audio_levels.passed,
        }

    if report.video_levels:
        result["video_levels"] = {
            "ymin": report.video_levels.ymin,
            "ymax": report.video_levels.ymax,
            "yavg": report.video_levels.yavg,
            "passed": report.video_levels.passed,
        }

    return result


# ---------------------------------------------------------------------------
# POST /delivery/broadcast-qc/audio
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/broadcast-qc/audio", methods=["POST"])
@require_csrf
@async_job("broadcast_qc_audio")
def broadcast_qc_audio_route(job_id, filepath, data):
    """Audio-only broadcast QC check (EBU R128 / ATSC A/85 loudness)."""
    from opencut.core.broadcast_qc import check_audio_levels

    standard = str(data.get("standard", "ebu_r128")).strip().lower()
    if standard not in ("ebu_r128", "atsc_a85", "arib_tr_b32"):
        standard = "ebu_r128"

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    from dataclasses import asdict

    result = check_audio_levels(
        video_path=filepath,
        standard=standard,
        on_progress=_progress,
    )

    return {
        "integrated_loudness": result.integrated_loudness,
        "loudness_range": result.loudness_range,
        "true_peak": result.true_peak,
        "standard": result.standard,
        "target_loudness": result.target_loudness,
        "passed": result.passed,
        "checks": [asdict(c) for c in result.checks],
    }


# ---------------------------------------------------------------------------
# POST /delivery/broadcast-qc/report
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/broadcast-qc/report", methods=["POST"])
@require_csrf
@async_job("broadcast_qc_report")
def broadcast_qc_report_route(job_id, filepath, data):
    """Generate a broadcast QC report file (JSON or text)."""
    from opencut.core.broadcast_qc import check_broadcast_standards, generate_qc_report

    standard = str(data.get("standard", "ebu_r128")).strip().lower()
    if standard not in ("ebu_r128", "atsc_a85", "arib_tr_b32"):
        standard = "ebu_r128"

    report_format = str(data.get("format", "json")).strip().lower()
    if report_format not in ("json", "text"):
        report_format = "json"

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Running broadcast QC checks...")
    qc_results = check_broadcast_standards(
        video_path=filepath,
        standard=standard,
        on_progress=lambda pct, msg: _progress(min(pct // 2, 45), msg),
    )

    # Determine output path
    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    ext = ".txt" if report_format == "text" else ".json"
    base = os.path.splitext(os.path.basename(filepath))[0]
    report_path = os.path.join(out_dir, f"{base}_qc_report{ext}")

    _progress(50, "Generating report file...")
    result = generate_qc_report(
        results=qc_results,
        output_path=report_path,
        format=report_format,
        on_progress=lambda pct, msg: _progress(50 + pct // 2, msg),
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/thumbnail-ab
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/thumbnail-ab", methods=["POST"])
@require_csrf
@async_job("thumbnail_ab")
def thumbnail_ab_route(job_id, filepath, data):
    """Score candidate frames and generate thumbnail variants."""
    from opencut.core.thumbnail_ab import (
        create_thumbnail_grid,
        generate_variants,
        score_frames,
    )

    count = safe_int(data.get("count", 5), 5, min_val=1, max_val=20)
    sample_count = safe_int(data.get("sample_count", 20), 20, min_val=5, max_val=100)
    text = str(data.get("text", ""))
    width = safe_int(data.get("width", 1280), 1280, min_val=320, max_val=3840)
    height = safe_int(data.get("height", 720), 720, min_val=180, max_val=2160)
    create_grid = safe_bool(data.get("create_grid", True), True)

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Scoring candidate frames...")
    top_frames = score_frames(
        video_path=filepath,
        count=count,
        sample_count=sample_count,
        on_progress=lambda pct, msg: _progress(min(pct // 2, 40), msg),
    )

    if not top_frames:
        raise RuntimeError("No suitable frames found in video.")

    # Generate variants for the best frame
    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    _progress(45, "Generating thumbnail variants...")
    variants = generate_variants(
        frame_path=top_frames[0].frame_path,
        text=text,
        output_dir=out_dir,
        width=width,
        height=height,
        on_progress=lambda pct, msg: _progress(45 + pct // 4, msg),
    )

    result = {
        "top_frames": [
            {
                "timestamp": f.timestamp,
                "sharpness": f.sharpness,
                "brightness": f.brightness,
                "contrast": f.contrast,
                "score": f.overall_score,
                "path": f.frame_path,
            }
            for f in top_frames
        ],
        "variants": [
            {
                "type": v.variant_type,
                "path": v.path,
                "label": v.label,
            }
            for v in variants
        ],
    }

    # Create grid if requested
    if create_grid and variants:
        _progress(75, "Creating comparison grid...")
        variant_paths = [v.path for v in variants if os.path.isfile(v.path)]
        if variant_paths:
            grid_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_thumb_grid.jpg")
            grid = create_thumbnail_grid(
                thumbnails=variant_paths,
                output_path_str=grid_path,
                columns=min(2, len(variant_paths)),
                on_progress=lambda pct, msg: _progress(75 + pct // 5, msg),
            )
            result["grid"] = grid

    return result


# ---------------------------------------------------------------------------
# POST /delivery/end-screen
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/end-screen", methods=["POST"])
@require_csrf
@async_job("end_screen", filepath_required=False)
def end_screen_route(job_id, filepath, data):
    """Render an animated end screen video from a template."""
    from opencut.core.end_screen import generate_end_screen

    template = str(data.get("template", "youtube_classic")).strip()
    duration = safe_float(data.get("duration", 10.0), 10.0, min_val=5.0, max_val=20.0)
    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=3840)
    height = safe_int(data.get("height", 1080), 1080, min_val=360, max_val=2160)
    fps = safe_int(data.get("fps", 30), 30, min_val=24, max_val=60)
    fade_duration = safe_float(data.get("fade_duration", 1.0), 1.0, min_val=0.0, max_val=5.0)

    out_path = data.get("output_path", "")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_end_screen(
        template=template,
        data=data,
        duration=duration,
        output_path_str=out_path,
        width=width,
        height=height,
        fps=fps,
        fade_duration=fade_duration,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/end-screen/templates (synchronous)
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/end-screen/templates", methods=["POST"])
@require_csrf
def end_screen_templates_route():
    """List available end screen templates."""
    from opencut.core.end_screen import list_templates

    templates = list_templates()
    return jsonify({"templates": templates})


# ---------------------------------------------------------------------------
# POST /delivery/news-ticker
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/news-ticker", methods=["POST"])
@require_csrf
@async_job("news_ticker")
def news_ticker_route(job_id, filepath, data):
    """Overlay a scrolling news ticker on a video."""
    from opencut.core.news_ticker import create_ticker

    text_content = data.get("text", data.get("text_content", ""))
    if not text_content:
        raise ValueError("No ticker text provided. Pass 'text' in the request body.")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = data.get("output_path", "")
    if not out_path:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(out_dir, f"{base}_ticker.mp4")

    result = create_ticker(
        text_content=text_content,
        video_path=filepath,
        output_path_str=out_path,
        speed=safe_int(data.get("speed", 100), 100, min_val=10, max_val=1000),
        font_size=safe_int(data.get("font_size", 48), 48, min_val=12, max_val=200),
        font_color=str(data.get("font_color", "white")),
        bg_color=str(data.get("bg_color", "black@0.7")),
        direction=str(data.get("direction", "left")),
        position=str(data.get("position", "bottom")),
        margin=safe_int(data.get("margin", 20), 20, min_val=0, max_val=200),
        separator=str(data.get("separator", "   +++   ")),
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/news-ticker/standalone
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/news-ticker/standalone", methods=["POST"])
@require_csrf
@async_job("news_ticker_standalone", filepath_required=False)
def news_ticker_standalone_route(job_id, filepath, data):
    """Create a standalone ticker overlay video (no source video needed)."""
    from opencut.core.news_ticker import create_ticker_overlay

    text_content = data.get("text", data.get("text_content", ""))
    if not text_content:
        raise ValueError("No ticker text provided. Pass 'text' in the request body.")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = create_ticker_overlay(
        text_content=text_content,
        duration=safe_float(data.get("duration", 10.0), 10.0, min_val=1.0, max_val=300.0),
        output_path_str=data.get("output_path", ""),
        width=safe_int(data.get("width", 1920), 1920, min_val=320, max_val=3840),
        height=safe_int(data.get("height", 80), 80, min_val=30, max_val=500),
        speed=safe_int(data.get("speed", 100), 100, min_val=10, max_val=1000),
        font_size=safe_int(data.get("font_size", 48), 48, min_val=12, max_val=200),
        font_color=str(data.get("font_color", "white")),
        bg_color=str(data.get("bg_color", "000000")),
        direction=str(data.get("direction", "left")),
        separator=str(data.get("separator", "   +++   ")),
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/hls
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/hls", methods=["POST"])
@require_csrf
@async_job("hls_package")
def hls_package_route(job_id, filepath, data):
    """Generate an HLS streaming package with multi-quality renditions."""
    from opencut.core.streaming_package import create_hls_package

    renditions = data.get("renditions", None)
    if isinstance(renditions, str):
        renditions = [r.strip() for r in renditions.split(",") if r.strip()]

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = create_hls_package(
        video_path=filepath,
        output_dir=data.get("output_dir", ""),
        renditions=renditions,
        segment_duration=safe_int(data.get("segment_duration", 6), 6, min_val=2, max_val=10),
        include_zip=safe_bool(data.get("include_zip", True), True),
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/dash
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/dash", methods=["POST"])
@require_csrf
@async_job("dash_package")
def dash_package_route(job_id, filepath, data):
    """Generate a DASH streaming package with .mpd manifest."""
    from opencut.core.streaming_package import create_dash_package

    renditions = data.get("renditions", None)
    if isinstance(renditions, str):
        renditions = [r.strip() for r in renditions.split(",") if r.strip()]

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = create_dash_package(
        video_path=filepath,
        output_dir=data.get("output_dir", ""),
        renditions=renditions,
        segment_duration=safe_int(data.get("segment_duration", 4), 4, min_val=2, max_val=10),
        include_zip=safe_bool(data.get("include_zip", True), True),
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/caption/ebu-tt
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/caption/ebu-tt", methods=["POST"])
@require_csrf
@async_job("caption_ebu_tt", filepath_required=False)
def caption_ebu_tt_route(job_id, filepath, data):
    """Export captions as EBU-TT (European Broadcasting Union) XML."""
    from opencut.core.broadcast_cc import export_ebu_tt

    captions = data.get("captions", data.get("segments", []))
    if not captions:
        raise ValueError("No caption data provided. Pass 'captions' in the request body.")

    out_path = data.get("output_path", "")
    if not out_path:
        import tempfile
        out_path = os.path.join(tempfile.gettempdir(), "opencut_ebu_tt.xml")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_ebu_tt(
        captions=captions,
        output_path=out_path,
        language=str(data.get("language", "en")),
        title=str(data.get("title", "")),
        frame_rate=safe_float(data.get("frame_rate", 30.0), 30.0, min_val=23.0, max_val=60.0),
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/caption/ttml
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/caption/ttml", methods=["POST"])
@require_csrf
@async_job("caption_ttml", filepath_required=False)
def caption_ttml_route(job_id, filepath, data):
    """Export captions as TTML / IMSC1 XML."""
    from opencut.core.broadcast_cc import export_ttml

    captions = data.get("captions", data.get("segments", []))
    if not captions:
        raise ValueError("No caption data provided. Pass 'captions' in the request body.")

    out_path = data.get("output_path", "")
    if not out_path:
        import tempfile
        out_path = os.path.join(tempfile.gettempdir(), "opencut_ttml.xml")

    profile = str(data.get("profile", "imsc1")).strip().lower()
    if profile not in ("imsc1", "ttml"):
        profile = "imsc1"

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_ttml(
        captions=captions,
        output_path=out_path,
        language=str(data.get("language", "en")),
        title=str(data.get("title", "")),
        profile=profile,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/caption/embed-cc
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/caption/embed-cc", methods=["POST"])
@require_csrf
@async_job("caption_embed_cc")
def caption_embed_cc_route(job_id, filepath, data):
    """Embed CEA-608/708 closed captions into a video file."""
    from opencut.core.broadcast_cc import embed_cea608

    captions = data.get("captions", data.get("segments", []))
    if not captions:
        raise ValueError("No caption data provided. Pass 'captions' in the request body.")

    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = data.get("output_path", "")
    if not out_path:
        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        out_path = os.path.join(out_dir, f"{base}_cc{ext}")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = embed_cea608(
        video_path=filepath,
        captions=captions,
        output_path=out_path,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /delivery/fcpxml
# ---------------------------------------------------------------------------
@delivery_bp.route("/delivery/fcpxml", methods=["POST"])
@require_csrf
@async_job("fcpxml_export", filepath_required=False)
def fcpxml_export_route(job_id, filepath, data):
    """Export a sequence as FCPXML 1.11 XML file."""
    from opencut.core.fcpxml_export import export_fcpxml

    sequence = data.get("sequence", data)
    out_path = data.get("output_path", "")
    if not out_path:
        import tempfile
        out_path = os.path.join(tempfile.gettempdir(), "opencut_export.fcpxml")

    project_name = str(data.get("project_name", data.get("name", "")))

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_fcpxml(
        sequence=sequence,
        output_path=out_path,
        project_name=project_name,
        on_progress=_progress,
    )

    return result
