"""
OpenCut Multi-View & Content Repurposing Routes

Blueprint: multiview_repurpose_bp

Endpoints:
  - POST /split-screen/create      -- create split-screen composite
  - GET  /split-screen/layouts      -- list available layouts
  - POST /reaction/create           -- create reaction video
  - POST /comparison/export         -- export before/after video
  - POST /multicam-grid/export      -- export multicam grid view
  - POST /repurpose/extract-shorts  -- long-to-shorts pipeline
  - POST /repurpose/video-to-blog   -- generate blog post from video
  - POST /repurpose/podcast-bundle  -- full podcast processing bundle
  - POST /repurpose/content-calendar-- generate content calendar
  - POST /repurpose/social-captions -- generate platform-specific captions
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int, validate_output_path

logger = logging.getLogger("opencut")

multiview_repurpose_bp = Blueprint("multiview_repurpose", __name__)


# ---------------------------------------------------------------------------
# Split-Screen
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/split-screen/layouts", methods=["GET"])
def split_screen_layouts():
    """List available split-screen layouts."""
    try:
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        return jsonify({"layouts": layouts})
    except Exception as exc:
        return safe_error(exc, "split_screen_layouts")


@multiview_repurpose_bp.route("/split-screen/create", methods=["POST"])
@require_csrf
@async_job("split_screen", filepath_required=False)
def split_screen_create(job_id, filepath, data):
    """Create a split-screen composite from multiple videos."""
    from opencut.core.split_screen import create_split_screen
    from opencut.jobs import _update_job

    video_paths = data.get("video_paths", [])
    if not video_paths:
        raise ValueError("video_paths is required (list of file paths)")

    layout_name = data.get("layout", "side_by_side")
    custom_layout = data.get("custom_layout")
    output_width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    output_height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    border_width = safe_int(data.get("border_width", 0), 0, min_val=0, max_val=20)
    border_color = data.get("border_color", "black")
    gap = safe_int(data.get("gap", 0), 0, min_val=0, max_val=50)
    output_path = data.get("output_path")
    if output_path:
        output_path = validate_output_path(output_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_split_screen(
        video_paths=video_paths,
        layout_name=layout_name,
        custom_layout=custom_layout,
        output_width=output_width,
        output_height=output_height,
        border_width=border_width,
        border_color=border_color,
        gap=gap,
        output_path_str=output_path,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "layout_name": result.layout_name,
        "cell_count": result.cell_count,
        "width": result.width,
        "height": result.height,
    }


# ---------------------------------------------------------------------------
# Reaction Video
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/reaction/create", methods=["POST"])
@require_csrf
@async_job("reaction_video", filepath_required=False)
def reaction_create(job_id, filepath, data):
    """Create a reaction video composite."""
    from opencut.core.reaction_template import create_reaction_video
    from opencut.jobs import _update_job

    content_path = data.get("content_path", "")
    reaction_path = data.get("reaction_path", "")
    if not content_path or not reaction_path:
        raise ValueError("content_path and reaction_path are required")

    preset_name = data.get("preset", "corner_pip")
    output_width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    output_height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    auto_sync = safe_bool(data.get("auto_sync", False))
    audio_offset = safe_float(data.get("audio_offset", 0), 0)
    duck_level = safe_float(data.get("duck_level", 0.3), 0.3, min_val=0, max_val=1)
    output_path = data.get("output_path")
    if output_path:
        output_path = validate_output_path(output_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_reaction_video(
        content_path=content_path,
        reaction_path=reaction_path,
        preset_name=preset_name,
        output_width=output_width,
        output_height=output_height,
        auto_sync=auto_sync,
        audio_offset=audio_offset,
        duck_level=duck_level,
        output_path_str=output_path,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "preset_name": result.preset_name,
        "audio_offset": result.audio_offset,
        "width": result.width,
        "height": result.height,
    }


# ---------------------------------------------------------------------------
# Before/After Comparison Export
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/comparison/export", methods=["POST"])
@require_csrf
@async_job("comparison_export", filepath_required=False)
def comparison_export(job_id, filepath, data):
    """Export a before/after comparison video."""
    from opencut.core.video_compare import export_comparison_video
    from opencut.jobs import _update_job

    original = data.get("original", "")
    processed = data.get("processed", "")
    if not original or not processed:
        raise ValueError("original and processed paths are required")

    mode = data.get("mode", "vertical_wipe")
    label_original = data.get("label_original", "Original")
    label_processed = data.get("label_processed", "Processed")
    wipe_speed = safe_float(data.get("wipe_speed", 1.0), 1.0, min_val=0.1, max_val=5.0)
    output_path = data.get("output_path")
    if output_path:
        output_path = validate_output_path(output_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_comparison_video(
        original=original,
        processed=processed,
        mode=mode,
        out_path=output_path,
        label_original=label_original,
        label_processed=label_processed,
        wipe_speed=wipe_speed,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Multicam Grid
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/multicam-grid/export", methods=["POST"])
@require_csrf
@async_job("multicam_grid", filepath_required=False)
def multicam_grid_export(job_id, filepath, data):
    """Export a multicam grid view."""
    from opencut.core.multicam_grid import export_multicam_grid
    from opencut.jobs import _update_job

    video_paths = data.get("video_paths", [])
    if not video_paths:
        raise ValueError("video_paths is required")

    output_width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    output_height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    show_timecode = safe_bool(data.get("show_timecode", True), True)
    show_audio_meters = safe_bool(data.get("show_audio_meters", True), True)
    active_speaker = safe_bool(data.get("active_speaker_highlight", False))
    label_names = data.get("label_names")
    output_path = data.get("output_path")
    if output_path:
        output_path = validate_output_path(output_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_multicam_grid(
        video_paths=video_paths,
        output_width=output_width,
        output_height=output_height,
        show_timecode=show_timecode,
        show_audio_meters=show_audio_meters,
        active_speaker_highlight=active_speaker,
        label_names=label_names,
        output_path_str=output_path,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "grid_size": result.grid_size,
        "cell_count": result.cell_count,
        "width": result.width,
        "height": result.height,
    }


# ---------------------------------------------------------------------------
# Long-to-Shorts Extraction
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/repurpose/extract-shorts", methods=["POST"])
@require_csrf
@async_job("extract_shorts")
def repurpose_extract_shorts(job_id, filepath, data):
    """Extract multiple short clips from a long-form video."""
    from opencut.core.long_to_shorts import extract_shorts
    from opencut.jobs import _update_job

    num_shorts = safe_int(data.get("num_shorts", 5), 5, min_val=1, max_val=20)
    min_duration = safe_float(data.get("min_duration", 15), 15, min_val=5, max_val=120)
    max_duration = safe_float(data.get("max_duration", 60), 60, min_val=10, max_val=300)
    reframe = safe_bool(data.get("reframe_vertical", True), True)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_shorts(
        input_path=filepath,
        num_shorts=num_shorts,
        min_duration=min_duration,
        max_duration=max_duration,
        reframe_vertical=reframe,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return {
        "output_dir": result.output_dir,
        "total_shorts": result.total_shorts,
        "csv_path": result.csv_path,
        "segments": [
            {
                "index": s.index,
                "title": s.title,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "output_path": s.output_path,
            }
            for s in result.segments
        ],
    }


# ---------------------------------------------------------------------------
# Video to Blog Post
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/repurpose/video-to-blog", methods=["POST"])
@require_csrf
@async_job("video_to_blog")
def repurpose_video_to_blog(job_id, filepath, data):
    """Generate a blog post from a video."""
    from opencut.core.video_to_blog import generate_blog_post
    from opencut.jobs import _update_job

    tone = data.get("tone", "professional")
    extract_screenshots = safe_bool(data.get("extract_screenshots", True), True)
    output_format = data.get("output_format", "both")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_blog_post(
        video_path=filepath,
        tone=tone,
        extract_screenshots=extract_screenshots,
        output_format=output_format,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return {
        "title": result.title,
        "output_dir": result.output_dir,
        "word_count": result.word_count,
        "section_count": len(result.sections),
        "image_count": len(result.image_paths),
        "markdown_preview": result.markdown[:2000],
        "seo": {
            "title": result.seo.title,
            "meta_description": result.seo.meta_description,
            "keywords": result.seo.keywords,
            "slug": result.seo.slug,
            "reading_time_min": result.seo.reading_time_min,
        } if result.seo else {},
    }


# ---------------------------------------------------------------------------
# Podcast Bundle
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/repurpose/podcast-bundle", methods=["POST"])
@require_csrf
@async_job("podcast_bundle")
def repurpose_podcast_bundle(job_id, filepath, data):
    """Create a full podcast processing bundle."""
    from opencut.core.podcast_bundle import create_podcast_bundle
    from opencut.jobs import _update_job

    title = data.get("title", "")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    export_formats = data.get("export_formats")
    max_clips = safe_int(data.get("max_highlight_clips", 3), 3, min_val=0, max_val=10)
    audiogram = safe_bool(data.get("generate_audiogram", True), True)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_podcast_bundle(
        audio_path=filepath,
        title=title,
        output_dir=output_dir,
        export_formats=export_formats,
        max_highlight_clips=max_clips,
        generate_audiogram_flag=audiogram,
        on_progress=_progress,
    )

    return {
        "output_dir": result.output_dir,
        "clean_audio_path": result.clean_audio_path,
        "chapter_count": len(result.chapters),
        "highlight_count": len(result.highlight_clips),
        "audiogram_path": result.audiogram_path,
        "has_show_notes": bool(result.show_notes_markdown),
        "manifest": result.manifest,
    }


# ---------------------------------------------------------------------------
# Content Calendar
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/repurpose/content-calendar", methods=["POST"])
@require_csrf
def repurpose_content_calendar():
    """Generate a cross-platform content calendar."""
    try:
        from opencut.core.content_calendar import generate_content_calendar

        data = request.get_json(force=True) or {}

        clips = data.get("clips", [])
        if not clips:
            return jsonify({"error": "clips list is required"}), 400

        platforms = data.get("platforms")
        start_date = data.get("start_date")
        weeks = safe_int(data.get("weeks", 4), 4, min_val=1, max_val=52)
        output_format = data.get("output_format", "both")
        output_dir = data.get("output_dir", "")
        if output_dir:
            output_dir = validate_path(output_dir)

        result = generate_content_calendar(
            clips=clips,
            platforms=platforms,
            start_date=start_date,
            weeks=weeks,
            output_format=output_format,
            output_dir=output_dir,
        )

        return jsonify({
            "total_posts": result.total_posts,
            "date_range": result.date_range,
            "platform_breakdown": result.platform_breakdown,
            "csv_path": result.csv_path,
            "ics_path": result.ics_path,
            "scheduled_posts": [
                {
                    "date": p.date,
                    "time": p.time,
                    "platform": p.platform,
                    "title": p.title,
                    "status": p.status,
                }
                for p in result.scheduled_posts[:50]
            ],
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return safe_error(exc, "content_calendar")


# ---------------------------------------------------------------------------
# Social Captions
# ---------------------------------------------------------------------------
@multiview_repurpose_bp.route("/repurpose/social-captions", methods=["POST"])
@require_csrf
@async_job("social_captions", filepath_required=False)
def repurpose_social_captions(job_id, filepath, data):
    """Generate platform-specific captions from transcript."""
    from opencut.core.social_captions import generate_platform_caption
    from opencut.jobs import _update_job

    transcript = data.get("transcript", "").strip()
    if not transcript:
        raise ValueError("transcript text is required")

    platform = data.get("platform", "twitter")
    tone = data.get("tone", "professional")
    custom_hashtags = data.get("custom_hashtags", [])

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_platform_caption(
        transcript=transcript,
        platform=platform,
        tone=tone,
        custom_hashtags=custom_hashtags,
        on_progress=_progress,
    )

    return {
        "platform": result.platform,
        "caption": result.caption,
        "hashtags": result.hashtags,
        "char_count": result.char_count,
        "tone": result.tone,
    }
