"""
OpenCut Documentary & Event Routes

Routes for selects bin, string-out reels, archival conform,
brand kit, guest compilation, photo montage, and event recap.
"""

import logging

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
)

logger = logging.getLogger("opencut")

documentary_bp = Blueprint("documentary", __name__)


# ===================================================================
# 14.2 — Selects Bin
# ===================================================================

@documentary_bp.route("/selects/rate", methods=["POST"])
@require_csrf
@async_job("selects_rate", filepath_required=True)
def selects_rate(job_id, filepath, data):
    """Rate a clip 1-5 stars in the selects bin."""
    from opencut.core.selects_bin import rate_clip

    rating = safe_int(data.get("rating", 0), 0, min_val=0, max_val=5)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = rate_clip(
        clip_path=filepath,
        rating=rating,
        on_progress=_progress,
    )
    return result


@documentary_bp.route("/selects/tag", methods=["POST"])
@require_csrf
@async_job("selects_tag", filepath_required=True)
def selects_tag(job_id, filepath, data):
    """Apply tags to a clip in the selects bin."""
    from opencut.core.selects_bin import tag_clip

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    mode = data.get("mode", "set").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = tag_clip(
        clip_path=filepath,
        tags=tags,
        mode=mode,
        on_progress=_progress,
    )
    return result


@documentary_bp.route("/selects/search", methods=["POST"])
@require_csrf
@async_job("selects_search", filepath_required=False)
def selects_search(job_id, filepath, data):
    """Search the selects bin by rating, tags, and text."""
    from opencut.core.selects_bin import search_selects

    filters = {
        "min_rating": data.get("min_rating"),
        "max_rating": data.get("max_rating"),
        "tags": data.get("tags", []),
        "any_tags": data.get("any_tags", []),
        "search": data.get("search", ""),
        "limit": safe_int(data.get("limit", 100), 100, min_val=1, max_val=1000),
        "offset": safe_int(data.get("offset", 0), 0, min_val=0),
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = search_selects(filters=filters, on_progress=_progress)

    return {
        "clips": [
            {
                "clip_path": c.clip_path,
                "rating": c.rating,
                "tags": c.tags,
                "notes": c.notes,
                "duration": c.duration,
            }
            for c in result.clips
        ],
        "total": result.total,
        "filters": result.filters_applied,
    }


@documentary_bp.route("/selects/metadata", methods=["POST"])
@require_csrf
@async_job("selects_metadata", filepath_required=True)
def selects_metadata(job_id, filepath, data):
    """Get metadata for a clip in the selects bin."""
    from opencut.core.selects_bin import get_clip_metadata

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = get_clip_metadata(clip_path=filepath, on_progress=_progress)
    return result


@documentary_bp.route("/selects/export", methods=["POST"])
@require_csrf
@async_job("selects_export", filepath_required=False)
def selects_export(job_id, filepath, data):
    """Export selects matching filters to JSON or CSV."""
    from opencut.core.selects_bin import export_selects

    filters = data.get("filters", {})
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)
    fmt = data.get("format", "json").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_selects(
        filters=filters,
        output_path_str=output,
        format=fmt,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 14.3 — String-Out Reel
# ===================================================================

@documentary_bp.route("/stringout/generate", methods=["POST"])
@require_csrf
@async_job("stringout_generate", filepath_required=False)
def stringout_generate(job_id, filepath, data):
    """Generate a string-out reel from selects or clip list."""
    from opencut.core.stringout_reel import generate_stringout

    clip_paths = data.get("clip_paths", [])
    filter_criteria = data.get("filter_criteria", None)
    order = data.get("order", "rating").strip()
    gap_seconds = safe_float(data.get("gap_seconds", 0), 0, min_val=0, max_val=10)
    target_res = data.get("target_resolution", None)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    # Validate clip paths
    if clip_paths:
        clip_paths = [validate_filepath(p) for p in clip_paths if p.strip()]

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_stringout(
        filter_criteria=filter_criteria,
        output_path_str=output,
        clip_paths=clip_paths if clip_paths else None,
        order=order,
        gap_seconds=gap_seconds,
        target_resolution=target_res,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "duration": result.duration,
        "clip_count": result.clip_count,
        "chapters": [
            {"title": c.title, "start": c.start_time, "end": c.end_time}
            for c in result.chapters
        ],
        "skipped": result.skipped,
    }


@documentary_bp.route("/stringout/chapters", methods=["POST"])
@require_csrf
@async_job("stringout_chapters", filepath_required=True)
def stringout_chapters(job_id, filepath, data):
    """Add chapter markers to an existing video."""
    from opencut.core.stringout_reel import add_chapter_markers

    chapters = data.get("chapters", [])
    if not chapters:
        raise ValueError("No chapters provided")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = add_chapter_markers(
        clips=chapters,
        output_path_str=filepath,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 14.4 — Archival Conform
# ===================================================================

@documentary_bp.route("/conform/analyze", methods=["POST"])
@require_csrf
@async_job("conform_analyze", filepath_required=False)
def conform_analyze(job_id, filepath, data):
    """Analyze clips for conformance against target settings."""
    from opencut.core.archival_conform import analyze_conformance

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("No file paths provided")
    file_paths = [validate_filepath(p) for p in file_paths if p.strip()]

    target = data.get("target_settings", {})
    if not target:
        target = {"fps": 30.0, "width": 1920, "height": 1080,
                  "pix_fmt": "yuv420p", "color_space": "bt709"}

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = analyze_conformance(
        file_paths=file_paths,
        target_settings=target,
        on_progress=_progress,
    )

    return {
        "issues": [
            {
                "file_path": i.file_path,
                "issue_type": i.issue_type,
                "current_value": i.current_value,
                "target_value": i.target_value,
                "severity": i.severity,
            }
            for i in result.issues
        ],
        "files_analyzed": result.files_analyzed,
        "files_with_issues": result.files_with_issues,
        "all_conformant": result.all_conformant,
    }


@documentary_bp.route("/conform/clip", methods=["POST"])
@require_csrf
@async_job("conform_clip", filepath_required=True)
def conform_clip_route(job_id, filepath, data):
    """Conform a single clip to target settings."""
    from opencut.core.archival_conform import conform_clip

    target = data.get("target_settings", {})
    if not target:
        target = {"fps": 30.0, "width": 1920, "height": 1080,
                  "pix_fmt": "yuv420p", "color_space": "bt709"}

    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)
    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "conformed", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = conform_clip(
        video_path=filepath,
        target_settings=target,
        output_path_str=output,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "input_path": result.input_path,
        "changes_applied": result.changes_applied,
        "duration": result.duration,
    }


@documentary_bp.route("/conform/batch", methods=["POST"])
@require_csrf
@async_job("conform_batch", filepath_required=False)
def conform_batch_route(job_id, filepath, data):
    """Batch-conform multiple clips to target settings."""
    from opencut.core.archival_conform import batch_conform

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("No file paths provided")
    file_paths = [validate_filepath(p) for p in file_paths if p.strip()]

    target = data.get("target_settings", {})
    if not target:
        target = {"fps": 30.0, "width": 1920, "height": 1080}
    output_dir = data.get("output_dir", None)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_conform(
        file_paths=file_paths,
        target_settings=target,
        output_dir=output_dir,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 15.1 — Brand Kit
# ===================================================================

@documentary_bp.route("/brand/load", methods=["POST"])
@require_csrf
@async_job("brand_load", filepath_required=False)
def brand_load(job_id, filepath, data):
    """Load a brand kit from a JSON config file."""
    from opencut.core.brand_kit import load_brand_kit

    config_path = data.get("config_path", "").strip()
    if not config_path:
        raise ValueError("No config_path provided")
    config_path = validate_filepath(config_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    kit = load_brand_kit(config_path=config_path, on_progress=_progress)

    return {
        "name": kit.name,
        "primary_color": kit.primary_color,
        "secondary_color": kit.secondary_color,
        "accent_color": kit.accent_color,
        "font_family": kit.font_family,
        "logo_path": kit.logo_path,
        "logo_position": kit.logo_position,
        "status": "loaded",
    }


@documentary_bp.route("/brand/check", methods=["POST"])
@require_csrf
@async_job("brand_check", filepath_required=True)
def brand_check(job_id, filepath, data):
    """Check video compliance against a brand kit."""
    from opencut.core.brand_kit import BrandKit, check_brand_compliance, load_brand_kit

    config_path = data.get("config_path", "").strip()
    if config_path:
        config_path = validate_filepath(config_path)
        kit = load_brand_kit(config_path)
    else:
        # Build BrandKit from inline data
        kit = BrandKit(
            name=data.get("brand_name", "Custom"),
            primary_color=data.get("primary_color", "#FFFFFF"),
            logo_path=data.get("logo_path", ""),
            watermark_text=data.get("watermark_text", ""),
        )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    report = check_brand_compliance(
        video_path=filepath,
        brand_kit=kit,
        on_progress=_progress,
    )

    return {
        "compliant": report.compliant,
        "score": report.score,
        "brand_kit": report.brand_kit_name,
        "issues": [
            {
                "type": i.issue_type,
                "description": i.description,
                "severity": i.severity,
                "auto_fixable": i.auto_fixable,
            }
            for i in report.issues
        ],
    }


@documentary_bp.route("/brand/auto-correct", methods=["POST"])
@require_csrf
@async_job("brand_correct", filepath_required=True)
def brand_auto_correct(job_id, filepath, data):
    """Auto-correct brand compliance issues."""
    from opencut.core.brand_kit import BrandKit, auto_correct_brand, load_brand_kit

    config_path = data.get("config_path", "").strip()
    if config_path:
        config_path = validate_filepath(config_path)
        kit = load_brand_kit(config_path)
    else:
        kit = BrandKit(
            name=data.get("brand_name", "Custom"),
            primary_color=data.get("primary_color", "#FFFFFF"),
            logo_path=data.get("logo_path", ""),
            logo_position=data.get("logo_position", "top_right"),
            watermark_text=data.get("watermark_text", ""),
        )

    add_logo = safe_bool(data.get("add_logo", True), True)
    add_watermark = safe_bool(data.get("add_watermark", True), True)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "branded", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_correct_brand(
        video_path=filepath,
        brand_kit=kit,
        output_path_str=output,
        add_logo=add_logo,
        add_watermark=add_watermark,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 48.2 — Guest Message Compilation
# ===================================================================

@documentary_bp.route("/guest/compile", methods=["POST"])
@require_csrf
@async_job("guest_compile", filepath_required=False)
def guest_compile(job_id, filepath, data):
    """Compile guest message videos from a folder."""
    from opencut.core.guest_compilation import NameCardStyle, compile_guest_messages

    folder_path = data.get("folder_path", "").strip()
    if not folder_path:
        raise ValueError("No folder_path provided")
    folder_path = validate_filepath(folder_path)

    trim_silence = safe_bool(data.get("trim_silence", True), True)
    normalize_audio = safe_bool(data.get("normalize_audio", True), True)
    add_name_cards = safe_bool(data.get("add_name_cards", True), True)
    transition = data.get("transition", "crossfade").strip()
    transition_dur = safe_float(data.get("transition_duration", 0.5), 0.5,
                                min_val=0, max_val=3)
    target_w = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    target_h = safe_int(data.get("height", 1080), 1080, min_val=360, max_val=4320)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    style = None
    style_data = data.get("name_style", None)
    if style_data:
        style = NameCardStyle(
            font_size=safe_int(style_data.get("font_size", 42), 42, min_val=12, max_val=200),
            font_color=style_data.get("font_color", "white"),
            bg_color=style_data.get("bg_color", "black@0.6"),
            position=style_data.get("position", "bottom_left"),
            display_duration=safe_float(style_data.get("display_duration", 4.0), 4.0),
        )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = compile_guest_messages(
        folder_path=folder_path,
        output_path_str=output,
        trim_silence=trim_silence,
        normalize_audio=normalize_audio,
        add_name_cards=add_name_cards,
        name_style=style,
        transition=transition,
        transition_duration=transition_dur,
        target_width=target_w,
        target_height=target_h,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "total_duration": result.total_duration,
        "message_count": result.message_count,
        "messages": [
            {
                "guest_name": m.guest_name,
                "original_duration": m.original_duration,
                "trimmed_duration": m.trimmed_duration,
                "silence_removed": m.silence_removed,
            }
            for m in result.messages
        ],
        "skipped": result.skipped,
    }


@documentary_bp.route("/guest/process-single", methods=["POST"])
@require_csrf
@async_job("guest_process", filepath_required=True)
def guest_process_single(job_id, filepath, data):
    """Process a single guest message video."""
    from opencut.core.guest_compilation import process_single_message

    guest_name = data.get("guest_name", None)
    trim_silence = safe_bool(data.get("trim_silence", True), True)
    normalize_audio = safe_bool(data.get("normalize_audio", True), True)
    add_name_card = safe_bool(data.get("add_name_card", True), True)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "processed", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = process_single_message(
        video_path=filepath,
        output_path_str=output,
        guest_name=guest_name,
        trim_silence=trim_silence,
        normalize_audio=normalize_audio,
        add_name_card=add_name_card,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "guest_name": result.guest_name,
        "original_duration": result.original_duration,
        "trimmed_duration": result.trimmed_duration,
        "silence_removed": result.silence_removed,
        "audio_normalized": result.audio_normalized,
    }


@documentary_bp.route("/guest/name-card", methods=["POST"])
@require_csrf
@async_job("guest_namecard", filepath_required=False)
def guest_name_card(job_id, filepath, data):
    """Generate a standalone name lower-third card."""
    from opencut.core.guest_compilation import NameCardStyle, generate_name_card

    name = data.get("name", "").strip()
    if not name:
        raise ValueError("No name provided")

    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=360, max_val=4320)
    duration = safe_float(data.get("duration", 4.0), 4.0, min_val=1, max_val=30)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    style_data = data.get("style", {})
    style = NameCardStyle(
        font_size=safe_int(style_data.get("font_size", 42), 42),
        font_color=style_data.get("font_color", "white"),
        bg_color=style_data.get("bg_color", "black@0.6"),
        position=style_data.get("position", "bottom_left"),
    ) if style_data else None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_name_card(
        name=name,
        style=style,
        width=width,
        height=height,
        output_path_str=output,
        duration=duration,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 48.3 — Photo+Video Montage
# ===================================================================

@documentary_bp.route("/montage/create", methods=["POST"])
@require_csrf
@async_job("montage_create", filepath_required=False)
def montage_create(job_id, filepath, data):
    """Create a photo+video montage with optional music sync."""
    from opencut.core.photo_montage import create_montage

    media_paths = data.get("media_paths", [])
    if not media_paths:
        raise ValueError("No media_paths provided")
    media_paths = [validate_filepath(p) for p in media_paths if p.strip()]

    music_path = data.get("music_path", None)
    if music_path:
        music_path = validate_filepath(music_path.strip())

    image_duration = safe_float(data.get("image_duration", 5.0), 5.0,
                                min_val=1, max_val=30)
    transition = data.get("transition", "crossfade").strip()
    transition_dur = safe_float(data.get("transition_duration", 0.5), 0.5,
                                min_val=0, max_val=3)
    ken_burns = data.get("ken_burns_effect", "random").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=360, max_val=4320)
    sync_music = safe_bool(data.get("sync_to_music", True), True)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_montage(
        media_paths=media_paths,
        music_path=music_path,
        output_path_str=output,
        image_duration=image_duration,
        transition=transition,
        transition_duration=transition_dur,
        ken_burns_effect=ken_burns,
        width=width,
        height=height,
        sync_to_music=sync_music,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "total_duration": result.total_duration,
        "segment_count": result.segment_count,
        "image_count": result.image_count,
        "video_count": result.video_count,
        "music_synced": result.music_synced,
    }


@documentary_bp.route("/montage/ken-burns", methods=["POST"])
@require_csrf
@async_job("montage_kenburns", filepath_required=True)
def montage_ken_burns(job_id, filepath, data):
    """Apply Ken Burns effect to a single image."""
    from opencut.core.photo_montage import apply_ken_burns

    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=1, max_val=60)
    focus = data.get("focus_point", None)
    if focus and isinstance(focus, list) and len(focus) == 2:
        focus = (float(focus[0]), float(focus[1]))
    else:
        focus = None

    effect = data.get("effect", "zoom_in").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=640, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=360, max_val=4320)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "kenburns", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_ken_burns(
        image_path=filepath,
        duration=duration,
        focus_point=focus,
        output_path_str=output,
        width=width,
        height=height,
        effect=effect,
        on_progress=_progress,
    )
    return result


# ===================================================================
# 48.4 — Event Recap Reel
# ===================================================================

@documentary_bp.route("/recap/score", methods=["POST"])
@require_csrf
@async_job("recap_score", filepath_required=True)
def recap_score(job_id, filepath, data):
    """Score event video segments for highlight selection."""
    from opencut.core.event_recap import RecapConfig, score_event_segments

    config_data = data.get("config", {})
    config = RecapConfig(
        segment_analysis_interval=safe_float(
            config_data.get("interval", 2.0), 2.0, min_val=0.5, max_val=10),
        audio_weight=safe_float(config_data.get("audio_weight", 0.35), 0.35),
        motion_weight=safe_float(config_data.get("motion_weight", 0.30), 0.30),
        variety_weight=safe_float(config_data.get("variety_weight", 0.20), 0.20),
        pacing_weight=safe_float(config_data.get("pacing_weight", 0.15), 0.15),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    segments = score_event_segments(
        video_path=filepath,
        config=config,
        on_progress=_progress,
    )

    return {
        "segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "audio_score": s.audio_score,
                "motion_score": s.motion_score,
                "combined_score": s.combined_score,
            }
            for s in segments
        ],
        "segment_count": len(segments),
    }


@documentary_bp.route("/recap/generate", methods=["POST"])
@require_csrf
@async_job("recap_generate", filepath_required=True)
def recap_generate(job_id, filepath, data):
    """Generate an event recap reel from multi-hour footage."""
    from opencut.core.event_recap import RecapConfig, generate_recap

    target_duration = safe_float(data.get("target_duration", 180), 180,
                                 min_val=10, max_val=1800)
    config_data = data.get("config", {})
    config = RecapConfig(
        target_duration=target_duration,
        min_segment_length=safe_float(
            config_data.get("min_segment_length", 3.0), 3.0, min_val=1, max_val=30),
        max_segment_length=safe_float(
            config_data.get("max_segment_length", 30.0), 30.0, min_val=5, max_val=120),
        transition=config_data.get("transition", "crossfade"),
        fade_in=safe_float(config_data.get("fade_in", 1.0), 1.0, min_val=0, max_val=5),
        fade_out=safe_float(config_data.get("fade_out", 1.0), 1.0, min_val=0, max_val=5),
        include_audio=safe_bool(config_data.get("include_audio", True), True),
    )

    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)
    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "recap", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_recap(
        video_path=filepath,
        target_duration=target_duration,
        output_path_str=output,
        config=config,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "total_duration": result.total_duration,
        "segments_selected": result.segments_selected,
        "source_duration": result.source_duration,
        "compression_ratio": result.compression_ratio,
        "segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "audio_score": s.audio_score,
                "motion_score": s.motion_score,
                "combined_score": s.combined_score,
            }
            for s in result.segments
        ],
    }
