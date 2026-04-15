"""
OpenCut Editing Workflow Routes

Routes for script/storyboard, multi-platform publish, paper edit,
template assembly, timeline copilot, programmatic video,
multi-language subtitles, speaker layout, and ceremony auto-edit.
"""

import logging
import os

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
)

logger = logging.getLogger("opencut")

editing_wf_bp = Blueprint("editing_wf", __name__)


# ---------------------------------------------------------------------------
# Helper: validate list of file paths
# ---------------------------------------------------------------------------
def _validate_file_list(data, key="file_paths", required=True, max_files=50):
    """Extract and validate a list of file paths from request data."""
    paths = data.get(key, [])
    if not isinstance(paths, list):
        raise ValueError(f"{key} must be a list")
    if required and not paths:
        raise ValueError(f"{key} must be a non-empty list")
    if len(paths) > max_files:
        raise ValueError(f"Too many files: {len(paths)} (max {max_files})")
    validated = []
    for p in paths:
        if isinstance(p, str) and p.strip():
            validated.append(validate_filepath(p.strip()))
    return validated


# ===================================================================
# 4.7 — Script/Storyboard Integration
# ===================================================================

@editing_wf_bp.route("/script/parse", methods=["POST"])
@require_csrf
@async_job("script_parse", filepath_required=True, filepath_param="script_path")
def script_parse(job_id, filepath, data):
    """Parse a script file (text or PDF) into structured elements."""
    from opencut.core.script_storyboard import parse_script

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    lines = parse_script(filepath, on_progress=_progress)

    return {
        "line_count": len(lines),
        "lines": [
            {
                "index": ln.index,
                "type": ln.line_type,
                "text": ln.text[:200],
                "character": ln.character,
                "scene": ln.scene,
            }
            for ln in lines[:500]
        ],
    }


@editing_wf_bp.route("/script/align", methods=["POST"])
@require_csrf
@async_job("script_align", filepath_required=True, filepath_param="script_path")
def script_align(job_id, filepath, data):
    """Align script to transcript and find missing coverage."""
    from opencut.core.script_storyboard import (
        align_script_to_transcript,
        find_missing_coverage,
        parse_script,
    )

    transcript = data.get("transcript", [])
    if not isinstance(transcript, list) or not transcript:
        raise ValueError("transcript must be a non-empty list of segments")

    threshold = safe_float(data.get("threshold", 0.3), 0.3, min_val=0.0, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Parsing script...")
    script = parse_script(filepath)

    _progress(30, "Aligning to transcript...")
    alignment = align_script_to_transcript(
        script, transcript, threshold=threshold, on_progress=_progress,
    )

    _progress(80, "Finding missing coverage...")
    missing = find_missing_coverage(alignment)

    covered = sum(1 for a in alignment if a.covered)

    return {
        "total_lines": len(script),
        "covered": covered,
        "missing": len(missing),
        "alignment": [
            {
                "index": a.script_line.index,
                "type": a.script_line.line_type,
                "text": a.script_line.text[:120],
                "covered": a.covered,
                "confidence": a.confidence,
                "start": a.transcript_start,
                "end": a.transcript_end,
            }
            for a in alignment[:300]
        ],
        "missing_lines": [
            {
                "index": m.script_line.index,
                "text": m.script_line.text[:120],
                "reason": m.reason,
            }
            for m in missing[:100]
        ],
    }


@editing_wf_bp.route("/script/broll", methods=["POST"])
@require_csrf
@async_job("script_broll", filepath_required=True, filepath_param="script_path")
def script_broll(job_id, filepath, data):
    """Suggest B-roll clips based on script content."""
    from opencut.core.script_storyboard import parse_script, suggest_broll_from_script

    media_library = data.get("media_library", [])
    if not isinstance(media_library, list):
        media_library = []

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Parsing script...")
    script = parse_script(filepath)

    _progress(30, "Generating B-roll suggestions...")
    suggestions = suggest_broll_from_script(
        script, media_library=media_library, on_progress=_progress,
    )

    return {
        "suggestion_count": len(suggestions),
        "suggestions": [
            {
                "line_index": s.script_line.index,
                "line_text": s.script_line.text[:100],
                "suggestion": s.suggestion,
                "keywords": s.keywords,
            }
            for s in suggestions[:200]
        ],
    }


# ===================================================================
# 5.1 — Multi-Platform Batch Publish
# ===================================================================

@editing_wf_bp.route("/publish/queue", methods=["POST"])
@require_csrf
@async_job("publish_queue", filepath_required=True, filepath_param="video_path")
def publish_queue(job_id, filepath, data):
    """Create a publish queue for multiple platforms."""
    from opencut.core.multi_publish import PublishConfig, create_publish_queue

    platforms = data.get("platforms", [])
    if not isinstance(platforms, list) or not platforms:
        raise ValueError("platforms must be a non-empty list")

    raw_config = data.get("config", {})
    config = {}
    if isinstance(raw_config, dict):
        for plat, conf in raw_config.items():
            if isinstance(conf, dict):
                config[plat] = PublishConfig(
                    platform=plat,
                    title=conf.get("title", ""),
                    description=conf.get("description", ""),
                    tags=conf.get("tags", []),
                    privacy=conf.get("privacy", "private"),
                )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    queue = create_publish_queue(filepath, platforms, config=config, on_progress=_progress)

    return {
        "queue_size": queue.total,
        "items": [
            {"platform": item.platform, "status": item.status}
            for item in queue.items
        ],
    }


@editing_wf_bp.route("/publish/export", methods=["POST"])
@require_csrf
@async_job("publish_export", filepath_required=True, filepath_param="video_path")
def publish_export(job_id, filepath, data):
    """Export a video formatted for a specific platform."""
    from opencut.core.multi_publish import export_for_platform

    platform = data.get("platform", "").strip().lower()
    if not platform:
        raise ValueError("platform is required")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    out_path = export_for_platform(
        filepath, platform, output_dir=output_dir, on_progress=_progress,
    )

    return {"output_path": out_path, "platform": platform}


@editing_wf_bp.route("/publish/upload", methods=["POST"])
@require_csrf
@async_job("publish_upload", filepath_required=True, filepath_param="video_path")
def publish_upload(job_id, filepath, data):
    """Publish/upload a video to a platform."""
    from opencut.core.multi_publish import PublishConfig, publish_to_platform

    platform = data.get("platform", "").strip().lower()
    if not platform:
        raise ValueError("platform is required")

    credentials = data.get("credentials", {})
    config_data = data.get("config", {})
    config = PublishConfig(
        platform=platform,
        title=config_data.get("title", ""),
        description=config_data.get("description", ""),
        tags=config_data.get("tags", []),
        privacy=config_data.get("privacy", "private"),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = publish_to_platform(
        filepath, platform, credentials=credentials,
        config=config, on_progress=_progress,
    )

    return result


# ===================================================================
# 14.1 — Paper Edit from Transcript
# ===================================================================

@editing_wf_bp.route("/paper-edit/create", methods=["POST"])
@require_csrf
@async_job("paper_edit_create", filepath_required=False)
def paper_edit_create(job_id, filepath, data):
    """Create a paper edit from transcript selections."""
    from opencut.core.paper_edit import create_paper_edit

    transcript = data.get("transcript", [])
    selections = data.get("selections", [])
    if not isinstance(selections, list) or not selections:
        raise ValueError("selections must be a non-empty list")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    paper_edit = create_paper_edit(
        transcript, selections, on_progress=_progress,
    )

    return {
        "selection_count": len(paper_edit.selections),
        "total_duration": paper_edit.total_duration,
        "selections": [
            {
                "order": s.order,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "text": s.text[:120],
                "label": s.label,
                "speaker": s.speaker,
            }
            for s in paper_edit.selections
        ],
    }


@editing_wf_bp.route("/paper-edit/assemble", methods=["POST"])
@require_csrf
@async_job("paper_edit_assemble", filepath_required=True, filepath_param="video_path")
def paper_edit_assemble(job_id, filepath, data):
    """Assemble a video from paper edit selections."""
    from opencut.core.paper_edit import PaperEdit, PaperEditSelection, assemble_from_paper_edit

    selections_raw = data.get("selections", [])
    if not isinstance(selections_raw, list) or not selections_raw:
        raise ValueError("selections must be a non-empty list")

    sels = []
    for i, s in enumerate(selections_raw):
        sels.append(PaperEditSelection(
            start=safe_float(s.get("start", 0), 0),
            end=safe_float(s.get("end", 0), 0),
            text=s.get("text", ""),
            label=s.get("label", f"Sel {i + 1}"),
            order=i,
        ))

    paper_edit = PaperEdit(selections=sels)
    paper_edit._recalc()

    output_dir = data.get("output_dir", "")
    out_path = ""
    if output_dir:
        out_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        out_path = _op(filepath, "paper_edit", out_dir)

    gap = safe_float(data.get("gap_seconds", 0), 0, min_val=0, max_val=10)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = assemble_from_paper_edit(
        filepath, paper_edit, output_path_str=out_path,
        gap_seconds=gap, on_progress=_progress,
    )

    return result


@editing_wf_bp.route("/paper-edit/export", methods=["POST"])
@require_csrf
@async_job("paper_edit_export", filepath_required=False)
def paper_edit_export(job_id, filepath, data):
    """Export a paper edit to JSON, TXT, or EDL."""
    from opencut.core.paper_edit import PaperEdit, PaperEditSelection, export_paper_edit

    selections_raw = data.get("selections", [])
    if not isinstance(selections_raw, list) or not selections_raw:
        raise ValueError("selections must be a non-empty list")

    sels = []
    for i, s in enumerate(selections_raw):
        sels.append(PaperEditSelection(
            start=safe_float(s.get("start", 0), 0),
            end=safe_float(s.get("end", 0), 0),
            text=s.get("text", ""),
            label=s.get("label", f"Sel {i + 1}"),
            order=i,
        ))

    paper_edit = PaperEdit(selections=sels)
    paper_edit._recalc()

    fmt = data.get("format", "json").strip().lower()
    output_path_str = data.get("output_path", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_paper_edit(
        paper_edit, format=fmt, output_path_str=output_path_str,
        on_progress=_progress,
    )

    return result


# ===================================================================
# 15.3 — Template-Based Video Assembly
# ===================================================================

@editing_wf_bp.route("/template/list", methods=["POST"])
@require_csrf
@async_job("template_list", filepath_required=False)
def template_list(job_id, filepath, data):
    """List available video templates."""
    from opencut.core.template_assembly import list_templates

    category = data.get("category", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    templates = list_templates(category=category, on_progress=_progress)

    return {"templates": templates, "count": len(templates)}


@editing_wf_bp.route("/template/fill", methods=["POST"])
@require_csrf
@async_job("template_fill", filepath_required=False)
def template_fill(job_id, filepath, data):
    """Fill a template with media assignments."""
    from opencut.core.template_assembly import fill_template, load_template

    template_name = data.get("template_name", "").strip()
    if not template_name:
        raise ValueError("template_name is required")

    media = data.get("media_assignments", {})
    if not isinstance(media, dict):
        raise ValueError("media_assignments must be a dict")

    text_overrides = data.get("text_overrides", {})

    # Validate media paths
    validated_media = {}
    for name, path in media.items():
        validated_media[name] = validate_filepath(path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    template = load_template(template_name, on_progress=_progress)
    filled = fill_template(template, validated_media, text_overrides, on_progress=_progress)

    return {
        "template_name": template_name,
        "assignments": len(filled.assignments),
        "text_overrides": len(filled.text_overrides),
    }


@editing_wf_bp.route("/template/assemble", methods=["POST"])
@require_csrf
@async_job("template_assemble", filepath_required=False)
def template_assemble(job_id, filepath, data):
    """Assemble a video from a filled template."""
    from opencut.core.template_assembly import (
        assemble_from_template,
        fill_template,
        load_template,
    )

    template_name = data.get("template_name", "").strip()
    if not template_name:
        raise ValueError("template_name is required")

    media = data.get("media_assignments", {})
    if not isinstance(media, dict) or not media:
        raise ValueError("media_assignments must be a non-empty dict")

    text_overrides = data.get("text_overrides", {})
    output_path_str = data.get("output_path", "")

    # Validate media paths
    validated_media = {}
    for name, path in media.items():
        validated_media[name] = validate_filepath(path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    template = load_template(template_name)
    filled = fill_template(template, validated_media, text_overrides)
    result = assemble_from_template(filled, output_path_str=output_path_str, on_progress=_progress)

    return result


# ===================================================================
# 21.1 — Multimodal Timeline Copilot
# ===================================================================

@editing_wf_bp.route("/copilot/query", methods=["POST"])
@require_csrf
@async_job("copilot_query", filepath_required=False)
def copilot_query(job_id, filepath, data):
    """Process a natural language timeline query."""
    from opencut.core.timeline_copilot import (
        build_timeline_context,
        process_copilot_query,
    )

    query = data.get("query", "").strip()
    if not query:
        raise ValueError("query is required")

    video_path = data.get("video_path", "").strip()
    transcript = data.get("transcript", [])
    scenes = data.get("scenes", [])

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    context = None
    if video_path:
        try:
            vp = validate_filepath(video_path)
            context = build_timeline_context(
                vp, transcript=transcript, scenes=scenes,
                on_progress=_progress,
            )
        except (ValueError, FileNotFoundError):
            pass

    action = process_copilot_query(query, context=context, on_progress=_progress)

    return {
        "action_type": action.action_type,
        "description": action.description,
        "parameters": action.parameters,
        "confidence": action.confidence,
        "result": action.result,
    }


@editing_wf_bp.route("/copilot/execute", methods=["POST"])
@require_csrf
@async_job("copilot_execute", filepath_required=True, filepath_param="video_path")
def copilot_execute(job_id, filepath, data):
    """Execute a copilot action on a video."""
    from opencut.core.timeline_copilot import (
        CopilotAction,
        execute_copilot_action,
        process_copilot_query,
    )

    query = data.get("query", "").strip()
    action_data = data.get("action", {})

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if action_data and isinstance(action_data, dict):
        action = CopilotAction(
            action_type=action_data.get("action_type", "info"),
            description=action_data.get("description", ""),
            parameters=action_data.get("parameters", {}),
            confidence=safe_float(action_data.get("confidence", 0.5), 0.5),
        )
    elif query:
        action = process_copilot_query(query, on_progress=_progress)
    else:
        raise ValueError("Either query or action is required")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    result = execute_copilot_action(
        action, filepath, output_dir=output_dir, on_progress=_progress,
    )

    return {
        "action_type": action.action_type,
        "executed": action.executed,
        "result": result,
    }


# ===================================================================
# 21.5 — Programmatic Video from Data
# ===================================================================

@editing_wf_bp.route("/data-video/create", methods=["POST"])
@require_csrf
@async_job("data_video_create", filepath_required=False)
def data_video_create(job_id, filepath, data):
    """Generate a single video from data and template."""
    from opencut.core.programmatic_video import DataVideoTemplate, create_data_video

    template_data = data.get("template", {})
    if not isinstance(template_data, dict):
        raise ValueError("template must be a dict")

    data_row = data.get("data_row", {})
    if not isinstance(data_row, dict) or not data_row:
        raise ValueError("data_row must be a non-empty dict")

    output_path_str = data.get("output_path", "")
    if not output_path_str:
        import tempfile
        fd, output_path_str = tempfile.mkstemp(suffix=".mp4", prefix="data_video_")
        os.close(fd)

    template = DataVideoTemplate(
        name=template_data.get("name", "custom"),
        background=template_data.get("background", "black"),
        width=safe_int(template_data.get("width", 1920), 1920, min_val=2, max_val=3840),
        height=safe_int(template_data.get("height", 1080), 1080, min_val=2, max_val=2160),
        duration=safe_float(template_data.get("duration", 10), 10, min_val=0.1, max_val=3600),
        fps=safe_float(template_data.get("fps", 30), 30, min_val=1, max_val=120),
        text_fields=template_data.get("text_fields", []),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_data_video(template, data_row, output_path_str, on_progress=_progress)

    return result


@editing_wf_bp.route("/data-video/batch", methods=["POST"])
@require_csrf
@async_job("data_video_batch", filepath_required=True, filepath_param="csv_path")
def data_video_batch(job_id, filepath, data):
    """Generate batch videos from CSV data and template."""
    from opencut.core.programmatic_video import DataVideoTemplate, batch_data_videos

    template_data = data.get("template", {})
    if not isinstance(template_data, dict):
        raise ValueError("template must be a dict")

    output_dir = data.get("output_dir", "")
    if not output_dir:
        output_dir = _resolve_output_dir(filepath, "")
    else:
        output_dir = _resolve_output_dir(filepath, output_dir)

    filename_field = data.get("filename_field", "")

    template = DataVideoTemplate(
        name=template_data.get("name", "custom"),
        background=template_data.get("background", "black"),
        width=safe_int(template_data.get("width", 1920), 1920, min_val=2, max_val=3840),
        height=safe_int(template_data.get("height", 1080), 1080, min_val=2, max_val=2160),
        duration=safe_float(template_data.get("duration", 10), 10, min_val=0.1, max_val=3600),
        fps=safe_float(template_data.get("fps", 30), 30, min_val=1, max_val=120),
        text_fields=template_data.get("text_fields", []),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_data_videos(
        template, filepath, output_dir,
        filename_field=filename_field, on_progress=_progress,
    )

    return result


# ===================================================================
# 24.2 — Multi-Language Subtitle Editing
# ===================================================================

@editing_wf_bp.route("/multilang/create", methods=["POST"])
@require_csrf
@async_job("multilang_create", filepath_required=False)
def multilang_create(job_id, filepath, data):
    """Create a multi-language subtitle project."""
    from opencut.core.multilang_subtitle import create_multilang_project

    base_timing = data.get("base_timing", [])
    if not isinstance(base_timing, list) or not base_timing:
        raise ValueError("base_timing must be a non-empty list")

    languages = data.get("languages", [])
    if not isinstance(languages, list) or not languages:
        raise ValueError("languages must be a non-empty list")

    source_text = data.get("source_text", [])
    source_language = data.get("source_language", "")
    project_name = data.get("project_name", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    project = create_multilang_project(
        base_timing, languages,
        source_text=source_text,
        source_language=source_language,
        project_name=project_name,
        on_progress=_progress,
    )

    return {
        "project_name": project.project_name,
        "entry_count": project.entry_count(),
        "languages": project.languages,
        "source_language": project.source_language,
    }


@editing_wf_bp.route("/multilang/update", methods=["POST"])
@require_csrf
@async_job("multilang_update", filepath_required=False)
def multilang_update(job_id, filepath, data):
    """Update translations for a language in the project."""
    from opencut.core.multilang_subtitle import (
        create_multilang_project,
        update_language_text,
    )

    # Reconstruct project from data (stateless API)
    project_data = data.get("project", {})
    if not isinstance(project_data, dict):
        raise ValueError("project data is required")

    base_timing = project_data.get("base_timing", [])
    languages = project_data.get("languages", [])
    if not base_timing or not languages:
        raise ValueError("project must contain base_timing and languages")

    project = create_multilang_project(base_timing, languages)

    # Restore existing translations
    for lang, texts in project_data.get("translations", {}).items():
        if lang in project.languages and isinstance(texts, list):
            project.translations[lang] = texts[:len(project.timing)]

    language = data.get("language", "").strip()
    translations = data.get("translations", [])
    if not isinstance(translations, list):
        raise ValueError("translations must be a list")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = update_language_text(
        project, language, translations, on_progress=_progress,
    )

    return result


@editing_wf_bp.route("/multilang/export", methods=["POST"])
@require_csrf
@async_job("multilang_export", filepath_required=False)
def multilang_export(job_id, filepath, data):
    """Export subtitles for a specific language."""
    from opencut.core.multilang_subtitle import (
        create_multilang_project,
        export_language,
    )

    project_data = data.get("project", {})
    if not isinstance(project_data, dict):
        raise ValueError("project data is required")

    base_timing = project_data.get("base_timing", [])
    languages = project_data.get("languages", [])
    if not base_timing or not languages:
        raise ValueError("project must contain base_timing and languages")

    project = create_multilang_project(base_timing, languages)

    for lang, texts in project_data.get("translations", {}).items():
        if lang in project.languages and isinstance(texts, list):
            project.translations[lang] = texts[:len(project.timing)]

    language = data.get("language", "").strip()
    fmt = data.get("format", "srt").strip().lower()
    output_path_str = data.get("output_path", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = export_language(
        project, language, format=fmt,
        output_path=output_path_str, on_progress=_progress,
    )

    return result


@editing_wf_bp.route("/multilang/sync", methods=["POST"])
@require_csrf
@async_job("multilang_sync", filepath_required=False)
def multilang_sync(job_id, filepath, data):
    """Apply a timing change to the shared timing track."""
    from opencut.core.multilang_subtitle import (
        create_multilang_project,
        sync_timing_change,
    )

    project_data = data.get("project", {})
    if not isinstance(project_data, dict):
        raise ValueError("project data is required")

    base_timing = project_data.get("base_timing", [])
    languages = project_data.get("languages", [])
    if not base_timing or not languages:
        raise ValueError("project must contain base_timing and languages")

    project = create_multilang_project(base_timing, languages)

    for lang, texts in project_data.get("translations", {}).items():
        if lang in project.languages and isinstance(texts, list):
            project.translations[lang] = texts[:len(project.timing)]

    timing_update = data.get("timing_update", {})
    if not isinstance(timing_update, dict) or "operation" not in timing_update:
        raise ValueError("timing_update with operation is required")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = sync_timing_change(project, timing_update, on_progress=_progress)

    return result


# ===================================================================
# 40.2 — Multi-Speaker Layout Engine
# ===================================================================

@editing_wf_bp.route("/speaker-layout/create", methods=["POST"])
@require_csrf
@async_job("speaker_layout", filepath_required=False)
def speaker_layout_create(job_id, filepath, data):
    """Create a multi-speaker layout video."""
    from opencut.core.speaker_layout import LayoutConfig, create_speaker_layout

    video_paths = _validate_file_list(data, "video_paths", required=True, max_files=12)
    layout_type = data.get("layout_type", "grid").strip().lower()
    output_path_str = data.get("output_path", "")

    config = LayoutConfig(
        layout_type=layout_type,
        width=safe_int(data.get("width", 1920), 1920, min_val=2, max_val=3840),
        height=safe_int(data.get("height", 1080), 1080, min_val=2, max_val=2160),
        gap=safe_int(data.get("gap", 4), 4, min_val=0, max_val=50),
        spotlight_index=safe_int(data.get("spotlight_index", 0), 0, min_val=0),
        pip_size_pct=safe_float(data.get("pip_size_pct", 25), 25, min_val=5, max_val=50),
        pip_position=data.get("pip_position", "bottom_right"),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_speaker_layout(
        video_paths, layout_type=layout_type,
        config=config, output_path_str=output_path_str,
        on_progress=_progress,
    )

    return result


@editing_wf_bp.route("/speaker-layout/active", methods=["POST"])
@require_csrf
@async_job("speaker_layout_active", filepath_required=False)
def speaker_layout_active(job_id, filepath, data):
    """Apply active-speaker switching based on diarization."""
    from opencut.core.speaker_layout import LayoutConfig, apply_active_speaker

    video_paths = _validate_file_list(data, "video_paths", required=True, max_files=12)
    diarization = data.get("diarization", [])
    if not isinstance(diarization, list) or not diarization:
        raise ValueError("diarization must be a non-empty list")

    speaker_map = data.get("speaker_map", None)
    output_path_str = data.get("output_path", "")

    config = LayoutConfig(
        width=safe_int(data.get("width", 1920), 1920, min_val=2, max_val=3840),
        height=safe_int(data.get("height", 1080), 1080, min_val=2, max_val=2160),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_active_speaker(
        video_paths, diarization,
        output_path_str=output_path_str,
        speaker_map=speaker_map,
        config=config,
        on_progress=_progress,
    )

    return result


# ===================================================================
# 48.1 — Multi-Camera Ceremony Auto-Edit
# ===================================================================

@editing_wf_bp.route("/ceremony/auto-edit", methods=["POST"])
@require_csrf
@async_job("ceremony_autoedit", filepath_required=False)
def ceremony_auto_edit(job_id, filepath, data):
    """Auto-edit multi-camera ceremony footage."""
    from opencut.core.ceremony_autoedit import CeremonyConfig, auto_edit_ceremony

    camera_paths = _validate_file_list(data, "camera_paths", required=True, max_files=20)
    if len(camera_paths) < 2:
        raise ValueError("At least 2 camera files are required")

    output_path_str = data.get("output_path", "")

    config = CeremonyConfig(
        segment_duration=safe_float(data.get("segment_duration", 5), 5, min_val=1, max_val=60),
        min_segment=safe_float(data.get("min_segment", 2), 2, min_val=0.5, max_val=30),
        max_segment=safe_float(data.get("max_segment", 15), 15, min_val=5, max_val=120),
        audio_weight=safe_float(data.get("audio_weight", 0.5), 0.5, min_val=0, max_val=1),
        motion_weight=safe_float(data.get("motion_weight", 0.3), 0.3, min_val=0, max_val=1),
        variety_weight=safe_float(data.get("variety_weight", 0.2), 0.2, min_val=0, max_val=1),
        width=safe_int(data.get("width", 1920), 1920, min_val=2, max_val=3840),
        height=safe_int(data.get("height", 1080), 1080, min_val=2, max_val=2160),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_edit_ceremony(
        camera_paths, output_path_str=output_path_str,
        config=config, on_progress=_progress,
    )

    return result


@editing_wf_bp.route("/ceremony/score", methods=["POST"])
@require_csrf
@async_job("ceremony_score", filepath_required=False)
def ceremony_score(job_id, filepath, data):
    """Score camera angles at a specific timestamp."""
    from opencut.core.ceremony_autoedit import CeremonyConfig, score_camera_angles

    camera_paths = _validate_file_list(data, "camera_paths", required=True, max_files=20)
    timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0)
    segment_duration = safe_float(data.get("segment_duration", 5), 5, min_val=1, max_val=60)
    previous_camera = safe_int(data.get("previous_camera", -1), -1, min_val=-1)

    config = CeremonyConfig(
        audio_weight=safe_float(data.get("audio_weight", 0.5), 0.5, min_val=0, max_val=1),
        motion_weight=safe_float(data.get("motion_weight", 0.3), 0.3, min_val=0, max_val=1),
        variety_weight=safe_float(data.get("variety_weight", 0.2), 0.2, min_val=0, max_val=1),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    scores = score_camera_angles(
        camera_paths, timestamp,
        segment_duration=segment_duration,
        config=config,
        previous_camera=previous_camera,
        on_progress=_progress,
    )

    return {
        "timestamp": timestamp,
        "scores": [
            {
                "camera_index": s.camera_index,
                "audio_energy": s.audio_energy,
                "motion_score": s.motion_score,
                "variety_score": s.variety_score,
                "total_score": s.total_score,
            }
            for s in scores
        ],
    }
