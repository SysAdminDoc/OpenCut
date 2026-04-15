"""
OpenCut Advanced Timeline Automation Routes (Category 74)

Endpoints for AI rough cut, auto-mix, smart trim, batch timeline
operations, and template-based assembly.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

timeline_auto_bp = Blueprint("timeline_auto", __name__)


# ---------------------------------------------------------------------------
# POST /api/timeline/rough-cut
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/rough-cut", methods=["POST"])
@require_csrf
@async_job("rough_cut", filepath_required=False)
def rough_cut(job_id, filepath, data):
    """Generate rough cut from script text and media clips."""
    from opencut.core.auto_rough_cut import ASSEMBLY_MODES, assemble_rough_cut

    script_text = data.get("script", "")
    media_paths = data.get("media_paths", [])
    mode = data.get("mode", "strict")
    target_duration = safe_float(data.get("target_duration"), 120.0, 1.0, 7200.0)
    fps = safe_float(data.get("fps"), 30.0, 1.0, 120.0)

    if not media_paths or not isinstance(media_paths, list):
        raise ValueError("'media_paths' must be a non-empty list of file paths")

    # Validate each media path
    validated_paths = []
    for p in media_paths:
        if isinstance(p, str) and p.strip():
            validated_paths.append(validate_filepath(p.strip()))
    if not validated_paths:
        raise ValueError("No valid media paths provided")

    if mode not in ASSEMBLY_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(ASSEMBLY_MODES)}")

    # LLM config (optional)
    llm_config = None
    llm_data = data.get("llm_config")
    if llm_data and isinstance(llm_data, dict):
        try:
            from opencut.core.llm import LLMConfig
            llm_config = LLMConfig(
                provider=llm_data.get("provider", "ollama"),
                model=llm_data.get("model", "llama3.2"),
                api_key=llm_data.get("api_key", ""),
                base_url=llm_data.get("base_url", "http://localhost:11434"),
                temperature=safe_float(llm_data.get("temperature"), 0.3, 0.0, 2.0),
                max_tokens=safe_int(llm_data.get("max_tokens"), 2000, 100, 8000),
            )
        except ImportError:
            logger.debug("LLM module not available, proceeding without LLM")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Assembling rough cut... {pct}%")

    result = assemble_rough_cut(
        script_text=script_text,
        media_paths=validated_paths,
        mode=mode,
        target_duration=target_duration,
        llm_config=llm_config,
        fps=fps,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/auto-mix
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/auto-mix", methods=["POST"])
@require_csrf
@async_job("auto_mix", filepath_required=False)
def auto_mix(job_id, filepath, data):
    """Auto-mix multiple audio tracks with ducking and level matching."""
    from opencut.core.auto_mix import DUCKING_PROFILES, auto_mix as run_auto_mix

    tracks = data.get("tracks", [])
    profile = data.get("profile", "podcast")
    mix_down = bool(data.get("mix_down", False))
    output_file = data.get("output_file", "")

    if not tracks or not isinstance(tracks, list):
        raise ValueError("'tracks' must be a non-empty list of track objects")

    if profile not in DUCKING_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Use: {', '.join(DUCKING_PROFILES.keys())}"
        )

    # Validate track file paths
    validated_tracks = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        fp = t.get("file_path", "")
        if isinstance(fp, str) and fp.strip():
            t["file_path"] = validate_filepath(fp.strip())
        validated_tracks.append(t)

    if not validated_tracks:
        raise ValueError("No valid tracks provided")

    # Validate output file path if mix_down requested
    if mix_down and output_file:
        out_dir = _resolve_output_dir(validated_tracks[0]["file_path"],
                                      data.get("output_dir", ""))
        if not output_file.startswith(("/", "\\")) and ":" not in output_file:
            import os
            output_file = os.path.join(out_dir, output_file)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Auto-mixing... {pct}%")

    result = run_auto_mix(
        tracks=validated_tracks,
        profile=profile,
        output_file=output_file,
        mix_down=mix_down,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/auto-mix/preview
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/auto-mix/preview", methods=["POST"])
@require_csrf
@async_job("auto_mix_preview", filepath_required=False)
def auto_mix_preview(job_id, filepath, data):
    """Preview mix settings for first N seconds."""
    from opencut.core.auto_mix import DUCKING_PROFILES, preview_mix

    tracks = data.get("tracks", [])
    profile = data.get("profile", "podcast")
    preview_seconds = safe_float(data.get("preview_seconds"), 10.0, 1.0, 60.0)

    if not tracks or not isinstance(tracks, list):
        raise ValueError("'tracks' must be a non-empty list of track objects")

    if profile not in DUCKING_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Use: {', '.join(DUCKING_PROFILES.keys())}"
        )

    # Validate track file paths
    validated_tracks = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        fp = t.get("file_path", "")
        if isinstance(fp, str) and fp.strip():
            t["file_path"] = validate_filepath(fp.strip())
        validated_tracks.append(t)

    if not validated_tracks:
        raise ValueError("No valid tracks provided")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Generating preview... {pct}%")

    result = preview_mix(
        tracks=validated_tracks,
        profile=profile,
        preview_seconds=preview_seconds,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/smart-trim
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/smart-trim", methods=["POST"])
@require_csrf
@async_job("smart_trim")
def smart_trim_route(job_id, filepath, data):
    """Find optimal trim points for a single clip."""
    from opencut.core.smart_trim import TRIM_MODES, smart_trim

    mode = data.get("mode", "tight")
    if mode not in TRIM_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(TRIM_MODES.keys())}")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Analyzing trim points... {pct}%")

    result = smart_trim(
        file_path=filepath,
        mode=mode,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/smart-trim/batch
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/smart-trim/batch", methods=["POST"])
@require_csrf
@async_job("smart_trim_batch", filepath_required=False)
def smart_trim_batch(job_id, filepath, data):
    """Batch smart trim for multiple clips."""
    from opencut.core.smart_trim import TRIM_MODES, batch_smart_trim

    file_paths = data.get("file_paths", [])
    mode = data.get("mode", "tight")

    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("'file_paths' must be a non-empty list")

    if mode not in TRIM_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(TRIM_MODES.keys())}")

    # Validate paths
    validated = []
    for p in file_paths:
        if isinstance(p, str) and p.strip():
            validated.append(validate_filepath(p.strip()))
    if not validated:
        raise ValueError("No valid file paths provided")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Batch trimming... {pct}%")

    result = batch_smart_trim(
        file_paths=validated,
        mode=mode,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/batch-ops
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/batch-ops", methods=["POST"])
@require_csrf
@async_job("batch_ops", filepath_required=False)
def batch_ops(job_id, filepath, data):
    """Execute batch timeline operations (operation pipeline)."""
    from opencut.core.batch_timeline_ops import execute_pipeline

    clips = data.get("clips", [])
    operations = data.get("operations", [])

    if not clips or not isinstance(clips, list):
        raise ValueError("'clips' must be a non-empty list")

    if not operations or not isinstance(operations, list):
        raise ValueError("'operations' must be a non-empty list")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Processing batch operations... {pct}%")

    result = execute_pipeline(
        clips_data=clips,
        operations=operations,
        dry_run=False,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/batch-ops/preview
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/batch-ops/preview", methods=["POST"])
@require_csrf
@async_job("batch_ops_preview", filepath_required=False)
def batch_ops_preview(job_id, filepath, data):
    """Dry-run preview of batch timeline operations."""
    from opencut.core.batch_timeline_ops import execute_pipeline

    clips = data.get("clips", [])
    operations = data.get("operations", [])

    if not clips or not isinstance(clips, list):
        raise ValueError("'clips' must be a non-empty list")

    if not operations or not isinstance(operations, list):
        raise ValueError("'operations' must be a non-empty list")

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Previewing batch operations... {pct}%")

    result = execute_pipeline(
        clips_data=clips,
        operations=operations,
        dry_run=True,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/assemble
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/assemble", methods=["POST"])
@require_csrf
@async_job("template_assemble", filepath_required=False)
def template_assemble(job_id, filepath, data):
    """Assemble video from a template and media map."""
    from opencut.core.template_assembly_adv import (
        BUILTIN_TEMPLATES,
        Template,
        assemble_from_template,
    )

    template_name = data.get("template", "")
    custom_template = data.get("custom_template")
    media_map = data.get("media_map", {})

    if not media_map or not isinstance(media_map, dict):
        raise ValueError("'media_map' must be a non-empty dict mapping slot names to paths")

    # Resolve template
    if custom_template and isinstance(custom_template, dict):
        template = Template.from_dict(custom_template)
    elif template_name in BUILTIN_TEMPLATES:
        template = BUILTIN_TEMPLATES[template_name]()
    else:
        available = ", ".join(BUILTIN_TEMPLATES.keys())
        raise ValueError(
            f"Unknown template '{template_name}'. Available: {available}. "
            "Or provide 'custom_template' dict."
        )

    # Validate media paths in map
    validated_map = {}
    for slot_name, path in media_map.items():
        if isinstance(path, str) and path.strip():
            # Text slots don't need file validation
            slot = next((s for s in template.slots if s.name == slot_name), None)
            if slot and slot.slot_type == "text":
                validated_map[slot_name] = path
            elif os.path.isfile(path.strip()):
                validated_map[slot_name] = path.strip()
            else:
                logger.warning("Media file not found for slot '%s': %s", slot_name, path)
                validated_map[slot_name] = path.strip()
        else:
            validated_map[slot_name] = path

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Assembling from template... {pct}%")

    result = assemble_from_template(
        template=template,
        media_map=validated_map,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# GET /api/timeline/templates
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/templates", methods=["GET"])
def list_templates_route():
    """List all available assembly templates."""
    from opencut.core.template_assembly_adv import list_templates
    return jsonify({"templates": list_templates()})


# ---------------------------------------------------------------------------
# POST /api/timeline/templates/validate
# ---------------------------------------------------------------------------
@timeline_auto_bp.route("/api/timeline/templates/validate", methods=["POST"])
@require_csrf
def validate_template_route():
    """Validate a custom template definition."""
    from opencut.core.template_assembly_adv import validate_template

    data = request.get_json(force=True) or {}
    template_data = data.get("template", data)

    if not template_data or not isinstance(template_data, dict):
        return jsonify({"error": "No template data provided"}), 400

    result = validate_template(template_data)
    return jsonify(result.to_dict())
