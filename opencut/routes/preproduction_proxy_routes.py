"""
OpenCut Pre-Production & Proxy Management Routes (Sections 59-60)

Blueprint: preproduction_proxy_bp

Endpoints:
- POST /storyboard/from-script       — Generate storyboard from script text
- POST /storyboard/shot-list         — Generate shot list from screenplay
- POST /storyboard/mood-board        — Generate mood board from footage
- POST /rough-cut/from-script        — Script-to-roughcut assembly
- POST /proxy/auto-ingest            — Auto proxy generation
- POST /proxy/swap-check             — Verify proxy/original swap status
- POST /proxy/relink                 — Media relinking assistant
- POST /proxy/detect-duplicates      — Find near-duplicate media
"""

import logging
import os
import tempfile

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

preproduction_proxy_bp = Blueprint("preproduction_proxy", __name__)


# ---------------------------------------------------------------------------
# 59.1 — Storyboard from Script
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/storyboard/from-script", methods=["POST"])
@require_csrf
@async_job("storyboard_from_script", filepath_required=False)
def storyboard_from_script(job_id, filepath, data):
    """Generate a storyboard from script text with optional AI images."""
    script_text = data.get("script_text", "").strip()
    if not script_text:
        raise ValueError("script_text is required")

    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "opencut_storyboard")

    columns = safe_int(data.get("columns", 3), 3, min_val=1, max_val=6)
    panel_width = safe_int(data.get("panel_width", 640), 640, min_val=200, max_val=1920)
    panel_height = safe_int(data.get("panel_height", 360), 360, min_val=120, max_val=1080)
    export_pdf = safe_bool(data.get("export_pdf", True), True)
    use_sd = safe_bool(data.get("use_stable_diffusion", False), False)
    sd_url = str(data.get("sd_api_url", "http://127.0.0.1:7860"))[:200]

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.ai_storyboard import generate_storyboard_from_script
    result = generate_storyboard_from_script(
        script_text=script_text,
        output_dir=output_dir,
        on_progress=_p,
        use_stable_diffusion=use_sd,
        sd_api_url=sd_url,
        columns=columns,
        panel_width=panel_width,
        panel_height=panel_height,
        export_pdf=export_pdf,
    )

    return {
        "total_shots": result.total_shots,
        "grid_path": result.grid_path,
        "pdf_path": result.pdf_path,
        "panels": len(result.panels),
    }


# ---------------------------------------------------------------------------
# 59.2 — Shot List from Screenplay
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/storyboard/shot-list", methods=["POST"])
@require_csrf
@async_job("shot_list_generate", filepath_required=False)
def shot_list_generate(job_id, filepath, data):
    """Generate a shot list from screenplay text."""
    screenplay_text = data.get("screenplay_text", "").strip()
    if not screenplay_text:
        raise ValueError("screenplay_text is required")

    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "opencut_shot_list")

    is_fountain = safe_bool(data.get("is_fountain", False), False)
    export_csv = safe_bool(data.get("export_csv", True), True)
    export_json = safe_bool(data.get("export_json", True), True)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.shot_list_gen import generate_shot_list
    result = generate_shot_list(
        screenplay_text=screenplay_text,
        output_dir=output_dir,
        is_fountain=is_fountain,
        export_csv=export_csv,
        export_json=export_json,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 59.3 — Mood Board from Footage
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/storyboard/mood-board", methods=["POST"])
@require_csrf
@async_job("mood_board_generate")
def mood_board_generate(job_id, filepath, data):
    """Generate a mood board from video footage."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""
    if not output_dir:
        output_dir = _resolve_output_dir(filepath, "")

    num_keyframes = safe_int(data.get("num_keyframes", 8), 8, min_val=2, max_val=30)
    num_colors = safe_int(data.get("num_colors", 5), 5, min_val=2, max_val=10)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.mood_board import generate_mood_board
    result = generate_mood_board(
        video_path=filepath,
        output_dir=output_dir,
        num_keyframes=num_keyframes,
        num_colors=num_colors,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 59.4 — Script-to-Rough-Cut Assembly
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/rough-cut/from-script", methods=["POST"])
@require_csrf
@async_job("rough_cut_from_script", filepath_required=False)
def rough_cut_from_script(job_id, filepath, data):
    """Assemble a rough cut from script and footage."""
    script_text = data.get("script_text", "").strip()
    if not script_text:
        raise ValueError("script_text is required")

    clip_paths = data.get("clip_paths", [])
    if not isinstance(clip_paths, list):
        raise ValueError("clip_paths must be a list")

    # Validate each clip path
    validated_clips = []
    for cp in clip_paths:
        try:
            validated_clips.append(validate_filepath(str(cp)))
        except ValueError:
            logger.warning("Skipping invalid clip path: %s", cp)

    transcript_map = data.get("transcript_map", {})
    if not isinstance(transcript_map, dict):
        transcript_map = {}

    # Validate transcript paths
    validated_transcripts = {}
    for clip, tp in transcript_map.items():
        try:
            validated_transcripts[str(clip)] = validate_filepath(str(tp))
        except ValueError:
            logger.warning("Skipping invalid transcript path: %s", tp)

    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "opencut_rough_cut")

    output_format = str(data.get("output_format", "xml"))[:10]
    if output_format not in ("xml", "otio"):
        output_format = "xml"

    match_threshold = safe_float(data.get("match_threshold", 0.4), 0.4,
                                 min_val=0.1, max_val=1.0)
    fps = safe_float(data.get("fps", 24.0), 24.0, min_val=1.0, max_val=120.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.script_to_roughcut import assemble_rough_cut
    result = assemble_rough_cut(
        script_text=script_text,
        clip_paths=validated_clips,
        transcript_map=validated_transcripts,
        output_dir=output_dir,
        output_format=output_format,
        match_threshold=match_threshold,
        fps=fps,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 60.1 — Auto Proxy Ingest
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/proxy/auto-ingest", methods=["POST"])
@require_csrf
@async_job("proxy_auto_ingest", filepath_required=False)
def proxy_auto_ingest(job_id, filepath, data):
    """Auto-detect high-res clips in a folder and generate proxies."""
    folder_path = data.get("folder_path", "").strip()
    if not folder_path:
        raise ValueError("folder_path is required")
    try:
        folder_path = validate_path(folder_path)
    except ValueError as e:
        raise ValueError(f"Invalid folder_path: {e}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"folder_path is not a directory: {folder_path}")

    threshold = safe_int(data.get("threshold_resolution", 1920), 1920,
                         min_val=720, max_val=8192)
    preset = str(data.get("proxy_preset", "720p"))[:20]
    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""
    recursive = safe_bool(data.get("recursive", True), True)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.proxy_gen import auto_proxy_ingest
    result = auto_proxy_ingest(
        folder_path=folder_path,
        threshold_resolution=threshold,
        proxy_preset=preset,
        output_dir=output_dir,
        recursive=recursive,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 60.2 — Proxy Swap Check
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/proxy/swap-check", methods=["POST"])
@require_csrf
@async_job("proxy_swap_check", filepath_required=False)
def proxy_swap_check(job_id, filepath, data):
    """Check timeline clips against proxy manifest for swap readiness."""
    clip_paths = data.get("clip_paths", [])
    if not isinstance(clip_paths, list) or not clip_paths:
        raise ValueError("clip_paths must be a non-empty list")

    proxy_dirs = data.get("proxy_dirs", [])
    if not isinstance(proxy_dirs, list):
        proxy_dirs = []

    validated_dirs = []
    for d in proxy_dirs:
        try:
            validated_dirs.append(validate_path(str(d)))
        except ValueError:
            pass

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.proxy_swap import check_proxy_swap
    result = check_proxy_swap(
        clip_paths=clip_paths,
        proxy_dirs=validated_dirs if validated_dirs else None,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 60.3 — Media Relink
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/proxy/relink", methods=["POST"])
@require_csrf
@async_job("media_relink", filepath_required=False)
def media_relink(job_id, filepath, data):
    """Find replacement files for offline/missing media."""
    offline_paths = data.get("offline_paths", [])
    if not isinstance(offline_paths, list) or not offline_paths:
        raise ValueError("offline_paths must be a non-empty list")

    search_dirs = data.get("search_dirs", [])
    if not isinstance(search_dirs, list) or not search_dirs:
        raise ValueError("search_dirs must be a non-empty list")

    validated_dirs = []
    for d in search_dirs:
        try:
            validated_dirs.append(validate_path(str(d)))
        except ValueError:
            logger.warning("Skipping invalid search dir: %s", d)

    if not validated_dirs:
        raise ValueError("No valid search directories provided")

    offline_metadata = data.get("offline_metadata", {})
    if not isinstance(offline_metadata, dict):
        offline_metadata = {}

    recursive = safe_bool(data.get("recursive", True), True)
    fuzzy_threshold = safe_float(data.get("fuzzy_threshold", 0.6), 0.6,
                                 min_val=0.3, max_val=1.0)
    auto_resolve = safe_float(data.get("auto_resolve_threshold", 0.9), 0.9,
                              min_val=0.5, max_val=1.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.media_relink import relink_media
    result = relink_media(
        offline_paths=offline_paths,
        search_dirs=validated_dirs,
        offline_metadata=offline_metadata,
        recursive=recursive,
        fuzzy_threshold=fuzzy_threshold,
        auto_resolve_threshold=auto_resolve,
        on_progress=_p,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# 60.4 — Detect Duplicates
# ---------------------------------------------------------------------------
@preproduction_proxy_bp.route("/proxy/detect-duplicates", methods=["POST"])
@require_csrf
@async_job("detect_duplicates", filepath_required=False)
def detect_duplicates(job_id, filepath, data):
    """Find near-duplicate video files in a folder."""
    folder_path = data.get("folder_path", "").strip()
    if not folder_path:
        raise ValueError("folder_path is required")
    try:
        folder_path = validate_path(folder_path)
    except ValueError as e:
        raise ValueError(f"Invalid folder_path: {e}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"folder_path is not a directory: {folder_path}")

    threshold = safe_float(data.get("threshold", 0.80), 0.80,
                           min_val=0.5, max_val=1.0)
    recursive = safe_bool(data.get("recursive", True), True)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.duplicate_detect import detect_near_duplicates
    result = detect_near_duplicates(
        folder_path=folder_path,
        threshold=threshold,
        recursive=recursive,
        on_progress=_p,
    )

    return {
        "groups": result,
        "total_groups": len(result),
        "total_potential_savings": sum(g.get("potential_savings", 0) for g in result),
    }
