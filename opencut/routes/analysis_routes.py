"""
OpenCut Analysis Routes

Endpoints for video analysis, caption compliance, and subtitle timing:
- Shot type classification
- Caption compliance checking and auto-fix
- Shot-change-aware subtitle timing
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")

analysis_bp = Blueprint("analysis", __name__)


# ---------------------------------------------------------------------------
# POST /analysis/shot-classify
# ---------------------------------------------------------------------------
@analysis_bp.route("/analysis/shot-classify", methods=["POST"])
@require_csrf
@async_job("shot_classify")
def analysis_shot_classify(job_id, filepath, data):
    """Classify all shots in a video by type."""
    from opencut.core.shot_classify import classify_shots

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = classify_shots(filepath, on_progress=_on_progress)

    return {
        "total_shots": result.total_shots,
        "duration": result.duration,
        "type_distribution": result.type_distribution,
        "shots": [
            {
                "start": s.start,
                "end": s.end,
                "shot_type": s.shot_type,
                "confidence": s.confidence,
            }
            for s in result.shots
        ],
    }


# ---------------------------------------------------------------------------
# POST /analysis/shot-classify-frame (synchronous)
# ---------------------------------------------------------------------------
@analysis_bp.route("/analysis/shot-classify-frame", methods=["POST"])
@require_csrf
def analysis_shot_classify_frame():
    """Classify a single frame image by shot type."""
    from opencut.core.shot_classify import classify_single_frame
    from opencut.security import validate_filepath

    data = request.get_json(silent=True) or {}
    frame_path = data.get("filepath", "").strip()
    if not frame_path:
        return jsonify({"error": "filepath is required"}), 400

    try:
        frame_path = validate_filepath(frame_path)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400

    result = classify_single_frame(frame_path)
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /caption/compliance
# ---------------------------------------------------------------------------
@analysis_bp.route("/caption/compliance", methods=["POST"])
@require_csrf
@async_job("caption_compliance", filepath_param="srt_path")
def caption_compliance(job_id, filepath, data):
    """Check SRT subtitles against a compliance standard."""
    from opencut.core.caption_compliance import check_caption_compliance

    standard = data.get("standard", "netflix").strip().lower()
    if standard not in ("netflix", "bbc", "fcc", "youtube"):
        standard = "netflix"

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = check_caption_compliance(
        filepath,
        standard=standard,
        on_progress=_on_progress,
    )

    return {
        "overall_pass": result.overall_pass,
        "pass_rate": result.pass_rate,
        "standard": result.standard,
        "total_subtitles": result.total_subtitles,
        "checked_rules": result.checked_rules,
        "total_violations": len(result.violations),
        "violations": [
            {
                "line_num": v.line_num,
                "start_time": v.start_time,
                "violation_type": v.violation_type,
                "description": v.description,
                "severity": v.severity,
                "fix_suggestion": v.fix_suggestion,
            }
            for v in result.violations
        ],
    }


# ---------------------------------------------------------------------------
# POST /caption/compliance/fix
# ---------------------------------------------------------------------------
@analysis_bp.route("/caption/compliance/fix", methods=["POST"])
@require_csrf
@async_job("caption_compliance_fix", filepath_param="srt_path")
def caption_compliance_fix(job_id, filepath, data):
    """Auto-fix compliance violations in an SRT file."""
    from opencut.core.caption_compliance import auto_fix_compliance

    standard = data.get("standard", "netflix").strip().lower()
    if standard not in ("netflix", "bbc", "fcc", "youtube"):
        standard = "netflix"

    output = data.get("output_path", "").strip() or None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_fix_compliance(
        filepath,
        standard=standard,
        output_path_str=output,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /subtitle/snap-to-cuts
# ---------------------------------------------------------------------------
@analysis_bp.route("/subtitle/snap-to-cuts", methods=["POST"])
@require_csrf
@async_job("subtitle_snap", filepath_param="srt_path")
def subtitle_snap_to_cuts(job_id, filepath, data):
    """Snap subtitle timing to scene cut points."""
    from opencut.core.subtitle_timing import snap_subtitles_to_cuts

    cut_times = data.get("cut_times", [])
    if not isinstance(cut_times, list):
        cut_times = []
    # Sanitize
    cut_times = [float(c) for c in cut_times if isinstance(c, (int, float))]

    min_gap_frames = safe_int(data.get("min_gap_frames", 2), 2, min_val=0, max_val=30)
    fps = safe_float(data.get("fps", 24.0), 24.0, min_val=1.0, max_val=240.0)
    output = data.get("output_path", "").strip() or None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = snap_subtitles_to_cuts(
        srt_path=filepath,
        cut_times=cut_times,
        min_gap_frames=min_gap_frames,
        fps=fps,
        output_path=output,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /subtitle/auto-snap
# ---------------------------------------------------------------------------
@analysis_bp.route("/subtitle/auto-snap", methods=["POST"])
@require_csrf
@async_job("subtitle_auto_snap", filepath_param="srt_path")
def subtitle_auto_snap(job_id, filepath, data):
    """Auto-detect cuts from video and snap subtitle timing."""
    from opencut.core.subtitle_timing import auto_snap_subtitles
    from opencut.security import validate_filepath

    video_path = data.get("video_path", "").strip()
    if not video_path:
        raise ValueError("video_path is required")

    video_path = validate_filepath(video_path)
    output = data.get("output_path", "").strip() or None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_snap_subtitles(
        srt_path=filepath,
        video_path=video_path,
        output_path=output,
        on_progress=_on_progress,
    )

    return result
