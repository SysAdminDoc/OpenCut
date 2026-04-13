"""
OpenCut QC/QA Routes

Quality-control check endpoints for video and audio files:
black frames, frozen frames, audio phase, silence gaps,
leader detection, and full QC check.
"""

import logging

from flask import Blueprint

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float

logger = logging.getLogger("opencut")

qc_bp = Blueprint("qc", __name__)


# ---------------------------------------------------------------------------
# POST /qc/black-frames
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/black-frames", methods=["POST"])
@require_csrf
@async_job("qc_black_frames")
def qc_black_frames(job_id, filepath, data):
    """Detect black frame regions in a video."""
    from opencut.core.qc_checks import detect_black_frames

    threshold = safe_float(data.get("threshold", 0.98), 0.98, min_val=0.0, max_val=1.0)
    min_duration = safe_float(data.get("min_duration", 0.5), 0.5, min_val=0.0, max_val=60.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_black_frames(
        filepath,
        threshold=threshold,
        min_duration=min_duration,
        on_progress=_on_progress,
    )

    return {
        "total_regions": len(result.frames),
        "total_black_duration": result.total_black_duration,
        "file_duration": result.file_duration,
        "black_percentage": result.black_percentage,
        "frames": [
            {"start": f.start, "end": f.end, "duration": f.duration}
            for f in result.frames
        ],
    }


# ---------------------------------------------------------------------------
# POST /qc/frozen-frames
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/frozen-frames", methods=["POST"])
@require_csrf
@async_job("qc_frozen_frames")
def qc_frozen_frames(job_id, filepath, data):
    """Detect frozen (still) frame regions in a video."""
    from opencut.core.qc_checks import detect_frozen_frames

    noise_threshold = safe_float(data.get("noise_threshold", 0.001), 0.001, min_val=0.0, max_val=1.0)
    duration_threshold = safe_float(data.get("duration_threshold", 2.0), 2.0, min_val=0.1, max_val=60.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_frozen_frames(
        filepath,
        noise_threshold=noise_threshold,
        duration_threshold=duration_threshold,
        on_progress=_on_progress,
    )

    return {
        "total_regions": len(result.frames),
        "total_frozen_duration": result.total_frozen_duration,
        "file_duration": result.file_duration,
        "frozen_percentage": result.frozen_percentage,
        "frames": [
            {"start": f.start, "end": f.end, "duration": f.duration}
            for f in result.frames
        ],
    }


# ---------------------------------------------------------------------------
# POST /qc/audio-phase
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/audio-phase", methods=["POST"])
@require_csrf
@async_job("qc_audio_phase")
def qc_audio_phase(job_id, filepath, data):
    """Check audio phase correlation for out-of-phase issues."""
    from opencut.core.qc_checks import check_audio_phase

    threshold = safe_float(data.get("threshold", -0.5), -0.5, min_val=-1.0, max_val=1.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = check_audio_phase(
        filepath,
        threshold=threshold,
        on_progress=_on_progress,
    )

    return {
        "has_phase_problems": result.has_phase_problems,
        "overall_avg_phase": result.overall_avg_phase,
        "file_duration": result.file_duration,
        "total_issues": len(result.issues),
        "issues": [
            {"start": i.start, "end": i.end, "avg_phase": i.avg_phase}
            for i in result.issues
        ],
    }


# ---------------------------------------------------------------------------
# POST /qc/silence-gaps
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/silence-gaps", methods=["POST"])
@require_csrf
@async_job("qc_silence_gaps")
def qc_silence_gaps(job_id, filepath, data):
    """Detect silence gaps in audio track."""
    from opencut.core.qc_checks import detect_silence_gaps

    noise_db = safe_float(data.get("noise_db", -50), -50, min_val=-100.0, max_val=0.0)
    min_duration = safe_float(data.get("min_duration", 2.0), 2.0, min_val=0.1, max_val=60.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_silence_gaps(
        filepath,
        noise_db=noise_db,
        min_duration=min_duration,
        on_progress=_on_progress,
    )

    return {
        "total_gaps": len(result.gaps),
        "total_silence_duration": result.total_silence_duration,
        "file_duration": result.file_duration,
        "silence_percentage": result.silence_percentage,
        "gaps": [
            {"start": g.start, "end": g.end, "duration": g.duration}
            for g in result.gaps
        ],
    }


# ---------------------------------------------------------------------------
# POST /qc/leader-detect
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/leader-detect", methods=["POST"])
@require_csrf
@async_job("qc_leader_detect")
def qc_leader_detect(job_id, filepath, data):
    """Detect color bars, reference tone, and slate in video leader."""
    from opencut.core.qc_checks import detect_leader_elements

    scan_duration = safe_float(data.get("scan_duration", 120.0), 120.0, min_val=1.0, max_val=600.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_leader_elements(
        filepath,
        scan_duration=scan_duration,
        on_progress=_on_progress,
    )

    return {
        "bars_detected": result.bars_detected,
        "bars_end_time": result.bars_end_time,
        "tone_detected": result.tone_detected,
        "tone_end_time": result.tone_end_time,
        "slate_detected": result.slate_detected,
        "slate_end_time": result.slate_end_time,
        "recommended_trim_point": result.recommended_trim_point,
    }


# ---------------------------------------------------------------------------
# POST /qc/full-check
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/full-check", methods=["POST"])
@require_csrf
@async_job("qc_full_check")
def qc_full_check(job_id, filepath, data):
    """Run all QC checks and return a combined report."""
    from opencut.core.qc_checks import run_full_qc

    # Per-check configuration from request
    black_threshold = safe_float(data.get("black_threshold", 0.98), 0.98, min_val=0.0, max_val=1.0)
    black_min_duration = safe_float(data.get("black_min_duration", 0.5), 0.5, min_val=0.0, max_val=60.0)
    freeze_noise = safe_float(data.get("freeze_noise", 0.001), 0.001, min_val=0.0, max_val=1.0)
    freeze_duration = safe_float(data.get("freeze_duration", 2.0), 2.0, min_val=0.1, max_val=60.0)
    phase_threshold = safe_float(data.get("phase_threshold", -0.5), -0.5, min_val=-1.0, max_val=1.0)
    silence_noise_db = safe_float(data.get("silence_noise_db", -50), -50, min_val=-100.0, max_val=0.0)
    silence_min_duration = safe_float(data.get("silence_min_duration", 2.0), 2.0, min_val=0.1, max_val=60.0)
    leader_scan_duration = safe_float(data.get("leader_scan_duration", 120.0), 120.0, min_val=1.0, max_val=600.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    report = run_full_qc(
        filepath,
        on_progress=_on_progress,
        black_threshold=black_threshold,
        black_min_duration=black_min_duration,
        freeze_noise=freeze_noise,
        freeze_duration=freeze_duration,
        phase_threshold=phase_threshold,
        silence_noise_db=silence_noise_db,
        silence_min_duration=silence_min_duration,
        leader_scan_duration=leader_scan_duration,
    )

    # Build response
    result = {
        "passed": report.passed,
        "issues_summary": report.issues_summary,
    }

    if report.black_frames:
        result["black_frames"] = {
            "total_regions": len(report.black_frames.frames),
            "total_black_duration": report.black_frames.total_black_duration,
            "black_percentage": report.black_frames.black_percentage,
            "frames": [
                {"start": f.start, "end": f.end, "duration": f.duration}
                for f in report.black_frames.frames
            ],
        }

    if report.frozen_frames:
        result["frozen_frames"] = {
            "total_regions": len(report.frozen_frames.frames),
            "total_frozen_duration": report.frozen_frames.total_frozen_duration,
            "frozen_percentage": report.frozen_frames.frozen_percentage,
            "frames": [
                {"start": f.start, "end": f.end, "duration": f.duration}
                for f in report.frozen_frames.frames
            ],
        }

    if report.audio_phase:
        result["audio_phase"] = {
            "has_phase_problems": report.audio_phase.has_phase_problems,
            "overall_avg_phase": report.audio_phase.overall_avg_phase,
            "total_issues": len(report.audio_phase.issues),
            "issues": [
                {"start": i.start, "end": i.end, "avg_phase": i.avg_phase}
                for i in report.audio_phase.issues
            ],
        }

    if report.silence_gaps:
        result["silence_gaps"] = {
            "total_gaps": len(report.silence_gaps.gaps),
            "total_silence_duration": report.silence_gaps.total_silence_duration,
            "silence_percentage": report.silence_gaps.silence_percentage,
            "gaps": [
                {"start": g.start, "end": g.end, "duration": g.duration}
                for g in report.silence_gaps.gaps
            ],
        }

    if report.leader:
        result["leader"] = {
            "bars_detected": report.leader.bars_detected,
            "bars_end_time": report.leader.bars_end_time,
            "tone_detected": report.leader.tone_detected,
            "tone_end_time": report.leader.tone_end_time,
            "slate_detected": report.leader.slate_detected,
            "slate_end_time": report.leader.slate_end_time,
            "recommended_trim_point": report.leader.recommended_trim_point,
        }

    return result


# ---------------------------------------------------------------------------
# POST /qc/dropouts
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/dropouts", methods=["POST"])
@require_csrf
@async_job("qc_dropouts")
def qc_dropouts(job_id, filepath, data):
    """Detect dropout and glitch artifacts in a video."""
    from opencut.core.qc_checks import detect_dropouts

    ssim_threshold = safe_float(data.get("ssim_threshold", 0.5), 0.5, min_val=0.0, max_val=1.0)
    block_threshold = safe_float(data.get("block_threshold", 0.3), 0.3, min_val=0.0, max_val=1.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_dropouts(
        filepath,
        ssim_threshold=ssim_threshold,
        block_threshold=block_threshold,
        on_progress=_on_progress,
    )

    return {
        "total_dropouts": result.total_dropouts,
        "file_duration": result.file_duration,
        "frames_analyzed": result.frames_analyzed,
        "dropouts": [
            {
                "frame_num": d.frame_num,
                "timestamp": d.timestamp,
                "type": d.type,
                "severity": d.severity,
                "description": d.description,
            }
            for d in result.dropouts
        ],
    }


# ---------------------------------------------------------------------------
# POST /qc/report
# ---------------------------------------------------------------------------
@qc_bp.route("/qc/report", methods=["POST"])
@require_csrf
@async_job("qc_report")
def qc_report(job_id, filepath, data):
    """Generate a comprehensive QC report for a video file."""
    from opencut.core.qc_checks import generate_qc_report

    ruleset = data.get("ruleset", "broadcast").strip().lower()
    if ruleset not in ("broadcast", "netflix", "youtube"):
        ruleset = "broadcast"

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    report = generate_qc_report(
        filepath,
        ruleset=ruleset,
        on_progress=_on_progress,
    )

    return {
        "overall_verdict": report.overall_verdict,
        "total_issues": report.total_issues,
        "critical_count": report.critical_count,
        "warning_count": report.warning_count,
        "ruleset": report.ruleset,
        "file_duration": report.file_duration,
        "per_check": [
            {
                "check_name": c.check_name,
                "status": c.status,
                "details": c.details,
                "issues": c.issues,
            }
            for c in report.per_check
        ],
        "html_report": report.html_report,
    }
