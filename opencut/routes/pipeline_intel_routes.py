"""
OpenCut Pipeline Intelligence Routes

Endpoints for pipeline health monitoring, scheduled jobs, smart content
routing, processing time estimation, and resource monitoring.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")

pipeline_intel_bp = Blueprint("pipeline_intel", __name__)


# =========================================================================
# Pipeline Health
# =========================================================================

@pipeline_intel_bp.route("/api/pipeline/health", methods=["GET"])
def pipeline_health():
    """Get pipeline health dashboard data."""
    from opencut.core.pipeline_health import get_pipeline_health

    timeframe = safe_int(request.args.get("timeframe_hours", 24), 24, 1)
    result = get_pipeline_health(timeframe_hours=timeframe)
    return jsonify(result.to_dict())


@pipeline_intel_bp.route("/api/pipeline/errors", methods=["GET"])
def pipeline_errors():
    """Get pipeline error summary."""
    from opencut.core.pipeline_health import get_error_summary

    timeframe = safe_int(request.args.get("timeframe_hours", 24), 24, 1)
    errors = get_error_summary(timeframe_hours=timeframe)
    return jsonify({"errors": [e.to_dict() for e in errors]})


@pipeline_intel_bp.route("/api/pipeline/health/record", methods=["POST"])
@require_csrf
def pipeline_health_record():
    """Record a pipeline health metric."""
    from opencut.core.pipeline_health import record_metric

    data = request.get_json(silent=True) or {}
    operation = data.get("operation", "")
    if not operation:
        return jsonify({"error": "operation is required"}), 400

    duration_s = safe_float(data.get("duration_s", 0), 0.0, 0.0)
    success = bool(data.get("success", True))

    metric = record_metric(
        operation=operation,
        duration_s=duration_s,
        success=success,
        error_type=data.get("error_type", ""),
        error_message=data.get("error_message", ""),
        cpu_pct=safe_float(data.get("cpu_pct", 0), 0.0, 0.0),
        gpu_pct=safe_float(data.get("gpu_pct", 0), 0.0, 0.0),
        ram_mb=safe_float(data.get("ram_mb", 0), 0.0, 0.0),
        disk_write_mb=safe_float(data.get("disk_write_mb", 0), 0.0, 0.0),
    )
    return jsonify(metric.to_dict())


# =========================================================================
# Scheduled Jobs
# =========================================================================

@pipeline_intel_bp.route("/api/pipeline/schedules", methods=["POST"])
@require_csrf
def create_schedule():
    """Create a new scheduled job."""
    from opencut.core.scheduled_jobs import create_schedule as _create

    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    cron_expr = data.get("cron_expr", "")
    if not name or not cron_expr:
        return jsonify({"error": "name and cron_expr are required"}), 400

    try:
        job = _create(
            name=name,
            cron_expr=cron_expr,
            job_config=data.get("job_config"),
            enabled=bool(data.get("enabled", True)),
            tags=data.get("tags"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(job.to_dict()), 201


@pipeline_intel_bp.route("/api/pipeline/schedules", methods=["GET"])
def list_schedules():
    """List all scheduled jobs."""
    from opencut.core.scheduled_jobs import list_schedules as _list

    enabled_only = request.args.get("enabled_only", "").lower() in ("1", "true", "yes")
    job_type = request.args.get("job_type", "") or None
    tag = request.args.get("tag", "") or None

    jobs = _list(enabled_only=enabled_only, job_type=job_type, tag=tag)
    return jsonify({"schedules": [j.to_dict() for j in jobs]})


@pipeline_intel_bp.route("/api/pipeline/schedules/<schedule_id>", methods=["GET"])
def get_schedule(schedule_id):
    """Get a single scheduled job by ID."""
    from opencut.core.scheduled_jobs import get_schedule as _get

    job = _get(schedule_id)
    if job is None:
        return jsonify({"error": "Schedule not found"}), 404
    return jsonify(job.to_dict())


@pipeline_intel_bp.route("/api/pipeline/schedules/<schedule_id>", methods=["PUT"])
@require_csrf
def update_schedule(schedule_id):
    """Update an existing scheduled job."""
    from opencut.core.scheduled_jobs import update_schedule as _update

    data = request.get_json(silent=True) or {}

    try:
        updated = _update(
            schedule_id=schedule_id,
            name=data.get("name"),
            cron_expr=data.get("cron_expr"),
            enabled=data.get("enabled"),
            job_config=data.get("job_config"),
            tags=data.get("tags"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if updated is None:
        return jsonify({"error": "Schedule not found"}), 404
    return jsonify(updated.to_dict())


@pipeline_intel_bp.route("/api/pipeline/schedules/<schedule_id>", methods=["DELETE"])
@require_csrf
def delete_schedule(schedule_id):
    """Delete a scheduled job."""
    from opencut.core.scheduled_jobs import delete_schedule as _delete

    deleted = _delete(schedule_id)
    if not deleted:
        return jsonify({"error": "Schedule not found"}), 404
    return jsonify({"deleted": True, "schedule_id": schedule_id})


@pipeline_intel_bp.route("/api/pipeline/schedules/due", methods=["GET"])
def check_due_schedules():
    """Check which schedules are due for execution."""
    from opencut.core.scheduled_jobs import check_due_jobs

    due = check_due_jobs()
    return jsonify({"due_jobs": [j.to_dict() for j in due]})


@pipeline_intel_bp.route("/api/pipeline/schedules/history", methods=["GET"])
def schedule_history():
    """Get scheduled job execution history."""
    from opencut.core.scheduled_jobs import get_job_history

    schedule_id = request.args.get("schedule_id", "") or None
    limit = safe_int(request.args.get("limit", 50), 50, 1)
    history = get_job_history(schedule_id=schedule_id, limit=limit)
    return jsonify({"history": [h.to_dict() for h in history]})


# =========================================================================
# Smart Content Routing
# =========================================================================

@pipeline_intel_bp.route("/api/video/classify-content", methods=["POST"])
@require_csrf
@async_job("classify_content")
def classify_content(job_id, filepath, data):
    """Classify video content type and suggest workflows."""
    from opencut.core.smart_route import classify_content as _classify

    face_count = safe_int(data.get("face_count", -1), -1)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    classification = _classify(
        video_path=filepath,
        face_count=face_count,
        on_progress=_progress,
    )
    return classification.to_dict()


@pipeline_intel_bp.route("/api/video/suggest-workflow", methods=["POST"])
@require_csrf
def suggest_workflow():
    """Suggest optimal workflow from a classification result."""
    from opencut.core.smart_route import ContentClassification, suggest_workflow as _suggest

    data = request.get_json(silent=True) or {}
    classification_data = data.get("classification", {})
    if not classification_data:
        return jsonify({"error": "classification object is required"}), 400

    classification = ContentClassification(
        content_type=classification_data.get("content_type", "unknown"),
        confidence=float(classification_data.get("confidence", 0)),
        label=classification_data.get("label", ""),
        description=classification_data.get("description", ""),
        scores=classification_data.get("scores", {}),
        video_traits=classification_data.get("video_traits", {}),
    )

    result = _suggest(classification)
    return jsonify(result.to_dict())


@pipeline_intel_bp.route("/api/video/content-types", methods=["GET"])
def list_content_types():
    """List all known content types and their indicators."""
    from opencut.core.smart_route import CONTENT_TYPES

    return jsonify({"content_types": CONTENT_TYPES})


# =========================================================================
# Processing Time Estimation
# =========================================================================

@pipeline_intel_bp.route("/api/pipeline/estimate", methods=["POST"])
@require_csrf
@async_job("estimate_time")
def estimate_processing_time(job_id, filepath, data):
    """Estimate processing time for an operation on a video."""
    from opencut.core.process_estimate import estimate_processing_time as _estimate

    operation = data.get("operation", "")
    if not operation:
        return {"error": "operation is required"}

    params = data.get("params", {})
    gpu = data.get("gpu_available")
    if gpu is not None:
        gpu = bool(gpu)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _estimate(
        video_path=filepath,
        operation=operation,
        params=params,
        gpu_available=gpu,
        on_progress=_progress,
    )
    return result.to_dict()


@pipeline_intel_bp.route("/api/pipeline/estimate/batch", methods=["POST"])
@require_csrf
@async_job("batch_estimate")
def batch_estimate_time(job_id, filepath, data):
    """Estimate processing time for multiple sequential operations."""
    from opencut.core.process_estimate import batch_estimate as _batch

    operations = data.get("operations", [])
    if not operations:
        return {"error": "operations list is required"}

    gpu = data.get("gpu_available")
    if gpu is not None:
        gpu = bool(gpu)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = _batch(
        video_path=filepath,
        operations=operations,
        gpu_available=gpu,
        on_progress=_progress,
    )
    return result.to_dict()


@pipeline_intel_bp.route("/api/pipeline/estimate/baselines", methods=["GET"])
def get_estimate_baselines():
    """List operation baselines used for estimation."""
    from opencut.core.process_estimate import OPERATION_BASELINES

    items = {}
    for op, info in OPERATION_BASELINES.items():
        items[op] = {
            "description": info["description"],
            "base_ratio": info["base_ratio"],
            "gpu_factor": info["gpu_factor"],
            "resolution_scale": info.get("resolution_scale", False),
        }
    return jsonify({"baselines": items})


@pipeline_intel_bp.route("/api/pipeline/estimate/accuracy", methods=["GET"])
def get_estimate_accuracy():
    """Get estimation accuracy statistics."""
    from opencut.core.process_estimate import get_estimate_accuracy as _accuracy

    operation = request.args.get("operation", "") or None
    days = safe_int(request.args.get("days", 30), 30, 1)
    result = _accuracy(operation=operation, days=days)
    return jsonify(result)


# =========================================================================
# Resource Monitoring
# =========================================================================

@pipeline_intel_bp.route("/api/pipeline/resources", methods=["GET"])
def resource_snapshot():
    """Get current resource utilization snapshot."""
    from opencut.core.resource_monitor import get_resource_snapshot

    extra_paths = request.args.getlist("disk_path")
    snap = get_resource_snapshot(extra_disk_paths=extra_paths or None)
    return jsonify(snap.to_dict())


@pipeline_intel_bp.route("/api/pipeline/resources/gpu", methods=["GET"])
def resource_gpu():
    """Get GPU information."""
    from opencut.core.resource_monitor import get_gpu_info

    gpus = get_gpu_info()
    return jsonify({"gpus": [g.to_dict() for g in gpus]})


@pipeline_intel_bp.route("/api/pipeline/resources/history", methods=["GET"])
def resource_history():
    """Get resource utilization history."""
    from opencut.core.resource_monitor import get_resource_history

    minutes = safe_int(request.args.get("minutes", 60), 60, 1)
    history = get_resource_history(minutes=minutes)
    return jsonify({"history": [s.to_dict() for s in history]})


@pipeline_intel_bp.route("/api/pipeline/resources/check", methods=["POST"])
@require_csrf
def resource_availability_check():
    """Check resource availability against requirements."""
    from opencut.core.resource_monitor import check_resource_availability

    data = request.get_json(silent=True) or {}
    requirements = data.get("requirements", {})
    result = check_resource_availability(requirements=requirements)
    return jsonify(result.to_dict())
