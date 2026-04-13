"""
OpenCut Remote Processing & Real-Time AI Routes

Endpoints for remote render-node management and real-time AI
preview sessions.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float

logger = logging.getLogger("opencut")

remote_realtime_bp = Blueprint("remote_realtime", __name__)


# ===================================================================
# Remote Processing — Node Management
# ===================================================================

@remote_realtime_bp.route("/remote/register-node", methods=["POST"])
@require_csrf
def register_node():
    """Register a remote OpenCut render node."""
    try:
        from opencut.core.remote_process import register_node as do_register

        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "url is required"}), 400

        api_key = data.get("api_key", "")
        name = data.get("name", "")

        node = do_register(url=url, api_key=api_key, name=name)
        return jsonify({"node": node.to_dict()})
    except Exception as exc:
        return safe_error(exc, "register_node")


@remote_realtime_bp.route("/remote/nodes", methods=["GET"])
def list_nodes():
    """List all registered remote nodes."""
    try:
        from opencut.core.remote_process import list_nodes as do_list

        nodes = do_list()
        return jsonify({"nodes": [n.to_dict() for n in nodes]})
    except Exception as exc:
        return safe_error(exc, "list_nodes")


@remote_realtime_bp.route("/remote/ping", methods=["POST"])
@require_csrf
def ping_node():
    """Ping a remote node to check health."""
    try:
        from opencut.core.remote_process import ping_node as do_ping

        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "url is required"}), 400

        api_key = data.get("api_key", "")
        result = do_ping(url=url, api_key=api_key)
        return jsonify({"status": "online", "health": result})
    except Exception as exc:
        return safe_error(exc, "ping_node")


@remote_realtime_bp.route("/remote/submit", methods=["POST"])
@require_csrf
@async_job("remote_submit", filepath_required=True)
def submit_remote_job(job_id, filepath, data):
    """Submit a job to a remote render node."""
    from opencut.core.remote_process import submit_remote_job as do_submit

    node_url = (data.get("node_url") or "").strip()
    if not node_url:
        raise ValueError("node_url is required")

    endpoint = (data.get("endpoint") or "").strip()
    if not endpoint:
        raise ValueError("endpoint is required")

    params = data.get("params", {})

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = do_submit(
        node_url=node_url,
        file_path=filepath,
        endpoint=endpoint,
        params=params,
        on_progress=_on_progress,
    )

    return result.to_dict()


@remote_realtime_bp.route("/remote/job/<job_id>", methods=["GET"])
def check_remote_job(job_id):
    """Check the status of a tracked remote job."""
    try:
        from opencut.core.remote_process import get_remote_job

        rjob = get_remote_job(job_id)
        if rjob is None:
            return jsonify({"error": "Remote job not found"}), 404
        return jsonify(rjob.to_dict())
    except Exception as exc:
        return safe_error(exc, "check_remote_job")


@remote_realtime_bp.route("/remote/download", methods=["POST"])
@require_csrf
def download_remote_result():
    """Download the result file from a completed remote job."""
    try:
        from opencut.core.remote_process import download_result

        data = request.get_json(silent=True) or {}
        node_url = (data.get("node_url") or "").strip()
        remote_job_id = (data.get("remote_job_id") or "").strip()
        local_path = (data.get("local_path") or "").strip()

        if not node_url:
            return jsonify({"error": "node_url is required"}), 400
        if not remote_job_id:
            return jsonify({"error": "remote_job_id is required"}), 400
        if not local_path:
            return jsonify({"error": "local_path is required"}), 400

        result_path = download_result(
            node_url=node_url,
            remote_job_id=remote_job_id,
            local_path=local_path,
        )
        return jsonify({"path": result_path})
    except Exception as exc:
        return safe_error(exc, "download_remote_result")


# ===================================================================
# Real-Time AI Preview
# ===================================================================

@remote_realtime_bp.route("/realtime/create-session", methods=["POST"])
@require_csrf
def create_realtime_session():
    """Create a real-time AI preview session."""
    try:
        from opencut.core.realtime_ai import (
            RealtimeConfig,
            create_session,
        )

        data = request.get_json(silent=True) or {}
        model_name = (data.get("model") or "style_transfer").strip()
        video_path = (data.get("video_path") or "").strip()

        if not video_path:
            return jsonify({"error": "video_path is required"}), 400

        config = None
        if any(k in data for k in ("resolution_scale", "target_fps", "device", "params")):
            config = RealtimeConfig(
                model=model_name,
                resolution_scale=safe_float(
                    data.get("resolution_scale", 0.5), default=0.5,
                    min_val=0.1, max_val=1.0,
                ),
                target_fps=safe_float(
                    data.get("target_fps", 12), default=12.0,
                    min_val=1.0, max_val=60.0,
                ),
                device=data.get("device", "cpu"),
                params=data.get("params", {}),
            )

        session = create_session(
            model_name=model_name,
            video_path=video_path,
            config=config,
        )
        return jsonify({"session": session.to_dict()})
    except Exception as exc:
        return safe_error(exc, "create_realtime_session")


@remote_realtime_bp.route("/realtime/frame", methods=["POST"])
@require_csrf
def get_realtime_frame():
    """Get a single preview frame at a specific timestamp."""
    try:
        from opencut.core.realtime_ai import get_preview_frame

        data = request.get_json(silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        timestamp = safe_float(data.get("timestamp", 0), default=0.0, min_val=0.0)

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        frame = get_preview_frame(session_id=session_id, timestamp=timestamp)
        return jsonify({"frame": frame.to_dict()})
    except Exception as exc:
        return safe_error(exc, "get_realtime_frame")


@remote_realtime_bp.route("/realtime/update-params", methods=["POST"])
@require_csrf
def update_realtime_params():
    """Update effect parameters for a running session."""
    try:
        from opencut.core.realtime_ai import update_params

        data = request.get_json(silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        params = data.get("params", {})

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not params:
            return jsonify({"error": "params is required"}), 400

        session = update_params(session_id=session_id, params=params)
        return jsonify({"session": session.to_dict()})
    except Exception as exc:
        return safe_error(exc, "update_realtime_params")


@remote_realtime_bp.route("/realtime/stop", methods=["POST"])
@require_csrf
def stop_realtime_session():
    """Stop a running real-time preview session."""
    try:
        from opencut.core.realtime_ai import stop_session

        data = request.get_json(silent=True) or {}
        session_id = (data.get("session_id") or "").strip()

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        stopped = stop_session(session_id=session_id)
        if not stopped:
            return jsonify({"error": "Session not found"}), 404

        return jsonify({"stopped": True, "session_id": session_id})
    except Exception as exc:
        return safe_error(exc, "stop_realtime_session")


@remote_realtime_bp.route("/realtime/models", methods=["GET"])
def list_realtime_models():
    """List available real-time AI models."""
    try:
        from opencut.core.realtime_ai import get_cache_stats, list_realtime_models

        models = list_realtime_models()
        return jsonify({
            "models": models,
            "cache": get_cache_stats(),
        })
    except Exception as exc:
        return safe_error(exc, "list_realtime_models")
