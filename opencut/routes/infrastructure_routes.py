"""
OpenCut Infrastructure & Platform Routes

Endpoints for auto-update, model quantization, MCP tools,
model downloads, Apple Silicon detection, and GPU dashboard.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf, validate_filepath, validate_output_path

logger = logging.getLogger("opencut")

infra_bp = Blueprint("infra", __name__)


# ===================================================================
# 7.3 — Auto-Update
# ===================================================================

@infra_bp.route("/api/system/check-updates", methods=["GET"])
def check_updates():
    """Check for available updates."""
    try:
        from opencut.core.auto_update import check_for_updates
        include_pre = request.args.get("include_prerelease", "false").lower() == "true"
        result = check_for_updates(include_prerelease=include_pre)
        return jsonify(result.to_dict())
    except Exception as exc:
        return safe_error(exc, "check_updates")


@infra_bp.route("/api/system/update", methods=["POST"])
@require_csrf
def trigger_update():
    """Trigger an update via the specified method."""
    try:
        from opencut.core.auto_update import trigger_update as do_update
        data = request.get_json(silent=True) or {}
        method = data.get("method", "pip")
        result = do_update(method=method)
        status_code = 200 if result.success else 500
        return jsonify(result.to_dict()), status_code
    except Exception as exc:
        return safe_error(exc, "trigger_update")


# ===================================================================
# 7.4 — Model Quantization
# ===================================================================

@infra_bp.route("/api/models/quantization", methods=["GET"])
def list_quantizable():
    """List models available for quantization."""
    try:
        from opencut.core.model_quantization import (
            list_quantizable_models,
            recommend_quantization,
        )
        models = list_quantizable_models()
        recommendation = recommend_quantization()
        return jsonify({
            "models": [m.to_dict() for m in models],
            "recommendation": recommendation.to_dict(),
        })
    except Exception as exc:
        return safe_error(exc, "list_quantizable")


@infra_bp.route("/api/models/quantize", methods=["POST"])
@require_csrf
def quantize_model():
    """Quantize a model to the requested precision."""
    try:
        from opencut.core.model_quantization import quantize_model as do_quantize
        data = request.get_json(silent=True) or {}
        model_path = data.get("model_path", "")
        if not model_path:
            return jsonify({"error": "model_path is required"}), 400
        validate_filepath(model_path)
        precision = data.get("precision", "int8")
        output = data.get("output_path")
        if output:
            output = validate_output_path(output)
        result = do_quantize(
            model_path=model_path,
            target_precision=precision,
            output_path=output,
        )
        status_code = 200 if result.success else 400
        return jsonify(result.to_dict()), status_code
    except Exception as exc:
        return safe_error(exc, "quantize_model")


# ===================================================================
# 7.7 — MCP Tools
# ===================================================================

@infra_bp.route("/api/mcp/tools", methods=["GET"])
def mcp_tools():
    """List all available MCP tool definitions."""
    try:
        from opencut.core.mcp_tools import (
            get_compound_tools,
            get_mcp_tools,
            list_available_operations,
        )
        category = request.args.get("category")
        tools = get_mcp_tools()
        if category:
            tools = [t for t in tools if t.get("category") == category]
        return jsonify({
            "tools": tools,
            "compound_tools": get_compound_tools(),
            "operations": list_available_operations(),
        })
    except Exception as exc:
        return safe_error(exc, "mcp_tools")


@infra_bp.route("/api/mcp/compound", methods=["POST"])
@require_csrf
def mcp_compound():
    """Execute a compound MCP tool."""
    try:
        from opencut.core.mcp_tools import execute_compound_tool
        data = request.get_json(silent=True) or {}
        tool_name = data.get("tool_name", "")
        params = data.get("params", {})
        if not tool_name:
            return jsonify({"error": "tool_name is required"}), 400
        result = execute_compound_tool(tool_name=tool_name, params=params)
        status_code = 200 if result.success else 400
        return jsonify(result.to_dict()), status_code
    except Exception as exc:
        return safe_error(exc, "mcp_compound")


# ===================================================================
# 10.1 — Model Download Manager
# ===================================================================

@infra_bp.route("/api/models/download", methods=["POST"])
@require_csrf
def download_model():
    """Queue a model for background download."""
    try:
        from opencut.core.model_manager import queue_download
        data = request.get_json(silent=True) or {}
        model_name = data.get("model_name", "")
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400
        url = data.get("url")
        throttle = int(data.get("throttle_kbps", 0))
        progress = queue_download(
            model_name=model_name,
            url=url,
            throttle_kbps=throttle,
        )
        return jsonify(progress.to_dict())
    except Exception as exc:
        return safe_error(exc, "download_model")


@infra_bp.route("/api/models/progress", methods=["GET"])
def download_progress():
    """Get download progress for a model."""
    try:
        from opencut.core.model_manager import get_download_progress
        model_name = request.args.get("model_name", "")
        if not model_name:
            return jsonify({"error": "model_name query parameter is required"}), 400
        progress = get_download_progress(model_name)
        return jsonify(progress.to_dict())
    except Exception as exc:
        return safe_error(exc, "download_progress")


@infra_bp.route("/api/models/installed", methods=["GET"])
def installed_models():
    """List installed models."""
    try:
        from opencut.core.model_manager import list_installed_models
        models = list_installed_models()
        return jsonify({"models": [m.to_dict() for m in models]})
    except Exception as exc:
        return safe_error(exc, "installed_models")


# ===================================================================
# 10.4 — Apple Silicon / MPS Acceleration
# ===================================================================

@infra_bp.route("/api/system/apple-silicon", methods=["GET"])
def apple_silicon_info():
    """Detect Apple Silicon hardware and MPS availability."""
    try:
        from opencut.core.apple_silicon import (
            detect_apple_silicon,
            get_recommended_device,
        )
        info = detect_apple_silicon()
        # Optionally include device recommendation for a specific operation
        operation = request.args.get("operation")
        recommendation = None
        if operation:
            recommendation = get_recommended_device(operation).to_dict()
        resp = info.to_dict()
        if recommendation:
            resp["device_recommendation"] = recommendation
        return jsonify(resp)
    except Exception as exc:
        return safe_error(exc, "apple_silicon_info")


# ===================================================================
# 32.4 — GPU Memory Management Dashboard
# ===================================================================

@infra_bp.route("/api/gpu/status", methods=["GET"])
def gpu_status():
    """Get current GPU/VRAM status."""
    try:
        from opencut.core.gpu_dashboard import get_gpu_info, get_vram_status
        status = get_vram_status()
        gpus = get_gpu_info()
        return jsonify({
            "status": status.to_dict(),
            "gpus": [g.to_dict() for g in gpus],
        })
    except Exception as exc:
        return safe_error(exc, "gpu_status")


@infra_bp.route("/api/gpu/models", methods=["GET"])
def gpu_models():
    """List models currently loaded in GPU memory."""
    try:
        from opencut.core.gpu_dashboard import get_loaded_models
        models = get_loaded_models()
        return jsonify({"models": [m.to_dict() for m in models]})
    except Exception as exc:
        return safe_error(exc, "gpu_models")


@infra_bp.route("/api/gpu/unload", methods=["POST"])
@require_csrf
def gpu_unload():
    """Unload a model from GPU memory."""
    try:
        from opencut.core.gpu_dashboard import recommend_unload, unload_model
        data = request.get_json(silent=True) or {}
        model_name = data.get("model_name")
        required_vram = data.get("required_vram")

        if model_name:
            success = unload_model(model_name)
            return jsonify({"success": success, "model": model_name})
        elif required_vram is not None:
            rec = recommend_unload(float(required_vram))
            return jsonify(rec.to_dict())
        else:
            return jsonify({"error": "Provide model_name to unload or required_vram for recommendation."}), 400
    except Exception as exc:
        return safe_error(exc, "gpu_unload")
