"""
OpenCut Architecture Routes

Provides endpoints for the FastAPI migration layer and GPU worker
process isolation.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf, safe_int

logger = logging.getLogger("opencut")

architecture_bp = Blueprint("architecture", __name__)


# ===========================================================================
# FastAPI / OpenAPI Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /architecture/openapi — full OpenAPI spec
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/openapi", methods=["GET"])
def architecture_openapi():
    """Return the combined OpenAPI spec for the application."""
    try:
        from flask import current_app

        from opencut.core.fastapi_app import generate_openapi_spec

        spec = generate_openapi_spec(flask_app=current_app)
        return jsonify(spec)
    except Exception as exc:
        return safe_error(exc, "architecture_openapi")


# ---------------------------------------------------------------------------
# GET /architecture/routes — list all registered routes with params
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/routes", methods=["GET"])
def architecture_routes():
    """List all registered Flask routes with methods and parameters."""
    try:
        from flask import current_app

        from opencut.core.fastapi_app import OpenCutFastAPI

        wrapper = OpenCutFastAPI(current_app)
        routes = wrapper.discover_flask_routes()
        return jsonify({
            "routes": [r.to_dict() for r in routes],
            "total": len(routes),
        })
    except Exception as exc:
        return safe_error(exc, "architecture_routes")


# ---------------------------------------------------------------------------
# GET /architecture/health — system health check
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/health", methods=["GET"])
def architecture_health():
    """Architecture layer health check."""
    try:
        import time

        status = {
            "status": "ok",
            "timestamp": time.time(),
            "components": {},
        }

        # Check FastAPI availability
        try:
            import fastapi
            status["components"]["fastapi"] = {
                "available": True,
                "version": getattr(fastapi, "__version__", "unknown"),
            }
        except ImportError:
            status["components"]["fastapi"] = {
                "available": False,
                "version": None,
            }

        # Check Pydantic availability
        try:
            import pydantic
            status["components"]["pydantic"] = {
                "available": True,
                "version": getattr(pydantic, "__version__", "unknown"),
            }
        except ImportError:
            status["components"]["pydantic"] = {
                "available": False,
                "version": None,
            }

        # Check GPU worker pool
        from opencut.core.process_isolation import get_worker_pool
        pool = get_worker_pool()
        status["components"]["gpu_worker_pool"] = {
            "initialized": pool is not None,
            "status": pool.get_status() if pool else None,
        }

        # Check nvidia-smi
        from opencut.core.process_isolation import _query_nvidia_smi
        gpus = _query_nvidia_smi()
        status["components"]["gpu"] = {
            "available": len(gpus) > 0,
            "count": len(gpus),
            "gpus": gpus,
        }

        return jsonify(status)
    except Exception as exc:
        return safe_error(exc, "architecture_health")


# ===========================================================================
# GPU Worker Pool Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /architecture/worker-pool/create — create GPU worker pool
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/worker-pool/create", methods=["POST"])
@require_csrf
def worker_pool_create():
    """Create or replace the GPU worker pool.

    Expects JSON::

        {
            "max_workers": 2,
            "vram_budget_mb": 8192
        }
    """
    try:
        from opencut.core.process_isolation import create_worker_pool

        data = request.get_json(force=True) or {}
        max_workers = safe_int(data.get("max_workers", 2), default=2, min_val=1, max_val=16)
        vram_budget_mb = safe_int(
            data.get("vram_budget_mb", 8192), default=8192, min_val=512, max_val=65536
        )

        pool = create_worker_pool(
            max_workers=max_workers,
            vram_budget_mb=vram_budget_mb,
        )
        return jsonify({
            "status": "created",
            "pool": pool.get_status(),
        })
    except Exception as exc:
        return safe_error(exc, "worker_pool_create")


# ---------------------------------------------------------------------------
# GET /architecture/worker-pool/status — pool status
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/worker-pool/status", methods=["GET"])
def worker_pool_status():
    """Return the current GPU worker pool status."""
    try:
        from opencut.core.process_isolation import get_worker_pool

        pool = get_worker_pool()
        if pool is None:
            return jsonify({
                "error": "Worker pool not initialized",
                "code": "POOL_NOT_INITIALIZED",
                "suggestion": "Create a pool first via POST /architecture/worker-pool/create",
            }), 404

        return jsonify(pool.get_status())
    except Exception as exc:
        return safe_error(exc, "worker_pool_status")


# ---------------------------------------------------------------------------
# POST /architecture/worker-pool/submit — submit isolated job
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/worker-pool/submit", methods=["POST"])
@require_csrf
def worker_pool_submit():
    """Submit a job to the GPU worker pool.

    Expects JSON::

        {
            "model_name": "whisper-large",
            "function_path": "opencut.core.llm.run_inference",
            "args": [],
            "kwargs": {},
            "vram_required": 4096,
            "timeout": 300
        }
    """
    try:
        from opencut.core.process_isolation import (
            WorkerConfig,
            get_worker_pool,
        )

        pool = get_worker_pool()
        if pool is None:
            return jsonify({
                "error": "Worker pool not initialized",
                "code": "POOL_NOT_INITIALIZED",
                "suggestion": "Create a pool first via POST /architecture/worker-pool/create",
            }), 404

        data = request.get_json(force=True) or {}
        model_name = data.get("model_name", "").strip()
        function_path = data.get("function_path", "").strip()

        if not model_name:
            return jsonify({
                "error": "model_name is required",
                "code": "INVALID_INPUT",
            }), 400
        if not function_path:
            return jsonify({
                "error": "function_path is required",
                "code": "INVALID_INPUT",
            }), 400

        args = tuple(data.get("args", []))
        kwargs = data.get("kwargs", {})
        vram_required = safe_int(data.get("vram_required", 2048), default=2048, min_val=256)
        timeout = safe_int(data.get("timeout", 600), default=600, min_val=10)

        config = WorkerConfig(
            model_name=model_name,
            vram_required=vram_required,
            timeout=timeout,
        )

        worker_id, worker = pool.submit(
            model_name=model_name,
            function_path=function_path,
            args=args,
            kwargs=kwargs,
            config=config,
        )

        if worker is None:
            return jsonify({
                "error": "Pool is full or VRAM budget exceeded",
                "code": "POOL_FULL",
                "worker_id": worker_id,
                "suggestion": "Wait for workers to finish or increase the pool capacity.",
            }), 429

        return jsonify({
            "worker_id": worker_id,
            "worker": worker.to_dict(),
        })
    except Exception as exc:
        return safe_error(exc, "worker_pool_submit")


# ---------------------------------------------------------------------------
# POST /architecture/worker-pool/kill — kill specific worker
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/worker-pool/kill", methods=["POST"])
@require_csrf
def worker_pool_kill():
    """Kill a specific GPU worker.

    Expects JSON::

        {"worker_id": "gpu-worker-1"}
    """
    try:
        from opencut.core.process_isolation import get_worker_pool

        pool = get_worker_pool()
        if pool is None:
            return jsonify({
                "error": "Worker pool not initialized",
                "code": "POOL_NOT_INITIALIZED",
            }), 404

        data = request.get_json(force=True) or {}
        worker_id = data.get("worker_id", "").strip()
        if not worker_id:
            return jsonify({
                "error": "worker_id is required",
                "code": "INVALID_INPUT",
            }), 400

        killed = pool.kill_worker(worker_id)
        return jsonify({
            "killed": killed,
            "worker_id": worker_id,
        })
    except Exception as exc:
        return safe_error(exc, "worker_pool_kill")


# ---------------------------------------------------------------------------
# POST /architecture/worker-pool/cleanup — cleanup pool
# ---------------------------------------------------------------------------
@architecture_bp.route("/architecture/worker-pool/cleanup", methods=["POST"])
@require_csrf
def worker_pool_cleanup():
    """Clean up all GPU workers in the pool."""
    try:
        from opencut.core.process_isolation import get_worker_pool

        pool = get_worker_pool()
        if pool is None:
            return jsonify({
                "error": "Worker pool not initialized",
                "code": "POOL_NOT_INITIALIZED",
            }), 404

        count = pool.cleanup()
        return jsonify({
            "cleaned_up": count,
            "status": "cleaned",
        })
    except Exception as exc:
        return safe_error(exc, "worker_pool_cleanup")
