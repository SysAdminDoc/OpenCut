"""
OpenCut Camera Solver & Autonomous Agent Routes

Endpoints for 3D camera solving, ground plane detection, 3D overlay
rendering, and the LLM-powered autonomous editing agent.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
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

solver_agent_bp = Blueprint("solver_agent", __name__)


# ---------------------------------------------------------------------------
# Camera Solver — Solve camera motion
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/camera-solver/solve", methods=["POST"])
@require_csrf
@async_job("camera_solve")
def camera_solve(job_id, filepath, data):
    """Solve camera motion from video footage."""
    from opencut.core.camera_solver import solve_camera

    start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
    end_frame = data.get("end_frame")
    if end_frame is not None:
        end_frame = safe_int(end_frame, None, min_val=0)
    method = data.get("method", "ORB").strip().upper()
    if method not in ("ORB", "SIFT"):
        method = "ORB"
    max_frames = safe_int(data.get("max_frames", 300), 300, min_val=10, max_val=5000)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = solve_camera(
        video_path=filepath,
        start_frame=start_frame,
        end_frame=end_frame,
        method=method,
        max_frames=max_frames,
        on_progress=_progress,
    )

    return {
        "success": result.success,
        "total_frames": result.total_frames,
        "feature_count": result.feature_count,
        "reprojection_error": result.reprojection_error,
        "point_cloud_size": len(result.point_cloud),
        "error": result.error,
        "frames": [
            {
                "frame_number": f.frame_number,
                "position": list(f.position),
                "rotation": list(f.rotation),
                "focal_length": f.focal_length,
            }
            for f in result.frames[:100]  # Cap frames in response
        ],
    }


# ---------------------------------------------------------------------------
# Camera Solver — Detect ground plane
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/camera-solver/ground-plane", methods=["POST"])
@require_csrf
@async_job("ground_plane")
def camera_ground_plane(job_id, filepath, data):
    """Detect ground plane from a camera solve result."""
    from opencut.core.camera_solver import (
        CameraFrame,
        CameraSolve,
        detect_ground_plane,
    )

    solve_data = data.get("solve_result")
    if not solve_data or not isinstance(solve_data, dict):
        raise ValueError("No solve_result provided")

    # Reconstruct CameraSolve from JSON
    frames = []
    for f in solve_data.get("frames", []):
        frames.append(CameraFrame(
            position=tuple(f.get("position", [0, 0, 0])),
            rotation=tuple(f.get("rotation", [0, 0, 0])),
            focal_length=f.get("focal_length", 35.0),
            frame_number=f.get("frame_number", 0),
        ))

    point_cloud = [tuple(p) for p in solve_data.get("point_cloud", [])]

    solve_result = CameraSolve(
        frames=frames,
        point_cloud=point_cloud,
        reprojection_error=solve_data.get("reprojection_error", 0.0),
        success=solve_data.get("success", True),
    )

    iterations = safe_int(data.get("iterations", 1000), 1000, min_val=100, max_val=10000)
    distance_threshold = safe_float(data.get("distance_threshold", 0.1), 0.1,
                                     min_val=0.001, max_val=10.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_ground_plane(
        solve_result=solve_result,
        iterations=iterations,
        distance_threshold=distance_threshold,
        on_progress=_progress,
    )

    return {
        "normal": list(result.normal),
        "distance": result.distance,
        "origin": list(result.origin),
        "inlier_count": result.inlier_count,
        "confidence": result.confidence,
    }


# ---------------------------------------------------------------------------
# Camera Solver — Place object on plane
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/camera-solver/place-object", methods=["POST"])
@require_csrf
def camera_place_object():
    """Project a 3D object position to 2D pixel coordinates."""
    try:
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )

        data = request.get_json(force=True) or {}

        solve_data = data.get("solve_result")
        if not solve_data or not isinstance(solve_data, dict):
            return jsonify({"error": "No solve_result provided"}), 400

        frames = []
        for f in solve_data.get("frames", []):
            frames.append(CameraFrame(
                position=tuple(f.get("position", [0, 0, 0])),
                rotation=tuple(f.get("rotation", [0, 0, 0])),
                focal_length=f.get("focal_length", 35.0),
                frame_number=f.get("frame_number", 0),
            ))

        solve_result = CameraSolve(
            frames=frames,
            point_cloud=[tuple(p) for p in solve_data.get("point_cloud", [])],
            success=solve_data.get("success", True),
        )

        gp_data = data.get("ground_plane", {})
        ground_plane = GroundPlane(
            normal=tuple(gp_data.get("normal", [0, 1, 0])),
            distance=gp_data.get("distance", 0.0),
            origin=tuple(gp_data.get("origin", [0, 0, 0])),
        )

        obj_data = data.get("object", {})
        scene_object = SceneObject(
            text_or_image=obj_data.get("text_or_image", ""),
            position_3d=tuple(obj_data.get("position_3d", [0, 0, 5])),
            scale=safe_float(obj_data.get("scale", 1.0), 1.0, min_val=0.01),
            rotation=tuple(obj_data.get("rotation", [0, 0, 0])),
        )

        frame_number = safe_int(data.get("frame_number", 0), 0, min_val=0)

        result = place_object_on_plane(
            solve_result=solve_result,
            ground_plane=ground_plane,
            object_config=scene_object,
            frame_number=frame_number,
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "camera_place_object")


# ---------------------------------------------------------------------------
# Camera Solver — Render 3D overlay
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/camera-solver/render", methods=["POST"])
@require_csrf
@async_job("camera_render")
def camera_render(job_id, filepath, data):
    """Render 3D-tracked objects composited into video."""
    from opencut.core.camera_solver import (
        CameraFrame,
        CameraSolve,
        SceneObject,
        render_3d_overlay,
    )

    solve_data = data.get("solve_result")
    if not solve_data or not isinstance(solve_data, dict):
        raise ValueError("No solve_result provided")

    frames = []
    for f in solve_data.get("frames", []):
        frames.append(CameraFrame(
            position=tuple(f.get("position", [0, 0, 0])),
            rotation=tuple(f.get("rotation", [0, 0, 0])),
            focal_length=f.get("focal_length", 35.0),
            frame_number=f.get("frame_number", 0),
        ))

    solve_result = CameraSolve(
        frames=frames,
        point_cloud=[tuple(p) for p in solve_data.get("point_cloud", [])],
        reprojection_error=solve_data.get("reprojection_error", 0.0),
        success=solve_data.get("success", True),
    )

    objects_data = data.get("objects", [])
    objects = []
    for obj in objects_data:
        objects.append(SceneObject(
            text_or_image=obj.get("text_or_image", ""),
            position_3d=tuple(obj.get("position_3d", [0, 0, 5])),
            scale=safe_float(obj.get("scale", 1.0), 1.0, min_val=0.01),
            rotation=tuple(obj.get("rotation", [0, 0, 0])),
            color=obj.get("color", "#FFFFFF"),
            font_size=safe_int(obj.get("font_size", 48), 48, min_val=8, max_val=200),
            opacity=safe_float(obj.get("opacity", 1.0), 1.0, min_val=0.0, max_val=1.0),
        ))

    output_dir = data.get("output_dir", "")
    out = data.get("output_path", None) or None

    if out is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        out = _op(filepath, "3d_overlay", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_3d_overlay(
        video_path=filepath,
        solve_result=solve_result,
        objects=objects,
        out_path=out,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Camera Solver — Export camera path
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/camera-solver/export", methods=["POST"])
@require_csrf
@async_job("camera_export", filepath_required=False)
def camera_export(job_id, filepath, data):
    """Export camera path data for use in 3D software."""
    from opencut.core.camera_solver import (
        CameraFrame,
        CameraSolve,
        export_camera_path,
    )

    solve_data = data.get("solve_result")
    if not solve_data or not isinstance(solve_data, dict):
        raise ValueError("No solve_result provided")

    frames = []
    for f in solve_data.get("frames", []):
        frames.append(CameraFrame(
            position=tuple(f.get("position", [0, 0, 0])),
            rotation=tuple(f.get("rotation", [0, 0, 0])),
            focal_length=f.get("focal_length", 35.0),
            frame_number=f.get("frame_number", 0),
        ))

    solve_result = CameraSolve(
        frames=frames,
        point_cloud=[tuple(p) for p in solve_data.get("point_cloud", [])],
        reprojection_error=solve_data.get("reprojection_error", 0.0),
        success=solve_data.get("success", True),
        total_frames=len(frames),
        feature_count=solve_data.get("feature_count", 0),
    )

    export_format = data.get("format", "json").strip().lower()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Exporting camera path")

    result = export_camera_path(
        solve_result=solve_result,
        format=export_format,
    )

    _progress(100, "Export complete")
    return result


# ---------------------------------------------------------------------------
# Agent — Create plan
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/agent/create-plan", methods=["POST"])
@require_csrf
@async_job("agent_plan", filepath_required=False)
def agent_create_plan(job_id, filepath, data):
    """Create an editing plan from natural language instructions."""
    from opencut.core.autonomous_agent import create_plan

    goal_text = data.get("goal", "").strip()
    if not goal_text:
        raise ValueError("No goal text provided")

    file_paths = data.get("file_paths", [])
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Validate file paths
    validated_paths = []
    for fp in file_paths:
        if fp and fp.strip():
            validated_paths.append(validate_filepath(fp.strip()))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    plan = create_plan(
        goal_text=goal_text,
        available_files=validated_paths,
        on_progress=_progress,
    )

    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "step_count": len(plan.steps),
        "steps": [
            {
                "action": s.action,
                "params": s.params,
                "status": s.status,
            }
            for s in plan.steps
        ],
        "goal": plan.goal.description,
    }


# ---------------------------------------------------------------------------
# Agent — Execute plan
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/agent/execute-plan", methods=["POST"])
@require_csrf
@async_job("agent_execute", filepath_required=False)
def agent_execute_plan(job_id, filepath, data):
    """Execute an existing editing plan."""
    from opencut.core.autonomous_agent import execute_plan, get_plan

    plan_id = data.get("plan_id", "").strip()
    if not plan_id:
        raise ValueError("No plan_id provided")

    plan = get_plan(plan_id)
    if plan is None:
        raise ValueError(f"Plan not found: {plan_id}")

    max_retries = safe_int(data.get("max_retries", 1), 1, min_val=0, max_val=5)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = execute_plan(
        plan=plan,
        max_retries=max_retries,
        on_progress=_progress,
    )

    return {
        "plan_id": result.plan.plan_id,
        "status": result.plan.status,
        "steps_completed": result.steps_completed,
        "steps_failed": result.steps_failed,
        "total_duration": result.total_duration,
        "output_paths": result.output_paths,
        "summary": result.summary,
        "execution_log": result.plan.execution_log,
    }


# ---------------------------------------------------------------------------
# Agent — Full auto-edit pipeline
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/agent/auto-edit", methods=["POST"])
@require_csrf
@async_job("agent_auto_edit", filepath_required=False)
def agent_auto_edit(job_id, filepath, data):
    """Full automatic edit pipeline from natural language."""
    from opencut.core.autonomous_agent import agent_edit

    goal_text = data.get("goal", "").strip()
    if not goal_text:
        raise ValueError("No goal text provided")

    file_paths = data.get("file_paths", [])
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    validated_paths = []
    for fp in file_paths:
        if fp and fp.strip():
            validated_paths.append(validate_filepath(fp.strip()))

    out_path = data.get("output_path", None)
    if out_path:
        out_path = validate_output_path(out_path)
    if out_path:
        out_path = out_path.strip() or None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = agent_edit(
        goal_text=goal_text,
        file_paths=validated_paths,
        out_path=out_path,
        on_progress=_progress,
    )

    return {
        "plan_id": result.plan.plan_id,
        "status": result.plan.status,
        "steps_completed": result.steps_completed,
        "steps_failed": result.steps_failed,
        "total_duration": result.total_duration,
        "output_paths": result.output_paths,
        "summary": result.summary,
        "execution_log": result.plan.execution_log,
    }


# ---------------------------------------------------------------------------
# Agent — Get plan status
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/agent/plan/<plan_id>", methods=["GET"])
def agent_plan_status(plan_id):
    """Get the current status of an editing plan."""
    try:
        from opencut.core.autonomous_agent import get_plan

        plan = get_plan(plan_id)
        if plan is None:
            return jsonify({"error": f"Plan not found: {plan_id}"}), 404

        return jsonify({
            "plan_id": plan.plan_id,
            "status": plan.status,
            "goal": plan.goal.description,
            "current_step": plan.current_step_idx,
            "total_steps": len(plan.steps),
            "steps": [
                {
                    "action": s.action,
                    "status": s.status,
                    "error": s.error,
                    "duration_seconds": s.duration_seconds,
                }
                for s in plan.steps
            ],
            "execution_log": plan.execution_log,
        })

    except Exception as e:
        return safe_error(e, "agent_plan_status")


# ---------------------------------------------------------------------------
# Agent — List available tools
# ---------------------------------------------------------------------------
@solver_agent_bp.route("/agent/tools", methods=["GET"])
def agent_list_tools():
    """List all tools available to the editing agent."""
    try:
        from opencut.core.autonomous_agent import list_tools

        tools = list_tools()
        return jsonify({
            "tools": tools,
            "count": len(tools),
        })

    except Exception as e:
        return safe_error(e, "agent_list_tools")
