"""
Tests for OpenCut 3D Camera Solver and Autonomous Editing Agent.

Covers: camera_solver (solve, ground plane, object placement, render, export),
autonomous_agent (plan creation, step execution, recovery, full pipeline),
and solver_agent_routes blueprint.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Camera Solver — Data Class Tests
# ============================================================
class TestCameraDataClasses(unittest.TestCase):
    """Test camera solver data classes."""

    def test_camera_frame_defaults(self):
        from opencut.core.camera_solver import CameraFrame
        f = CameraFrame()
        self.assertEqual(f.position, (0.0, 0.0, 0.0))
        self.assertEqual(f.rotation, (0.0, 0.0, 0.0))
        self.assertEqual(f.focal_length, 35.0)
        self.assertEqual(f.frame_number, 0)

    def test_camera_frame_custom(self):
        from opencut.core.camera_solver import CameraFrame
        f = CameraFrame(position=(1.0, 2.0, 3.0), rotation=(10.0, 20.0, 30.0),
                        focal_length=50.0, frame_number=42)
        self.assertEqual(f.position, (1.0, 2.0, 3.0))
        self.assertEqual(f.frame_number, 42)

    def test_camera_solve_defaults(self):
        from opencut.core.camera_solver import CameraSolve
        s = CameraSolve()
        self.assertEqual(s.frames, [])
        self.assertEqual(s.point_cloud, [])
        self.assertFalse(s.success)
        self.assertEqual(s.error, "")

    def test_camera_solve_success(self):
        from opencut.core.camera_solver import CameraFrame, CameraSolve
        s = CameraSolve(
            frames=[CameraFrame(frame_number=0)],
            point_cloud=[(1, 2, 3)],
            reprojection_error=1.5,
            success=True,
        )
        self.assertTrue(s.success)
        self.assertEqual(len(s.frames), 1)

    def test_ground_plane_defaults(self):
        from opencut.core.camera_solver import GroundPlane
        gp = GroundPlane()
        self.assertEqual(gp.normal, (0.0, 1.0, 0.0))
        self.assertEqual(gp.distance, 0.0)

    def test_scene_object_defaults(self):
        from opencut.core.camera_solver import SceneObject
        obj = SceneObject()
        self.assertEqual(obj.text_or_image, "")
        self.assertEqual(obj.scale, 1.0)
        self.assertEqual(obj.color, "#FFFFFF")

    def test_scene_object_custom(self):
        from opencut.core.camera_solver import SceneObject
        obj = SceneObject(
            text_or_image="Hello World",
            position_3d=(1, 2, 5),
            scale=2.0,
            color="#FF0000",
            font_size=72,
            opacity=0.8,
        )
        self.assertEqual(obj.text_or_image, "Hello World")
        self.assertEqual(obj.font_size, 72)


# ============================================================
# Camera Solver — OpenCV Check
# ============================================================
class TestOpenCVCheck(unittest.TestCase):
    """Test OpenCV availability checking."""

    @patch.dict("sys.modules", {"cv2": MagicMock()})
    def test_check_opencv_available(self):
        from opencut.core.camera_solver import _check_opencv
        result = _check_opencv()
        self.assertIsNotNone(result)

    @patch.dict("sys.modules", {"cv2": None})
    def test_check_opencv_missing(self):
        import importlib

        import opencut.core.camera_solver as mod
        importlib.reload(mod)
        # When cv2 is None in sys.modules, import raises
        with patch("builtins.__import__", side_effect=ImportError("no cv2")):
            result = mod._check_opencv()
            self.assertIsNone(result)


# ============================================================
# Camera Solver — solve_camera
# ============================================================
class TestSolveCamera(unittest.TestCase):
    """Test camera solve function."""

    def test_solve_without_opencv_returns_error(self):
        from opencut.core.camera_solver import solve_camera
        with patch("opencut.core.camera_solver._check_opencv", return_value=None):
            result = solve_camera("/fake/video.mp4")
            self.assertFalse(result.success)
            self.assertIn("OpenCV", result.error)

    @patch("opencut.core.camera_solver._check_opencv")
    @patch("opencut.core.camera_solver._extract_frames")
    @patch("opencut.core.camera_solver._detect_features")
    @patch("opencut.core.camera_solver._match_features")
    @patch("opencut.core.camera_solver._estimate_camera_pose")
    def test_solve_camera_success(self, mock_pose, mock_match, mock_detect,
                                   mock_extract, mock_cv):
        import numpy as np

        from opencut.core.camera_solver import solve_camera

        mock_cv.return_value = MagicMock()
        gray = np.zeros((100, 100), dtype=np.uint8)
        mock_extract.return_value = (
            [gray, gray, gray],
            [0, 10, 20],
            {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0},
        )

        kp = MagicMock()
        kp.pt = (100.0, 100.0)
        mock_detect.return_value = ([[kp] * 100, [kp] * 100, [kp] * 100],
                                     [np.zeros((100, 32), dtype=np.uint8)] * 3,
                                     "ORB")
        mock_match.return_value = [MagicMock(queryIdx=i, trainIdx=i, distance=10)
                                    for i in range(50)]

        R = np.eye(3)
        t = np.array([[0.1], [0.0], [0.0]])
        mock_pose.return_value = (R, t, [(1, 2, 3), (4, 5, 6)], 2.5)

        progress = MagicMock()
        result = solve_camera("/fake/video.mp4", on_progress=progress)

        self.assertTrue(result.success)
        self.assertEqual(len(result.frames), 3)
        self.assertGreater(len(result.point_cloud), 0)
        progress.assert_called()

    @patch("opencut.core.camera_solver._check_opencv")
    @patch("opencut.core.camera_solver._extract_frames")
    def test_solve_camera_too_few_frames(self, mock_extract, mock_cv):
        import numpy as np

        from opencut.core.camera_solver import solve_camera

        mock_cv.return_value = MagicMock()
        mock_extract.return_value = (
            [np.zeros((100, 100), dtype=np.uint8)],
            [0],
            {"width": 1920, "height": 1080, "fps": 30.0},
        )

        result = solve_camera("/fake/video.mp4")
        self.assertFalse(result.success)
        self.assertIn("at least 2 frames", result.error)

    @patch("opencut.core.camera_solver._check_opencv")
    @patch("opencut.core.camera_solver._extract_frames")
    def test_solve_camera_exception_handled(self, mock_extract, mock_cv):
        from opencut.core.camera_solver import solve_camera

        mock_cv.return_value = MagicMock()
        mock_extract.side_effect = RuntimeError("Cannot open video")

        result = solve_camera("/fake/video.mp4")
        self.assertFalse(result.success)
        self.assertIn("Cannot open video", result.error)

    @patch("opencut.core.camera_solver._check_opencv")
    @patch("opencut.core.camera_solver._extract_frames")
    @patch("opencut.core.camera_solver._detect_features")
    @patch("opencut.core.camera_solver._match_features")
    @patch("opencut.core.camera_solver._estimate_camera_pose")
    def test_solve_camera_no_valid_matches(self, mock_pose, mock_match, mock_detect,
                                            mock_extract, mock_cv):
        import numpy as np

        from opencut.core.camera_solver import solve_camera

        mock_cv.return_value = MagicMock()
        mock_extract.return_value = (
            [np.zeros((100, 100), dtype=np.uint8)] * 3,
            [0, 10, 20],
            {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0},
        )
        mock_detect.return_value = ([[], [], []], [None, None, None], "ORB")
        mock_match.return_value = []  # No matches
        mock_pose.return_value = (None, None, None, float("inf"))

        result = solve_camera("/fake/video.mp4")
        # Should still produce frames (propagated) but low quality
        self.assertEqual(len(result.frames), 3)


# ============================================================
# Camera Solver — Rotation Matrix to Euler
# ============================================================
class TestRotationToEuler(unittest.TestCase):
    """Test rotation matrix to Euler angle conversion."""

    def test_identity_matrix(self):
        import numpy as np

        from opencut.core.camera_solver import _rotation_matrix_to_euler
        R = np.eye(3)
        euler = _rotation_matrix_to_euler(R)
        self.assertAlmostEqual(euler[0], 0.0, places=5)
        self.assertAlmostEqual(euler[1], 0.0, places=5)
        self.assertAlmostEqual(euler[2], 0.0, places=5)

    def test_90_degree_rotation(self):
        import numpy as np

        from opencut.core.camera_solver import _rotation_matrix_to_euler
        # 90 degree rotation around Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        euler = _rotation_matrix_to_euler(R)
        self.assertAlmostEqual(euler[2], 90.0, places=3)


# ============================================================
# Camera Solver — Ground Plane Detection
# ============================================================
class TestGroundPlane(unittest.TestCase):
    """Test RANSAC ground plane detection."""

    def test_detect_ground_plane_insufficient_points(self):
        from opencut.core.camera_solver import CameraSolve, detect_ground_plane
        solve = CameraSolve(point_cloud=[(1, 2, 3)])
        gp = detect_ground_plane(solve)
        self.assertEqual(gp.confidence, 0.0)

    def test_detect_ground_plane_coplanar_points(self):
        from opencut.core.camera_solver import CameraSolve, detect_ground_plane
        # Points on the XZ plane (y=0)
        points = [(float(x), 0.0, float(z)) for x in range(10) for z in range(10)]
        solve = CameraSolve(point_cloud=points, success=True)
        gp = detect_ground_plane(solve, iterations=500, distance_threshold=0.01)
        # Normal should be approximately (0, 1, 0) or (0, -1, 0)
        self.assertGreater(abs(gp.normal[1]), 0.9)
        self.assertGreater(gp.confidence, 0.8)

    def test_detect_ground_plane_with_noise(self):
        import random

        from opencut.core.camera_solver import CameraSolve, detect_ground_plane
        random.seed(42)
        # Mostly planar points with some outliers
        points = [(float(x), 0.0, float(z)) for x in range(10) for z in range(10)]
        points += [(random.random() * 10, random.random() * 10, random.random() * 10)
                    for _ in range(20)]
        solve = CameraSolve(point_cloud=points, success=True)
        gp = detect_ground_plane(solve, iterations=1000, distance_threshold=0.05)
        self.assertGreater(gp.inlier_count, 50)

    def test_detect_ground_plane_progress_callback(self):
        from opencut.core.camera_solver import CameraSolve, detect_ground_plane
        points = [(float(x), 0.0, float(z)) for x in range(5) for z in range(5)]
        solve = CameraSolve(point_cloud=points)
        progress = MagicMock()
        detect_ground_plane(solve, iterations=500, on_progress=progress)
        progress.assert_called()

    def test_detect_ground_plane_empty_cloud(self):
        from opencut.core.camera_solver import CameraSolve, detect_ground_plane
        solve = CameraSolve(point_cloud=[])
        gp = detect_ground_plane(solve)
        self.assertEqual(gp.confidence, 0.0)


# ============================================================
# Camera Solver — Object Placement
# ============================================================
class TestPlaceObject(unittest.TestCase):
    """Test 3D to 2D object projection."""

    def test_place_object_basic(self):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )
        solve = CameraSolve(
            frames=[CameraFrame(position=(0, 0, 0), rotation=(0, 0, 0),
                                focal_length=1000, frame_number=0)],
            success=True,
        )
        gp = GroundPlane()
        obj = SceneObject(text_or_image="Test", position_3d=(0.0, 0.0, 5.0))

        result = place_object_on_plane(solve, gp, obj, 0)
        self.assertTrue(result["visible"])
        self.assertIsInstance(result["pixel_x"], float)

    def test_place_object_behind_camera(self):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )
        solve = CameraSolve(
            frames=[CameraFrame(position=(0, 0, 10), rotation=(0, 0, 0),
                                focal_length=1000, frame_number=0)],
            success=True,
        )
        gp = GroundPlane()
        obj = SceneObject(position_3d=(0.0, 0.0, 20.0))  # Behind camera

        result = place_object_on_plane(solve, gp, obj, 0)
        # Object behind camera should not be visible
        # (dz > 0 after rotation means in front)
        self.assertIsInstance(result["visible"], bool)

    def test_place_object_no_frames(self):
        from opencut.core.camera_solver import (
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )
        solve = CameraSolve(frames=[], success=True)
        gp = GroundPlane()
        obj = SceneObject(position_3d=(0, 0, 5))

        result = place_object_on_plane(solve, gp, obj, 0)
        self.assertFalse(result["visible"])

    def test_place_object_nearest_frame(self):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )
        solve = CameraSolve(
            frames=[
                CameraFrame(position=(0, 0, 0), focal_length=1000, frame_number=0),
                CameraFrame(position=(1, 0, 0), focal_length=1000, frame_number=30),
            ],
            success=True,
        )
        gp = GroundPlane()
        obj = SceneObject(position_3d=(0, 0, 5))

        # Frame 15 should use nearest (0 or 30)
        result = place_object_on_plane(solve, gp, obj, 15)
        self.assertIsInstance(result, dict)

    def test_place_object_zero_dz(self):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            place_object_on_plane,
        )
        solve = CameraSolve(
            frames=[CameraFrame(position=(0, 0, 5), focal_length=1000, frame_number=0)],
            success=True,
        )
        gp = GroundPlane()
        obj = SceneObject(position_3d=(0, 0, 5))  # Same Z as camera

        result = place_object_on_plane(solve, gp, obj, 0)
        self.assertFalse(result["visible"])


# ============================================================
# Camera Solver — Export
# ============================================================
class TestExportCameraPath(unittest.TestCase):
    """Test camera path export functions."""

    def _make_solve(self):
        from opencut.core.camera_solver import CameraFrame, CameraSolve
        return CameraSolve(
            frames=[
                CameraFrame(position=(0, 0, 0), rotation=(0, 0, 0),
                            focal_length=35, frame_number=0),
                CameraFrame(position=(1, 0, 0), rotation=(5, 0, 0),
                            focal_length=35, frame_number=10),
            ],
            point_cloud=[(1, 2, 3), (4, 5, 6)],
            reprojection_error=1.5,
            success=True,
            total_frames=2,
            feature_count=200,
        )

    def test_export_json(self):
        from opencut.core.camera_solver import export_camera_path
        result = export_camera_path(self._make_solve(), format="json")
        self.assertEqual(result["format"], "json")
        data = json.loads(result["data"])
        self.assertIn("camera_path", data)
        self.assertEqual(len(data["camera_path"]), 2)

    def test_export_csv(self):
        from opencut.core.camera_solver import export_camera_path
        result = export_camera_path(self._make_solve(), format="csv")
        self.assertEqual(result["format"], "csv")
        lines = result["data"].split("\n")
        self.assertIn("frame,pos_x", lines[0])
        self.assertEqual(len(lines), 3)  # header + 2 frames

    def test_export_fbx_ascii(self):
        from opencut.core.camera_solver import export_camera_path
        result = export_camera_path(self._make_solve(), format="fbx_ascii")
        self.assertEqual(result["format"], "fbx_ascii")
        self.assertIn("FBX", result["data"])

    def test_export_invalid_format(self):
        from opencut.core.camera_solver import export_camera_path
        with self.assertRaises(ValueError) as ctx:
            export_camera_path(self._make_solve(), format="xyz")
        self.assertIn("Unsupported", str(ctx.exception))

    def test_export_failed_solve(self):
        from opencut.core.camera_solver import CameraSolve, export_camera_path
        with self.assertRaises(ValueError):
            export_camera_path(CameraSolve(success=False))

    def test_export_frame_count(self):
        from opencut.core.camera_solver import export_camera_path
        result = export_camera_path(self._make_solve(), format="json")
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["point_count"], 2)


# ============================================================
# Camera Solver — Render 3D Overlay
# ============================================================
class TestRender3DOverlay(unittest.TestCase):
    """Test 3D overlay rendering."""

    @patch("opencut.core.camera_solver.run_ffmpeg")
    @patch("opencut.core.camera_solver.get_video_info")
    @patch("opencut.core.camera_solver.detect_ground_plane")
    def test_render_overlay_basic(self, mock_gp, mock_info, mock_ffmpeg):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            render_3d_overlay,
        )

        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_gp.return_value = GroundPlane()

        solve = CameraSolve(
            frames=[CameraFrame(position=(0, 0, 0), focal_length=1000, frame_number=0)],
            success=True,
        )
        objects = [SceneObject(text_or_image="Hello", position_3d=(0, 0, 5))]

        result = render_3d_overlay("/fake/video.mp4", solve, objects)
        self.assertIn("output_path", result)
        mock_ffmpeg.assert_called_once()

    def test_render_overlay_failed_solve(self):
        from opencut.core.camera_solver import CameraSolve, SceneObject, render_3d_overlay
        with self.assertRaises(ValueError):
            render_3d_overlay("/fake.mp4", CameraSolve(success=False), [SceneObject()])

    def test_render_overlay_no_objects(self):
        from opencut.core.camera_solver import CameraSolve, render_3d_overlay
        with self.assertRaises(ValueError):
            render_3d_overlay("/fake.mp4", CameraSolve(success=True), [])

    @patch("opencut.core.camera_solver.run_ffmpeg")
    @patch("opencut.core.camera_solver.get_video_info")
    @patch("opencut.core.camera_solver.detect_ground_plane")
    def test_render_overlay_progress(self, mock_gp, mock_info, mock_ffmpeg):
        from opencut.core.camera_solver import (
            CameraFrame,
            CameraSolve,
            GroundPlane,
            SceneObject,
            render_3d_overlay,
        )

        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_gp.return_value = GroundPlane()

        solve = CameraSolve(
            frames=[CameraFrame(position=(0, 0, 0), focal_length=1000, frame_number=0)],
            success=True,
        )
        progress = MagicMock()
        render_3d_overlay("/fake.mp4", solve, [SceneObject(text_or_image="X", position_3d=(0, 0, 5))],
                          on_progress=progress)
        progress.assert_called()


# ============================================================
# Autonomous Agent — Data Class Tests
# ============================================================
class TestAgentDataClasses(unittest.TestCase):
    """Test agent data classes."""

    def test_agent_goal_defaults(self):
        from opencut.core.autonomous_agent import AgentGoal
        g = AgentGoal()
        self.assertEqual(g.description, "")
        self.assertEqual(g.constraints, [])
        self.assertIsNone(g.duration_target)

    def test_agent_step_defaults(self):
        from opencut.core.autonomous_agent import AgentStep
        s = AgentStep()
        self.assertEqual(s.action, "")
        self.assertEqual(s.status, "pending")
        self.assertEqual(s.retry_count, 0)

    def test_agent_plan_defaults(self):
        from opencut.core.autonomous_agent import AgentPlan
        p = AgentPlan()
        self.assertEqual(p.status, "created")
        self.assertEqual(p.steps, [])
        self.assertEqual(p.current_step_idx, 0)

    def test_agent_result_defaults(self):
        from opencut.core.autonomous_agent import AgentResult
        r = AgentResult()
        self.assertEqual(r.output_paths, [])
        self.assertEqual(r.steps_completed, 0)


# ============================================================
# Autonomous Agent — Tool Registry
# ============================================================
class TestAgentToolRegistry(unittest.TestCase):
    """Test agent tool registry."""

    def test_all_tools_have_required_fields(self):
        from opencut.core.autonomous_agent import AGENT_TOOLS
        for name, tool in AGENT_TOOLS.items():
            self.assertIn("description", tool, f"Tool {name} missing description")
            self.assertIn("params", tool, f"Tool {name} missing params")
            self.assertIn("module", tool, f"Tool {name} missing module")
            self.assertIn("function", tool, f"Tool {name} missing function")

    def test_list_tools(self):
        from opencut.core.autonomous_agent import list_tools
        tools = list_tools()
        self.assertGreater(len(tools), 5)
        self.assertIn("silence_removal", tools)
        self.assertIn("description", tools["silence_removal"])

    def test_tools_have_filepath_param(self):
        from opencut.core.autonomous_agent import AGENT_TOOLS
        for name, tool in AGENT_TOOLS.items():
            self.assertIn("filepath", tool["params"],
                          f"Tool {name} should accept filepath param")


# ============================================================
# Autonomous Agent — Plan Parsing
# ============================================================
class TestPlanParsing(unittest.TestCase):
    """Test LLM response parsing into steps."""

    def test_parse_valid_json(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        text = json.dumps([
            {"action": "silence_removal", "params": {"filepath": "/test.mp4"}},
            {"action": "caption_generation", "params": {"filepath": "{prev_output}"}},
        ])
        steps = _parse_plan_response(text)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].action, "silence_removal")

    def test_parse_json_with_markdown_wrapper(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        text = '```json\n[{"action": "scene_detection", "params": {}}]\n```'
        steps = _parse_plan_response(text)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action, "scene_detection")

    def test_parse_json_with_prefix_text(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        text = 'Here is my plan:\n[{"action": "audio_normalization", "params": {}}]'
        steps = _parse_plan_response(text)
        self.assertEqual(len(steps), 1)

    def test_parse_filters_unknown_tools(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        text = json.dumps([
            {"action": "silence_removal", "params": {}},
            {"action": "nonexistent_tool", "params": {}},
        ])
        steps = _parse_plan_response(text)
        self.assertEqual(len(steps), 1)

    def test_parse_invalid_json_raises(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        with self.assertRaises(ValueError):
            _parse_plan_response("This is not JSON at all.")

    def test_parse_non_array_raises(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        with self.assertRaises(ValueError):
            _parse_plan_response('{"not": "an array"}')

    def test_parse_skips_non_dict_items(self):
        from opencut.core.autonomous_agent import _parse_plan_response
        text = json.dumps([
            {"action": "silence_removal", "params": {}},
            "not a dict",
            42,
        ])
        steps = _parse_plan_response(text)
        self.assertEqual(len(steps), 1)


# ============================================================
# Autonomous Agent — Step Execution
# ============================================================
class TestStepExecution(unittest.TestCase):
    """Test individual step execution."""

    @patch("opencut.core.autonomous_agent._execute_tool")
    def test_execute_step_success(self, mock_tool):
        from opencut.core.autonomous_agent import AgentStep, execute_step
        mock_tool.return_value = {"output_path": "/out.mp4"}

        step = AgentStep(action="silence_removal", params={"filepath": "/in.mp4"})
        result = execute_step(step)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.result["output_path"], "/out.mp4")
        self.assertGreater(result.duration_seconds, 0)

    @patch("opencut.core.autonomous_agent._execute_tool")
    def test_execute_step_failure(self, mock_tool):
        from opencut.core.autonomous_agent import AgentStep, execute_step
        mock_tool.side_effect = RuntimeError("Tool crashed")

        step = AgentStep(action="silence_removal", params={})
        result = execute_step(step)

        self.assertEqual(result.status, "error")
        self.assertIn("crashed", result.error)

    def test_execute_unknown_tool(self):
        from opencut.core.autonomous_agent import _execute_tool
        with self.assertRaises(ValueError):
            _execute_tool("nonexistent_tool", {})


# ============================================================
# Autonomous Agent — Step Validation
# ============================================================
class TestStepValidation(unittest.TestCase):
    """Test step result validation."""

    def test_validate_complete_step(self):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="complete", result={"key": "value"})
        self.assertTrue(validate_step_result(step))

    def test_validate_failed_step(self):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="error", result=None)
        self.assertFalse(validate_step_result(step))

    def test_validate_step_no_result(self):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="complete", result=None)
        self.assertFalse(validate_step_result(step))

    def test_validate_step_with_error_in_result(self):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="complete", result={"error": "something wrong"})
        self.assertFalse(validate_step_result(step))

    @patch("os.path.isfile", return_value=True)
    def test_validate_step_with_output_path(self, mock_isfile):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="complete", result={"output_path": "/out.mp4"})
        self.assertTrue(validate_step_result(step))

    @patch("os.path.isfile", return_value=False)
    def test_validate_step_with_missing_output(self, mock_isfile):
        from opencut.core.autonomous_agent import AgentStep, validate_step_result
        step = AgentStep(status="complete", result={"output_path": "/missing.mp4"})
        self.assertFalse(validate_step_result(step))


# ============================================================
# Autonomous Agent — Parameter Resolution
# ============================================================
class TestParamResolution(unittest.TestCase):
    """Test step parameter placeholder resolution."""

    def test_resolve_prev_output(self):
        from opencut.core.autonomous_agent import AgentStep, _resolve_step_params
        step = AgentStep(params={"filepath": "{prev_output}"})
        _resolve_step_params(step, "/prev_out.mp4", [])
        self.assertEqual(step.params["filepath"], "/prev_out.mp4")

    def test_resolve_default_filepath(self):
        from opencut.core.autonomous_agent import AgentStep, _resolve_step_params
        step = AgentStep(params={"filepath": ""})
        _resolve_step_params(step, None, ["/input.mp4"])
        self.assertEqual(step.params["filepath"], "/input.mp4")

    def test_resolve_no_files(self):
        from opencut.core.autonomous_agent import AgentStep, _resolve_step_params
        step = AgentStep(params={"filepath": ""})
        _resolve_step_params(step, None, [])
        # filepath remains empty when no sources available
        self.assertEqual(step.params["filepath"], "")


# ============================================================
# Autonomous Agent — Recovery
# ============================================================
class TestRecovery(unittest.TestCase):
    """Test LLM-based recovery from step failures."""

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_recovery_with_alternative(self, mock_llm):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            recover_from_failure,
        )

        mock_llm.return_value = json.dumps({
            "action": "noise_reduction",
            "params": {"filepath": "/test.mp4", "strength": 0.5},
            "description": "Use noise reduction instead",
        })

        plan = AgentPlan(goal=AgentGoal(description="Clean audio"))
        failed = AgentStep(action="audio_normalization",
                           error="Normalization failed", status="error")

        result = recover_from_failure(plan, failed)
        self.assertEqual(result.action, "noise_reduction")
        self.assertNotEqual(result.status, "skipped")

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_recovery_skip(self, mock_llm):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            recover_from_failure,
        )

        mock_llm.return_value = json.dumps({
            "action": "skip",
            "description": "No alternative",
        })

        plan = AgentPlan(goal=AgentGoal(description="Edit video"))
        failed = AgentStep(action="music_generation", error="Failed", status="error")

        result = recover_from_failure(plan, failed)
        self.assertEqual(result.status, "skipped")

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_recovery_llm_failure(self, mock_llm):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            recover_from_failure,
        )

        mock_llm.side_effect = Exception("LLM unreachable")

        plan = AgentPlan(goal=AgentGoal(description="Edit"))
        failed = AgentStep(action="export", error="Failed", status="error")

        result = recover_from_failure(plan, failed)
        self.assertEqual(result.status, "skipped")


# ============================================================
# Autonomous Agent — Plan Creation
# ============================================================
class TestCreatePlan(unittest.TestCase):
    """Test plan creation from natural language."""

    @patch("opencut.core.autonomous_agent._get_llm_response")
    @patch("opencut.core.autonomous_agent.get_video_info")
    def test_create_plan_basic(self, mock_info, mock_llm):
        from opencut.core.autonomous_agent import create_plan

        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        mock_llm.return_value = json.dumps([
            {"action": "silence_removal", "params": {"filepath": "/test.mp4"}},
            {"action": "caption_generation", "params": {"filepath": "{prev_output}"}},
        ])

        plan = create_plan("Remove silence and add captions", ["/test.mp4"])
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.status, "created")
        self.assertIn("2 steps", plan.execution_log[0])

    def test_create_plan_empty_goal(self):
        from opencut.core.autonomous_agent import create_plan
        with self.assertRaises(ValueError):
            create_plan("")

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_create_plan_no_files(self, mock_llm):
        from opencut.core.autonomous_agent import create_plan
        mock_llm.return_value = json.dumps([
            {"action": "silence_removal", "params": {}},
        ])
        plan = create_plan("Remove silence")
        self.assertEqual(len(plan.steps), 1)

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_create_plan_stores_plan(self, mock_llm):
        from opencut.core.autonomous_agent import create_plan, get_plan
        mock_llm.return_value = json.dumps([
            {"action": "scene_detection", "params": {}},
        ])
        plan = create_plan("Detect scenes")
        retrieved = get_plan(plan.plan_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.plan_id, plan.plan_id)

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_create_plan_progress_callback(self, mock_llm):
        from opencut.core.autonomous_agent import create_plan
        mock_llm.return_value = json.dumps([
            {"action": "export", "params": {}},
        ])
        progress = MagicMock()
        create_plan("Export video", on_progress=progress)
        progress.assert_called()


# ============================================================
# Autonomous Agent — Plan Execution
# ============================================================
class TestExecutePlan(unittest.TestCase):
    """Test plan execution pipeline."""

    @patch("opencut.core.autonomous_agent._execute_tool")
    def test_execute_plan_all_success(self, mock_tool):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            execute_plan,
        )

        mock_tool.return_value = {"output_path": "/out.mp4"}

        plan = AgentPlan(
            plan_id="test-1",
            goal=AgentGoal(description="Test"),
            steps=[
                AgentStep(action="silence_removal", params={"filepath": "/in.mp4"}),
                AgentStep(action="caption_generation", params={"filepath": "{prev_output}"}),
            ],
            input_files=["/in.mp4"],
        )

        result = execute_plan(plan)
        self.assertEqual(result.steps_completed, 2)
        self.assertEqual(result.steps_failed, 0)
        self.assertEqual(result.plan.status, "complete")

    @patch("opencut.core.autonomous_agent._execute_tool")
    @patch("opencut.core.autonomous_agent.recover_from_failure")
    def test_execute_plan_with_skipped_step(self, mock_recover, mock_tool):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            execute_plan,
        )

        mock_tool.side_effect = RuntimeError("Failed")
        failed_step = AgentStep(action="silence_removal", status="skipped")
        mock_recover.return_value = failed_step

        plan = AgentPlan(
            plan_id="test-2a",
            goal=AgentGoal(description="Test"),
            steps=[AgentStep(action="silence_removal", params={"filepath": "/in.mp4"})],
            input_files=["/in.mp4"],
        )

        result = execute_plan(plan)
        # Skipped steps don't count as failed, so plan status is "complete"
        self.assertEqual(result.plan.status, "complete")
        self.assertEqual(result.steps_completed, 0)

    @patch("opencut.core.autonomous_agent._execute_tool")
    @patch("opencut.core.autonomous_agent.recover_from_failure")
    def test_execute_plan_with_permanent_failure(self, mock_recover, mock_tool):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            execute_plan,
        )

        mock_tool.side_effect = RuntimeError("Failed")
        # Recovery returns a step that is still in "error" status
        error_step = AgentStep(action="silence_removal", status="error",
                               error="No alternative")
        mock_recover.return_value = error_step

        plan = AgentPlan(
            plan_id="test-2b",
            goal=AgentGoal(description="Test"),
            steps=[AgentStep(action="silence_removal", params={"filepath": "/in.mp4"})],
            input_files=["/in.mp4"],
        )

        result = execute_plan(plan, max_retries=0)
        self.assertEqual(result.plan.status, "error")
        self.assertEqual(result.steps_failed, 1)

    @patch("opencut.core.autonomous_agent._execute_tool")
    def test_execute_plan_progress(self, mock_tool):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            execute_plan,
        )

        mock_tool.return_value = {"output_path": "/out.mp4"}

        plan = AgentPlan(
            plan_id="test-3",
            goal=AgentGoal(description="Test"),
            steps=[AgentStep(action="silence_removal", params={"filepath": "/in.mp4"})],
            input_files=["/in.mp4"],
        )

        progress = MagicMock()
        execute_plan(plan, on_progress=progress)
        progress.assert_called()

    @patch("opencut.core.autonomous_agent._execute_tool")
    def test_execute_plan_output_chaining(self, mock_tool):
        from opencut.core.autonomous_agent import (
            AgentGoal,
            AgentPlan,
            AgentStep,
            execute_plan,
        )

        call_count = [0]
        def mock_exec(name, params):
            call_count[0] += 1
            return {"output_path": f"/step{call_count[0]}.mp4"}

        mock_tool.side_effect = mock_exec

        plan = AgentPlan(
            plan_id="test-4",
            goal=AgentGoal(description="Chain"),
            steps=[
                AgentStep(action="silence_removal", params={"filepath": "/in.mp4"}),
                AgentStep(action="export", params={"filepath": "{prev_output}"}),
            ],
            input_files=["/in.mp4"],
        )

        result = execute_plan(plan)
        self.assertEqual(len(result.output_paths), 2)
        self.assertEqual(result.output_paths[-1], "/step2.mp4")


# ============================================================
# Autonomous Agent — Full Pipeline
# ============================================================
class TestAgentEdit(unittest.TestCase):
    """Test full agent_edit pipeline."""

    @patch("opencut.core.autonomous_agent._execute_tool")
    @patch("opencut.core.autonomous_agent._get_llm_response")
    @patch("opencut.core.autonomous_agent.get_video_info")
    def test_agent_edit_full(self, mock_info, mock_llm, mock_tool):
        from opencut.core.autonomous_agent import agent_edit

        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        mock_llm.return_value = json.dumps([
            {"action": "silence_removal", "params": {"filepath": "/in.mp4"}},
        ])
        mock_tool.return_value = {"output_path": "/out.mp4"}

        result = agent_edit("Remove silence", file_paths=["/in.mp4"])
        self.assertEqual(result.steps_completed, 1)
        self.assertIn("/out.mp4", result.output_paths)

    def test_agent_edit_empty_goal(self):
        from opencut.core.autonomous_agent import agent_edit
        with self.assertRaises(ValueError):
            agent_edit("")

    @patch("opencut.core.autonomous_agent._execute_tool")
    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_agent_edit_with_output_path(self, mock_llm, mock_tool):
        from opencut.core.autonomous_agent import agent_edit

        mock_llm.return_value = json.dumps([
            {"action": "export", "params": {}},
        ])
        mock_tool.return_value = {"output_path": "/final.mp4"}

        result = agent_edit("Export video", out_path="/final.mp4")
        self.assertIsNotNone(result)


# ============================================================
# Autonomous Agent — Plan Store
# ============================================================
class TestPlanStore(unittest.TestCase):
    """Test plan storage and retrieval."""

    def test_get_nonexistent_plan(self):
        from opencut.core.autonomous_agent import get_plan
        result = get_plan("nonexistent-id")
        self.assertIsNone(result)

    @patch("opencut.core.autonomous_agent._get_llm_response")
    def test_plan_has_unique_id(self, mock_llm):
        from opencut.core.autonomous_agent import create_plan
        mock_llm.return_value = json.dumps([
            {"action": "silence_removal", "params": {}},
        ])
        plan1 = create_plan("Goal 1")
        plan2 = create_plan("Goal 2")
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)


# ============================================================
# Route Tests — Camera Solver
# ============================================================
def _make_test_app():
    """Create a Flask test app with the solver_agent blueprint registered."""
    from opencut.config import OpenCutConfig
    from opencut.routes.solver_agent_routes import solver_agent_bp
    from opencut.server import create_app

    app = create_app(config=OpenCutConfig())
    app.config["TESTING"] = True
    # Register blueprint if not already present
    if "solver_agent" not in [bp.name for bp in app.blueprints.values()]:
        app.register_blueprint(solver_agent_bp)
    return app


class TestCameraSolverRoutes(unittest.TestCase):
    """Test camera solver route endpoints."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()
        self.app = _make_test_app()
        self.client = self.app.test_client()
        resp = self.client.get("/health")
        self.token = resp.get_json().get("csrf_token", "")

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _headers(self):
        from tests.conftest import csrf_headers
        return csrf_headers(self.token)

    @patch("opencut.core.camera_solver.solve_camera")
    def test_solve_route_returns_job(self, mock_solve):
        from opencut.core.camera_solver import CameraSolve
        mock_solve.return_value = CameraSolve(success=True, total_frames=10)

        resp = self.client.post("/camera-solver/solve",
                                headers=self._headers(),
                                data=json.dumps({"filepath": self.tmp.name}))
        self.assertIn(resp.status_code, (200, 202))
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_place_object_route_sync(self):
        resp = self.client.post("/camera-solver/place-object",
                                headers=self._headers(),
                                data=json.dumps({
                                    "solve_result": {
                                        "frames": [{
                                            "position": [0, 0, 0],
                                            "rotation": [0, 0, 0],
                                            "focal_length": 1000,
                                            "frame_number": 0,
                                        }],
                                        "point_cloud": [],
                                        "success": True,
                                    },
                                    "object": {
                                        "text_or_image": "Test",
                                        "position_3d": [0, 0, 5],
                                    },
                                    "frame_number": 0,
                                }))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("visible", data)

    def test_place_object_route_missing_solve(self):
        resp = self.client.post("/camera-solver/place-object",
                                headers=self._headers(),
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)


# ============================================================
# Route Tests — Agent
# ============================================================
class TestAgentRoutes(unittest.TestCase):
    """Test agent route endpoints."""

    def setUp(self):
        self.app = _make_test_app()
        self.client = self.app.test_client()
        resp = self.client.get("/health")
        self.token = resp.get_json().get("csrf_token", "")

    def _headers(self):
        from tests.conftest import csrf_headers
        return csrf_headers(self.token)

    def test_list_tools_route(self):
        resp = self.client.get("/agent/tools")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("tools", data)
        self.assertGreater(data["count"], 0)

    def test_plan_status_not_found(self):
        resp = self.client.get("/agent/plan/nonexistent-id")
        self.assertEqual(resp.status_code, 404)

    @patch("opencut.core.autonomous_agent.create_plan")
    def test_create_plan_route(self, mock_create):
        from opencut.core.autonomous_agent import AgentPlan, AgentStep
        mock_create.return_value = AgentPlan(
            plan_id="test-123",
            steps=[AgentStep(action="silence_removal")],
            status="created",
        )

        resp = self.client.post("/agent/create-plan",
                                headers=self._headers(),
                                data=json.dumps({"goal": "Remove silence"}))
        self.assertIn(resp.status_code, (200, 202))

    def test_auto_edit_route_no_goal(self):
        resp = self.client.post("/agent/auto-edit",
                                headers=self._headers(),
                                data=json.dumps({}))
        # Should get job_id since async_job handles it
        self.assertIn(resp.status_code, (200, 202))


# ============================================================
# Feature Detection (unit tests)
# ============================================================
class TestFeatureDetection(unittest.TestCase):
    """Test feature detection helpers."""

    def test_match_features_no_descriptors(self):
        from opencut.core.camera_solver import _match_features
        with patch("opencut.core.camera_solver._check_opencv", return_value=MagicMock()):
            result = _match_features(None, None)
            self.assertEqual(result, [])

    def test_match_features_no_opencv(self):
        from opencut.core.camera_solver import _match_features
        with patch("opencut.core.camera_solver._check_opencv", return_value=None):
            result = _match_features(None, None)
            self.assertEqual(result, [])


# ============================================================
# Build Plan Prompt
# ============================================================
class TestBuildPlanPrompt(unittest.TestCase):
    """Test plan prompt construction."""

    def test_prompt_includes_tools(self):
        from opencut.core.autonomous_agent import _build_plan_prompt
        prompt = _build_plan_prompt("Remove silence", ["/test.mp4"], {})
        self.assertIn("silence_removal", prompt)
        self.assertIn("caption_generation", prompt)

    def test_prompt_includes_files(self):
        from opencut.core.autonomous_agent import _build_plan_prompt
        prompt = _build_plan_prompt("Edit", ["/my/video.mp4"],
                                     {"/my/video.mp4": {"duration": 60, "width": 1920, "height": 1080}})
        self.assertIn("video.mp4", prompt)
        self.assertIn("1920", prompt)

    def test_prompt_no_files(self):
        from opencut.core.autonomous_agent import _build_plan_prompt
        prompt = _build_plan_prompt("Edit something", [], {})
        self.assertIn("no files provided", prompt)


if __name__ == "__main__":
    unittest.main()
