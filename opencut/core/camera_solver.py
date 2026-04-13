"""
OpenCut 3D Camera Solver

Solves camera motion from footage to determine 3D camera path, enabling
addition of text/objects locked to the 3D scene.

Uses OpenCV for feature detection (ORB/SIFT), essential matrix estimation
via cv2.recoverPose, and RANSAC-based ground plane fitting from
reconstructed 3D points.

Requires: opencv-python (optional, returns error if missing)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraFrame:
    """Camera pose for a single frame."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    focal_length: float = 35.0
    frame_number: int = 0


@dataclass
class CameraSolve:
    """Complete camera solve result."""
    frames: List[CameraFrame] = field(default_factory=list)
    point_cloud: List[Tuple[float, float, float]] = field(default_factory=list)
    reprojection_error: float = 0.0
    success: bool = False
    error: str = ""
    total_frames: int = 0
    feature_count: int = 0


@dataclass
class GroundPlane:
    """Detected ground plane in 3D scene."""
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    distance: float = 0.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    inlier_count: int = 0
    confidence: float = 0.0


@dataclass
class SceneObject:
    """Object to place in 3D scene."""
    text_or_image: str = ""
    position_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: str = "#FFFFFF"
    font_size: int = 48
    opacity: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _check_opencv():
    """Check if OpenCV is available, return module or None."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _extract_frames(video_path, start_frame=0, end_frame=None, max_frames=300,
                    on_progress=None):
    """Extract frames from video as numpy arrays via FFmpeg + OpenCV."""
    cv2 = _check_opencv()
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for camera solving. "
            "Install with: pip install opencv-python"
        )

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0) or 30.0
    duration = info.get("duration", 0.0) or 0.0
    total_frames = int(duration * fps)

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame < 0:
        start_frame = 0

    frame_range = end_frame - start_frame
    step = max(1, frame_range // max_frames)

    if on_progress:
        on_progress(5, "Opening video for frame extraction")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    frame_numbers = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame
        while idx < end_frame and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if (idx - start_frame) % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                frame_numbers.append(idx)
            idx += 1
            if on_progress and len(frames) % 20 == 0:
                pct = 5 + int(15 * len(frames) / max_frames)
                on_progress(pct, f"Extracted {len(frames)} frames")
    finally:
        cap.release()

    return frames, frame_numbers, info


def _detect_features(frames, method="ORB", on_progress=None):
    """Detect and describe features in all frames."""
    cv2 = _check_opencv()
    if cv2 is None:
        raise RuntimeError("OpenCV required for feature detection")

    if method.upper() == "SIFT":
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
        except AttributeError:
            detector = cv2.ORB_create(nFeatures=2000)
            method = "ORB"
    else:
        detector = cv2.ORB_create(nFeatures=2000)

    all_keypoints = []
    all_descriptors = []

    for i, frame in enumerate(frames):
        kps, desc = detector.detectAndCompute(frame, None)
        all_keypoints.append(kps)
        all_descriptors.append(desc)
        if on_progress and i % 10 == 0:
            pct = 20 + int(20 * i / len(frames))
            on_progress(pct, f"Detecting features: frame {i + 1}/{len(frames)}")

    return all_keypoints, all_descriptors, method


def _match_features(desc1, desc2, method="ORB"):
    """Match features between two frames."""
    cv2 = _check_opencv()
    if cv2 is None:
        return []

    if desc1 is None or desc2 is None:
        return []

    if method.upper() == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        try:
            raw_matches = bf.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        # Lowe's ratio test
        good = []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        return good
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(desc1, desc2)
        except cv2.error:
            return []
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:500]


def _estimate_camera_pose(kps1, kps2, matches, focal_length, pp):
    """Estimate relative camera pose from matched features."""
    cv2 = _check_opencv()
    import numpy as np

    if len(matches) < 8:
        return None, None, None, float("inf")

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, focal_length, pp,
                                    cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None, None, None, float("inf")

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, focal=focal_length,
                                          pp=pp, mask=mask)

    # Triangulate points for error estimation
    inlier_pts1 = pts1[pose_mask.ravel() > 0]
    inlier_pts2 = pts2[pose_mask.ravel() > 0]

    if len(inlier_pts1) < 4:
        return R, t, [], float("inf")

    # Build projection matrices
    K = np.array([[focal_length, 0, pp[0]],
                  [0, focal_length, pp[1]],
                  [0, 0, 1]], dtype=np.float64)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # Filter invalid points (behind camera or too far)
    valid = []
    for p in pts3d:
        if p[2] > 0 and abs(p[0]) < 100 and abs(p[1]) < 100 and abs(p[2]) < 100:
            valid.append(tuple(p))

    # Reprojection error
    if len(valid) > 0:
        projected = P2 @ np.hstack([np.array(valid), np.ones((len(valid), 1))]).T
        projected = (projected[:2] / projected[2]).T
        errors = np.sqrt(np.sum((projected[:len(inlier_pts2)] - inlier_pts2[:len(projected)]) ** 2, axis=1))
        reproj_error = float(np.mean(errors)) if len(errors) > 0 else float("inf")
    else:
        reproj_error = float("inf")

    return R, t, valid, reproj_error


def _rotation_matrix_to_euler(R):
    """Convert 3x3 rotation matrix to Euler angles (degrees)."""

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    return (math.degrees(x), math.degrees(y), math.degrees(z))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_camera(
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    method: str = "ORB",
    max_frames: int = 300,
    on_progress: Optional[Callable] = None,
) -> CameraSolve:
    """
    Solve camera motion from footage to determine 3D camera path.

    Extracts frames, detects features, matches between consecutive frames,
    and estimates camera poses via essential matrix decomposition.

    Args:
        video_path: Source video file.
        start_frame: First frame to analyze.
        end_frame: Last frame to analyze (None = all).
        method: Feature detection method - "ORB" (fast, free) or "SIFT".
        max_frames: Maximum frames to analyze.
        on_progress: Progress callback(pct, msg).

    Returns:
        CameraSolve with camera frames and point cloud.
    """
    cv2 = _check_opencv()
    if cv2 is None:
        return CameraSolve(
            success=False,
            error="OpenCV is required for camera solving. "
                  "Install with: pip install opencv-python",
        )

    import numpy as np

    if on_progress:
        on_progress(1, "Starting camera solve")

    try:
        # Step 1: Extract frames
        frames, frame_numbers, info = _extract_frames(
            video_path, start_frame, end_frame, max_frames, on_progress
        )

        if len(frames) < 2:
            return CameraSolve(
                success=False,
                error="Need at least 2 frames for camera solve",
            )

        # Step 2: Detect features
        all_kps, all_descs, used_method = _detect_features(
            frames, method, on_progress
        )

        # Step 3: Match features between consecutive frames and estimate poses
        width = info.get("width", 1920) or 1920
        height = info.get("height", 1080) or 1080
        focal_length = max(width, height) * 0.8
        pp = (width / 2.0, height / 2.0)

        camera_frames = []
        all_points = []
        total_reproj_error = 0.0
        valid_pairs = 0

        # First frame is at origin
        camera_frames.append(CameraFrame(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            focal_length=focal_length,
            frame_number=frame_numbers[0],
        ))

        cumulative_R = np.eye(3)
        cumulative_t = np.zeros((3, 1))

        total_features = sum(
            len(kps) for kps in all_kps if kps is not None
        )

        for i in range(1, len(frames)):
            if on_progress:
                pct = 40 + int(40 * i / len(frames))
                on_progress(pct, f"Solving pose: frame {i + 1}/{len(frames)}")

            matches = _match_features(all_descs[i - 1], all_descs[i], used_method)

            if len(matches) < 8:
                # Not enough matches, propagate previous pose
                prev = camera_frames[-1]
                camera_frames.append(CameraFrame(
                    position=prev.position,
                    rotation=prev.rotation,
                    focal_length=focal_length,
                    frame_number=frame_numbers[i],
                ))
                continue

            R, t, points_3d, reproj_err = _estimate_camera_pose(
                all_kps[i - 1], all_kps[i], matches, focal_length, pp
            )

            if R is not None:
                cumulative_R = R @ cumulative_R
                cumulative_t = R @ cumulative_t + t

                pos = tuple(-cumulative_R.T @ cumulative_t.flatten())
                rot = _rotation_matrix_to_euler(cumulative_R)

                camera_frames.append(CameraFrame(
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    rotation=(float(rot[0]), float(rot[1]), float(rot[2])),
                    focal_length=focal_length,
                    frame_number=frame_numbers[i],
                ))

                all_points.extend(points_3d)

                if reproj_err < float("inf"):
                    total_reproj_error += reproj_err
                    valid_pairs += 1
            else:
                prev = camera_frames[-1]
                camera_frames.append(CameraFrame(
                    position=prev.position,
                    rotation=prev.rotation,
                    focal_length=focal_length,
                    frame_number=frame_numbers[i],
                ))

        avg_reproj = (total_reproj_error / valid_pairs) if valid_pairs > 0 else float("inf")
        success = valid_pairs > 0 and avg_reproj < 50.0

        if on_progress:
            on_progress(90, "Camera solve complete")

        return CameraSolve(
            frames=camera_frames,
            point_cloud=all_points[:10000],  # Cap point cloud size
            reprojection_error=avg_reproj,
            success=success,
            total_frames=len(camera_frames),
            feature_count=total_features,
        )

    except Exception as e:
        logger.exception("Camera solve failed: %s", e)
        return CameraSolve(
            success=False,
            error=f"Camera solve failed: {e}",
        )


def detect_ground_plane(
    solve_result: CameraSolve,
    iterations: int = 1000,
    distance_threshold: float = 0.1,
    on_progress: Optional[Callable] = None,
) -> GroundPlane:
    """
    Detect ground plane from camera solve point cloud using RANSAC.

    Args:
        solve_result: CameraSolve with point_cloud data.
        iterations: RANSAC iterations.
        distance_threshold: Max distance for inlier classification.
        on_progress: Progress callback(pct, msg).

    Returns:
        GroundPlane with normal, distance, and origin.
    """
    import random

    if on_progress:
        on_progress(10, "Starting ground plane detection")

    points = solve_result.point_cloud
    if len(points) < 3:
        return GroundPlane(
            normal=(0.0, 1.0, 0.0),
            distance=0.0,
            origin=(0.0, 0.0, 0.0),
            confidence=0.0,
        )

    best_normal = (0.0, 1.0, 0.0)
    best_distance = 0.0
    best_inliers = 0
    best_origin = (0.0, 0.0, 0.0)

    for iteration in range(iterations):
        # Pick 3 random points
        sample = random.sample(points, min(3, len(points)))
        if len(sample) < 3:
            break

        p1, p2, p3 = sample[0], sample[1], sample[2]

        # Compute plane normal via cross product
        v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])

        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]

        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-10:
            continue

        nx /= length
        ny /= length
        nz /= length

        d = -(nx * p1[0] + ny * p1[1] + nz * p1[2])

        # Count inliers
        inlier_count = 0
        for pt in points:
            dist = abs(nx * pt[0] + ny * pt[1] + nz * pt[2] + d)
            if dist < distance_threshold:
                inlier_count += 1

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = (nx, ny, nz)
            best_distance = d
            best_origin = p1

        if on_progress and iteration % 200 == 0:
            pct = 10 + int(80 * iteration / iterations)
            on_progress(pct, f"RANSAC iteration {iteration}/{iterations}")

    confidence = best_inliers / len(points) if len(points) > 0 else 0.0

    if on_progress:
        on_progress(100, "Ground plane detection complete")

    return GroundPlane(
        normal=best_normal,
        distance=best_distance,
        origin=best_origin,
        inlier_count=best_inliers,
        confidence=confidence,
    )


def place_object_on_plane(
    solve_result: CameraSolve,
    ground_plane: GroundPlane,
    object_config: SceneObject,
    frame_number: int,
) -> Dict:
    """
    Project a 3D position onto 2D pixel coordinates for a given frame.

    Args:
        solve_result: CameraSolve result.
        ground_plane: Detected ground plane.
        object_config: Object to place.
        frame_number: Target frame number.

    Returns:
        Dict with pixel_x, pixel_y, scale, rotation, visible.
    """
    # Find the camera frame closest to requested frame
    camera_frame = None
    for cf in solve_result.frames:
        if cf.frame_number == frame_number:
            camera_frame = cf
            break

    if camera_frame is None and solve_result.frames:
        # Find nearest frame
        camera_frame = min(
            solve_result.frames,
            key=lambda f: abs(f.frame_number - frame_number),
        )

    if camera_frame is None:
        return {
            "pixel_x": 0, "pixel_y": 0, "scale": 0.0,
            "rotation": 0.0, "visible": False,
        }

    # Project 3D point to 2D using pinhole camera model
    obj_pos = object_config.position_3d
    cam_pos = camera_frame.position

    # Relative position
    dx = obj_pos[0] - cam_pos[0]
    dy = obj_pos[1] - cam_pos[1]
    dz = obj_pos[2] - cam_pos[2]

    # Simple pinhole projection
    f = camera_frame.focal_length
    if abs(dz) < 1e-6:
        return {
            "pixel_x": 0, "pixel_y": 0, "scale": 0.0,
            "rotation": 0.0, "visible": False,
        }

    # Apply rotation (simplified - use Euler angles)
    rx, ry, rz = [math.radians(a) for a in camera_frame.rotation]
    cos_ry = math.cos(ry)
    sin_ry = math.sin(ry)
    cos_rx = math.cos(rx)
    sin_rx = math.sin(rx)

    # Rotate around Y
    rx2 = cos_ry * dx + sin_ry * dz
    rz2 = -sin_ry * dx + cos_ry * dz

    # Rotate around X
    ry2 = cos_rx * dy - sin_rx * rz2
    rz3 = sin_rx * dy + cos_rx * rz2

    if rz3 <= 0:
        return {
            "pixel_x": 0, "pixel_y": 0, "scale": 0.0,
            "rotation": 0.0, "visible": False,
        }

    pixel_x = f * rx2 / rz3
    pixel_y = f * ry2 / rz3

    # Scale based on distance
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    apparent_scale = object_config.scale * (f / max(distance, 0.1))

    return {
        "pixel_x": float(pixel_x),
        "pixel_y": float(pixel_y),
        "scale": float(apparent_scale),
        "rotation": float(math.degrees(rz)),
        "visible": True,
        "distance": float(distance),
    }


def render_3d_overlay(
    video_path: str,
    solve_result: CameraSolve,
    objects: List[SceneObject],
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Composite 3D-tracked objects into video.

    Renders text/image overlays locked to 3D positions using the
    camera solve data for each frame.

    Args:
        video_path: Source video file.
        solve_result: CameraSolve result with camera path.
        objects: List of SceneObject to render.
        out_path: Output video path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path and rendering stats.
    """
    if not solve_result.success:
        raise ValueError("Cannot render overlay: camera solve was not successful")

    if not objects:
        raise ValueError("No objects to render")

    if out_path is None:
        out_path = output_path(video_path, "3d_overlay")

    info = get_video_info(video_path)
    width = info.get("width", 1920) or 1920
    height = info.get("height", 1080) or 1080
    info.get("fps", 30.0) or 30.0
    info.get("duration", 0.0) or 0.0

    if on_progress:
        on_progress(5, "Preparing 3D overlay render")

    # Build a ground plane for projection
    gp = detect_ground_plane(solve_result)

    # Pre-compute object positions for each camera frame
    frame_overlays = {}
    for cf in solve_result.frames:
        overlays_for_frame = []
        for obj in objects:
            proj = place_object_on_plane(solve_result, gp, obj, cf.frame_number)
            if proj["visible"]:
                overlays_for_frame.append({
                    "text": obj.text_or_image,
                    "x": proj["pixel_x"] + width / 2,
                    "y": proj["pixel_y"] + height / 2,
                    "scale": proj["scale"],
                    "color": obj.color,
                    "font_size": obj.font_size,
                    "opacity": obj.opacity,
                })
        frame_overlays[cf.frame_number] = overlays_for_frame

    if on_progress:
        on_progress(20, "Building FFmpeg overlay filter")

    # Build drawtext filter chain using first frame's overlay as static overlay
    # For a full implementation, this would use frame-by-frame rendering
    drawtext_parts = []
    first_frame = solve_result.frames[0] if solve_result.frames else None
    if first_frame and first_frame.frame_number in frame_overlays:
        for i, overlay in enumerate(frame_overlays[first_frame.frame_number]):
            text = overlay["text"].replace("'", "\\'").replace(":", "\\:")
            x = int(overlay["x"])
            y = int(overlay["y"])
            fs = max(8, int(overlay["font_size"] * overlay["scale"]))
            color = overlay["color"]
            alpha = overlay["opacity"]
            drawtext_parts.append(
                f"drawtext=text='{text}':x={x}:y={y}"
                f":fontsize={fs}:fontcolor={color}:alpha={alpha}"
            )

    if drawtext_parts:
        vf = ",".join(drawtext_parts)
    else:
        vf = "null"

    if on_progress:
        on_progress(30, "Rendering overlay video")

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac")
        .output(out_path)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "3D overlay render complete")

    return {
        "output_path": out_path,
        "objects_rendered": len(objects),
        "frames_processed": len(solve_result.frames),
        "width": width,
        "height": height,
    }


def export_camera_path(
    solve_result: CameraSolve,
    format: str = "json",
) -> Dict:
    """
    Export camera path data for use in 3D software.

    Args:
        solve_result: CameraSolve result.
        format: Export format - "json", "csv", or "fbx_ascii".

    Returns:
        Dict with exported data as string and metadata.
    """
    if not solve_result.success:
        raise ValueError("Cannot export: camera solve was not successful")

    if format == "json":
        data = {
            "camera_path": [
                {
                    "frame": cf.frame_number,
                    "position": list(cf.position),
                    "rotation": list(cf.rotation),
                    "focal_length": cf.focal_length,
                }
                for cf in solve_result.frames
            ],
            "point_cloud": [list(p) for p in solve_result.point_cloud[:5000]],
            "reprojection_error": solve_result.reprojection_error,
            "total_frames": solve_result.total_frames,
            "feature_count": solve_result.feature_count,
        }
        return {
            "format": "json",
            "data": json.dumps(data, indent=2),
            "frame_count": len(solve_result.frames),
            "point_count": len(solve_result.point_cloud),
        }

    elif format == "csv":
        lines = ["frame,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,focal_length"]
        for cf in solve_result.frames:
            lines.append(
                f"{cf.frame_number},{cf.position[0]:.6f},{cf.position[1]:.6f},"
                f"{cf.position[2]:.6f},{cf.rotation[0]:.6f},{cf.rotation[1]:.6f},"
                f"{cf.rotation[2]:.6f},{cf.focal_length:.2f}"
            )
        return {
            "format": "csv",
            "data": "\n".join(lines),
            "frame_count": len(solve_result.frames),
            "point_count": len(solve_result.point_cloud),
        }

    elif format == "fbx_ascii":
        # Minimal FBX ASCII representation for camera animation
        fbx_lines = [
            "; FBX ASCII - Camera Path Export",
            "; Generated by OpenCut Camera Solver",
            "FBXHeaderExtension: {",
            "    FBXHeaderVersion: 1003",
            "    FBXVersion: 7400",
            "}",
            "",
            "Objects: {",
            '    Model: "Camera", "Camera" {',
            "        Properties70: {",
        ]
        for cf in solve_result.frames:
            fbx_lines.append(
                f"            ; Frame {cf.frame_number}: "
                f"pos=({cf.position[0]:.4f}, {cf.position[1]:.4f}, {cf.position[2]:.4f}) "
                f"rot=({cf.rotation[0]:.4f}, {cf.rotation[1]:.4f}, {cf.rotation[2]:.4f})"
            )
        fbx_lines.extend([
            "        }",
            "    }",
            "}",
        ])
        return {
            "format": "fbx_ascii",
            "data": "\n".join(fbx_lines),
            "frame_count": len(solve_result.frames),
            "point_count": len(solve_result.point_cloud),
        }

    else:
        raise ValueError(f"Unsupported export format: {format}. Use 'json', 'csv', or 'fbx_ascii'.")
