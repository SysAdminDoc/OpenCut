"""
OpenCut Motion Transfer Module

Full-body motion transfer: extract a pose sequence from a source video
and animate a target person image with those poses.  Inspired by
AnimateAnyone 2 / MimicMotion.

Pipeline:
  1. Extract pose sequence from source video via MediaPipe Pose.
  2. Generate frames of the target person in source poses (AI model).
  3. Assemble generated frames into output video.

Fallbacks when AI models are unavailable:
  - Export pose data as JSON for external tools (ComfyUI, etc.).
  - Generate stick-figure animation overlaid on the target image.

Functions:
    extract_pose_sequence - Extract per-frame pose data from a video
    transfer_motion       - Animate a target image with source motion
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

# Reuse the landmark map from body_effects
_MP_LANDMARK_MAP = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Skeleton connections for stick-figure rendering
_SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]

SUPPORTED_MODELS = ["auto", "animate_anyone", "mimic_motion", "stick_figure"]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PoseSequence:
    """Extracted pose sequence from a video."""

    poses: List[Dict[str, Tuple[float, float, float]]] = field(default_factory=list)
    fps: float = 30.0
    duration: float = 0.0


@dataclass
class MotionTransferResult:
    """Result of motion transfer."""

    output_path: str = ""
    source_duration: float = 0.0
    frames_generated: int = 0
    model_used: str = "stick_figure"


# ---------------------------------------------------------------------------
# Pose Extraction
# ---------------------------------------------------------------------------


def extract_pose_sequence(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> PoseSequence:
    """Extract per-frame pose keypoints from *video_path* via MediaPipe Pose.

    Args:
        video_path: Path to the source motion video.
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        PoseSequence with per-frame keypoint dicts.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Loading pose model...")

    if not ensure_package("mediapipe", "mediapipe", on_progress):
        raise RuntimeError("mediapipe is required but could not be installed")
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required but could not be installed")

    import cv2
    import mediapipe as mp

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    duration = info.get("duration", 0.0)

    if on_progress:
        on_progress(10, "Extracting pose sequence...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    poses: List[Dict[str, Tuple[float, float, float]]] = []
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            h, w = frame.shape[:2]

            frame_kp: Dict[str, Tuple[float, float, float]] = {}
            if results.pose_landmarks:
                for part_name, lm_idx in _MP_LANDMARK_MAP.items():
                    lm = results.pose_landmarks.landmark[lm_idx]
                    frame_kp[part_name] = (
                        round(lm.x * w, 2),
                        round(lm.y * h, 2),
                        round(lm.visibility, 4),
                    )

            poses.append(frame_kp)
            frame_idx += 1

            if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                pct = 10 + int(80 * frame_idx / total_frames)
                on_progress(min(pct, 90), f"Extracted poses: {frame_idx}/{total_frames}")
    finally:
        cap.release()
        pose.close()

    if on_progress:
        on_progress(95, "Pose extraction complete")

    return PoseSequence(poses=poses, fps=fps, duration=duration)


# ---------------------------------------------------------------------------
# Motion Transfer
# ---------------------------------------------------------------------------


def _try_ai_model(model: str) -> Optional[str]:
    """Attempt to import an AI motion-transfer model. Returns model name or None."""
    if model in ("auto", "animate_anyone"):
        try:
            import importlib
            importlib.import_module("animate_anyone")
            return "animate_anyone"
        except ImportError:
            pass
    if model in ("auto", "mimic_motion"):
        try:
            import importlib
            importlib.import_module("mimic_motion")
            return "mimic_motion"
        except ImportError:
            pass
    return None


def _render_stick_figure_frames(
    target_image_path: str,
    pose_seq: PoseSequence,
    frame_dir: str,
    on_progress: Optional[Callable] = None,
) -> int:
    """Render stick-figure animation frames overlaid on the target image.

    Returns the number of frames written.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("opencv-python-headless is required for stick figure rendering")

    import cv2

    target = cv2.imread(target_image_path)
    if target is None:
        raise FileNotFoundError(f"Could not read target image: {target_image_path}")

    h, w = target.shape[:2]
    count = 0

    for i, frame_kp in enumerate(pose_seq.poses):
        canvas = target.copy()

        # Draw skeleton lines
        for p1_name, p2_name in _SKELETON_CONNECTIONS:
            p1 = frame_kp.get(p1_name)
            p2 = frame_kp.get(p2_name)
            if p1 and p2 and p1[2] > 0.3 and p2[2] > 0.3:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw keypoint circles
        for part_name, kp in frame_kp.items():
            if kp[2] > 0.3:
                cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        out_frame = os.path.join(frame_dir, f"frame_{i:06d}.png")
        cv2.imwrite(out_frame, canvas)
        count += 1

        if on_progress and i % max(1, len(pose_seq.poses) // 10) == 0:
            pct = 50 + int(30 * i / max(1, len(pose_seq.poses)))
            on_progress(min(pct, 80), f"Rendering frame {i}/{len(pose_seq.poses)}")

    return count


def _export_pose_json(pose_seq: PoseSequence, output_dir: str) -> str:
    """Export the pose sequence as a JSON file for external tools."""
    json_path = os.path.join(output_dir, "pose_sequence.json")
    data = {
        "fps": pose_seq.fps,
        "duration": pose_seq.duration,
        "frame_count": len(pose_seq.poses),
        "poses": [
            {k: list(v) for k, v in frame_kp.items()}
            for frame_kp in pose_seq.poses
        ],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def transfer_motion(
    source_video: str,
    target_image: str,
    output: Optional[str] = None,
    model: str = "auto",
    on_progress: Optional[Callable] = None,
) -> MotionTransferResult:
    """Transfer motion from *source_video* onto the person in *target_image*.

    Args:
        source_video: Path to the video providing motion.
        target_image: Path to the image of the person to animate.
        output: Output video path. Auto-generated if None.
        model: AI model to use — ``"auto"``, ``"animate_anyone"``,
               ``"mimic_motion"``, or ``"stick_figure"``.
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        MotionTransferResult with output path and metadata.
    """
    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if not os.path.isfile(target_image):
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Choose from: {SUPPORTED_MODELS}")

    if output is None:
        output = _output_path(source_video, "motion_transfer")

    if on_progress:
        on_progress(5, "Extracting pose sequence from source video...")

    pose_seq = extract_pose_sequence(source_video, on_progress=on_progress)

    if on_progress:
        on_progress(40, "Pose extraction complete, preparing transfer...")

    # Try AI model first unless explicitly stick_figure
    model_used = "stick_figure"
    if model != "stick_figure":
        resolved = _try_ai_model(model)
        if resolved:
            model_used = resolved
            logger.info("Using AI model: %s", resolved)
            # Placeholder: AI model integration would generate frames here
            # For now, fall through to stick_figure
            model_used = "stick_figure"
            logger.info("AI model %s loaded but generation not yet integrated, using stick_figure", resolved)

    if on_progress:
        on_progress(45, f"Generating frames via {model_used}...")

    # Render stick-figure frames
    frame_dir = tempfile.mkdtemp(prefix="opencut_motion_")

    # Also export pose JSON for external tool usage
    _export_pose_json(pose_seq, frame_dir)

    frames_generated = _render_stick_figure_frames(
        target_image, pose_seq, frame_dir, on_progress=on_progress,
    )

    if frames_generated == 0:
        raise RuntimeError("No frames were generated")

    if on_progress:
        on_progress(85, "Assembling output video...")

    # Assemble frames into video via FFmpeg
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    fps = pose_seq.fps if pose_seq.fps > 0 else 30.0

    cmd = (
        FFmpegCmd()
        .option("framerate", str(fps))
        .input(frame_pattern)
        .video_codec("libx264", crf=18, preset="fast")
        .faststart()
        .output(output)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "Motion transfer complete")

    return MotionTransferResult(
        output_path=output,
        source_duration=pose_seq.duration,
        frames_generated=frames_generated,
        model_used=model_used,
    )
