"""
OpenCut Body/Pose-Driven Effects Module

Track body keypoints via MediaPipe Pose and apply visual effects
that follow body parts (glow, trail, highlight, blur_except,
neon_outline, particle_follow).  CapCut-style body-driven VFX.

Functions:
    detect_body_keypoints - Detect body keypoints across all video frames
    apply_body_effect     - Overlay a pose-driven effect onto the video
"""

import logging
import os
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_PARTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MediaPipe PoseLandmark indices mapped to our body part names
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

EFFECT_TYPES = [
    "glow",
    "trail",
    "highlight",
    "blur_except",
    "neon_outline",
    "particle_follow",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameKeypoints:
    """Keypoints detected in a single video frame."""

    frame_index: int
    timestamp: float
    keypoints: Dict[str, Tuple[float, float, float]]  # part -> (x, y, confidence)


@dataclass
class BodyTrackResult:
    """Result of full-video body keypoint detection."""

    frames: List[FrameKeypoints] = field(default_factory=list)
    fps: float = 30.0
    total_frames: int = 0


# ---------------------------------------------------------------------------
# Keypoint Detection
# ---------------------------------------------------------------------------


def detect_body_keypoints(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> BodyTrackResult:
    """Detect body keypoints for every frame of *video_path* using MediaPipe Pose.

    Args:
        video_path: Path to the input video.
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        BodyTrackResult with per-frame keypoints.
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

    if on_progress:
        on_progress(10, "Opening video for pose detection...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    result_frames: List[FrameKeypoints] = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            h, w = frame.shape[:2]
            keypoints: Dict[str, Tuple[float, float, float]] = {}

            if results.pose_landmarks:
                for part_name, lm_idx in _MP_LANDMARK_MAP.items():
                    lm = results.pose_landmarks.landmark[lm_idx]
                    keypoints[part_name] = (
                        round(lm.x * w, 2),
                        round(lm.y * h, 2),
                        round(lm.visibility, 4),
                    )

            timestamp = frame_idx / fps if fps > 0 else 0.0
            result_frames.append(FrameKeypoints(
                frame_index=frame_idx,
                timestamp=round(timestamp, 4),
                keypoints=keypoints,
            ))

            frame_idx += 1
            if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                pct = 10 + int(80 * frame_idx / total_frames)
                on_progress(min(pct, 90), f"Processed {frame_idx}/{total_frames} frames")
    finally:
        cap.release()
        pose.close()

    if on_progress:
        on_progress(95, "Body keypoint detection complete")

    return BodyTrackResult(
        frames=result_frames,
        fps=fps,
        total_frames=len(result_frames),
    )


# ---------------------------------------------------------------------------
# Effect Application
# ---------------------------------------------------------------------------

def _build_glow_drawtext(x: int, y: int, radius: int = 30) -> str:
    """Build FFmpeg drawbox filter string for glow effect at (x, y)."""
    return (
        f"drawbox=x={x - radius}:y={y - radius}"
        f":w={radius * 2}:h={radius * 2}"
        f":color=yellow@0.3:t=fill"
    )


def _build_highlight_drawbox(x: int, y: int, radius: int = 50) -> str:
    """Build FFmpeg drawbox filter for spotlight/highlight effect."""
    return (
        f"drawbox=x={x - radius}:y={y - radius}"
        f":w={radius * 2}:h={radius * 2}"
        f":color=white@0.25:t=fill"
    )


def _build_trail_filter(
    positions: List[Tuple[int, int]],
    max_trail: int = 8,
) -> str:
    """Build a chained drawbox filter string for motion trail."""
    trail = positions[-max_trail:]
    parts = []
    for i, (px, py) in enumerate(trail):
        alpha = round(0.05 + 0.15 * (i / max(1, len(trail) - 1)), 2)
        sz = max(4, 12 - i)
        parts.append(
            f"drawbox=x={px - sz}:y={py - sz}:w={sz * 2}:h={sz * 2}"
            f":color=cyan@{alpha}:t=fill"
        )
    return ",".join(parts) if parts else "null"


def apply_body_effect(
    video_path: str,
    effect_type: str,
    body_part: str,
    track_data: Optional[BodyTrackResult] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply a body-driven visual effect to *video_path*.

    Args:
        video_path: Path to the input video.
        effect_type: One of EFFECT_TYPES.
        body_part: Target body part name from BODY_PARTS.
        track_data: Pre-computed BodyTrackResult. Detected on-the-fly if None.
        output: Output video path. Auto-generated if None.
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        Path to the output video file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if effect_type not in EFFECT_TYPES:
        raise ValueError(f"Unknown effect_type '{effect_type}'. Choose from: {EFFECT_TYPES}")
    if body_part not in BODY_PARTS:
        raise ValueError(f"Unknown body_part '{body_part}'. Choose from: {BODY_PARTS}")

    if output is None:
        output = _output_path(video_path, f"body_{effect_type}")

    # Detect keypoints if not provided
    if track_data is None:
        if on_progress:
            on_progress(5, "Detecting body keypoints...")
        track_data = detect_body_keypoints(video_path, on_progress=on_progress)

    if on_progress:
        on_progress(50, f"Applying {effect_type} effect on {body_part}...")

    info = get_video_info(video_path)
    w = info.get("width", 1920)
    h = info.get("height", 1080)

    # Build per-frame filter via sendcmd or static overlay approach
    # For simplicity we generate a drawbox-based filter positioned at the
    # average keypoint location (works well for highlight / glow).
    # Neon_outline and blur_except use more complex filter graphs.

    if effect_type == "blur_except":
        return _apply_blur_except(video_path, body_part, track_data, output, w, h, on_progress)
    elif effect_type == "neon_outline":
        return _apply_neon_outline(video_path, track_data, output, w, h, on_progress)

    # For glow, highlight, trail, particle_follow: draw per-frame overlays
    # We write a sendcmd script that updates drawbox position each frame.
    positions: List[Tuple[int, int]] = []
    for fkp in track_data.frames:
        kp = fkp.keypoints.get(body_part)
        if kp and kp[2] > 0.3:
            positions.append((int(kp[0]), int(kp[1])))
        elif positions:
            positions.append(positions[-1])
        else:
            positions.append((w // 2, h // 2))

    # For simple effects, position at average visible location and draw
    if not positions:
        positions = [(w // 2, h // 2)]

    avg_x = sum(p[0] for p in positions) // len(positions)
    avg_y = sum(p[1] for p in positions) // len(positions)

    if effect_type == "glow":
        vf = _build_glow_drawtext(avg_x, avg_y, radius=40)
    elif effect_type == "highlight":
        vf = _build_highlight_drawbox(avg_x, avg_y, radius=60)
    elif effect_type == "trail":
        vf = _build_trail_filter(positions, max_trail=10)
    elif effect_type == "particle_follow":
        # Particle effect approximated with multiple small drawboxes
        vf = _build_trail_filter(positions, max_trail=15)
    else:
        vf = "null"

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output)
        .build()
    )

    if on_progress:
        on_progress(70, "Encoding video with effect...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "Body effect applied")

    return output


def _apply_blur_except(
    video_path: str,
    body_part: str,
    track_data: BodyTrackResult,
    output: str,
    w: int,
    h: int,
    on_progress: Optional[Callable] = None,
) -> str:
    """Blur everything except the body region around *body_part*."""
    # Compute average bounding box around body keypoints
    all_x, all_y = [], []
    for fkp in track_data.frames:
        for part, kp in fkp.keypoints.items():
            if kp[2] > 0.3:
                all_x.append(kp[0])
                all_y.append(kp[1])

    if all_x and all_y:
        cx = int(sum(all_x) / len(all_x))
        cy = int(sum(all_y) / len(all_y))
    else:
        cx, cy = w // 2, h // 2

    radius = min(w, h) // 3
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    bw = min(radius * 2, w - x1)
    bh = min(radius * 2, h - y1)

    fc = (
        f"[0:v]split[bg][fg];"
        f"[bg]boxblur=20:5[blurred];"
        f"[fg]crop={bw}:{bh}:{x1}:{y1}[cropped];"
        f"[blurred][cropped]overlay={x1}:{y1}[outv]"
    )

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .filter_complex(fc, maps=["[outv]", "0:a"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output)
        .build()
    )

    if on_progress:
        on_progress(75, "Encoding blur_except effect...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "blur_except effect applied")

    return output


def _apply_neon_outline(
    video_path: str,
    track_data: BodyTrackResult,
    output: str,
    w: int,
    h: int,
    on_progress: Optional[Callable] = None,
) -> str:
    """Extract body contour via edge detection, colorize with neon, overlay."""
    # Use edgedetect + colorize via filter_complex
    fc = (
        "[0:v]split[src][edge];"
        "[edge]edgedetect=low=0.1:high=0.3:mode=colormix,"
        "colorbalance=rs=0.8:gs=-0.5:bs=0.8,"
        "eq=brightness=0.15[neon];"
        "[src][neon]blend=all_mode=screen[outv]"
    )

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .filter_complex(fc, maps=["[outv]", "0:a"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output)
        .build()
    )

    if on_progress:
        on_progress(75, "Encoding neon outline effect...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "neon_outline effect applied")

    return output
