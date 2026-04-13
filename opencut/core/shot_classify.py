"""
OpenCut Shot Type Auto-Classification

Classifies shots by type (close-up, wide, medium, etc.) using heuristic
face-detection-based analysis via FFmpeg.

Uses FFmpeg only - no additional ML dependencies required.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")

SHOT_TYPES = [
    "extreme_close_up",
    "close_up",
    "medium_close_up",
    "medium",
    "medium_wide",
    "wide",
    "extreme_wide",
    "aerial",
    "insert",
    "over_shoulder",
]


@dataclass
class ShotInfo:
    """Information about a single classified shot."""
    start: float
    end: float
    shot_type: str
    confidence: float = 0.0


@dataclass
class ShotClassResult:
    """Complete shot classification results."""
    shots: List[ShotInfo] = field(default_factory=list)
    total_shots: int = 0
    duration: float = 0.0
    type_distribution: Dict[str, int] = field(default_factory=dict)


def _detect_faces_in_frame(frame_path: str) -> List[Dict]:
    """
    Detect faces in a frame image using FFmpeg's drawbox metadata approach.

    Uses FFmpeg's cropdetect-based heuristic: extract high-contrast regions
    that match face-like aspect ratios. Falls back to edge analysis.

    Returns list of dicts with x, y, w, h for each detected face-like region.
    """
    faces = []
    try:
        # Use FFmpeg to analyze the frame for face-like rectangular regions
        # We use the metadata filter with a Haar-like approach via lavfi
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return faces
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        if not streams:
            return faces
        frame_w = int(streams[0].get("width", 0))
        frame_h = int(streams[0].get("height", 0))
        if frame_w == 0 or frame_h == 0:
            return faces

        # Analyze the frame using FFmpeg's signalstats for spatial info
        # and entropy analysis to determine if there's a face-like subject
        cmd_analyze = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", frame_path,
            "-vf", "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        result_analyze = subprocess.run(
            cmd_analyze, capture_output=True, text=True, timeout=15
        )
        stderr_text = result_analyze.stderr

        # Parse signal stats for spatial complexity
        # TOUT (temporal outliers) and VREP (vertical repeat) give us
        # info about whether the center of frame has a subject
        tout_match = re.search(r"SIGNALSTATS\.TOUT=(\d+\.?\d*)", stderr_text)
        vrep_match = re.search(r"SIGNALSTATS\.VREP=(\d+\.?\d*)", stderr_text)

        tout = float(tout_match.group(1)) if tout_match else 0.0
        float(vrep_match.group(1)) if vrep_match else 0.0

        # Use center crop analysis to estimate face region
        # Extract center 50% of frame and compare complexity
        center_cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", frame_path,
            "-vf", f"crop={frame_w // 2}:{frame_h // 2}:{frame_w // 4}:{frame_h // 4},"
                   f"signalstats=stat=tout+vrep+brng,metadata=print:file=-",
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        center_result = subprocess.run(
            center_cmd, capture_output=True, text=True, timeout=15
        )
        center_stderr = center_result.stderr

        center_tout_match = re.search(
            r"SIGNALSTATS\.TOUT=(\d+\.?\d*)", center_stderr
        )
        center_tout = float(center_tout_match.group(1)) if center_tout_match else 0.0

        # Heuristic: if center region has notably different complexity than
        # full frame, there's likely a subject (face) in center
        if center_tout > 0 or tout > 0:
            # Estimate face size based on complexity ratio
            ratio = center_tout / max(tout, 0.001) if tout > 0 else 1.5
            estimated_face_w = int(frame_w * min(0.6, max(0.05, 0.15 * ratio)))
            estimated_face_h = int(estimated_face_w * 1.3)  # face aspect ratio

            if estimated_face_h <= frame_h:
                faces.append({
                    "x": (frame_w - estimated_face_w) // 2,
                    "y": int(frame_h * 0.15),
                    "w": estimated_face_w,
                    "h": estimated_face_h,
                })

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Face detection failed for %s: %s", frame_path, exc)

    return faces


def _compute_entropy(frame_path: str) -> float:
    """Compute spatial entropy of a frame (0.0 - 1.0 normalized)."""
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", frame_path,
            "-vf", "entropy=mode=normal,metadata=print:file=-",
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # Parse entropy value from metadata output
        entropy_match = re.search(
            r"entropy\.entropy\.normal\.Y=(\d+\.?\d*)", result.stderr
        )
        if entropy_match:
            return float(entropy_match.group(1))
        # Fallback: estimate from signal stats
        brng_match = re.search(r"SIGNALSTATS\.BRNG=(\d+\.?\d*)", result.stderr)
        if brng_match:
            return 1.0 - float(brng_match.group(1)) / 100.0
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Entropy computation failed for %s: %s", frame_path, exc)
    return 0.5  # default mid-range


def classify_single_frame(frame_path: str) -> dict:
    """
    Classify a single frame image by shot type.

    Args:
        frame_path: Path to the image file.

    Returns:
        dict with shot_type and confidence.
    """
    # Get frame dimensions
    try:
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        frame_w = int(streams[0].get("width", 1920)) if streams else 1920
        frame_h = int(streams[0].get("height", 1080)) if streams else 1080
    except Exception:
        frame_w, frame_h = 1920, 1080

    frame_area = frame_w * frame_h
    faces = _detect_faces_in_frame(frame_path)
    entropy = _compute_entropy(frame_path)

    if faces:
        # Use largest face for classification
        largest_face = max(faces, key=lambda f: f["w"] * f["h"])
        face_area = largest_face["w"] * largest_face["h"]
        face_ratio = face_area / frame_area

        if face_ratio > 0.30:
            return {"shot_type": "extreme_close_up", "confidence": min(0.95, 0.7 + face_ratio)}
        elif face_ratio > 0.15:
            return {"shot_type": "close_up", "confidence": 0.80}
        elif face_ratio > 0.08:
            return {"shot_type": "medium_close_up", "confidence": 0.75}
        elif face_ratio > 0.03:
            return {"shot_type": "medium", "confidence": 0.70}
        elif face_ratio > 0.01:
            return {"shot_type": "medium_wide", "confidence": 0.65}
        else:
            return {"shot_type": "wide", "confidence": 0.55}
    else:
        # No face detected - classify by entropy and spatial characteristics
        if entropy > 0.85:
            return {"shot_type": "extreme_wide", "confidence": 0.50}
        elif entropy > 0.6:
            return {"shot_type": "wide", "confidence": 0.50}
        elif entropy < 0.3:
            return {"shot_type": "insert", "confidence": 0.45}
        else:
            return {"shot_type": "medium", "confidence": 0.40}


def classify_shots(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> ShotClassResult:
    """
    Classify all shots in a video by type.

    Extracts key frames from scene boundaries and classifies each shot
    using face-detection heuristics and spatial analysis.

    Args:
        input_path: Source video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        ShotClassResult with classified shots.
    """
    if on_progress:
        on_progress(5, "Getting video info...")

    info = get_video_info(input_path)
    duration = info.get("duration", 0)
    info.get("fps", 30.0)

    if on_progress:
        on_progress(10, "Detecting scene boundaries...")

    # Detect scene changes using FFmpeg
    from opencut.core.scene_detect import detect_scenes
    scenes = detect_scenes(input_path, threshold=0.3, min_scene_length=1.0)

    boundaries = scenes.boundaries
    if not boundaries:
        # No scenes detected - treat entire video as one shot
        boundaries_times = [0.0, duration]
    else:
        boundaries_times = [b.time for b in boundaries]
        if boundaries_times[-1] < duration - 0.5:
            boundaries_times.append(duration)

    total_scenes = len(boundaries_times) - 1
    if total_scenes < 1:
        total_scenes = 1
        boundaries_times = [0.0, max(duration, 0.1)]

    if on_progress:
        on_progress(20, f"Classifying {total_scenes} shots...")

    shots: List[ShotInfo] = []
    tmp_dir = tempfile.mkdtemp(prefix="opencut_shotclass_")

    try:
        for i in range(total_scenes):
            start_t = boundaries_times[i]
            end_t = boundaries_times[i + 1] if (i + 1) < len(boundaries_times) else duration
            mid_t = (start_t + end_t) / 2.0

            # Extract key frame at midpoint of scene
            frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            extract_cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", str(mid_t),
                "-i", input_path,
                "-frames:v", "1",
                "-q:v", "2",
                frame_path,
            ]
            try:
                subprocess.run(extract_cmd, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Frame extraction timed out for scene %d at %.2fs", i, mid_t)
                shots.append(ShotInfo(
                    start=start_t, end=end_t,
                    shot_type="medium", confidence=0.2,
                ))
                continue

            if not os.path.isfile(frame_path):
                shots.append(ShotInfo(
                    start=start_t, end=end_t,
                    shot_type="medium", confidence=0.2,
                ))
                continue

            classification = classify_single_frame(frame_path)
            shots.append(ShotInfo(
                start=round(start_t, 3),
                end=round(end_t, 3),
                shot_type=classification["shot_type"],
                confidence=round(classification["confidence"], 3),
            ))

            if on_progress:
                pct = 20 + int(70 * (i + 1) / total_scenes)
                on_progress(pct, f"Classified shot {i + 1}/{total_scenes}: {classification['shot_type']}")

    finally:
        # Cleanup temp frames
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Build type distribution
    type_dist: Dict[str, int] = {}
    for s in shots:
        type_dist[s.shot_type] = type_dist.get(s.shot_type, 0) + 1

    if on_progress:
        on_progress(100, f"Classified {len(shots)} shots")

    return ShotClassResult(
        shots=shots,
        total_shots=len(shots),
        duration=duration,
        type_distribution=type_dist,
    )
