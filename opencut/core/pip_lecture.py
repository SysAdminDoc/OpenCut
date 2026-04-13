"""
OpenCut PiP Lecture Processing (Feature 33.2)

Detect Picture-in-Picture camera region via contour analysis,
separate speaker from screen content, create side-by-side layouts.

Uses FFmpeg for video processing. Optional OpenCV for advanced
PiP region detection with fallback to common PiP positions.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PipRegion:
    """Detected PiP camera region."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0   # 0.0 - 1.0
    position: str = ""        # e.g. "bottom-right", "top-left"

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "confidence": round(self.confidence, 3),
            "position": self.position,
        }


@dataclass
class PipDetectionResult:
    """Result of PiP region detection."""
    region: dict
    video_width: int
    video_height: int
    method: str  # "contour", "template", "fallback"

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "method": self.method,
        }


@dataclass
class PipExtractionResult:
    """Result of extracting PiP streams."""
    speaker_path: str
    screen_path: str
    output_dir: str

    def to_dict(self) -> dict:
        return {
            "speaker_path": self.speaker_path,
            "screen_path": self.screen_path,
            "output_dir": self.output_dir,
        }


@dataclass
class SideBySideResult:
    """Result of creating side-by-side layout."""
    output_path: str
    layout: str
    resolution: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "layout": self.layout,
            "resolution": list(self.resolution),
        }


# ---------------------------------------------------------------------------
# Common PiP positions (fallback when OpenCV unavailable)
# ---------------------------------------------------------------------------

_PIP_POSITIONS = {
    "bottom-right": lambda w, h, pw, ph: (w - pw - 20, h - ph - 20),
    "bottom-left": lambda w, h, pw, ph: (20, h - ph - 20),
    "top-right": lambda w, h, pw, ph: (w - pw - 20, 20),
    "top-left": lambda w, h, pw, ph: (20, 20),
}

# Typical PiP sizes as fraction of video dimensions
_PIP_SIZE_FRACTIONS = [
    (0.25, 0.25),   # quarter size
    (0.20, 0.20),   # fifth size
    (0.33, 0.33),   # third size
]


# ---------------------------------------------------------------------------
# Frame extraction for analysis
# ---------------------------------------------------------------------------

def _extract_analysis_frames(video_path: str, count: int = 5) -> List[str]:
    """Extract evenly-spaced frames from video for analysis.

    Returns list of temporary PNG file paths.
    """
    import tempfile

    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0:
        duration = 30.0  # assume 30s if unknown

    frames = []
    temp_dir = tempfile.mkdtemp(prefix="opencut_pip_")

    for i in range(count):
        ts = (duration / (count + 1)) * (i + 1)
        out_file = os.path.join(temp_dir, f"frame_{i:03d}.png")

        cmd = (
            FFmpegCmd()
            .pre_input("ss", f"{ts:.3f}")
            .input(video_path)
            .frames(1)
            .option("q:v", "2")
            .output(out_file)
            .build()
        )

        try:
            run_ffmpeg(cmd, timeout=15)
            if os.path.isfile(out_file):
                frames.append(out_file)
        except RuntimeError:
            pass

    return frames


def _cleanup_frames(frame_paths: List[str]):
    """Remove temporary frame files and their parent directory."""
    for p in frame_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    if frame_paths:
        parent = os.path.dirname(frame_paths[0])
        try:
            os.rmdir(parent)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# PiP detection via contour analysis
# ---------------------------------------------------------------------------

def _detect_pip_contours(frame_paths: List[str], video_w: int, video_h: int) -> Optional[PipRegion]:
    """Detect PiP region using OpenCV contour analysis.

    Looks for a rectangular region that:
    1. Appears consistently in the same position across frames
    2. Has a size typical of PiP windows (10-35% of video area)
    3. Is located in a corner of the video
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    # Analyze each frame for rectangular contours
    candidate_regions = []

    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Must be roughly rectangular (4 vertices)
            if len(approx) < 4 or len(approx) > 6:
                continue

            x, y, w, h = cv2.boundingRect(approx)

            # Size constraints: PiP typically 10-35% of each dimension
            w_frac = w / video_w
            h_frac = h / video_h

            if not (0.10 <= w_frac <= 0.40 and 0.10 <= h_frac <= 0.40):
                continue

            # Aspect ratio should be roughly 4:3 or 16:9
            aspect = w / max(h, 1)
            if not (0.5 <= aspect <= 2.5):
                continue

            # Must be near a corner
            cx, cy = x + w // 2, y + h // 2
            is_corner = False
            corner_margin = 0.35
            if (cx < video_w * corner_margin or cx > video_w * (1 - corner_margin)):
                if (cy < video_h * corner_margin or cy > video_h * (1 - corner_margin)):
                    is_corner = True

            if not is_corner:
                continue

            # Determine position name
            position = ""
            if cx > video_w / 2:
                position = "bottom-right" if cy > video_h / 2 else "top-right"
            else:
                position = "bottom-left" if cy > video_h / 2 else "top-left"

            candidate_regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "position": position,
            })

    if not candidate_regions:
        return None

    # Find the most consistent region (appears in similar position across frames)
    # Group by position
    from collections import Counter
    pos_counts = Counter(r["position"] for r in candidate_regions)
    best_pos = pos_counts.most_common(1)[0][0]

    # Average the regions at that position
    pos_regions = [r for r in candidate_regions if r["position"] == best_pos]
    avg_x = int(sum(r["x"] for r in pos_regions) / len(pos_regions))
    avg_y = int(sum(r["y"] for r in pos_regions) / len(pos_regions))
    avg_w = int(sum(r["w"] for r in pos_regions) / len(pos_regions))
    avg_h = int(sum(r["h"] for r in pos_regions) / len(pos_regions))

    confidence = len(pos_regions) / len(frame_paths)

    return PipRegion(
        x=avg_x, y=avg_y, w=avg_w, h=avg_h,
        confidence=min(confidence, 1.0),
        position=best_pos,
    )


def _detect_pip_fallback(video_w: int, video_h: int) -> PipRegion:
    """Fallback PiP detection: assume standard bottom-right quarter-size PiP."""
    pip_w = int(video_w * 0.25)
    pip_h = int(video_h * 0.25)
    pip_x = video_w - pip_w - 20
    pip_y = video_h - pip_h - 20

    return PipRegion(
        x=pip_x, y=pip_y, w=pip_w, h=pip_h,
        confidence=0.3,
        position="bottom-right",
    )


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_pip_region(
    video_path: str,
    sample_count: int = 5,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect PiP camera region in a video.

    Extracts sample frames, runs contour analysis to find a
    consistently-positioned rectangular region typical of PiP webcams.

    Args:
        video_path: Source video file path.
        sample_count: Number of frames to sample for analysis.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with region, video_width, video_height, method.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    video_w, video_h = info["width"], info["height"]

    if on_progress:
        on_progress(10, "Extracting sample frames...")

    frame_paths = _extract_analysis_frames(video_path, count=sample_count)

    try:
        if on_progress:
            on_progress(40, "Analyzing frames for PiP region...")

        # Try contour-based detection
        pip = _detect_pip_contours(frame_paths, video_w, video_h)
        method = "contour"

        if pip is None:
            if on_progress:
                on_progress(60, "Contour detection failed, using fallback...")
            pip = _detect_pip_fallback(video_w, video_h)
            method = "fallback"
    finally:
        _cleanup_frames(frame_paths)

    if on_progress:
        on_progress(100, "Done")

    result = PipDetectionResult(
        region=pip.to_dict(),
        video_width=video_w,
        video_height=video_h,
        method=method,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Stream extraction
# ---------------------------------------------------------------------------

def extract_pip_streams(
    video_path: str,
    pip_region: dict,
    output_dir: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Separate speaker (PiP) from screen content into two video files.

    Crops the PiP region for the speaker video and masks/fills it
    in the screen video.

    Args:
        video_path: Source video file path.
        pip_region: Dict with x, y, w, h of the PiP region.
        output_dir: Directory for output files.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with speaker_path, screen_path, output_dir.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If pip_region is invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    required = ("x", "y", "w", "h")
    for key in required:
        if key not in pip_region:
            raise ValueError(f"pip_region missing required key: {key}")

    rx = int(pip_region["x"])
    ry = int(pip_region["y"])
    rw = int(pip_region["w"])
    rh = int(pip_region["h"])

    if rw <= 0 or rh <= 0:
        raise ValueError("PiP region width and height must be > 0")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    # 1. Extract speaker (crop PiP region)
    if on_progress:
        on_progress(10, "Extracting speaker stream...")

    speaker_path = os.path.join(output_dir, f"{base}_speaker.mp4")
    crop_filter = f"crop={rw}:{rh}:{rx}:{ry}"

    cmd_speaker = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(crop_filter)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="128k")
        .faststart()
        .output(speaker_path)
        .build()
    )

    run_ffmpeg(cmd_speaker)

    # 2. Extract screen content (fill PiP region with black)
    if on_progress:
        on_progress(50, "Extracting screen content...")

    screen_path = os.path.join(output_dir, f"{base}_screen.mp4")
    fill_filter = f"drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black:t=fill"

    cmd_screen = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(fill_filter)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="128k")
        .faststart()
        .output(screen_path)
        .build()
    )

    run_ffmpeg(cmd_screen)

    if on_progress:
        on_progress(100, "Done")

    result = PipExtractionResult(
        speaker_path=speaker_path,
        screen_path=screen_path,
        output_dir=output_dir,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Side-by-side layout
# ---------------------------------------------------------------------------

def create_side_by_side(
    speaker_path: str,
    screen_path: str,
    output_path_str: str,
    layout: str = "speaker-left",
    speaker_scale: float = 0.33,
    output_width: int = 1920,
    output_height: int = 1080,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a side-by-side video from speaker and screen streams.

    Args:
        speaker_path: Path to speaker video.
        screen_path: Path to screen content video.
        output_path_str: Output file path.
        layout: Layout mode - "speaker-left", "speaker-right",
                "speaker-top", "speaker-bottom".
        speaker_scale: Speaker video width as fraction of output (0.1 - 0.5).
        output_width: Output video width.
        output_height: Output video height.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, layout, resolution.

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If layout is invalid.
    """
    if not os.path.isfile(speaker_path):
        raise FileNotFoundError(f"Speaker video not found: {speaker_path}")
    if not os.path.isfile(screen_path):
        raise FileNotFoundError(f"Screen video not found: {screen_path}")

    valid_layouts = ("speaker-left", "speaker-right", "speaker-top", "speaker-bottom")
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout: {layout!r}. Valid: {', '.join(valid_layouts)}")

    speaker_scale = max(0.1, min(0.5, speaker_scale))

    if on_progress:
        on_progress(10, f"Building {layout} layout...")

    ow, oh = output_width, output_height

    if layout in ("speaker-left", "speaker-right"):
        sw = int(ow * speaker_scale)
        sh = oh
        scw = ow - sw
        sch = oh

        if layout == "speaker-left":
            # Speaker on left, screen on right
            fc = (
                f"[0:v]scale={sw}:{sh}:force_original_aspect_ratio=decrease,"
                f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:color=black[spk];"
                f"[1:v]scale={scw}:{sch}:force_original_aspect_ratio=decrease,"
                f"pad={scw}:{sch}:(ow-iw)/2:(oh-ih)/2:color=black[scr];"
                f"[spk][scr]hstack=inputs=2[outv]"
            )
        else:
            # Screen on left, speaker on right
            fc = (
                f"[1:v]scale={scw}:{sch}:force_original_aspect_ratio=decrease,"
                f"pad={scw}:{sch}:(ow-iw)/2:(oh-ih)/2:color=black[scr];"
                f"[0:v]scale={sw}:{sh}:force_original_aspect_ratio=decrease,"
                f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:color=black[spk];"
                f"[scr][spk]hstack=inputs=2[outv]"
            )
    else:
        # Top/bottom layout
        spk_h = int(oh * speaker_scale)
        scr_h = oh - spk_h

        if layout == "speaker-top":
            fc = (
                f"[0:v]scale={ow}:{spk_h}:force_original_aspect_ratio=decrease,"
                f"pad={ow}:{spk_h}:(ow-iw)/2:(oh-ih)/2:color=black[spk];"
                f"[1:v]scale={ow}:{scr_h}:force_original_aspect_ratio=decrease,"
                f"pad={ow}:{scr_h}:(ow-iw)/2:(oh-ih)/2:color=black[scr];"
                f"[spk][scr]vstack=inputs=2[outv]"
            )
        else:
            fc = (
                f"[1:v]scale={ow}:{scr_h}:force_original_aspect_ratio=decrease,"
                f"pad={ow}:{scr_h}:(ow-iw)/2:(oh-ih)/2:color=black[scr];"
                f"[0:v]scale={ow}:{spk_h}:force_original_aspect_ratio=decrease,"
                f"pad={ow}:{spk_h}:(ow-iw)/2:(oh-ih)/2:color=black[spk];"
                f"[scr][spk]vstack=inputs=2[outv]"
            )

    if on_progress:
        on_progress(30, "Encoding side-by-side video...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path_str)), exist_ok=True)

    cmd = (
        FFmpegCmd()
        .input(speaker_path)
        .input(screen_path)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .option("shortest")
        .faststart()
        .output(output_path_str)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Done")

    result = SideBySideResult(
        output_path=output_path_str,
        layout=layout,
        resolution=(ow, oh),
    )
    return result.to_dict()
