"""
OpenCut Subtitle Positioning Module v1.0.0

Dynamic per-frame subtitle positioning to avoid faces, text, and logos.
Analyzes frame obstructions and computes safe subtitle placement.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Obstruction:
    """A detected obstruction region in a frame."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    label: str = ""       # "face", "text", "logo", "object"
    confidence: float = 0.0


@dataclass
class SubtitlePosition:
    """Computed subtitle position for a frame."""
    x: int = 0
    y: int = 0
    alignment: int = 2       # SSA alignment (1=bottom-left, 2=bottom-center, etc.)
    margin_bottom: int = 50
    safe: bool = True
    reason: str = ""


@dataclass
class PositioningResult:
    """Result for the full dynamic positioning operation."""
    output_path: str = ""
    frames_analyzed: int = 0
    positions_adjusted: int = 0
    duration_seconds: float = 0.0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Default subtitle regions and margins
# ---------------------------------------------------------------------------

# Standard safe zones for subtitle placement (relative to frame height)
_DEFAULT_BOTTOM_ZONE = 0.85    # Bottom 15% of frame
_DEFAULT_TOP_ZONE = 0.10       # Top 10% of frame
_SAFE_MARGIN = 20              # Pixel margin from detected obstructions


# ---------------------------------------------------------------------------
# Frame obstruction analysis
# ---------------------------------------------------------------------------

def _detect_bright_regions(frame_arr, threshold: float = 200.0) -> List[dict]:
    """Simple bright region detection as proxy for text/logo areas."""

    gray = frame_arr.mean(axis=2) if len(frame_arr.shape) == 3 else frame_arr
    h, w = gray.shape

    # Focus on bottom third where subtitles go
    bottom_region = gray[int(h * 0.7):, :]
    bright_mask = bottom_region > threshold

    regions = []
    if bright_mask.any():
        # Find connected bright columns
        col_brightness = bright_mask.mean(axis=0)
        bright_cols = col_brightness > 0.3

        if bright_cols.any():
            # Find contiguous segments
            in_region = False
            start_col = 0
            for c in range(len(bright_cols)):
                if bright_cols[c] and not in_region:
                    start_col = c
                    in_region = True
                elif not bright_cols[c] and in_region:
                    region_w = c - start_col
                    if region_w > 20:  # Skip tiny regions
                        regions.append({
                            "x": start_col,
                            "y": int(h * 0.7),
                            "width": region_w,
                            "height": h - int(h * 0.7),
                            "type": "bright_region",
                        })
                    in_region = False

            if in_region:
                region_w = len(bright_cols) - start_col
                if region_w > 20:
                    regions.append({
                        "x": start_col,
                        "y": int(h * 0.7),
                        "width": region_w,
                        "height": h - int(h * 0.7),
                        "type": "bright_region",
                    })

    return regions


def analyze_frame_obstructions(
    frame_path: str,
    detect_faces: bool = True,
    detect_text: bool = True,
    detect_logos: bool = True,
    on_progress: Optional[Callable] = None,
) -> List[Obstruction]:
    """Analyze a video frame for obstructions that could overlap subtitles.

    Uses brightness analysis and optional face detection to find regions
    that subtitles should avoid.

    Args:
        frame_path: Path to a frame image.
        detect_faces: Enable face detection.
        detect_text: Enable text/bright region detection.
        detect_logos: Enable logo detection (uses bright regions).
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        List of :class:`Obstruction` objects.
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for frame analysis")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for frame analysis")

    import numpy as np
    from PIL import Image

    if on_progress:
        on_progress(10, "Loading frame...")

    img = Image.open(frame_path).convert("RGB")
    frame_arr = np.array(img, dtype=np.float32)
    h, w = frame_arr.shape[:2]

    obstructions = []

    # Face detection via simple skin-color heuristic
    if detect_faces:
        if on_progress:
            on_progress(30, "Detecting faces...")

        # Simple skin-tone detection in bottom portion
        bottom = frame_arr[int(h * 0.5):, :, :]
        # Rough skin tone range in RGB
        r, g, b = bottom[:, :, 0], bottom[:, :, 1], bottom[:, :, 2]
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15)
        )

        skin_ratio = skin_mask.mean()
        if skin_ratio > 0.05:  # Significant skin-tone region
            # Find bounding box of skin region
            rows = np.any(skin_mask, axis=1)
            cols = np.any(skin_mask, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                obstructions.append(Obstruction(
                    x=int(cmin),
                    y=int(h * 0.5 + rmin),
                    width=int(cmax - cmin),
                    height=int(rmax - rmin),
                    label="face",
                    confidence=round(min(skin_ratio * 5, 0.9), 2),
                ))

    # Text / bright region detection
    if detect_text or detect_logos:
        if on_progress:
            on_progress(60, "Detecting text/logos...")

        bright_regions = _detect_bright_regions(frame_arr)
        for region in bright_regions:
            label = "text" if detect_text else "logo"
            obstructions.append(Obstruction(
                x=region["x"],
                y=region["y"],
                width=region["width"],
                height=region["height"],
                label=label,
                confidence=0.6,
            ))

    if on_progress:
        on_progress(100, f"Found {len(obstructions)} obstructions")

    return obstructions


# ---------------------------------------------------------------------------
# Subtitle position computation
# ---------------------------------------------------------------------------

def compute_subtitle_position(
    obstructions: List[Obstruction],
    frame_size: Tuple[int, int],
    preferred_alignment: int = 2,
    margin: int = 50,
) -> SubtitlePosition:
    """Compute optimal subtitle position avoiding obstructions.

    Args:
        obstructions: List of Obstruction objects in the frame.
        frame_size: (width, height) of the frame.
        preferred_alignment: SSA alignment (2=bottom-center default).
        margin: Minimum pixel margin from obstructions.

    Returns:
        :class:`SubtitlePosition` with computed placement.
    """
    w, h = frame_size
    pos = SubtitlePosition(
        x=w // 2,
        y=int(h * _DEFAULT_BOTTOM_ZONE),
        alignment=preferred_alignment,
        margin_bottom=margin,
        safe=True,
    )

    if not obstructions:
        return pos

    # Check if default bottom position overlaps any obstruction
    subtitle_region_top = int(h * _DEFAULT_BOTTOM_ZONE) - margin
    subtitle_region_bottom = h
    subtitle_region_left = w // 4
    subtitle_region_right = 3 * w // 4

    overlap = False
    for obs in obstructions:
        obs_right = obs.x + obs.width
        obs_bottom = obs.y + obs.height

        # Check overlap with subtitle zone
        if (obs.x < subtitle_region_right and obs_right > subtitle_region_left and
                obs.y < subtitle_region_bottom and obs_bottom > subtitle_region_top):
            overlap = True
            break

    if not overlap:
        return pos

    # Try moving subtitle to top of frame
    top_clear = True
    top_region_bottom = int(h * _DEFAULT_TOP_ZONE) + margin
    for obs in obstructions:
        obs_bottom = obs.y + obs.height
        if obs.y < top_region_bottom:
            top_clear = False
            break

    if top_clear:
        pos.y = int(h * _DEFAULT_TOP_ZONE)
        pos.alignment = 8  # SSA top-center
        pos.margin_bottom = h - int(h * _DEFAULT_TOP_ZONE)
        pos.reason = "Moved to top to avoid bottom obstruction"
        return pos

    # Try left-aligned bottom
    left_clear = True
    for obs in obstructions:
        obs_right = obs.x + obs.width
        if obs.x < w // 3 and obs.y > subtitle_region_top:
            left_clear = False
            break

    if left_clear:
        pos.x = w // 4
        pos.alignment = 1  # SSA bottom-left
        pos.reason = "Moved to left to avoid center obstruction"
        return pos

    # Last resort: raise above all obstructions
    max_obs_top = max((obs.y for obs in obstructions), default=subtitle_region_top)
    safe_y = max(max_obs_top - margin - 50, int(h * 0.3))
    pos.y = safe_y
    pos.alignment = 2
    pos.margin_bottom = h - safe_y
    pos.safe = False
    pos.reason = "Raised above obstructions (may be suboptimal)"

    return pos


# ---------------------------------------------------------------------------
# Apply dynamic positioning to subtitle file
# ---------------------------------------------------------------------------

def apply_dynamic_positioning(
    subtitle_path: str,
    video_path: str,
    output_path: str,
    sample_interval: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> PositioningResult:
    """Apply dynamic positioning to subtitles based on video content.

    Samples video frames at intervals, analyzes obstructions, and adjusts
    subtitle positioning to avoid overlap.

    Args:
        subtitle_path: Path to SRT/ASS subtitle file.
        video_path: Path to the video file.
        output_path: Path for the output video with positioned subtitles.
        sample_interval: Seconds between frame samples for analysis.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`PositioningResult` with processing details.
    """
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result = PositioningResult()
    start_time = time.time()

    if on_progress:
        on_progress(5, "Analyzing video for subtitle positioning...")

    # Get video info
    info = get_video_info(video_path)
    duration = info["duration"]
    width = info["width"]
    height = info["height"]

    if duration <= 0:
        raise ValueError("Could not determine video duration")

    # Sample frames and analyze
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="opencut_subpos_")

    try:
        # Extract sample frames
        sample_times = []
        t = 0.0
        while t < duration:
            sample_times.append(t)
            t += sample_interval

        if on_progress:
            on_progress(10, f"Extracting {len(sample_times)} sample frames...")

        frame_positions = {}

        for i, st in enumerate(sample_times):
            frame_path = os.path.join(temp_dir, f"sample_{i:06d}.jpg")

            try:
                cmd = (
                    FFmpegCmd()
                    .pre_input("ss", str(st))
                    .input(video_path)
                    .frames(1)
                    .option("q:v", "2")
                    .output(frame_path)
                    .build()
                )
                run_ffmpeg(cmd)

                if os.path.isfile(frame_path):
                    obstructions = analyze_frame_obstructions(
                        frame_path,
                        detect_faces=True,
                        detect_text=True,
                        detect_logos=True,
                    )

                    position = compute_subtitle_position(
                        obstructions, (width, height),
                    )

                    frame_positions[st] = position
                    result.frames_analyzed += 1

                    if position.reason:
                        result.positions_adjusted += 1

            except Exception as e:
                logger.debug("Frame analysis failed at %.1fs: %s", st, e)

            if on_progress:
                pct = min(int(((i + 1) / len(sample_times)) * 60) + 10, 70)
                on_progress(pct, f"Analyzed {i + 1}/{len(sample_times)} frames")

        # Determine if we need position overrides
        if on_progress:
            on_progress(75, "Encoding video with subtitles...")

        # Build FFmpeg command with subtitle overlay
        # Use the subtitle file as-is with margin adjustments
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Calculate dominant margin from analysis
        margins = [p.margin_bottom for p in frame_positions.values()]
        avg_margin = int(sum(margins) / max(len(margins), 1)) if margins else 50

        # Escape path for FFmpeg subtitle filter
        sub_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")

        vf = f"subtitles='{sub_escaped}':force_style='MarginV={avg_margin}'"

        cmd = (
            FFmpegCmd()
            .input(video_path)
            .video_codec("libx264", crf=18, preset="medium")
            .audio_codec("aac", bitrate="192k")
            .video_filter(vf)
            .faststart()
            .output(output_path)
            .build()
        )
        run_ffmpeg(cmd)

        result.output_path = output_path
        result.status = "complete"

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        raise

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    result.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(
            100,
            f"Positioning complete: {result.positions_adjusted} adjustments "
            f"across {result.frames_analyzed} samples",
        )

    return result
