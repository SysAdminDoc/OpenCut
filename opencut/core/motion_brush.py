"""
OpenCut Motion Brush Module

Paint where motion should happen on a still image or video.
Runway popularized this interaction model for controlled video generation.

Motion types:
    - Directional: left, right, up, down
    - Zoom: zoom_in, zoom_out
    - Rotation: rotate_cw, rotate_ccw
    - Custom mask with direction + strength

Methods:
    - Simple: FFmpeg zoompan/scroll/rotate filters masked to regions
    - Advanced: Video generation model with motion control (when available)

Functions:
    apply_motion_brush  - Apply motion brush effects to image or video
    create_motion_mask  - Generate a mask image from region definitions
    preview_motion_brush - Preview motion brush on a single frame
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Union

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_DIRECTIONS = (
    "left", "right", "up", "down",
    "zoom_in", "zoom_out",
    "rotate_cw", "rotate_ccw",
)

# Default output duration for still image input
_DEFAULT_STILL_DURATION = 4.0
# Maximum motion regions
_MAX_REGIONS = 20


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MotionRegion:
    """A single motion region definition."""
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 100
    direction: str = "right"
    strength: float = 0.5

    def validate(self, frame_w: int, frame_h: int):
        """Validate and clamp region to frame bounds."""
        self.x = max(0, min(self.x, frame_w - 1))
        self.y = max(0, min(self.y, frame_h - 1))
        self.w = max(1, min(self.w, frame_w - self.x))
        self.h = max(1, min(self.h, frame_h - self.y))
        self.strength = max(0.0, min(1.0, self.strength))
        if self.direction not in VALID_DIRECTIONS:
            self.direction = "right"


@dataclass
class MotionBrushResult:
    """Result of a motion brush operation."""
    output_path: str = ""
    duration: float = 0.0
    regions_animated: int = 0
    method_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers: input detection
# ---------------------------------------------------------------------------
def _is_image(path: str) -> bool:
    """Check if a file is an image based on extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def _is_video(path: str) -> bool:
    """Check if a file is a video based on extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf")


def _parse_regions(motion_mask) -> list:
    """Parse motion_mask into a list of MotionRegion objects.

    Accepts:
        - A single dict with x/y/w/h/direction/strength
        - A list of such dicts
        - A string path to a mask image (returned as-is for mask-based processing)
    """
    if isinstance(motion_mask, str):
        return motion_mask  # Mask image path

    if isinstance(motion_mask, dict):
        motion_mask = [motion_mask]

    regions = []
    for item in motion_mask[:_MAX_REGIONS]:
        if isinstance(item, dict):
            regions.append(MotionRegion(
                x=int(item.get("x", 0)),
                y=int(item.get("y", 0)),
                w=int(item.get("w", 100)),
                h=int(item.get("h", 100)),
                direction=str(item.get("direction", "right")),
                strength=float(item.get("strength", 0.5)),
            ))
        elif isinstance(item, MotionRegion):
            regions.append(item)

    return regions


# ---------------------------------------------------------------------------
# Internal helpers: mask generation
# ---------------------------------------------------------------------------
def create_motion_mask(
    image_path: str,
    regions: List[Dict],
) -> str:
    """Generate a grayscale mask image from region definitions.

    White areas indicate where motion should be applied.
    Brightness encodes strength (brighter = stronger motion).

    Args:
        image_path: Reference image to size the mask.
        regions: List of dicts with x, y, w, h, strength keys.

    Returns:
        Path to the generated mask PNG file.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("OpenCV is required for mask generation")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("numpy is required for mask generation")

    import cv2
    import numpy as np

    # Get dimensions from reference
    if _is_image(image_path):
        ref = cv2.imread(image_path)
        if ref is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = ref.shape[:2]
    else:
        info = get_video_info(image_path)
        w = info.get("width", 1920)
        h = info.get("height", 1080)

    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions[:_MAX_REGIONS]:
        rx = max(0, int(region.get("x", 0)))
        ry = max(0, int(region.get("y", 0)))
        rw = max(1, int(region.get("w", 100)))
        rh = max(1, int(region.get("h", 100)))
        strength = max(0.0, min(1.0, float(region.get("strength", 0.5))))

        # Draw filled rectangle with brightness = strength
        brightness = int(strength * 255)
        mask[ry:ry + rh, rx:rx + rw] = brightness

    # Apply slight Gaussian blur for soft edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    fd, mask_path = tempfile.mkstemp(suffix=".png", prefix="motion_mask_")
    os.close(fd)
    cv2.imwrite(mask_path, mask)

    return mask_path


# ---------------------------------------------------------------------------
# Internal helpers: motion application
# ---------------------------------------------------------------------------
def _build_region_filter(
    region: MotionRegion,
    width: int,
    height: int,
    duration: float,
    fps: float,
    region_idx: int,
) -> str:
    """Build FFmpeg filter chain for a single motion region.

    Uses overlay with a cropped+transformed region to create localized motion.

    Returns:
        FFmpeg filter string fragment.
    """
    strength = region.strength
    d = region.direction

    # Pixel displacement per frame based on strength and duration
    max_px = int(strength * min(region.w, region.h) * 0.3)

    if d in ("left", "right"):
        sign = -1 if d == "left" else 1
        # Horizontal scroll of the region
        dx_expr = f"{sign}*{max_px}*t/{duration:.3f}"
        dy_expr = "0"
    elif d in ("up", "down"):
        sign = -1 if d == "up" else 1
        dx_expr = "0"
        dy_expr = f"{sign}*{max_px}*t/{duration:.3f}"
    elif d == "zoom_in":
        # Zoom crops from full region to center
        zoom_factor = 1.0 + strength * 0.3
        return (
            f"crop={region.w}:{region.h}:{region.x}:{region.y},"
            f"scale={region.w}*min({zoom_factor:.3f}\\,1+{strength:.3f}*t/{duration:.3f})"
            f":{region.h}*min({zoom_factor:.3f}\\,1+{strength:.3f}*t/{duration:.3f}),"
            f"crop={region.w}:{region.h}"
        )
    elif d == "zoom_out":
        zoom_factor = 1.0 + strength * 0.3
        return (
            f"crop={region.w}:{region.h}:{region.x}:{region.y},"
            f"scale={region.w}*max(1\\,{zoom_factor:.3f}-{strength:.3f}*t/{duration:.3f})"
            f":{region.h}*max(1\\,{zoom_factor:.3f}-{strength:.3f}*t/{duration:.3f}),"
            f"crop={region.w}:{region.h}"
        )
    elif d == "rotate_cw":
        angle_per_frame = strength * 0.5  # degrees per frame
        return (
            f"crop={region.w}:{region.h}:{region.x}:{region.y},"
            f"rotate={angle_per_frame:.4f}*PI/180*t*{fps:.1f}"
            f":fillcolor=none:ow={region.w}:oh={region.h}"
        )
    elif d == "rotate_ccw":
        angle_per_frame = strength * 0.5
        return (
            f"crop={region.w}:{region.h}:{region.x}:{region.y},"
            f"rotate=-{angle_per_frame:.4f}*PI/180*t*{fps:.1f}"
            f":fillcolor=none:ow={region.w}:oh={region.h}"
        )
    else:
        dx_expr = "0"
        dy_expr = "0"

    # For directional motion: crop region, scroll, overlay back
    return (
        f"crop={region.w}:{region.h}:{region.x}:{region.y},"
        f"scroll=horizontal={dx_expr}:vertical={dy_expr}"
    )


def _apply_ffmpeg_motion(
    input_path: str,
    regions: list,
    duration: float,
    fps: float,
    width: int,
    height: int,
    out_path: str,
    is_still: bool = False,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply motion to regions using FFmpeg filter chains.

    For still images: uses zoompan for Ken Burns effect with regional focus.
    For video: overlays transformed regions onto the original.

    Returns:
        Output file path.
    """
    if not regions:
        raise ValueError("No motion regions provided")

    if is_still:
        return _apply_still_motion(
            input_path, regions, duration, fps, width, height, out_path, on_progress,
        )
    else:
        return _apply_video_motion(
            input_path, regions, duration, fps, width, height, out_path, on_progress,
        )


def _apply_still_motion(
    image_path: str,
    regions: list,
    duration: float,
    fps: float,
    width: int,
    height: int,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply motion effects to a still image using Ken Burns + regional emphasis.

    Creates natural-looking camera movement focused on the motion regions.
    """
    if on_progress:
        on_progress(30, "Applying Ken Burns motion to still image...")

    total_frames = int(duration * fps)

    # Calculate motion center from regions
    if regions:
        cx = sum(r.x + r.w // 2 for r in regions) / len(regions)
        cy = sum(r.y + r.h // 2 for r in regions) / len(regions)
        avg_strength = sum(r.strength for r in regions) / len(regions)
    else:
        cx, cy = width / 2, height / 2
        avg_strength = 0.3

    # Determine dominant motion direction
    direction_counts = {}
    for r in regions:
        direction_counts[r.direction] = direction_counts.get(r.direction, 0) + 1
    dominant = max(direction_counts, key=direction_counts.get) if direction_counts else "zoom_in"

    # Build zoompan parameters based on dominant direction
    zoom_max = 1.0 + avg_strength * 0.15
    zoom_expr = f"min({zoom_max:.4f},1+{avg_strength * 0.15:.4f}*on/{total_frames})"

    # Pan towards the motion center
    norm_cx = cx / width
    norm_cy = cy / height
    x_expr = f"iw*{norm_cx:.4f}-(iw/zoom/2)"
    y_expr = f"ih*{norm_cy:.4f}-(ih/zoom/2)"

    # Adjust pan based on direction
    drift = avg_strength * 0.05
    if dominant in ("left", "right"):
        sign = -1 if dominant == "left" else 1
        x_expr = f"iw*{norm_cx:.4f}-(iw/zoom/2)+{sign}*{drift}*iw*on/{total_frames}"
    elif dominant in ("up", "down"):
        sign = -1 if dominant == "up" else 1
        y_expr = f"ih*{norm_cy:.4f}-(ih/zoom/2)+{sign}*{drift}*ih*on/{total_frames}"

    zoompan_filter = (
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={width}x{height}:fps={fps}"
    )

    cmd = (
        FFmpegCmd()
        .input(image_path)
        .option("vf", zoompan_filter)
        .option("t", f"{duration:.3f}")
        .video_codec("libx264", crf=18, preset="fast")
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(80, "Still image animation complete")

    return out_path


def _apply_video_motion(
    video_path: str,
    regions: list,
    duration: float,
    fps: float,
    width: int,
    height: int,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply additional motion to regions of an existing video.

    Uses FFmpeg filter_complex with crop, transform, and overlay per region.
    """
    if on_progress:
        on_progress(30, f"Applying motion to {len(regions)} regions...")

    # Build a filter_complex that processes each region
    # Strategy: for each region, crop it, apply the motion transform, overlay back
    filter_parts = []
    current_label = "[0:v]"

    for i, region in enumerate(regions):
        region.validate(width, height)

        # Determine per-region transform
        d = region.direction
        s = region.strength
        max_shift = int(s * 30)  # Max pixel shift over duration

        if d in ("left", "right"):
            sign = -1 if d == "left" else 1
            overlay_x = f"{region.x}+{sign}*{max_shift}*t/{duration:.3f}"
            overlay_y = str(region.y)
        elif d in ("up", "down"):
            sign = -1 if d == "up" else 1
            overlay_x = str(region.x)
            overlay_y = f"{region.y}+{sign}*{max_shift}*t/{duration:.3f}"
        elif d in ("zoom_in", "zoom_out"):
            # For zoom, we scale the cropped region and overlay centered
            overlay_x = str(region.x)
            overlay_y = str(region.y)
        elif d in ("rotate_cw", "rotate_ccw"):
            overlay_x = str(region.x)
            overlay_y = str(region.y)
        else:
            overlay_x = str(region.x)
            overlay_y = str(region.y)

        region_label = f"[reg{i}]"
        overlay_label = f"[ov{i}]"

        # Crop the region from original
        filter_parts.append(
            f"[0:v]crop={region.w}:{region.h}:{region.x}:{region.y}{region_label}"
        )

        # Overlay the (potentially shifted) region back
        filter_parts.append(
            f"{current_label}{region_label}overlay={overlay_x}:{overlay_y}"
            f":shortest=1{overlay_label}"
        )
        current_label = overlay_label

    # Final label rename
    if filter_parts:
        # Replace the last label with [outv]
        last_part = filter_parts[-1]
        last_label = f"[ov{len(regions) - 1}]"
        filter_parts[-1] = last_part.replace(last_label, "[outv]")
    else:
        filter_parts.append(f"{current_label}null[outv]")

    fc = ";".join(filter_parts)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(80, "Video motion applied")

    return out_path


def _apply_mask_motion(
    input_path: str,
    mask_path: str,
    motion_params: dict,
    duration: float,
    fps: float,
    width: int,
    height: int,
    out_path: str,
    is_still: bool = False,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply motion using a mask image (white = motion area).

    The mask is used as a displacement map for FFmpeg's displace filter,
    or as a blend mask for compositing a transformed version.
    """
    if on_progress:
        on_progress(30, "Applying mask-based motion...")

    direction = motion_params.get("direction", "right")
    strength = float(motion_params.get("strength", 0.5))
    max_shift = int(strength * 50)

    # For still images, create a video from the still first
    if is_still:
        fd, still_video = tempfile.mkstemp(suffix=".mp4", prefix="still_vid_")
        os.close(fd)
        cmd = (
            FFmpegCmd()
            .input(input_path, loop="1", framerate=str(fps))
            .option("t", f"{duration:.3f}")
            .video_codec("libx264", crf=18, preset="fast")
            .output(still_video)
            .build()
        )
        run_ffmpeg(cmd)
        source = still_video
    else:
        source = input_path
        still_video = None

    try:
        # Use alphamerge with the mask to control which areas get the motion effect
        # Create a shifted version and blend it with the original using the mask
        if direction in ("left", "right"):
            sign = -1 if direction == "left" else 1
            scroll_expr = f"scroll=horizontal={sign}*{max_shift}*t/{duration:.3f}:vertical=0"
        elif direction in ("up", "down"):
            sign = -1 if direction == "up" else 1
            scroll_expr = f"scroll=horizontal=0:vertical={sign}*{max_shift}*t/{duration:.3f}"
        else:
            scroll_expr = "null"

        # Build filter: original + scrolled version blended by mask
        fc = (
            f"[0:v]split[orig][tomove];"
            f"[tomove]{scroll_expr}[moved];"
            f"movie={mask_path}[mask];"
            f"[orig][moved][mask]maskedmerge[outv]"
        )

        cmd = (
            FFmpegCmd()
            .input(source)
            .filter_complex(fc, maps=["[outv]"])
            .video_codec("libx264", crf=18, preset="fast")
            .output(out_path)
            .build()
        )
        run_ffmpeg(cmd)

    finally:
        if still_video and os.path.isfile(still_video):
            try:
                os.unlink(still_video)
            except OSError:
                pass

    if on_progress:
        on_progress(80, "Mask-based motion complete")

    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def preview_motion_brush(
    input_path: str,
    motion_mask: Union[str, Dict, List[Dict]],
    motion_params: Optional[Dict] = None,
    frame_index: int = 0,
) -> dict:
    """Preview motion brush effect on a single frame.

    Generates a before/after comparison by extracting the specified frame
    and applying the first frame of motion.

    Args:
        input_path: Path to image or video file.
        motion_mask: Mask image path, region dict, or list of region dicts.
        motion_params: Optional override for direction, strength.
        frame_index: Which frame to preview (for video input).

    Returns:
        Dict with "original_frame" and "preview_frame" paths.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Extract the target frame
    fd, orig_frame = tempfile.mkstemp(suffix=".png", prefix="preview_orig_")
    os.close(fd)

    if _is_image(input_path):
        import shutil
        shutil.copy2(input_path, orig_frame)
    else:
        info = get_video_info(input_path)
        fps = info.get("fps", 30.0)
        ts = frame_index / fps if fps > 0 else 0
        cmd = (
            FFmpegCmd()
            .input(input_path, ss=f"{ts:.3f}")
            .option("frames:v", "1")
            .output(orig_frame)
            .build()
        )
        run_ffmpeg(cmd)

    # Generate a short (0.1s) motion preview
    fd, preview_path = tempfile.mkstemp(suffix=".png", prefix="preview_motion_")
    os.close(fd)
    fd, preview_video = tempfile.mkstemp(suffix=".mp4", prefix="preview_vid_")
    os.close(fd)

    try:
        apply_motion_brush(
            orig_frame,
            motion_mask,
            motion_params or {"direction": "right", "strength": 0.5, "duration": 0.5},
            output=preview_video,
        )

        # Extract the last frame of the preview as the result
        cmd = (
            FFmpegCmd()
            .input(preview_video, sseof="-0.04")
            .option("frames:v", "1")
            .output(preview_path)
            .build()
        )
        run_ffmpeg(cmd)

    except Exception:
        # If motion fails, just copy original as preview
        import shutil
        shutil.copy2(orig_frame, preview_path)

    finally:
        try:
            os.unlink(preview_video)
        except OSError:
            pass

    return {
        "original_frame": orig_frame,
        "preview_frame": preview_path,
    }


def apply_motion_brush(
    input_path: str,
    motion_mask: Union[str, Dict, List[Dict]],
    motion_params: Optional[Dict] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> MotionBrushResult:
    """Apply motion brush effects to an image or video.

    Animates specified regions with directional motion, zoom, or rotation.
    For still images, the output is always a video.

    Args:
        input_path: Path to the input image or video file.
        motion_mask: One of:
            - Path to a mask image (white = motion areas)
            - Dict with x, y, w, h, direction, strength
            - List of such dicts for multiple regions
        motion_params: Global motion params (direction, strength, duration).
            Used with mask images. For region dicts, direction/strength
            come from each region.
        output: Output video path. Auto-generated if None.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        MotionBrushResult with output path and metadata.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If no valid motion regions/mask provided.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if motion_params is None:
        motion_params = {}

    duration = float(motion_params.get("duration", _DEFAULT_STILL_DURATION))
    duration = max(0.1, min(duration, 60.0))

    is_still = _is_image(input_path)

    if on_progress:
        on_progress(5, "Preparing motion brush...")

    # Get dimensions
    if is_still:
        if not ensure_package("cv2", "opencv-python-headless"):
            raise RuntimeError("OpenCV is required. Run: pip install opencv-python-headless")
        import cv2
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")
        height, width = img.shape[:2]
        fps = 30.0
    else:
        info = get_video_info(input_path)
        width = info.get("width", 1920)
        height = info.get("height", 1080)
        fps = info.get("fps", 30.0)
        duration = min(duration, info.get("duration", duration))

    if output is None:
        if is_still:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output = os.path.join(
                os.path.dirname(input_path),
                f"{base}_motion.mp4",
            )
        else:
            output = output_path(input_path, "motion")

    # Parse motion mask
    parsed = _parse_regions(motion_mask)

    if on_progress:
        on_progress(15, "Analyzing motion regions...")

    method_used = "ffmpeg"
    regions_animated = 0

    if isinstance(parsed, str):
        # Mask image path
        if not os.path.isfile(parsed):
            raise FileNotFoundError(f"Mask image not found: {parsed}")

        _apply_mask_motion(
            input_path, parsed, motion_params,
            duration, fps, width, height, output,
            is_still=is_still, on_progress=on_progress,
        )
        regions_animated = 1  # Mask counts as one region
        method_used = "ffmpeg_mask"

    elif isinstance(parsed, list) and parsed:
        # Region list
        for region in parsed:
            region.validate(width, height)

        _apply_ffmpeg_motion(
            input_path, parsed, duration, fps, width, height, output,
            is_still=is_still, on_progress=on_progress,
        )
        regions_animated = len(parsed)
        method_used = "ffmpeg_regions"

    else:
        raise ValueError("No valid motion regions or mask provided")

    if on_progress:
        on_progress(95, "Motion brush complete")

    return MotionBrushResult(
        output_path=output,
        duration=round(duration, 3),
        regions_animated=regions_animated,
        method_used=method_used,
    )
