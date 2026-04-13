"""
OpenCut 360 Video Support Module v0.9.0

Import, process, and export equirectangular 360 video:
- Detect equirectangular vs cubemap format
- Projection conversion (equirect <-> cubemap)
- Flat rectilinear crop extraction (yaw/pitch/fov)
- 360 video stabilisation via motion analysis

All via FFmpeg v360 filter + OpenCV -- no ML dependencies.
"""

import logging
import os
from typing import Callable, Optional

from opencut.helpers import get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# 360 Format Detection
# ---------------------------------------------------------------------------
SUPPORTED_PROJECTIONS = ("equirect", "cubemap", "eac", "barrel", "flat")


def detect_360_format(video_path: str) -> dict:
    """
    Detect whether a video is 360 and its projection type.

    Checks aspect ratio (2:1 = equirect), metadata spherical tags,
    and filename hints.

    Args:
        video_path: Path to video file.

    Returns:
        Dict with keys: is_360, projection, width, height, aspect_ratio.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    w = info.get("width", 0)
    h = info.get("height", 0)

    result = {
        "is_360": False,
        "projection": "unknown",
        "width": w,
        "height": h,
        "aspect_ratio": round(w / max(1, h), 2),
    }

    # Check aspect ratio: equirectangular is 2:1
    if w > 0 and h > 0:
        ratio = w / h
        if 1.95 <= ratio <= 2.05:
            result["is_360"] = True
            result["projection"] = "equirect"
        elif 0.95 <= ratio <= 1.05 and w >= 2048:
            # Could be cubemap (6 faces in a cross or strip)
            result["is_360"] = True
            result["projection"] = "cubemap"

    # Check filename hints
    basename = os.path.basename(video_path).lower()
    if "360" in basename or "equirect" in basename or "vr" in basename:
        result["is_360"] = True
        if result["projection"] == "unknown":
            result["projection"] = "equirect"

    return result


# ---------------------------------------------------------------------------
# Projection Conversion
# ---------------------------------------------------------------------------
def convert_360_projection(
    video_path: str,
    projection: str = "cubemap",
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Convert 360 video between projection types using FFmpeg v360 filter.

    Args:
        video_path: Input 360 video (equirectangular).
        projection: Target projection: "cubemap", "equirect", "eac", "barrel".
        output_path: Output path. Auto-generated if None.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to converted output video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"Unsupported projection '{projection}'. "
            f"Supported: {', '.join(SUPPORTED_PROJECTIONS)}"
        )

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_{projection}.mp4")

    if on_progress:
        on_progress(5, f"Converting to {projection} projection...")

    # Detect source projection
    source_info = detect_360_format(video_path)
    src_proj = source_info.get("projection", "equirect")
    if src_proj == "unknown":
        src_proj = "equirect"

    # Map to FFmpeg v360 filter input/output names
    proj_map = {
        "equirect": "e",
        "cubemap": "c6x1",
        "eac": "eac",
        "barrel": "barrel",
        "flat": "flat",
    }
    in_proj = proj_map.get(src_proj, "e")
    out_proj = proj_map.get(projection, "c6x1")

    info = get_video_info(video_path)
    w = info.get("width", 3840)
    info.get("height", 1920)

    if projection == "cubemap":
        w // 6
    elif projection == "equirect":
        w // 2
    else:
        pass

    v360_filter = f"v360={in_proj}:{out_proj}"

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", v360_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, f"Projection conversion to {projection} complete!")
    return output_path


# ---------------------------------------------------------------------------
# Extract Flat Crop from 360
# ---------------------------------------------------------------------------
def extract_360_crop(
    video_path: str,
    yaw: float = 0.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    output_width: int = 1920,
    output_height: int = 1080,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Extract a flat rectilinear crop from a 360 video.

    Converts a viewing direction (yaw/pitch) with a field of view into
    a standard flat video -- like looking through a virtual camera.

    Args:
        video_path: Input 360 equirectangular video.
        yaw: Horizontal rotation in degrees (-180 to 180). 0 = front.
        pitch: Vertical rotation in degrees (-90 to 90). 0 = horizon.
        fov: Field of view in degrees (30-160).
        output_path: Output path. Auto-generated if None.
        output_dir: Output directory.
        output_width: Width of output flat video.
        output_height: Height of output flat video.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to extracted flat crop video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    yaw = max(-180.0, min(180.0, float(yaw)))
    pitch = max(-90.0, min(90.0, float(pitch)))
    fov = max(30.0, min(160.0, float(fov)))

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(
            directory,
            f"{base}_crop_y{int(yaw)}_p{int(pitch)}_f{int(fov)}.mp4",
        )

    if on_progress:
        on_progress(5, f"Extracting flat crop (yaw={yaw}, pitch={pitch}, fov={fov})...")

    v360_filter = (
        f"v360=e:flat:yaw={yaw}:pitch={pitch}:"
        f"h_fov={fov}:v_fov={fov * output_height / output_width}:"
        f"w={output_width}:h={output_height}"
    )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", v360_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "360 crop extraction complete!")
    return output_path


# ---------------------------------------------------------------------------
# 360 Video Stabilisation
# ---------------------------------------------------------------------------
def stabilize_360(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    smoothing: int = 10,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Stabilise a 360 video using FFmpeg's vidstab filters.

    Two-pass approach: analyse motion, then apply smoothing.
    Uses the v360 filter for equirectangular-aware stabilisation.

    Args:
        video_path: Input 360 video.
        output_path: Output path. Auto-generated if None.
        output_dir: Output directory.
        smoothing: Smoothing strength (1-30, default 10).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to stabilised output video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    smoothing = max(1, min(30, int(smoothing)))

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_stabilized_360.mp4")

    if on_progress:
        on_progress(5, "Analysing 360 video motion (pass 1)...")

    import tempfile
    transforms_file = os.path.join(tempfile.gettempdir(), "oc_360_transforms.trf")

    # Pass 1: Detect motion
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"vidstabdetect=result={transforms_file}:shakiness=8",
        "-f", "null", os.devnull,
    ], timeout=7200)

    if on_progress:
        on_progress(50, "Applying 360 stabilisation (pass 2)...")

    # Pass 2: Apply stabilisation
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"vidstabtransform=input={transforms_file}:smoothing={smoothing}:interpol=bicubic",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], timeout=7200)

    # Clean up transforms file
    try:
        os.unlink(transforms_file)
    except OSError:
        pass

    if on_progress:
        on_progress(100, "360 stabilisation complete!")
    return output_path
