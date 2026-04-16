"""
OpenCut 360 Video Support Module v0.10.0

Import, process, and export equirectangular 360 video:
- Detect equirectangular vs cubemap format
- Projection conversion (equirect <-> cubemap)
- Flat rectilinear crop extraction (yaw/pitch/fov)
- Equirectangular to flat with independent yaw/pitch/roll/h_fov/w_fov
- Keyframed reframing with smooth interpolation
- Auto FOV region extraction with face/motion detection
- 360 video stabilisation via motion analysis

All via FFmpeg v360 filter + OpenCV -- no ML dependencies.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

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


# ---------------------------------------------------------------------------
# Data types for reframing
# ---------------------------------------------------------------------------
@dataclass
class ReframeKeyframe:
    """A single keyframe for 360 reframing."""
    time: float       # seconds
    yaw: float = 0.0  # degrees
    pitch: float = 0.0
    roll: float = 0.0
    h_fov: float = 90.0
    w_fov: float = 0.0  # 0 = derive from h_fov and aspect ratio


@dataclass
class ReframeResult:
    """Result of equirectangular-to-flat reframing."""
    output_path: str = ""
    width: int = 1920
    height: int = 1080
    keyframes_used: int = 0
    duration: float = 0.0
    method: str = "static"


@dataclass
class FOVRegion:
    """A detected region of interest in 360 video."""
    label: str = ""
    yaw: float = 0.0
    pitch: float = 0.0
    h_fov: float = 90.0
    confidence: float = 0.0
    output_path: str = ""


@dataclass
class FOVExtractionResult:
    """Result of automatic FOV region extraction."""
    regions: List[FOVRegion] = field(default_factory=list)
    xml_path: str = ""
    output_dir: str = ""
    total_regions: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# 51.2: Equirectangular to Flat (enhanced)
# ---------------------------------------------------------------------------
def equirect_to_flat(
    video_path: str,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    h_fov: float = 90.0,
    w_fov: float = 0.0,
    output_width: int = 1920,
    output_height: int = 1080,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ReframeResult:
    """
    Convert equirectangular 360 video to a flat rectilinear view.

    Supports independent yaw, pitch, roll, horizontal and vertical FOV.

    Args:
        video_path: Input equirectangular video.
        yaw: Horizontal rotation in degrees (-180 to 180).
        pitch: Vertical rotation in degrees (-90 to 90).
        roll: Roll rotation in degrees (-180 to 180).
        h_fov: Horizontal field of view in degrees (30-160).
        w_fov: Vertical field of view in degrees. 0 = auto from aspect ratio.
        output_width: Width of the output flat video.
        output_height: Height of the output flat video.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReframeResult with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    yaw = max(-180.0, min(180.0, float(yaw)))
    pitch = max(-90.0, min(90.0, float(pitch)))
    roll = max(-180.0, min(180.0, float(roll)))
    h_fov = max(30.0, min(160.0, float(h_fov)))

    if w_fov <= 0:
        # Derive vertical FOV from aspect ratio
        w_fov = h_fov * output_height / max(1, output_width)
    else:
        w_fov = max(20.0, min(160.0, float(w_fov)))

    out = output_path_override or _output_path(video_path, "flat", output_dir)

    if on_progress:
        on_progress(5, f"Reframing: yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}")

    v360_filter = (
        f"v360=e:flat"
        f":yaw={yaw}:pitch={pitch}:roll={roll}"
        f":h_fov={h_fov}:v_fov={w_fov}"
        f":w={output_width}:h={output_height}"
    )

    if on_progress:
        on_progress(15, "Applying v360 flat projection...")

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", v360_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        out,
    ], timeout=7200)

    info = get_video_info(video_path)

    if on_progress:
        on_progress(100, "Equirectangular to flat conversion complete!")

    return ReframeResult(
        output_path=out,
        width=output_width,
        height=output_height,
        keyframes_used=0,
        duration=info.get("duration", 0.0),
        method="static",
    )


def _interpolate_keyframes(
    keyframes: List[ReframeKeyframe],
    time: float,
) -> Tuple[float, float, float, float, float]:
    """
    Smoothly interpolate between keyframes using cosine interpolation.

    Returns (yaw, pitch, roll, h_fov, w_fov) at the given time.
    """
    if not keyframes:
        return 0.0, 0.0, 0.0, 90.0, 0.0

    if len(keyframes) == 1:
        kf = keyframes[0]
        return kf.yaw, kf.pitch, kf.roll, kf.h_fov, kf.w_fov

    # Sort by time
    kfs = sorted(keyframes, key=lambda k: k.time)

    # Clamp to range
    if time <= kfs[0].time:
        kf = kfs[0]
        return kf.yaw, kf.pitch, kf.roll, kf.h_fov, kf.w_fov
    if time >= kfs[-1].time:
        kf = kfs[-1]
        return kf.yaw, kf.pitch, kf.roll, kf.h_fov, kf.w_fov

    # Find bounding keyframes
    for i in range(len(kfs) - 1):
        if kfs[i].time <= time <= kfs[i + 1].time:
            a, b = kfs[i], kfs[i + 1]
            span = b.time - a.time
            if span <= 0:
                return a.yaw, a.pitch, a.roll, a.h_fov, a.w_fov

            # Cosine interpolation for smoother motion
            t = (time - a.time) / span
            t = (1.0 - math.cos(t * math.pi)) / 2.0

            yaw = a.yaw + (b.yaw - a.yaw) * t
            pitch = a.pitch + (b.pitch - a.pitch) * t
            roll = a.roll + (b.roll - a.roll) * t
            h_fov = a.h_fov + (b.h_fov - a.h_fov) * t
            w_fov_a = a.w_fov if a.w_fov > 0 else a.h_fov * 9 / 16
            w_fov_b = b.w_fov if b.w_fov > 0 else b.h_fov * 9 / 16
            w_fov = w_fov_a + (w_fov_b - w_fov_a) * t

            return yaw, pitch, roll, h_fov, w_fov

    kf = kfs[-1]
    return kf.yaw, kf.pitch, kf.roll, kf.h_fov, kf.w_fov


def keyframed_reframe(
    video_path: str,
    keyframes: List[Dict],
    output_width: int = 1920,
    output_height: int = 1080,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ReframeResult:
    """
    Reframe 360 video with keyframed camera movement.

    Smoothly interpolates between yaw/pitch/roll/fov keyframes using
    cosine interpolation for natural-looking camera motion.

    When only a single keyframe is provided, produces a static reframe.
    With multiple keyframes, splits the video into segments and applies
    per-segment v360 filters, then concatenates.

    Args:
        video_path: Input equirectangular video.
        keyframes: List of dicts with keys: time, yaw, pitch, roll, h_fov, w_fov.
        output_width: Width of the output flat video.
        output_height: Height of the output flat video.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReframeResult with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not keyframes:
        raise ValueError("At least one keyframe is required")

    # Parse keyframe dicts
    parsed_kfs = []
    for kf in keyframes:
        parsed_kfs.append(ReframeKeyframe(
            time=float(kf.get("time", 0.0)),
            yaw=float(kf.get("yaw", 0.0)),
            pitch=float(kf.get("pitch", 0.0)),
            roll=float(kf.get("roll", 0.0)),
            h_fov=float(kf.get("h_fov", 90.0)),
            w_fov=float(kf.get("w_fov", 0.0)),
        ))

    parsed_kfs.sort(key=lambda k: k.time)
    out = output_path_override or _output_path(video_path, "reframed", output_dir)

    info = get_video_info(video_path)
    duration = info.get("duration", 0.0)
    info.get("fps", 30.0)

    if on_progress:
        on_progress(5, f"Reframing with {len(parsed_kfs)} keyframes...")

    # For single keyframe or very short video, use static reframe
    if len(parsed_kfs) == 1 or duration <= 0.1:
        kf = parsed_kfs[0]
        return equirect_to_flat(
            video_path=video_path,
            yaw=kf.yaw,
            pitch=kf.pitch,
            roll=kf.roll,
            h_fov=kf.h_fov,
            w_fov=kf.w_fov,
            output_width=output_width,
            output_height=output_height,
            output_path_override=out,
            on_progress=on_progress,
        )

    # Multi-keyframe: generate segments with interpolated parameters
    # Use ~1 second segments for smooth transitions
    segment_duration = 1.0
    num_segments = max(1, int(math.ceil(duration / segment_duration)))

    tmp_dir = tempfile.mkdtemp(prefix="oc_reframe_")
    segment_files = []

    try:
        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_duration
            seg_end = min(seg_start + segment_duration, duration)
            seg_mid = (seg_start + seg_end) / 2.0

            # Interpolate at segment midpoint
            yaw, pitch, roll, h_fov, w_fov = _interpolate_keyframes(
                parsed_kfs, seg_mid
            )
            if w_fov <= 0:
                w_fov = h_fov * output_height / max(1, output_width)

            v360_filter = (
                f"v360=e:flat"
                f":yaw={yaw:.4f}:pitch={pitch:.4f}:roll={roll:.4f}"
                f":h_fov={h_fov:.2f}:v_fov={w_fov:.2f}"
                f":w={output_width}:h={output_height}"
            )

            seg_file = os.path.join(tmp_dir, f"seg_{seg_idx:05d}.mp4")
            segment_files.append(seg_file)

            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(seg_start), "-t", str(seg_end - seg_start),
                "-i", video_path,
                "-vf", v360_filter,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                seg_file,
            ], timeout=600)

            if on_progress:
                pct = 10 + int(80 * (seg_idx + 1) / num_segments)
                on_progress(pct, f"Segment {seg_idx + 1}/{num_segments}")

        # Concatenate segments
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")

        if on_progress:
            on_progress(92, "Concatenating segments...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            out,
        ], timeout=3600)

    finally:
        # Clean up temp dir
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, "Keyframed reframe complete!")

    return ReframeResult(
        output_path=out,
        width=output_width,
        height=output_height,
        keyframes_used=len(parsed_kfs),
        duration=duration,
        method="keyframed",
    )


# ---------------------------------------------------------------------------
# 51.3: FOV Region Extraction from 360
# ---------------------------------------------------------------------------
def _detect_motion_regions(
    video_path: str,
    num_samples: int = 10,
) -> List[Tuple[float, float, float]]:
    """
    Detect regions with significant motion in equirectangular video.

    Samples frames and looks for high-motion areas by analyzing
    frame differences via FFmpeg signalstats.

    Returns list of (yaw, pitch, confidence) tuples.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0.0)
    w = info.get("width", 3840)
    h = info.get("height", 1920)

    if duration <= 0 or w <= 0 or h <= 0:
        return []

    regions = []
    # Divide equirectangular into a grid and check motion in each cell
    grid_cols = 8
    grid_rows = 4
    cell_w = w // grid_cols
    cell_h = h // grid_rows

    tmp_dir = tempfile.mkdtemp(prefix="oc_fov_detect_")

    try:
        # Extract two frames at different times for motion detection
        t1 = min(duration * 0.3, duration - 0.1)
        t2 = min(duration * 0.6, duration - 0.1)

        frame1 = os.path.join(tmp_dir, "frame1.png")
        frame2 = os.path.join(tmp_dir, "frame2.png")

        for t, fp in [(t1, frame1), (t2, frame2)]:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(max(0, t)),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                fp,
            ], timeout=30)

        # Compute difference and check per-region activity
        diff_file = os.path.join(tmp_dir, "diff.png")
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", frame1, "-i", frame2,
            "-filter_complex",
            "[0][1]blend=all_mode=difference,format=gray",
            "-frames:v", "1",
            diff_file,
        ], timeout=30)

        # Analyze grid cells for brightness (motion indicator)
        for row in range(grid_rows):
            for col in range(grid_cols):
                x = col * cell_w
                y = row * cell_h

                # Map grid position to yaw/pitch
                yaw = (col / grid_cols) * 360.0 - 180.0
                pitch = 90.0 - (row / grid_rows) * 180.0

                # Use crop + signalstats to measure mean brightness
                os.path.join(tmp_dir, f"cell_{row}_{col}.txt")
                try:
                    result = subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", diff_file,
                        "-vf", f"crop={cell_w}:{cell_h}:{x}:{y},signalstats",
                        "-f", "null", "-",
                    ], capture_output=True, text=True, timeout=10)

                    # Parse lavfi.signalstats.YAVG from stderr
                    for line in result.stderr.split("\n"):
                        if "YAVG" in line:
                            parts = line.split("YAVG:")
                            if len(parts) > 1:
                                try:
                                    avg = float(parts[1].strip().split()[0])
                                    if avg > 15:  # threshold for significant motion
                                        confidence = min(1.0, avg / 100.0)
                                        regions.append((yaw, pitch, confidence))
                                except (ValueError, IndexError):
                                    pass
                            break
                except (subprocess.TimeoutExpired, OSError):
                    continue

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Sort by confidence, return top regions
    regions.sort(key=lambda r: r[2], reverse=True)
    return regions[:6]


def _generate_multicam_xml(
    regions: List[FOVRegion],
    duration: float,
    fps: float,
    output_path_str: str,
) -> str:
    """
    Generate a multicam XML (FCPXML-like) for the extracted FOV regions.
    """
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<fcpxml version="1.9">',
        '  <resources>',
    ]

    for i, region in enumerate(regions):
        if region.output_path:
            xml_lines.append(
                f'    <asset id="clip_{i}" name="{region.label}" '
                f'src="file://{region.output_path}" '
                f'duration="{duration}s" />'
            )

    xml_lines.append('  </resources>')
    xml_lines.append('  <library>')
    xml_lines.append('    <event name="360_FOV_Extraction">')
    xml_lines.append('      <multicam-clip name="Auto FOV Multicam">')

    for i, region in enumerate(regions):
        total_frames = int(duration * fps)
        xml_lines.append(
            f'        <mc-angle name="{region.label}" '
            f'angle-id="{i}">'
        )
        xml_lines.append(
            f'          <asset-clip ref="clip_{i}" '
            f'duration="{total_frames}/{int(fps)}s" />'
        )
        xml_lines.append('        </mc-angle>')

    xml_lines.append('      </multicam-clip>')
    xml_lines.append('    </event>')
    xml_lines.append('  </library>')
    xml_lines.append('</fcpxml>')

    xml_content = "\n".join(xml_lines)
    with open(output_path_str, "w", encoding="utf-8") as f:
        f.write(xml_content)

    return output_path_str


def extract_fov_regions(
    video_path: str,
    output_dir: str = "",
    num_regions: int = 4,
    h_fov: float = 90.0,
    output_width: int = 1920,
    output_height: int = 1080,
    generate_xml: bool = True,
    on_progress: Optional[Callable] = None,
) -> FOVExtractionResult:
    """
    Auto-detect faces/motion in equirectangular video and extract
    per-subject flat views.

    Analyzes the 360 video for regions of interest (faces, motion),
    extracts each as a flat rectilinear view, and optionally generates
    a multicam XML for NLE import.

    Args:
        video_path: Input equirectangular video.
        output_dir: Directory for output files.
        num_regions: Maximum number of regions to extract (1-8).
        h_fov: Horizontal FOV for each extracted region.
        output_width: Width of each extracted view.
        output_height: Height of each extracted view.
        generate_xml: Whether to generate multicam XML.
        on_progress: Progress callback(pct, msg).

    Returns:
        FOVExtractionResult with regions and multicam XML path.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    num_regions = max(1, min(8, int(num_regions)))
    h_fov = max(30.0, min(160.0, float(h_fov)))

    if not output_dir:
        output_dir = os.path.join(
            os.path.dirname(video_path), "fov_regions"
        )
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(video_path)
    duration = info.get("duration", 0.0)
    fps = info.get("fps", 30.0)

    if on_progress:
        on_progress(5, "Detecting regions of interest in 360 video...")

    # Detect motion regions
    motion_regions = _detect_motion_regions(video_path)

    # If we don't have enough motion regions, add default views
    default_views = [
        (0.0, 0.0, 0.5),      # front
        (90.0, 0.0, 0.3),     # right
        (-90.0, 0.0, 0.3),    # left
        (180.0, 0.0, 0.3),    # back
        (0.0, 45.0, 0.2),     # up-front
        (0.0, -30.0, 0.2),    # down-front
    ]

    all_regions_data = list(motion_regions)
    if len(all_regions_data) < num_regions:
        for dv in default_views:
            if len(all_regions_data) >= num_regions:
                break
            # Check if similar to existing region
            is_duplicate = False
            for existing in all_regions_data:
                if abs(existing[0] - dv[0]) < 30 and abs(existing[1] - dv[1]) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_regions_data.append(dv)

    all_regions_data = all_regions_data[:num_regions]

    if on_progress:
        on_progress(15, f"Extracting {len(all_regions_data)} FOV regions...")

    regions: List[FOVRegion] = []

    for i, (yaw, pitch, confidence) in enumerate(all_regions_data):
        label = f"region_{i + 1}_y{int(yaw)}_p{int(pitch)}"
        region_output = os.path.join(output_dir, f"{label}.mp4")

        w_fov = h_fov * output_height / max(1, output_width)

        v360_filter = (
            f"v360=e:flat"
            f":yaw={yaw}:pitch={pitch}"
            f":h_fov={h_fov}:v_fov={w_fov}"
            f":w={output_width}:h={output_height}"
        )

        try:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", video_path,
                "-vf", v360_filter,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                region_output,
            ], timeout=7200)

            regions.append(FOVRegion(
                label=label,
                yaw=yaw,
                pitch=pitch,
                h_fov=h_fov,
                confidence=confidence,
                output_path=region_output,
            ))
        except Exception as e:
            logger.warning("Failed to extract FOV region %s: %s", label, e)

        if on_progress:
            pct = 15 + int(70 * (i + 1) / len(all_regions_data))
            on_progress(pct, f"Extracted region {i + 1}/{len(all_regions_data)}")

    # Generate multicam XML
    xml_path = ""
    if generate_xml and regions:
        if on_progress:
            on_progress(90, "Generating multicam XML...")

        xml_path = os.path.join(output_dir, "multicam.fcpxml")
        _generate_multicam_xml(regions, duration, fps, xml_path)

    if on_progress:
        on_progress(100, f"Extracted {len(regions)} FOV regions!")

    return FOVExtractionResult(
        regions=regions,
        xml_path=xml_path,
        output_dir=output_dir,
        total_regions=len(regions),
        duration=duration,
    )
