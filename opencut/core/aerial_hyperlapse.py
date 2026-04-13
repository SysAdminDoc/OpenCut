"""
OpenCut Aerial Hyperlapse Module

Create stabilised aerial hyperlapses from drone footage by sampling
frames at GPS-distance intervals and applying aggressive stabilisation.

Pipeline:
  1. Parse GPS track from DJI SRT / GPX
  2. Sample frame indices by GPS distance intervals
  3. Extract sampled frames via FFmpeg
  4. Re-encode with aggressive stabilisation (``vidstabdetect`` / ``vidstabtransform``)
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class GpsPoint:
    """A single GPS coordinate with optional metadata."""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    timestamp: float = 0.0


@dataclass
class HyperlapseResult:
    """Result of aerial hyperlapse creation."""
    output_path: str = ""
    total_frames: int = 0
    sampled_frames: int = 0
    total_distance_m: float = 0.0
    speed_factor: float = 1.0
    stabilised: bool = False


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _haversine(p1: GpsPoint, p2: GpsPoint) -> float:
    """Haversine distance in metres between two GPS points."""
    R = 6371000.0
    lat1 = math.radians(p1.latitude)
    lat2 = math.radians(p2.latitude)
    dlat = math.radians(p2.latitude - p1.latitude)
    dlon = math.radians(p2.longitude - p1.longitude)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_by_gps_distance(
    gps_points: List[Dict],
    interval_meters: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> List[int]:
    """Select frame indices sampled at fixed GPS distance intervals.

    Walks through the GPS track and picks the frame closest to each
    distance milestone.

    Args:
        gps_points: List of dicts with ``latitude``, ``longitude`` keys
            (and optionally ``altitude``, ``timestamp``).
        interval_meters: Distance between samples in metres.
        on_progress: Optional ``(pct, msg)`` callback.

    Returns:
        List of 0-based frame indices to keep.
    """
    if on_progress:
        on_progress(10, "Computing GPS distance sampling...")

    if not gps_points or interval_meters <= 0:
        return list(range(len(gps_points)))

    points = [
        GpsPoint(
            latitude=float(p.get("latitude", 0)),
            longitude=float(p.get("longitude", 0)),
            altitude=float(p.get("altitude", 0)),
            timestamp=float(p.get("timestamp", 0)),
        )
        for p in gps_points
    ]

    selected = [0]
    accumulated = 0.0

    for i in range(1, len(points)):
        d = _haversine(points[i - 1], points[i])
        accumulated += d
        if accumulated >= interval_meters:
            selected.append(i)
            accumulated = 0.0

    # Always include last frame
    if selected[-1] != len(points) - 1:
        selected.append(len(points) - 1)

    if on_progress:
        on_progress(100, f"Selected {len(selected)} frames by GPS distance")

    logger.info("GPS distance sampling: %d/%d frames at %.1fm intervals",
                len(selected), len(points), interval_meters)
    return selected


def stabilize_aerial(
    video_path: str,
    out_path: str = "",
    shakiness: int = 10,
    accuracy: int = 15,
    smoothing: int = 30,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply aggressive stabilisation to an aerial video clip.

    Uses FFmpeg's vidstab two-pass process for maximum quality.

    Args:
        video_path: Input video path.
        out_path: Output path (auto-generated if empty).
        shakiness: vidstabdetect shakiness (1-10).
        accuracy: vidstabdetect accuracy (1-15).
        smoothing: vidstabtransform smoothing (frames).
        on_progress: Optional progress callback.

    Returns:
        Path to the stabilised output file.
    """
    if on_progress:
        on_progress(10, "Starting aerial stabilisation...")

    if not out_path:
        out_path = output_path(video_path, "stabilized")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    shakiness = max(1, min(10, shakiness))
    accuracy = max(1, min(15, accuracy))
    smoothing = max(1, min(100, smoothing))

    # Two-pass vidstab
    trf_fd = tempfile.NamedTemporaryFile(suffix=".trf", delete=False)
    trf_path = trf_fd.name
    trf_fd.close()

    try:
        # Pass 1: detect
        if on_progress:
            on_progress(20, "Stabilisation pass 1: motion analysis...")

        detect_filter = (
            f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}"
            f":result='{trf_path}'"
        )

        cmd1 = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-vf", detect_filter,
            "-f", "null",
            os.devnull,
        ]
        run_ffmpeg(cmd1)

        # Pass 2: transform
        if on_progress:
            on_progress(60, "Stabilisation pass 2: applying transform...")

        transform_filter = (
            f"vidstabtransform=input='{trf_path}':smoothing={smoothing}"
            f":interpol=bicubic:crop=black:zoom=5,unsharp=5:5:0.8"
        )

        cmd2 = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-vf", transform_filter,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "copy",
            out_path,
        ]
        run_ffmpeg(cmd2)

    finally:
        try:
            os.unlink(trf_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Stabilisation complete")

    logger.info("Stabilised aerial video: %s", out_path)
    return out_path


def create_aerial_hyperlapse(
    video_path: str,
    srt_path: str,
    out_path: str = "",
    interval_meters: float = 10.0,
    speed_factor: float = 0.0,
    stabilize: bool = True,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> HyperlapseResult:
    """Create a GPS-sampled, stabilised aerial hyperlapse.

    Full pipeline: parse GPS, sample by distance, extract frames,
    optionally stabilise.

    Args:
        video_path: Source drone video.
        srt_path: DJI SRT or GPX file with GPS data.
        out_path: Output path (auto-generated if empty).
        interval_meters: GPS distance interval for frame sampling.
        speed_factor: If > 0, override GPS sampling with simple speed-up.
        stabilize: Whether to apply aggressive stabilisation.
        fps: Output frame rate.
        on_progress: Optional progress callback.

    Returns:
        HyperlapseResult with output path and metadata.
    """
    if on_progress:
        on_progress(5, "Parsing GPS data...")

    if not out_path:
        out_path = output_path(video_path, "hyperlapse")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Parse GPS track
    from opencut.core.flight_path_map import parse_gps_track
    gps_points = parse_gps_track(srt_path)

    info = get_video_info(video_path)
    total_frames = int(info.get("duration", 0) * info.get("fps", 30))

    # Build GPS point dicts for sampling
    gps_dicts = [{"latitude": p.latitude, "longitude": p.longitude,
                   "altitude": p.altitude, "timestamp": p.timestamp}
                  for p in gps_points]

    if speed_factor > 0:
        # Simple speed-up mode
        if on_progress:
            on_progress(20, f"Speed-up mode: {speed_factor}x...")

        setpts = f"setpts={1.0/speed_factor}*PTS"
        atempo_val = min(speed_factor, 2.0)

        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-vf", setpts,
            "-af", f"atempo={atempo_val}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-r", str(fps),
            out_path,
        ]
        run_ffmpeg(cmd)

        sampled_count = max(1, int(total_frames / speed_factor))

    else:
        # GPS distance sampling
        if on_progress:
            on_progress(20, "Sampling frames by GPS distance...")

        selected_indices = sample_by_gps_distance(
            gps_dicts, interval_meters=interval_meters,
        )
        sampled_count = len(selected_indices)

        # Use select filter with frame numbers
        if selected_indices and total_frames > 0:
            # Map GPS point indices to approximate frame numbers
            frame_ratio = total_frames / max(1, len(gps_dicts))
            frame_nums = [int(idx * frame_ratio) for idx in selected_indices]
            frame_nums = sorted(set(fn for fn in frame_nums if fn < total_frames))

            if frame_nums:
                select_expr = "+".join(f"eq(n,{fn})" for fn in frame_nums[:500])
                vf = f"select='{select_expr}',setpts=N/{fps}/TB"

                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-i", video_path,
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-r", str(fps),
                    "-an",
                    out_path,
                ]

                if on_progress:
                    on_progress(50, "Extracting sampled frames...")

                run_ffmpeg(cmd)
            else:
                # Fallback: simple speed-up
                cmd = [
                    get_ffmpeg_path(), "-y",
                    "-i", video_path,
                    "-vf", "setpts=0.1*PTS",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-r", str(fps), "-an",
                    out_path,
                ]
                run_ffmpeg(cmd)
        else:
            # No valid GPS data, fall back to 10x speed
            cmd = [
                get_ffmpeg_path(), "-y",
                "-i", video_path,
                "-vf", "setpts=0.1*PTS",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", str(fps), "-an",
                out_path,
            ]
            run_ffmpeg(cmd)

    # Optionally stabilise
    stabilised = False
    if stabilize:
        if on_progress:
            on_progress(70, "Stabilising hyperlapse...")

        tmp_stab = out_path + ".stab.mp4"
        try:
            stabilize_aerial(
                video_path=out_path,
                out_path=tmp_stab,
                on_progress=on_progress,
            )
            # Replace output with stabilised version
            os.replace(tmp_stab, out_path)
            stabilised = True
        except Exception as e:
            logger.warning("Stabilisation failed, using unstabilised: %s", e)
            try:
                os.unlink(tmp_stab)
            except OSError:
                pass

    # Calculate total distance
    total_dist = 0.0
    for i in range(1, len(gps_points)):
        p1 = GpsPoint(latitude=gps_points[i-1].latitude,
                       longitude=gps_points[i-1].longitude)
        p2 = GpsPoint(latitude=gps_points[i].latitude,
                       longitude=gps_points[i].longitude)
        total_dist += _haversine(p1, p2)

    effective_speed = speed_factor if speed_factor > 0 else (
        total_frames / max(1, sampled_count)
    )

    if on_progress:
        on_progress(100, "Aerial hyperlapse complete")

    logger.info("Created aerial hyperlapse: %s (%d sampled frames, %.0fm)",
                out_path, sampled_count, total_dist)

    return HyperlapseResult(
        output_path=out_path,
        total_frames=total_frames,
        sampled_frames=sampled_count,
        total_distance_m=total_dist,
        speed_factor=effective_speed,
        stabilised=stabilised,
    )
