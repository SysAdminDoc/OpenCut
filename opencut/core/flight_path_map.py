"""
OpenCut Flight Path Map Module

Extract GPS coordinates from DJI SRT or GPX files and render an
animated map overlay showing the flight path.  Uses FFmpeg drawtext
and drawing filters to produce a self-contained overlay clip (no
external map tile service needed for basic mode).

GPS parsing supports:
  - DJI SRT (subtitle format with embedded telemetry)
  - GPX (XML-based GPS exchange format)
"""

import logging
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class GpsPoint:
    """A single GPS coordinate with optional metadata."""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    timestamp: float = 0.0
    speed: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FlightMapResult:
    """Result of flight path map rendering."""
    output_path: str = ""
    total_points: int = 0
    total_distance_m: float = 0.0
    duration: float = 0.0
    bounds: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GPS Parsing
# ---------------------------------------------------------------------------

def parse_gps_track(
    srt_path: str,
    on_progress: Optional[Callable] = None,
) -> List[GpsPoint]:
    """Parse GPS coordinates from a DJI SRT or GPX file.

    Auto-detects format by file extension and content.

    Args:
        srt_path: Path to SRT or GPX file.
        on_progress: Optional ``(pct, msg)`` callback.

    Returns:
        List of GpsPoint objects in chronological order.
    """
    if on_progress:
        on_progress(10, "Reading GPS track file...")

    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"GPS file not found: {srt_path}")

    ext = os.path.splitext(srt_path)[1].lower()

    if ext == ".gpx":
        points = _parse_gpx(srt_path)
    else:
        points = _parse_dji_srt_gps(srt_path)

    if on_progress:
        on_progress(100, f"Parsed {len(points)} GPS points")

    logger.info("Parsed %d GPS points from %s", len(points), srt_path)
    return points


def _parse_dji_srt_gps(srt_path: str) -> List[GpsPoint]:
    """Parse GPS data from DJI SRT subtitle file."""
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    points = []
    blocks = re.split(r"\n\s*\n", content.strip())

    lat_re = re.compile(r"\[?latitude[:\s]+([+-]?\d+\.?\d*)\]?", re.IGNORECASE)
    lon_re = re.compile(r"\[?longitude[:\s]+([+-]?\d+\.?\d*)\]?", re.IGNORECASE)
    alt_re = re.compile(r"\[?(?:altitude|rel_alt)[:\s]+([+-]?\d+\.?\d*)\]?", re.IGNORECASE)
    ts_re = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")

    for block in blocks:
        lat_m = lat_re.search(block)
        lon_m = lon_re.search(block)
        if not lat_m or not lon_m:
            continue

        lat = float(lat_m.group(1))
        lon = float(lon_m.group(1))

        alt = 0.0
        alt_m = alt_re.search(block)
        if alt_m:
            alt = float(alt_m.group(1))

        ts = 0.0
        ts_m = ts_re.search(block)
        if ts_m:
            ts = (int(ts_m.group(1)) * 3600 + int(ts_m.group(2)) * 60 +
                  int(ts_m.group(3)) + int(ts_m.group(4)) / 1000.0)

        points.append(GpsPoint(
            latitude=lat, longitude=lon, altitude=alt, timestamp=ts,
        ))

    return points


def _parse_gpx(gpx_path: str) -> List[GpsPoint]:
    """Parse GPS data from GPX XML file."""
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    points = []
    for trkpt in root.iter(f"{ns}trkpt"):
        lat = float(trkpt.get("lat", 0))
        lon = float(trkpt.get("lon", 0))

        alt = 0.0
        ele = trkpt.find(f"{ns}ele")
        if ele is not None and ele.text:
            alt = float(ele.text)

        points.append(GpsPoint(latitude=lat, longitude=lon, altitude=alt))

    return points


# ---------------------------------------------------------------------------
# Distance / projection helpers
# ---------------------------------------------------------------------------

def _haversine(p1: GpsPoint, p2: GpsPoint) -> float:
    """Haversine distance in meters between two GPS points."""
    R = 6371000.0
    lat1 = math.radians(p1.latitude)
    lat2 = math.radians(p2.latitude)
    dlat = math.radians(p2.latitude - p1.latitude)
    dlon = math.radians(p2.longitude - p1.longitude)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _total_distance(points: List[GpsPoint]) -> float:
    """Total distance in meters for a track."""
    if len(points) < 2:
        return 0.0
    return sum(_haversine(points[i], points[i + 1]) for i in range(len(points) - 1))


def _get_bounds(points: List[GpsPoint]) -> Dict[str, float]:
    """Get geographic bounding box."""
    if not points:
        return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    return {
        "min_lat": min(lats), "max_lat": max(lats),
        "min_lon": min(lons), "max_lon": max(lons),
    }


def _project_point(
    point: GpsPoint,
    bounds: Dict[str, float],
    width: int,
    height: int,
    margin: int = 40,
) -> Tuple[int, int]:
    """Project a GPS point to pixel coordinates within bounds."""
    lat_range = bounds["max_lat"] - bounds["min_lat"]
    lon_range = bounds["max_lon"] - bounds["min_lon"]

    if lat_range == 0:
        lat_range = 0.001
    if lon_range == 0:
        lon_range = 0.001

    x_frac = (point.longitude - bounds["min_lon"]) / lon_range
    y_frac = 1.0 - (point.latitude - bounds["min_lat"]) / lat_range

    x = margin + int(x_frac * (width - 2 * margin))
    y = margin + int(y_frac * (height - 2 * margin))
    return x, y


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_flight_map(
    gps_points: List[GpsPoint],
    out_path: str,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
    duration: float = 0.0,
    line_color: str = "yellow",
    bg_color: str = "0x1a1a2e",
    on_progress: Optional[Callable] = None,
) -> FlightMapResult:
    """Render an animated flight path map video from GPS points.

    Creates a video showing the flight path being drawn progressively.
    Uses FFmpeg drawing filters for rendering.

    Args:
        gps_points: List of GpsPoint objects.
        out_path: Output video file path.
        resolution: (width, height) tuple.
        fps: Frame rate.
        duration: Total video duration (0 = auto from points).
        line_color: FFmpeg color for the path line.
        bg_color: Background color hex.
        on_progress: Optional progress callback.

    Returns:
        FlightMapResult with output path and metadata.
    """
    if on_progress:
        on_progress(10, "Preparing flight map...")

    if len(gps_points) < 2:
        raise ValueError("Need at least 2 GPS points to render a flight map")

    w, h = resolution
    bounds = _get_bounds(gps_points)
    total_dist = _total_distance(gps_points)

    if duration <= 0:
        duration = max(5.0, len(gps_points) / fps)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Build drawtext overlay showing coordinates, alt, distance at each frame
    # We create a simple animated line rendering via sequential drawbox filters
    # For a production version we'd use Python rendering, but this uses pure FFmpeg
    projected = [_project_point(p, bounds, w, h) for p in gps_points]

    # Build a filter with drawbox for each segment that appears over time
    draw_filters = []
    n = len(projected)
    for i in range(1, n):
        t_start = (i - 1) / (n - 1) * duration
        # Draw a small box at each point (simulating line segments)
        x, y = projected[i]
        draw_filters.append(
            f"drawbox=x={x-1}:y={y-1}:w=3:h=3:color={line_color}:t=fill"
            f":enable='gte(t,{t_start:.3f})'"
        )

    # Add start/end markers
    sx, sy = projected[0]
    ex, ey = projected[-1]
    draw_filters.append(f"drawbox=x={sx-3}:y={sy-3}:w=7:h=7:color=green:t=fill")
    draw_filters.append(
        f"drawbox=x={ex-3}:y={ey-3}:w=7:h=7:color=red:t=fill"
        f":enable='gte(t,{duration * 0.9:.3f})'"
    )

    vf = ",".join(draw_filters) if draw_filters else "null"

    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi",
        "-i", f"color=c={bg_color}:s={w}x{h}:r={fps}:d={duration}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        out_path,
    ]

    if on_progress:
        on_progress(40, "Rendering flight map animation...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Flight map rendered")

    logger.info("Rendered flight map: %s (%d points, %.1fm)", out_path, n, total_dist)

    return FlightMapResult(
        output_path=out_path,
        total_points=n,
        total_distance_m=total_dist,
        duration=duration,
        bounds=bounds,
    )


def create_map_overlay(
    video_path: str,
    gps_track: List[GpsPoint],
    out_path: str = "",
    position: str = "bottom_right",
    size_fraction: float = 0.25,
    opacity: float = 0.8,
    on_progress: Optional[Callable] = None,
) -> FlightMapResult:
    """Overlay an animated flight map onto a video.

    Renders a small map overlay in the corner of the video showing
    the flight path.

    Args:
        video_path: Source video path.
        gps_track: GPS points for the flight path.
        out_path: Output path (auto-generated if empty).
        position: Overlay position (top_left, top_right, bottom_left, bottom_right).
        size_fraction: Size of map overlay relative to video (0.1 - 0.5).
        opacity: Overlay opacity (0.0 - 1.0).
        on_progress: Optional progress callback.

    Returns:
        FlightMapResult with composited output.
    """
    if on_progress:
        on_progress(5, "Preparing map overlay...")

    if not out_path:
        out_path = output_path(video_path, "flightmap")

    info = get_video_info(video_path)
    v_width = info.get("width", 1920)
    v_height = info.get("height", 1080)
    v_duration = info.get("duration", 30.0)

    size_fraction = max(0.1, min(0.5, size_fraction))
    map_w = int(v_width * size_fraction)
    map_h = int(v_height * size_fraction)

    # Render map to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        if on_progress:
            on_progress(20, "Rendering map clip...")

        render_flight_map(
            gps_points=gps_track,
            out_path=tmp_path,
            resolution=(map_w, map_h),
            duration=v_duration,
        )

        if on_progress:
            on_progress(60, "Compositing map overlay...")

        positions = {
            "top_left": "10:10",
            "top_right": f"{v_width - map_w - 10}:10",
            "bottom_left": f"10:{v_height - map_h - 10}",
            "bottom_right": f"{v_width - map_w - 10}:{v_height - map_h - 10}",
        }
        pos_str = positions.get(position, positions["bottom_right"])

        opacity = max(0.0, min(1.0, opacity))

        filter_complex = (
            f"[1:v]format=yuva420p,colorchannelmixer=aa={opacity:.2f}[map];"
            f"[0:v][map]overlay={pos_str}[out]"
        )

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-i", tmp_path,
            "-filter_complex", filter_complex,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            "-shortest",
            out_path,
        ]

        run_ffmpeg(cmd)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Map overlay complete")

    total_dist = _total_distance(gps_track)
    bounds = _get_bounds(gps_track)

    logger.info("Created flight map overlay: %s", out_path)

    return FlightMapResult(
        output_path=out_path,
        total_points=len(gps_track),
        total_distance_m=total_dist,
        duration=v_duration,
        bounds=bounds,
    )
