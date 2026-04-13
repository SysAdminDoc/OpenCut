"""
OpenCut Telemetry Data Overlay Module

Overlay telemetry data from DJI SRT files or CSV onto video:
- Parse DJI SRT format (GPS, altitude, speed, battery, gimbal)
- Parse generic CSV telemetry data
- Render telemetry as text overlay via FFmpeg drawtext filters

All via FFmpeg drawtext - no additional dependencies.
"""

import csv
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class TelemetryFrame:
    """A single telemetry data point."""
    timestamp: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    speed: float = 0.0
    distance: float = 0.0
    battery: float = 0.0
    gimbal_angle: float = 0.0


@dataclass
class TelemetryOverlayResult:
    """Result of telemetry overlay processing."""
    output_path: str = ""
    fields_shown: List[str] = field(default_factory=list)
    telemetry_points: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# DJI SRT Parsing
# ---------------------------------------------------------------------------

def parse_dji_srt(srt_path: str) -> List[TelemetryFrame]:
    """Parse a DJI drone SRT file into telemetry frames.

    DJI SRT files contain subtitle entries with embedded telemetry:
    - GPS coordinates: [latitude: X] [longitude: Y]
    - Altitude: [altitude: Z] or [rel_alt: Z]
    - Speed: various formats
    - Battery: [battery: N%]
    - Gimbal: [gimbal_angle: X]

    Args:
        srt_path: Path to DJI SRT file.

    Returns:
        List of TelemetryFrame with parsed data.
    """
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    frames = []

    # Split into SRT subtitle blocks
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Parse timestamp from SRT timecode line
        timestamp = 0.0
        for line in lines:
            tc_match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", line.strip())
            if tc_match:
                h, m, s, ms = tc_match.groups()
                timestamp = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                break

        # Parse telemetry fields from data lines
        text = " ".join(lines)
        tf = TelemetryFrame(timestamp=round(timestamp, 3))

        # Latitude
        lat_match = re.search(r"\[?latitude[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if lat_match:
            try:
                tf.latitude = float(lat_match.group(1))
            except ValueError:
                pass

        # Longitude
        lon_match = re.search(r"\[?longitude[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if lon_match:
            try:
                tf.longitude = float(lon_match.group(1))
            except ValueError:
                pass

        # Altitude (try multiple patterns)
        alt_match = re.search(r"\[?(?:rel_)?alt(?:itude)?[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if alt_match:
            try:
                tf.altitude = float(alt_match.group(1))
            except ValueError:
                pass

        # Speed
        speed_match = re.search(r"\[?speed[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if speed_match:
            try:
                tf.speed = float(speed_match.group(1))
            except ValueError:
                pass

        # Distance
        dist_match = re.search(r"\[?distance[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if dist_match:
            try:
                tf.distance = float(dist_match.group(1))
            except ValueError:
                pass

        # Battery
        batt_match = re.search(r"\[?battery[:\s]+(-?[\d.]+)%?\]?", text, re.IGNORECASE)
        if batt_match:
            try:
                tf.battery = float(batt_match.group(1))
            except ValueError:
                pass

        # Gimbal angle
        gimbal_match = re.search(r"\[?gimbal[_\s]?angle[:\s]+(-?[\d.]+)\]?", text, re.IGNORECASE)
        if gimbal_match:
            try:
                tf.gimbal_angle = float(gimbal_match.group(1))
            except ValueError:
                pass

        # Only add if we got meaningful data (not just a bare subtitle index)
        if tf.latitude != 0 or tf.longitude != 0 or tf.altitude != 0 or tf.speed != 0 or tf.battery != 0:
            frames.append(tf)

    return frames


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

def parse_telemetry_csv(csv_path: str) -> List[TelemetryFrame]:
    """Parse a generic CSV file into telemetry frames.

    Expected columns (case-insensitive):
    timestamp (or time), latitude (or lat), longitude (or lon/lng),
    altitude (or alt), speed, distance, battery, gimbal_angle (or gimbal).

    Args:
        csv_path: Path to CSV file.

    Returns:
        List of TelemetryFrame with parsed data.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frames = []
    # Column name aliases
    aliases = {
        "timestamp": ["timestamp", "time", "time_s", "time_sec"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng", "long"],
        "altitude": ["altitude", "alt", "height", "elevation"],
        "speed": ["speed", "velocity", "speed_mph", "speed_kmh", "speed_ms"],
        "distance": ["distance", "dist", "distance_m"],
        "battery": ["battery", "batt", "battery_pct", "battery_level"],
        "gimbal_angle": ["gimbal_angle", "gimbal", "pitch", "camera_angle"],
    }

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return frames

        # Map CSV headers to our field names
        header_map = {}
        for header in reader.fieldnames:
            h_lower = header.strip().lower()
            for field_name, alias_list in aliases.items():
                if h_lower in alias_list:
                    header_map[header] = field_name
                    break

        for row in reader:
            tf = TelemetryFrame()
            for csv_col, our_field in header_map.items():
                val_str = row.get(csv_col, "").strip()
                if val_str:
                    try:
                        val = float(val_str)
                        setattr(tf, our_field, val)
                    except ValueError:
                        pass

            # Only add if we got some data
            if any(getattr(tf, f) != 0 for f in ["latitude", "longitude", "altitude", "speed", "battery"]):
                frames.append(tf)

    return frames


# ---------------------------------------------------------------------------
# Overlay Rendering
# ---------------------------------------------------------------------------

# Field display formatters
FIELD_FORMATTERS = {
    "altitude": lambda tf: f"ALT: {tf.altitude:.1f}m",
    "speed": lambda tf: f"SPD: {tf.speed:.1f}m/s",
    "gps": lambda tf: f"GPS: {tf.latitude:.6f}, {tf.longitude:.6f}",
    "battery": lambda tf: f"BAT: {tf.battery:.0f}%",
    "gimbal": lambda tf: f"GIMBAL: {tf.gimbal_angle:.1f}",
    "distance": lambda tf: f"DIST: {tf.distance:.1f}m",
}

# Position presets (x, y offsets from edges)
POSITION_PRESETS = {
    "bottom-left": {"x": 20, "y_from_bottom": 20, "align": "left"},
    "bottom-right": {"x_from_right": 20, "y_from_bottom": 20, "align": "right"},
    "top-left": {"x": 20, "y": 20, "align": "left"},
    "top-right": {"x_from_right": 20, "y": 20, "align": "right"},
}


def _escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # Escape colons, backslashes, single quotes, and percent signs
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    return text


def _build_drawtext_filters(
    telemetry: List[TelemetryFrame],
    fields: List[str],
    position: str = "bottom-left",
    font_size: int = 18,
    font_color: str = "white",
    bg_color: str = "black@0.5",
    video_duration: float = 0.0,
) -> str:
    """Build FFmpeg drawtext filter chain for telemetry overlay.

    Generates per-frame text using drawtext with enable='between(t,start,end)'
    for each telemetry data window.
    """
    pos = POSITION_PRESETS.get(position, POSITION_PRESETS["bottom-left"])

    # Calculate position
    if "x_from_right" in pos:
        x_expr = f"w-tw-{pos['x_from_right']}"
    else:
        x_expr = str(pos.get("x", 20))

    if "y_from_bottom" in pos:
        base_y_expr = f"h-{pos['y_from_bottom']}"
        y_direction = -1  # stack upward
    else:
        base_y_expr = str(pos.get("y", 20))
        y_direction = 1  # stack downward

    filters = []
    line_height = font_size + 6

    # Sort telemetry by timestamp
    sorted_telem = sorted(telemetry, key=lambda t: t.timestamp)

    # Group telemetry into time windows
    # For each time window, create drawtext filters for each field
    for i in range(len(sorted_telem)):
        tf = sorted_telem[i]
        # Time window: from this frame to next frame (or end of video)
        t_start = tf.timestamp
        if i + 1 < len(sorted_telem):
            t_end = sorted_telem[i + 1].timestamp
        else:
            t_end = video_duration if video_duration > 0 else t_start + 10

        enable = f"between(t\\,{t_start:.3f}\\,{t_end:.3f})"

        for field_idx, field_name in enumerate(fields):
            formatter = FIELD_FORMATTERS.get(field_name)
            if formatter is None:
                continue

            text = _escape_drawtext(formatter(tf))

            # Calculate Y position for this line
            if y_direction < 0:
                y_offset = (len(fields) - field_idx) * line_height
                y_pos = f"{base_y_expr}-{y_offset}"
            else:
                y_offset = field_idx * line_height
                y_pos = f"{base_y_expr}+{y_offset}"

            filters.append(
                f"drawtext=text='{text}'"
                f":fontsize={font_size}"
                f":fontcolor={font_color}"
                f":box=1:boxcolor={bg_color}:boxborderw=5"
                f":x={x_expr}:y={y_pos}"
                f":enable='{enable}'"
            )

    return ",".join(filters) if filters else "null"


def overlay_telemetry(
    video_path: str,
    telemetry: List[dict],
    fields: Optional[List[str]] = None,
    position: str = "bottom-left",
    font_size: int = 18,
    font_color: str = "white",
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Overlay telemetry data onto video using drawtext filters.

    Args:
        video_path: Source video file.
        telemetry: List of telemetry dicts or TelemetryFrame objects.
        fields: Which data fields to show: ["altitude", "speed", "gps", "battery", "gimbal", "distance"].
                Defaults to all available fields.
        position: Overlay position: "bottom-left", "bottom-right", "top-left", "top-right".
        font_size: Font size for overlay text (8-72).
        font_color: Font color name or hex.
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, fields_shown, telemetry_points, duration.
    """
    if on_progress:
        on_progress(5, "Preparing telemetry overlay...")

    info = get_video_info(video_path)
    duration = info["duration"]
    font_size = max(8, min(72, font_size))

    if output_path_str is None:
        output_path_str = output_path(video_path, "telemetry")

    # Normalize telemetry data
    telem_frames = []
    for entry in telemetry:
        if isinstance(entry, TelemetryFrame):
            telem_frames.append(entry)
        elif isinstance(entry, dict):
            telem_frames.append(TelemetryFrame(
                timestamp=float(entry.get("timestamp", 0)),
                latitude=float(entry.get("latitude", 0)),
                longitude=float(entry.get("longitude", 0)),
                altitude=float(entry.get("altitude", 0)),
                speed=float(entry.get("speed", 0)),
                distance=float(entry.get("distance", 0)),
                battery=float(entry.get("battery", 0)),
                gimbal_angle=float(entry.get("gimbal_angle", 0)),
            ))

    if not telem_frames:
        raise ValueError("No telemetry data provided")

    # Determine which fields to show
    if fields is None:
        # Auto-detect: show fields that have non-zero data
        available = set()
        for tf in telem_frames:
            if tf.altitude != 0:
                available.add("altitude")
            if tf.speed != 0:
                available.add("speed")
            if tf.latitude != 0 or tf.longitude != 0:
                available.add("gps")
            if tf.battery != 0:
                available.add("battery")
            if tf.gimbal_angle != 0:
                available.add("gimbal")
            if tf.distance != 0:
                available.add("distance")
        fields = sorted(available) if available else ["altitude", "speed", "gps"]

    # Validate fields
    valid_fields = [f for f in fields if f in FIELD_FORMATTERS]
    if not valid_fields:
        valid_fields = ["altitude", "speed"]

    if on_progress:
        on_progress(15, f"Building drawtext filters for {len(valid_fields)} fields...")

    # For many telemetry points, limit drawtext filters to avoid command-line overflow
    # FFmpeg has practical limits on filter chain length
    max_points = 500
    if len(telem_frames) > max_points:
        # Downsample to max_points
        step = len(telem_frames) / max_points
        sampled = [telem_frames[int(i * step)] for i in range(max_points)]
        telem_frames = sampled

    position = position if position in POSITION_PRESETS else "bottom-left"

    vf = _build_drawtext_filters(
        telemetry=telem_frames,
        fields=valid_fields,
        position=position,
        font_size=font_size,
        font_color=font_color,
        video_duration=duration,
    )

    if on_progress:
        on_progress(30, "Rendering telemetry overlay...")

    # If filter is too long for command line, write to a filter script file
    if len(vf) > 8000:
        tmp_dir = tempfile.mkdtemp(prefix="opencut_telem_")
        filter_script = os.path.join(tmp_dir, "filters.txt")
        with open(filter_script, "w", encoding="utf-8") as f:
            f.write(vf)

        cmd = [
            get_ffmpeg_path(), "-i", video_path,
            "-filter_script:v", filter_script,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-y", output_path_str,
        ]
    else:
        cmd = [
            get_ffmpeg_path(), "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-y", output_path_str,
        ]

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Telemetry overlay complete")

    return {
        "output_path": output_path_str,
        "fields_shown": valid_fields,
        "telemetry_points": len(telem_frames),
        "duration": duration,
    }
