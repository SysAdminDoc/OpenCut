"""
OpenCut Dynamic Subtitle Positioning

Per-frame analysis to detect faces, text/graphics, and bright/busy regions
in the lower portion of the frame. Repositions subtitles to avoid covering
important visual content. Generates ASS format with per-line positioning
overrides.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Positioning zones
# ---------------------------------------------------------------------------
ZONES = {
    "bottom_center": {"x_pct": 0.5, "y_pct": 0.9, "label": "Bottom Center"},
    "top_center": {"x_pct": 0.5, "y_pct": 0.1, "label": "Top Center"},
    "bottom_left": {"x_pct": 0.2, "y_pct": 0.9, "label": "Bottom Left"},
    "bottom_right": {"x_pct": 0.8, "y_pct": 0.9, "label": "Bottom Right"},
    "top_left": {"x_pct": 0.2, "y_pct": 0.1, "label": "Top Left"},
    "top_right": {"x_pct": 0.8, "y_pct": 0.1, "label": "Top Right"},
}

ZONE_PRIORITY = [
    "bottom_center",
    "top_center",
    "bottom_left",
    "bottom_right",
    "top_left",
    "top_right",
]


@dataclass
class ObstructionInfo:
    """Information about a detected obstruction in a frame region."""

    zone: str = ""
    obstruction_type: str = ""  # face, text, graphic, bright, busy
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""

    timestamp: float = 0.0
    obstructions: List[ObstructionInfo] = field(default_factory=list)
    obstructed_zones: List[str] = field(default_factory=list)
    best_zone: str = "bottom_center"


@dataclass
class PositionedSubtitle:
    """A subtitle with determined position."""

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    zone: str = "bottom_center"
    x: int = 0
    y: int = 0
    repositioned: bool = False
    obstruction_reason: str = ""


@dataclass
class PositionResult:
    """Result of dynamic subtitle positioning."""

    positioned_subtitles: List[PositionedSubtitle] = field(default_factory=list)
    repositioned_count: int = 0
    obstruction_types: Dict[str, int] = field(default_factory=dict)
    total_segments: int = 0
    frames_analyzed: int = 0


@dataclass
class Obstruction:
    """Backward-compatible obstruction record used by batch-data routes."""

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    label: str = ""
    confidence: float = 0.0


@dataclass
class SubtitlePosition:
    """Single subtitle placement decision for compatibility endpoints."""

    x: int = 0
    y: int = 0
    alignment: int = 2
    margin_bottom: int = 40
    safe: bool = True
    reason: str = ""


@dataclass
class PositioningResult:
    """Minimal route-friendly result container."""

    output_path: str = ""
    status: str = "pending"
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "output_path": self.output_path,
            "status": self.status,
        }
        payload.update(self.details)
        return payload


_ALIGNMENT_TO_ZONE = {
    1: "bottom_left",
    2: "bottom_center",
    3: "bottom_right",
    7: "top_left",
    8: "top_center",
    9: "top_right",
}
_ZONE_TO_ALIGNMENT = {
    zone: alignment for alignment, zone in _ALIGNMENT_TO_ZONE.items()
}


# ---------------------------------------------------------------------------
# Zone coordinate calculation
# ---------------------------------------------------------------------------
def zone_to_pixels(
    zone: str,
    video_width: int = 1920,
    video_height: int = 1080,
    margin_x: int = 40,
    margin_y: int = 40,
) -> Tuple[int, int]:
    """Convert a zone name to pixel coordinates for subtitle placement.

    Args:
        zone: Zone name (bottom_center, top_center, etc.).
        video_width: Frame width in pixels.
        video_height: Frame height in pixels.
        margin_x: Horizontal margin from edges.
        margin_y: Vertical margin from edges.

    Returns:
        (x, y) pixel coordinates for subtitle anchor point.
    """
    zone_info = ZONES.get(zone, ZONES["bottom_center"])
    x = int(zone_info["x_pct"] * video_width)
    y = int(zone_info["y_pct"] * video_height)
    # Clamp to margins
    x = max(margin_x, min(x, video_width - margin_x))
    y = max(margin_y, min(y, video_height - margin_y))
    return x, y


# ---------------------------------------------------------------------------
# Frame analysis (simplified without heavy ML deps)
# ---------------------------------------------------------------------------
def _check_bottom_third(
    frame_data: Optional[Dict],
    video_width: int,
    video_height: int,
) -> List[ObstructionInfo]:
    """Check the bottom third of the frame for obstructions.

    Uses pre-computed analysis data (face locations, text regions, etc.)
    rather than performing ML inference directly. When frame_data is None
    or empty, assumes no obstructions.

    Args:
        frame_data: Dict with optional keys:
            'faces': list of {x, y, w, h} face bounding boxes
            'text_regions': list of {x, y, w, h} text/OCR regions
            'bright_regions': list of {x, y, w, h, intensity} bright spots
            'edge_density': float (0-1) edge density in bottom third
        video_width: Frame width.
        video_height: Frame height.

    Returns:
        List of ObstructionInfo for obstructed zones.
    """
    if not frame_data:
        return []

    obstructions: List[ObstructionInfo] = []
    bottom_y = video_height * 2 // 3

    # Check faces
    for face in frame_data.get("faces", []):
        fx = int(face.get("x", 0))
        fy = int(face.get("y", 0))
        fw = int(face.get("w", 0))
        fh = int(face.get("h", 0))
        # Face in bottom third?
        if fy + fh > bottom_y:
            zone = _bbox_to_zone(fx, fy, fw, fh, video_width, video_height)
            obstructions.append(ObstructionInfo(
                zone=zone,
                obstruction_type="face",
                confidence=float(face.get("confidence", 0.9)),
                bbox=(fx, fy, fw, fh),
            ))

    # Check text/graphic regions
    for region in frame_data.get("text_regions", []):
        rx = int(region.get("x", 0))
        ry = int(region.get("y", 0))
        rw = int(region.get("w", 0))
        rh = int(region.get("h", 0))
        if ry + rh > bottom_y:
            zone = _bbox_to_zone(rx, ry, rw, rh, video_width, video_height)
            obstructions.append(ObstructionInfo(
                zone=zone,
                obstruction_type="text",
                confidence=float(region.get("confidence", 0.8)),
                bbox=(rx, ry, rw, rh),
            ))

    # Check bright regions
    for bright in frame_data.get("bright_regions", []):
        bx = int(bright.get("x", 0))
        by = int(bright.get("y", 0))
        bw = int(bright.get("w", 0))
        bh = int(bright.get("h", 0))
        intensity = float(bright.get("intensity", 0))
        if by + bh > bottom_y and intensity > 0.8:
            zone = _bbox_to_zone(bx, by, bw, bh, video_width, video_height)
            obstructions.append(ObstructionInfo(
                zone=zone,
                obstruction_type="bright",
                confidence=intensity,
                bbox=(bx, by, bw, bh),
            ))

    # Edge density (general busyness)
    edge_density = float(frame_data.get("edge_density", 0))
    if edge_density > 0.6:
        obstructions.append(ObstructionInfo(
            zone="bottom_center",
            obstruction_type="busy",
            confidence=edge_density,
            bbox=(0, bottom_y, video_width, video_height - bottom_y),
        ))

    return obstructions


def _bbox_to_zone(
    x: int, y: int, w: int, h: int,
    video_width: int, video_height: int,
) -> str:
    """Map a bounding box center to the nearest positioning zone."""
    cx = x + w // 2
    cy = y + h // 2
    # Determine which zone this overlaps
    if cy > video_height * 0.66:
        # Bottom region
        if cx < video_width * 0.33:
            return "bottom_left"
        if cx > video_width * 0.66:
            return "bottom_right"
        return "bottom_center"
    # Top region
    if cx < video_width * 0.33:
        return "top_left"
    if cx > video_width * 0.66:
        return "top_right"
    return "top_center"


def _find_best_zone(
    obstructed_zones: List[str],
    priority: Optional[List[str]] = None,
) -> str:
    """Find the best unobstructed zone using priority order.

    Args:
        obstructed_zones: List of zone names that are obstructed.
        priority: Zone priority order (default: ZONE_PRIORITY).

    Returns:
        Best available zone name.
    """
    order = priority or ZONE_PRIORITY
    obstructed_set = set(obstructed_zones)
    for zone in order:
        if zone not in obstructed_set:
            return zone
    # All obstructed, fall back to top_center (least likely to overlap
    # with most content types)
    return "top_center"


def analyze_frame(
    frame_data: Optional[Dict],
    video_width: int = 1920,
    video_height: int = 1080,
) -> FrameAnalysis:
    """Analyze a single frame for subtitle positioning obstructions.

    Args:
        frame_data: Pre-computed analysis data for the frame.
        video_width: Frame width.
        video_height: Frame height.

    Returns:
        FrameAnalysis with detected obstructions and best zone.
    """
    obstructions = _check_bottom_third(frame_data, video_width, video_height)
    obstructed_zones = list({o.zone for o in obstructions})
    best_zone = _find_best_zone(obstructed_zones)

    return FrameAnalysis(
        obstructions=obstructions,
        obstructed_zones=obstructed_zones,
        best_zone=best_zone,
    )


# ---------------------------------------------------------------------------
# Batch subtitle positioning
# ---------------------------------------------------------------------------
def position_subtitles(
    subtitles: List[Dict],
    frame_analyses: Optional[Dict[float, Dict]] = None,
    video_width: int = 1920,
    video_height: int = 1080,
    analyze_keyframes_only: bool = True,
    on_progress: Optional[Callable] = None,
) -> PositionResult:
    """Apply dynamic positioning to subtitle segments.

    In batch mode with analyze_keyframes_only=True, only analyzes frames at
    subtitle in-points rather than every frame.

    Args:
        subtitles: List of subtitle dicts with 'start', 'end', 'text'.
        frame_analyses: Pre-computed per-frame analysis data. Dict mapping
            timestamps to frame data dicts with 'faces', 'text_regions', etc.
        video_width: Frame width for coordinate calculation.
        video_height: Frame height for coordinate calculation.
        analyze_keyframes_only: Only analyze at subtitle start times.
        on_progress: Progress callback.

    Returns:
        PositionResult with positioned subtitles.
    """
    if not subtitles:
        return PositionResult()

    analyses = frame_analyses or {}
    result_subs: List[PositionedSubtitle] = []
    obstruction_counts: Dict[str, int] = {}
    repositioned = 0
    frames_analyzed = 0

    total = len(subtitles)
    for i, sub in enumerate(subtitles):
        start = float(sub.get("start", 0))
        end = float(sub.get("end", 0))
        text = str(sub.get("text", ""))

        # Find closest frame analysis
        frame_data = None
        if analyses:
            # Find closest timestamp
            closest_ts = None
            min_dist = float("inf")
            for ts in analyses:
                dist = abs(float(ts) - start)
                if dist < min_dist:
                    min_dist = dist
                    closest_ts = ts
            if closest_ts is not None and min_dist < 1.0:
                frame_data = analyses[closest_ts]

        # Analyze frame
        analysis = analyze_frame(frame_data, video_width, video_height)
        analysis.timestamp = start
        frames_analyzed += 1

        # Determine position
        zone = analysis.best_zone
        x, y = zone_to_pixels(zone, video_width, video_height)
        is_repositioned = zone != "bottom_center"

        reason = ""
        if is_repositioned and analysis.obstructions:
            types = [o.obstruction_type for o in analysis.obstructions]
            reason = ", ".join(set(types))
            for t in types:
                obstruction_counts[t] = obstruction_counts.get(t, 0) + 1
            repositioned += 1

        result_subs.append(PositionedSubtitle(
            index=i + 1,
            start=start,
            end=end,
            text=text,
            zone=zone,
            x=x,
            y=y,
            repositioned=is_repositioned,
            obstruction_reason=reason,
        ))

        if on_progress:
            on_progress(int(((i + 1) / total) * 95))

    if on_progress:
        on_progress(100)

    return PositionResult(
        positioned_subtitles=result_subs,
        repositioned_count=repositioned,
        obstruction_types=obstruction_counts,
        total_segments=len(result_subs),
        frames_analyzed=frames_analyzed,
    )


# ---------------------------------------------------------------------------
# Single frame preview
# ---------------------------------------------------------------------------
def preview_position(
    text: str,
    frame_data: Optional[Dict],
    video_width: int = 1920,
    video_height: int = 1080,
) -> Dict:
    """Preview subtitle position for a single frame.

    Args:
        text: Subtitle text to position.
        frame_data: Pre-computed frame analysis data.
        video_width: Frame width.
        video_height: Frame height.

    Returns:
        Dict with zone, x, y, obstructions, ass_override.
    """
    analysis = analyze_frame(frame_data, video_width, video_height)
    x, y = zone_to_pixels(analysis.best_zone, video_width, video_height)

    ass_override = f"{{\\pos({x},{y})}}"

    return {
        "zone": analysis.best_zone,
        "zone_label": ZONES.get(analysis.best_zone, {}).get("label", ""),
        "x": x,
        "y": y,
        "obstructions": [
            {
                "type": o.obstruction_type,
                "zone": o.zone,
                "confidence": o.confidence,
            }
            for o in analysis.obstructions
        ],
        "obstructed_zones": analysis.obstructed_zones,
        "ass_override": ass_override,
    }


# ---------------------------------------------------------------------------
# ASS export with per-line positioning
# ---------------------------------------------------------------------------
def export_positioned_ass(
    result: PositionResult,
    title: str = "OpenCut Positioned Subtitles",
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """Export positioned subtitles as ASS with per-line positioning overrides.

    Each subtitle line gets a {\\pos(x,y)} override based on the computed
    best position for that segment.
    """
    header = (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,40,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    events: List[str] = []
    for sub in result.positioned_subtitles:
        start = _seconds_to_ass(sub.start)
        end = _seconds_to_ass(sub.end)
        text = sub.text.replace("\n", "\\N")
        # Add position override
        pos_tag = f"{{\\pos({sub.x},{sub.y})}}"
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{pos_tag}{text}"
        )

    return header + "\n".join(events) + "\n"


def _seconds_to_ass(s: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def export_to_file(
    result: PositionResult,
    output_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """Export positioned subtitles to ASS file. Returns output path."""
    content = export_positioned_ass(result, video_width=video_width,
                                    video_height=video_height)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(
        "Exported %d positioned subtitles to %s",
        len(result.positioned_subtitles), output_path,
    )
    return output_path


def analyze_frame_obstructions(
    frame_path: str,
    detect_faces: bool = True,
    detect_text: bool = True,
    detect_logos: bool = True,
) -> List[Obstruction]:
    """Lightweight obstruction analysis for a rendered frame image.

    The compatibility routes only need a safe, dependency-light heuristic.
    We currently detect bright/busy lower-third regions and return an empty
    list when optional imaging dependencies are unavailable.
    """
    if not os.path.exists(frame_path):
        raise FileNotFoundError(frame_path)

    try:
        from PIL import Image, ImageStat
    except ImportError:
        logger.debug("Pillow unavailable; skipping frame obstruction analysis")
        return []

    try:
        with Image.open(frame_path) as img:
            gray = img.convert("L")
            width, height = gray.size
            lower_third_top = height * 2 // 3
            lower_third = gray.crop((0, lower_third_top, width, height))
            stat = ImageStat.Stat(lower_third)
            mean_brightness = float(stat.mean[0]) if stat.mean else 0.0
            extrema = stat.extrema[0] if stat.extrema else (0, 0)
    except OSError as exc:
        raise ValueError(f"Could not read frame image: {exc}") from exc

    obstructions: List[Obstruction] = []
    if detect_text or detect_logos or detect_faces:
        if mean_brightness >= 180.0 or extrema[1] >= 240:
            confidence = max(mean_brightness / 255.0, 0.7)
            obstructions.append(Obstruction(
                x=0,
                y=lower_third_top,
                width=width,
                height=height - lower_third_top,
                label="bright",
                confidence=min(confidence, 1.0),
            ))
    return obstructions


def compute_subtitle_position(
    obstructions: List[Obstruction],
    frame_size: Tuple[int, int],
    preferred_alignment: int = 2,
    margin: int = 50,
) -> SubtitlePosition:
    """Choose a safe subtitle placement from simple obstruction boxes."""
    frame_width, frame_height = frame_size
    preferred_zone = _ALIGNMENT_TO_ZONE.get(preferred_alignment, "bottom_center")

    obstructed_zones = []
    obstruction_labels: Dict[str, set] = {}
    for obstruction in obstructions:
        zone = _bbox_to_zone(
            obstruction.x,
            obstruction.y,
            obstruction.width,
            obstruction.height,
            frame_width,
            frame_height,
        )
        obstructed_zones.append(zone)
        obstruction_labels.setdefault(zone, set()).add(obstruction.label or "content")

    priority = [preferred_zone]
    priority.extend(zone for zone in ZONE_PRIORITY if zone != preferred_zone)
    best_zone = _find_best_zone(obstructed_zones, priority)
    x, y = zone_to_pixels(
        best_zone,
        frame_width,
        frame_height,
        margin_x=margin,
        margin_y=margin,
    )

    reason = ""
    if preferred_zone in obstructed_zones:
        labels = sorted(label for label in obstruction_labels.get(preferred_zone, set()) if label)
        reason = f"Avoided {preferred_zone.replace('_', ' ')} obstruction"
        if labels:
            reason += f": {', '.join(labels)}"
    elif best_zone != preferred_zone:
        reason = f"Moved subtitles to {best_zone.replace('_', ' ')}"

    return SubtitlePosition(
        x=x,
        y=y,
        alignment=_ZONE_TO_ALIGNMENT.get(best_zone, 2),
        margin_bottom=margin,
        safe=best_zone not in set(obstructed_zones),
        reason=reason,
    )
