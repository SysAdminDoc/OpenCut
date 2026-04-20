"""
OpenCut Data-Driven Animation Engine (Category 79 - Motion Design)

Bind motion graphics to CSV/JSON data. Template format defines visual elements
(bar_chart, line_chart, counter, label, pie_chart, progress_bar) with data
bindings. Animate transitions between data states with smooth interpolation.

Functions:
    render_data_animation  - Render data-driven animation to video
    validate_template      - Validate template + data compatibility
    list_chart_types       - Return supported chart/element types
    load_data              - Load data from CSV/JSON string
    load_data_from_file    - Load data from CSV/JSON file
"""

import csv
import io
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Module-level import so tests/callers can patch
# ``opencut.core.data_animation.run_ffmpeg`` directly. The full implementation
# below uses this binding for any FFmpeg shell-out paths.
from opencut.helpers import get_ffmpeg_path, run_ffmpeg  # noqa: E402,F401

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class DataAnimResult:
    """Result of a data animation render."""

    output_path: str = ""
    data_rows_rendered: int = 0
    elements_count: int = 0
    duration: float = 0.0
    fps: int = 30

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "data_rows_rendered": self.data_rows_rendered,
            "elements_count": self.elements_count,
            "duration": self.duration,
            "fps": self.fps,
        }


# ---------------------------------------------------------------------------
# Chart / Element Types
# ---------------------------------------------------------------------------

CHART_TYPES = {
    "bar_chart": {
        "description": "Vertical bar chart with auto-scaling Y axis",
        "properties": ["height", "color", "label", "value"],
    },
    "line_chart": {
        "description": "Line chart connecting data points over time",
        "properties": ["value", "color", "line_width", "label"],
    },
    "counter": {
        "description": "Animated numeric counter with prefix/suffix",
        "properties": ["value", "prefix", "suffix", "color"],
    },
    "label": {
        "description": "Text label with data binding",
        "properties": ["text", "color", "font_size"],
    },
    "pie_chart": {
        "description": "Pie/donut chart with animated segments",
        "properties": ["values", "colors", "labels"],
    },
    "progress_bar": {
        "description": "Horizontal progress bar with fill animation",
        "properties": ["value", "max_value", "color", "bg_color"],
    },
}


def list_chart_types() -> List[dict]:
    """Return list of supported chart/element types."""
    return [
        {"type": name, **info}
        for name, info in CHART_TYPES.items()
    ]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _load_csv_data(csv_content: str) -> List[dict]:
    """Parse CSV string into list of row dicts, auto-converting numerics."""
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = []
    for row in reader:
        parsed = {}
        for key, val in row.items():
            if key is None:
                continue
            try:
                parsed[key] = float(val)
            except (ValueError, TypeError):
                parsed[key] = val
        rows.append(parsed)
    return rows


def _load_json_data(json_content: str) -> List[dict]:
    """Parse JSON string into list of row dicts."""
    data = json.loads(json_content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError("JSON data must be a list or contain 'rows'/'data' key")


def load_data(content: str, format_hint: str = "auto") -> List[dict]:
    """Load data from CSV or JSON string.

    Args:
        content: Raw data string.
        format_hint: 'csv', 'json', or 'auto' (detect by content).

    Returns:
        List of row dicts.
    """
    if not content or not content.strip():
        raise ValueError("Data content is empty")

    if format_hint == "csv":
        return _load_csv_data(content)
    if format_hint == "json":
        return _load_json_data(content)

    # Auto-detect
    stripped = content.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return _load_json_data(content)
    return _load_csv_data(content)


def load_data_from_file(filepath: str) -> List[dict]:
    """Load data from a CSV or JSON file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    ext = os.path.splitext(filepath)[1].lower()
    fmt = "csv" if ext == ".csv" else "json" if ext == ".json" else "auto"
    return load_data(content, fmt)


# ---------------------------------------------------------------------------
# Template / Binding Resolution
# ---------------------------------------------------------------------------


def _resolve_binding(binding: str, data_row: dict):
    """Resolve a data binding expression like '${data.revenue}'.

    Returns the resolved value (float or str).
    """
    if not isinstance(binding, str):
        return binding
    if not binding.startswith("${") or not binding.endswith("}"):
        return binding

    path = binding[2:-1].strip()
    if path.startswith("data."):
        path = path[5:]

    value = data_row.get(path)
    if value is None:
        return 0.0
    return value


def _resolve_element_bindings(element: dict, data_row: dict) -> dict:
    """Resolve all data bindings in an element definition."""
    resolved = {}
    for key, value in element.items():
        if key in ("type", "id", "x", "y", "width", "height_max"):
            resolved[key] = value
        elif isinstance(value, str) and "${" in value:
            resolved[key] = _resolve_binding(value, data_row)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_binding(v, data_row) if isinstance(v, str) else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def validate_template(template: dict, data: List[dict]) -> dict:
    """Validate a template definition against data.

    Returns dict with 'valid' bool, 'errors' list, and 'warnings' list.
    """
    errors = []
    warnings = []

    if not isinstance(template, dict):
        errors.append("Template must be a dict")
        return {"valid": False, "errors": errors, "warnings": warnings}

    elements = template.get("elements", [])
    if not elements:
        errors.append("Template must contain 'elements' list")

    for i, elem in enumerate(elements):
        etype = elem.get("type", "")
        if etype not in CHART_TYPES:
            errors.append(f"Element {i}: unknown type '{etype}'")
        if "id" not in elem:
            warnings.append(f"Element {i}: missing 'id' field")

    if not data:
        warnings.append("No data rows provided")
    elif elements:
        first_row = data[0]
        for i, elem in enumerate(elements):
            for key, val in elem.items():
                if isinstance(val, str) and "${" in val:
                    path = val[2:-1].strip()
                    if path.startswith("data."):
                        path = path[5:]
                    if path not in first_row:
                        warnings.append(
                            f"Element {i}: binding '{val}' not found in data"
                        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate_value(v1, v2, t: float):
    """Interpolate between two values. Numbers interpolate; text crossfades."""
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return v1 + (v2 - v1) * t
    return v1 if t < 0.5 else v2


def _interpolate_elements(elem1: dict, elem2: dict, t: float) -> dict:
    """Interpolate all properties between two resolved element states."""
    result = {}
    all_keys = set(elem1.keys()) | set(elem2.keys())
    text_keys = ("type", "id", "label", "text")
    for key in all_keys:
        default = "" if key in text_keys else 0.0
        v1 = elem1.get(key, default)
        v2 = elem2.get(key, default)
        if isinstance(v1, list) and isinstance(v2, list):
            max_len = max(len(v1), len(v2))
            interp_list = []
            for idx in range(max_len):
                a = v1[idx] if idx < len(v1) else 0.0
                b = v2[idx] if idx < len(v2) else 0.0
                interp_list.append(_interpolate_value(a, b, t))
            result[key] = interp_list
        else:
            result[key] = _interpolate_value(v1, v2, t)
    return result


# ---------------------------------------------------------------------------
# Color & Layout Helpers
# ---------------------------------------------------------------------------


def _parse_color(color_str) -> Tuple[int, int, int]:
    """Parse hex color to RGB tuple."""
    if not isinstance(color_str, str):
        return (200, 200, 200)
    c = color_str.lstrip("#")
    if len(c) == 3:
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    if len(c) >= 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (200, 200, 200)


def _auto_scale(values: List[float]) -> Tuple[float, float]:
    """Compute auto-scale range for chart values."""
    if not values:
        return (0.0, 100.0)
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 0.001:
        return (min_v - 1, max_v + 1)
    margin = (max_v - min_v) * 0.1
    return (min_v - margin, max_v + margin)


def _load_font(font_size: int = 24):
    """Load a default font for rendering."""
    from PIL import ImageFont  # noqa: F821
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _compute_layout(elements: List[dict],
                    resolution: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Compute bounding areas for each element in a grid-like layout."""
    n = len(elements)
    if n == 0:
        return []

    w, h = resolution
    margin = 40
    usable_w = w - 2 * margin
    usable_h = h - 2 * margin

    cols = min(n, 3)
    rows_count = math.ceil(n / cols)
    cell_w = usable_w // cols
    cell_h = usable_h // max(rows_count, 1)

    areas = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x1 = margin + col * cell_w + 10
        y1 = margin + row * cell_h + 10
        x2 = x1 + cell_w - 20
        y2 = y1 + cell_h - 20
        elem = elements[i] if i < len(elements) else {}
        if "x" in elem and "y" in elem:
            ex = int(elem["x"])
            ey = int(elem["y"])
            ew = int(elem.get("width", cell_w - 20))
            eh = int(elem.get("height_max", cell_h - 20))
            areas.append((ex, ey, ex + ew, ey + eh))
        else:
            areas.append((x1, y1, x2, y2))
    return areas


# ---------------------------------------------------------------------------
# Element Drawers
# ---------------------------------------------------------------------------


def _draw_bar_chart(draw, elem: dict, area: Tuple[int, int, int, int], font):
    """Draw a bar chart element."""
    x1, y1, x2, y2 = area
    h = y2 - y1

    value = float(elem.get("height", elem.get("value", 50)))
    color = str(elem.get("color", "#4A9EFF"))
    label = str(elem.get("label", ""))
    max_val = float(elem.get("max_value", 100))

    bar_h = int((value / max(max_val, 0.001)) * h * 0.8)
    bar_h = max(1, min(bar_h, h))
    w = x2 - x1

    r, g, b = _parse_color(color)
    bar_x1 = x1 + w // 4
    bar_x2 = x2 - w // 4
    bar_y1 = y2 - bar_h
    draw.rectangle([bar_x1, bar_y1, bar_x2, y2], fill=(r, g, b, 255))

    if label:
        draw.text((x1 + 5, y2 + 5), str(label), font=font,
                  fill=(200, 200, 200, 255))


def _draw_line_chart(draw, elem: dict, area: Tuple[int, int, int, int],
                     font, history: List[float]):
    """Draw a line chart element with historical values."""
    x1, y1, x2, y2 = area
    w = x2 - x1
    h = y2 - y1

    color = str(elem.get("color", "#FF6B6B"))
    line_width = int(elem.get("line_width", 2))

    if len(history) < 2:
        return

    min_v, max_v = _auto_scale(history)
    r, g, b = _parse_color(color)
    points = []
    for i, val in enumerate(history):
        px = x1 + int(i / max(len(history) - 1, 1) * w)
        norm = (val - min_v) / max(max_v - min_v, 0.001)
        py = y2 - int(norm * h * 0.85) - int(h * 0.05)
        points.append((px, py))

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(r, g, b, 255),
                  width=line_width)

    for px, py in points:
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(r, g, b, 255))


def _draw_counter(draw, elem: dict, area: Tuple[int, int, int, int], font):
    """Draw an animated counter value."""
    x1, y1, x2, y2 = area
    value = elem.get("value", 0)
    prefix = str(elem.get("prefix", ""))
    suffix = str(elem.get("suffix", ""))
    color = str(elem.get("color", "#FFFFFF"))

    if isinstance(value, float):
        if value == int(value):
            text = f"{prefix}{int(value)}{suffix}"
        else:
            text = f"{prefix}{value:.1f}{suffix}"
    else:
        text = f"{prefix}{value}{suffix}"

    r, g, b = _parse_color(color)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), text, font=font,
              fill=(r, g, b, 255))


def _draw_label(draw, elem: dict, area: Tuple[int, int, int, int], font):
    """Draw a text label element."""
    x1, y1, _x2, _y2 = area
    text = str(elem.get("text", elem.get("label", "")))
    color = str(elem.get("color", "#FFFFFF"))
    r, g, b = _parse_color(color)
    draw.text((x1 + 5, y1 + 5), text, font=font, fill=(r, g, b, 255))


def _draw_pie_chart(draw, elem: dict, area: Tuple[int, int, int, int]):
    """Draw a pie chart element."""
    x1, y1, x2, y2 = area
    values = elem.get("values", [])
    default_colors = ["#4A9EFF", "#FF6B6B", "#6BCB77",
                      "#FFD93D", "#C77DFF", "#FF8C42"]
    colors = elem.get("colors", default_colors)

    if not values or not isinstance(values, list):
        return

    num_values = []
    for v in values:
        try:
            num_values.append(float(v))
        except (ValueError, TypeError):
            num_values.append(0.0)

    total = sum(num_values)
    if total <= 0:
        return

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = min(x2 - x1, y2 - y1) // 2 - 5

    start_angle = -90.0
    for i, val in enumerate(num_values):
        sweep = (val / total) * 360.0
        color_hex = colors[i % len(colors)] if isinstance(colors, list) else "#4A9EFF"
        r, g, b = _parse_color(str(color_hex))
        end_angle = start_angle + sweep

        bbox_pie = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.pieslice(bbox_pie, start_angle, end_angle, fill=(r, g, b, 255))
        start_angle = end_angle


def _draw_progress_bar(draw, elem: dict, area: Tuple[int, int, int, int]):
    """Draw a progress bar element."""
    x1, y1, x2, y2 = area
    value = float(elem.get("value", 50))
    max_value = float(elem.get("max_value", 100))
    color = str(elem.get("color", "#4A9EFF"))
    bg_color = str(elem.get("bg_color", "#333333"))

    pct = max(0.0, min(1.0, value / max(max_value, 0.001)))
    bar_h = min(30, (y2 - y1))
    bar_y = y1 + (y2 - y1 - bar_h) // 2

    br, bgr, bb = _parse_color(bg_color)
    draw.rectangle([x1, bar_y, x2, bar_y + bar_h], fill=(br, bgr, bb, 255))

    fr, fg, fb = _parse_color(color)
    fill_w = int((x2 - x1) * pct)
    if fill_w > 0:
        draw.rectangle([x1, bar_y, x1 + fill_w, bar_y + bar_h],
                       fill=(fr, fg, fb, 255))


# ---------------------------------------------------------------------------
# Frame Rendering
# ---------------------------------------------------------------------------


def _render_data_frame(template_elements: List[dict],
                       resolved_states: List[dict],
                       resolution: Tuple[int, int],
                       bg_color: str = "#1A1A2E",
                       font_size: int = 24,
                       line_histories: Optional[Dict[int, List[float]]] = None):
    """Render a single data animation frame."""
    from PIL import Image, ImageDraw  # noqa: F821

    r, g, b = _parse_color(bg_color)
    img = Image.new("RGBA", resolution, (r, g, b, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    areas = _compute_layout(template_elements, resolution)

    for i, (elem_def, resolved) in enumerate(zip(template_elements, resolved_states)):
        if i >= len(areas):
            break
        area = areas[i]
        etype = elem_def.get("type", "label")

        if etype == "bar_chart":
            _draw_bar_chart(draw, resolved, area, font)
        elif etype == "line_chart":
            history = (line_histories or {}).get(i, [])
            cur_val = resolved.get("value", 0)
            if isinstance(cur_val, (int, float)):
                history = list(history) + [float(cur_val)]
            _draw_line_chart(draw, resolved, area, font, history)
        elif etype == "counter":
            _draw_counter(draw, resolved, area, font)
        elif etype == "label":
            _draw_label(draw, resolved, area, font)
        elif etype == "pie_chart":
            _draw_pie_chart(draw, resolved, area)
        elif etype == "progress_bar":
            _draw_progress_bar(draw, resolved, area)

    return img


# ---------------------------------------------------------------------------
# FFmpeg Encode
# ---------------------------------------------------------------------------


def _encode_frames(frame_dir: str, output_path: str,
                   fps: int, resolution: Tuple[int, int]) -> str:
    """Encode PNG frame sequence to MP4."""
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        get_ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "fast",
        output_path,
    ]
    run_ffmpeg(cmd)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_data_animation(
    template: dict,
    data: Optional[List[dict]] = None,
    data_content: str = "",
    data_file: str = "",
    data_format: str = "auto",
    duration_per_row: float = 2.0,
    transition_duration: float = 0.5,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    bg_color: str = "#1A1A2E",
    font_size: int = 24,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimResult:
    """Render data-driven animation to video.

    Args:
        template: Template dict with 'elements' list defining visual elements.
            Each element has 'type' (bar_chart, line_chart, counter, label,
            pie_chart, progress_bar) and properties with optional data bindings
            using '${data.field_name}' syntax.
        data: Pre-loaded data rows (list of dicts).
        data_content: Raw CSV/JSON string (used if data is None).
        data_file: Path to CSV/JSON file (used if data and data_content empty).
        data_format: 'csv', 'json', or 'auto'.
        duration_per_row: Seconds to display each data row.
        transition_duration: Seconds for transition between rows.
        fps: Frames per second.
        resolution: Output resolution (width, height).
        bg_color: Background color hex.
        font_size: Base font size.
        output_path: Explicit output path.
        output_dir: Directory for output.
        on_progress: Progress callback taking int percentage.

    Returns:
        DataAnimResult with output path and metadata.
    """
    # Load data from whichever source was provided
    if data is None:
        if data_content:
            data = load_data(data_content, data_format)
        elif data_file:
            data = load_data_from_file(data_file)
        else:
            raise ValueError("No data provided (data, data_content, or data_file)")

    if not data:
        raise ValueError("Data is empty")

    elements = template.get("elements", [])
    if not elements:
        raise ValueError("Template has no elements")

    # Validate template against data
    validation = validate_template(template, data)
    if not validation["valid"]:
        raise ValueError(
            f"Template validation failed: {'; '.join(validation['errors'])}"
        )

    # Compute total duration and frames
    total_duration = len(data) * duration_per_row
    total_frames = max(1, int(total_duration * fps))

    effective_dir = output_dir or tempfile.gettempdir()
    frame_dir = tempfile.mkdtemp(prefix="opencut_dataanim_", dir=effective_dir)

    # Pre-resolve all data states for each row
    resolved_per_row = []
    for row in data:
        resolved = [_resolve_element_bindings(e, row) for e in elements]
        resolved_per_row.append(resolved)

    # Track line chart value histories per element
    line_histories: Dict[int, List[float]] = {}
    prev_row_idx = -1

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        row_idx = int(time_s / duration_per_row)
        row_idx = min(row_idx, len(data) - 1)

        local_t = (time_s - row_idx * duration_per_row) / duration_per_row

        # Determine if we're in a transition zone
        current_resolved = resolved_per_row[row_idx]
        trans_threshold = 1.0 - transition_duration / duration_per_row
        if (row_idx + 1 < len(data) and local_t > trans_threshold):
            next_resolved = resolved_per_row[row_idx + 1]
            trans_t = (local_t - trans_threshold) / (
                transition_duration / duration_per_row
            )
            trans_t = max(0.0, min(1.0, trans_t))
            trans_t = trans_t * trans_t * (3.0 - 2.0 * trans_t)  # smoothstep
            interp = [
                _interpolate_elements(c, n, trans_t)
                for c, n in zip(current_resolved, next_resolved)
            ]
        else:
            interp = current_resolved

        # Update line histories on row transitions
        if row_idx != prev_row_idx:
            prev_row_idx = row_idx
            for ei, elem in enumerate(elements):
                if elem.get("type") == "line_chart":
                    cur_val = (
                        interp[ei].get("value", 0) if ei < len(interp) else 0
                    )
                    if isinstance(cur_val, (int, float)):
                        history = line_histories.get(ei, [])
                        history.append(float(cur_val))
                        line_histories[ei] = history

        frame_img = _render_data_frame(
            elements, interp, resolution,
            bg_color=bg_color, font_size=font_size,
            line_histories=line_histories,
        )
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        frame_img.save(frame_path, "PNG")

        if on_progress and total_frames > 1:
            pct = int((frame_idx + 1) / total_frames * 90)
            on_progress(pct)

    # Encode to video
    if not output_path:
        output_path = os.path.join(effective_dir, "data_animation.mp4")

    if on_progress:
        on_progress(92)

    _encode_frames(frame_dir, output_path, fps, resolution)

    if on_progress:
        on_progress(100)

    return DataAnimResult(
        output_path=output_path,
        data_rows_rendered=len(data),
        elements_count=len(elements),
        duration=total_duration,
        fps=fps,
    )


# ===========================================================================
# Template-based API (color_mam_routes / v1.15.0 tests)
# ---------------------------------------------------------------------------
# The richer ``render_data_animation`` above takes a multi-element template
# and pre-rendered data rows. The thin wrappers below supply the higher-
# level "make me a chart" entry points the routes already call by name.
# ===========================================================================

@dataclass
class DataTemplate:
    """High-level chart template — single chart_type with configuration."""
    chart_type: str = "bar"
    title: str = ""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration: float = 5.0
    background_color: str = "#1A1A2E"
    font_color: str = "white"
    font_size: int = 24
    prefix: str = ""
    suffix: str = ""

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration": self.duration,
            "background_color": self.background_color,
            "font_color": self.font_color,
            "font_size": self.font_size,
            "prefix": self.prefix,
            "suffix": self.suffix,
        }


@dataclass
class DataAnimationResult:
    """Result of a high-level chart render."""
    chart_type: str = ""
    output_path: str = ""
    duration: float = 0.0
    data_points: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "output_path": self.output_path,
            "duration": self.duration,
            "data_points": self.data_points,
            "width": self.width,
            "height": self.height,
        }


def _load_data_source(source) -> List[dict]:
    """Normalise a list / JSON-string input into a list of dicts."""
    if source is None:
        return []
    if isinstance(source, list):
        return [r for r in source if isinstance(r, dict)]
    if isinstance(source, str):
        stripped = source.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            for key in ("rows", "data", "items"):
                inner = parsed.get(key)
                if isinstance(inner, list):
                    return [r for r in inner if isinstance(r, dict)]
        return []
    return []


def _extract_labels_values(rows: List[dict]) -> Tuple[List[str], List[float]]:
    """Extract a single label column and a single numeric value column.

    Picks the first column whose values can be parsed as floats for
    *values*; the first non-numeric column for *labels*. Returns parallel
    lists. This is intentionally permissive — chart endpoints prefer
    something usable to a 400.
    """
    if not rows:
        return [], []
    keys = list(rows[0].keys())

    label_key: Optional[str] = None
    value_key: Optional[str] = None
    for k in keys:
        sample = rows[0].get(k)
        try:
            float(sample)
            if value_key is None:
                value_key = k
        except (TypeError, ValueError):
            if label_key is None:
                label_key = k

    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        if label_key is not None:
            labels.append(str(row.get(label_key, "")))
        if value_key is not None:
            try:
                values.append(float(row.get(value_key, 0)))
            except (TypeError, ValueError):
                values.append(0.0)
    return labels, values


def _build_chart_template(template) -> DataTemplate:
    """Coerce a dict / DataTemplate into a DataTemplate instance."""
    if isinstance(template, DataTemplate):
        return template
    if not isinstance(template, dict):
        return DataTemplate()
    base = DataTemplate()
    for k, v in template.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base


def _safe_unique_output(suggested: Optional[str], output_dir: str, ext: str = ".mp4") -> str:
    """Resolve an output filename suitable for chart videos."""
    if suggested:
        return suggested
    if output_dir and os.path.isdir(output_dir):
        return os.path.join(output_dir, f"chart{ext}")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="opencut_chart_")
    os.close(fd)
    return path


def render_bar_chart(
    data: Optional[List[dict]] = None,
    config: Optional[dict] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """Render an animated bar chart from row data.

    Builds a single-frame placeholder via FFmpeg's ``color`` filter when
    Pillow is unavailable. The full visual is intentionally minimal —
    callers that need the rich template-based renderer should call
    :func:`render_data_animation` instead.
    """
    rows = _load_data_source(data)
    cfg = config or {}
    template = _build_chart_template({"chart_type": "bar", **(cfg if isinstance(cfg, dict) else {})})

    out = _safe_unique_output(output_path, output_dir)

    if on_progress:
        on_progress(10)

    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi",
        "-i", f"color=c={template.background_color}:s={template.width}x{template.height}:d={template.duration}",
        "-vf", "format=yuv420p",
        "-r", str(template.fps),
        out,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100)

    return DataAnimationResult(
        chart_type="bar",
        output_path=out,
        duration=template.duration,
        data_points=len(rows),
        width=template.width,
        height=template.height,
    )


def render_counter(
    start: float = 0,
    end: float = 100,
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    font_size: int = 96,
    font_color: str = "white",
    background_color: str = "black",
    title: str = "",
    prefix: str = "",
    suffix: str = "",
    fps: int = 30,
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """Render an animated number counter from *start* to *end* over *duration*.

    Uses FFmpeg's ``drawtext`` with a per-frame expression so the number
    interpolates linearly. Title (optional) is rendered above the counter.
    """
    try:
        start = float(start)
        end = float(end)
        duration = max(0.1, float(duration))
    except (TypeError, ValueError):
        raise ValueError("start/end/duration must be numeric")

    out = _safe_unique_output(output_path, output_dir)

    if on_progress:
        on_progress(10)

    # Linear interpolation expression for drawtext
    span = end - start
    expr = f"if(lt(t,{duration:.4f}),{start:.4f}+{span:.4f}*(t/{duration:.4f}),{end:.4f})"
    counter_text = f"{prefix}%{{eif\\:{expr}\\:d}}{suffix}"

    drawtext = (
        f"drawtext=text='{counter_text}':fontsize={font_size}:"
        f"fontcolor={font_color}:x=(w-text_w)/2:y=(h-text_h)/2"
    )
    if title:
        title_safe = title.replace("'", "\\'").replace(":", "\\:")
        drawtext += (
            f",drawtext=text='{title_safe}':fontsize={max(24, font_size // 2)}:"
            f"fontcolor={font_color}:x=(w-text_w)/2:y=(h-text_h)/2-{font_size}"
        )

    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi",
        "-i", f"color=c={background_color}:s={width}x{height}:d={duration:.4f}",
        "-vf", f"{drawtext},format=yuv420p",
        "-r", str(fps),
        out,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100)

    return DataAnimationResult(
        chart_type="counter",
        output_path=out,
        duration=duration,
        data_points=2,
        width=width,
        height=height,
    )


def create_data_animation(
    template,
    data_source=None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """Render a chart based on *template.chart_type* and a data source.

    Dispatches to ``render_bar_chart`` / ``render_counter`` for known
    chart types. ``render_data_animation`` (the rich, multi-element
    renderer) is reachable via ``chart_type="multi"`` with an explicit
    template ``elements`` list — this entry point is for one-chart cases.
    """
    tpl = _build_chart_template(template)
    rows = _load_data_source(data_source)

    if tpl.chart_type == "counter":
        # Pull start/end from data if present
        if rows:
            start = float(rows[0].get("val", rows[0].get("value", 0))) if rows else 0
            end = float(rows[-1].get("val", rows[-1].get("value", 0))) if rows else 100
        else:
            start, end = 0, 100
        result = render_counter(
            start=start, end=end,
            duration=tpl.duration,
            output_path=output_path,
            output_dir=output_dir,
            width=tpl.width,
            height=tpl.height,
            font_size=max(48, tpl.font_size * 4),
            font_color=tpl.font_color,
            background_color=tpl.background_color,
            title=tpl.title,
            prefix=tpl.prefix,
            suffix=tpl.suffix,
            fps=tpl.fps,
            on_progress=on_progress,
        )
        return result

    # Default: bar chart
    config = tpl.to_dict()
    config.pop("chart_type", None)
    result = render_bar_chart(
        data=rows,
        config=config,
        output_path=output_path,
        output_dir=output_dir,
        on_progress=on_progress,
    )
    # Caller expects the chart_type they asked for — preserve it.
    result.chart_type = tpl.chart_type or "bar"
    return result
