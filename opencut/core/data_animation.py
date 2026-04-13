"""
OpenCut Data-Driven Animation (26.2)

Bind motion graphics to CSV/JSON data sources:
- Animated bar charts from data
- Animated counters (number tickers)
- Template-based data visualization
- Support for CSV and JSON data sources

Uses FFmpeg drawtext and drawbox filters for rendering.
"""

import csv
import io
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHART_TYPES = ["bar", "horizontal_bar", "counter", "label_value"]

DEFAULT_COLORS = [
    "#4285F4", "#EA4335", "#FBBC05", "#34A853",
    "#FF6D01", "#46BDC6", "#7B1FA2", "#C2185B",
    "#00897B", "#FFB300",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DataTemplate:
    """Template for data-driven animation."""
    chart_type: str = "bar"
    title: str = ""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration: float = 5.0
    background_color: str = "black"
    text_color: str = "white"
    bar_colors: List[str] = field(default_factory=lambda: list(DEFAULT_COLORS))
    font_size: int = 36
    title_size: int = 48
    padding: int = 60
    animate_in: bool = True
    show_values: bool = True
    show_labels: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataAnimationResult:
    """Result from a data animation render."""
    output_path: str = ""
    chart_type: str = ""
    data_points: int = 0
    duration: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal: Data loading
# ---------------------------------------------------------------------------
def _load_data_source(data_source: Union[str, List, Dict]) -> List[Dict]:
    """Load data from CSV path, JSON path, JSON string, or direct list."""
    if isinstance(data_source, list):
        return data_source

    if isinstance(data_source, dict):
        return [data_source]

    if isinstance(data_source, str):
        # Try as file path first
        if os.path.isfile(data_source):
            ext = os.path.splitext(data_source)[1].lower()
            if ext == ".csv":
                return _load_csv(data_source)
            elif ext == ".json":
                with open(data_source, "r") as f:
                    loaded = json.load(f)
                return loaded if isinstance(loaded, list) else [loaded]

        # Try as JSON string
        try:
            loaded = json.loads(data_source)
            return loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try as CSV string
        try:
            return _parse_csv_string(data_source)
        except Exception:
            pass

    return []


def _load_csv(filepath: str) -> List[Dict]:
    """Load CSV file into list of dicts."""
    rows = []
    with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_csv_string(csv_str: str) -> List[Dict]:
    """Parse CSV string into list of dicts."""
    rows = []
    reader = csv.DictReader(io.StringIO(csv_str))
    for row in reader:
        rows.append(dict(row))
    return rows


def _extract_labels_values(data: List[Dict]) -> Tuple[List[str], List[float]]:
    """Extract label-value pairs from data dicts."""
    labels = []
    values = []
    for item in data:
        # Auto-detect label and value fields
        label = ""
        value = 0.0
        for k, v in item.items():
            try:
                value = float(v)
                # The other field is the label
                for lk, lv in item.items():
                    if lk != k:
                        label = str(lv)
                        break
                break
            except (ValueError, TypeError):
                label = str(v)
                continue

        if not label:
            label = f"Item {len(labels) + 1}"
        labels.append(label)
        values.append(value)

    return labels, values


# ---------------------------------------------------------------------------
# Build FFmpeg filter for bar chart
# ---------------------------------------------------------------------------
def _build_bar_chart_filter(
    labels: List[str],
    values: List[float],
    template: DataTemplate,
    vertical: bool = True,
) -> str:
    """Build FFmpeg filter_complex for animated bar chart."""
    n = len(values)
    if n == 0:
        return "null"

    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1

    w = template.width
    h = template.height
    pad = template.padding
    chart_w = w - 2 * pad
    chart_h = h - 3 * pad  # extra padding for title

    filters = []

    if vertical:
        bar_width = max(2, chart_w // (n * 2))
        gap = max(1, bar_width // 2)

        for i, (label, val) in enumerate(zip(labels, values)):
            bar_h = int((val / max_val) * chart_h * 0.8)
            bar_x = pad + i * (bar_width + gap)
            bar_y = h - pad - bar_h
            color = template.bar_colors[i % len(template.bar_colors)]

            # Animated: bar grows from bottom using expression
            progress = f"min(1,t/{max(0.3, template.duration*0.6)})"
            animated_h = f"({bar_h}*{progress})" if template.animate_in else str(bar_h)
            animated_y = f"({h-pad}-{bar_h}*{progress})" if template.animate_in else str(bar_y)

            filters.append(
                f"drawbox=x={bar_x}:y='{animated_y}':"
                f"w={bar_width}:h='{animated_h}':"
                f"color={color}:t=fill"
            )

            if template.show_labels:
                escaped = label.replace("'", "\\'").replace(":", "\\:")
                filters.append(
                    f"drawtext=text='{escaped}':"
                    f"fontsize={template.font_size//2}:"
                    f"fontcolor={template.text_color}:"
                    f"x={bar_x}:y={h-pad+5}"
                )
    else:
        bar_height = max(2, chart_h // (n * 2))
        gap = max(1, bar_height // 2)

        for i, (label, val) in enumerate(zip(labels, values)):
            bar_w = int((val / max_val) * chart_w * 0.8)
            bar_x = pad + pad  # offset for labels
            bar_y = pad * 2 + i * (bar_height + gap)
            color = template.bar_colors[i % len(template.bar_colors)]

            progress = f"min(1,t/{max(0.3, template.duration*0.6)})"
            animated_w = f"({bar_w}*{progress})" if template.animate_in else str(bar_w)

            filters.append(
                f"drawbox=x={bar_x}:y={bar_y}:"
                f"w='{animated_w}':h={bar_height}:"
                f"color={color}:t=fill"
            )

            if template.show_labels:
                escaped = label.replace("'", "\\'").replace(":", "\\:")
                filters.append(
                    f"drawtext=text='{escaped}':"
                    f"fontsize={template.font_size//2}:"
                    f"fontcolor={template.text_color}:"
                    f"x={pad//2}:y={bar_y+bar_height//4}"
                )

    # Title
    if template.title:
        escaped_title = template.title.replace("'", "\\'").replace(":", "\\:")
        filters.append(
            f"drawtext=text='{escaped_title}':"
            f"fontsize={template.title_size}:"
            f"fontcolor={template.text_color}:"
            f"x=(w-text_w)/2:y={pad//2}"
        )

    return ",".join(filters) if filters else "null"


# ---------------------------------------------------------------------------
# Create Data Animation (generic)
# ---------------------------------------------------------------------------
def create_data_animation(
    template: Optional[DataTemplate] = None,
    data_source: Union[str, List, Dict, None] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """
    Create an animated visualization from a data source and template.

    Args:
        template: DataTemplate with chart type and style settings.
        data_source: CSV/JSON file path, JSON string, or list of dicts.
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        DataAnimationResult with rendered video path.
    """
    tmpl = template if isinstance(template, DataTemplate) else DataTemplate()
    if isinstance(template, dict):
        tmpl = DataTemplate(
            chart_type=template.get("chart_type", "bar"),
            title=template.get("title", ""),
            width=int(template.get("width", 1920)),
            height=int(template.get("height", 1080)),
            fps=int(template.get("fps", 30)),
            duration=float(template.get("duration", 5)),
            background_color=template.get("background_color", "black"),
            text_color=template.get("text_color", "white"),
            font_size=int(template.get("font_size", 36)),
            title_size=int(template.get("title_size", 48)),
            padding=int(template.get("padding", 60)),
            animate_in=template.get("animate_in", True),
            show_values=template.get("show_values", True),
            show_labels=template.get("show_labels", True),
        )

    data = _load_data_source(data_source) if data_source else []
    labels, values = _extract_labels_values(data)

    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"data_anim_{tmpl.chart_type}.mp4")

    if on_progress:
        on_progress(10, f"Rendering {tmpl.chart_type} chart with {len(data)} points...")

    if tmpl.chart_type == "counter":
        start = values[0] if values else 0
        end = values[-1] if len(values) > 1 else (values[0] if values else 100)
        return render_counter(
            start=start, end=end, duration=tmpl.duration,
            output_path=output_path,
            width=tmpl.width, height=tmpl.height,
            fps=tmpl.fps, font_size=tmpl.font_size * 2,
            font_color=tmpl.text_color,
            background_color=tmpl.background_color,
            title=tmpl.title,
            on_progress=on_progress,
        )

    vertical = tmpl.chart_type != "horizontal_bar"
    vf = _build_bar_chart_filter(labels, values, tmpl, vertical=vertical)

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(
            f"color=c={tmpl.background_color}:s={tmpl.width}x{tmpl.height}"
            f":d={tmpl.duration}:r={tmpl.fps}"
        )
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(tmpl.duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Data animation rendered.")

    return DataAnimationResult(
        output_path=output_path,
        chart_type=tmpl.chart_type,
        data_points=len(data),
        duration=tmpl.duration,
        width=tmpl.width,
        height=tmpl.height,
    )


# ---------------------------------------------------------------------------
# Render Bar Chart
# ---------------------------------------------------------------------------
def render_bar_chart(
    data: Union[List[Dict], str],
    config: Optional[Dict] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """
    Render an animated bar chart from data.

    Args:
        data: List of {label: X, value: Y} dicts, CSV/JSON path, or string.
        config: Chart configuration dict.
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        DataAnimationResult with rendered video path.
    """
    cfg = config or {}
    tmpl = DataTemplate(
        chart_type="bar",
        title=cfg.get("title", ""),
        width=int(cfg.get("width", 1920)),
        height=int(cfg.get("height", 1080)),
        duration=float(cfg.get("duration", 5)),
        background_color=cfg.get("background_color", "black"),
        text_color=cfg.get("text_color", "white"),
    )

    return create_data_animation(
        template=tmpl,
        data_source=data,
        output_path=output_path,
        output_dir=output_dir,
        on_progress=on_progress,
    )


# ---------------------------------------------------------------------------
# Render Counter (number ticker)
# ---------------------------------------------------------------------------
def render_counter(
    start: float = 0,
    end: float = 100,
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    font_size: int = 96,
    font_color: str = "white",
    background_color: str = "black",
    title: str = "",
    decimal_places: int = 0,
    prefix: str = "",
    suffix: str = "",
    on_progress: Optional[Callable] = None,
) -> DataAnimationResult:
    """
    Render an animated number counter (ticker).

    Args:
        start: Starting number.
        end: Ending number.
        duration: Animation duration in seconds.
        output_path: Explicit output file path.
        width: Output width.
        height: Output height.
        fps: Frame rate.
        font_size: Counter font size.
        font_color: Font color.
        background_color: Background color.
        title: Optional title above counter.
        decimal_places: Number of decimal places.
        prefix: Text before the number (e.g., "$").
        suffix: Text after the number (e.g., "%").
        on_progress: Callback(percent, message).

    Returns:
        DataAnimationResult with rendered video path.
    """
    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, "data_counter.mp4")

    if on_progress:
        on_progress(10, f"Rendering counter {start} -> {end}...")

    # FFmpeg drawtext with expression for counting
    # t goes from 0 to duration, we interpolate start to end
    progress = f"min(1,t/{max(0.01, duration)})"
    value_expr = f"{start}+({end}-{start})*{progress}"

    if decimal_places > 0:
        # Use %f format in drawtext expression
        pass
    else:
        pass

    prefix_esc = prefix.replace("'", "\\'").replace(":", "\\:").replace("%", "%%")
    suffix_esc = suffix.replace("'", "\\'").replace(":", "\\:").replace("%", "%%")
    text_expr = f"{prefix_esc}%{{eif\\:{value_expr}\\:d}}{suffix_esc}"

    drawtext = (
        f"drawtext=text='{text_expr}':"
        f"fontsize={font_size}:fontcolor={font_color}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2"
    )

    # Add title if provided
    if title:
        escaped_title = title.replace("'", "\\'").replace(":", "\\:")
        drawtext += (
            f",drawtext=text='{escaped_title}':"
            f"fontsize={font_size//2}:fontcolor={font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-{font_size}"
        )

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(
            f"color=c={background_color}:s={width}x{height}"
            f":d={duration}:r={fps}"
        )
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Counter rendered.")

    return DataAnimationResult(
        output_path=output_path,
        chart_type="counter",
        data_points=2,
        duration=duration,
        width=width,
        height=height,
    )
