"""
OpenCut AI Scene Description / Alt-Text Generator

Extracts key frames at given timestamps and generates textual descriptions
using LLM when available, falling back to heuristic frame analysis (dominant
colors, brightness, edge density).

Uses FFmpeg for frame extraction and ffprobe for metadata.
"""

import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SceneDescription:
    """Description for a single scene."""
    timestamp: float
    description: str = ""
    alt_text: str = ""
    dominant_colors: List[str] = field(default_factory=list)
    brightness: float = 0.0
    edge_density: float = 0.0
    tags: List[str] = field(default_factory=list)
    method: str = "heuristic"  # "heuristic" or "llm"


@dataclass
class SceneDescriptionResult:
    """Complete scene description results."""
    descriptions: List[SceneDescription] = field(default_factory=list)
    total_scenes: int = 0
    duration: float = 0.0
    method: str = "heuristic"


# ---------------------------------------------------------------------------
# Color name mapping (RGB buckets -> human readable)
# ---------------------------------------------------------------------------
_COLOR_NAMES = {
    (0, 0, 0): "black",
    (128, 128, 128): "gray",
    (255, 255, 255): "white",
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (255, 128, 0): "orange",
    (128, 0, 128): "purple",
    (0, 255, 255): "cyan",
    (255, 192, 203): "pink",
    (139, 69, 19): "brown",
    (0, 128, 0): "dark green",
    (0, 0, 128): "dark blue",
    (128, 0, 0): "dark red",
    (245, 222, 179): "beige",
}


def _closest_color_name(r: int, g: int, b: int) -> str:
    """Map an RGB value to the closest named color."""
    best_name = "unknown"
    best_dist = float("inf")
    for (cr, cg, cb), name in _COLOR_NAMES.items():
        dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Frame analysis helpers
# ---------------------------------------------------------------------------
def _extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at the given timestamp."""
    cmd = (
        FFmpegCmd()
        .pre_input("ss", str(timestamp))
        .input(video_path)
        .frames(1)
        .option("q:v", "2")
        .output(output_path)
        .build()
    )
    try:
        run_ffmpeg(cmd, timeout=30)
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    except RuntimeError as exc:
        logger.debug("Frame extraction at %.2fs failed: %s", timestamp, exc)
        return False


def _analyze_frame_colors(frame_path: str) -> List[str]:
    """Analyze dominant colors in a frame using FFmpeg histogram."""
    try:
        # Scale down to 4x4 to get average color regions
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", frame_path,
            "-vf", "scale=4:4,format=rgb24,signalstats=stat=tout+vrep+brng",
            "-frames:v", "1",
            "-f", "rawvideo", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or len(result.stdout) < 48:
            return ["unknown"]

        # Parse 4x4 RGB pixels and find dominant color clusters
        pixels = result.stdout
        color_counts: Dict[str, int] = {}
        for i in range(0, min(len(pixels), 48), 3):
            r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
            name = _closest_color_name(r, g, b)
            color_counts[name] = color_counts.get(name, 0) + 1

        # Return top 3 colors sorted by frequency
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_colors[:3]]
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Color analysis failed for %s: %s", frame_path, exc)
        return ["unknown"]


def _analyze_frame_brightness(frame_path: str) -> float:
    """Compute average brightness of a frame (0.0 - 1.0)."""
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", frame_path,
            "-vf", "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # Parse YAVG (average luminance) from signalstats
        yavg_match = re.search(r"SIGNALSTATS\.YAVG=(\d+\.?\d*)", result.stderr)
        if yavg_match:
            # YAVG is 0-255, normalize to 0-1
            return float(yavg_match.group(1)) / 255.0

        # Fallback: read raw grayscale pixel values
        gray_cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", frame_path,
            "-vf", "scale=8:8,format=gray",
            "-frames:v", "1",
            "-f", "rawvideo", "-",
        ]
        gray_result = subprocess.run(gray_cmd, capture_output=True, timeout=15)
        if gray_result.returncode == 0 and gray_result.stdout:
            pixels = gray_result.stdout
            avg = sum(pixels) / len(pixels)
            return avg / 255.0

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Brightness analysis failed for %s: %s", frame_path, exc)
    return 0.5


def _analyze_frame_edges(frame_path: str) -> float:
    """Compute edge density of a frame (0.0 - 1.0). Higher = more detail/edges."""
    try:
        # Apply edge detection and measure the result
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", frame_path,
            "-vf", "edgedetect=low=0.1:high=0.3,scale=16:16,format=gray",
            "-frames:v", "1",
            "-f", "rawvideo", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0 and result.stdout:
            pixels = result.stdout
            # Edge pixels are bright (white), non-edge are dark
            edge_pixels = sum(1 for p in pixels if p > 128)
            return edge_pixels / max(len(pixels), 1)
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Edge analysis failed for %s: %s", frame_path, exc)
    return 0.5


def _generate_heuristic_description(
    colors: List[str],
    brightness: float,
    edge_density: float,
    video_info: dict,
) -> str:
    """Generate a textual description from heuristic frame analysis."""
    parts = []

    # Brightness description
    if brightness < 0.15:
        parts.append("A very dark scene")
    elif brightness < 0.35:
        parts.append("A dimly lit scene")
    elif brightness < 0.65:
        parts.append("A moderately lit scene")
    elif brightness < 0.85:
        parts.append("A bright scene")
    else:
        parts.append("A very brightly lit scene")

    # Color description
    meaningful_colors = [c for c in colors if c != "unknown"]
    if meaningful_colors:
        color_str = ", ".join(meaningful_colors[:3])
        parts.append(f"with dominant {color_str} tones")

    # Detail/complexity description
    if edge_density > 0.6:
        parts.append("containing high visual detail and complex elements")
    elif edge_density > 0.3:
        parts.append("with moderate visual complexity")
    elif edge_density < 0.1:
        parts.append("with a clean, minimalist composition")

    # Aspect ratio hint
    w = video_info.get("width", 1920)
    h = video_info.get("height", 1080)
    aspect = w / max(h, 1)
    if aspect > 2.0:
        parts.append("in ultra-wide format")
    elif aspect > 1.85:
        parts.append("in cinematic widescreen")
    elif aspect < 1.0:
        parts.append("in vertical/portrait format")

    description = " ".join(parts) + "."
    return description


def _generate_alt_text(
    colors: List[str],
    brightness: float,
    edge_density: float,
) -> str:
    """Generate concise alt-text for accessibility."""
    brightness_word = "dark" if brightness < 0.35 else "bright" if brightness > 0.65 else "neutral"
    detail_word = "detailed" if edge_density > 0.4 else "simple"
    color_str = " and ".join([c for c in colors[:2] if c != "unknown"]) or "mixed"
    return f"Video frame: {brightness_word} {detail_word} scene with {color_str} tones"


def _infer_tags(
    colors: List[str],
    brightness: float,
    edge_density: float,
) -> List[str]:
    """Infer descriptive tags from frame analysis."""
    tags = []
    if brightness < 0.2:
        tags.append("dark")
        tags.append("low-key")
    elif brightness > 0.8:
        tags.append("bright")
        tags.append("high-key")

    if edge_density > 0.5:
        tags.append("detailed")
        tags.append("complex")
    elif edge_density < 0.15:
        tags.append("minimal")
        tags.append("clean")

    for color in colors:
        if color != "unknown":
            tags.append(color)

    if "blue" in colors and brightness > 0.5:
        tags.append("outdoor")
        tags.append("sky")
    if "green" in colors or "dark green" in colors:
        tags.append("nature")
    if "black" in colors and brightness < 0.3:
        tags.append("night")
    if "orange" in colors or "yellow" in colors:
        tags.append("warm")

    return tags


# ---------------------------------------------------------------------------
# LLM-based description
# ---------------------------------------------------------------------------
def _generate_llm_description(
    frame_path: str,
    colors: List[str],
    brightness: float,
    edge_density: float,
    timestamp: float,
    llm_config: Optional[dict] = None,
) -> Optional[str]:
    """Attempt to generate a description via LLM. Returns None if unavailable."""
    try:
        from opencut.core.llm import LLMConfig, query_llm

        config = LLMConfig()
        if llm_config:
            if "provider" in llm_config:
                config.provider = llm_config["provider"]
            if "model" in llm_config:
                config.model = llm_config["model"]
            if "api_key" in llm_config:
                config.api_key = llm_config["api_key"]
            if "base_url" in llm_config:
                config.base_url = llm_config["base_url"]

        color_info = ", ".join(colors[:3]) if colors else "unknown"
        bright_desc = "dark" if brightness < 0.35 else "bright" if brightness > 0.65 else "moderately lit"
        detail_desc = "high detail" if edge_density > 0.5 else "moderate detail" if edge_density > 0.2 else "low detail"

        prompt = (
            f"Describe a video scene at timestamp {timestamp:.1f}s. "
            f"Frame analysis shows: dominant colors are {color_info}, "
            f"the scene is {bright_desc} with {detail_desc}. "
            f"Write a concise 1-2 sentence description suitable for alt-text "
            f"and video accessibility. Focus on what might be visually present."
        )
        system_prompt = (
            "You are a video accessibility assistant. Generate concise, "
            "descriptive alt-text for video scenes based on frame analysis data."
        )

        response = query_llm(prompt, config=config, system_prompt=system_prompt)
        if response.text and not response.text.startswith("LLM error:"):
            return response.text.strip()

    except (ImportError, Exception) as exc:
        logger.debug("LLM description generation failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def describe_scene(
    video_path: str,
    timestamp: float,
    llm_config: Optional[dict] = None,
    on_progress: Optional[Callable] = None,
) -> SceneDescription:
    """
    Generate a description for a single scene at the given timestamp.

    Extracts a key frame and analyzes it using heuristic methods
    (dominant colors, brightness, edge density). Optionally uses LLM
    for richer descriptions when available.

    Args:
        video_path: Path to video file.
        timestamp: Time in seconds to extract frame.
        llm_config: Optional dict with LLM provider settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        SceneDescription with analysis results.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, f"Extracting frame at {timestamp:.1f}s...")

    video_info = get_video_info(video_path)
    duration = video_info.get("duration", 0)
    # Clamp timestamp to video duration
    if duration > 0:
        timestamp = max(0.0, min(timestamp, duration))

    tmp_dir = tempfile.mkdtemp(prefix="opencut_scenedesc_")
    frame_path = os.path.join(tmp_dir, "frame.jpg")

    try:
        extracted = _extract_frame(video_path, timestamp, frame_path)
        if not extracted:
            logger.warning("Could not extract frame at %.2fs, returning minimal description", timestamp)
            return SceneDescription(
                timestamp=timestamp,
                description="Unable to extract frame for analysis.",
                alt_text="Video frame (extraction failed)",
                method="heuristic",
            )

        if on_progress:
            on_progress(30, "Analyzing frame colors...")

        colors = _analyze_frame_colors(frame_path)

        if on_progress:
            on_progress(50, "Analyzing brightness and detail...")

        brightness = _analyze_frame_brightness(frame_path)
        edge_density = _analyze_frame_edges(frame_path)

        if on_progress:
            on_progress(70, "Generating description...")

        tags = _infer_tags(colors, brightness, edge_density)
        method = "heuristic"

        # Try LLM first if config provided
        llm_desc = None
        if llm_config:
            llm_desc = _generate_llm_description(
                frame_path, colors, brightness, edge_density,
                timestamp, llm_config,
            )

        if llm_desc:
            description = llm_desc
            alt_text = llm_desc[:200] if len(llm_desc) > 200 else llm_desc
            method = "llm"
        else:
            description = _generate_heuristic_description(colors, brightness, edge_density, video_info)
            alt_text = _generate_alt_text(colors, brightness, edge_density)

        if on_progress:
            on_progress(100, "Scene description complete")

        return SceneDescription(
            timestamp=timestamp,
            description=description,
            alt_text=alt_text,
            dominant_colors=colors,
            brightness=round(brightness, 3),
            edge_density=round(edge_density, 3),
            tags=tags,
            method=method,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def describe_all_scenes(
    video_path: str,
    scene_timestamps: Optional[List[float]] = None,
    llm_config: Optional[dict] = None,
    on_progress: Optional[Callable] = None,
) -> SceneDescriptionResult:
    """
    Generate descriptions for all scenes in a video.

    If scene_timestamps is not provided, auto-detects scenes using
    FFmpeg scene change detection.

    Args:
        video_path: Path to video file.
        scene_timestamps: List of timestamps to describe. Auto-detects if None.
        llm_config: Optional dict with LLM provider settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        SceneDescriptionResult with all scene descriptions.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_info = get_video_info(video_path)
    duration = video_info.get("duration", 0)

    if on_progress:
        on_progress(5, "Preparing scene analysis...")

    # Auto-detect scenes if timestamps not provided
    if scene_timestamps is None:
        if on_progress:
            on_progress(10, "Detecting scene boundaries...")
        from opencut.core.scene_detect import detect_scenes
        scene_info = detect_scenes(video_path, threshold=0.3, min_scene_length=2.0)
        scene_timestamps = [b.time for b in scene_info.boundaries]

    if not scene_timestamps:
        scene_timestamps = [0.0]

    total = len(scene_timestamps)
    descriptions: List[SceneDescription] = []
    method_used = "heuristic"

    for i, ts in enumerate(scene_timestamps):
        pct_base = 15 + int(80 * i / total)
        if on_progress:
            on_progress(pct_base, f"Describing scene {i + 1}/{total} at {ts:.1f}s...")

        desc = describe_scene(
            video_path,
            timestamp=ts,
            llm_config=llm_config,
        )
        descriptions.append(desc)
        if desc.method == "llm":
            method_used = "llm"

    if on_progress:
        on_progress(100, f"Described {total} scenes")

    return SceneDescriptionResult(
        descriptions=descriptions,
        total_scenes=total,
        duration=duration,
        method=method_used,
    )
