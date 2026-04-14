"""
OpenCut Mood Board Generator (59.3)

Extract keyframes from footage, analyze visual properties, and generate
a mood board image:
- Extract keyframes at intervals via FFmpeg
- Analyze dominant colors (k-means clustering), brightness, contrast, saturation
- Generate mood board image (Pillow): keyframes + color swatches + style tags
- Suggest matching LUTs based on color temperature / mood

Uses FFmpeg for frame extraction and Pillow for image assembly.
"""

import logging
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import ensure_package, get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ColorInfo:
    """A dominant color extracted from footage."""
    r: int = 0
    g: int = 0
    b: int = 0
    hex_code: str = "#000000"
    percentage: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameAnalysis:
    """Analysis of a single keyframe."""
    frame_path: str = ""
    timestamp: float = 0.0
    brightness: float = 0.0       # 0-255
    contrast: float = 0.0         # std dev of luminance
    saturation: float = 0.0       # average saturation 0-255
    dominant_colors: List[ColorInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class MoodBoardResult:
    """Complete mood board output."""
    mood_board_path: str = ""
    keyframes: List[FrameAnalysis] = field(default_factory=list)
    overall_brightness: float = 0.0
    overall_contrast: float = 0.0
    overall_saturation: float = 0.0
    palette: List[ColorInfo] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    suggested_luts: List[str] = field(default_factory=list)
    total_keyframes: int = 0

    def to_dict(self) -> dict:
        return {
            "mood_board_path": self.mood_board_path,
            "total_keyframes": self.total_keyframes,
            "overall_brightness": round(self.overall_brightness, 1),
            "overall_contrast": round(self.overall_contrast, 1),
            "overall_saturation": round(self.overall_saturation, 1),
            "palette": [c.to_dict() for c in self.palette],
            "style_tags": self.style_tags,
            "suggested_luts": self.suggested_luts,
        }


# ---------------------------------------------------------------------------
# LUT Suggestions
# ---------------------------------------------------------------------------
_LUT_SUGGESTIONS = {
    "warm": ["Film Warm", "Golden Hour", "Sunset Glow", "Vintage Amber"],
    "cool": ["Blue Steel", "Moonlight", "Ice Cold", "Teal Shadows"],
    "high_contrast": ["Bleach Bypass", "Cross Process", "Cinema Contrast"],
    "low_contrast": ["Soft Film", "Pastel Dream", "Matte Finish"],
    "desaturated": ["Bleach Bypass", "Film Noir BW", "Muted Earth"],
    "vibrant": ["Technicolor", "Vivid Pop", "Saturated Cinema"],
    "dark": ["Film Noir", "Dark Moody", "Shadow Play"],
    "bright": ["Overexposed Film", "Day Light", "Clean Bright"],
}


def _suggest_luts(brightness: float, contrast: float, saturation: float,
                  palette: List[ColorInfo]) -> List[str]:
    """Suggest LUTs based on visual analysis."""
    luts = set()

    # Brightness-based
    if brightness < 80:
        luts.update(_LUT_SUGGESTIONS["dark"])
    elif brightness > 170:
        luts.update(_LUT_SUGGESTIONS["bright"])

    # Contrast-based
    if contrast > 60:
        luts.update(_LUT_SUGGESTIONS["high_contrast"])
    elif contrast < 30:
        luts.update(_LUT_SUGGESTIONS["low_contrast"])

    # Saturation-based
    if saturation < 40:
        luts.update(_LUT_SUGGESTIONS["desaturated"])
    elif saturation > 120:
        luts.update(_LUT_SUGGESTIONS["vibrant"])

    # Color temperature from palette
    if palette:
        avg_r = sum(c.r for c in palette) / len(palette)
        avg_b = sum(c.b for c in palette) / len(palette)
        if avg_r > avg_b + 30:
            luts.update(_LUT_SUGGESTIONS["warm"])
        elif avg_b > avg_r + 30:
            luts.update(_LUT_SUGGESTIONS["cool"])

    return sorted(luts)[:6]


def _derive_style_tags(brightness: float, contrast: float, saturation: float,
                       palette: List[ColorInfo]) -> List[str]:
    """Generate style descriptor tags from analysis."""
    tags = []

    if brightness < 80:
        tags.append("dark")
    elif brightness < 130:
        tags.append("medium-key")
    else:
        tags.append("bright")

    if contrast > 60:
        tags.append("high-contrast")
    elif contrast < 30:
        tags.append("low-contrast")

    if saturation < 40:
        tags.append("desaturated")
    elif saturation > 120:
        tags.append("vibrant")
    else:
        tags.append("natural")

    if palette:
        avg_r = sum(c.r for c in palette) / len(palette)
        avg_b = sum(c.b for c in palette) / len(palette)
        if avg_r > avg_b + 30:
            tags.append("warm-tones")
        elif avg_b > avg_r + 30:
            tags.append("cool-tones")
        else:
            tags.append("neutral")

    return tags


# ---------------------------------------------------------------------------
# Keyframe Extraction
# ---------------------------------------------------------------------------
def extract_keyframes(
    video_path: str,
    num_frames: int = 8,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> List[str]:
    """
    Extract evenly-spaced keyframes from a video using FFmpeg.

    Args:
        video_path: Path to the video file.
        num_frames: Number of keyframes to extract.
        output_dir: Directory for frame images.
        on_progress: Callback(pct, msg).

    Returns:
        List of extracted frame image paths.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        duration = 10.0  # fallback

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_mood_")
    os.makedirs(output_dir, exist_ok=True)

    num_frames = max(1, min(num_frames, 30))
    interval = duration / (num_frames + 1)
    frame_paths = []

    for i in range(num_frames):
        ts = interval * (i + 1)
        frame_path = os.path.join(output_dir, f"keyframe_{i:03d}.png")

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.isfile(frame_path):
                frame_paths.append(frame_path)
        except Exception as e:
            logger.warning("Failed to extract keyframe at %.1fs: %s", ts, e)

        if on_progress:
            pct = int((i + 1) / num_frames * 100)
            on_progress(pct, f"Extracting frame {i + 1}/{num_frames}...")

    return frame_paths


# ---------------------------------------------------------------------------
# Color Analysis
# ---------------------------------------------------------------------------
def _analyze_colors_pillow(image_path: str, num_colors: int = 5) -> List[ColorInfo]:
    """
    Extract dominant colors using Pillow's quantize (k-means approximation).

    Args:
        image_path: Path to the image.
        num_colors: Number of dominant colors to extract.

    Returns:
        List of ColorInfo objects.
    """
    ensure_package("PIL", "Pillow")
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        # Resize for speed
        img = img.resize((100, 100))
        # Quantize to num_colors
        quantized = img.quantize(colors=num_colors, method=0)
        palette = quantized.getpalette()[:num_colors * 3]
        histogram = quantized.histogram()[:num_colors]
        total_pixels = sum(histogram) or 1

        colors = []
        for i in range(min(num_colors, len(histogram))):
            r = palette[i * 3] if i * 3 < len(palette) else 0
            g = palette[i * 3 + 1] if i * 3 + 1 < len(palette) else 0
            b = palette[i * 3 + 2] if i * 3 + 2 < len(palette) else 0
            pct = (histogram[i] / total_pixels) * 100

            colors.append(ColorInfo(
                r=r, g=g, b=b,
                hex_code=f"#{r:02x}{g:02x}{b:02x}",
                percentage=round(pct, 1),
            ))

        return sorted(colors, key=lambda c: c.percentage, reverse=True)
    except Exception as e:
        logger.warning("Color analysis failed for %s: %s", image_path, e)
        return []


def _analyze_frame(image_path: str, timestamp: float = 0.0) -> FrameAnalysis:
    """Analyze a single frame for brightness, contrast, saturation, and colors."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageStat

    analysis = FrameAnalysis(frame_path=image_path, timestamp=timestamp)

    try:
        img = Image.open(image_path).convert("RGB")

        # Brightness (mean of luminance)
        grayscale = img.convert("L")
        stat = ImageStat.Stat(grayscale)
        analysis.brightness = stat.mean[0]
        analysis.contrast = stat.stddev[0]

        # Saturation (convert to HSV, measure S channel)
        hsv = img.convert("HSV")
        s_channel = hsv.split()[1]
        s_stat = ImageStat.Stat(s_channel)
        analysis.saturation = s_stat.mean[0]

        # Dominant colors
        analysis.dominant_colors = _analyze_colors_pillow(image_path)

    except Exception as e:
        logger.warning("Frame analysis failed for %s: %s", image_path, e)

    return analysis


# ---------------------------------------------------------------------------
# Mood Board Rendering
# ---------------------------------------------------------------------------
def render_mood_board(
    keyframes: List[FrameAnalysis],
    palette: List[ColorInfo],
    style_tags: List[str],
    output_path: str,
    board_width: int = 1920,
    board_height: int = 1080,
) -> str:
    """
    Generate a mood board image with keyframes, color swatches, and style tags.

    Args:
        keyframes: Analyzed keyframes with paths.
        palette: Overall color palette.
        style_tags: Descriptive style tags.
        output_path: Where to save the mood board image.
        board_width: Output width.
        board_height: Output height.

    Returns:
        Path to the mood board image.
    """
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFont

    board = Image.new("RGB", (board_width, board_height), (25, 25, 30))
    draw = ImageDraw.Draw(board)

    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    margin = 30

    # Title
    draw.text((margin, margin // 2), "MOOD BOARD", fill=(255, 255, 255), font=font_large)

    # Keyframe grid area (top 65% of board)
    grid_top = margin + 40
    grid_height = int(board_height * 0.60)
    grid_width = board_width - 2 * margin

    if keyframes:
        # Calculate grid layout
        n = len(keyframes)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        cell_w = grid_width // cols
        cell_h = grid_height // rows
        padding = 5

        for i, kf in enumerate(keyframes):
            col = i % cols
            row = i // cols
            x = margin + col * cell_w + padding
            y = grid_top + row * cell_h + padding
            w = cell_w - 2 * padding
            h = cell_h - 2 * padding

            try:
                img = Image.open(kf.frame_path).convert("RGB")
                img = img.resize((w, h))
                board.paste(img, (x, y))
            except Exception:
                draw.rectangle([x, y, x + w, y + h], fill=(50, 50, 55))

    # Color palette section (bottom 35%)
    palette_top = grid_top + grid_height + 20
    swatch_size = 60
    swatch_gap = 10

    draw.text((margin, palette_top - 5), "COLOR PALETTE",
              fill=(200, 200, 200), font=font_medium)
    palette_y = palette_top + 25

    for i, color in enumerate(palette[:8]):
        x = margin + i * (swatch_size + swatch_gap)
        # Swatch rectangle
        draw.rounded_rectangle(
            [x, palette_y, x + swatch_size, palette_y + swatch_size],
            radius=5, fill=(color.r, color.g, color.b),
            outline=(100, 100, 100),
        )
        # Hex label
        draw.text((x, palette_y + swatch_size + 3),
                  color.hex_code, fill=(150, 150, 150), font=font_small)
        draw.text((x, palette_y + swatch_size + 17),
                  f"{color.percentage:.0f}%", fill=(120, 120, 120), font=font_small)

    # Style tags section (right side)
    tags_x = board_width - margin - 300
    draw.text((tags_x, palette_top - 5), "STYLE",
              fill=(200, 200, 200), font=font_medium)
    tag_y = palette_top + 25
    for tag in style_tags[:8]:
        draw.text((tags_x, tag_y), f"  {tag}", fill=(180, 200, 255), font=font_medium)
        tag_y += 24

    board.save(output_path, "PNG")
    return output_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def generate_mood_board(
    video_path: str,
    output_dir: str,
    num_keyframes: int = 8,
    num_colors: int = 5,
    on_progress: Optional[Callable] = None,
) -> MoodBoardResult:
    """
    Generate a mood board from video footage.

    Pipeline:
    1. Extract keyframes from video
    2. Analyze each frame (colors, brightness, contrast, saturation)
    3. Compute overall palette and style tags
    4. Render mood board image
    5. Suggest matching LUTs

    Args:
        video_path: Path to the source video.
        output_dir: Directory for output files.
        num_keyframes: Number of keyframes to extract.
        num_colors: Number of dominant colors per frame.
        on_progress: Callback(pct, msg).

    Returns:
        MoodBoardResult with analysis and output path.
    """
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow is required for mood board generation")

    os.makedirs(output_dir, exist_ok=True)
    result = MoodBoardResult()

    # Phase 1: Extract keyframes
    if on_progress:
        on_progress(5, "Extracting keyframes...")

    frames_dir = os.path.join(output_dir, "keyframes")
    frame_paths = extract_keyframes(
        video_path, num_frames=num_keyframes,
        output_dir=frames_dir,
    )

    if not frame_paths:
        raise RuntimeError("No keyframes could be extracted from video")

    # Phase 2: Analyze each frame
    if on_progress:
        on_progress(30, "Analyzing frames...")

    info = get_video_info(video_path)
    duration = info.get("duration", 0) or 10.0
    interval = duration / (num_keyframes + 1)

    for i, fp in enumerate(frame_paths):
        ts = interval * (i + 1)
        analysis = _analyze_frame(fp, timestamp=ts)
        result.keyframes.append(analysis)

        if on_progress:
            pct = 30 + int((i + 1) / len(frame_paths) * 30)
            on_progress(pct, f"Analyzing frame {i + 1}/{len(frame_paths)}...")

    result.total_keyframes = len(result.keyframes)

    # Phase 3: Compute overall metrics
    if on_progress:
        on_progress(65, "Computing overall palette...")

    if result.keyframes:
        result.overall_brightness = sum(
            kf.brightness for kf in result.keyframes
        ) / len(result.keyframes)
        result.overall_contrast = sum(
            kf.contrast for kf in result.keyframes
        ) / len(result.keyframes)
        result.overall_saturation = sum(
            kf.saturation for kf in result.keyframes
        ) / len(result.keyframes)

    # Merge all frame colors into an overall palette
    all_colors: dict = {}  # hex -> (r, g, b, total_pct, count)
    for kf in result.keyframes:
        for c in kf.dominant_colors:
            key = c.hex_code
            if key in all_colors:
                existing = all_colors[key]
                all_colors[key] = (
                    c.r, c.g, c.b,
                    existing[3] + c.percentage,
                    existing[4] + 1,
                )
            else:
                all_colors[key] = (c.r, c.g, c.b, c.percentage, 1)

    # Top colors by aggregate percentage
    sorted_colors = sorted(all_colors.values(), key=lambda x: x[3], reverse=True)
    total_pct = sum(c[3] for c in sorted_colors) or 1
    for r, g, b, pct, _ in sorted_colors[:num_colors]:
        result.palette.append(ColorInfo(
            r=r, g=g, b=b,
            hex_code=f"#{r:02x}{g:02x}{b:02x}",
            percentage=round((pct / total_pct) * 100, 1),
        ))

    # Phase 4: Style tags and LUT suggestions
    result.style_tags = _derive_style_tags(
        result.overall_brightness, result.overall_contrast,
        result.overall_saturation, result.palette,
    )
    result.suggested_luts = _suggest_luts(
        result.overall_brightness, result.overall_contrast,
        result.overall_saturation, result.palette,
    )

    # Phase 5: Render mood board image
    if on_progress:
        on_progress(80, "Rendering mood board...")

    board_path = os.path.join(output_dir, "mood_board.png")
    render_mood_board(
        result.keyframes, result.palette, result.style_tags,
        board_path,
    )
    result.mood_board_path = board_path

    if on_progress:
        on_progress(100, f"Mood board complete: {result.total_keyframes} keyframes analyzed")

    return result
