"""
OpenCut Caption Burn-in Module v0.8.0

Burn subtitles/captions directly into video pixels:
- ASS/SRT/VTT subtitle overlay via FFmpeg subtitles filter
- Styled burn-in using OpenCut caption styles
- Position control (top, center, bottom)
- Font embedding for consistent rendering

Burns subtitles permanently into the video stream (hardcoded).
Use this when the target player doesn't support soft subs.
"""

import logging
import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _run_ffmpeg(cmd: List[str], timeout: int = 7200) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


def _get_video_info(filepath: str) -> Dict:
    import json
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        data = json.loads(result.stdout.decode())
        stream = data["streams"][0]
        return {
            "width": int(stream.get("width", 1920)),
            "height": int(stream.get("height", 1080)),
        }
    except Exception:
        return {"width": 1920, "height": 1080}


# ---------------------------------------------------------------------------
# Burn-in from subtitle file
# ---------------------------------------------------------------------------
def burnin_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    font_size: int = 0,
    margin_bottom: int = 0,
    force_style: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Burn a subtitle file (SRT/ASS/VTT) into video.

    Args:
        subtitle_path: Path to .srt, .ass, or .vtt file.
        font_size: Override font size (0 = use subtitle file's default).
        margin_bottom: Override bottom margin in pixels.
        force_style: FFmpeg force_style override string for SRT/VTT.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_subtitled{ext}")

    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    if on_progress:
        on_progress(10, "Burning subtitles into video...")

    ext_lower = os.path.splitext(subtitle_path)[1].lower()

    # Escape path for FFmpeg filter (Windows backslashes, colons, etc.)
    escaped_sub = subtitle_path.replace("\\", "/").replace(":", "\\:")

    if ext_lower == ".ass" or ext_lower == ".ssa":
        # ASS subtitles: use ass filter (respects all ASS styling)
        vf = f"ass='{escaped_sub}'"
    else:
        # SRT/VTT: use subtitles filter with optional force_style
        style_parts = []
        if font_size > 0:
            style_parts.append(f"FontSize={font_size}")
        if margin_bottom > 0:
            style_parts.append(f"MarginV={margin_bottom}")
        if force_style:
            style_parts.append(force_style)

        if style_parts:
            style_str = ",".join(style_parts)
            vf = f"subtitles='{escaped_sub}':force_style='{style_str}'"
        else:
            vf = f"subtitles='{escaped_sub}'"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Subtitles burned in!")
    return output_path


# ---------------------------------------------------------------------------
# Burn-in from caption segments (generates ASS then burns)
# ---------------------------------------------------------------------------
BURNIN_STYLES = {
    "default": {
        "label": "Default White",
        "fontname": "Arial",
        "fontsize": 48,
        "primary_color": "&H00FFFFFF",     # white
        "outline_color": "&H00000000",     # black
        "outline": 3,
        "shadow": 1,
        "alignment": 2,
        "margin_v": 40,
    },
    "bold_yellow": {
        "label": "Bold Yellow",
        "fontname": "Impact",
        "fontsize": 56,
        "primary_color": "&H0000FFFF",     # yellow (BGR)
        "outline_color": "&H00000000",
        "outline": 4,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 50,
    },
    "boxed_dark": {
        "label": "Dark Box",
        "fontname": "Arial",
        "fontsize": 44,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",        # semi-transparent black
        "borderstyle": 3,                  # opaque box
        "outline": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 35,
    },
    "neon_cyan": {
        "label": "Neon Cyan",
        "fontname": "Arial Bold",
        "fontsize": 50,
        "primary_color": "&H00FFFF00",     # cyan (BGR)
        "outline_color": "&H00800000",
        "outline": 3,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 45,
    },
    "cinematic_serif": {
        "label": "Cinematic Serif",
        "fontname": "Georgia",
        "fontsize": 42,
        "primary_color": "&H00D2D2DC",     # warm white
        "outline_color": "&H00000000",
        "outline": 0,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 60,
    },
    "top_center": {
        "label": "Top Center",
        "fontname": "Arial",
        "fontsize": 44,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline": 3,
        "shadow": 1,
        "alignment": 8,                    # top center
        "margin_v": 30,
    },
}


def burnin_segments(
    video_path: str,
    segments: List[Dict],
    output_path: Optional[str] = None,
    output_dir: str = "",
    style: str = "default",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Burn caption segments directly into video.

    Takes raw segments (with start, end, text) and generates
    a temporary ASS file, then burns it into the video.

    Args:
        segments: List of {"start": float, "end": float, "text": str}.
        style: Burn-in style preset name from BURNIN_STYLES.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_captioned{ext}")

    if not segments:
        raise ValueError("No caption segments provided")

    style_cfg = BURNIN_STYLES.get(style, BURNIN_STYLES["default"])
    info = _get_video_info(video_path)

    if on_progress:
        on_progress(5, "Generating subtitle file...")

    # Generate temporary ASS file
    tmp_ass = tempfile.NamedTemporaryFile(suffix=".ass", mode="w",
                                          encoding="utf-8", delete=False)
    try:
        _write_ass_file(tmp_ass, segments, style_cfg, info)
        tmp_ass.close()

        if on_progress:
            on_progress(10, "Burning captions into video...")

        result = burnin_subtitles(
            video_path, tmp_ass.name,
            output_path=output_path,
            on_progress=on_progress,
        )
        return result
    finally:
        if os.path.exists(tmp_ass.name):
            os.unlink(tmp_ass.name)


def _write_ass_file(f, segments: List[Dict], style: Dict, info: Dict):
    """Write an ASS subtitle file with styled events."""
    w, h = info["width"], info["height"]
    font = style.get("fontname", "Arial")
    size = style.get("fontsize", 48)
    pc = style.get("primary_color", "&H00FFFFFF")
    oc = style.get("outline_color", "&H00000000")
    bc = style.get("back_color", "&H00000000")
    outline = style.get("outline", 3)
    shadow = style.get("shadow", 1)
    align = style.get("alignment", 2)
    margin_v = style.get("margin_v", 40)
    borderstyle = style.get("borderstyle", 1)

    f.write("[Script Info]\n")
    f.write("Title: OpenCut Burn-in\n")
    f.write("ScriptType: v4.00+\n")
    f.write(f"PlayResX: {w}\n")
    f.write(f"PlayResY: {h}\n")
    f.write("ScaledBorderAndShadow: yes\n\n")

    f.write("[V4+ Styles]\n")
    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    f.write(f"Style: Default,{font},{size},{pc},{pc},{oc},{bc},-1,0,0,0,100,100,0,0,{borderstyle},{outline},{shadow},{align},20,20,{margin_v},1\n\n")

    f.write("[Events]\n")
    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    for seg in segments:
        start = _format_ass_time(seg.get("start", 0))
        end = _format_ass_time(seg.get("end", 0))
        text = seg.get("text", "").strip().replace("\n", "\\N")
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Available styles
# ---------------------------------------------------------------------------
def get_burnin_styles() -> List[Dict]:
    """Return available burn-in styles."""
    return [
        {"name": k, "label": v["label"]}
        for k, v in BURNIN_STYLES.items()
    ]
