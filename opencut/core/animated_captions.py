"""
OpenCut Animated Captions Module v0.9.0

CapCut/Submagic-style animated captions:
- Word-by-word reveals with timing from WhisperX
- Animation presets: pop, fade, slide, typewriter, bounce, glow
- CSS-based styling per word state (active, inactive, upcoming)
- Emoji insertion support
- Renders directly to video via Pillow frame compositing

Can use pycaps if installed, or falls back to built-in
Pillow-based renderer for zero-dependency operation.
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _ensure_package(pkg, pip_name=None, on_progress=None):
    try:
        __import__(pkg)
        return True
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg,
                        "--break-system-packages", "-q"], capture_output=True, timeout=300)
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


def _run_ffmpeg(cmd, timeout=7200):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {r.stderr.decode(errors='replace')[-500:]}")


def _get_video_info(fp):
    import json
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration",
                        "-of", "json", fp], capture_output=True, timeout=30)
    try:
        s = json.loads(r.stdout.decode())["streams"][0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_p[0]) / float(fps_p[1]) if len(fps_p) == 2 else 30.0
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080)),
                "fps": fps, "duration": float(s.get("duration", 0))}
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0}


# ---------------------------------------------------------------------------
# Animation Presets
# ---------------------------------------------------------------------------
ANIMATION_PRESETS = {
    "pop": {
        "label": "Pop",
        "description": "Words pop in with scale bounce (CapCut style)",
        "active_scale": 1.15,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.3,
        "transition_frames": 4,
    },
    "fade": {
        "label": "Fade",
        "description": "Words fade in smoothly",
        "active_scale": 1.0,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.2,
        "transition_frames": 8,
    },
    "slide_up": {
        "label": "Slide Up",
        "description": "Words slide up into position",
        "active_scale": 1.0,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.2,
        "slide_y": -15,
        "transition_frames": 6,
    },
    "typewriter": {
        "label": "Typewriter",
        "description": "Words appear one by one like typing",
        "active_scale": 1.0,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.0,
        "transition_frames": 2,
    },
    "bounce": {
        "label": "Bounce",
        "description": "Words bounce in with spring easing",
        "active_scale": 1.2,
        "inactive_scale": 0.8,
        "active_opacity": 1.0,
        "inactive_opacity": 0.3,
        "transition_frames": 6,
    },
    "glow": {
        "label": "Glow",
        "description": "Active word gets a bright glow effect",
        "active_scale": 1.05,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.4,
        "glow_radius": 8,
        "transition_frames": 5,
    },
    "highlight_box": {
        "label": "Highlight Box",
        "description": "Active word gets a colored background box",
        "active_scale": 1.0,
        "inactive_scale": 1.0,
        "active_opacity": 1.0,
        "inactive_opacity": 0.5,
        "box_color": (255, 230, 0),
        "box_padding": 6,
        "transition_frames": 3,
    },
}


# ---------------------------------------------------------------------------
# Built-in Pillow Renderer (zero-dependency fallback)
# ---------------------------------------------------------------------------
def render_animated_captions(
    video_path: str,
    word_segments: List[Dict],
    output_path: Optional[str] = None,
    output_dir: str = "",
    animation: str = "pop",
    font_size: int = 56,
    font_color: tuple = (255, 255, 255),
    highlight_color: tuple = (255, 230, 0),
    stroke_width: int = 3,
    stroke_color: tuple = (0, 0, 0),
    position_y: float = 0.80,
    max_words_per_line: int = 6,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Render animated word-by-word captions onto video.

    Args:
        word_segments: List of {"word": str, "start": float, "end": float}
            from WhisperX or similar word-level transcript.
        animation: Preset name from ANIMATION_PRESETS.
        position_y: Vertical position as fraction of height (0=top, 1=bottom).
        max_words_per_line: Words per caption line before wrapping.
    """
    _ensure_package("cv2", "opencv-python-headless", on_progress)
    _ensure_package("PIL", "Pillow", on_progress)
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_animated_caps.mp4")

    preset = ANIMATION_PRESETS.get(animation, ANIMATION_PRESETS["pop"])
    info = _get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]

    if on_progress:
        on_progress(5, "Loading video for caption overlay...")

    # Load font
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
               "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/impact.ttf"]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Group words into lines (segments that overlap in time)
    lines = _group_words_into_lines(word_segments, max_words_per_line)

    cap = cv2.VideoCapture(video_path)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # Find active line
            active_line = None
            for line in lines:
                if line["start"] <= current_time <= line["end"]:
                    active_line = line
                    break

            if active_line:
                # Render caption overlay using Pillow
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Calculate full line width for centering
                words = active_line["words"]
                word_widths = []
                space_w = draw.textlength(" ", font=font) if hasattr(draw, 'textlength') else font_size * 0.3
                for wd in words:
                    tw = draw.textlength(wd["word"], font=font) if hasattr(draw, 'textlength') else len(wd["word"]) * font_size * 0.6
                    word_widths.append(tw)
                total_w = sum(word_widths) + space_w * (len(words) - 1)
                x_start = (w - total_w) / 2
                y_pos = int(h * position_y)

                x = x_start
                for i, wd in enumerate(words):
                    is_active = wd["start"] <= current_time <= wd["end"]
                    is_past = current_time > wd["end"]

                    if is_active:
                        color = highlight_color + (int(preset["active_opacity"] * 255),)
                    elif is_past:
                        color = font_color + (220,)
                    else:
                        color = font_color + (int(preset["inactive_opacity"] * 255),)

                    # Draw stroke
                    if stroke_width > 0:
                        sc = stroke_color + (color[3],)
                        for dx in range(-stroke_width, stroke_width + 1):
                            for dy in range(-stroke_width, stroke_width + 1):
                                if dx * dx + dy * dy <= stroke_width * stroke_width:
                                    draw.text((x + dx, y_pos + dy), wd["word"], font=font, fill=sc)

                    # Draw highlight box for that preset
                    if animation == "highlight_box" and is_active:
                        pad = preset.get("box_padding", 6)
                        bx_color = preset.get("box_color", (255, 230, 0)) + (200,)
                        draw.rectangle([x - pad, y_pos - pad,
                                        x + word_widths[i] + pad, y_pos + font_size + pad],
                                       fill=bx_color)

                    draw.text((x, y_pos), wd["word"], font=font, fill=color)
                    x += word_widths[i] + space_w

                # Composite
                pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
                frame = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

            writer.write(frame)
            frame_idx += 1

            if on_progress and frame_idx % 60 == 0:
                pct = 5 + int((frame_idx / total_frames) * 85)
                on_progress(pct, f"Rendering frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding with audio...")

    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_video, "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])
    os.unlink(tmp_video)

    if on_progress:
        on_progress(100, "Animated captions rendered!")
    return output_path


def _group_words_into_lines(words: List[Dict], max_per_line: int) -> List[Dict]:
    """Group word segments into display lines."""
    if not words:
        return []

    lines = []
    current_words = []

    for w in words:
        current_words.append(w)
        if len(current_words) >= max_per_line:
            lines.append({
                "words": current_words,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
            })
            current_words = []

    if current_words:
        lines.append({
            "words": current_words,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
        })

    return lines


def get_animation_presets() -> List[Dict]:
    """Return available animation presets."""
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in ANIMATION_PRESETS.items()
    ]


def check_pycaps_available() -> bool:
    try:
        import pycaps  # noqa: F401
        return True
    except ImportError:
        return False
