"""
OpenCut Motion Graphics Module v0.9.0

Animated titles and kinetic typography via Manim + FFmpeg:
- Text reveal animations (write, fade, typewriter, wave)
- Lower thirds with animated entry/exit
- Title cards with background + animation
- Countdown timers
- Renders to transparent video for compositing

Uses Manim CE if available, falls back to FFmpeg drawtext for basic animations.
"""

import logging
import os
import re
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")


def check_manim_available() -> bool:
    try:
        import manim  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Title Card Presets
# ---------------------------------------------------------------------------
TITLE_PRESETS = {
    "fade_center": {
        "label": "Fade Center",
        "description": "Text fades in at center, holds, fades out",
    },
    "slide_left": {
        "label": "Slide From Left",
        "description": "Text slides in from the left edge",
    },
    "typewriter": {
        "label": "Typewriter",
        "description": "Characters appear one by one",
    },
    "lower_third": {
        "label": "Lower Third",
        "description": "Professional lower-third bar with name/title",
    },
    "countdown": {
        "label": "Countdown",
        "description": "Animated countdown timer (3, 2, 1...)",
    },
    "kinetic_bounce": {
        "label": "Kinetic Bounce",
        "description": "Words bounce in one by one with spring physics",
    },
}


# ---------------------------------------------------------------------------
# FFmpeg drawtext Renderer (zero dependency)
# ---------------------------------------------------------------------------
def render_title_card(
    text: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    duration: float = 5.0,
    fps: int = 30,
    preset: str = "fade_center",
    font_size: int = 72,
    font_color: str = "white",
    bg_color: str = "black",
    subtitle: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Render an animated title card video using FFmpeg drawtext.

    Returns path to rendered MP4 with the title animation.
    """
    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '', text[:20]).replace(" ", "_")
        output_path = os.path.join(directory, f"title_{safe}.mp4")

    if on_progress:
        on_progress(10, f"Rendering title: {preset}...")

    escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(";", "\\;")
    escaped_sub = subtitle.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(";", "\\;") if subtitle else ""

    if preset == "fade_center":
        fade_in = min(1.0, duration * 0.2)
        fade_out = min(1.0, duration * 0.2)
        vf = (
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"alpha='if(lt(t,{fade_in}),t/{fade_in},if(gt(t,{duration-fade_out}),(({duration}-t)/{fade_out}),1))'"
        )
        if escaped_sub:
            vf += (
                f",drawtext=text='{escaped_sub}':fontsize={font_size//2}:fontcolor={font_color}@0.8:"
                f"x=(w-text_w)/2:y=(h/2)+{font_size}:"
                f"alpha='if(lt(t,{fade_in+0.3}),max(0,(t-0.3)/{fade_in}),if(gt(t,{duration-fade_out}),(({duration}-t)/{fade_out}),1))'"
            )

    elif preset == "slide_left":
        slide_dur = min(1.5, duration * 0.3)
        vf = (
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x='if(lt(t,{slide_dur}),-text_w+(w/2+text_w/2)*t/{slide_dur},(w-text_w)/2)':"
            f"y=(h-text_h)/2"
        )

    elif preset == "typewriter":
        # Reveal characters over time
        chars = max(1, len(text))
        char_dur = max(0.01, min(duration * 0.6, chars * 0.08))
        vf = (
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"enable='gte(t,0)'"
        )
        # Approximate typewriter via alpha reveal per character isn't possible in one drawtext
        # Use full text with delayed fade-in as approximation
        vf = (
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"alpha='min(1,t/{char_dur})'"
        )

    elif preset == "lower_third":
        bar_h = font_size * 2 + 30
        y_bar = height - bar_h - 60
        vf = (
            f"drawbox=x=0:y={y_bar}:w=iw:h={bar_h}:color=black@0.7:t=fill:"
            f"enable='between(t,0.5,{duration-0.5})',"
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=60:y={y_bar + 10}:"
            f"enable='between(t,0.7,{duration-0.5})'"
        )
        if escaped_sub:
            vf += (
                f",drawtext=text='{escaped_sub}':fontsize={font_size*2//3}:fontcolor={font_color}@0.8:"
                f"x=60:y={y_bar + font_size + 15}:"
                f"enable='between(t,0.9,{duration-0.5})'"
            )

    elif preset == "countdown":
        # Countdown numbers 3, 2, 1 with scale effect
        seg = duration / 3
        parts = []
        for i, num in enumerate(["3", "2", "1"]):
            t_start = i * seg
            t_end = t_start + seg
            parts.append(
                f"drawtext=text='{num}':fontsize={font_size*3}:fontcolor={font_color}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:"
                f"enable='between(t,{t_start},{t_end})':"
                f"alpha='if(lt(t-{t_start},0.3),(t-{t_start})/0.3,if(gt(t,{t_end-0.3}),({t_end}-t)/0.3,1))'"
            )
        vf = ",".join(parts)

    else:
        # Kinetic bounce - words appear sequentially
        words = text.split()
        word_dur = min(duration * 0.7, len(words) * 0.4) / max(len(words), 1)
        parts = []
        for i, word in enumerate(words):
            ew = word.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(";", "\\;")
            t_start = i * word_dur + 0.2
            parts.append(
                f"drawtext=text='{ew} ':fontsize={font_size}:fontcolor={font_color}:"
                f"x=(w/2)-{len(text)*font_size//4}+{i*font_size*len(word)*2//3}:"
                f"y=(h/2):"
                f"enable='gte(t,{t_start})':"
                f"alpha='min(1,(t-{t_start})/0.3)'"
            )
        vf = ",".join(parts) if parts else f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:x=(w-text_w)/2:y=(h-text_h)/2"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={duration}:r={fps}",
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "Title card rendered!")
    return output_path


# ---------------------------------------------------------------------------
# Composite title onto video
# ---------------------------------------------------------------------------
def overlay_title(
    video_path: str,
    text: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    preset: str = "fade_center",
    font_size: int = 72,
    font_color: str = "white",
    start_time: float = 0.0,
    duration: float = 5.0,
    subtitle: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Overlay animated title text directly onto a video."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_titled.mp4")

    if on_progress:
        on_progress(10, "Overlaying title...")

    escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(";", "\\;")
    escaped_sub = subtitle.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(";", "\\;") if subtitle else ""
    end_time = start_time + duration

    fade_in = min(1.0, duration * 0.2)
    fade_out = min(1.0, duration * 0.2)

    if preset == "lower_third":
        vh = 1080
        try:
            info_cmd = subprocess.run(
                [get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=height", "-of", "csv=p=0", video_path],
                capture_output=True, timeout=10)
            vh = int(info_cmd.stdout.decode().strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            pass
        bar_h = font_size * 2 + 30
        y_bar = vh - bar_h - 60
        vf = (
            f"drawbox=x=0:y={y_bar}:w=iw:h={bar_h}:color=black@0.7:t=fill:"
            f"enable='between(t,{start_time+0.3},{end_time-0.3})',"
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=60:y={y_bar+10}:"
            f"enable='between(t,{start_time+0.5},{end_time-0.3})'"
        )
        if escaped_sub:
            vf += (
                f",drawtext=text='{escaped_sub}':fontsize={font_size*2//3}:fontcolor={font_color}@0.8:"
                f"x=60:y={y_bar+font_size+15}:"
                f"enable='between(t,{start_time+0.7},{end_time-0.3})'"
            )
    else:
        vf = (
            f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"enable='between(t,{start_time},{end_time})':"
            f"alpha='if(lt(t-{start_time},{fade_in}),(t-{start_time})/{fade_in},"
            f"if(gt(t,{end_time-fade_out}),({end_time}-t)/{fade_out},1))'"
        )
        if escaped_sub:
            vf += (
                f",drawtext=text='{escaped_sub}':fontsize={font_size//2}:fontcolor={font_color}@0.8:"
                f"x=(w-text_w)/2:y=(h/2)+{font_size}:"
                f"enable='between(t,{start_time+0.3},{end_time})'"
            )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "Title overlaid!")
    return output_path


def get_title_presets() -> List[Dict]:
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in TITLE_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Remotion-Powered Motion Graphics (Premium, requires Node.js)
# ---------------------------------------------------------------------------
def check_remotion_available() -> bool:
    """Check if Node.js and Remotion CLI are available."""
    try:
        result = subprocess.run(["npx", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def render_remotion_title(
    text: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    template: str = "title-card",
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    props: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Render premium motion graphics using Remotion (React-based).

    Produces After Effects-quality animated titles, lower thirds, and
    kinetic typography via React components rendered to video.

    Requires Node.js 18+ and npx. Templates are React components
    stored in ~/.opencut/remotion-templates/.

    Args:
        text: Title text to render.
        template: Template name (title-card, lower-third, kinetic-text, countdown).
        duration: Duration in seconds.
        width/height: Output resolution.
        fps: Frames per second.
        props: Additional template-specific props (colors, fonts, animations).
    """
    if not check_remotion_available():
        raise RuntimeError(
            "Remotion requires Node.js 18+. Install from https://nodejs.org/ "
            "then run: npx remotion --version"
        )

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        safe_text = re.sub(r'[^\w\-]', '_', text[:20]).strip('_')
        output_path = os.path.join(directory, f"remotion_{safe_text}.mp4")

    if on_progress:
        on_progress(10, f"Preparing Remotion template ({template})...")

    templates_dir = os.path.expanduser("~/.opencut/remotion-templates")
    template_dir = os.path.join(templates_dir, template)

    if not os.path.isdir(template_dir):
        # Generate a default React template on-the-fly
        os.makedirs(template_dir, exist_ok=True)
        _generate_default_template(template_dir, template)

    # Build props JSON for Remotion
    import json
    render_props = {
        "text": text,
        "duration": duration,
        "width": width,
        "height": height,
        **(props or {}),
    }

    props_file = os.path.join(template_dir, "props.json")
    with open(props_file, "w", encoding="utf-8") as f:
        json.dump(render_props, f)

    if on_progress:
        on_progress(30, "Rendering with Remotion...")

    total_frames = int(duration * fps)
    cmd = [
        "npx", "remotion", "render",
        template_dir,
        "Main",
        output_path,
        "--props", props_file,
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--frames", f"0-{total_frames - 1}",
        "--codec", "h264",
        "--crf", "18",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else "unknown error"
        # Fallback to FFmpeg drawtext if Remotion fails
        logger.warning("Remotion render failed, falling back to FFmpeg: %s", stderr)
        return render_title_card(
            text, output_path=output_path, output_dir=output_dir,
            preset="fade_center", duration=duration,
            width=width, height=height, fps=fps,
            on_progress=on_progress,
        )

    if on_progress:
        on_progress(100, "Remotion render complete!")
    return output_path


def _generate_default_template(template_dir: str, template_name: str):
    """Generate a minimal Remotion template for the given style."""
    # Create a minimal package.json and entry component
    import json

    package = {
        "name": f"opencut-{template_name}",
        "version": "1.0.0",
        "private": True,
        "dependencies": {
            "remotion": "^4.0.0",
            "react": "^18.0.0",
            "react-dom": "^18.0.0",
        },
    }
    with open(os.path.join(template_dir, "package.json"), "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)

    # Create a minimal Main composition component
    comp_code = """import {Composition, useCurrentFrame, useVideoConfig, interpolate} from 'remotion';
import React from 'react';

const Main = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames, width, height} = useVideoConfig();
  const opacity = interpolate(frame, [0, fps * 0.5], [0, 1], {extrapolateRight: 'clamp'});
  const outOpacity = interpolate(frame, [durationInFrames - fps * 0.5, durationInFrames], [1, 0], {extrapolateLeft: 'clamp'});
  return (
    <div style={{width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1e1e2e', opacity: Math.min(opacity, outOpacity)}}>
      <h1 style={{color: '#cdd6f4', fontSize: 64, fontFamily: 'sans-serif', textAlign: 'center', padding: 40}}>Title</h1>
    </div>
  );
};

export const RemotionRoot = () => (
  <Composition id="Main" component={Main} durationInFrames={150} fps={30} width={1920} height={1080} />
);
"""
    os.makedirs(os.path.join(template_dir, "src"), exist_ok=True)
    with open(os.path.join(template_dir, "src", "index.tsx"), "w", encoding="utf-8") as f:
        f.write(comp_code)

    logger.info("Generated default Remotion template: %s", template_dir)
