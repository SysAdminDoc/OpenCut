"""
Template-Based Video Assembly (15.3)

Pre-built video templates with placeholder zones. Users drop media
into zones and assemble the final video.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TemplatePlaceholder:
    """A placeholder zone in a video template."""
    name: str
    zone_type: str = "video"  # "video", "image", "text", "audio"
    start: float = 0.0
    end: float = 0.0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    required: bool = True
    default_text: str = ""


@dataclass
class VideoTemplate:
    """A video template with placeholder zones."""
    name: str
    description: str = ""
    duration: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    placeholders: List[TemplatePlaceholder] = field(default_factory=list)
    background_color: str = "black"
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilledTemplate:
    """A template with media assigned to placeholders."""
    template: VideoTemplate
    assignments: Dict[str, str] = field(default_factory=dict)  # placeholder_name -> file_path
    text_overrides: Dict[str, str] = field(default_factory=dict)  # placeholder_name -> text


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------
_BUILTIN_TEMPLATES: Dict[str, VideoTemplate] = {
    "intro_outro": VideoTemplate(
        name="intro_outro",
        description="Intro title card + main content + outro card",
        duration=0,
        width=1920, height=1080,
        placeholders=[
            TemplatePlaceholder(
                name="intro_text", zone_type="text",
                start=0.0, end=3.0,
                x=0, y=0, width=1920, height=1080,
                default_text="Title Here",
            ),
            TemplatePlaceholder(
                name="main_content", zone_type="video",
                start=3.0, end=0.0,
                x=0, y=0, width=1920, height=1080,
                required=True,
            ),
            TemplatePlaceholder(
                name="outro_text", zone_type="text",
                start=0.0, end=3.0,
                x=0, y=0, width=1920, height=1080,
                default_text="Thanks for watching",
            ),
        ],
        category="general",
        tags=["intro", "outro", "title"],
    ),
    "split_screen": VideoTemplate(
        name="split_screen",
        description="Side-by-side split screen layout",
        duration=0,
        width=1920, height=1080,
        placeholders=[
            TemplatePlaceholder(
                name="left_video", zone_type="video",
                start=0.0, end=0.0,
                x=0, y=0, width=960, height=1080,
                required=True,
            ),
            TemplatePlaceholder(
                name="right_video", zone_type="video",
                start=0.0, end=0.0,
                x=960, y=0, width=960, height=1080,
                required=True,
            ),
        ],
        category="comparison",
        tags=["split", "comparison", "side-by-side"],
    ),
    "picture_in_picture": VideoTemplate(
        name="picture_in_picture",
        description="Main video with small overlay in corner",
        duration=0,
        width=1920, height=1080,
        placeholders=[
            TemplatePlaceholder(
                name="main_video", zone_type="video",
                start=0.0, end=0.0,
                x=0, y=0, width=1920, height=1080,
                required=True,
            ),
            TemplatePlaceholder(
                name="pip_video", zone_type="video",
                start=0.0, end=0.0,
                x=1440, y=30, width=450, height=253,
                required=False,
            ),
        ],
        category="overlay",
        tags=["pip", "overlay", "webcam"],
    ),
    "social_vertical": VideoTemplate(
        name="social_vertical",
        description="9:16 vertical video with top/bottom text zones",
        duration=0,
        width=1080, height=1920,
        placeholders=[
            TemplatePlaceholder(
                name="top_text", zone_type="text",
                start=0.0, end=0.0,
                x=0, y=0, width=1080, height=200,
                default_text="Caption here",
            ),
            TemplatePlaceholder(
                name="main_video", zone_type="video",
                start=0.0, end=0.0,
                x=0, y=200, width=1080, height=1520,
                required=True,
            ),
            TemplatePlaceholder(
                name="bottom_text", zone_type="text",
                start=0.0, end=0.0,
                x=0, y=1720, width=1080, height=200,
                default_text="@handle",
            ),
        ],
        category="social",
        tags=["vertical", "tiktok", "reels", "shorts"],
    ),
    "tutorial": VideoTemplate(
        name="tutorial",
        description="Screen recording with webcam overlay and title bar",
        duration=0,
        width=1920, height=1080,
        placeholders=[
            TemplatePlaceholder(
                name="screen_recording", zone_type="video",
                start=0.0, end=0.0,
                x=0, y=60, width=1920, height=1020,
                required=True,
            ),
            TemplatePlaceholder(
                name="webcam", zone_type="video",
                start=0.0, end=0.0,
                x=1500, y=700, width=400, height=300,
                required=False,
            ),
            TemplatePlaceholder(
                name="title_bar", zone_type="text",
                start=0.0, end=0.0,
                x=0, y=0, width=1920, height=60,
                default_text="Tutorial Title",
            ),
        ],
        category="education",
        tags=["tutorial", "screencast", "education"],
    ),
}


# ---------------------------------------------------------------------------
# List / Load Templates
# ---------------------------------------------------------------------------
def list_templates(
    category: str = "",
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """List available video templates.

    Args:
        category: Filter by category (empty for all).
        on_progress: Optional callback(pct, msg).

    Returns:
        List of template summaries.
    """
    if on_progress:
        on_progress(50, "Loading templates...")

    results = []
    for name, tmpl in _BUILTIN_TEMPLATES.items():
        if category and tmpl.category != category:
            continue
        results.append({
            "name": tmpl.name,
            "description": tmpl.description,
            "category": tmpl.category,
            "tags": tmpl.tags,
            "width": tmpl.width,
            "height": tmpl.height,
            "placeholder_count": len(tmpl.placeholders),
            "placeholders": [
                {"name": p.name, "type": p.zone_type, "required": p.required}
                for p in tmpl.placeholders
            ],
        })

    if on_progress:
        on_progress(100, f"Found {len(results)} templates")

    return results


def load_template(
    template_name: str,
    on_progress: Optional[Callable] = None,
) -> VideoTemplate:
    """Load a video template by name.

    Args:
        template_name: Name of the template to load.
        on_progress: Optional callback(pct, msg).

    Returns:
        VideoTemplate object.
    """
    tmpl = _BUILTIN_TEMPLATES.get(template_name)
    if not tmpl:
        available = sorted(_BUILTIN_TEMPLATES.keys())
        raise ValueError(
            f"Template not found: {template_name}. "
            f"Available: {available}"
        )

    if on_progress:
        on_progress(100, f"Loaded template: {template_name}")

    return tmpl


# ---------------------------------------------------------------------------
# Fill Template
# ---------------------------------------------------------------------------
def fill_template(
    template: VideoTemplate,
    media_assignments: Dict[str, str],
    text_overrides: Optional[Dict[str, str]] = None,
    on_progress: Optional[Callable] = None,
) -> FilledTemplate:
    """Assign media files to template placeholders.

    Args:
        template: VideoTemplate to fill.
        media_assignments: Map of placeholder_name -> file_path.
        text_overrides: Optional map of text placeholder_name -> text.
        on_progress: Optional callback(pct, msg).

    Returns:
        FilledTemplate with assignments.
    """
    if on_progress:
        on_progress(10, "Validating assignments...")

    text_overrides = text_overrides or {}

    # Validate required placeholders
    for ph in template.placeholders:
        if ph.required and ph.zone_type in ("video", "image", "audio"):
            if ph.name not in media_assignments:
                raise ValueError(f"Required placeholder not filled: {ph.name}")

    # Validate media files exist
    for name, path in media_assignments.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Media file not found for {name}: {path}")
        # Verify placeholder exists
        valid_names = {p.name for p in template.placeholders}
        if name not in valid_names:
            raise ValueError(
                f"Unknown placeholder: {name}. "
                f"Available: {sorted(valid_names)}"
            )

    filled = FilledTemplate(
        template=template,
        assignments=dict(media_assignments),
        text_overrides=dict(text_overrides),
    )

    if on_progress:
        on_progress(100, f"Template filled: {len(media_assignments)} assignments")

    return filled


# ---------------------------------------------------------------------------
# Assemble from Template
# ---------------------------------------------------------------------------
def assemble_from_template(
    filled_template: FilledTemplate,
    output_path_str: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Assemble a video from a filled template.

    Args:
        filled_template: FilledTemplate with media assignments.
        output_path_str: Output path (auto-generated if empty).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, duration, template_name.
    """
    tmpl = filled_template.template

    if on_progress:
        on_progress(5, f"Assembling template: {tmpl.name}...")

    # Determine output path
    first_video = None
    for ph in tmpl.placeholders:
        if ph.zone_type == "video" and ph.name in filled_template.assignments:
            first_video = filled_template.assignments[ph.name]
            break

    if not first_video:
        raise ValueError("No video assigned to any placeholder")

    if not output_path_str:
        out_dir = os.path.dirname(os.path.abspath(first_video))
        output_path_str = output_path(first_video, f"template_{tmpl.name}", out_dir)

    # Get info about assigned videos
    video_assignments = {}
    for ph in tmpl.placeholders:
        if ph.zone_type == "video" and ph.name in filled_template.assignments:
            path = filled_template.assignments[ph.name]
            video_assignments[ph.name] = {
                "path": path,
                "info": get_video_info(path),
                "placeholder": ph,
            }

    if on_progress:
        on_progress(20, "Building filter graph...")

    # Build FFmpeg command based on template type
    if tmpl.name == "split_screen":
        result = _assemble_split_screen(video_assignments, tmpl, output_path_str, on_progress)
    elif tmpl.name == "picture_in_picture":
        result = _assemble_pip(video_assignments, tmpl, output_path_str, on_progress)
    else:
        # Generic sequential assembly
        result = _assemble_sequential(
            video_assignments, filled_template, tmpl, output_path_str, on_progress,
        )

    return result


def _assemble_split_screen(assignments, tmpl, output_path_str, on_progress):
    """Assemble a split-screen layout."""
    left = assignments.get("left_video", {})
    right = assignments.get("right_video", {})

    if not left or not right:
        raise ValueError("Split screen requires both left_video and right_video")

    hw = tmpl.width // 2
    fc = (
        f"[0:v]scale={hw}:{tmpl.height}:force_original_aspect_ratio=decrease,"
        f"pad={hw}:{tmpl.height}:(ow-iw)/2:(oh-ih)/2[left];"
        f"[1:v]scale={hw}:{tmpl.height}:force_original_aspect_ratio=decrease,"
        f"pad={hw}:{tmpl.height}:(ow-iw)/2:(oh-ih)/2[right];"
        f"[left][right]hstack=inputs=2[outv];"
        f"[0:a][1:a]amix=inputs=2:duration=longest[outa]"
    )

    cmd = (FFmpegCmd()
           .input(left["path"])
           .input(right["path"])
           .filter_complex(fc, maps=["[outv]", "[outa]"])
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .output(output_path_str)
           .build())

    if on_progress:
        on_progress(50, "Encoding split screen...")

    run_ffmpeg(cmd)

    duration = max(
        left["info"]["duration"],
        right["info"]["duration"],
    )

    if on_progress:
        on_progress(100, "Split screen complete")

    return {
        "output_path": output_path_str,
        "duration": duration,
        "template_name": tmpl.name,
    }


def _assemble_pip(assignments, tmpl, output_path_str, on_progress):
    """Assemble a picture-in-picture layout."""
    main = assignments.get("main_video", {})
    if not main:
        raise ValueError("PiP requires main_video")

    pip = assignments.get("pip_video")
    if not pip:
        # No PiP, just copy main
        cmd = (FFmpegCmd()
               .input(main["path"])
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "duration": main["info"]["duration"],
            "template_name": tmpl.name,
        }

    pip_ph = [p for p in tmpl.placeholders if p.name == "pip_video"][0]
    fc = (
        f"[1:v]scale={pip_ph.width}:{pip_ph.height}:force_original_aspect_ratio=decrease,"
        f"pad={pip_ph.width}:{pip_ph.height}:(ow-iw)/2:(oh-ih)/2[pip];"
        f"[0:v][pip]overlay={pip_ph.x}:{pip_ph.y}:shortest=1[outv]"
    )

    cmd = (FFmpegCmd()
           .input(main["path"])
           .input(pip["path"])
           .filter_complex(fc, maps=["[outv]", "0:a"])
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .output(output_path_str)
           .build())

    if on_progress:
        on_progress(50, "Encoding PiP layout...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "PiP assembly complete")

    return {
        "output_path": output_path_str,
        "duration": main["info"]["duration"],
        "template_name": tmpl.name,
    }


def _assemble_sequential(assignments, filled_template, tmpl, output_path_str, on_progress):
    """Generic sequential assembly for templates like intro_outro."""
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="opencut_template_")
    segment_files = []
    total_dur = 0.0

    try:
        idx = 0
        for ph in tmpl.placeholders:
            if ph.zone_type == "text":
                # Generate text card
                text = filled_template.text_overrides.get(ph.name, ph.default_text)
                dur = ph.end - ph.start if ph.end > ph.start else 3.0
                seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
                _generate_text_card(
                    text, tmpl.width, tmpl.height, dur,
                    seg_path, tmpl.background_color,
                )
                segment_files.append(seg_path)
                total_dur += dur
                idx += 1

            elif ph.zone_type == "video" and ph.name in filled_template.assignments:
                src = filled_template.assignments[ph.name]
                seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
                info = get_video_info(src)

                # Re-encode to match template dimensions
                vf = (
                    f"scale={tmpl.width}:{tmpl.height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={tmpl.width}:{tmpl.height}:(ow-iw)/2:(oh-ih)/2:black"
                )
                cmd = (FFmpegCmd()
                       .input(src)
                       .video_filter(vf)
                       .video_codec("libx264", crf=18, preset="fast")
                       .audio_codec("aac", bitrate="192k")
                       .output(seg_path)
                       .build())
                run_ffmpeg(cmd)
                segment_files.append(seg_path)
                total_dur += info["duration"]
                idx += 1

        if not segment_files:
            raise ValueError("No segments to assemble")

        if on_progress:
            on_progress(70, "Concatenating segments...")

        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                safe = seg.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_path)
               .copy_streams()
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd)

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, "Template assembly complete")

    return {
        "output_path": output_path_str,
        "duration": total_dur,
        "template_name": tmpl.name,
    }


def _generate_text_card(
    text: str, width: int, height: int, duration: float,
    output_path_str: str, bg_color: str = "black",
):
    """Generate a video card with text using FFmpeg."""
    # Escape text for FFmpeg drawtext
    safe_text = text.replace("'", "\\'").replace(":", "\\:")

    vf = (
        f"drawtext=text='{safe_text}':"
        f"fontsize=48:fontcolor=white:"
        f"x=(w-text_w)/2:y=(h-text_h)/2"
    )

    cmd = (FFmpegCmd()
           .option("f", "lavfi")
           .input(f"color=c={bg_color}:s={width}x{height}:d={duration}:r=30")
           .option("f", "lavfi")
           .input(f"anullsrc=r=48000:cl=stereo:d={duration}")
           .video_filter(vf)
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .option("shortest")
           .output(output_path_str)
           .build())
    run_ffmpeg(cmd)
