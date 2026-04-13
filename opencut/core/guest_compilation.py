"""
OpenCut Guest Message Compilation Module (48.2)

Auto-trim silence, normalize audio, generate name lower-thirds
from filenames, and concatenate with transitions.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class ProcessedMessage:
    """Result of processing a single guest message."""
    input_path: str = ""
    output_path: str = ""
    guest_name: str = ""
    original_duration: float = 0.0
    trimmed_duration: float = 0.0
    silence_removed: float = 0.0
    audio_normalized: bool = False


@dataclass
class NameCardStyle:
    """Styling for name lower-third cards."""
    font_family: str = "Arial"
    font_size: int = 42
    font_color: str = "white"
    bg_color: str = "black@0.6"
    position: str = "bottom_left"  # bottom_left, bottom_center, center
    margin: int = 40
    padding: int = 16
    display_duration: float = 4.0
    fade_in: float = 0.5
    fade_out: float = 0.5


@dataclass
class CompilationResult:
    """Result of compiling all guest messages."""
    output_path: str = ""
    total_duration: float = 0.0
    message_count: int = 0
    messages: List[ProcessedMessage] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def _extract_name_from_filename(filename: str) -> str:
    """Extract a display name from a filename.

    Handles patterns like:
    - "John_Smith.mp4" -> "John Smith"
    - "john-smith-message.mp4" -> "John Smith"
    - "03_Jane_Doe_congrats.mov" -> "Jane Doe"
    - "message_from_Bob.mp4" -> "Bob"
    """
    base = os.path.splitext(filename)[0]

    # Remove common prefixes/suffixes
    for pattern in ["message_from_", "msg_from_", "from_", "message_", "msg_"]:
        if base.lower().startswith(pattern):
            base = base[len(pattern):]
    for pattern in ["_message", "_msg", "_congrats", "_greeting",
                    "_wishes", "_video", "_clip"]:
        if base.lower().endswith(pattern):
            base = base[:len(base) - len(pattern)]

    # Remove leading numbers/underscores (e.g., "03_")
    base = re.sub(r"^\d+[_\-\s]*", "", base)

    # Replace separators with spaces
    name = re.sub(r"[_\-]+", " ", base).strip()

    # Title case
    name = name.title()

    return name if name else "Guest"


def _detect_silence_boundaries(video_path: str, threshold_db: float = -35.0,
                                min_silence_sec: float = 0.5) -> tuple:
    """Detect leading/trailing silence and return content boundaries.

    Returns (start_time, end_time) of the non-silent portion.
    """
    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0:
        return 0.0, 0.0

    # Use FFmpeg silencedetect to find silence regions
    cmd = [
        get_ffmpeg_path(), "-i", video_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ]

    import subprocess as _sp
    result = _sp.run(cmd, capture_output=True, timeout=120)
    stderr = result.stderr.decode(errors="replace")

    # Parse silence start/end from stderr
    silence_regions = []
    silence_start = None
    for line in stderr.split("\n"):
        if "silence_start:" in line:
            try:
                silence_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                silence_start = None
        elif "silence_end:" in line and silence_start is not None:
            try:
                parts = line.split("silence_end:")[1].strip().split()
                silence_end = float(parts[0])
                silence_regions.append((silence_start, silence_end))
            except (ValueError, IndexError):
                pass
            silence_start = None

    if not silence_regions:
        return 0.0, duration

    # Find content start (after leading silence)
    content_start = 0.0
    if silence_regions[0][0] < 0.1:  # silence starts at beginning
        content_start = silence_regions[0][1]

    # Find content end (before trailing silence)
    content_end = duration
    if silence_regions[-1][1] >= duration - 0.1:  # silence ends at end
        content_end = silence_regions[-1][0]

    # Ensure valid range
    content_start = max(0.0, content_start - 0.2)  # small padding
    content_end = min(duration, content_end + 0.2)

    if content_end <= content_start:
        return 0.0, duration

    return round(content_start, 3), round(content_end, 3)


def generate_name_card(
    name: str,
    style: Optional[NameCardStyle] = None,
    width: int = 1920,
    height: int = 1080,
    output_path_str: Optional[str] = None,
    duration: float = 4.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a name lower-third card as a video segment.

    Args:
        name: Display name to show.
        style: NameCardStyle for customization.
        width: Video width.
        height: Video height.
        output_path_str: Output path. Auto-generated if None.
        duration: Duration of the name card in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, name, duration.
    """
    if not name:
        raise ValueError("Name must not be empty")

    if style is None:
        style = NameCardStyle()

    if output_path_str is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", prefix="namecard_",
                                           delete=False)
        output_path_str = tmp.name
        tmp.close()

    if on_progress:
        on_progress(10, f"Generating name card for {name}...")

    # Position calculations
    margin = style.margin
    padding = style.padding
    fontsize = style.font_size

    if style.position == "bottom_center":
        x_expr = "(w-tw)/2"
        y_expr = f"h-{margin}-th"
    elif style.position == "center":
        x_expr = "(w-tw)/2"
        y_expr = "(h-th)/2"
    else:  # bottom_left
        x_expr = str(margin)
        y_expr = f"h-{margin}-th"

    escaped_name = name.replace("'", "'\\''").replace(":", "\\:")

    # Build drawtext with background box and fade
    fade_in_expr = f"if(lt(t,{style.fade_in}),t/{style.fade_in},1)"
    fade_out_start = duration - style.fade_out
    fade_out_expr = f"if(gt(t,{fade_out_start}),(({duration}-t)/{style.fade_out}),1)"
    alpha_expr = f"min({fade_in_expr},{fade_out_expr})"

    drawtext = (
        f"drawtext=text='{escaped_name}'"
        f":fontsize={fontsize}"
        f":fontcolor={style.font_color}"
        f":box=1:boxcolor={style.bg_color}:boxborderw={padding}"
        f":x={x_expr}:y={y_expr}"
        f":alpha='{alpha_expr}'"
    )

    # Generate a black background with the name card
    cmd = (FFmpegCmd()
           .pre_input("f", "lavfi")
           .input(f"color=c=black:s={width}x{height}:d={duration}:r=30")
           .video_codec("libx264", crf=18, preset="fast")
           .video_filter(drawtext)
           .option("t", str(duration))
           .output(output_path_str)
           .build())

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Name card generated: {name}")

    return {
        "output_path": output_path_str,
        "name": name,
        "duration": duration,
    }


def process_single_message(
    video_path: str,
    output_path_str: Optional[str] = None,
    guest_name: Optional[str] = None,
    trim_silence: bool = True,
    normalize_audio: bool = True,
    add_name_card: bool = True,
    name_style: Optional[NameCardStyle] = None,
    target_width: int = 1920,
    target_height: int = 1080,
    on_progress: Optional[Callable] = None,
) -> ProcessedMessage:
    """Process a single guest message video.

    Args:
        video_path: Input video path.
        output_path_str: Output path. Auto-generated if None.
        guest_name: Display name. Auto-extracted from filename if None.
        trim_silence: Remove leading/trailing silence.
        normalize_audio: Normalize audio levels.
        add_name_card: Overlay a name lower-third.
        name_style: Style for the name card.
        target_width: Target output width.
        target_height: Target output height.
        on_progress: Progress callback(pct, msg).

    Returns:
        ProcessedMessage with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, f"Processing: {os.path.basename(video_path)}...")

    info = get_video_info(video_path)
    original_duration = info["duration"]

    if guest_name is None:
        guest_name = _extract_name_from_filename(os.path.basename(video_path))

    if output_path_str is None:
        output_path_str = output_path(video_path, "processed")

    if name_style is None:
        name_style = NameCardStyle()

    # Determine trim points
    content_start = 0.0
    content_end = original_duration

    if trim_silence and original_duration > 1.0:
        if on_progress:
            on_progress(15, "Detecting silence boundaries...")
        content_start, content_end = _detect_silence_boundaries(video_path)

    trimmed_duration = content_end - content_start
    silence_removed = original_duration - trimmed_duration

    if on_progress:
        on_progress(35, "Building processing pipeline...")

    # Build video filter chain
    vf_parts = []

    # Scale to target resolution
    vf_parts.append(
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )

    # Name card overlay (drawtext)
    if add_name_card and guest_name:
        margin = name_style.margin
        padding = name_style.padding
        fontsize = name_style.font_size
        card_dur = name_style.display_duration
        fade_in = name_style.fade_in
        fade_out = name_style.fade_out
        escaped_name = guest_name.replace("'", "'\\''").replace(":", "\\:")

        # Show name card for first N seconds with fade
        enable_expr = f"between(t,0,{card_dur})"
        alpha_expr = (
            f"if(lt(t,{fade_in}),t/{fade_in},"
            f"if(gt(t,{card_dur - fade_out}),({card_dur}-t)/{fade_out},1))"
        )

        if name_style.position == "bottom_center":
            x_expr = "(w-tw)/2"
        else:
            x_expr = str(margin)
        y_expr = f"h-{margin}-th"

        vf_parts.append(
            f"drawtext=text='{escaped_name}'"
            f":fontsize={fontsize}"
            f":fontcolor={name_style.font_color}"
            f":box=1:boxcolor={name_style.bg_color}:boxborderw={padding}"
            f":x={x_expr}:y={y_expr}"
            f":alpha='{alpha_expr}'"
            f":enable='{enable_expr}'"
        )

    # Build audio filter chain
    af_parts = []
    if normalize_audio:
        af_parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    if on_progress:
        on_progress(50, "Encoding processed message...")

    # Build FFmpeg command
    builder = FFmpegCmd()
    builder.input(video_path)

    if content_start > 0:
        builder.seek(start=content_start)
    if content_end < original_duration:
        builder.option("t", str(trimmed_duration))

    vf_chain = ",".join(vf_parts)
    af_chain = ",".join(af_parts) if af_parts else None

    builder.video_codec("libx264", crf=18, preset="fast")
    builder.audio_codec("aac", bitrate="192k")
    builder.video_filter(vf_chain)
    if af_chain:
        builder.audio_filter(af_chain)
    builder.option("ac", "2")
    builder.option("ar", "48000")
    builder.faststart()
    builder.output(output_path_str)

    cmd = builder.build()
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Processed: {guest_name}")

    return ProcessedMessage(
        input_path=video_path,
        output_path=output_path_str,
        guest_name=guest_name,
        original_duration=round(original_duration, 3),
        trimmed_duration=round(trimmed_duration, 3),
        silence_removed=round(silence_removed, 3),
        audio_normalized=normalize_audio,
    )


def compile_guest_messages(
    folder_path: str,
    output_path_str: Optional[str] = None,
    trim_silence: bool = True,
    normalize_audio: bool = True,
    add_name_cards: bool = True,
    name_style: Optional[NameCardStyle] = None,
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    target_width: int = 1920,
    target_height: int = 1080,
    on_progress: Optional[Callable] = None,
) -> CompilationResult:
    """Auto-compile guest message videos from a folder.

    Args:
        folder_path: Directory containing guest message video files.
        output_path_str: Output file path. Auto-generated if None.
        trim_silence: Trim leading/trailing silence.
        normalize_audio: Normalize audio levels.
        add_name_cards: Add name lower-thirds from filenames.
        name_style: NameCardStyle for lower-thirds.
        transition: Transition type: 'cut', 'crossfade', 'fade_to_black'.
        transition_duration: Duration of transitions in seconds.
        target_width: Target output width.
        target_height: Target output height.
        on_progress: Progress callback(pct, msg).

    Returns:
        CompilationResult with output path, processed messages, etc.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find video files
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts"}
    video_files = []
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in video_extensions:
            video_files.append(os.path.join(folder_path, fname))

    if not video_files:
        raise ValueError(f"No video files found in {folder_path}")

    if on_progress:
        on_progress(5, f"Found {len(video_files)} guest message videos...")

    if output_path_str is None:
        output_path_str = os.path.join(folder_path, "guest_compilation.mp4")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_guestcomp_")
    messages = []
    skipped = []

    try:
        # Process each message
        for idx, vf in enumerate(video_files):
            if on_progress:
                pct = 5 + int(60 * idx / len(video_files))
                on_progress(pct, f"Processing message {idx + 1}/{len(video_files)}...")

            try:
                seg_path = os.path.join(tmp_dir, f"msg_{idx:04d}.mp4")
                msg = process_single_message(
                    video_path=vf,
                    output_path_str=seg_path,
                    trim_silence=trim_silence,
                    normalize_audio=normalize_audio,
                    add_name_card=add_name_cards,
                    name_style=name_style,
                    target_width=target_width,
                    target_height=target_height,
                )
                messages.append(msg)
            except Exception as e:
                logger.warning("Skipping %s: %s", os.path.basename(vf), e)
                skipped.append(vf)

        if not messages:
            raise RuntimeError("No messages could be processed")

        if on_progress:
            on_progress(75, f"Concatenating {len(messages)} messages...")

        # Write concat list
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for msg in messages:
                f.write(f"file '{msg.output_path}'\n")

        # Concatenate all processed messages
        cmd = (FFmpegCmd()
               .pre_input("f", "concat")
               .pre_input("safe", "0")
               .input(concat_file)
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd)

        total_duration = sum(m.trimmed_duration for m in messages)

        if on_progress:
            on_progress(100, f"Compiled {len(messages)} messages, "
                             f"total {total_duration:.1f}s")

        return CompilationResult(
            output_path=output_path_str,
            total_duration=round(total_duration, 3),
            message_count=len(messages),
            messages=messages,
            skipped=skipped,
        )

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
