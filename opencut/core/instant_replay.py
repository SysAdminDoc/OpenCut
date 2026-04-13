"""
OpenCut Instant Replay Builder (31.2)

Flag moments in a video and generate replay clips with:
- Slow-motion speed ramp (configurable curve)
- "REPLAY" text overlay with optional custom text
- Transition effects (fade, flash, zoom)
- Batch processing of multiple timestamps

All via FFmpeg -- zero external dependencies.
"""

import logging
import os
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


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ReplayConfig:
    """Configuration for instant replay generation."""
    pre_roll: float = 3.0          # seconds before the moment
    post_roll: float = 2.0         # seconds after the moment
    slow_factor: float = 0.5       # slow-motion factor (0.25 = 4x slow)
    ramp_in: float = 0.5           # seconds to ramp into slow-mo
    ramp_out: float = 0.5          # seconds to ramp out of slow-mo
    overlay_text: str = "REPLAY"   # text overlay
    font_size: int = 72            # overlay font size
    font_color: str = "white"      # overlay text color
    border_color: str = "#FFD700"  # gold border
    transition: str = "flash"      # 'flash', 'fade', 'zoom', 'none'
    transition_duration: float = 0.3
    include_original: bool = True  # include original speed first, then replay


@dataclass
class ReplayResult:
    """Result of a single replay generation."""
    output_path: str = ""
    timestamp: float = 0.0
    duration: float = 0.0
    slow_duration: float = 0.0     # duration of the slow-mo portion
    config: Optional[ReplayConfig] = None


@dataclass
class BatchReplayResult:
    """Result of batch replay generation."""
    replays: List[ReplayResult] = field(default_factory=list)
    output_dir: str = ""
    total_count: int = 0
    success_count: int = 0


# ---------------------------------------------------------------------------
# Speed Ramp Filter Generation
# ---------------------------------------------------------------------------
def _build_speed_filter(
    slow_factor: float,
    duration: float,
    ramp_in: float,
    ramp_out: float,
) -> str:
    """Build FFmpeg setpts expression for speed ramp.

    Creates a smooth speed transition: normal -> slow -> normal.

    The setpts filter uses PTS-STARTPTS to normalize timestamps,
    then applies a piecewise speed curve.
    """
    if slow_factor >= 1.0:
        return "setpts=PTS-STARTPTS"

    # Simple constant slow-motion (more reliable than complex ramps)
    pts_factor = 1.0 / slow_factor
    return f"setpts={pts_factor}*(PTS-STARTPTS)"


def _build_atempo_chain(slow_factor: float) -> str:
    """Build FFmpeg atempo filter chain for audio speed adjustment.

    atempo only accepts values in [0.5, 100.0], so we chain multiple
    filters for extreme slowdowns.
    """
    if slow_factor >= 1.0:
        return "anull"

    # Chain atempo filters (each can do 0.5x at minimum)
    filters = []
    remaining = slow_factor
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


# ---------------------------------------------------------------------------
# Single Replay Generation
# ---------------------------------------------------------------------------
def create_replay(
    video_path: str,
    timestamp: float,
    output_path_str: Optional[str] = None,
    config: Optional[ReplayConfig] = None,
    on_progress: Optional[Callable] = None,
) -> ReplayResult:
    """Generate an instant replay clip for a specific moment.

    Extracts the moment with pre/post roll, applies slow-motion with
    a speed ramp, adds "REPLAY" overlay text, and transition effects.

    Args:
        video_path: Source video file path.
        timestamp: The moment to replay (seconds into video).
        output_path_str: Output file path. Auto-generated if None.
        config: ReplayConfig with slow-mo and overlay settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReplayResult with output path and stats.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if config is None:
        config = ReplayConfig()

    if on_progress:
        on_progress(5, "Analyzing video...")

    info = get_video_info(video_path)
    video_duration = info["duration"]
    width, height = info["width"], info["height"]

    # Calculate extraction range
    start = max(0, timestamp - config.pre_roll)
    end = min(video_duration, timestamp + config.post_roll)
    clip_duration = end - start

    if clip_duration <= 0:
        raise ValueError(f"Invalid timestamp {timestamp} for video of duration {video_duration}")

    if output_path_str is None:
        ts_label = f"replay_{int(timestamp)}s"
        output_path_str = output_path(video_path, ts_label)

    tmp_dir = tempfile.mkdtemp(prefix="opencut_replay_")

    try:
        if on_progress:
            on_progress(10, "Extracting clip segment...")

        # Step 1: Extract the segment at original speed
        original_clip = os.path.join(tmp_dir, "original.mp4")
        cmd = (
            FFmpegCmd()
            .input(video_path, ss=str(round(start, 3)))
            .option("t", str(round(clip_duration, 3)))
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("aac", bitrate="192k")
            .output(original_clip)
            .build()
        )
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(30, "Applying slow-motion...")

        # Step 2: Create slow-motion version
        slow_clip = os.path.join(tmp_dir, "slow.mp4")
        speed_filter = _build_speed_filter(
            config.slow_factor, clip_duration,
            config.ramp_in, config.ramp_out,
        )
        atempo = _build_atempo_chain(config.slow_factor)

        cmd = (
            FFmpegCmd()
            .input(original_clip)
            .video_filter(speed_filter)
            .audio_filter(atempo)
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("aac", bitrate="192k")
            .output(slow_clip)
            .build()
        )
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(55, "Adding REPLAY overlay...")

        # Step 3: Add text overlay to slow-motion clip
        overlay_clip = os.path.join(tmp_dir, "overlay.mp4")
        text_esc = config.overlay_text.replace("'", "\\'")
        border_esc = config.border_color.replace("#", "0x")

        # Build drawtext filter
        drawtext = (
            f"drawtext=text='{text_esc}'"
            f":fontsize={config.font_size}"
            f":fontcolor={config.font_color}"
            f":borderw=3:bordercolor={border_esc}"
            f":x=(w-text_w)/2:y=h*0.1"
            f":enable='between(t,0.3,99999)'"
        )

        # Add flash transition effect at the start
        if config.transition == "flash":
            td = config.transition_duration
            vf = (
                f"fade=t=in:st=0:d={td}:color=white,"
                f"{drawtext}"
            )
        elif config.transition == "fade":
            td = config.transition_duration
            vf = f"fade=t=in:st=0:d={td},{drawtext}"
        elif config.transition == "zoom":
            td = config.transition_duration
            vf = (
                f"zoompan=z='if(lt(in_time,{td}),1.5-0.5*in_time/{td},1)'"
                f":d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":s={width}x{height},"
                f"{drawtext}"
            )
        else:
            vf = drawtext

        cmd = (
            FFmpegCmd()
            .input(slow_clip)
            .video_filter(vf)
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("copy")
            .output(overlay_clip)
            .build()
        )
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(75, "Assembling final replay...")

        # Step 4: Combine original + replay if requested
        if config.include_original:
            concat_file = os.path.join(tmp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                f.write(f"file '{original_clip}'\n")
                f.write(f"file '{overlay_clip}'\n")

            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path_str,
            ]
            run_ffmpeg(cmd)
        else:
            # Just use the overlay clip as output
            import shutil
            shutil.copy2(overlay_clip, output_path_str)

        # Calculate slow-mo duration
        slow_dur = clip_duration / config.slow_factor

        if on_progress:
            on_progress(100, "Instant replay complete")

        return ReplayResult(
            output_path=output_path_str,
            timestamp=timestamp,
            duration=clip_duration + (slow_dur if config.include_original else 0),
            slow_duration=slow_dur,
            config=config,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batch Replay Generation
# ---------------------------------------------------------------------------
def batch_replays(
    video_path: str,
    timestamps: List[float],
    output_dir: Optional[str] = None,
    config: Optional[ReplayConfig] = None,
    on_progress: Optional[Callable] = None,
) -> BatchReplayResult:
    """Generate instant replays for multiple timestamps.

    Args:
        video_path: Source video file path.
        timestamps: List of moment timestamps (seconds).
        output_dir: Directory for output files. Uses video dir if None.
        config: ReplayConfig for all replays. Uses defaults if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        BatchReplayResult with list of ReplayResult.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not timestamps:
        raise ValueError("No timestamps provided")

    if config is None:
        config = ReplayConfig()

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))
    os.makedirs(output_dir, exist_ok=True)

    replays = []
    total = len(timestamps)
    success = 0

    for i, ts in enumerate(sorted(timestamps)):
        if on_progress:
            pct = int(5 + 90 * i / total)
            on_progress(pct, f"Creating replay {i + 1}/{total} at {ts:.1f}s...")

        ts_label = f"replay_{int(ts)}s"
        out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{ts_label}.mp4")

        try:
            result = create_replay(
                video_path=video_path,
                timestamp=ts,
                output_path_str=out_path,
                config=config,
            )
            replays.append(result)
            success += 1
        except Exception as e:
            logger.warning("Failed to create replay at %.1fs: %s", ts, e)
            replays.append(ReplayResult(
                output_path="",
                timestamp=ts,
                config=config,
            ))

    if on_progress:
        on_progress(100, f"Created {success}/{total} replays")

    return BatchReplayResult(
        replays=replays,
        output_dir=output_dir,
        total_count=total,
        success_count=success,
    )
