"""
OpenCut Photo+Video Montage Module (48.3)

Mix photos (with Ken Burns pan/zoom) and video clips,
sync cuts to music beats for dynamic montages.
"""

import logging
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts"}


@dataclass
class KenBurnsConfig:
    """Configuration for Ken Burns effect on a single image."""
    start_scale: float = 1.0
    end_scale: float = 1.3
    start_x: float = 0.5  # normalized 0-1 center point
    start_y: float = 0.5
    end_x: float = 0.5
    end_y: float = 0.5
    duration: float = 5.0


@dataclass
class MontageSegment:
    """A segment in the montage (either photo or video)."""
    media_path: str
    is_image: bool = False
    duration: float = 5.0
    start_time: float = 0.0  # for video clips, trim start
    ken_burns: Optional[KenBurnsConfig] = None


@dataclass
class MontageResult:
    """Result of montage creation."""
    output_path: str = ""
    total_duration: float = 0.0
    segment_count: int = 0
    image_count: int = 0
    video_count: int = 0
    music_synced: bool = False


def _is_image(filepath: str) -> bool:
    """Check if file is a supported image format."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _is_video(filepath: str) -> bool:
    """Check if file is a supported video format."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in VIDEO_EXTENSIONS


def _detect_focus_point(image_path: str) -> Tuple[float, float]:
    """Detect the focal point of an image for Ken Burns.

    Uses a simple center-of-mass approach on luminance.
    Falls back to center (0.5, 0.5) if detection fails.
    """
    # Try using PIL for basic face/focus detection
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=float)

        # Weight by brightness (focus tends to be on lighter areas)
        total = arr.sum()
        if total > 0:
            ys, xs = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
            cy = float((ys * arr).sum() / total) / arr.shape[0]
            cx = float((xs * arr).sum() / total) / arr.shape[1]
            # Clamp to inner 80% to avoid extreme edges
            cx = max(0.2, min(0.8, cx))
            cy = max(0.2, min(0.8, cy))
            return cx, cy
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Focus detection failed for %s: %s", image_path, e)

    return 0.5, 0.5


def apply_ken_burns(
    image_path: str,
    duration: float = 5.0,
    focus_point: Optional[Tuple[float, float]] = None,
    output_path_str: Optional[str] = None,
    width: int = 1920,
    height: int = 1080,
    effect: str = "zoom_in",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Apply Ken Burns pan/zoom effect to a still image.

    Args:
        image_path: Path to the source image.
        duration: Duration of the output video segment.
        focus_point: (x, y) normalized focal point. Auto-detected if None.
        output_path_str: Output video path. Auto-generated if None.
        width: Output video width.
        height: Output video height.
        effect: Effect type: 'zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'random'.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, duration, effect applied.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if on_progress:
        on_progress(10, f"Creating Ken Burns effect for {os.path.basename(image_path)}...")

    if focus_point is None:
        focus_point = _detect_focus_point(image_path)

    fx, fy = focus_point

    if effect == "random":
        effect = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

    if output_path_str is None:
        output_path_str = output_path(image_path, "kenburns")
        output_path_str = os.path.splitext(output_path_str)[0] + ".mp4"

    fps = 30
    total_frames = int(duration * fps)

    # Build zoompan filter based on effect type
    # zoompan uses 'z' for zoom, 'x'/'y' for pan position
    if effect == "zoom_in":
        zoom_expr = "min(zoom+0.001,1.3)"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"
    elif effect == "zoom_out":
        zoom_expr = "if(eq(on,1),1.3,max(zoom-0.001,1.0))"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"
    elif effect == "pan_left":
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*on/{total_frames}"
        y_expr = "(ih-ih/zoom)/2"
    elif effect == "pan_right":
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "(ih-ih/zoom)/2"
    else:
        zoom_expr = "min(zoom+0.001,1.3)"
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"

    vf = (
        f"scale=8000:-1,"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )

    if on_progress:
        on_progress(30, "Rendering Ken Burns animation...")

    cmd = (FFmpegCmd()
           .pre_input("loop", "1")
           .input(image_path)
           .video_codec("libx264", crf=18, preset="fast")
           .video_filter(vf)
           .option("t", str(duration))
           .option("r", str(fps))
           .output(output_path_str)
           .build())

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Ken Burns {effect} complete")

    return {
        "output_path": output_path_str,
        "duration": duration,
        "effect": effect,
        "focus_point": list(focus_point),
    }


def _detect_beats(music_path: str) -> List[float]:
    """Detect beat timestamps in a music file. Returns list of beat times in seconds."""
    # Try librosa first
    try:
        import librosa
        y, sr = librosa.load(music_path, sr=22050, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return [round(float(t), 3) for t in beat_times]
    except ImportError:
        pass

    # FFmpeg fallback: estimate from audio energy
    try:
        from opencut.core.beat_cuts import _detect_beats_energy
        markers = _detect_beats_energy(music_path)
        return [m.time for m in markers]
    except ImportError:
        pass

    # Last resort: regular intervals at ~120 BPM
    info = get_video_info(music_path)
    duration = info.get("duration", 120)
    interval = 0.5  # 120 BPM
    return [round(t, 3) for t in _frange(interval, duration, interval)]


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    current = start
    while current < stop:
        yield current
        current += step


def sync_to_beats(
    clips: List[Dict],
    beat_times: List[float],
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """Assign durations to clips based on beat intervals.

    Args:
        clips: List of dicts with 'media_path' and optional 'min_duration'.
        beat_times: Sorted list of beat timestamps.
        on_progress: Progress callback(pct, msg).

    Returns:
        Updated clips list with 'duration' and 'beat_time' fields.
    """
    if not clips:
        raise ValueError("At least one clip is required")
    if not beat_times:
        raise ValueError("At least one beat time is required")

    if on_progress:
        on_progress(10, "Syncing clips to beats...")

    # Calculate beat intervals
    intervals = []
    for i in range(len(beat_times) - 1):
        intervals.append(beat_times[i + 1] - beat_times[i])
    if intervals:
        intervals.append(intervals[-1])  # extend last

    # Assign beats to clips
    result = []
    beat_idx = 0
    for idx, clip in enumerate(clips):
        if beat_idx >= len(beat_times):
            beat_idx = beat_idx % max(len(beat_times), 1)

        min_dur = clip.get("min_duration", 0)
        # Accumulate beat intervals until we meet minimum duration
        clip_duration = 0.0
        start_beat = beat_idx
        while clip_duration < max(min_dur, 0.5) and beat_idx < len(intervals):
            clip_duration += intervals[beat_idx]
            beat_idx += 1

        if clip_duration <= 0:
            clip_duration = 2.0  # fallback

        result.append({
            **clip,
            "duration": round(clip_duration, 3),
            "beat_time": beat_times[start_beat] if start_beat < len(beat_times) else 0,
        })

    if on_progress:
        on_progress(100, f"Synced {len(result)} clips to beats")

    return result


def create_montage(
    media_paths: List[str],
    music_path: Optional[str] = None,
    output_path_str: Optional[str] = None,
    image_duration: float = 5.0,
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    ken_burns_effect: str = "random",
    width: int = 1920,
    height: int = 1080,
    sync_to_music: bool = True,
    on_progress: Optional[Callable] = None,
) -> MontageResult:
    """Create a photo+video montage with optional music sync.

    Args:
        media_paths: List of image and/or video file paths.
        music_path: Optional music track to sync to.
        output_path_str: Output file path. Auto-generated if None.
        image_duration: Default duration for images (if no music sync).
        transition: Transition type: 'cut', 'crossfade'.
        transition_duration: Duration of transitions.
        ken_burns_effect: Ken Burns effect for images: 'zoom_in', 'zoom_out',
                          'pan_left', 'pan_right', 'random'.
        width: Output width.
        height: Output height.
        sync_to_music: Whether to sync cuts to music beats.
        on_progress: Progress callback(pct, msg).

    Returns:
        MontageResult with output path, duration, segment counts.
    """
    if not media_paths:
        raise ValueError("At least one media file is required")

    valid_paths = [p for p in media_paths if os.path.isfile(p) and
                   (_is_image(p) or _is_video(p))]
    if not valid_paths:
        raise ValueError("No valid image or video files found")

    if on_progress:
        on_progress(5, f"Preparing montage with {len(valid_paths)} media files...")

    if output_path_str is None:
        output_path_str = output_path(valid_paths[0], "montage")
        output_path_str = os.path.splitext(output_path_str)[0] + ".mp4"

    # Detect beats if music provided
    beat_times = None
    music_synced = False
    if music_path and os.path.isfile(music_path) and sync_to_music:
        if on_progress:
            on_progress(10, "Detecting beats in music...")
        beat_times = _detect_beats(music_path)
        music_synced = len(beat_times) > 0

    # Plan segment durations
    segments = []
    for p in valid_paths:
        is_img = _is_image(p)
        info = {} if is_img else get_video_info(p)
        segments.append({
            "media_path": p,
            "is_image": is_img,
            "min_duration": image_duration if is_img else max(2.0, info.get("duration", 5.0)),
        })

    # Sync to beats if available
    if beat_times and music_synced:
        segments = sync_to_beats(segments, beat_times)
    else:
        for seg in segments:
            if "duration" not in seg:
                seg["duration"] = seg.get("min_duration", image_duration)

    if on_progress:
        on_progress(20, f"Creating {len(segments)} montage segments...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_montage_")
    image_count = 0
    video_count = 0

    try:
        segment_files = []

        for idx, seg in enumerate(segments):
            if on_progress:
                pct = 20 + int(50 * idx / len(segments))
                on_progress(pct, f"Processing segment {idx + 1}/{len(segments)}...")

            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            seg_duration = seg.get("duration", image_duration)

            if seg.get("is_image", _is_image(seg["media_path"])):
                # Ken Burns for images
                apply_ken_burns(
                    image_path=seg["media_path"],
                    duration=seg_duration,
                    output_path_str=seg_path,
                    width=width,
                    height=height,
                    effect=ken_burns_effect,
                )
                image_count += 1
            else:
                # Trim and scale video clip
                cmd = (FFmpegCmd()
                       .input(seg["media_path"])
                       .video_codec("libx264", crf=18, preset="fast")
                       .audio_codec("aac", bitrate="192k")
                       .video_filter(
                           f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                       )
                       .option("t", str(seg_duration))
                       .option("r", "30")
                       .option("an")  # strip audio, we'll add music
                       .output(seg_path)
                       .build())
                run_ffmpeg(cmd)
                video_count += 1

            segment_files.append(seg_path)

        if on_progress:
            on_progress(75, "Concatenating segments...")

        # Concat all segments
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        concat_video = os.path.join(tmp_dir, "concat.mp4")
        cmd = (FFmpegCmd()
               .pre_input("f", "concat")
               .pre_input("safe", "0")
               .input(concat_file)
               .video_codec("libx264", crf=18, preset="fast")
               .option("an")
               .output(concat_video)
               .build())
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(88, "Adding music track..." if music_path else "Finalizing...")

        total_duration = sum(seg.get("duration", image_duration) for seg in segments)

        # Merge with music or finalize
        if music_path and os.path.isfile(music_path):
            cmd = (FFmpegCmd()
                   .input(concat_video)
                   .input(music_path)
                   .map("0:v:0")
                   .map("1:a:0")
                   .video_codec("copy")
                   .audio_codec("aac", bitrate="192k")
                   .option("shortest")
                   .option("t", str(total_duration))
                   .faststart()
                   .output(output_path_str)
                   .build())
            run_ffmpeg(cmd)
        else:
            # No music, just copy the concat output
            import shutil
            shutil.copy2(concat_video, output_path_str)

        if on_progress:
            on_progress(100, f"Montage complete: {image_count} photos, "
                             f"{video_count} videos, {total_duration:.1f}s")

        return MontageResult(
            output_path=output_path_str,
            total_duration=round(total_duration, 3),
            segment_count=len(segments),
            image_count=image_count,
            video_count=video_count,
            music_synced=music_synced,
        )

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
