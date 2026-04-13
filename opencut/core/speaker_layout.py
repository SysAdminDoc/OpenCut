"""
Multi-Speaker Layout Engine (40.2)

Auto-layout multiple speaker videos in grid, spotlight, or PiP
arrangements, driven by speaker diarization data.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LayoutConfig:
    """Configuration for multi-speaker layout."""
    layout_type: str = "grid"  # "grid", "spotlight", "pip", "sidebar"
    width: int = 1920
    height: int = 1080
    background_color: str = "black"
    gap: int = 4  # pixels between grid cells
    spotlight_index: int = 0  # which speaker to spotlight
    pip_size_pct: float = 25.0  # PiP size as percentage of canvas
    pip_position: str = "bottom_right"  # "top_left", "top_right", "bottom_left", "bottom_right"
    border_color: str = ""  # optional border around active speaker
    border_width: int = 3
    transition_frames: int = 5


# ---------------------------------------------------------------------------
# Layout Calculation
# ---------------------------------------------------------------------------
def _calc_grid(n: int, width: int, height: int, gap: int) -> List[Dict]:
    """Calculate grid positions for n speakers."""
    if n == 0:
        return []
    if n == 1:
        return [{"x": 0, "y": 0, "w": width, "h": height}]

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w = (width - gap * (cols - 1)) // cols
    cell_h = (height - gap * (rows - 1)) // rows

    # Ensure even dimensions
    cell_w = cell_w - (cell_w % 2)
    cell_h = cell_h - (cell_h % 2)

    positions = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        positions.append({"x": x, "y": y, "w": cell_w, "h": cell_h})

    return positions


def _calc_spotlight(
    n: int, width: int, height: int, spotlight_idx: int,
) -> List[Dict]:
    """Calculate spotlight layout: one large, rest small at bottom."""
    if n == 0:
        return []
    if n == 1:
        return [{"x": 0, "y": 0, "w": width, "h": height}]

    # Main speaker gets 75% of height
    main_h = int(height * 0.75)
    main_h = main_h - (main_h % 2)
    thumb_h = height - main_h
    thumb_h = thumb_h - (thumb_h % 2)

    positions = []
    others = n - 1
    thumb_w = width // max(others, 1)
    thumb_w = thumb_w - (thumb_w % 2)

    for i in range(n):
        if i == spotlight_idx:
            positions.append({"x": 0, "y": 0, "w": width, "h": main_h})
        else:
            thumb_idx = i if i < spotlight_idx else i - 1
            x = thumb_idx * thumb_w
            positions.append({
                "x": x, "y": main_h,
                "w": thumb_w, "h": thumb_h,
            })

    return positions


def _calc_pip(
    n: int, width: int, height: int,
    pip_size_pct: float, pip_position: str,
) -> List[Dict]:
    """Calculate PiP layout: first speaker full, rest as small overlays."""
    if n == 0:
        return []

    positions = [{"x": 0, "y": 0, "w": width, "h": height}]

    if n == 1:
        return positions

    pip_w = int(width * pip_size_pct / 100)
    pip_h = int(height * pip_size_pct / 100)
    pip_w = pip_w - (pip_w % 2)
    pip_h = pip_h - (pip_h % 2)
    margin = 20

    for i in range(1, n):
        if pip_position == "top_left":
            x, y = margin, margin + (i - 1) * (pip_h + margin)
        elif pip_position == "top_right":
            x, y = width - pip_w - margin, margin + (i - 1) * (pip_h + margin)
        elif pip_position == "bottom_left":
            x, y = margin, height - pip_h - margin - (i - 1) * (pip_h + margin)
        else:  # bottom_right
            x, y = width - pip_w - margin, height - pip_h - margin - (i - 1) * (pip_h + margin)

        positions.append({"x": max(0, x), "y": max(0, y), "w": pip_w, "h": pip_h})

    return positions


# ---------------------------------------------------------------------------
# Create Speaker Layout
# ---------------------------------------------------------------------------
def create_speaker_layout(
    video_paths: List[str],
    layout_type: str = "grid",
    config: Optional[LayoutConfig] = None,
    output_path_str: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Create a multi-speaker layout video.

    Args:
        video_paths: List of video file paths (one per speaker).
        layout_type: Layout type — "grid", "spotlight", "pip", "sidebar".
        config: Optional LayoutConfig.
        output_path_str: Output file path.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, layout_type, speaker_count.
    """
    if not video_paths:
        raise ValueError("At least one video path is required")

    for p in video_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Video not found: {p}")

    config = config or LayoutConfig(layout_type=layout_type)
    config.layout_type = layout_type
    n = len(video_paths)

    if on_progress:
        on_progress(5, f"Creating {layout_type} layout for {n} speakers...")

    # Calculate positions
    if layout_type == "spotlight":
        positions = _calc_spotlight(n, config.width, config.height, config.spotlight_index)
    elif layout_type == "pip":
        positions = _calc_pip(n, config.width, config.height, config.pip_size_pct, config.pip_position)
    elif layout_type == "sidebar":
        positions = _calc_spotlight(n, config.width, config.height, 0)
    else:
        positions = _calc_grid(n, config.width, config.height, config.gap)

    if not output_path_str:
        out_dir = os.path.dirname(os.path.abspath(video_paths[0]))
        output_path_str = output_path(video_paths[0], f"layout_{layout_type}", out_dir)

    if on_progress:
        on_progress(20, "Building filter graph...")

    # Build FFmpeg filter complex
    cmd = FFmpegCmd()

    for p in video_paths:
        cmd.input(p)

    # Build filter complex
    filter_parts = []
    # Create background
    filter_parts.append(
        f"color=c={config.background_color}:s={config.width}x{config.height}:"
        f"d=3600:r=30[bg]"
    )

    # Scale each input to its cell size
    for i, pos in enumerate(positions):
        filter_parts.append(
            f"[{i}:v]scale={pos['w']}:{pos['h']}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={pos['w']}:{pos['h']}:(ow-iw)/2:(oh-ih)/2:"
            f"{config.background_color}[v{i}]"
        )

    # Overlay each speaker onto background
    prev = "bg"
    for i, pos in enumerate(positions):
        out_label = f"out{i}" if i < n - 1 else "outv"
        filter_parts.append(
            f"[{prev}][v{i}]overlay={pos['x']}:{pos['y']}:shortest=1[{out_label}]"
        )
        prev = out_label

    # Mix audio from all speakers
    if n > 1:
        audio_inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_parts.append(
            f"{audio_inputs}amix=inputs={n}:duration=shortest[outa]"
        )
        audio_map = "[outa]"
    else:
        audio_map = "0:a"

    fc = ";".join(filter_parts)
    cmd.filter_complex(fc, maps=["[outv]", audio_map])

    cmd = (cmd
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .output(output_path_str))

    if on_progress:
        on_progress(40, "Encoding layout...")

    run_ffmpeg(cmd.build())

    if on_progress:
        on_progress(100, f"{layout_type} layout complete")

    return {
        "output_path": output_path_str,
        "layout_type": layout_type,
        "speaker_count": n,
        "positions": positions,
        "width": config.width,
        "height": config.height,
    }


# ---------------------------------------------------------------------------
# Active Speaker Switching
# ---------------------------------------------------------------------------
def apply_active_speaker(
    video_paths: List[str],
    diarization: List[dict],
    output_path_str: str = "",
    speaker_map: Optional[Dict[str, int]] = None,
    config: Optional[LayoutConfig] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Apply active-speaker spotlight switching based on diarization.

    Generates a sequence where the speaking person is shown prominently,
    using cut-based switching (no complex real-time compositing).

    Args:
        video_paths: List of video files, one per speaker/camera.
        diarization: Diarization segments [{'speaker', 'start', 'end'}].
        output_path_str: Output file path.
        speaker_map: Map speaker labels to video_paths indices.
        config: Optional LayoutConfig.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, switches, duration.
    """
    if not video_paths:
        raise ValueError("At least one video path is required")
    if not diarization:
        raise ValueError("Diarization data is required")

    for p in video_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Video not found: {p}")

    config = config or LayoutConfig()

    # Auto-assign speakers if no map provided
    if not speaker_map:
        speakers = []
        for seg in sorted(diarization, key=lambda s: s.get("start", 0)):
            spk = seg.get("speaker", "")
            if spk and spk not in speakers:
                speakers.append(spk)
        speaker_map = {spk: i % len(video_paths) for i, spk in enumerate(speakers)}

    if on_progress:
        on_progress(10, "Planning speaker switches...")

    if not output_path_str:
        out_dir = os.path.dirname(os.path.abspath(video_paths[0]))
        output_path_str = output_path(video_paths[0], "active_speaker", out_dir)

    # Sort diarization by start time and merge short segments
    sorted_segs = sorted(diarization, key=lambda s: float(s.get("start", 0)))

    # Build cut list: for each segment, select the speaker's video
    import shutil
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="opencut_active_spk_")
    segment_files = []
    switches = 0

    try:
        total = len(sorted_segs)
        for i, seg in enumerate(sorted_segs):
            if on_progress and total > 0:
                pct = 10 + int((i / total) * 60)
                on_progress(pct, f"Processing segment {i + 1}/{total}...")

            speaker = seg.get("speaker", "")
            vid_idx = speaker_map.get(speaker, 0)
            vid_idx = min(vid_idx, len(video_paths) - 1)
            src = video_paths[vid_idx]

            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            if end <= start:
                continue

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            vf = (
                f"scale={config.width}:{config.height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black"
            )
            cmd = (FFmpegCmd()
                   .input(src, ss=start, to=end)
                   .video_filter(vf)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .output(seg_path)
                   .build())
            run_ffmpeg(cmd)
            segment_files.append(seg_path)
            switches += 1

        if not segment_files:
            raise ValueError("No valid segments to assemble")

        if on_progress:
            on_progress(80, "Concatenating segments...")

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
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if on_progress:
        on_progress(100, "Active speaker layout complete")

    return {
        "output_path": output_path_str,
        "switches": switches,
        "speaker_count": len(video_paths),
        "speaker_map": speaker_map,
    }
