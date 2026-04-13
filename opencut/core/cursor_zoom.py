"""
OpenCut Cursor Zoom Module

Auto zoom-to-cursor in screen recordings:
- Detect click regions by analyzing cursor position changes + static moments
- Extract frames at 2fps via FFmpeg, analyze motion center of mass
- Apply smooth zoom in/out at each click using FFmpeg zoompan filter

All via FFmpeg + basic frame analysis - no heavy dependencies.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class ClickRegion:
    """A detected cursor click location."""
    timestamp: float
    x: int
    y: int
    confidence: float = 0.0


@dataclass
class CursorZoomResult:
    """Result of cursor zoom processing."""
    output_path: str = ""
    click_regions: List[ClickRegion] = field(default_factory=list)
    zoom_factor: float = 2.0
    zoom_duration: float = 1.5
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Frame Analysis Helpers
# ---------------------------------------------------------------------------

def _read_raw_frame(path: str, width: int, height: int) -> Optional[bytes]:
    """Read a raw RGB frame from file."""
    expected = width * height * 3
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) >= expected:
            return data[:expected]
        return None
    except OSError:
        return None


def _frame_region_energy(
    frame: bytes, width: int, height: int,
    rx: int, ry: int, rw: int, rh: int,
) -> float:
    """Compute average pixel intensity in a region of a raw RGB frame."""
    total = 0
    count = 0
    for y in range(max(0, ry), min(height, ry + rh)):
        for x in range(max(0, rx), min(width, rx + rw)):
            offset = (y * width + x) * 3
            if offset + 2 < len(frame):
                total += frame[offset] + frame[offset + 1] + frame[offset + 2]
                count += 3
    return total / count if count > 0 else 0.0


def _diff_frames(
    frame_a: bytes, frame_b: bytes, width: int, height: int,
    block_size: int = 32,
) -> List[dict]:
    """Compare two raw RGB frames and find blocks with highest change.

    Returns list of {x, y, energy} dicts for blocks with significant change.
    """
    blocks = []
    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            diff_sum = 0
            count = 0
            bw = min(block_size, width - bx)
            bh = min(block_size, height - by)
            for dy in range(bh):
                for dx in range(bw):
                    off = ((by + dy) * width + (bx + dx)) * 3
                    if off + 2 < len(frame_a) and off + 2 < len(frame_b):
                        diff_sum += (
                            abs(frame_a[off] - frame_b[off])
                            + abs(frame_a[off + 1] - frame_b[off + 1])
                            + abs(frame_a[off + 2] - frame_b[off + 2])
                        )
                        count += 3
            energy = diff_sum / count if count > 0 else 0.0
            if energy > 5.0:  # threshold for meaningful change
                blocks.append({"x": bx + bw // 2, "y": by + bh // 2, "energy": energy})
    return blocks


def _center_of_mass(blocks: List[dict]) -> Optional[dict]:
    """Compute weighted center of mass from diff blocks."""
    if not blocks:
        return None
    total_w = sum(b["energy"] for b in blocks)
    if total_w < 1e-6:
        return None
    cx = sum(b["x"] * b["energy"] for b in blocks) / total_w
    cy = sum(b["y"] * b["energy"] for b in blocks) / total_w
    return {"x": int(cx), "y": int(cy), "energy": total_w}


# ---------------------------------------------------------------------------
# Click Detection
# ---------------------------------------------------------------------------

def detect_click_regions(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> List[ClickRegion]:
    """Detect click regions in a screen recording.

    Analyzes frames at 2fps for cursor position changes. A click is detected
    when the cursor stops briefly (small motion region persists across frames).

    Args:
        input_path: Path to screen recording video.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ClickRegion with detected click positions and timestamps.
    """
    if on_progress:
        on_progress(5, "Analyzing video for cursor clicks...")

    info = get_video_info(input_path)
    width, height = info["width"], info["height"]
    info["duration"]
    fps_extract = 2  # extract at 2fps for analysis

    # Create temp directory for extracted frames
    tmp_dir = tempfile.mkdtemp(prefix="opencut_cursor_")
    try:
        # Extract frames at 2fps as raw RGB
        frame_pattern = os.path.join(tmp_dir, "frame_%05d.raw")
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-vf", f"fps={fps_extract},scale=640:360",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-y", frame_pattern,
        ]
        # Use individual frame files via image2
        frame_pattern_png = os.path.join(tmp_dir, "frame_%05d.png")
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-vf", f"fps={fps_extract},scale=640:360",
            "-y", frame_pattern_png,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(30, "Extracting motion regions...")

        # Read frames and detect motion hotspots
        analysis_w, analysis_h = 640, 360
        scale_x = width / analysis_w
        scale_y = height / analysis_h

        # Convert PNGs to raw for analysis (or read via ffmpeg pipe)
        frame_files = sorted(
            f for f in os.listdir(tmp_dir) if f.endswith(".png")
        )

        prev_frame = None
        motion_history = []  # list of (timestamp, center_x, center_y, energy)

        for i, fname in enumerate(frame_files):
            fpath = os.path.join(tmp_dir, fname)
            # Convert single frame to raw via ffmpeg
            raw_path = fpath + ".raw"
            cmd_raw = [
                get_ffmpeg_path(), "-i", fpath,
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-y", raw_path,
            ]
            try:
                run_ffmpeg(cmd_raw)
            except RuntimeError:
                continue

            frame_data = _read_raw_frame(raw_path, analysis_w, analysis_h)
            if frame_data is None:
                continue

            if prev_frame is not None:
                blocks = _diff_frames(prev_frame, frame_data, analysis_w, analysis_h, block_size=16)
                com = _center_of_mass(blocks)
                if com:
                    timestamp = i / fps_extract
                    motion_history.append((timestamp, com["x"], com["y"], com["energy"]))

            prev_frame = frame_data

            if on_progress and len(frame_files) > 0:
                pct = 30 + int(50 * (i + 1) / len(frame_files))
                on_progress(pct, f"Analyzing frame {i + 1}/{len(frame_files)}...")

        if on_progress:
            on_progress(85, "Identifying click positions...")

        # Detect clicks: small, concentrated motion followed by stillness
        click_regions = []
        i = 0
        while i < len(motion_history):
            ts, mx, my, energy = motion_history[i]
            # Look for a burst of small motion (cursor moving to click target)
            # followed by a pause (the click)
            if energy < 500:  # small concentrated motion
                # Check if next frame has very low motion (click pause)
                if i + 1 < len(motion_history):
                    next_energy = motion_history[i + 1][3]
                    if next_energy < 200:  # stillness after motion = click
                        click_x = int(mx * scale_x)
                        click_y = int(my * scale_y)
                        confidence = min(1.0, (500 - energy) / 500 * 0.5 + (200 - next_energy) / 200 * 0.5)
                        click_regions.append(ClickRegion(
                            timestamp=ts,
                            x=click_x,
                            y=click_y,
                            confidence=round(confidence, 3),
                        ))
                        i += 2  # skip the stillness frame
                        continue
            i += 1

        if on_progress:
            on_progress(100, f"Detected {len(click_regions)} click regions")

        return click_regions

    finally:
        # Cleanup temp directory
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Zoom Application
# ---------------------------------------------------------------------------

def _ease_in_out(t: float) -> float:
    """Smooth ease-in/ease-out curve."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def apply_cursor_zoom(
    input_path: str,
    click_regions: List[ClickRegion],
    zoom_factor: float = 2.0,
    zoom_duration: float = 1.5,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Apply smooth zoom-to-cursor at detected click positions.

    Generates zoom keyframes: at each click, smooth zoom in to click position,
    hold briefly, then zoom back out with ease-in/ease-out.

    Args:
        input_path: Source video file.
        click_regions: List of ClickRegion from detect_click_regions.
        zoom_factor: Maximum zoom level (1.0-5.0). Default 2.0.
        zoom_duration: Duration of each zoom in/out cycle in seconds.
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, click_count, zoom_factor, duration.
    """
    if on_progress:
        on_progress(5, "Preparing zoom keyframes...")

    zoom_factor = max(1.1, min(5.0, zoom_factor))
    zoom_duration = max(0.5, min(5.0, zoom_duration))

    info = get_video_info(input_path)
    width, height = info["width"], info["height"]
    fps = info["fps"]
    duration = info["duration"]

    if output_path_str is None:
        output_path_str = output_path(input_path, "cursor_zoom")

    if not click_regions:
        # No clicks detected, just copy input
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-c", "copy", "-y", output_path_str,
        ]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "click_count": 0,
            "zoom_factor": zoom_factor,
            "duration": duration,
        }

    if on_progress:
        on_progress(15, f"Building zoompan filter for {len(click_regions)} clicks...")

    # Build zoompan filter expression
    # zoompan works with per-frame zoom/x/y expressions
    total_frames = int(duration * fps)
    half_dur = zoom_duration / 2.0
    half_frames = int(half_dur * fps)

    # Build per-frame zoom/pan schedule as zoompan expressions
    # We use conditional expressions in zoompan filter
    zoom_parts = []
    x_parts = []
    y_parts = []

    for cr in click_regions:
        # Frame range for this zoom event
        center_frame = int(cr.timestamp * fps)
        start_frame = max(0, center_frame - half_frames)
        end_frame = min(total_frames, center_frame + half_frames)

        # Normalize click position for zoompan (0-based from top-left)
        # zoompan x/y are the top-left corner of the visible area
        norm_x = cr.x / width
        norm_y = cr.y / height

        # Zoom expression: ramp up then down within the frame range
        # Use if(between(on,start,mid), ease_in, if(between(on,mid,end), ease_out, 1))
        mid_frame = center_frame
        zoom_in = (
            f"if(between(on,{start_frame},{mid_frame}),"
            f"{1.0}+({zoom_factor - 1.0})*((on-{start_frame})/{max(1, mid_frame - start_frame)})"
            f"*((on-{start_frame})/{max(1, mid_frame - start_frame)}),"  # quadratic ease-in
        )
        zoom_out = (
            f"if(between(on,{mid_frame},{end_frame}),"
            f"{zoom_factor}-({zoom_factor - 1.0})*((on-{mid_frame})/{max(1, end_frame - mid_frame)})"
            f"*((on-{mid_frame})/{max(1, end_frame - mid_frame)}),"  # quadratic ease-out
        )
        zoom_parts.append(f"{zoom_in}{zoom_out}0))")

        # X/Y pan: move toward click position as we zoom in
        target_x = max(0, min(int(norm_x * width - width / (2 * zoom_factor)), int(width - width / zoom_factor)))
        target_y = max(0, min(int(norm_y * height - height / (2 * zoom_factor)), int(height - height / zoom_factor)))

        x_expr_in = (
            f"if(between(on,{start_frame},{mid_frame}),"
            f"{target_x}*(on-{start_frame})/{max(1, mid_frame - start_frame)},"
        )
        x_expr_out = (
            f"if(between(on,{mid_frame},{end_frame}),"
            f"{target_x}*(1-(on-{mid_frame})/{max(1, end_frame - mid_frame)}),"
        )
        x_parts.append(f"{x_expr_in}{x_expr_out}0))")

        y_expr_in = (
            f"if(between(on,{start_frame},{mid_frame}),"
            f"{target_y}*(on-{start_frame})/{max(1, mid_frame - start_frame)},"
        )
        y_expr_out = (
            f"if(between(on,{mid_frame},{end_frame}),"
            f"{target_y}*(1-(on-{mid_frame})/{max(1, end_frame - mid_frame)}),"
        )
        y_parts.append(f"{y_expr_in}{y_expr_out}0))")

    # Combine all zoom expressions (sum all zooms, clamp to max)
    # For simplicity, use nested if/else for non-overlapping zoom events
    "+".join(zoom_parts) if zoom_parts else "1"
    "+".join(x_parts) if x_parts else "0"
    "+".join(y_parts) if y_parts else "0"

    # Use zoompan filter
    # Fallback to simpler approach: generate a filter_complex with scale+crop per segment
    if on_progress:
        on_progress(30, "Applying zoom effects...")

    # Build filter: use zoompan with frame-level expressions

    # For complex expressions, fall back to simpler segment-based approach
    # Split into segments: before/during/after each zoom, concat
    segments = _build_zoom_segments(input_path, click_regions, zoom_factor, zoom_duration, info)

    if on_progress:
        on_progress(50, "Rendering zoom segments...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_czoom_")
    try:
        segment_files = []
        concat_list = os.path.join(tmp_dir, "concat.txt")

        for idx, seg in enumerate(segments):
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            segment_files.append(seg_path)

            if seg["type"] == "passthrough":
                # Copy segment as-is
                cmd = [
                    get_ffmpeg_path(), "-i", input_path,
                    "-ss", str(seg["start"]), "-t", str(seg["duration"]),
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-y", seg_path,
                ]
                run_ffmpeg(cmd)
            elif seg["type"] == "zoom":
                # Apply zoom via scale+crop
                crop_w = int(width / seg["zoom"])
                crop_h = int(height / seg["zoom"])
                crop_x = max(0, min(seg["cx"] - crop_w // 2, width - crop_w))
                crop_y = max(0, min(seg["cy"] - crop_h // 2, height - crop_h))

                vf = (
                    f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
                    f"scale={width}:{height}:flags=lanczos"
                )
                cmd = [
                    get_ffmpeg_path(), "-i", input_path,
                    "-ss", str(seg["start"]), "-t", str(seg["duration"]),
                    "-vf", vf,
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-y", seg_path,
                ]
                run_ffmpeg(cmd)

            if on_progress:
                pct = 50 + int(40 * (idx + 1) / len(segments))
                on_progress(pct, f"Rendered segment {idx + 1}/{len(segments)}")

        # Write concat list
        with open(concat_list, "w") as f:
            for sp in segment_files:
                f.write(f"file '{sp}'\n")

        if on_progress:
            on_progress(92, "Concatenating segments...")

        # Concat all segments
        cmd = [
            get_ffmpeg_path(), "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-y", output_path_str,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Cursor zoom applied")

        return {
            "output_path": output_path_str,
            "click_count": len(click_regions),
            "zoom_factor": zoom_factor,
            "duration": duration,
        }

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


def _build_zoom_segments(
    input_path: str,
    click_regions: List[ClickRegion],
    zoom_factor: float,
    zoom_duration: float,
    info: dict,
) -> List[dict]:
    """Build a list of video segments: passthrough and zoom sections.

    Each zoom event is split into zoom-in, hold, zoom-out subsegments
    approximated by a static crop at different zoom levels.
    """
    duration = info["duration"]
    _width, _height = info["width"], info["height"]
    half_dur = zoom_duration / 2.0
    ramp_steps = 3  # number of ramp steps for smooth zoom

    # Sort click regions by timestamp
    sorted_clicks = sorted(click_regions, key=lambda c: c.timestamp)

    segments = []
    current_time = 0.0

    for cr in sorted_clicks:
        zoom_start = max(0, cr.timestamp - half_dur)
        zoom_end = min(duration, cr.timestamp + half_dur)

        # Passthrough before this zoom
        if zoom_start > current_time + 0.05:
            segments.append({
                "type": "passthrough",
                "start": current_time,
                "duration": zoom_start - current_time,
            })

        # Zoom ramp in
        ramp_in_dur = (cr.timestamp - zoom_start) / ramp_steps if ramp_steps > 0 else 0
        for step in range(ramp_steps):
            t = (step + 1) / ramp_steps
            eased = _ease_in_out(t * 0.5)  # first half of ease
            z = 1.0 + (zoom_factor - 1.0) * eased
            seg_start = zoom_start + step * ramp_in_dur
            segments.append({
                "type": "zoom",
                "start": seg_start,
                "duration": max(0.05, ramp_in_dur),
                "zoom": z,
                "cx": cr.x,
                "cy": cr.y,
            })

        # Zoom ramp out
        ramp_out_dur = (zoom_end - cr.timestamp) / ramp_steps if ramp_steps > 0 else 0
        for step in range(ramp_steps):
            t = (step + 1) / ramp_steps
            eased = _ease_in_out(0.5 + t * 0.5)  # second half of ease
            z = 1.0 + (zoom_factor - 1.0) * (1.0 - eased)
            seg_start = cr.timestamp + step * ramp_out_dur
            segments.append({
                "type": "zoom",
                "start": seg_start,
                "duration": max(0.05, ramp_out_dur),
                "zoom": max(1.01, z),
                "cx": cr.x,
                "cy": cr.y,
            })

        current_time = zoom_end

    # Final passthrough after last zoom
    if current_time < duration - 0.05:
        segments.append({
            "type": "passthrough",
            "start": current_time,
            "duration": duration - current_time,
        })

    return segments
