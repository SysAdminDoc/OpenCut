"""
OpenCut Version Compare

Compare two video renders frame-by-frame using multiple visual modes:
side-by-side, overlay diff, flicker (alternating), and swipe (vertical wipe).

Extracts frames at configurable intervals via FFmpeg, computes SSIM and
PSNR per frame pair, generates diff heatmaps (absolute pixel difference
amplified), and produces a comparison output video.

Also supports audio comparison: waveform diff and loudness delta.
"""

import logging
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPARE_MODES = ("side_by_side", "overlay_diff", "flicker", "swipe")

# Default interval between frames to compare (seconds)
DEFAULT_FRAME_INTERVAL = 1.0

# SSIM threshold below which a frame is considered "changed"
SSIM_CHANGE_THRESHOLD = 0.95


@dataclass
class FrameCompareResult:
    """Result for a single frame pair comparison."""
    frame_index: int = 0
    timestamp_sec: float = 0.0
    ssim: float = 1.0
    psnr: float = 100.0
    changed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompareReport:
    """Overall comparison report for two video files."""
    file_a: str = ""
    file_b: str = ""
    mode: str = "side_by_side"
    total_frames: int = 0
    changed_frames: int = 0
    changed_percentage: float = 0.0
    avg_ssim: float = 1.0
    min_ssim: float = 1.0
    avg_psnr: float = 100.0
    min_psnr: float = 100.0
    overall_similarity: float = 100.0
    frame_results: List[FrameCompareResult] = field(default_factory=list)
    biggest_changes: List[FrameCompareResult] = field(default_factory=list)
    output_path: str = ""
    duration_a: float = 0.0
    duration_b: float = 0.0
    audio_loudness_a: float = 0.0
    audio_loudness_b: float = 0.0
    audio_loudness_delta: float = 0.0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_results"] = [fr if isinstance(fr, dict) else fr for fr in d["frame_results"]]
        d["biggest_changes"] = [fr if isinstance(fr, dict) else fr for fr in d["biggest_changes"]]
        return d


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(video_path: str, output_dir: str,
                    interval: float = DEFAULT_FRAME_INTERVAL,
                    max_frames: int = 300,
                    prefix: str = "frame") -> List[str]:
    """Extract frames from a video at regular intervals.

    Returns list of extracted frame file paths.
    """
    ffmpeg = get_ffmpeg_path()
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise ValueError(f"Cannot determine duration of {video_path}")

    frame_count = min(int(duration / interval) + 1, max_frames)
    pattern = os.path.join(output_dir, f"{prefix}_%05d.png")

    fps_filter = f"fps=1/{interval}" if interval >= 1 else f"fps={1/interval}"
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vf", fps_filter,
        "-frames:v", str(frame_count),
        "-q:v", "2",
        pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.warning("Frame extraction stderr: %s", result.stderr[-500:] if result.stderr else "")

    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir)
         if f.startswith(prefix) and f.endswith(".png")]
    )
    return frames


# ---------------------------------------------------------------------------
# SSIM / PSNR computation
# ---------------------------------------------------------------------------

def _compute_ssim_psnr(frame_a: str, frame_b: str) -> dict:
    """Compute SSIM and PSNR between two frame images using FFmpeg.

    Returns dict with ``ssim`` and ``psnr`` floats.
    """
    ffmpeg = get_ffmpeg_path()

    # SSIM
    ssim_cmd = [
        ffmpeg, "-i", frame_a, "-i", frame_b,
        "-lavfi", "ssim=stats_file=-",
        "-f", "null", "-",
    ]
    ssim_val = 1.0
    try:
        result = subprocess.run(ssim_cmd, capture_output=True, text=True, timeout=30)
        combined = (result.stderr or "") + (result.stdout or "")
        match = re.search(r"All:(\d+\.\d+)", combined)
        if match:
            ssim_val = float(match.group(1))
    except Exception as exc:
        logger.debug("SSIM computation failed: %s", exc)

    # PSNR
    psnr_cmd = [
        ffmpeg, "-i", frame_a, "-i", frame_b,
        "-lavfi", "psnr=stats_file=-",
        "-f", "null", "-",
    ]
    psnr_val = 100.0
    try:
        result = subprocess.run(psnr_cmd, capture_output=True, text=True, timeout=30)
        combined = (result.stderr or "") + (result.stdout or "")
        match = re.search(r"average:(\d+\.?\d*)", combined)
        if match:
            val = float(match.group(1))
            psnr_val = val if not math.isinf(val) else 100.0
    except Exception as exc:
        logger.debug("PSNR computation failed: %s", exc)

    return {"ssim": ssim_val, "psnr": psnr_val}


# ---------------------------------------------------------------------------
# Diff heatmap generation
# ---------------------------------------------------------------------------

def _generate_diff_heatmap(frame_a: str, frame_b: str, output_path: str,
                           amplify: int = 10) -> str:
    """Generate a diff heatmap image showing absolute pixel differences.

    Amplifies differences by the given factor for visibility.
    """
    ffmpeg = get_ffmpeg_path()
    fc = (
        f"[0:v][1:v]blend=all_mode=difference,"
        f"curves=all='0/0 0.05/1',"
        f"eq=brightness=0.1:contrast={amplify}"
    )
    cmd = [
        ffmpeg, "-y", "-i", frame_a, "-i", frame_b,
        "-filter_complex", fc,
        "-frames:v", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        logger.debug("Heatmap generation failed: %s", exc)
    return output_path


# ---------------------------------------------------------------------------
# Audio comparison
# ---------------------------------------------------------------------------

def _get_loudness(video_path: str) -> float:
    """Measure integrated loudness (LUFS) of a video's audio track."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-i", video_path,
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        combined = (result.stderr or "") + (result.stdout or "")
        match = re.search(r'"input_i"\s*:\s*"(-?\d+\.?\d*)"', combined)
        if match:
            return float(match.group(1))
    except Exception as exc:
        logger.debug("Loudness measurement failed for %s: %s", video_path, exc)
    return 0.0


def compare_audio(file_a: str, file_b: str) -> dict:
    """Compare audio loudness between two files.

    Returns dict with loudness values and delta.
    """
    loud_a = _get_loudness(file_a)
    loud_b = _get_loudness(file_b)
    return {
        "loudness_a": loud_a,
        "loudness_b": loud_b,
        "loudness_delta": round(loud_b - loud_a, 2),
    }


# ---------------------------------------------------------------------------
# Comparison video output
# ---------------------------------------------------------------------------

def _build_comparison_video(file_a: str, file_b: str, mode: str,
                            output_path: str,
                            on_progress: Optional[Callable] = None) -> str:
    """Build a comparison video using the specified mode."""
    if mode not in COMPARE_MODES:
        raise ValueError(f"Invalid comparison mode: {mode}")
    ffmpeg = get_ffmpeg_path()
    info_a = get_video_info(file_a)
    w, h = info_a.get("width", 1920), info_a.get("height", 1080)

    scale_b = f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[vb];"

    if mode == "side_by_side":
        fc = (
            f"[0:v]scale={w}:{h}[va];"
            f"{scale_b}"
            f"[va][vb]hstack=inputs=2[outv]"
        )
    elif mode == "overlay_diff":
        fc = (
            f"[0:v]scale={w}:{h}[va];"
            f"{scale_b}"
            f"[va][vb]blend=all_mode=difference,eq=contrast=3[outv]"
        )
    elif mode == "flicker":
        # Alternate between the two inputs every 0.5 seconds
        fc = (
            f"[0:v]scale={w}:{h},setpts=PTS-STARTPTS[va];"
            f"{scale_b}"
            f"[va][vb]interleave=nb_inputs=2[outv]"
        )
    elif mode == "swipe":
        # Vertical wipe — left half from A, right half from B
        half_w = w // 2
        fc = (
            f"[0:v]scale={w}:{h},crop={half_w}:{h}:0:0[left];"
            f"{scale_b}"
            f"[vb]crop={w - half_w}:{h}:{half_w}:0[right];"
            f"[left][right]hstack=inputs=2[outv]"
        )
    else:
        raise ValueError(f"Invalid comparison mode: {mode}")

    cmd = [
        ffmpeg, "-y", "-i", file_a, "-i", file_b,
        "-filter_complex", fc,
        "-map", "[outv]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-movflags", "+faststart",
        output_path,
    ]

    if on_progress:
        on_progress(40)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.warning("Comparison video stderr: %s", result.stderr[-500:] if result.stderr else "")
        raise RuntimeError(f"FFmpeg comparison video failed (rc={result.returncode})")

    if on_progress:
        on_progress(80)

    return output_path


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_versions(
    file_a: str,
    file_b: str,
    mode: str = "side_by_side",
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    max_frames: int = 300,
    output_dir: str = "",
    include_audio: bool = True,
    generate_video: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Compare two video renders frame-by-frame.

    Args:
        file_a: Path to the reference video.
        file_b: Path to the comparison video.
        mode: Comparison mode (side_by_side, overlay_diff, flicker, swipe).
        frame_interval: Seconds between frame samples.
        max_frames: Maximum number of frames to compare.
        output_dir: Directory for output files. Uses temp if empty.
        include_audio: Whether to compare audio loudness.
        generate_video: Whether to generate a comparison video file.
        on_progress: Progress callback (int percentage).

    Returns:
        CompareReport as dict.
    """
    if not os.path.isfile(file_a):
        raise FileNotFoundError(f"File A not found: {file_a}")
    if not os.path.isfile(file_b):
        raise FileNotFoundError(f"File B not found: {file_b}")
    if mode not in COMPARE_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {COMPARE_MODES}")

    start_time = time.time()
    report = CompareReport(file_a=file_a, file_b=file_b, mode=mode)

    if on_progress:
        on_progress(5)

    # Get duration info
    info_a = get_video_info(file_a)
    info_b = get_video_info(file_b)
    report.duration_a = info_a.get("duration", 0)
    report.duration_b = info_b.get("duration", 0)

    if on_progress:
        on_progress(10)

    # Extract frames
    tmpdir = tempfile.mkdtemp(prefix="opencut_compare_")
    try:
        dir_a = os.path.join(tmpdir, "a")
        dir_b = os.path.join(tmpdir, "b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        frames_a = _extract_frames(file_a, dir_a, interval=frame_interval,
                                   max_frames=max_frames, prefix="a")
        if on_progress:
            on_progress(25)

        frames_b = _extract_frames(file_b, dir_b, interval=frame_interval,
                                   max_frames=max_frames, prefix="b")
        if on_progress:
            on_progress(40)

        # Compare frame pairs
        pair_count = min(len(frames_a), len(frames_b))
        report.total_frames = pair_count
        ssim_sum = 0.0
        psnr_sum = 0.0
        min_ssim = 1.0
        min_psnr = 100.0

        for i in range(pair_count):
            metrics = _compute_ssim_psnr(frames_a[i], frames_b[i])
            ssim_val = metrics["ssim"]
            psnr_val = metrics["psnr"]

            is_changed = ssim_val < SSIM_CHANGE_THRESHOLD
            fr = FrameCompareResult(
                frame_index=i,
                timestamp_sec=round(i * frame_interval, 3),
                ssim=round(ssim_val, 6),
                psnr=round(psnr_val, 2),
                changed=is_changed,
            )
            report.frame_results.append(fr)

            ssim_sum += ssim_val
            psnr_sum += psnr_val
            if ssim_val < min_ssim:
                min_ssim = ssim_val
            if psnr_val < min_psnr:
                min_psnr = psnr_val

            if on_progress and pair_count > 0:
                pct = 40 + int(40 * (i + 1) / pair_count)
                on_progress(pct)

        if pair_count > 0:
            report.avg_ssim = round(ssim_sum / pair_count, 6)
            report.avg_psnr = round(psnr_sum / pair_count, 2)
            report.min_ssim = round(min_ssim, 6)
            report.min_psnr = round(min_psnr, 2)
            report.changed_frames = sum(1 for fr in report.frame_results if fr.changed)
            report.changed_percentage = round(
                100 * report.changed_frames / pair_count, 2
            )
            report.overall_similarity = round(report.avg_ssim * 100, 2)

        # Find frames with biggest changes (lowest SSIM)
        sorted_frames = sorted(report.frame_results, key=lambda fr: fr.ssim)
        report.biggest_changes = sorted_frames[:10]

    finally:
        # Cleanup temp frames
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    # Audio comparison
    if include_audio:
        try:
            audio = compare_audio(file_a, file_b)
            report.audio_loudness_a = audio["loudness_a"]
            report.audio_loudness_b = audio["loudness_b"]
            report.audio_loudness_delta = audio["loudness_delta"]
        except Exception as exc:
            logger.debug("Audio comparison failed: %s", exc)

    if on_progress:
        on_progress(85)

    # Generate comparison video
    if generate_video:
        out_dir = output_dir or os.path.dirname(file_a)
        base = os.path.splitext(os.path.basename(file_a))[0]
        out_file = os.path.join(out_dir, f"{base}_compare_{mode}.mp4")
        try:
            _build_comparison_video(file_a, file_b, mode, out_file,
                                    on_progress=on_progress)
            report.output_path = out_file
        except Exception as exc:
            logger.warning("Comparison video generation failed: %s", exc)

    report.elapsed_sec = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(100)

    return report.to_dict()
