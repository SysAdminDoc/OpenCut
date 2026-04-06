"""
Color Matching

Matches the color profile of a source clip to a reference clip
using histogram matching on each YCbCr channel.
"""

import logging
import os
import tempfile
from typing import List

try:
    from ..helpers import get_ffmpeg_path, run_ffmpeg
except ImportError:
    try:
        from opencut.helpers import get_ffmpeg_path, run_ffmpeg
    except ImportError:
        run_ffmpeg = None  # type: ignore
        get_ffmpeg_path = None  # type: ignore

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False
    logger.debug("opencv-python not installed; color_match features unavailable. "
                 "Install with: pip install opencv-python")

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NP_AVAILABLE = False
    logger.debug("numpy not installed; color_match features unavailable. "
                 "Install with: pip install numpy")


def _require_cv2():
    if not _CV2_AVAILABLE or not _NP_AVAILABLE:
        raise RuntimeError(
            "opencv-python and numpy are required for color matching. "
            "Install with: pip install opencv-python numpy"
        )


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _build_cumulative_hist(frames: list, channel: int, bins: int = 256) -> "np.ndarray":
    """
    Build a normalised cumulative histogram for one channel across multiple frames.

    Args:
        frames: List of YCbCr frames (numpy arrays, shape HxWx3, uint8).
        channel: Channel index (0=Y, 1=Cb, 2=Cr).
        bins: Number of histogram bins.

    Returns:
        Normalised CDF array of shape (bins,).
    """
    hist = np.zeros(bins, dtype=np.float64)
    for frame in frames:
        ch = frame[:, :, channel].ravel()
        h, _ = np.histogram(ch, bins=bins, range=(0, 256))
        hist += h.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    cdf = np.cumsum(hist)
    return cdf


def _build_lut(src_cdf: "np.ndarray", ref_cdf: "np.ndarray") -> "np.ndarray":
    """
    Build a lookup table mapping source pixel values to matched values.

    Uses histogram specification (matching src CDF to ref CDF).

    Returns:
        uint8 LUT of shape (256,).
    """
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
            ref_idx += 1
        lut[src_val] = ref_idx
    return lut


def _apply_luts(frame_ycbcr: "np.ndarray", luts: List["np.ndarray"]) -> "np.ndarray":
    """Apply per-channel LUTs to a YCbCr frame, return modified frame."""
    result = frame_ycbcr.copy()
    for ch, lut in enumerate(luts):
        result[:, :, ch] = cv2.LUT(frame_ycbcr[:, :, ch], lut)
    return result


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str, n: int) -> List["np.ndarray"]:
    """
    Extract N frames evenly distributed across a video.

    Returns:
        List of BGR frames (numpy uint8 arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
    n = max(1, min(n, total))

    positions = [int(i * total / n) for i in range(n)]
    frames = []
    try:
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# Color stats
# ---------------------------------------------------------------------------

def extract_color_stats(video_path: str, sample_frames: int = 5) -> dict:
    """
    Extract mean, std, and histogram stats for a video's YCbCr channels.

    Args:
        video_path: Path to the video file.
        sample_frames: Number of frames to sample.

    Returns:
        Dict with "mean" [Y, Cb, Cr], "std" [Y, Cb, Cr], "histogram" per channel.

    Raises:
        RuntimeError: If cv2/numpy are not installed or video cannot be opened.
    """
    _require_cv2()

    bgr_frames = _sample_frames(video_path, sample_frames)
    if not bgr_frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    ycbcr_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in bgr_frames]

    channel_names = ["Y", "Cr", "Cb"]
    mean_vals = []
    std_vals = []
    histograms = {}

    for ch, name in enumerate(channel_names):
        data = np.concatenate([f[:, :, ch].ravel() for f in ycbcr_frames])
        mean_vals.append(float(np.mean(data)))
        std_vals.append(float(np.std(data)))
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        histograms[name] = hist.tolist()

    return {
        "mean": mean_vals,
        "std": std_vals,
        "histogram": histograms,
    }


# ---------------------------------------------------------------------------
# Main color match function
# ---------------------------------------------------------------------------

def color_match_video(
    source_path: str,
    reference_path: str,
    output_path: str,
    strength: float = 1.0,
    sample_frame_count: int = 5,
    on_progress=None,
) -> str:
    """
    Match the color profile of a source video to a reference video.

    Uses histogram matching on each YCbCr channel (Y=luma, Cb/Cr=chroma).
    Audio from the source is preserved unchanged.

    Args:
        source_path: Path to the source (input) video.
        reference_path: Path to the reference video whose look to match.
        output_path: Path for the output video.
        strength: Blend between original (0.0) and fully matched (1.0) look.
        sample_frame_count: Frames to sample from reference for histogram building.

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If cv2/numpy are not installed, videos cannot be opened,
                      or FFmpeg fails during audio merge.
    """
    _require_cv2()

    strength = max(0.0, min(1.0, float(strength)))

    logger.debug("Building reference color histogram from %s (%d frames)",
                 reference_path, sample_frame_count)

    # Sample frames from reference and convert to YCbCr (release BGR immediately)
    ref_bgr = _sample_frames(reference_path, sample_frame_count)
    if not ref_bgr:
        raise RuntimeError(f"Could not extract frames from reference: {reference_path}")
    ref_ycbcr = [cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in ref_bgr]
    del ref_bgr  # Free ~31MB per 5 frames at 1080p

    # Also sample source for CDF building (release BGR immediately)
    src_bgr_samples = _sample_frames(source_path, sample_frame_count)
    if not src_bgr_samples:
        raise RuntimeError(f"Could not extract frames from source: {source_path}")
    src_ycbcr_samples = [cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb) for f in src_bgr_samples]
    del src_bgr_samples  # Free ~31MB

    # Build LUTs for each channel
    luts = []
    for ch in range(3):  # Y, Cb, Cr
        src_cdf = _build_cumulative_hist(src_ycbcr_samples, ch)
        ref_cdf = _build_cumulative_hist(ref_ycbcr, ch)
        luts.append(_build_lut(src_cdf, ref_cdf))

    # Free histogram source frames — only LUTs needed from here
    del ref_ycbcr, src_ycbcr_samples

    logger.debug("Processing source video frame-by-frame: %s", source_path)

    # Open source for frame-by-frame processing
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to temp video (no audio yet)
    fd, temp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(temp_video)
        except OSError:
            pass
        raise RuntimeError("Failed to create VideoWriter for color match output")

    try:
        frame_count = 0
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            ycbcr = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
            matched_ycbcr = _apply_luts(ycbcr, luts)
            matched_bgr = cv2.cvtColor(matched_ycbcr, cv2.COLOR_YCrCb2BGR)

            if strength < 1.0:
                matched_bgr = cv2.addWeighted(
                    bgr_frame, 1.0 - strength,
                    matched_bgr, strength,
                    0,
                )

            writer.write(matched_bgr)
            frame_count += 1

            if frame_count % 100 == 0:
                pct = int(frame_count / max(total_frames, 1) * 100)
                logger.debug("Color match progress: %d%%", pct)

            if on_progress and total_frames > 0:
                if frame_count % 50 == 0 or frame_count % max(1, int(total_frames / 10)) == 0:
                    pct = int(frame_count / total_frames * 100)
                    on_progress(pct)

        if on_progress:
            on_progress(100)
    finally:
        cap.release()
        writer.release()

    logger.info("Merging audio from source into color-matched output")

    # Merge original audio from source using FFmpeg
    _ffmpeg = get_ffmpeg_path() if get_ffmpeg_path else "ffmpeg"
    cmd = [
        _ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", temp_video,
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "1:a?",   # copy audio if present, silently skip if not
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        if run_ffmpeg is not None:
            # run_ffmpeg raises RuntimeError on non-zero exit, returns stderr str
            run_ffmpeg(cmd, timeout=3600)
        else:
            import subprocess as _sp
            result = _sp.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                stderr = result.stderr[-500:] if result.stderr else "unknown error"
                raise RuntimeError(f"FFmpeg audio merge failed: {stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
    finally:
        try:
            os.unlink(temp_video)
        except OSError:
            pass

    logger.info("Color match complete: %s", output_path)
    return output_path
