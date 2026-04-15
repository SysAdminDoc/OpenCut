"""
OpenCut A/B Comparison — Side-by-Side Preview Data Generator

Given original and processed video files, generate comparison data for
the frontend: extract matched frame pairs at intervals, compute
per-frame quality metrics (SSIM, PSNR, colour delta), and generate
composite frames in various comparison modes.

Modes: side_by_side, overlay_blend, wipe_horizontal, wipe_vertical,
split_diagonal, checkerboard.
"""

import hashlib
import logging
import math
import os
import subprocess as _sp
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPARE_MODES = (
    "side_by_side",
    "overlay_blend",
    "wipe_horizontal",
    "wipe_vertical",
    "split_diagonal",
    "checkerboard",
)

DEFAULT_WIDTH = 854
DEFAULT_HEIGHT = 480
DEFAULT_INTERVAL = 1.0  # seconds between comparison frames


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FrameMetrics:
    """Quality metrics for a single frame pair."""
    timestamp: float = 0.0
    ssim: float = 0.0
    psnr: float = 0.0
    color_delta: float = 0.0
    mse: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompareFrame:
    """A single comparison composite frame."""
    timestamp: float = 0.0
    mode: str = ""
    composite_path: str = ""
    original_path: str = ""
    processed_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompareResult:
    """Result of an A/B comparison generation."""
    frames: List[CompareFrame] = field(default_factory=list)
    metrics: List[FrameMetrics] = field(default_factory=list)
    overall_ssim: float = 0.0
    overall_psnr: float = 0.0
    overall_color_delta: float = 0.0
    mode: str = ""
    frame_count: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frames"] = [f.to_dict() if hasattr(f, "to_dict") else f
                       for f in self.frames]
        d["metrics"] = [m.to_dict() if hasattr(m, "to_dict") else m
                        for m in self.metrics]
        return d


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------
def _extract_frame(video_path: str, timestamp: float,
                   output_path: str, width: int, height: int) -> str:
    """Extract a single frame at *timestamp* scaled to width x height."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",
        "-y", output_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Frame extraction failed at ts={timestamp}: {stderr}")
    return output_path


def _extract_frame_pair(
    original: str, processed: str, timestamp: float,
    output_dir: str, width: int, height: int, pair_id: str,
) -> Tuple[str, str]:
    """Extract matched frames from both videos."""
    orig_path = os.path.join(output_dir, f"orig_{pair_id}.jpg")
    proc_path = os.path.join(output_dir, f"proc_{pair_id}.jpg")
    _extract_frame(original, timestamp, orig_path, width, height)
    _extract_frame(processed, timestamp, proc_path, width, height)
    return orig_path, proc_path


# ---------------------------------------------------------------------------
# Quality metrics via PIL (numpy optional)
# ---------------------------------------------------------------------------
def _compute_metrics(orig_path: str, proc_path: str) -> FrameMetrics:
    """Compute SSIM, PSNR, MSE, and colour delta between two frame images.

    Uses PIL for pixel access.  Numpy is used for vectorised math when
    available; falls back to pure-Python otherwise.
    """
    try:
        from PIL import Image  # noqa: F811
    except ImportError:
        logger.warning("PIL not available; returning zero metrics")
        return FrameMetrics()

    try:
        img_a = Image.open(orig_path).convert("RGB")
        img_b = Image.open(proc_path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to open frames for metrics: %s", e)
        return FrameMetrics()

    # Ensure same size
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size)

    try:
        import numpy as np  # noqa: F811
        arr_a = np.asarray(img_a, dtype=np.float64)
        arr_b = np.asarray(img_b, dtype=np.float64)
        return _metrics_numpy(arr_a, arr_b)
    except ImportError:
        return _metrics_pure(img_a, img_b)


def _metrics_numpy(arr_a, arr_b) -> FrameMetrics:
    """Compute metrics using numpy arrays (H, W, 3) float64."""
    import numpy as np  # noqa: F811

    diff = arr_a - arr_b
    mse = float(np.mean(diff ** 2))

    # PSNR
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(10 * math.log10((255.0 ** 2) / mse))
    psnr = min(psnr, 100.0)

    # Simplified SSIM (luminance channel)
    gray_a = 0.299 * arr_a[:, :, 0] + 0.587 * arr_a[:, :, 1] + 0.114 * arr_a[:, :, 2]
    gray_b = 0.299 * arr_b[:, :, 0] + 0.587 * arr_b[:, :, 1] + 0.114 * arr_b[:, :, 2]

    mu_a = float(np.mean(gray_a))
    mu_b = float(np.mean(gray_b))
    sigma_a_sq = float(np.var(gray_a))
    sigma_b_sq = float(np.var(gray_b))
    sigma_ab = float(np.mean((gray_a - mu_a) * (gray_b - mu_b)))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    ssim_den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim = float(ssim_num / ssim_den) if ssim_den > 0 else 0.0
    ssim = max(0.0, min(1.0, ssim))

    # Colour delta (mean Euclidean distance in RGB)
    color_delta = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=2))))

    return FrameMetrics(
        ssim=round(ssim, 6),
        psnr=round(psnr, 2),
        mse=round(mse, 2),
        color_delta=round(color_delta, 4),
    )


def _metrics_pure(img_a, img_b) -> FrameMetrics:
    """Pure-Python metrics fallback (no numpy)."""
    pixels_a = list(img_a.getdata())
    pixels_b = list(img_b.getdata())
    n = len(pixels_a)
    if n == 0:
        return FrameMetrics()

    sum_sq = 0.0
    sum_color_dist = 0.0
    for pa, pb in zip(pixels_a, pixels_b):
        dr = pa[0] - pb[0]
        dg = pa[1] - pb[1]
        db = pa[2] - pb[2]
        sum_sq += (dr * dr + dg * dg + db * db) / 3.0
        sum_color_dist += math.sqrt(dr * dr + dg * dg + db * db)

    mse = sum_sq / n
    psnr = 10 * math.log10((255.0 ** 2) / mse) if mse > 1e-10 else 100.0
    psnr = min(psnr, 100.0)
    color_delta = sum_color_dist / n

    # Simplified SSIM estimate
    ssim = max(0.0, 1.0 - (mse / (255.0 ** 2)))

    return FrameMetrics(
        ssim=round(ssim, 6),
        psnr=round(psnr, 2),
        mse=round(mse, 2),
        color_delta=round(color_delta, 4),
    )


# ---------------------------------------------------------------------------
# Composite frame generation
# ---------------------------------------------------------------------------
def _composite_side_by_side(orig: str, proc: str, out: str,
                            width: int, height: int) -> str:
    """Generate side-by-side composite (original | processed)."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", orig, "-i", proc,
        "-filter_complex",
        f"[0:v]scale={width}:{height}[a];[1:v]scale={width}:{height}[b];[a][b]hstack=inputs=2[out]",
        "-map", "[out]",
        "-frames:v", "1", "-q:v", "2", "-y", out,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Side-by-side composite failed: {stderr}")
    return out


def _composite_overlay_blend(orig: str, proc: str, out: str,
                             width: int, height: int,
                             opacity: float = 0.5) -> str:
    """Generate blended overlay composite."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", orig, "-i", proc,
        "-filter_complex",
        f"[0:v]scale={width}:{height}[a];[1:v]scale={width}:{height}[b];"
        f"[a][b]blend=all_mode=overlay:all_opacity={opacity}[out]",
        "-map", "[out]",
        "-frames:v", "1", "-q:v", "2", "-y", out,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"Overlay blend failed: {stderr}")
    return out


def _composite_wipe(orig: str, proc: str, out: str,
                    width: int, height: int,
                    position: float = 0.5,
                    horizontal: bool = True) -> str:
    """Generate a wipe-transition composite (vertical or horizontal split)."""
    try:
        from PIL import Image  # noqa: F811
    except ImportError:
        # Fallback to side-by-side
        return _composite_side_by_side(orig, proc, out, width, height)

    img_a = Image.open(orig).convert("RGB").resize((width, height))
    img_b = Image.open(proc).convert("RGB").resize((width, height))
    composite = img_a.copy()

    pos = max(0.0, min(1.0, position))
    if horizontal:
        split = int(width * pos)
        region = img_b.crop((split, 0, width, height))
        composite.paste(region, (split, 0))
    else:
        split = int(height * pos)
        region = img_b.crop((0, split, width, height))
        composite.paste(region, (0, split))

    composite.save(out, "JPEG", quality=90)
    return out


def _composite_diagonal(orig: str, proc: str, out: str,
                        width: int, height: int) -> str:
    """Diagonal split composite."""
    try:
        from PIL import Image, ImageDraw  # noqa: F811
    except ImportError:
        return _composite_side_by_side(orig, proc, out, width, height)

    img_a = Image.open(orig).convert("RGB").resize((width, height))
    img_b = Image.open(proc).convert("RGB").resize((width, height))

    # Create mask: white triangle top-right
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(0, 0), (width, 0), (width, height)], fill=255)

    composite = Image.composite(img_b, img_a, mask)
    composite.save(out, "JPEG", quality=90)
    return out


def _composite_checkerboard(orig: str, proc: str, out: str,
                            width: int, height: int,
                            block_size: int = 64) -> str:
    """Checkerboard pattern composite."""
    try:
        from PIL import Image  # noqa: F811
    except ImportError:
        return _composite_side_by_side(orig, proc, out, width, height)

    img_a = Image.open(orig).convert("RGB").resize((width, height))
    img_b = Image.open(proc).convert("RGB").resize((width, height))

    pixels_a = img_a.load()
    pixels_b = img_b.load()
    composite = Image.new("RGB", (width, height))
    pixels_c = composite.load()

    for y in range(height):
        row_block = y // block_size
        for x in range(width):
            col_block = x // block_size
            if (row_block + col_block) % 2 == 0:
                pixels_c[x, y] = pixels_a[x, y]
            else:
                pixels_c[x, y] = pixels_b[x, y]

    composite.save(out, "JPEG", quality=90)
    return out


_COMPOSITE_FUNCS = {
    "side_by_side": _composite_side_by_side,
    "overlay_blend": _composite_overlay_blend,
    "wipe_horizontal": lambda o, p, out, w, h: _composite_wipe(o, p, out, w, h, horizontal=True),
    "wipe_vertical": lambda o, p, out, w, h: _composite_wipe(o, p, out, w, h, horizontal=False),
    "split_diagonal": _composite_diagonal,
    "checkerboard": _composite_checkerboard,
}


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
def generate_comparison(
    original_path: str,
    processed_path: str,
    mode: str = "side_by_side",
    timestamps: Optional[List[float]] = None,
    num_frames: int = 5,
    interval: float = DEFAULT_INTERVAL,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    wipe_position: float = 0.5,
    output_dir: str = "",
    compute_metrics: bool = True,
    on_progress: Optional[Callable] = None,
) -> CompareResult:
    """Generate A/B comparison frames and metrics.

    Args:
        original_path: Path to original (reference) video.
        processed_path: Path to processed video.
        mode: Comparison display mode.
        timestamps: Explicit timestamps to compare.
        num_frames: Number of frames if timestamps not given.
        interval: Seconds between frames if timestamps not given.
        width: Output frame width.
        height: Output frame height.
        wipe_position: Split position for wipe modes (0.0 - 1.0).
        output_dir: Directory for output files.
        compute_metrics: Whether to compute quality metrics.
        on_progress: Progress callback (percentage int).

    Returns:
        CompareResult with frames, metrics, and overall scores.
    """
    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"Original video not found: {original_path}")
    if not os.path.isfile(processed_path):
        raise FileNotFoundError(f"Processed video not found: {processed_path}")
    if mode not in COMPARE_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Available: {list(COMPARE_MODES)}")

    t0 = time.time()

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_compare_")
    os.makedirs(output_dir, exist_ok=True)

    # Determine timestamps
    if timestamps is None:
        info = get_video_info(original_path)
        duration = info.get("duration", 0)
        if duration <= 0:
            duration = num_frames * interval
        num_frames = max(1, min(num_frames, 100))
        timestamps = [i * interval for i in range(num_frames) if i * interval < duration]
        if not timestamps:
            timestamps = [0.0]

    if on_progress:
        on_progress(5)

    frames = []
    metrics_list = []
    total = len(timestamps)
    composite_fn = _COMPOSITE_FUNCS.get(mode, _composite_side_by_side)

    for idx, ts in enumerate(timestamps):
        pair_id = hashlib.sha256(
            f"{original_path}:{processed_path}:{ts}".encode()
        ).hexdigest()[:10]

        try:
            orig_frame, proc_frame = _extract_frame_pair(
                original_path, processed_path, ts,
                output_dir, width, height, pair_id,
            )

            # Compute metrics
            if compute_metrics:
                m = _compute_metrics(orig_frame, proc_frame)
                m.timestamp = ts
                metrics_list.append(m)

            # Generate composite
            comp_path = os.path.join(output_dir, f"compare_{mode}_{pair_id}.jpg")
            if mode in ("wipe_horizontal", "wipe_vertical"):
                horiz = mode == "wipe_horizontal"
                _composite_wipe(orig_frame, proc_frame, comp_path,
                                width, height, wipe_position, horiz)
            else:
                composite_fn(orig_frame, proc_frame, comp_path, width, height)

            frames.append(CompareFrame(
                timestamp=ts,
                mode=mode,
                composite_path=comp_path,
                original_path=orig_frame,
                processed_path=proc_frame,
            ))
        except Exception as e:
            logger.warning("Comparison failed at ts=%.2f: %s", ts, e)
            continue

        if on_progress:
            pct = int(10 + (idx + 1) / total * 85)
            on_progress(pct)

    # Compute overall metrics
    overall_ssim = 0.0
    overall_psnr = 0.0
    overall_color_delta = 0.0
    if metrics_list:
        overall_ssim = sum(m.ssim for m in metrics_list) / len(metrics_list)
        overall_psnr = sum(m.psnr for m in metrics_list) / len(metrics_list)
        overall_color_delta = sum(m.color_delta for m in metrics_list) / len(metrics_list)

    elapsed = (time.time() - t0) * 1000
    if on_progress:
        on_progress(100)

    return CompareResult(
        frames=frames,
        metrics=metrics_list,
        overall_ssim=round(overall_ssim, 6),
        overall_psnr=round(overall_psnr, 2),
        overall_color_delta=round(overall_color_delta, 4),
        mode=mode,
        frame_count=len(frames),
        processing_time_ms=round(elapsed, 1),
    )


def generate_wipe_frame(
    original_path: str,
    processed_path: str,
    timestamp: float = 0.0,
    position: float = 0.5,
    horizontal: bool = True,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    output_dir: str = "",
) -> dict:
    """Generate a single wipe-transition comparison frame.

    Intended for interactive scrubbing where the frontend sends the
    wipe position as the user drags a slider.
    """
    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"Original not found: {original_path}")
    if not os.path.isfile(processed_path):
        raise FileNotFoundError(f"Processed not found: {processed_path}")

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_wipe_")
    os.makedirs(output_dir, exist_ok=True)

    pair_id = hashlib.sha256(
        f"{original_path}:{processed_path}:{timestamp}:{position}".encode()
    ).hexdigest()[:10]

    orig_frame = os.path.join(output_dir, f"wipe_orig_{pair_id}.jpg")
    proc_frame = os.path.join(output_dir, f"wipe_proc_{pair_id}.jpg")
    _extract_frame(original_path, timestamp, orig_frame, width, height)
    _extract_frame(processed_path, timestamp, proc_frame, width, height)

    comp_path = os.path.join(output_dir, f"wipe_comp_{pair_id}.jpg")
    _composite_wipe(orig_frame, proc_frame, comp_path, width, height,
                    position, horizontal)

    return {
        "composite_path": comp_path,
        "original_path": orig_frame,
        "processed_path": proc_frame,
        "timestamp": timestamp,
        "wipe_position": position,
        "horizontal": horizontal,
    }


def get_compare_metrics(
    original_path: str,
    processed_path: str,
    timestamps: Optional[List[float]] = None,
    num_frames: int = 10,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Compute quality metrics between two videos without generating composites.

    Returns dict with per-frame and overall metrics.
    """
    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"Original not found: {original_path}")
    if not os.path.isfile(processed_path):
        raise FileNotFoundError(f"Processed not found: {processed_path}")

    output_dir = tempfile.mkdtemp(prefix="opencut_metrics_")

    if timestamps is None:
        info = get_video_info(original_path)
        duration = info.get("duration", 0)
        if duration <= 0:
            duration = 10.0
        num_frames = max(1, min(num_frames, 100))
        step = duration / num_frames
        timestamps = [i * step for i in range(num_frames)]

    metrics_list = []
    total = len(timestamps)

    for idx, ts in enumerate(timestamps):
        pair_id = hashlib.sha256(
            f"metrics:{original_path}:{processed_path}:{ts}".encode()
        ).hexdigest()[:10]
        try:
            orig_frame, proc_frame = _extract_frame_pair(
                original_path, processed_path, ts,
                output_dir, width, height, pair_id,
            )
            m = _compute_metrics(orig_frame, proc_frame)
            m.timestamp = ts
            metrics_list.append(m)

            # Clean up extracted frames
            for p in (orig_frame, proc_frame):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Metrics extraction failed at ts=%.2f: %s", ts, e)

        if on_progress:
            pct = int((idx + 1) / total * 100)
            on_progress(pct)

    # Clean temp dir
    try:
        os.rmdir(output_dir)
    except OSError:
        pass

    overall_ssim = 0.0
    overall_psnr = 0.0
    overall_color_delta = 0.0
    if metrics_list:
        overall_ssim = sum(m.ssim for m in metrics_list) / len(metrics_list)
        overall_psnr = sum(m.psnr for m in metrics_list) / len(metrics_list)
        overall_color_delta = sum(m.color_delta for m in metrics_list) / len(metrics_list)

    return {
        "frame_metrics": [m.to_dict() for m in metrics_list],
        "overall_ssim": round(overall_ssim, 6),
        "overall_psnr": round(overall_psnr, 2),
        "overall_color_delta": round(overall_color_delta, 4),
        "frame_count": len(metrics_list),
    }


def list_compare_modes() -> List[dict]:
    """Return available comparison modes with descriptions."""
    descs = {
        "side_by_side": "Original and processed shown side by side",
        "overlay_blend": "Alpha-blended overlay of both frames",
        "wipe_horizontal": "Horizontal wipe split at configurable position",
        "wipe_vertical": "Vertical wipe split at configurable position",
        "split_diagonal": "Diagonal split from corner to corner",
        "checkerboard": "Alternating blocks from each source",
    }
    return [{"id": m, "name": m.replace("_", " ").title(),
             "description": descs.get(m, "")} for m in COMPARE_MODES]
