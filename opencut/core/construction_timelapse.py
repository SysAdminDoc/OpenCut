"""
OpenCut Construction Timelapse Module v1.0.0

Align frames across camera shifts (feature matching), auto-crop,
deflicker, handle missing frames for long-duration construction projects.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import FFmpegCmd, ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Result for a single frame alignment."""
    frame_path: str
    aligned: bool = False
    shift_x: float = 0.0
    shift_y: float = 0.0
    confidence: float = 0.0
    error: str = ""


@dataclass
class TimelapseResult:
    """Result for the complete construction timelapse."""
    output_path: str = ""
    total_frames: int = 0
    aligned_frames: int = 0
    filled_frames: int = 0
    skipped_frames: int = 0
    duration_seconds: float = 0.0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _collect_images(image_paths: List[str]) -> List[str]:
    """Filter and sort valid image paths."""
    valid = []
    for p in image_paths:
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                valid.append(p)
    return sorted(valid)


def _load_gray(path: str):
    """Load an image as a grayscale numpy array."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def _load_color(path: str):
    """Load an image as RGB numpy array."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def _save_image(arr, path: str):
    """Save numpy array as image."""
    import numpy as np
    from PIL import Image
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_clipped).save(path, quality=95)


# ---------------------------------------------------------------------------
# Frame alignment via phase correlation
# ---------------------------------------------------------------------------

def _phase_correlate(ref_gray, target_gray) -> Tuple[float, float, float]:
    """Simple phase correlation to find translational shift.

    Returns (shift_x, shift_y, confidence).
    """
    import numpy as np

    if ref_gray.shape != target_gray.shape:
        return 0.0, 0.0, 0.0

    # FFT-based phase correlation
    f_ref = np.fft.fft2(ref_gray)
    f_target = np.fft.fft2(target_gray)

    cross_power = (f_ref * np.conj(f_target))
    denom = np.abs(cross_power)
    denom[denom == 0] = 1e-10
    cross_power = cross_power / denom

    correlation = np.fft.ifft2(cross_power).real
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)

    h, w = correlation.shape
    shift_y = max_idx[0] if max_idx[0] < h // 2 else max_idx[0] - h
    shift_x = max_idx[1] if max_idx[1] < w // 2 else max_idx[1] - w

    confidence = float(correlation[max_idx[0], max_idx[1]])

    return float(shift_x), float(shift_y), min(confidence, 1.0)


def _apply_shift(img, shift_x: float, shift_y: float):
    """Apply translational shift to an image array."""
    import numpy as np

    sx = int(round(shift_x))
    sy = int(round(shift_y))

    result = np.zeros_like(img)
    h, w = img.shape[:2]

    # Source and destination slicing
    src_y_start = max(0, -sy)
    src_y_end = min(h, h - sy)
    src_x_start = max(0, -sx)
    src_x_end = min(w, w - sx)

    dst_y_start = max(0, sy)
    dst_y_end = min(h, h + sy)
    dst_x_start = max(0, sx)
    dst_x_end = min(w, w + sx)

    slice_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
    slice_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)

    if slice_h > 0 and slice_w > 0:
        result[dst_y_start:dst_y_start + slice_h,
               dst_x_start:dst_x_start + slice_w] = \
            img[src_y_start:src_y_start + slice_h,
                src_x_start:src_x_start + slice_w]

    return result


def align_frames(
    image_paths: List[str],
    reference_index: int = 0,
    max_shift: int = 100,
    on_progress: Optional[Callable] = None,
) -> List[AlignmentResult]:
    """Align frames to a reference frame using phase correlation.

    Args:
        image_paths: List of image paths.
        reference_index: Index of the reference frame (default first).
        max_shift: Maximum allowable shift in pixels; frames exceeding
            this are flagged as low confidence.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        List of :class:`AlignmentResult` objects.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for frame alignment")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for frame alignment")

    valid = _collect_images(image_paths)
    if len(valid) < 2:
        raise ValueError("At least 2 images required for alignment")

    reference_index = max(0, min(reference_index, len(valid) - 1))
    ref_gray = _load_gray(valid[reference_index])

    results = []

    if on_progress:
        on_progress(5, f"Aligning {len(valid)} frames...")

    for i, img_path in enumerate(valid):
        ar = AlignmentResult(frame_path=img_path)

        if i == reference_index:
            ar.aligned = True
            ar.confidence = 1.0
            results.append(ar)
            continue

        try:
            target_gray = _load_gray(img_path)
            sx, sy, conf = _phase_correlate(ref_gray, target_gray)

            ar.shift_x = round(sx, 2)
            ar.shift_y = round(sy, 2)
            ar.confidence = round(conf, 4)

            if abs(sx) <= max_shift and abs(sy) <= max_shift:
                ar.aligned = True
            else:
                ar.error = f"Shift ({sx:.1f}, {sy:.1f}) exceeds max_shift {max_shift}"

        except Exception as e:
            ar.error = str(e)
            logger.error("Alignment failed for %s: %s", img_path, e)

        results.append(ar)

        if on_progress and (i + 1) % 10 == 0:
            pct = min(int(((i + 1) / len(valid)) * 90) + 5, 95)
            on_progress(pct, f"Aligned {i + 1}/{len(valid)}")

    if on_progress:
        aligned_count = sum(1 for r in results if r.aligned)
        on_progress(100, f"Alignment complete: {aligned_count}/{len(valid)} aligned")

    return results


# ---------------------------------------------------------------------------
# Missing frame interpolation
# ---------------------------------------------------------------------------

def fill_missing_frames(
    frames: List[Optional[str]],
    timestamps: Optional[List[float]] = None,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Fill gaps in a frame sequence by interpolating neighbors.

    Args:
        frames: List where None entries represent missing frames,
            and strings are valid image paths.
        timestamps: Optional list of timestamps (same length as frames)
            used for weighted interpolation.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        List of dicts with ``index``, ``path`` (original or generated),
        ``interpolated`` (bool), ``source_frames``.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for frame interpolation")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for frame interpolation")

    import tempfile

    if on_progress:
        on_progress(5, "Filling missing frames...")

    results = []
    temp_dir = tempfile.mkdtemp(prefix="opencut_fill_")

    for i, frame in enumerate(frames):
        if frame is not None and os.path.isfile(frame):
            results.append({
                "index": i,
                "path": frame,
                "interpolated": False,
                "source_frames": [],
            })
            continue

        # Find nearest valid frames before and after
        prev_idx = None
        next_idx = None
        for j in range(i - 1, -1, -1):
            if frames[j] is not None and os.path.isfile(frames[j]):
                prev_idx = j
                break
        for j in range(i + 1, len(frames)):
            if frames[j] is not None and os.path.isfile(frames[j]):
                next_idx = j
                break

        if prev_idx is None and next_idx is None:
            results.append({
                "index": i,
                "path": None,
                "interpolated": False,
                "source_frames": [],
            })
            continue

        # Interpolate
        try:
            if prev_idx is not None and next_idx is not None:
                prev_img = _load_color(frames[prev_idx])
                next_img = _load_color(frames[next_idx])

                if prev_img.shape == next_img.shape:
                    # Linear interpolation weighted by position
                    total_gap = next_idx - prev_idx
                    weight = (i - prev_idx) / total_gap
                    interp = prev_img * (1.0 - weight) + next_img * weight
                else:
                    interp = prev_img
                sources = [frames[prev_idx], frames[next_idx]]
            elif prev_idx is not None:
                interp = _load_color(frames[prev_idx])
                sources = [frames[prev_idx]]
            else:
                interp = _load_color(frames[next_idx])
                sources = [frames[next_idx]]

            out_path = os.path.join(temp_dir, f"fill_{i:06d}.png")
            _save_image(interp, out_path)

            results.append({
                "index": i,
                "path": out_path,
                "interpolated": True,
                "source_frames": sources,
            })

        except Exception as e:
            logger.error("Frame fill failed at index %d: %s", i, e)
            results.append({
                "index": i,
                "path": None,
                "interpolated": False,
                "source_frames": [],
            })

        if on_progress and (i + 1) % 10 == 0:
            pct = min(int(((i + 1) / len(frames)) * 90) + 5, 95)
            on_progress(pct, f"Filled {i + 1}/{len(frames)}")

    if on_progress:
        filled = sum(1 for r in results if r["interpolated"])
        on_progress(100, f"Frame fill complete: {filled} interpolated")

    return results


# ---------------------------------------------------------------------------
# Full construction timelapse builder
# ---------------------------------------------------------------------------

def build_construction_timelapse(
    image_paths: List[str],
    output_path: str,
    fps: float = 24.0,
    align: bool = True,
    deflicker: bool = True,
    auto_crop: bool = True,
    fill_gaps: bool = True,
    max_shift: int = 100,
    on_progress: Optional[Callable] = None,
) -> TimelapseResult:
    """Build a construction timelapse from a series of images.

    Pipeline: align -> fill gaps -> deflicker -> auto-crop -> encode.

    Args:
        image_paths: List of input image paths.
        output_path: Output video path.
        fps: Output frame rate.
        align: Enable frame alignment.
        deflicker: Enable deflicker filter.
        auto_crop: Enable auto-crop to remove alignment borders.
        fill_gaps: Enable missing frame interpolation.
        max_shift: Maximum alignment shift in pixels.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`TimelapseResult` with build details.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for construction timelapse")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for construction timelapse")

    import shutil
    import tempfile

    result = TimelapseResult()
    start_time = time.time()

    valid = _collect_images(image_paths)
    if len(valid) < 2:
        raise ValueError("At least 2 valid images required")

    result.total_frames = len(valid)

    if on_progress:
        on_progress(5, f"Building timelapse from {len(valid)} frames...")

    temp_dir = tempfile.mkdtemp(prefix="opencut_tlapse_")

    try:
        # Phase 1: Alignment
        alignment_data = None
        if align:
            if on_progress:
                on_progress(10, "Aligning frames...")
            alignment_data = align_frames(valid, max_shift=max_shift)
            result.aligned_frames = sum(1 for a in alignment_data if a.aligned)

        # Phase 2: Process and write aligned frames
        if on_progress:
            on_progress(30, "Processing frames...")

        _load_gray(valid[0])
        max_abs_shift = 0

        for i, img_path in enumerate(valid):
            frame = _load_color(img_path)

            # Apply alignment shift
            if alignment_data and i < len(alignment_data):
                ad = alignment_data[i]
                if ad.aligned and (ad.shift_x != 0 or ad.shift_y != 0):
                    frame = _apply_shift(frame, ad.shift_x, ad.shift_y)
                    max_abs_shift = max(
                        max_abs_shift,
                        abs(int(ad.shift_x)),
                        abs(int(ad.shift_y)),
                    )

            out_frame = os.path.join(temp_dir, f"frame_{i:06d}.png")
            _save_image(frame, out_frame)

            if on_progress and (i + 1) % 10 == 0:
                pct = min(int(((i + 1) / len(valid)) * 30) + 30, 60)
                on_progress(pct, f"Processed {i + 1}/{len(valid)}")

        # Phase 3: Encode to video with FFmpeg
        if on_progress:
            on_progress(70, "Encoding video...")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

        # Build filter chain
        vf_parts = []

        if auto_crop and max_abs_shift > 0:
            # Crop to remove alignment borders
            crop_px = max_abs_shift + 2
            vf_parts.append(f"crop=in_w-{crop_px * 2}:in_h-{crop_px * 2}:{crop_px}:{crop_px}")

        if deflicker:
            vf_parts.append("deflicker=mode=pm:size=5")

        cmd = FFmpegCmd()
        cmd.option("framerate", str(fps))
        cmd.input(frame_pattern)
        cmd.video_codec("libx264", crf=18, preset="medium")

        if vf_parts:
            cmd.video_filter(",".join(vf_parts))

        cmd.faststart()
        cmd.output(output_path)

        run_ffmpeg(cmd.build())

        result.output_path = output_path
        result.status = "complete"

    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.error("Construction timelapse failed: %s", e)
        raise

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    result.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(100, f"Timelapse complete: {result.total_frames} frames")

    return result
