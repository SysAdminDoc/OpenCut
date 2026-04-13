"""
OpenCut Object Effects Module v1.0.0

Pika-style physics-simulated effects on selected objects:
- squish:      vertical compress (bounce)
- melt:        downward displacement (drip)
- inflate:     scale from centre outward
- explode:     radial scatter with particles
- dissolve:    alpha fade with noise
- crystallize: Voronoi tessellation (scipy) or grid pixelation fallback

Pipeline: user clicks object -> SAM2 (or threshold fallback) generates mask
          -> effect applied per-frame via FFmpeg filters or Pillow rendering.
"""

import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Effect type registry
# ---------------------------------------------------------------------------
EFFECT_TYPES: Dict[str, str] = {
    "squish": "Vertical compress / bounce effect",
    "melt": "Downward displacement / drip effect",
    "inflate": "Scale outward from centre",
    "explode": "Radial scatter with particles",
    "dissolve": "Alpha fade with procedural noise",
    "crystallize": "Voronoi tessellation / mosaic",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectMask:
    """Per-frame binary masks and bounding boxes for a segmented object."""
    mask_frames: List  # list of numpy arrays or file paths (str)
    bbox_per_frame: List[Tuple[int, int, int, int]]  # (x, y, w, h)


@dataclass
class EffectConfig:
    """Configuration for an object effect."""
    effect_type: str = "squish"
    intensity: float = 0.7
    duration: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.effect_type not in EFFECT_TYPES:
            raise ValueError(
                f"Unknown effect type '{self.effect_type}'. "
                f"Valid types: {', '.join(sorted(EFFECT_TYPES))}"
            )
        self.intensity = max(0.0, min(1.0, float(self.intensity)))
        self.duration = max(0.1, float(self.duration))
        self.seed = int(self.seed)


@dataclass
class ObjectEffectResult:
    """Result of applying an object effect."""
    output_path: str
    frames_processed: int
    effect_applied: str


# ---------------------------------------------------------------------------
# Mask generation (SAM2 with threshold fallback)
# ---------------------------------------------------------------------------

def generate_effect_mask(
    video_path: str,
    click_point: Tuple[int, int],
    num_frames: int = 30,
    on_progress: Optional[Callable] = None,
) -> ObjectMask:
    """
    Generate an object mask from a click point on the first frame.

    Tries SAM2 segmentation if available, otherwise uses a simple
    colour-threshold fallback that flood-fills from the click point.

    Returns an ObjectMask with per-frame mask arrays and bounding boxes.
    """
    try:
        from opencut.core.object_removal import check_sam2_available
        if check_sam2_available():
            return _generate_mask_sam2(video_path, click_point, num_frames, on_progress)
    except ImportError:
        pass

    return _generate_mask_threshold(video_path, click_point, num_frames, on_progress)


def _generate_mask_sam2(
    video_path: str,
    click_point: Tuple[int, int],
    num_frames: int,
    on_progress: Optional[Callable],
) -> ObjectMask:
    """Generate mask via SAM2 point prompt."""
    from opencut.core.object_removal import generate_masks_sam2

    if on_progress:
        on_progress(5, "Running SAM2 segmentation...")

    tmp_dir = tempfile.mkdtemp(prefix="objfx_masks_")
    result = generate_masks_sam2(
        video_path,
        prompts=[{"type": "point", "x": click_point[0], "y": click_point[1], "label": 1}],
        output_dir=tmp_dir,
        model_size="tiny",
        on_progress=on_progress,
    )

    mask_dir = result["mask_dir"]
    mask_files = sorted(
        f for f in os.listdir(mask_dir) if f.endswith(".png")
    )[:num_frames]

    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("OpenCV required for mask processing")
    import cv2
    import numpy as np

    mask_frames = []
    bbox_per_frame = []
    for mf in mask_files:
        mask = cv2.imread(os.path.join(mask_dir, mf), cv2.IMREAD_GRAYSCALE)
        mask_frames.append(mask)
        coords = np.where(mask > 127)
        if len(coords[0]) > 0:
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            bbox_per_frame.append((x_min, y_min, x_max - x_min, y_max - y_min))
        else:
            bbox_per_frame.append((0, 0, 0, 0))

    return ObjectMask(mask_frames=mask_frames, bbox_per_frame=bbox_per_frame)


def _generate_mask_threshold(
    video_path: str,
    click_point: Tuple[int, int],
    num_frames: int,
    on_progress: Optional[Callable],
) -> ObjectMask:
    """
    Fallback mask generation using colour-threshold flood fill.

    Reads the first frame, samples the colour at click_point,
    and creates a mask of pixels within a tolerance.  The same mask
    is duplicated for all requested frames (static object assumption).
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError(
            "OpenCV is required for mask generation. "
            "Install with: pip install opencv-python-headless"
        )
    import cv2
    import numpy as np

    if on_progress:
        on_progress(10, "Extracting first frame for mask...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    h, w = frame.shape[:2]
    cx, cy = int(click_point[0]), int(click_point[1])
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    # Flood fill from click point
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    seed_colour = hsv[cy, cx].astype(int)

    tolerance = 30
    lower = np.clip(seed_colour - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(seed_colour + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if on_progress:
        on_progress(50, "Mask generated via threshold fallback")

    # Find bounding box
    coords = np.where(mask > 127)
    if len(coords[0]) > 0:
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    else:
        # Fallback: small region around click
        radius = 50
        bbox = (max(0, cx - radius), max(0, cy - radius), radius * 2, radius * 2)
        mask[max(0, cy - radius):min(h, cy + radius),
             max(0, cx - radius):min(w, cx + radius)] = 255

    mask_frames = [mask.copy() for _ in range(num_frames)]
    bbox_per_frame = [bbox] * num_frames

    if on_progress:
        on_progress(100, "Mask generation complete")

    return ObjectMask(mask_frames=mask_frames, bbox_per_frame=bbox_per_frame)


# ---------------------------------------------------------------------------
# Effect implementations
# ---------------------------------------------------------------------------

def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out curve (0->1)."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _apply_squish(frame, mask, bbox, progress, intensity, rng):
    """Vertical compress effect on masked region."""
    import numpy as np

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()
    squeeze = _ease_in_out(progress) * intensity
    new_h = max(1, int(bh * (1.0 - squeeze * 0.6)))
    y_offset = bh - new_h  # shift down as it squishes

    # Extract and resize the masked region
    region = frame[y:y + bh, x:x + bw].copy()
    mask_region = mask[y:y + bh, x:x + bw]

    import cv2
    squished = cv2.resize(region, (bw, new_h), interpolation=cv2.INTER_LINEAR)
    mask_squished = cv2.resize(mask_region, (bw, new_h), interpolation=cv2.INTER_NEAREST)

    # Paste back at offset position
    paste_y = min(y + y_offset, h - new_h)
    paste_y = max(0, paste_y)
    paste_h = min(new_h, h - paste_y)
    paste_w = min(bw, w - x)

    alpha = (mask_squished[:paste_h, :paste_w] > 127).astype(np.float32)
    if len(frame.shape) == 3:
        alpha = alpha[:, :, None]

    result[paste_y:paste_y + paste_h, x:x + paste_w] = (
        squished[:paste_h, :paste_w] * alpha +
        result[paste_y:paste_y + paste_h, x:x + paste_w] * (1.0 - alpha)
    ).astype(np.uint8)

    return result


def _apply_melt(frame, mask, bbox, progress, intensity, rng):
    """Downward displacement / drip effect."""

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()
    max_shift = int(bh * intensity * progress)

    for col_offset in range(bw):
        col_x = x + col_offset
        if col_x >= w:
            break
        # Sinusoidal variation per column for drip effect
        col_shift = int(max_shift * (0.5 + 0.5 * math.sin(col_offset * 0.15 + rng.random() * 0.5)))
        for row in range(min(bh, h - y)):
            src_y = y + row
            dst_y = min(src_y + col_shift, h - 1)
            if src_y < h and mask[src_y, col_x] > 127:
                result[dst_y, col_x] = frame[src_y, col_x]

    return result


def _apply_inflate(frame, mask, bbox, progress, intensity, rng):
    """Scale from centre outward."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()
    scale = 1.0 + _ease_in_out(progress) * intensity * 0.8

    cx = x + bw // 2
    cy = y + bh // 2
    new_w = int(bw * scale)
    new_h = int(bh * scale)

    region = frame[y:y + bh, x:x + bw].copy()
    mask_region = mask[y:y + bh, x:x + bw]

    scaled = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask_region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Compute paste coordinates (centred)
    px = max(0, cx - new_w // 2)
    py = max(0, cy - new_h // 2)
    pw = min(new_w, w - px)
    ph = min(new_h, h - py)
    # Source offsets if paste starts at edge
    sx = max(0, new_w // 2 - cx)
    sy = max(0, new_h // 2 - cy)

    alpha = (mask_scaled[sy:sy + ph, sx:sx + pw] > 127).astype(np.float32)
    if len(frame.shape) == 3:
        alpha = alpha[:, :, None]

    result[py:py + ph, px:px + pw] = (
        scaled[sy:sy + ph, sx:sx + pw] * alpha +
        result[py:py + ph, px:px + pw] * (1.0 - alpha)
    ).astype(np.uint8)

    return result


def _apply_explode(frame, mask, bbox, progress, intensity, rng):
    """Radial scatter with particles using Pillow."""
    import numpy as np

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()
    cx = x + bw // 2
    cy = y + bh // 2

    # Clear the masked region with progress
    alpha_clear = min(1.0, progress * intensity * 2.0)
    mask_region = mask[y:y + bh, x:x + bw]
    clear_mask = (mask_region > 127).astype(np.float32) * alpha_clear
    if len(frame.shape) == 3:
        clear_mask = clear_mask[:, :, None]
    bg = frame[y:y + bh, x:x + bw].copy()
    # Darken to simulate explosion
    darkened = (bg * 0.3).astype(np.uint8)
    result[y:y + bh, x:x + bw] = (
        darkened * clear_mask + bg * (1.0 - clear_mask)
    ).astype(np.uint8)

    # Draw particles radiating outward
    num_particles = int(40 * intensity)
    max_radius = int(max(bw, bh) * progress * intensity)

    for _ in range(num_particles):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.2, 1.0) * max_radius
        px = int(cx + math.cos(angle) * dist)
        py_pos = int(cy + math.sin(angle) * dist)
        size = rng.randint(1, max(2, int(4 * intensity)))

        if 0 <= px < w and 0 <= py_pos < h:
            # Sample colour from original frame at source
            src_x = min(max(x, px), x + bw - 1)
            src_y = min(max(y, py_pos), y + bh - 1)
            if src_x < w and src_y < h:
                colour = frame[src_y, src_x].tolist()
                y1, y2 = max(0, py_pos - size), min(h, py_pos + size)
                x1, x2 = max(0, px - size), min(w, px + size)
                result[y1:y2, x1:x2] = colour

    return result


def _apply_dissolve(frame, mask, bbox, progress, intensity, rng):
    """Alpha fade with procedural noise mask."""
    import numpy as np

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()

    # Generate noise pattern (deterministic per seed)
    noise = np.random.RandomState(rng.randint(0, 2**31)).rand(bh, bw)
    threshold = progress * intensity
    dissolve_mask = (noise < threshold).astype(np.float32)

    # Combine with object mask
    obj_mask = (mask[y:y + bh, x:x + bw] > 127).astype(np.float32)
    combined = dissolve_mask * obj_mask

    if len(frame.shape) == 3:
        combined = combined[:, :, None]

    # Blend to transparent (use background underneath)
    bg = frame[y:y + bh, x:x + bw].copy()
    # Fade to slightly noisy background
    noise_colour = (np.random.RandomState(rng.randint(0, 2**31)).rand(bh, bw, 3) * 30).astype(np.uint8)
    faded = np.clip(bg.astype(np.int16) - 80, 0, 255).astype(np.uint8) + noise_colour

    result[y:y + bh, x:x + bw] = (
        faded * combined + bg * (1.0 - combined)
    ).astype(np.uint8)

    return result


def _apply_crystallize(frame, mask, bbox, progress, intensity, rng):
    """Voronoi tessellation (scipy) or grid pixelation fallback."""
    import numpy as np

    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return frame

    result = frame.copy()
    region = frame[y:y + bh, x:x + bw].copy()
    obj_mask = (mask[y:y + bh, x:x + bw] > 127).astype(np.float32)

    cell_size = max(2, int(20 * (1.0 - progress * intensity * 0.9)))

    try:
        from scipy.spatial import Voronoi  # noqa: F401
        crystallized = _voronoi_crystallize(region, cell_size, rng)
    except ImportError:
        crystallized = _grid_pixelate(region, cell_size)

    if len(frame.shape) == 3:
        alpha = obj_mask[:, :, None]
    else:
        alpha = obj_mask

    result[y:y + bh, x:x + bw] = (
        crystallized * alpha + region * (1.0 - alpha)
    ).astype(np.uint8)

    return result


def _voronoi_crystallize(region, cell_size, rng):
    """Apply Voronoi tessellation to an image region."""
    import numpy as np
    from scipy.spatial import Voronoi

    h, w = region.shape[:2]
    num_points = max(4, (h * w) // (cell_size * cell_size))
    points = np.array([(rng.randint(0, w - 1), rng.randint(0, h - 1))
                       for _ in range(num_points)])

    result = region.copy()
    Voronoi(points)

    for i, point in enumerate(points):
        px, py = int(point[0]), int(point[1])
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        colour = region[py, px].tolist()

        # Fill region around point with sampled colour
        r = cell_size // 2
        y1, y2 = max(0, py - r), min(h, py + r)
        x1, x2 = max(0, px - r), min(w, px + r)
        result[y1:y2, x1:x2] = colour

    return result


def _grid_pixelate(region, cell_size):
    """Grid-based pixelation fallback when scipy is not available."""
    import cv2

    h, w = region.shape[:2]
    small = cv2.resize(
        region,
        (max(1, w // cell_size), max(1, h // cell_size)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


_EFFECT_FUNCTIONS = {
    "squish": _apply_squish,
    "melt": _apply_melt,
    "inflate": _apply_inflate,
    "explode": _apply_explode,
    "dissolve": _apply_dissolve,
    "crystallize": _apply_crystallize,
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def apply_object_effect(
    video_path: str,
    mask_or_points: ObjectMask,
    effect_config: EffectConfig,
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ObjectEffectResult:
    """
    Apply a physics-simulated effect to a masked object across video frames.

    Args:
        video_path:    Source video path.
        mask_or_points: ObjectMask with per-frame masks and bounding boxes.
        effect_config:  EffectConfig specifying effect type, intensity, etc.
        out_path:       Output path (auto-generated if None).
        on_progress:    Callback(pct: int, msg: str).

    Returns:
        ObjectEffectResult with output path, frame count, and effect name.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError(
            "OpenCV is required for object effects. "
            "Install with: pip install opencv-python-headless"
        )
    import cv2

    if on_progress:
        on_progress(5, f"Applying {effect_config.effect_type} effect...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0) or 30.0
    vid_w = info.get("width", 0)
    vid_h = info.get("height", 0)
    if vid_w <= 0 or vid_h <= 0:
        raise RuntimeError(f"Cannot determine video dimensions: {vid_w}x{vid_h}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if out_path is None:
        out_path = _output_path(video_path, f"fx_{effect_config.effect_type}")

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (vid_w, vid_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to create video writer")

    effect_fn = _EFFECT_FUNCTIONS[effect_config.effect_type]
    rng = random.Random(effect_config.seed)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effect_frames = len(mask_or_points.mask_frames)
    frames_processed = 0

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < effect_frames:
                mask = mask_or_points.mask_frames[frame_idx]
                bbox = mask_or_points.bbox_per_frame[frame_idx]

                # Ensure mask is a numpy array
                if isinstance(mask, str):
                    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

                # Resize mask if needed
                if mask.shape[:2] != (vid_h, vid_w):
                    mask = cv2.resize(mask, (vid_w, vid_h), interpolation=cv2.INTER_NEAREST)

                # Calculate progress within effect duration
                effect_duration_frames = max(1, int(effect_config.duration * fps))
                progress = min(1.0, frame_idx / effect_duration_frames)

                frame = effect_fn(frame, mask, bbox, progress, effect_config.intensity, rng)

            writer.write(frame)
            frames_processed += 1
            frame_idx += 1

            if on_progress and frame_idx % 30 == 0:
                pct = 5 + int((frame_idx / max(total_frames, 1)) * 85)
                on_progress(pct, f"Processing frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding final output with audio...")

    # Mux audio from source
    try:
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    if on_progress:
        on_progress(100, f"{effect_config.effect_type} effect applied!")

    return ObjectEffectResult(
        output_path=out_path,
        frames_processed=frames_processed,
        effect_applied=effect_config.effect_type,
    )


def preview_effect_frame(
    video_path: str,
    mask: ObjectMask,
    effect_config: EffectConfig,
    timestamp: float = 0.0,
) -> Dict:
    """
    Preview an effect on a single frame.

    Args:
        video_path:    Source video.
        mask:          ObjectMask (only the first frame's mask is used).
        effect_config: EffectConfig for the effect.
        timestamp:     Time in seconds to sample the frame.

    Returns:
        Dict with preview_path (PNG) and dimensions.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("OpenCV required for preview")
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame at timestamp {timestamp}")

    h, w = frame.shape[:2]
    mask_frame = mask.mask_frames[0]
    if isinstance(mask_frame, str):
        mask_frame = cv2.imread(mask_frame, cv2.IMREAD_GRAYSCALE)
    if mask_frame.shape[:2] != (h, w):
        mask_frame = cv2.resize(mask_frame, (w, h), interpolation=cv2.INTER_NEAREST)

    bbox = mask.bbox_per_frame[0]
    effect_fn = _EFFECT_FUNCTIONS[effect_config.effect_type]
    rng = random.Random(effect_config.seed)

    result_frame = effect_fn(frame, mask_frame, bbox, 0.5, effect_config.intensity, rng)

    preview_path = tempfile.NamedTemporaryFile(
        suffix=".png", prefix="objfx_preview_", delete=False
    ).name
    cv2.imwrite(preview_path, result_frame)

    return {
        "preview_path": preview_path,
        "width": w,
        "height": h,
        "effect_type": effect_config.effect_type,
    }


def get_available_effects() -> List[Dict[str, str]]:
    """Return list of available effect types with descriptions."""
    return [
        {"type": k, "description": v}
        for k, v in EFFECT_TYPES.items()
    ]
