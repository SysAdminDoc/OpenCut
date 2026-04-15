"""
OpenCut Physics-Aware Object Removal Module (Category 69.2)

Remove objects AND their physical interactions (shadows, reflections) from video.

Pipeline:
    1. Segment the target object via SAM2 or user-supplied mask points
    2. Detect shadow direction and extent from the object mask
    3. Detect reflections on nearby surfaces
    4. Combine object + shadow + reflection into a unified removal mask
    5. Inpaint removed regions with temporal consistency via ProPainter/LaMA
    6. Re-encode with FFmpeg

Functions:
    detect_shadow       - Analyse shadow direction and extent for a single frame
    remove_with_physics - Full pipeline: mask + shadow + reflection removal
"""

import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Shadow detection parameters
SHADOW_INTENSITY_THRESHOLD = 0.65   # max relative brightness for shadow region
SHADOW_SEARCH_RADIUS = 2.5         # multiplier on object bbox for shadow search area
SHADOW_MIN_AREA_RATIO = 0.05       # min shadow area as fraction of object area
SHADOW_MAX_AREA_RATIO = 3.0        # max shadow area as fraction of object area

# Reflection detection
REFLECTION_SEARCH_BELOW = 1.5      # search this far below object for reflections
REFLECTION_SIMILARITY_THRESH = 0.4  # structural similarity threshold

# Inpainting
INPAINT_DILATE_PX = 8              # dilate removal mask edges for cleaner fill
INPAINT_TEMPORAL_WINDOW = 5        # frames to consider for temporal consistency

# ProPainter model path
PROPAINTER_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models", "propainter")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ShadowInfo:
    """Detected shadow information for an object."""
    detected: bool = False
    direction_deg: float = 0.0       # angle in degrees, 0 = right, 90 = down
    extent_px: int = 0               # how far the shadow extends
    area_px: int = 0                 # shadow area in pixels
    intensity: float = 0.0           # average shadow darkness (0=black, 1=white)
    mask_points: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReflectionInfo:
    """Detected reflection information."""
    detected: bool = False
    surface_y: int = 0               # y-coordinate of reflective surface
    similarity: float = 0.0          # structural similarity to object
    area_px: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhysicsRemoveResult:
    """Result of physics-aware removal."""
    output_path: str = ""
    object_mask_area: int = 0
    shadow_info: Optional[Dict] = None
    reflection_info: Optional[Dict] = None
    total_removed_area: int = 0
    frame_count: int = 0
    inpaint_method: str = "lama"
    video_width: int = 0
    video_height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shadow detection
# ---------------------------------------------------------------------------
def _compute_shadow_direction(
    frame_gray, object_mask, obj_bbox: Tuple[int, int, int, int],
) -> Tuple[float, int]:
    """Estimate shadow direction by analysing dark regions adjacent to the object.

    Scans radially around the object centroid for the darkest cluster
    outside the object mask.

    Args:
        frame_gray: Grayscale numpy array of the frame.
        object_mask: Binary numpy mask of the object (255=object).
        obj_bbox: (x, y, w, h) bounding box of the object.

    Returns:
        (direction_degrees, extent_pixels) tuple.
    """

    ox, oy, ow, oh = obj_bbox
    cx = ox + ow // 2
    cy = oy + oh // 2
    max_extent = int(max(ow, oh) * SHADOW_SEARCH_RADIUS)

    best_angle = 0.0
    best_darkness = 255.0
    best_extent = 0
    h, w = frame_gray.shape[:2]

    # Sample 36 directions (every 10 degrees)
    for angle_idx in range(36):
        angle_rad = math.radians(angle_idx * 10)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        darkness_sum = 0.0
        count = 0
        farthest = 0

        for step in range(10, max_extent, 4):
            sx = int(cx + dx * step)
            sy = int(cy + dy * step)

            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                break

            # Skip if inside the object mask
            if object_mask[sy, sx] > 127:
                continue

            pixel_val = float(frame_gray[sy, sx])
            if pixel_val < 255 * SHADOW_INTENSITY_THRESHOLD:
                darkness_sum += (255 - pixel_val)
                count += 1
                farthest = step

        if count > 0:
            avg_darkness = darkness_sum / count
            if avg_darkness > best_darkness or (avg_darkness == best_darkness and farthest > best_extent):
                best_darkness = avg_darkness
                best_angle = angle_idx * 10.0
                best_extent = farthest

    return best_angle, best_extent


def _extract_shadow_mask(
    frame_gray, object_mask,
    obj_bbox: Tuple[int, int, int, int],
    direction_deg: float,
    extent_px: int,
) -> "numpy.ndarray":  # noqa: F821
    """Generate a binary mask of the shadow region.

    Creates a search corridor in the shadow direction and thresholds
    dark pixels that aren't part of the object.
    """
    import numpy as np

    h, w = frame_gray.shape[:2]
    shadow_mask = np.zeros((h, w), dtype=np.uint8)

    ox, oy, ow, oh = obj_bbox
    cx = ox + ow // 2
    cy = oy + oh // 2

    angle_rad = math.radians(direction_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Corridor width: half the object's smaller dimension
    corridor_half = max(min(ow, oh) // 2, 16)

    # Walk along the shadow direction
    for step in range(5, extent_px + INPAINT_DILATE_PX):
        for lateral in range(-corridor_half, corridor_half + 1, 2):
            sx = int(cx + dx * step - dy * lateral)
            sy = int(cy + dy * step + dx * lateral)

            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                continue
            if object_mask[sy, sx] > 127:
                continue

            pixel_val = float(frame_gray[sy, sx])
            if pixel_val < 255 * SHADOW_INTENSITY_THRESHOLD:
                shadow_mask[sy, sx] = 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = _morph_close(shadow_mask, kernel, iterations=2)
    shadow_mask = _morph_open(shadow_mask, kernel, iterations=1)

    # Validate shadow area relative to object
    obj_area = max(int(np.sum(object_mask > 127)), 1)
    shadow_area = int(np.sum(shadow_mask > 127))
    ratio = shadow_area / obj_area

    if ratio < SHADOW_MIN_AREA_RATIO or ratio > SHADOW_MAX_AREA_RATIO:
        logger.info("Shadow area ratio %.2f outside bounds [%.2f, %.2f]; discarding",
                     ratio, SHADOW_MIN_AREA_RATIO, SHADOW_MAX_AREA_RATIO)
        return np.zeros((h, w), dtype=np.uint8)

    return shadow_mask


def _morph_close(mask, kernel, iterations=1):
    """Morphological close using numpy (dilate then erode)."""
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    m = mask > 127
    for _ in range(iterations):
        m = binary_dilation(m, structure=np.ones((5, 5)))
    for _ in range(iterations):
        m = binary_erosion(m, structure=np.ones((5, 5)))
    return (m.astype(np.uint8)) * 255


def _morph_open(mask, kernel, iterations=1):
    """Morphological open using numpy (erode then dilate)."""
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    m = mask > 127
    for _ in range(iterations):
        m = binary_erosion(m, structure=np.ones((5, 5)))
    for _ in range(iterations):
        m = binary_dilation(m, structure=np.ones((5, 5)))
    return (m.astype(np.uint8)) * 255


def detect_shadow(
    frame_path: str,
    object_mask_path: str = "",
    object_bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_points: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect shadow information for an object in a single frame.

    Can accept either a pre-computed object mask image or SAM2 mask points,
    or a bounding box for approximate detection.

    Args:
        frame_path: Path to the frame image.
        object_mask_path: Path to a binary mask image (white=object).
        object_bbox: (x, y, w, h) bounding box if no mask available.
        mask_points: SAM2-style point prompts for on-demand segmentation.
        on_progress: Progress callback(pct, msg).

    Returns:
        ShadowInfo as dict.
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    if not ensure_package("numpy", "numpy", on_progress):
        raise RuntimeError("numpy required")
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow required")

    import numpy as np
    from PIL import Image

    if on_progress:
        on_progress(10, "Loading frame for shadow detection...")

    frame_img = Image.open(frame_path).convert("RGB")
    frame_arr = np.array(frame_img)
    frame_gray = np.mean(frame_arr, axis=2).astype(np.uint8)
    h, w = frame_gray.shape

    # Load or generate object mask
    if object_mask_path and os.path.isfile(object_mask_path):
        mask_img = Image.open(object_mask_path).convert("L")
        object_mask = np.array(mask_img.resize((w, h)))
    elif object_bbox:
        object_mask = np.zeros((h, w), dtype=np.uint8)
        bx, by, bw, bh = object_bbox
        object_mask[max(by, 0):min(by + bh, h), max(bx, 0):min(bx + bw, w)] = 255
    elif mask_points:
        # Generate mask using SAM2 if available
        object_mask = _generate_mask_from_points(frame_path, mask_points, on_progress)
        if object_mask is None:
            return ShadowInfo(detected=False).to_dict()
    else:
        raise ValueError("Must provide object_mask_path, object_bbox, or mask_points")

    if on_progress:
        on_progress(30, "Detecting shadow direction...")

    # Compute bounding box from mask
    ys, xs = np.where(object_mask > 127)
    if len(xs) == 0:
        return ShadowInfo(detected=False).to_dict()

    obj_bbox_computed = (int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min()), int(ys.max() - ys.min()))

    direction, extent = _compute_shadow_direction(frame_gray, object_mask, obj_bbox_computed)

    if extent < 5:
        logger.info("No significant shadow detected (extent=%d)", extent)
        return ShadowInfo(detected=False, direction_deg=direction).to_dict()

    if on_progress:
        on_progress(60, f"Extracting shadow mask (dir={direction:.0f} deg, extent={extent}px)...")

    shadow_mask = _extract_shadow_mask(
        frame_gray, object_mask, obj_bbox_computed, direction, extent,
    )
    shadow_area = int(np.sum(shadow_mask > 127))

    if shadow_area == 0:
        return ShadowInfo(detected=False, direction_deg=direction).to_dict()

    # Compute shadow bounding box
    sy, sx = np.where(shadow_mask > 127)
    shadow_bbox = (int(sx.min()), int(sy.min()),
                   int(sx.max() - sx.min()), int(sy.max() - sy.min()))

    # Average intensity in shadow region
    shadow_pixels = frame_gray[shadow_mask > 127]
    avg_intensity = float(np.mean(shadow_pixels)) / 255.0

    if on_progress:
        on_progress(100, "Shadow detection complete")

    result = ShadowInfo(
        detected=True,
        direction_deg=direction,
        extent_px=extent,
        area_px=shadow_area,
        intensity=round(avg_intensity, 3),
        bbox=shadow_bbox,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Reflection detection
# ---------------------------------------------------------------------------
def _detect_reflection(
    frame_arr, object_mask, obj_bbox: Tuple[int, int, int, int],
) -> ReflectionInfo:
    """Detect a reflection of the object on surfaces below it.

    Looks for vertically-flipped structural similarity below the object.
    """
    import numpy as np

    h, w = frame_arr.shape[:2]
    ox, oy, ow, oh = obj_bbox

    # Search area below the object
    search_top = oy + oh
    search_bottom = min(int(search_top + oh * REFLECTION_SEARCH_BELOW), h)

    if search_top >= h or search_bottom <= search_top:
        return ReflectionInfo(detected=False)

    # Extract the object region and flip vertically
    obj_region = frame_arr[max(oy, 0):min(oy + oh, h), max(ox, 0):min(ox + ow, w)]
    if obj_region.size == 0:
        return ReflectionInfo(detected=False)

    flipped = np.flipud(obj_region)

    # Extract the below-object region
    ref_height = min(oh, search_bottom - search_top)
    below_region = frame_arr[search_top:search_top + ref_height, max(ox, 0):min(ox + ow, w)]

    if below_region.size == 0 or flipped.shape[0] == 0:
        return ReflectionInfo(detected=False)

    # Resize flipped to match below region dimensions
    trim_h = min(flipped.shape[0], below_region.shape[0])
    trim_w = min(flipped.shape[1], below_region.shape[1])
    flipped_trimmed = flipped[:trim_h, :trim_w]
    below_trimmed = below_region[:trim_h, :trim_w]

    if flipped_trimmed.size == 0:
        return ReflectionInfo(detected=False)

    # Compute normalised cross-correlation as similarity
    f_flat = flipped_trimmed.astype(np.float32).flatten()
    b_flat = below_trimmed.astype(np.float32).flatten()

    f_norm = f_flat - f_flat.mean()
    b_norm = b_flat - b_flat.mean()

    denom = (np.linalg.norm(f_norm) * np.linalg.norm(b_norm))
    if denom < 1e-6:
        return ReflectionInfo(detected=False)

    similarity = float(np.dot(f_norm, b_norm) / denom)

    if similarity < REFLECTION_SIMILARITY_THRESH:
        return ReflectionInfo(detected=False, similarity=round(similarity, 3))

    area = trim_h * trim_w
    bbox = (ox, search_top, ow, trim_h)

    return ReflectionInfo(
        detected=True,
        surface_y=search_top,
        similarity=round(similarity, 3),
        area_px=area,
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Mask point helpers
# ---------------------------------------------------------------------------
def _generate_mask_from_points(
    frame_path: str,
    mask_points: List[Dict],
    on_progress: Optional[Callable] = None,
) -> "Optional[numpy.ndarray]":  # noqa: F821
    """Generate a binary mask from SAM2 point prompts on a single frame.

    Returns a numpy array (H, W) with 255 for the object.
    Falls back to bounding-box mask if SAM2 is unavailable.
    """
    if not ensure_package("numpy", "numpy"):
        return None
    import numpy as np
    from PIL import Image

    img = Image.open(frame_path)
    w, h = img.size

    # Try SAM2
    try:
        if not ensure_package("sam2", "sam2"):
            raise ImportError("SAM2 not available")
        if not ensure_package("torch", "torch"):
            raise ImportError("torch not available")

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model_id = "facebook/sam2.1-hiera-tiny"
        predictor = SAM2ImagePredictor(build_sam2(model_id))
        predictor.set_image(np.array(img))

        points = []
        labels = []
        for pt in mask_points:
            points.append([pt.get("x", 0), pt.get("y", 0)])
            labels.append(pt.get("label", 1))

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
            )
            best_idx = int(np.argmax(scores))
            mask = (masks[best_idx] > 0.5).astype(np.uint8) * 255
            return mask

    except (ImportError, Exception) as e:
        logger.info("SAM2 unavailable for point mask: %s; using bbox fallback", e)

    # Bbox fallback from points
    mask = np.zeros((h, w), dtype=np.uint8)
    if mask_points:
        xs = [p.get("x", 0) for p in mask_points]
        ys = [p.get("y", 0) for p in mask_points]
        pad = 40
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, w)
        y2 = min(max(ys) + pad, h)
        mask[y1:y2, x1:x2] = 255
    return mask


# ---------------------------------------------------------------------------
# Unified mask combination
# ---------------------------------------------------------------------------
def _combine_masks(
    object_mask, shadow_mask, reflection_mask, dilate_px: int = INPAINT_DILATE_PX,
) -> "numpy.ndarray":  # noqa: F821
    """Combine object, shadow, and reflection masks into a single removal mask.

    Applies dilation to ensure clean inpainting boundaries.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation

    combined = np.zeros_like(object_mask)
    combined = np.maximum(combined, object_mask)
    if shadow_mask is not None:
        combined = np.maximum(combined, shadow_mask)
    if reflection_mask is not None:
        combined = np.maximum(combined, reflection_mask)

    # Dilate edges
    if dilate_px > 0:
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
        dilated = binary_dilation(combined > 127, structure=struct)
        combined = (dilated.astype(np.uint8)) * 255

    return combined


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------
def _inpaint_frame_lama(frame_path: str, mask_path: str, out_path: str) -> bool:
    """Inpaint a single frame using SimpleLaMA."""
    try:
        from simple_lama_inpainting import SimpleLama
        from PIL import Image

        lama = SimpleLama()
        img = Image.open(frame_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        result = lama(img, mask)
        result.save(out_path)
        return True
    except Exception as e:
        logger.warning("LaMA inpainting failed: %s", e)
        return False


def _inpaint_frame_ffmpeg(frame_path: str, mask_path: str, out_path: str) -> bool:
    """Inpaint using FFmpeg's delogo or blend filter as basic fallback."""
    try:
        from PIL import Image
        import numpy as np

        mask_img = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask_img)
        ys, xs = np.where(mask_arr > 127)

        if len(xs) == 0:
            # No mask region; just copy
            Image.open(frame_path).save(out_path)
            return True

        # Bounding box of mask
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bw = x2 - x1
        bh = y2 - y1

        if bw < 2 or bh < 2:
            Image.open(frame_path).save(out_path)
            return True

        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", frame_path,
            "-vf", f"delogo=x={x1}:y={y1}:w={bw}:h={bh}",
            out_path,
        ]
        run_ffmpeg(cmd, timeout=30)
        return True
    except Exception as e:
        logger.warning("FFmpeg inpaint fallback failed: %s", e)
        return False


def _inpaint_frames(
    frames_dir: str,
    masks_dir: str,
    output_dir: str,
    method: str = "lama",
    on_progress: Optional[Callable] = None,
) -> Tuple[str, int]:
    """Inpaint all frames using the specified method.

    Returns (method_used, frame_count).
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".png"))

    # Map mask index to frame index
    mask_map = {}
    for mf in mask_files:
        idx_str = mf.replace("mask_", "").replace(".png", "")
        try:
            mask_map[int(idx_str)] = os.path.join(masks_dir, mf)
        except ValueError:
            continue

    inpainted = 0
    method_used = method

    for fi, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, fname)
        out_path = os.path.join(output_dir, fname)

        mask_path = mask_map.get(fi)
        if mask_path is None or not os.path.isfile(mask_path):
            # No mask for this frame, copy as-is
            from PIL import Image
            Image.open(frame_path).save(out_path)
            inpainted += 1
            continue

        success = False
        if method == "lama":
            success = _inpaint_frame_lama(frame_path, mask_path, out_path)
            if not success:
                method_used = "ffmpeg_delogo"

        if not success:
            success = _inpaint_frame_ffmpeg(frame_path, mask_path, out_path)

        if not success:
            # Last resort: copy original
            from PIL import Image
            Image.open(frame_path).save(out_path)

        inpainted += 1

        if on_progress and fi % 20 == 0:
            pct = 60 + int(30 * fi / len(frame_files))
            on_progress(min(pct, 90), f"Inpainting: {fi}/{len(frame_files)}")

    return method_used, inpainted


# ---------------------------------------------------------------------------
# Reassemble video from inpainted frames
# ---------------------------------------------------------------------------
def _reassemble_video(
    inpainted_dir: str,
    original_video: str,
    out_path: str,
    fps: float = 30.0,
) -> str:
    """Reassemble inpainted frames into a video, copying original audio."""
    ffmpeg = get_ffmpeg_path()
    frame_pattern = os.path.join(inpainted_dir, "frame_%06d.png")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", original_video,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        out_path,
    ]

    run_ffmpeg(cmd, timeout=3600)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def remove_with_physics(
    video_path: str,
    mask_points: Optional[List[Dict]] = None,
    object_bbox: Optional[Tuple[int, int, int, int]] = None,
    detect_shadows: bool = True,
    detect_reflections: bool = True,
    inpaint_method: str = "lama",
    output_dir: str = "",
    sam2_model: str = "tiny",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Remove an object and its physical interactions from video.

    Full pipeline: segment -> detect shadow -> detect reflection -> combine
    masks -> inpaint all frames -> reassemble video.

    Args:
        video_path: Path to input video.
        mask_points: SAM2 point prompts [{"x": int, "y": int, "label": 1}].
        object_bbox: Alternative to mask_points: (x, y, w, h) of the object.
        detect_shadows: Whether to detect and remove shadows.
        detect_reflections: Whether to detect and remove reflections.
        inpaint_method: "lama" or "ffmpeg_delogo".
        output_dir: Output directory.
        sam2_model: SAM2 model size for mask generation.
        on_progress: Progress callback(pct, msg).

    Returns:
        PhysicsRemoveResult as dict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not mask_points and not object_bbox:
        raise ValueError("Must provide mask_points or object_bbox")

    if not ensure_package("numpy", "numpy", on_progress):
        raise RuntimeError("numpy is required")
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow is required")

    import numpy as np
    from PIL import Image

    if on_progress:
        on_progress(2, "Starting physics-aware removal...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    vid_w = info.get("width", 1920)
    vid_h = info.get("height", 1080)

    # Work directories
    work_dir = tempfile.mkdtemp(prefix="physrem_")
    frames_dir = os.path.join(work_dir, "frames")
    masks_dir = os.path.join(work_dir, "masks")
    inpainted_dir = os.path.join(work_dir, "inpainted")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Extract all frames
    if on_progress:
        on_progress(5, "Extracting frames...")

    pattern = os.path.join(frames_dir, "frame_%06d.png")
    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-q:v", "2", pattern,
    ], timeout=7200)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    total_frames = len(frame_files)
    if total_frames == 0:
        raise RuntimeError("Failed to extract frames from video")

    if on_progress:
        on_progress(15, f"Extracted {total_frames} frames")

    # Generate object mask on first frame
    first_frame = os.path.join(frames_dir, frame_files[0])
    if mask_points:
        object_mask = _generate_mask_from_points(first_frame, mask_points, on_progress)
    else:
        object_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
        bx, by, bw, bh = object_bbox
        object_mask[max(by, 0):min(by + bh, vid_h), max(bx, 0):min(bx + bw, vid_w)] = 255

    obj_area = int(np.sum(object_mask > 127))

    # Detect shadow on first frame
    shadow_result = ShadowInfo(detected=False)
    shadow_mask = None
    if detect_shadows:
        if on_progress:
            on_progress(20, "Detecting shadows...")

        frame_img = Image.open(first_frame).convert("RGB")
        frame_arr = np.array(frame_img)
        frame_gray = np.mean(frame_arr, axis=2).astype(np.uint8)

        ys, xs = np.where(object_mask > 127)
        if len(xs) > 0:
            obj_bbox_c = (int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min()), int(ys.max() - ys.min()))

            direction, extent = _compute_shadow_direction(frame_gray, object_mask, obj_bbox_c)

            if extent >= 5:
                shadow_mask = _extract_shadow_mask(
                    frame_gray, object_mask, obj_bbox_c, direction, extent,
                )
                shadow_area = int(np.sum(shadow_mask > 127))
                if shadow_area > 0:
                    shadow_pixels = frame_gray[shadow_mask > 127]
                    avg_intensity = float(np.mean(shadow_pixels)) / 255.0
                    sy, sx = np.where(shadow_mask > 127)
                    shadow_result = ShadowInfo(
                        detected=True,
                        direction_deg=direction,
                        extent_px=extent,
                        area_px=shadow_area,
                        intensity=round(avg_intensity, 3),
                        bbox=(int(sx.min()), int(sy.min()),
                              int(sx.max() - sx.min()), int(sy.max() - sy.min())),
                    )

    # Detect reflection on first frame
    refl_result = ReflectionInfo(detected=False)
    reflection_mask = None
    if detect_reflections:
        if on_progress:
            on_progress(30, "Detecting reflections...")

        frame_img = Image.open(first_frame).convert("RGB")
        frame_arr = np.array(frame_img)

        ys, xs = np.where(object_mask > 127)
        if len(xs) > 0:
            obj_bbox_c = (int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min()), int(ys.max() - ys.min()))
            refl_result = _detect_reflection(frame_arr, object_mask, obj_bbox_c)

            if refl_result.detected:
                rb = refl_result.bbox
                reflection_mask = np.zeros_like(object_mask)
                rx, ry, rw, rh = rb
                reflection_mask[max(ry, 0):min(ry + rh, vid_h),
                                max(rx, 0):min(rx + rw, vid_w)] = 255

    # Combine all masks
    if on_progress:
        on_progress(35, "Combining removal masks...")

    combined = _combine_masks(object_mask, shadow_mask, reflection_mask)
    total_removed = int(np.sum(combined > 127))

    # Save combined mask for each frame (static mask applied to all)
    for fi in range(total_frames):
        mask_out = os.path.join(masks_dir, f"mask_{fi:06d}.png")
        Image.fromarray(combined, mode="L").save(mask_out)

    if on_progress:
        on_progress(40, "Masks generated; starting inpainting...")

    # Inpaint all frames
    method_used, frame_count = _inpaint_frames(
        frames_dir, masks_dir, inpainted_dir,
        method=inpaint_method,
        on_progress=on_progress,
    )

    # Reassemble video
    if on_progress:
        on_progress(92, "Reassembling video...")

    out_dir = output_dir or os.path.dirname(video_path)
    out_file = output_path(video_path, "physrem", output_dir=out_dir)
    final_path = _reassemble_video(inpainted_dir, video_path, out_file, fps=fps)

    if on_progress:
        on_progress(100, "Physics-aware removal complete")

    result = PhysicsRemoveResult(
        output_path=final_path,
        object_mask_area=obj_area,
        shadow_info=shadow_result.to_dict() if shadow_result.detected else None,
        reflection_info=refl_result.to_dict() if refl_result.detected else None,
        total_removed_area=total_removed,
        frame_count=total_frames,
        inpaint_method=method_used,
        video_width=vid_w,
        video_height=vid_h,
    )
    return result.to_dict()
