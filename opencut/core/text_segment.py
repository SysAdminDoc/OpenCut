"""
OpenCut Text-Based Video Segmentation Module (Category 69.1)

Natural language video segmentation: describe what to segment in plain text
and the system locates + tracks + exports the matching region as an alpha mask.

Pipeline:
    1. Parse text query into segmentation target descriptor
    2. Extract key frames from video
    3. Use CLIP embeddings to find matching region in each frame
    4. Optionally refine with SAM2 point-based segmentation
    5. Track segmentation mask across all frames
    6. Export mask as alpha-channel video or PNG sequence

Functions:
    find_target_region   - CLIP-based region matching on a single frame
    segment_by_text      - Full pipeline: query -> tracked mask video
"""

import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
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
# Default grid size for sliding-window region search
DEFAULT_GRID_COLS = 8
DEFAULT_GRID_ROWS = 6

# Minimum CLIP similarity score to accept a region match
MIN_CLIP_SIMILARITY = 0.18

# How many key frames to sample for initial target localisation
DEFAULT_SAMPLE_FRAMES = 8

# SAM2 model sizes available for refinement
SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RegionMatch:
    """A candidate region matching the text query in a single frame."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    score: float = 0.0
    frame_idx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cx(self) -> int:
        return self.x + self.width // 2

    @property
    def cy(self) -> int:
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class TextSegmentResult:
    """Result of a full text-based segmentation run."""
    output_path: str = ""
    mask_dir: str = ""
    query: str = ""
    frame_count: int = 0
    best_score: float = 0.0
    best_region: Optional[Dict] = None
    method: str = "clip"
    sam2_refined: bool = False
    video_width: int = 0
    video_height: int = 0
    fps: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------
# Common determiners and prepositions stripped before embedding
_STOP_WORDS = frozenset(
    "the a an in on at of to for with by from is are was were "
    "that this these those and or but".split()
)


def _clean_query(query: str) -> str:
    """Normalise and strip stop-words from a natural-language query.

    Keeps adjective+noun structure intact so CLIP gets a focused prompt.
    Example: 'the red car on the left' -> 'red car left'.
    """
    if not query or not query.strip():
        raise ValueError("Segmentation query must not be empty")
    tokens = query.lower().strip().split()
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    # If filtering removed everything, keep the original minus determiners
    if not filtered:
        filtered = [t for t in tokens if t not in {"the", "a", "an"}]
    return " ".join(filtered) if filtered else query.strip().lower()


def _build_clip_prompts(clean_query: str) -> List[str]:
    """Build several prompt variants for CLIP to improve matching.

    CLIP responds better to templated prompts than bare nouns.
    """
    base = clean_query
    prompts = [
        f"a photo of {base}",
        f"a video frame showing {base}",
        base,
        f"{base} in a scene",
        f"close-up of {base}",
    ]
    return prompts


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------
def _extract_frames(
    video_path: str,
    out_dir: str,
    count: int = DEFAULT_SAMPLE_FRAMES,
    fps_val: float = 0,
) -> List[str]:
    """Extract *count* evenly spaced frames (or all frames at given fps) to *out_dir*.

    Returns sorted list of extracted frame paths.
    """
    ffmpeg = get_ffmpeg_path()
    pattern = os.path.join(out_dir, "frame_%06d.png")

    if fps_val > 0:
        # Extract at a specific fps (for full-video processing)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", f"fps={fps_val}",
            "-q:v", "2", pattern,
        ]
    else:
        # Extract N evenly-spaced frames via select filter
        info = get_video_info(video_path)
        total_frames = max(int(info.get("duration", 1) * info.get("fps", 30)), 1)
        step = max(total_frames // max(count, 1), 1)
        select_expr = f"not(mod(n\\,{step}))"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", f"select='{select_expr}'",
            "-vsync", "vfr",
            "-frames:v", str(count),
            "-q:v", "2", pattern,
        ]

    run_ffmpeg(cmd, timeout=600)
    frames = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")
    )
    return frames


def _extract_all_frames(video_path: str, out_dir: str) -> List[str]:
    """Extract every frame to out_dir as PNGs. Returns sorted paths."""
    ffmpeg = get_ffmpeg_path()
    pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-q:v", "2", pattern,
    ]
    run_ffmpeg(cmd, timeout=7200)
    frames = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")
    )
    return frames


# ---------------------------------------------------------------------------
# CLIP-based region matching
# ---------------------------------------------------------------------------
def _load_clip(on_progress: Optional[Callable] = None):
    """Lazy-load CLIP model and processor, installing if needed.

    Returns (model, processor, torch) tuple.
    """
    if not ensure_package("transformers", "transformers", on_progress):
        raise RuntimeError("transformers library required for CLIP. Install: pip install transformers")
    if not ensure_package("torch", "torch", on_progress):
        raise RuntimeError("PyTorch required for CLIP. Install: pip install torch")
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow required. Install: pip install Pillow")

    import torch
    from PIL import Image  # noqa: F401
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    logger.info("Loading CLIP model: %s", model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    return model, processor, torch


def _sliding_window_regions(
    img_w: int,
    img_h: int,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    overlap: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """Generate sliding-window region boxes (x, y, w, h) covering the image.

    Each box is (img_w/grid_cols) wide, with *overlap* fractional overlap.
    Also includes 2x-sized windows for larger objects.
    """
    regions = []
    cell_w = max(img_w // grid_cols, 32)
    cell_h = max(img_h // grid_rows, 32)
    step_x = max(int(cell_w * (1 - overlap)), 1)
    step_y = max(int(cell_h * (1 - overlap)), 1)

    # Standard grid
    y = 0
    while y + cell_h <= img_h:
        x = 0
        while x + cell_w <= img_w:
            regions.append((x, y, cell_w, cell_h))
            x += step_x
        y += step_y

    # Larger 2x windows for bigger objects
    big_w = min(cell_w * 2, img_w)
    big_h = min(cell_h * 2, img_h)
    big_step_x = max(big_w // 2, 1)
    big_step_y = max(big_h // 2, 1)
    y = 0
    while y + big_h <= img_h:
        x = 0
        while x + big_w <= img_w:
            regions.append((x, y, big_w, big_h))
            x += big_step_x
        y += big_step_y

    return regions


def find_target_region(
    frame_path: str,
    query: str,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    on_progress: Optional[Callable] = None,
) -> RegionMatch:
    """Find the image region best matching *query* using CLIP similarity.

    Slides a grid of windows across the frame, encodes each crop, and
    scores against the text query embeddings.

    Args:
        frame_path: Path to a single frame image (PNG/JPG).
        query: Natural-language description of the target.
        grid_cols: Horizontal grid divisions.
        grid_rows: Vertical grid divisions.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        RegionMatch with location and score.
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    clean = _clean_query(query)
    prompts = _build_clip_prompts(clean)

    model, processor, torch = _load_clip(on_progress)
    from PIL import Image

    if on_progress:
        on_progress(20, "Scanning frame regions...")

    img = Image.open(frame_path).convert("RGB")
    img_w, img_h = img.size

    # Compute text embeddings for all prompt variants
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Average across prompt variants
        text_embed = text_embeds.mean(dim=0, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    regions = _sliding_window_regions(img_w, img_h, grid_cols, grid_rows)
    best = RegionMatch(score=-1.0)

    # Process regions in batches to avoid OOM
    batch_size = 32
    total_batches = math.ceil(len(regions) / batch_size)

    for bi in range(total_batches):
        batch = regions[bi * batch_size : (bi + 1) * batch_size]
        crops = []
        for (rx, ry, rw, rh) in batch:
            crop = img.crop((rx, ry, rx + rw, ry + rh))
            crops.append(crop)

        img_inputs = processor(images=crops, return_tensors="pt", padding=True)
        with torch.no_grad():
            img_embeds = model.get_image_features(**img_inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            sims = (img_embeds @ text_embed.T).squeeze(-1)

        for idx, sim_val in enumerate(sims.tolist()):
            if sim_val > best.score:
                rx, ry, rw, rh = batch[idx]
                best = RegionMatch(
                    x=rx, y=ry, width=rw, height=rh,
                    score=sim_val, frame_idx=0,
                )

        if on_progress:
            pct = 20 + int(60 * (bi + 1) / total_batches)
            on_progress(pct, f"Scanned {min((bi+1)*batch_size, len(regions))}/{len(regions)} regions")

    if best.score < MIN_CLIP_SIMILARITY:
        logger.warning("Best CLIP score %.3f below threshold %.3f for query '%s'",
                        best.score, MIN_CLIP_SIMILARITY, query)

    return best


# ---------------------------------------------------------------------------
# SAM2 mask refinement
# ---------------------------------------------------------------------------
def _refine_with_sam2(
    frames_dir: str,
    center_point: Tuple[int, int],
    mask_dir: str,
    model_size: str = "tiny",
    on_progress: Optional[Callable] = None,
) -> int:
    """Use SAM2 to generate precise segmentation masks from a centre point.

    Args:
        frames_dir: Directory containing extracted frame PNGs.
        center_point: (x, y) of the target region centre.
        mask_dir: Output directory for mask PNGs.
        model_size: SAM2 model variant.
        on_progress: Progress callback.

    Returns:
        Number of mask frames generated.
    """
    if not ensure_package("sam2", "sam2", on_progress):
        logger.warning("SAM2 not available; skipping refinement")
        return 0
    if not ensure_package("torch", "torch", on_progress):
        return 0

    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    model_id = SAM2_MODELS.get(model_size, SAM2_MODELS["tiny"])
    if on_progress:
        on_progress(60, f"Loading SAM2 ({model_size}) for refinement...")

    predictor = build_sam2_video_predictor(model_id)
    os.makedirs(mask_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frame_files:
        return 0

    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir)

        # Add click point on frame 0
        cx, cy = center_point
        points = np.array([[cx, cy]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

        # Propagate through all frames
        mask_count = 0
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            if len(masks) == 0:
                continue
            mask_np = (masks[0][0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, mode="L")
            mask_img.save(os.path.join(mask_dir, f"mask_{frame_idx:06d}.png"))
            mask_count += 1

            if on_progress and mask_count % 30 == 0:
                pct = 60 + int(30 * mask_count / len(frame_files))
                on_progress(min(pct, 90), f"SAM2 masks: {mask_count}/{len(frame_files)}")

    return mask_count


# ---------------------------------------------------------------------------
# Fallback mask generation (CLIP-only, bounding box)
# ---------------------------------------------------------------------------
def _generate_bbox_masks(
    frames: List[str],
    region: RegionMatch,
    mask_dir: str,
    on_progress: Optional[Callable] = None,
) -> int:
    """Generate binary bounding-box masks for each frame (no SAM2).

    Applies the same bounding box from the CLIP match to every frame.
    This is crude but functional without SAM2.

    Returns:
        Number of masks written.
    """
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for mask generation")
    from PIL import Image

    os.makedirs(mask_dir, exist_ok=True)

    for i, fpath in enumerate(frames):
        img = Image.open(fpath)
        w, h = img.size
        # Create black mask
        mask = Image.new("L", (w, h), 0)
        # Draw white rectangle for the region
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        x1 = max(region.x, 0)
        y1 = max(region.y, 0)
        x2 = min(region.x + region.width, w)
        y2 = min(region.y + region.height, h)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        mask.save(os.path.join(mask_dir, f"mask_{i:06d}.png"))

        if on_progress and i % 50 == 0:
            pct = 60 + int(30 * i / len(frames))
            on_progress(min(pct, 90), f"Generating masks: {i}/{len(frames)}")

    return len(frames)


# ---------------------------------------------------------------------------
# Alpha-channel video export
# ---------------------------------------------------------------------------
def _export_alpha_video(
    video_path: str,
    mask_dir: str,
    out_path: str,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """Merge original video with mask sequence into an alpha-channel video.

    Uses FFmpeg to overlay the mask as the alpha channel, outputting a
    VP9+WebM (supports alpha) or ProRes 4444 (supports alpha).

    Returns the output file path.
    """
    ffmpeg = get_ffmpeg_path()
    mask_pattern = os.path.join(mask_dir, "mask_%06d.png")

    # Check if masks exist
    mask_files = [f for f in os.listdir(mask_dir) if f.startswith("mask_") and f.endswith(".png")]
    if not mask_files:
        raise RuntimeError("No mask files found for alpha export")

    if on_progress:
        on_progress(92, "Encoding alpha-channel video...")

    # Use VP9 WebM for alpha support
    if not out_path.endswith(".webm"):
        out_path = os.path.splitext(out_path)[0] + ".webm"

    filter_complex = (
        "[0:v][1:v]alphamerge[merged]"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-framerate", str(fps), "-i", mask_pattern,
        "-filter_complex", filter_complex,
        "-map", "[merged]",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-crf", "18",
        "-b:v", "0",
        out_path,
    ]

    run_ffmpeg(cmd, timeout=3600)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def segment_by_text(
    video_path: str,
    query: str = "the red car",
    use_sam2: bool = True,
    sam2_model: str = "tiny",
    output_dir: str = "",
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    sample_frames: int = DEFAULT_SAMPLE_FRAMES,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Segment a described object from video using natural language.

    Full pipeline:
        1. Sample key frames and find best matching region via CLIP
        2. Optionally refine mask with SAM2
        3. Generate mask sequence for all frames
        4. Export as alpha-channel video

    Args:
        video_path: Path to input video file.
        query: Natural language description of the object to segment.
        use_sam2: Whether to refine with SAM2 (requires sam2 + torch).
        sam2_model: SAM2 model size if use_sam2 is True.
        output_dir: Directory for output files.
        grid_cols: CLIP sliding window horizontal divisions.
        grid_rows: CLIP sliding window vertical divisions.
        sample_frames: Number of key frames for initial CLIP search.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, mask_dir, query, frame_count, best_score,
        best_region, method, sam2_refined.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    clean = _clean_query(query)
    if on_progress:
        on_progress(2, f"Segmenting: '{clean}'...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    vid_w = info.get("width", 1920)
    vid_h = info.get("height", 1080)

    # Work directories
    work_dir = tempfile.mkdtemp(prefix="textseg_")
    sample_dir = os.path.join(work_dir, "samples")
    frames_dir = os.path.join(work_dir, "frames")
    mask_dir = os.path.join(work_dir, "masks")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    if on_progress:
        on_progress(5, "Extracting sample frames...")

    # Step 1: Extract sample frames
    sample_frames_list = _extract_frames(video_path, sample_dir, count=sample_frames)
    if not sample_frames_list:
        raise RuntimeError("Failed to extract any frames from video")

    if on_progress:
        on_progress(10, f"Searching {len(sample_frames_list)} frames for '{clean}'...")

    # Step 2: Find best matching region across sample frames
    best_region = RegionMatch(score=-1.0)
    for fi, fpath in enumerate(sample_frames_list):
        region = find_target_region(
            fpath, query,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
        )
        region.frame_idx = fi
        if region.score > best_region.score:
            best_region = region

        if on_progress:
            pct = 10 + int(30 * (fi + 1) / len(sample_frames_list))
            on_progress(pct, f"Frame {fi+1}/{len(sample_frames_list)}: score={region.score:.3f}")

    if best_region.score < 0:
        raise RuntimeError(f"Could not find any region matching '{query}'")

    logger.info("Best region for '%s': (%d,%d %dx%d) score=%.3f on frame %d",
                query, best_region.x, best_region.y, best_region.width,
                best_region.height, best_region.score, best_region.frame_idx)

    if on_progress:
        on_progress(45, "Extracting all frames...")

    # Step 3: Extract all frames for mask generation
    all_frames = _extract_all_frames(video_path, frames_dir)
    total_frames = len(all_frames)

    if on_progress:
        on_progress(55, f"Generating masks for {total_frames} frames...")

    # Step 4: Generate masks
    sam2_used = False
    if use_sam2:
        mask_count = _refine_with_sam2(
            frames_dir,
            (best_region.cx, best_region.cy),
            mask_dir,
            model_size=sam2_model,
            on_progress=on_progress,
        )
        if mask_count > 0:
            sam2_used = True
            logger.info("SAM2 generated %d mask frames", mask_count)

    if not sam2_used:
        logger.info("Using bounding-box masks (SAM2 unavailable or failed)")
        _generate_bbox_masks(all_frames, best_region, mask_dir, on_progress=on_progress)

    # Step 5: Export alpha video
    if on_progress:
        on_progress(90, "Exporting alpha-channel video...")

    out_dir = output_dir or os.path.dirname(video_path)
    out_file = output_path(video_path, "textseg", output_dir=out_dir)
    final_path = _export_alpha_video(video_path, mask_dir, out_file, fps=fps, on_progress=on_progress)

    if on_progress:
        on_progress(100, "Text segmentation complete")

    result = TextSegmentResult(
        output_path=final_path,
        mask_dir=mask_dir,
        query=query,
        frame_count=total_frames,
        best_score=best_region.score,
        best_region=best_region.to_dict(),
        method="clip+sam2" if sam2_used else "clip_bbox",
        sam2_refined=sam2_used,
        video_width=vid_w,
        video_height=vid_h,
        fps=fps,
    )
    return result.to_dict()
