"""
OpenCut Object Removal Module v0.9.0

Interactive object removal from video:
- SAM 2 segmentation: point/box prompt -> mask propagation across frames
- ProPainter video inpainting: temporally consistent fill
- LaMA single-frame inpainting fallback (lighter weight)
- Static watermark/logo removal with fixed mask

Pipeline: User marks object -> SAM 2 generates + tracks masks -> ProPainter fills.
"""

import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


def _ensure_package(pkg, pip_name=None, on_progress=None):
    try:
        __import__(pkg)
        return True
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg,
                            "--break-system-packages", "-q"], capture_output=True, timeout=600)
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


def _run_ffmpeg(cmd, timeout=7200):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {r.stderr.decode(errors='replace')[-500:]}")


def _get_video_info(fp):
    import json
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
                        "-of", "json", fp], capture_output=True, timeout=30)
    try:
        s = json.loads(r.stdout.decode())["streams"][0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_p[0]) / float(fps_p[1]) if len(fps_p) == 2 else 30.0
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080)),
                "fps": fps, "duration": float(s.get("duration", 0)),
                "frames": int(s.get("nb_frames", 0))}
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0, "frames": 0}


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------
def check_sam2_available() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


def check_propainter_available() -> bool:
    try:
        import torch
        return os.path.isdir(os.path.expanduser("~/.opencut/models/propainter"))
    except ImportError:
        return False


def check_lama_available() -> bool:
    try:
        from simple_lama_inpainting import SimpleLama  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# SAM 2 Mask Generation
# ---------------------------------------------------------------------------
def generate_masks_sam2(
    video_path: str,
    prompts: List[Dict],
    output_dir: str = "",
    model_size: str = "tiny",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Generate object masks across video frames using SAM 2.

    Args:
        prompts: List of prompts for frame 0:
            [{"type": "point", "x": 500, "y": 300, "label": 1}]
            [{"type": "box", "x1": 100, "y1": 100, "x2": 400, "y2": 400}]
        model_size: "tiny" (39MB), "small" (185MB), "base" (390MB), "large" (898MB)

    Returns dict with mask_dir (PNG masks per frame) and frame_count.
    """
    if not _ensure_package("sam2", "sam2", on_progress):
        raise RuntimeError("SAM 2 not installed. Run: pip install sam2")

    _ensure_package("cv2", "opencv-python-headless")
    import cv2
    import numpy as np
    import torch

    if on_progress:
        on_progress(5, f"Loading SAM 2 ({model_size})...")

    # Model checkpoint mapping
    model_map = {
        "tiny": "facebook/sam2-hiera-tiny",
        "small": "facebook/sam2-hiera-small",
        "base": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }
    model_id = model_map.get(model_size, model_map["tiny"])

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_id)

    mask_dir = os.path.join(output_dir or tempfile.gettempdir(), "sam2_masks")
    os.makedirs(mask_dir, exist_ok=True)

    if on_progress:
        on_progress(15, "Extracting frames...")

    # Extract frames to temp dir
    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-q:v", "2",
        os.path.join(frames_dir, "frame_%06d.jpg"),
    ])

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    total = len(frame_files)

    if on_progress:
        on_progress(25, f"Running SAM 2 on {total} frames...")

    # Initialize video predictor
    with torch.inference_mode():
        state = predictor.init_state(video_path=frames_dir)

        # Add prompts on frame 0
        for p in prompts:
            if p["type"] == "point":
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    points=np.array([[p["x"], p["y"]]], dtype=np.float32),
                    labels=np.array([p.get("label", 1)], dtype=np.int32),
                )
            elif p["type"] == "box":
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    box=np.array([p["x1"], p["y1"], p["x2"], p["y2"]], dtype=np.float32),
                )

        # Propagate masks
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            mask = (masks[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(mask_dir, f"mask_{frame_idx:06d}.png"), mask)

            if on_progress and frame_idx % 30 == 0:
                pct = 25 + int((frame_idx / max(total, 1)) * 60)
                on_progress(pct, f"Tracking object: frame {frame_idx}/{total}...")

    # Cleanup frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    if on_progress:
        on_progress(90, "Masks generated!")

    return {"mask_dir": mask_dir, "frame_count": total}


# ---------------------------------------------------------------------------
# Static Watermark Removal (LaMA - lightweight)
# ---------------------------------------------------------------------------
def remove_watermark_lama(
    video_path: str,
    mask_region: Dict,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Remove static watermark/logo using LaMA inpainting per-frame.

    Args:
        mask_region: {"x": int, "y": int, "width": int, "height": int}
            Rectangle covering the watermark area.
    """
    if not _ensure_package("simple_lama_inpainting", "simple-lama-inpainting", on_progress):
        raise RuntimeError("simple-lama-inpainting not installed")

    _ensure_package("cv2", "opencv-python-headless")
    import cv2
    import numpy as np
    from simple_lama_inpainting import SimpleLama

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_no_watermark.mp4")

    if on_progress:
        on_progress(5, "Loading LaMA model...")

    lama = SimpleLama()
    info = _get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]

    # Create static mask
    mask = np.zeros((h, w), dtype=np.uint8)
    rx, ry = mask_region["x"], mask_region["y"]
    rw, rh = mask_region["width"], mask_region["height"]
    # Expand slightly for better inpainting
    pad = 5
    mask[max(0, ry - pad):min(h, ry + rh + pad),
         max(0, rx - pad):min(w, rx + rw + pad)] = 255

    cap = cv2.VideoCapture(video_path)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0

    if on_progress:
        on_progress(10, "Removing watermark frame by frame...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to PIL for LaMA
            from PIL import Image
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)

            result = lama(pil_frame, pil_mask)
            result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            writer.write(result_cv)

            frame_idx += 1
            if on_progress and frame_idx % 30 == 0:
                pct = 10 + int((frame_idx / total_frames) * 80)
                on_progress(pct, f"Inpainting frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding with audio...")

    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_video, "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])
    os.unlink(tmp_video)

    if on_progress:
        on_progress(100, "Watermark removed!")
    return output_path


# ---------------------------------------------------------------------------
# FFmpeg Delogo (lightweight built-in alternative)
# ---------------------------------------------------------------------------
def remove_watermark_delogo(
    video_path: str,
    region: Dict,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Remove watermark using FFmpeg's built-in delogo filter (fast, no ML)."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_delogo.mp4")

    x, y = region["x"], region["y"]
    rw, rh = region["width"], region["height"]

    if on_progress:
        on_progress(10, "Applying delogo filter...")

    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"delogo=x={x}:y={y}:w={rw}:h={rh}:show=0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ])

    if on_progress:
        on_progress(100, "Delogo applied!")
    return output_path


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def get_removal_capabilities() -> Dict:
    return {
        "sam2": check_sam2_available(),
        "propainter": check_propainter_available(),
        "lama": check_lama_available(),
        "delogo": True,  # Always available via FFmpeg
    }
