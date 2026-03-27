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
import tempfile
from typing import Callable, Dict, List, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

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
        import torch  # noqa: F401
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
    if not ensure_package("sam2", "sam2", on_progress):
        raise RuntimeError("SAM 2 not installed. Run: pip install sam2")

    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")
    import shutil

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
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-q:v", "2",
            os.path.join(frames_dir, "frame_%06d.jpg"),
        ], timeout=7200)

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        total = len(frame_files)

        if on_progress:
            on_progress(25, f"Running SAM 2 on {total} frames...")

        # Initialize video predictor
        with torch.inference_mode():
            state = predictor.init_state(video_path=frames_dir)

            # Add prompts on frame 0
            for p in prompts:
                ptype = p.get("type", "")
                if ptype == "point":
                    if "x" not in p or "y" not in p:
                        raise ValueError("Point prompt missing 'x' or 'y' key")
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=np.array([[float(p["x"]), float(p["y"])]], dtype=np.float32),
                        labels=np.array([p.get("label", 1)], dtype=np.int32),
                    )
                elif ptype == "box":
                    for k in ("x1", "y1", "x2", "y2"):
                        if k not in p:
                            raise ValueError(f"Box prompt missing '{k}' key")
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        box=np.array([float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])], dtype=np.float32),
                    )

            # Propagate masks
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                mask = (masks[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(mask_dir, f"mask_{frame_idx:06d}.png"), mask)

                if on_progress and frame_idx % 30 == 0:
                    pct = 25 + int((frame_idx / max(total, 1)) * 60)
                    on_progress(pct, f"Tracking object: frame {frame_idx}/{total}...")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    if on_progress:
        on_progress(90, "Masks generated!")

    return {"mask_dir": mask_dir, "frame_count": total}


# ---------------------------------------------------------------------------
# Video Inpainting (ProPainter - temporally coherent)
# ---------------------------------------------------------------------------
def inpaint_video_propainter(
    video_path: str,
    mask_dir: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Inpaint masked regions in video using ProPainter (ICCV 2023).

    Temporally coherent video inpainting via flow completion + dual-domain
    propagation. Produces far better results than per-frame LAMA for video
    (no flickering, consistent background fill across frames).

    Args:
        video_path: Source video.
        mask_dir: Directory of binary mask PNGs (white = inpaint region).
    """
    import subprocess as _sp

    if not check_propainter_available():
        raise RuntimeError(
            "ProPainter not installed. Clone to ~/.opencut/models/propainter: "
            "git clone https://github.com/sczhou/ProPainter ~/.opencut/models/propainter"
        )

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_inpainted.mp4")

    if on_progress:
        on_progress(10, "Running ProPainter video inpainting...")

    propainter_dir = os.path.expanduser("~/.opencut/models/propainter")
    inference_script = os.path.join(propainter_dir, "inference_propainter.py")

    if not os.path.isfile(inference_script):
        raise RuntimeError(f"ProPainter inference script not found: {inference_script}")

    import sys
    cmd = [
        sys.executable, inference_script,
        "--video", video_path,
        "--mask", mask_dir,
        "--output", os.path.dirname(output_path),
        "--save_fps", "0",  # auto-detect from source
        "--fp16",
    ]

    result = _sp.run(cmd, capture_output=True, text=True, timeout=7200, cwd=propainter_dir)
    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"ProPainter failed: {stderr}")

    # ProPainter writes to output dir — find the result
    out_dir = os.path.dirname(output_path)
    candidates = [f for f in os.listdir(out_dir) if f.startswith("inpaint_") and f.endswith(".mp4")]
    if candidates:
        # Rename to expected output path
        src = os.path.join(out_dir, sorted(candidates)[-1])
        if src != output_path:
            import shutil
            shutil.move(src, output_path)

    if not os.path.isfile(output_path):
        raise RuntimeError("ProPainter produced no output file")

    # Mux audio from source
    temp_out = output_path + ".tmp.mp4"
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", output_path, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", temp_out,
        ], timeout=300)
        import shutil
        shutil.move(temp_out, output_path)
    except Exception:
        # If muxing fails, keep video-only output
        try:
            os.unlink(temp_out)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Video inpainting complete!")
    return output_path


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
    if not ensure_package("simple_lama_inpainting", "simple-lama-inpainting", on_progress):
        raise RuntimeError("simple-lama-inpainting not installed")

    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")
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
    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    if w <= 0 or h <= 0 or fps <= 0:
        raise RuntimeError(f"Cannot determine video properties: {w}x{h} @ {fps}fps")

    # Create static mask
    mask = np.zeros((h, w), dtype=np.uint8)
    rx = int(mask_region.get("x", 0))
    ry = int(mask_region.get("y", 0))
    rw = int(mask_region.get("width", 100))
    rh = int(mask_region.get("height", 100))
    # Expand slightly for better inpainting
    pad = 5
    mask[max(0, ry - pad):min(h, ry + rh + pad),
         max(0, rx - pad):min(w, rx + rw + pad)] = 255

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to initialize video writer")

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx = 0

    if on_progress:
        on_progress(10, "Removing watermark frame by frame...")

    from PIL import Image
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
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

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

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

    x, y = int(region.get("x", 0)), int(region.get("y", 0))
    rw, rh = int(region.get("width", 100)), int(region.get("height", 100))

    if on_progress:
        on_progress(10, "Applying delogo filter...")

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"delogo=x={x}:y={y}:w={rw}:h={rh}:show=0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "Delogo applied!")
    return output_path


# ---------------------------------------------------------------------------
# Auto Watermark Detection (Florence-2 or fallback)
# ---------------------------------------------------------------------------
def check_florence_available() -> bool:
    """Check if Florence-2 (auto watermark detection) is available."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def detect_watermark_region(
    video_path: str,
    prompt: str = "watermark",
    sample_frame: int = 30,
    on_progress: Optional[Callable] = None,
) -> Optional[Dict]:
    """
    Auto-detect watermark location using Florence-2 vision-language model.

    Analyzes a sample frame to find the watermark region, returning
    bounding box coordinates that can be passed to remove_watermark_lama.

    Args:
        video_path: Path to video file.
        prompt: What to detect ("watermark", "logo", "text overlay").
        sample_frame: Which frame to sample (0-indexed).
        on_progress: Progress callback.

    Returns:
        Dict with x, y, width, height of detected watermark region, or None if not found.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError:
        raise ImportError("Florence-2 requires transformers and torch. Install: pip install transformers torch")

    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required. Install: pip install opencv-python-headless")

    if on_progress:
        on_progress(10, "Extracting sample frame...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read sample frame")

    if on_progress:
        on_progress(20, "Loading Florence-2 model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "microsoft/Florence-2-base"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.warning("Florence-2 model load failed: %s", e)
        return _fallback_watermark_detection(frame, on_progress)

    if on_progress:
        on_progress(50, "Detecting watermark with Florence-2...")

    from PIL import Image
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Use grounding/detection task
    task = "<OPEN_VOCABULARY_DETECTION>"
    text_input = prompt

    inputs = processor(text=task + text_input, images=pil_frame, return_tensors="pt").to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=task, image_size=(pil_frame.width, pil_frame.height))

    # Parse detection results
    if result and task in result:
        detections = result[task]
        bboxes = detections.get("bboxes", [])
        if bboxes:
            # Take the first (highest confidence) detection
            bbox = bboxes[0]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if on_progress:
                on_progress(100, f"Watermark detected at ({x1}, {y1}) — ({x2}, {y2})")

            return {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": 0.9,  # Florence-2 doesn't return per-detection confidence
                "method": "florence2",
            }

    # No detection
    if on_progress:
        on_progress(90, "No watermark detected by Florence-2, trying fallback...")

    return _fallback_watermark_detection(frame, on_progress)


def _fallback_watermark_detection(frame, on_progress=None) -> Optional[Dict]:
    """
    Fallback watermark detection using edge/corner analysis.

    Watermarks are typically in corners and have consistent edges.
    Checks top-right and bottom-right corners for high-contrast text-like regions.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check corner regions (watermarks are usually in corners)
    corner_regions = [
        ("top-right", w * 2 // 3, 0, w, h // 6),
        ("bottom-right", w * 2 // 3, h * 5 // 6, w, h),
        ("top-left", 0, 0, w // 3, h // 6),
        ("bottom-left", 0, h * 5 // 6, w // 3, h),
        ("bottom-center", w // 4, h * 5 // 6, w * 3 // 4, h),
    ]

    best_region = None
    best_score = 0

    for name, x1, y1, x2, y2 in corner_regions:
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Edge density in this region
        edges = cv2.Canny(roi, 100, 200)
        edge_density = float(np.mean(edges)) / 255.0

        # Text-like features: high local contrast
        local_std = float(np.std(roi))

        score = edge_density * 0.6 + (local_std / 128.0) * 0.4

        if score > best_score and score > 0.15:  # Minimum threshold
            best_score = score
            best_region = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                           "confidence": round(min(score * 2, 1.0), 2), "method": "edge_fallback"}

    if on_progress:
        if best_region:
            on_progress(100, f"Possible watermark detected in corner (confidence: {best_region['confidence']:.0%})")
        else:
            on_progress(100, "No watermark detected")

    return best_region


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def get_removal_capabilities() -> Dict:
    return {
        "sam2": check_sam2_available(),
        "propainter": check_propainter_available(),
        "lama": check_lama_available(),
        "delogo": True,  # Always available via FFmpeg
        "florence2": check_florence_available(),
        "auto_detect": check_florence_available(),
    }
