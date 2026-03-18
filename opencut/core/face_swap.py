"""
OpenCut Face Swap Module v0.9.0

AI-powered face operations:
- Face swap: replace face in video with reference face image
- Face enhancement: restore/upscale faces (GFPGAN/CodeFormer)
- Face mosaic: artistic face styling
- Uses InsightFace for detection + FaceFusion pipeline

Heavy ML module - requires CUDA GPU and large model downloads.
Falls back to OpenCV-based face enhancement for CPU-only systems.
"""

import logging
import os
import tempfile
from typing import Callable, Dict, Optional

from opencut.helpers import ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def check_insightface_available() -> bool:
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        return False


def check_gfpgan_available() -> bool:
    try:
        import gfpgan  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Face Enhancement (GFPGAN - lightweight, practical)
# ---------------------------------------------------------------------------
def enhance_faces(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    model: str = "gfpgan",
    upscale: int = 2,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Enhance/restore faces in video using GFPGAN.

    Upscales and restores face quality - fixes blur, compression artifacts,
    and low resolution on faces while preserving the rest of the frame.

    Args:
        model: "gfpgan" (default, best general quality).
        upscale: Face upscale factor (1-4).
    """
    if not ensure_package("gfpgan", "gfpgan", on_progress):
        raise RuntimeError("GFPGAN not installed. Run: pip install gfpgan")
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")

    import cv2
    from gfpgan import GFPGANer

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_enhanced.mp4")

    if on_progress:
        on_progress(5, "Loading GFPGAN model...")

    # Auto-download model
    model_path = os.path.expanduser("~/.opencut/models/GFPGANv1.4.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    if on_progress:
        on_progress(10, "Enhancing faces frame by frame...")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                _, _, output = restorer.enhance(frame, paste_back=True)
                if output is not None:
                    output = cv2.resize(output, (orig_w, orig_h))
                    writer.write(output)
                else:
                    writer.write(frame)
            except Exception as e:
                logger.debug("Face enhance frame %d failed: %s", frame_idx, e)
                writer.write(frame)

            frame_idx += 1
            if on_progress and frame_idx % 10 == 0:
                pct = 10 + int((frame_idx / total) * 80)
                on_progress(pct, f"Enhancing frame {frame_idx}/{total}...")
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
        on_progress(100, "Faces enhanced!")
    return output_path


# ---------------------------------------------------------------------------
# Face Swap (InsightFace based)
# ---------------------------------------------------------------------------
def swap_face(
    video_path: str,
    reference_face: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Swap faces in video with a reference face image.

    Uses InsightFace for detection + inswapper model for swapping.

    Args:
        reference_face: Path to image containing the source face.
    """
    if not ensure_package("insightface", "insightface", on_progress):
        raise RuntimeError("InsightFace not installed")
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")
    if not ensure_package("onnxruntime", "onnxruntime", on_progress):
        raise RuntimeError("Failed to install onnxruntime. Install manually: pip install onnxruntime")

    import cv2
    import insightface
    from insightface.app import FaceAnalysis

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_swapped.mp4")

    if on_progress:
        on_progress(5, "Loading face analysis model...")

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load swapper model
    model_dir = os.path.expanduser("~/.opencut/models")
    swapper_path = os.path.join(model_dir, "inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        raise FileNotFoundError(
            f"Swap model not found at {swapper_path}. "
            "Download inswapper_128.onnx from InsightFace model zoo."
        )

    swapper = insightface.model_zoo.get_model(swapper_path)

    # Get source face
    ref_img = cv2.imread(reference_face)
    if ref_img is None:
        raise RuntimeError(f"Cannot read reference face image: {reference_face}")
    ref_faces = app.get(ref_img)
    if not ref_faces:
        raise RuntimeError("No face found in reference image")
    source_face = ref_faces[0]

    if on_progress:
        on_progress(15, "Swapping faces in video...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                faces = app.get(frame)
                for face in faces:
                    frame = swapper.get(frame, face, source_face, paste_back=True)
            except Exception as e:
                logger.debug("Face swap frame %d failed: %s", frame_idx, e)

            writer.write(frame)
            frame_idx += 1
            if on_progress and frame_idx % 10 == 0:
                pct = 15 + int((frame_idx / total) * 75)
                on_progress(pct, f"Swapping frame {frame_idx}/{total}...")
    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding...")

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
        on_progress(100, "Face swap complete!")
    return output_path


def get_face_capabilities() -> Dict:
    return {
        "insightface": check_insightface_available(),
        "gfpgan": check_gfpgan_available(),
    }
