"""
OpenCut Smart Upscaling Hub v1.29.0

Unified dispatcher that selects the best available upscaling backend
based on content type, available hardware, and installed packages.

Backend priority (highest quality first):
  1. FlashVSR  — real-time streaming SR, needs flashvsr + torch
  2. Real-ESRGAN — frame-by-frame AI SR, needs realesrgan + torch
  3. Video2x  — CLI pipeline (Real-ESRGAN + RIFE), needs video2x
  4. Lanczos  — FFmpeg bilinear/lanczos, always available

Content hints:
  auto       — score all available backends, pick best
  fast       — lanczos only (no GPU, instant)
  quality    — highest-tier available
  anime      — Real-ESRGAN anime model
  face       — Real-ESRGAN face-optimised (RealESRGAN_x4plus_anime_6B)
  film       — Video2x with temporal RIFE (temporal consistency priority)
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "pip install realesrgan basicsr torch  "
    "# for AI upscaling; lanczos always available via FFmpeg"
)

CONTENT_HINTS = ("auto", "fast", "quality", "anime", "face", "film")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class UpscaleHubResult:
    output: str = ""
    backend: str = "lanczos"
    scale: int = 2
    width_out: int = 0
    height_out: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "backend", "scale", "width_out", "height_out", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _check_realesrgan() -> bool:
    try:
        from realesrgan import RealESRGANer  # noqa: F401
        return True
    except ImportError:
        return False


def _check_flashvsr() -> bool:
    try:
        import importlib
        importlib.import_module("flashvsr")
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_video2x() -> bool:
    import shutil
    return shutil.which("video2x") is not None


def get_available_backends() -> Dict[str, bool]:
    return {
        "lanczos": True,
        "realesrgan": _check_realesrgan(),
        "video2x": _check_video2x(),
        "flashvsr": _check_flashvsr(),
    }


def _select_backend(hint: str, available: Dict[str, bool]) -> str:
    """
    Pick the best backend for the given content hint.

    Priority (given a hint):
      fast          → lanczos always
      film          → video2x (temporal) > realesrgan > lanczos
      anime         → realesrgan (anime model) > lanczos
      face          → realesrgan (face model) > lanczos
      quality/auto  → flashvsr > realesrgan > video2x > lanczos
    """
    if hint == "fast":
        return "lanczos"

    if hint == "film":
        if available.get("video2x"):
            return "video2x"
        if available.get("realesrgan"):
            return "realesrgan"
        return "lanczos"

    if hint in ("anime", "face"):
        if available.get("realesrgan"):
            return "realesrgan"
        return "lanczos"

    # auto / quality — highest tier first
    if available.get("flashvsr"):
        return "flashvsr"
    if available.get("realesrgan"):
        return "realesrgan"
    if available.get("video2x"):
        return "video2x"
    return "lanczos"


# ---------------------------------------------------------------------------
# Backend dispatchers
# ---------------------------------------------------------------------------

def _upscale_lanczos(video_path: str, scale: int, output: str, on_progress) -> str:
    from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg
    info = get_video_info(video_path)
    new_w = info["width"] * scale
    new_h = info["height"] * scale
    if on_progress:
        on_progress(10, f"Lanczos {info['width']}x{info['height']} → {new_w}x{new_h}...")
    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"scale={new_w}:{new_h}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output,
    ], timeout=14400)
    if on_progress:
        on_progress(100, "Lanczos complete")
    return output


def _upscale_realesrgan(video_path: str, scale: int, output: str,
                         hint: str, on_progress) -> str:
    """Dispatch to upscale_pro.upscale_realesrgan with model adjusted for hint."""
    from opencut.core.upscale_pro import upscale_realesrgan
    # Choose appropriate model for the content hint
    model_name = "RealESRGAN_x4plus"
    if hint == "anime":
        model_name = "RealESRGAN_x4plus_anime_6B"
    elif hint == "face":
        model_name = "RealESRGAN_x4plus"   # face model not separate in open weights

    return upscale_realesrgan(
        video_path,
        scale=scale,
        model_name=model_name,
        output_path=output,
        on_progress=on_progress,
    )


def _upscale_video2x(video_path: str, scale: int, output: str, on_progress) -> str:
    from opencut.core.upscale_pro import upscale_video2x
    return upscale_video2x(video_path, scale=scale, output_path=output,
                            on_progress=on_progress)


def _upscale_flashvsr(video_path: str, scale: int, output: str, on_progress) -> str:
    """FlashVSR real-time streaming SR."""
    import torch
    from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg

    if on_progress:
        on_progress(5, "Loading FlashVSR model...")

    import flashvsr  # type: ignore[import]

    info = get_video_info(video_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = flashvsr.FlashVSR(scale=scale).to(device).eval()

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    new_w = info["width"] * scale
    new_h = info["height"] * scale

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (new_w, new_h))

    try:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = frame[:, :, ::-1].copy()
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            with torch.inference_mode():
                sr = model(t).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (np.clip(sr, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
            writer.write(out)
            idx += 1
            if on_progress and idx % 10 == 0:
                on_progress(5 + int((idx / total) * 85), f"FlashVSR frame {idx}/{total}")
    finally:
        cap.release()
        writer.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if on_progress:
        on_progress(92, "Re-encoding with audio...")
    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_path, "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output,
    ], timeout=14400)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, "FlashVSR complete")
    return output


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def upscale(
    video_path: str,
    scale: int = 2,
    hint: str = "auto",
    backend: str = "auto",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> UpscaleHubResult:
    """
    Upscale video using the best available backend.

    Args:
        video_path: Input video file path.
        scale:      Integer scale factor (2 or 4).
        hint:       Content type hint — see CONTENT_HINTS.
                    Ignored when backend is not "auto".
        backend:    Force a specific backend (lanczos/realesrgan/video2x/flashvsr/auto).
        output:     Output path.  Auto-generated under ~/.opencut if None.
        on_progress: Callback ``(percent, message)``.

    Returns:
        UpscaleHubResult
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"video_path does not exist: {video_path!r}")

    scale = max(1, min(8, int(scale)))
    hint = hint if hint in CONTENT_HINTS else "auto"

    available = get_available_backends()

    # Resolve backend
    chosen = backend if backend in available else "auto"
    if chosen == "auto":
        chosen = _select_backend(hint, available)

    if not available.get(chosen, False):
        logger.warning("Requested backend %r unavailable; falling back to lanczos", chosen)
        chosen = "lanczos"

    # Build output path
    if output is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{base}_smart_{chosen}_{scale}x.mp4")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    notes = [f"Backend: {chosen}", f"Scale: {scale}x", f"Hint: {hint}"]

    if on_progress:
        on_progress(1, f"Upscaling via {chosen} ({scale}x)...")

    try:
        if chosen == "lanczos":
            _upscale_lanczos(video_path, scale, output, on_progress)
        elif chosen == "realesrgan":
            _upscale_realesrgan(video_path, scale, output, hint, on_progress)
        elif chosen == "video2x":
            _upscale_video2x(video_path, scale, output, on_progress)
        elif chosen == "flashvsr":
            _upscale_flashvsr(video_path, scale, output, on_progress)
        else:
            raise RuntimeError(f"Unknown backend: {chosen!r}")
    except Exception as exc:
        if chosen != "lanczos":
            logger.warning("Backend %r failed (%s); retrying with lanczos", chosen, exc)
            notes.append(f"Fallback to lanczos: {exc}")
            chosen = "lanczos"
            _upscale_lanczos(video_path, scale, output, on_progress)
        else:
            raise

    from opencut.helpers import get_video_info
    try:
        info_out = get_video_info(output)
        w_out = info_out.get("width", 0)
        h_out = info_out.get("height", 0)
    except Exception:
        w_out = h_out = 0

    return UpscaleHubResult(
        output=output,
        backend=chosen,
        scale=scale,
        width_out=w_out,
        height_out=h_out,
        notes=notes,
    )


def get_hub_info() -> Dict:
    """Return info dict for the /info endpoint."""
    available = get_available_backends()
    order = []
    for hint in CONTENT_HINTS:
        b = _select_backend(hint, available)
        order.append({"hint": hint, "selected_backend": b})

    return {
        "available_backends": available,
        "content_hints": CONTENT_HINTS,
        "auto_selection": order,
        "install_hint": INSTALL_HINT,
    }


__all__ = [
    "UpscaleHubResult",
    "INSTALL_HINT",
    "CONTENT_HINTS",
    "get_available_backends",
    "get_hub_info",
    "upscale",
]
