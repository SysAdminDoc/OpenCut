"""
OpenCut EchoMimic Lipsync v1.30.0

Portrait and half-body audio-driven lipsync via EchoMimic.

EchoMimic (BadToBest/EchoMimic) generates lip-synced video from a reference
portrait image + audio clip using diffusion-based motion generation.

Backend priority:
  1. echomimic PyPI package (if installed manually from the EchoMimic repo)
  2. diffusers + torch with the EchoMimic pipeline from HuggingFace

Model weights (~2 GB) auto-download to ~/.opencut/models/echomimic/ on
first use. Set OPENCUT_ECHOMIMIC_MODEL_DIR to override.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import, get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")


INSTALL_HINT = (
    "pip install torch diffusers accelerate transformers\n"
    "  # Model weights (~2 GB) will download automatically on first run.\n"
    "  # Set OPENCUT_ECHOMIMIC_MODEL_DIR to specify a local model directory."
)

_ECHOMIMIC_HF_REPO = "BadToBest/EchoMimic"
_DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models", "echomimic")
_MODES = ("portrait", "halfbody")


@dataclass
class EchoMimicResult:
    output: str = ""
    mode: str = "portrait"
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_echomimic_available() -> bool:
    """Returns True if EchoMimic package or diffusers+torch are available."""
    if _try_import("echomimic") is not None:
        return True
    return (
        _try_import("torch") is not None
        and _try_import("diffusers") is not None
    )


def _get_model_dir() -> str:
    return os.environ.get("OPENCUT_ECHOMIMIC_MODEL_DIR", _DEFAULT_MODEL_DIR)


# ---------------------------------------------------------------------------
# Backend: echomimic package
# ---------------------------------------------------------------------------
def _animate_via_echomimic_package(
    image_path: str,
    audio_path: str,
    mode: str,
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Use the echomimic Python package if installed."""
    echomimic = _try_import("echomimic")
    if echomimic is None:
        raise ImportError("echomimic package not available")

    if on_progress:
        on_progress(10, "Loading EchoMimic pipeline...")

    # Standard EchoMimic package API
    from echomimic.pipelines.pipeline_echo_mimic import EchoMimicPipeline  # type: ignore

    model_dir = _get_model_dir()
    pipe = EchoMimicPipeline.from_pretrained(model_dir)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    if on_progress:
        on_progress(30, f"Generating lipsync video ({mode} mode)...")

    # EchoMimic inference
    output = pipe(
        image=image,
        audio_path=audio_path,
        mode=mode,
        output_path=output_path,
    )

    # Normalise output — package may return path or object
    if isinstance(output, str):
        return output
    if hasattr(output, "video"):
        return output_path
    return output_path


# ---------------------------------------------------------------------------
# Backend: diffusers + HuggingFace
# ---------------------------------------------------------------------------
def _animate_via_diffusers(
    image_path: str,
    audio_path: str,
    mode: str,
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Use diffusers pipeline with EchoMimic weights from HuggingFace."""
    torch = _try_import("torch")
    diffusers = _try_import("diffusers")
    if torch is None or diffusers is None:
        raise ImportError("torch and diffusers are required for EchoMimic")

    import torch as _torch
    from diffusers import DiffusionPipeline  # type: ignore

    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    if on_progress:
        on_progress(5, "Loading EchoMimic model (may download ~2 GB on first run)...")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    dtype = _torch.float16 if device == "cuda" else _torch.float32

    # Try local first, fall back to HuggingFace download
    try:
        if os.path.isdir(model_dir) and any(
            f.endswith((".bin", ".safetensors", ".pt"))
            for f in os.listdir(model_dir)
        ):
            pipe = DiffusionPipeline.from_pretrained(
                model_dir, torch_dtype=dtype, trust_remote_code=True
            )
            logger.info("EchoMimic: loaded from local %s", model_dir)
        else:
            raise FileNotFoundError("No local weights")
    except Exception:
        logger.info("EchoMimic: downloading from HuggingFace %s", _ECHOMIMIC_HF_REPO)
        pipe = DiffusionPipeline.from_pretrained(
            _ECHOMIMIC_HF_REPO,
            torch_dtype=dtype,
            cache_dir=model_dir,
            trust_remote_code=True,
        )

    pipe = pipe.to(device)

    if on_progress:
        on_progress(25, "Preprocessing inputs...")

    import torchaudio  # type: ignore
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    # Load audio as tensor
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    duration_sec = waveform.shape[1] / sample_rate
    # EchoMimic typically runs at 25fps
    num_frames = max(8, int(duration_sec * 25))

    if on_progress:
        on_progress(35, f"Generating {num_frames} frames ({duration_sec:.1f}s)...")

    try:
        result = pipe(
            image=image,
            audio_waveform=waveform,
            audio_sample_rate=sample_rate,
            num_frames=num_frames,
            fps=25,
        )
        frames = result.frames[0] if hasattr(result, "frames") else result

        if on_progress:
            on_progress(75, "Encoding output video...")

        # Export frames + audio to mp4
        _export_frames_to_video(frames, audio_path, output_path, fps=25)

    except Exception as exc:
        # EchoMimic via diffusers may have a different pipeline interface
        # Try the generic __call__ with keyword args
        logger.debug("EchoMimic diffusers call failed (%s), trying fallback", exc)
        raise RuntimeError(
            f"EchoMimic pipeline failed: {exc}\n"
            "Ensure EchoMimic-compatible weights are downloaded to "
            f"{model_dir} or set OPENCUT_ECHOMIMIC_MODEL_DIR."
        )

    return output_path


def _export_frames_to_video(
    frames: Any,
    audio_path: str,
    output_path: str,
    fps: int = 25,
) -> None:
    """Write PIL frames or numpy arrays to MP4, then mux with audio."""
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    tmp_video = output_path + ".tmp_noaudio.mp4"

    # Determine frame size
    first_frame = frames[0] if hasattr(frames, "__getitem__") else None
    if first_frame is None:
        raise ValueError("No frames to export")

    if isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
    else:
        w, h = PILImage.fromarray(first_frame).size if hasattr(first_frame, "__array_interface__") else first_frame.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    try:
        for frame in frames:
            if isinstance(frame, np.ndarray):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                arr = np.array(frame)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    # Mux with original audio
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", tmp_video,
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        run_ffmpeg(cmd, timeout=300)
    finally:
        if os.path.isfile(tmp_video):
            os.unlink(tmp_video)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def animate(
    image_path: str,
    audio_path: str,
    mode: str = "portrait",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> EchoMimicResult:
    """
    Generate an audio-driven lip-synced video from a portrait image.

    Args:
        image_path: Path to a portrait or half-body reference image (JPG/PNG).
        audio_path: Path to the speech audio file (WAV/MP3).
        mode: "portrait" for head+shoulders, "halfbody" for full torso.
        output: Output video path. Auto-generated if None.
        on_progress: Optional callback(pct, msg).

    Returns:
        EchoMimicResult with output path, mode, duration, and notes.

    Raises:
        RuntimeError: If neither echomimic nor torch+diffusers are available.
        FileNotFoundError: If input files don't exist.
    """
    if not check_echomimic_available():
        raise RuntimeError(f"EchoMimic is not installed.\n    {INSTALL_HINT}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    mode = mode if mode in _MODES else "portrait"

    if output is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        ts = str(int(__import__("time").time()))
        output = os.path.join(
            os.path.expanduser("~"), ".opencut",
            f"opencut_echomimic_{base}_{ts}.mp4",
        )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    notes: List[str] = []

    # Try echomimic package first
    if _try_import("echomimic") is not None:
        try:
            result_path = _animate_via_echomimic_package(
                image_path, audio_path, mode, output, on_progress
            )
            notes.append("Backend: echomimic package")
        except Exception as exc:
            logger.warning("EchoMimic package failed: %s — trying diffusers", exc)
            notes.append(f"echomimic package failed: {exc}")
            result_path = _animate_via_diffusers(
                image_path, audio_path, mode, output, on_progress
            )
            notes.append("Backend: diffusers fallback")
    else:
        result_path = _animate_via_diffusers(
            image_path, audio_path, mode, output, on_progress
        )
        notes.append("Backend: diffusers")

    # Get output duration
    duration = 0.0
    try:
        from opencut.helpers import get_video_info
        info = get_video_info(result_path)
        duration = float(info.get("duration", 0) or 0)
    except Exception:
        pass

    if on_progress:
        on_progress(100, "EchoMimic lipsync complete")

    return EchoMimicResult(
        output=result_path,
        mode=mode,
        duration=round(duration, 3),
        notes=notes,
    )


__all__ = ["EchoMimicResult", "check_echomimic_available", "INSTALL_HINT", "animate"]
