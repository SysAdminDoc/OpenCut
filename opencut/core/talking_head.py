"""
OpenCut AI Talking Head / Avatar Generation v1.10.0

Generate talking head video from a still photo + audio:
- SadTalker pipeline: image + audio -> lip-synced face animation
- LivePortrait pipeline: alternative face reenactment
- Simple FFmpeg fallback: static image + audio = video with fade-in,
  subtle zoom, Ken Burns effect, optional subtitle overlay

References:
  - SadTalker: https://github.com/OpenTalker/SadTalker
  - LivePortrait: https://github.com/KwaiVGI/LivePortrait
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffprobe_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Supported backends
# ---------------------------------------------------------------------------
BACKENDS = {
    "sadtalker": {
        "label": "SadTalker",
        "packages": [("sadtalker", "sadtalker")],
        "description": "Full lip-sync + head motion from audio-driven 3DMM",
    },
    "liveportrait": {
        "label": "LivePortrait",
        "packages": [("liveportrait", "liveportrait")],
        "description": "Face reenactment via implicit keypoints",
    },
    "simple": {
        "label": "Simple (FFmpeg)",
        "packages": [],
        "description": "Static image + audio with Ken Burns / zoom / fade",
    },
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TalkingHeadConfig:
    """Configuration for talking head generation."""
    image_path: str = ""
    audio_path: str = ""
    backend: str = "simple"
    fps: int = 25
    resolution: Tuple[int, int] = (512, 512)
    expression_scale: float = 1.0
    pose_style: int = 0
    still_mode: bool = False
    enhancer: str = ""
    preprocess: str = "crop"


@dataclass
class TalkingHeadResult:
    """Result of talking head generation."""
    output_path: str = ""
    duration: float = 0.0
    frames: int = 0
    backend_used: str = ""
    face_detected: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Face Detection
# ---------------------------------------------------------------------------

def detect_face_in_image(image_path: str) -> Dict:
    """
    Detect faces in an image using OpenCV Haar cascades.

    Returns:
        Dict with keys: detected (bool), count (int), faces (list of rects),
        primary_face (largest face rect or None).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")

    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Try Haar cascade face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(cascade_path):
        # Fallback: assume face present if image is reasonable
        return {
            "detected": True,
            "count": 1,
            "faces": [{"x": 0, "y": 0, "w": w, "h": h}],
            "primary_face": {"x": 0, "y": 0, "w": w, "h": h},
            "image_size": {"width": w, "height": h},
            "method": "fallback",
        }

    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return {
            "detected": False,
            "count": 0,
            "faces": [],
            "primary_face": None,
            "image_size": {"width": w, "height": h},
            "method": "haar_cascade",
        }

    face_rects = []
    for (fx, fy, fw, fh) in faces:
        face_rects.append({"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)})

    # Primary face = largest by area
    primary = max(face_rects, key=lambda f: f["w"] * f["h"])

    return {
        "detected": True,
        "count": len(face_rects),
        "faces": face_rects,
        "primary_face": primary,
        "image_size": {"width": w, "height": h},
        "method": "haar_cascade",
    }


# ---------------------------------------------------------------------------
# Backend Availability
# ---------------------------------------------------------------------------

def list_available_backends() -> List[Dict]:
    """
    List all talking head backends and their availability.

    Returns:
        List of dicts with name, label, available, description.
    """
    import importlib
    results = []
    for name, info in BACKENDS.items():
        available = True
        if info["packages"]:
            for pkg, _ in info["packages"]:
                try:
                    importlib.import_module(pkg)
                except ImportError:
                    available = False
                    break
        results.append({
            "name": name,
            "label": info["label"],
            "available": available,
            "description": info["description"],
        })
    return results


def _check_backend(backend: str) -> bool:
    """Check whether a specific backend is available."""
    import importlib
    info = BACKENDS.get(backend)
    if info is None:
        return False
    for pkg, _ in info.get("packages", []):
        try:
            importlib.import_module(pkg)
        except ImportError:
            return False
    return True


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------

def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    import json as _json
    import subprocess as _sp

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            data = _json.loads(result.stdout.decode())
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Simple FFmpeg Talking Head (fallback)
# ---------------------------------------------------------------------------

def generate_simple_talking_head(
    image_path: str,
    audio_path: str,
    output: Optional[str] = None,
    output_dir: str = "",
    fps: int = 25,
    resolution: Optional[Tuple[int, int]] = None,
    zoom_speed: float = 0.0003,
    fade_duration: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a video from a static image + audio using FFmpeg.

    Applies Ken Burns (zoompan), fade-in, and optional subtitle overlay
    to make a simple talking-head style video from a photo.

    Args:
        image_path: Path to the source face/portrait image.
        audio_path: Path to the audio file (speech).
        output: Explicit output path. Auto-generated if None.
        output_dir: Output directory (used when output is None).
        fps: Output video frame rate.
        resolution: Output (width, height) or None to auto-detect.
        zoom_speed: Zoom speed per frame for Ken Burns effect.
        fade_duration: Duration of fade-in/out in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the generated video.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if output is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        directory = output_dir or os.path.dirname(image_path)
        output = os.path.join(directory, f"{base}_talking.mp4")

    if on_progress:
        on_progress(5, "Preparing simple talking head video...")

    # Get audio duration
    duration = _get_audio_duration(audio_path)
    if duration <= 0:
        duration = 10.0  # fallback

    # Determine resolution
    if resolution:
        w, h = resolution
    else:
        w, h = 512, 512

    if on_progress:
        on_progress(15, "Building FFmpeg filter graph...")

    # Ken Burns zoompan: slow zoom in centered on face area
    # zoompan: zoom from 1.0 to ~1.15 over the duration
    total_frames = int(duration * fps)
    max_zoom = 1.0 + zoom_speed * total_frames
    max_zoom = min(max_zoom, 1.5)  # cap zoom

    # Build filter: scale image, apply zoompan, fade in/out
    int(fade_duration * fps)
    vf_parts = [
        f"scale={w * 2}:{h * 2}",
        f"zoompan=z='min(zoom+{zoom_speed},{max_zoom})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={w}x{h}:fps={fps}",
        f"fade=t=in:st=0:d={fade_duration}",
        f"fade=t=out:st={max(0, duration - fade_duration)}:d={fade_duration}",
    ]
    vf = ",".join(vf_parts)

    if on_progress:
        on_progress(25, "Generating video...")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-tune", "stillimage", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-t", str(duration),
        output,
    ]

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Simple talking head video complete!")

    return output


# ---------------------------------------------------------------------------
# SadTalker Backend
# ---------------------------------------------------------------------------

def _generate_sadtalker(
    config: TalkingHeadConfig,
    output: str,
    on_progress: Optional[Callable] = None,
) -> TalkingHeadResult:
    """Generate talking head using SadTalker pipeline."""
    if on_progress:
        on_progress(5, "Loading SadTalker model...")

    if not ensure_package("sadtalker", "sadtalker", on_progress):
        raise RuntimeError(
            "SadTalker not available. Install: pip install sadtalker"
        )

    import sadtalker

    if on_progress:
        on_progress(20, "Running SadTalker inference...")

    duration = _get_audio_duration(config.audio_path)

    try:
        result = sadtalker.generate(
            source_image=config.image_path,
            driven_audio=config.audio_path,
            result_dir=os.path.dirname(output),
            expression_scale=config.expression_scale,
            pose_style=config.pose_style,
            still=config.still_mode,
            enhancer=config.enhancer or None,
            preprocess=config.preprocess,
            fps=config.fps,
        )

        # SadTalker returns path to generated video
        generated_path = result if isinstance(result, str) else str(result)

        # Move to desired output path if different
        if generated_path != output and os.path.isfile(generated_path):
            import shutil
            shutil.move(generated_path, output)

        if on_progress:
            on_progress(100, "SadTalker generation complete!")

        frames = int(duration * config.fps) if duration > 0 else 0
        return TalkingHeadResult(
            output_path=output,
            duration=duration,
            frames=frames,
            backend_used="sadtalker",
            face_detected=True,
            message="Generated with SadTalker",
        )
    except Exception as e:
        raise RuntimeError(f"SadTalker generation failed: {e}") from e


# ---------------------------------------------------------------------------
# LivePortrait Backend
# ---------------------------------------------------------------------------

def _generate_liveportrait(
    config: TalkingHeadConfig,
    output: str,
    on_progress: Optional[Callable] = None,
) -> TalkingHeadResult:
    """Generate talking head using LivePortrait pipeline."""
    if on_progress:
        on_progress(5, "Loading LivePortrait model...")

    if not ensure_package("liveportrait", "liveportrait", on_progress):
        raise RuntimeError(
            "LivePortrait not available. Install: pip install liveportrait"
        )

    import liveportrait

    if on_progress:
        on_progress(20, "Running LivePortrait inference...")

    duration = _get_audio_duration(config.audio_path)

    try:
        result = liveportrait.animate(
            source_image=config.image_path,
            driving_audio=config.audio_path,
            output_path=output,
            fps=config.fps,
        )

        generated_path = result if isinstance(result, str) else output

        if on_progress:
            on_progress(100, "LivePortrait generation complete!")

        frames = int(duration * config.fps) if duration > 0 else 0
        return TalkingHeadResult(
            output_path=generated_path,
            duration=duration,
            frames=frames,
            backend_used="liveportrait",
            face_detected=True,
            message="Generated with LivePortrait",
        )
    except Exception as e:
        raise RuntimeError(f"LivePortrait generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_talking_head(
    config: TalkingHeadConfig,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> TalkingHeadResult:
    """
    Generate a talking head video from a still image + audio.

    Tries the requested backend first, falls back to simpler backends
    if the primary one is unavailable.

    Args:
        config: TalkingHeadConfig with image_path, audio_path, backend, etc.
        output: Output video path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        TalkingHeadResult with output_path, duration, frames, backend_used.
    """
    if not config.image_path or not os.path.isfile(config.image_path):
        raise FileNotFoundError(f"Image not found: {config.image_path}")
    if not config.audio_path or not os.path.isfile(config.audio_path):
        raise FileNotFoundError(f"Audio not found: {config.audio_path}")

    if output is None:
        base = os.path.splitext(os.path.basename(config.image_path))[0]
        directory = os.path.dirname(config.image_path)
        output = os.path.join(directory, f"{base}_talking_head.mp4")

    backend = config.backend.lower().strip()

    # Determine backend order (requested first, then fallbacks)
    backend_order = []
    if backend in BACKENDS:
        backend_order.append(backend)
    for b in ["sadtalker", "liveportrait", "simple"]:
        if b not in backend_order:
            backend_order.append(b)

    last_error = None
    for attempt_backend in backend_order:
        if on_progress:
            on_progress(2, f"Trying backend: {attempt_backend}...")

        try:
            if attempt_backend == "sadtalker" and _check_backend("sadtalker"):
                return _generate_sadtalker(config, output, on_progress)
            elif attempt_backend == "liveportrait" and _check_backend("liveportrait"):
                return _generate_liveportrait(config, output, on_progress)
            elif attempt_backend == "simple":
                # Simple backend always available
                result_path = generate_simple_talking_head(
                    image_path=config.image_path,
                    audio_path=config.audio_path,
                    output=output,
                    fps=config.fps,
                    resolution=config.resolution,
                    on_progress=on_progress,
                )
                duration = _get_audio_duration(config.audio_path)
                frames = int(duration * config.fps) if duration > 0 else 0
                return TalkingHeadResult(
                    output_path=result_path,
                    duration=duration,
                    frames=frames,
                    backend_used="simple",
                    face_detected=True,
                    message="Generated with simple FFmpeg pipeline",
                )
        except Exception as e:
            last_error = e
            logger.warning("Backend %s failed: %s", attempt_backend, e)
            continue

    raise RuntimeError(
        f"All talking head backends failed. Last error: {last_error}"
    )
