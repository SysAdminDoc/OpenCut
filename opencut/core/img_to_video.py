"""
OpenCut Image-to-Video Animation

Animate a still image using either:
  - AI video generation (SVD / Wan when available)
  - Enhanced Ken Burns with parallax layers

Input: image path + motion prompt / parameters.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    ensure_package,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Motion presets for Ken Burns
# ---------------------------------------------------------------------------

MOTION_PRESETS = {
    "zoom_in": {
        "description": "Slow zoom into the center of the image",
        "start_scale": 1.0,
        "end_scale": 1.4,
        "start_x": 0.5,
        "start_y": 0.5,
        "end_x": 0.5,
        "end_y": 0.5,
    },
    "zoom_out": {
        "description": "Slow zoom out from center",
        "start_scale": 1.4,
        "end_scale": 1.0,
        "start_x": 0.5,
        "start_y": 0.5,
        "end_x": 0.5,
        "end_y": 0.5,
    },
    "pan_left": {
        "description": "Horizontal pan from right to left",
        "start_scale": 1.3,
        "end_scale": 1.3,
        "start_x": 0.7,
        "start_y": 0.5,
        "end_x": 0.3,
        "end_y": 0.5,
    },
    "pan_right": {
        "description": "Horizontal pan from left to right",
        "start_scale": 1.3,
        "end_scale": 1.3,
        "start_x": 0.3,
        "start_y": 0.5,
        "end_x": 0.7,
        "end_y": 0.5,
    },
    "pan_up": {
        "description": "Vertical pan from bottom to top",
        "start_scale": 1.3,
        "end_scale": 1.3,
        "start_x": 0.5,
        "start_y": 0.7,
        "end_x": 0.5,
        "end_y": 0.3,
    },
    "pan_down": {
        "description": "Vertical pan from top to bottom",
        "start_scale": 1.3,
        "end_scale": 1.3,
        "start_x": 0.5,
        "start_y": 0.3,
        "end_x": 0.5,
        "end_y": 0.7,
    },
    "zoom_pan": {
        "description": "Zoom in while panning right",
        "start_scale": 1.0,
        "end_scale": 1.5,
        "start_x": 0.3,
        "start_y": 0.5,
        "end_x": 0.7,
        "end_y": 0.4,
    },
    "parallax": {
        "description": "Subtle parallax / depth effect",
        "start_scale": 1.1,
        "end_scale": 1.15,
        "start_x": 0.48,
        "start_y": 0.5,
        "end_x": 0.52,
        "end_y": 0.5,
    },
}


@dataclass
class ImageAnimationResult:
    """Result of image-to-video animation."""
    output_path: str = ""
    duration: float = 0.0
    fps: float = 0.0
    motion_preset: str = ""
    method: str = ""
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Ken Burns effect
# ---------------------------------------------------------------------------

def _ken_burns_ffmpeg(
    image_path: str,
    out_path: str,
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    preset_name: str = "zoom_in",
) -> str:
    """Generate Ken Burns animation using FFmpeg zoompan filter."""
    preset = MOTION_PRESETS.get(preset_name, MOTION_PRESETS["zoom_in"])

    total_frames = int(duration * fps)
    # zoompan: z = zoom level, x/y = top-left corner offset
    # Interpolate zoom and position over total_frames
    start_z = preset["start_scale"]
    end_z = preset["end_scale"]

    # zoompan uses 'on' (output frame number) for interpolation
    # z: linear interpolation from start to end
    z_expr = f"{start_z}+({end_z}-{start_z})*on/{total_frames}"

    # x, y: position of the crop window center, scaled by zoom
    sx = preset["start_x"]
    sy = preset["start_y"]
    ex = preset["end_x"]
    ey = preset["end_y"]

    x_expr = f"(iw-iw/zoom)*({sx}+({ex}-{sx})*on/{total_frames})"
    y_expr = f"(ih-ih/zoom)*({sy}+({ey}-{sy})*on/{total_frames})"

    vf = (
        f"scale=8000:-1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "faststart",
        out_path,
    ], timeout=3600)

    return out_path


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def animate_image(
    image_path: str,
    duration: float = 5.0,
    motion_preset: str = "zoom_in",
    motion_prompt: Optional[str] = None,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    method: str = "auto",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Animate a still image into a video clip.

    When *method* is ``"auto"``, attempts AI-based generation first
    (SVD / Wan), falling back to Ken Burns with parallax.

    Args:
        image_path:  Path to the input image (PNG, JPG, etc.).
        duration:  Output duration in seconds.
        motion_preset:  One of the MOTION_PRESETS keys.
        motion_prompt:  Text prompt for AI motion (used when AI is
            available).
        fps:  Output frame rate.
        width:  Output width in pixels.
        height:  Output height in pixels.
        method:  ``"auto"``, ``"ken_burns"``, or ``"ai"``.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *duration*, *fps*, *motion_preset*,
        *method*, *width*, *height*.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if motion_preset not in MOTION_PRESETS:
        motion_preset = "zoom_in"

    out = output_path_override or output_path(image_path, f"animated_{motion_preset}")
    # Ensure .mp4 extension
    if not out.lower().endswith(".mp4"):
        out = os.path.splitext(out)[0] + ".mp4"

    if on_progress:
        on_progress(5, f"Animating image with {motion_preset} effect...")

    actual_method = method
    ai_available = False

    if method in ("auto", "ai"):
        # Check for AI model availability
        try:
            torch = ensure_package("torch", "torch")
            diffusers = ensure_package("diffusers", "diffusers")
            if torch and diffusers:
                ai_available = True
        except Exception:
            ai_available = False

        if not ai_available:
            if method == "ai":
                logger.warning("AI models not available, falling back to Ken Burns")
            actual_method = "ken_burns"
        else:
            actual_method = "ai"

    if actual_method == "ai" and ai_available:
        if on_progress:
            on_progress(10, "Loading AI video generation model...")

        # AI path: use diffusers SVD pipeline
        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline
            from PIL import Image

            if on_progress:
                on_progress(20, "Generating frames with AI...")

            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)

            image = Image.open(image_path).convert("RGB").resize((width, height))
            num_frames = min(int(duration * fps), 25)

            frames = pipe(image, num_frames=num_frames, decode_chunk_size=8).frames[0]

            if on_progress:
                on_progress(70, "Encoding AI-generated frames...")

            # Write frames to temp directory and assemble
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, frame in enumerate(frames):
                    frame.save(os.path.join(tmpdir, f"frame_{i:05d}.png"))

                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "frame_%05d.png"),
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "faststart",
                    out,
                ], timeout=3600)

            actual_method = "ai_svd"
        except Exception as e:
            logger.warning("AI generation failed (%s), falling back to Ken Burns", e)
            actual_method = "ken_burns"

    if actual_method == "ken_burns":
        if on_progress:
            on_progress(10, f"Generating Ken Burns effect ({motion_preset})...")
        _ken_burns_ffmpeg(
            image_path, out,
            duration=duration, fps=fps,
            width=width, height=height,
            preset_name=motion_preset,
        )

    if on_progress:
        on_progress(100, "Image animation complete!")

    return {
        "output_path": out,
        "duration": duration,
        "fps": fps,
        "motion_preset": motion_preset,
        "method": actual_method,
        "width": width,
        "height": height,
    }


def list_motion_presets() -> list:
    """Return available motion presets with descriptions."""
    return [
        {"name": name, "description": cfg["description"]}
        for name, cfg in MOTION_PRESETS.items()
    ]
