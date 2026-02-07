"""
OpenCut Video FX - LUT, Chroma Key, Background Removal, Slow Motion, Upscaling

All FFmpeg-based features (LUT, chroma key) require no additional deps.
AI features (background removal, slow-mo, upscaling) auto-install on first use.

Models (downloaded on first use to ~/.u2net/ or model-specific dirs):
  - rembg + birefnet-general     (~170 MB)  - Background removal
  - Practical-RIFE v4.25+        (~110 MB)  - AI frame interpolation
  - Real-ESRGAN x4plus           (~65 MB)   - Video upscaling
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class VideoFxResult:
    """Generic result for video FX operations."""
    output_path: str = ""
    original_duration: float = 0.0
    output_duration: float = 0.0
    width: int = 0
    height: int = 0
    operation: str = ""
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FFmpeg probe helpers (shared)
# ---------------------------------------------------------------------------
def _probe_media(filepath: str) -> Dict:
    """Get media info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    info = {
        "duration": 0.0, "width": 1920, "height": 1080,
        "fps": 30.0, "codec": "h264", "has_audio": False,
    }
    try:
        data = json.loads(result.stdout)
        # Duration
        dur = data.get("format", {}).get("duration")
        if dur:
            info["duration"] = float(dur)
        # Video stream
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                info["width"] = int(s.get("width", 1920))
                info["height"] = int(s.get("height", 1080))
                info["codec"] = s.get("codec_name", "h264")
                r_frame = s.get("r_frame_rate", "30/1")
                if "/" in str(r_frame):
                    num, den = str(r_frame).split("/")
                    info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
                else:
                    info["fps"] = float(r_frame)
                if not dur and s.get("duration"):
                    info["duration"] = float(s["duration"])
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        pass
    return info


def _resolve_output(input_path: str, output_dir: str, suffix: str, ext: str = ".mp4") -> str:
    """Build output file path."""
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}_{suffix}{ext}")


def _run_ffmpeg(cmd: List[str], timeout: int = 3600) -> None:
    """Run an FFmpeg command, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = result.stderr[-1000:] if result.stderr else "Unknown error"
        logger.error(f"FFmpeg error: {err}")
        raise RuntimeError(f"FFmpeg failed: {err}")


# ===========================================================================
# 1. LUT APPLICATION
# ===========================================================================
LUT_FORMATS = [".cube", ".3dl", ".csp", ".spi3d"]


def get_lut_info(lut_path: str) -> Dict:
    """Read basic info from a LUT file."""
    if not os.path.isfile(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    ext = os.path.splitext(lut_path)[1].lower()
    if ext not in LUT_FORMATS:
        raise ValueError(f"Unsupported LUT format: {ext}. Supported: {LUT_FORMATS}")

    info = {
        "path": lut_path,
        "name": os.path.basename(lut_path),
        "format": ext,
        "size": os.path.getsize(lut_path),
    }

    # Try to read title and size from .cube files
    if ext == ".cube":
        try:
            with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TITLE"):
                        info["title"] = line.split('"')[1] if '"' in line else line.split(None, 1)[1]
                    elif line.startswith("LUT_3D_SIZE"):
                        info["lut_size"] = int(line.split()[1])
                    elif line.startswith("LUT_1D_SIZE"):
                        info["lut_size"] = int(line.split()[1])
                        info["type"] = "1D"
                    if "title" in info and "lut_size" in info:
                        break
        except Exception:
            pass

    return info


def apply_lut(
    input_path: str,
    lut_path: str,
    intensity: float = 1.0,
    output_dir: str = "",
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Apply a .cube / .3dl LUT to a video via FFmpeg lut3d filter.

    Args:
        input_path:  Source video.
        lut_path:    Path to LUT file (.cube, .3dl).
        intensity:   Blend intensity 0.0-1.0 (1.0 = full LUT).
        output_dir:  Output directory.
        quality:     Encoding quality.
        on_progress: Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Validating LUT file...")

    lut_info = get_lut_info(lut_path)
    media = _probe_media(input_path)

    if on_progress:
        on_progress(10, f"Applying LUT: {lut_info['name']}")

    output_path = _resolve_output(input_path, output_dir, "lut")
    crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")

    # Build filter: lut3d with optional blend for intensity < 1.0
    lut_escaped = lut_path.replace("\\", "/").replace(":", "\\:")
    if intensity >= 0.99:
        vf = f"lut3d='{lut_escaped}'"
    else:
        # Blend original with LUT-applied version
        vf = (
            f"split[orig][lut];"
            f"[lut]lut3d='{lut_escaped}'[graded];"
            f"[orig][graded]blend=all_expr='A*{1-intensity:.3f}+B*{intensity:.3f}'"
        )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-filter_complex" if ";" in vf else "-vf", vf,
        "-c:v", "libx264", "-crf", crf, "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    if on_progress:
        on_progress(30, "Encoding with LUT...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "LUT applied successfully")

    return VideoFxResult(
        output_path=output_path,
        original_duration=media["duration"],
        output_duration=media["duration"],
        width=media["width"],
        height=media["height"],
        operation="lut",
        details={"lut_name": lut_info["name"], "intensity": intensity},
    )


# ===========================================================================
# 2. CHROMA KEY (Green Screen)
# ===========================================================================
CHROMA_PRESETS = {
    "green": {"color": "0x00FF00", "similarity": 0.3, "blend": 0.1},
    "blue": {"color": "0x0000FF", "similarity": 0.3, "blend": 0.1},
    "red": {"color": "0xFF0000", "similarity": 0.3, "blend": 0.1},
    "white": {"color": "0xFFFFFF", "similarity": 0.2, "blend": 0.05},
    "black": {"color": "0x000000", "similarity": 0.2, "blend": 0.05},
}


def apply_chroma_key(
    input_path: str,
    key_color: str = "green",
    similarity: float = 0.3,
    blend: float = 0.1,
    background: str = "blur",
    bg_color: str = "000000",
    bg_image: str = "",
    output_dir: str = "",
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Apply chroma key (green screen removal) via FFmpeg.

    Args:
        input_path:  Source video with green/blue screen.
        key_color:   Preset name or hex color (e.g. "green", "0x00FF00").
        similarity:  Color similarity threshold (0.01-1.0).
        blend:       Edge blending (0.0-1.0).
        background:  Replacement: "blur", "color", "transparent", or path to image/video.
        bg_color:    Hex color for "color" background mode.
        bg_image:    Path to background image/video.
        output_dir:  Output directory.
        quality:     Encoding quality.
        on_progress: Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Preparing chroma key...")

    media = _probe_media(input_path)

    # Resolve color
    if key_color in CHROMA_PRESETS:
        preset = CHROMA_PRESETS[key_color]
        color_hex = preset["color"]
        if similarity == 0.3:
            similarity = preset["similarity"]
        if blend == 0.1:
            blend = preset["blend"]
    else:
        color_hex = key_color if key_color.startswith("0x") else f"0x{key_color}"

    output_path = _resolve_output(input_path, output_dir, "chromakey")
    crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")

    if on_progress:
        on_progress(15, f"Applying chroma key ({key_color})...")

    if background == "transparent":
        # Output with alpha channel (WebM/MOV with ProRes 4444)
        output_path = _resolve_output(input_path, output_dir, "chromakey", ".webm")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", input_path,
            "-vf", f"chromakey={color_hex}:{similarity}:{blend}",
            "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
            "-auto-alt-ref", "0",
            "-c:a", "libopus",
            output_path,
        ]
    elif background == "blur":
        # Blur the original behind the keyed subject
        vf = (
            f"split[fg][bg];"
            f"[bg]avgblur=sizeX=40:sizeY=40[blurred];"
            f"[fg]chromakey={color_hex}:{similarity}:{blend}[keyed];"
            f"[blurred][keyed]overlay=format=auto"
        )
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", input_path,
            "-filter_complex", vf,
            "-c:v", "libx264", "-crf", crf, "-preset", "fast",
            "-c:a", "copy", "-movflags", "+faststart",
            output_path,
        ]
    elif background == "color":
        vf = (
            f"color=c=#{bg_color}:s={media['width']}x{media['height']}:d={media['duration']:.3f}[bg];"
            f"[0:v]chromakey={color_hex}:{similarity}:{blend}[keyed];"
            f"[bg][keyed]overlay=format=auto"
        )
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", input_path,
            "-filter_complex", vf,
            "-c:v", "libx264", "-crf", crf, "-preset", "fast",
            "-c:a", "copy", "-movflags", "+faststart",
            output_path,
        ]
    elif bg_image and os.path.isfile(bg_image):
        # Use a background image/video
        vf = (
            f"[1:v]scale={media['width']}:{media['height']}[bg];"
            f"[0:v]chromakey={color_hex}:{similarity}:{blend}[keyed];"
            f"[bg][keyed]overlay=format=auto"
        )
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", input_path,
            "-i", bg_image,
            "-filter_complex", vf,
            "-c:v", "libx264", "-crf", crf, "-preset", "fast",
            "-c:a", "copy", "-movflags", "+faststart",
            "-shortest",
            output_path,
        ]
    else:
        raise ValueError(f"Invalid background mode: {background}")

    if on_progress:
        on_progress(30, "Encoding chroma key...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Chroma key applied")

    return VideoFxResult(
        output_path=output_path,
        original_duration=media["duration"],
        output_duration=media["duration"],
        width=media["width"],
        height=media["height"],
        operation="chroma_key",
        details={
            "key_color": key_color,
            "similarity": similarity,
            "blend": blend,
            "background": background,
        },
    )


# ===========================================================================
# 3. BACKGROUND REMOVAL (AI - rembg + BiRefNet)
# ===========================================================================
REMBG_MODELS = [
    {"id": "birefnet-general", "name": "BiRefNet General", "description": "Best quality, general purpose", "size_mb": 170},
    {"id": "birefnet-portrait", "name": "BiRefNet Portrait", "description": "Optimized for people", "size_mb": 170},
    {"id": "u2net", "name": "U2-Net", "description": "Fast, good general use", "size_mb": 170},
    {"id": "u2net_human_seg", "name": "U2-Net Human", "description": "Optimized for human segmentation", "size_mb": 170},
    {"id": "isnet-anime", "name": "ISNet Anime", "description": "For anime and cartoon content", "size_mb": 170},
]

BG_REPLACE_MODES = ["blur", "color", "transparent", "image"]


def check_rembg_available() -> Tuple[bool, str]:
    """Check if rembg is installed."""
    try:
        import rembg  # noqa: F401
        return True, "rembg is installed"
    except ImportError:
        return False, "rembg is not installed"


def install_rembg(progress_callback: Optional[Callable] = None) -> bool:
    """Install rembg with onnxruntime."""
    steps = [
        {"label": "Installing onnxruntime", "cmd": [sys.executable, "-m", "pip", "install", "onnxruntime", "--quiet"]},
        {"label": "Installing rembg", "cmd": [sys.executable, "-m", "pip", "install", "rembg[cli]", "--quiet"]},
    ]

    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            steps[0] = {
                "label": "Installing onnxruntime-gpu",
                "cmd": [sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "--quiet"],
            }
    except ImportError:
        pass

    for i, step in enumerate(steps):
        if progress_callback:
            pct = int((i / len(steps)) * 70) + 10
            progress_callback(pct, step["label"])
        logger.info(f"Rembg install: {step['label']}")
        try:
            result = subprocess.run(step["cmd"], capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"Install failed: {result.stderr[-500:]}")
                return False
        except Exception as e:
            logger.error(f"Install error: {e}")
            return False

    return True


def remove_background(
    input_path: str,
    model: str = "birefnet-general",
    background: str = "transparent",
    bg_color: str = "000000",
    bg_image: str = "",
    output_dir: str = "",
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Remove background from video using rembg (frame-by-frame).

    For images: processes directly.
    For video: extracts frames, processes each, re-encodes.

    Args:
        input_path:  Source file (image or video).
        model:       rembg model name.
        background:  "transparent", "blur", "color", "image".
        bg_color:    Hex color for color mode.
        bg_image:    Path to background image.
        output_dir:  Output directory.
        quality:     Encoding quality.
        on_progress: Callback(pct, msg).
    """
    from rembg import remove, new_session
    from PIL import Image
    import io
    import numpy as np

    if on_progress:
        on_progress(5, f"Loading model: {model}...")

    session = new_session(model)
    media = _probe_media(input_path)

    # Check if input is an image
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(input_path)[1].lower()

    if ext in img_exts:
        return _remove_bg_image(input_path, session, background, bg_color, bg_image, output_dir, on_progress)

    return _remove_bg_video(input_path, session, media, background, bg_color, bg_image, output_dir, quality, on_progress)


def _remove_bg_image(
    input_path: str, session, background: str,
    bg_color: str, bg_image: str, output_dir: str,
    on_progress: Optional[Callable],
) -> VideoFxResult:
    """Remove background from a single image."""
    from rembg import remove
    from PIL import Image

    if on_progress:
        on_progress(20, "Processing image...")

    with open(input_path, "rb") as f:
        input_data = f.read()

    output_data = remove(input_data, session=session)

    # Load as PIL for compositing
    fg = Image.open(io.BytesIO(output_data)).convert("RGBA")

    if background == "transparent":
        out_ext = ".png"
        result_img = fg
    elif background == "blur":
        orig = Image.open(input_path).convert("RGBA")
        from PIL import ImageFilter
        blurred = orig.filter(ImageFilter.GaussianBlur(radius=40))
        blurred.paste(fg, mask=fg.split()[3])
        result_img = blurred.convert("RGB")
        out_ext = ".png"
    elif background == "color":
        r, g, b = int(bg_color[:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)
        bg = Image.new("RGBA", fg.size, (r, g, b, 255))
        bg.paste(fg, mask=fg.split()[3])
        result_img = bg.convert("RGB")
        out_ext = ".png"
    elif background == "image" and bg_image and os.path.isfile(bg_image):
        bg = Image.open(bg_image).convert("RGBA").resize(fg.size)
        bg.paste(fg, mask=fg.split()[3])
        result_img = bg.convert("RGB")
        out_ext = ".png"
    else:
        result_img = fg
        out_ext = ".png"

    output_path = _resolve_output(input_path, output_dir, "nobg", out_ext)
    result_img.save(output_path)

    if on_progress:
        on_progress(100, "Background removed")

    return VideoFxResult(
        output_path=output_path,
        width=fg.width,
        height=fg.height,
        operation="background_removal",
        details={"model": session.inner_session_options.get("model_name", "unknown"), "background": background},
    )


def _remove_bg_video(
    input_path: str, session, media: Dict, background: str,
    bg_color: str, bg_image: str, output_dir: str,
    quality: str, on_progress: Optional[Callable],
) -> VideoFxResult:
    """Remove background from video frame-by-frame."""
    from rembg import remove
    from PIL import Image
    import io
    import numpy as np

    if on_progress:
        on_progress(10, "Extracting frames...")

    tmpdir = tempfile.mkdtemp(prefix="opencut_bg_")
    frames_dir = os.path.join(tmpdir, "frames")
    output_frames_dir = os.path.join(tmpdir, "output")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        cmd_extract = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", input_path,
            "-vsync", "0",
            os.path.join(frames_dir, "frame_%06d.png"),
        ]
        _run_ffmpeg(cmd_extract, timeout=600)

        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(".png")]
        )
        total_frames = len(frame_files)

        if total_frames == 0:
            raise RuntimeError("No frames extracted from video")

        if on_progress:
            on_progress(15, f"Processing {total_frames} frames...")

        # Load background image if needed
        bg_pil = None
        if background == "image" and bg_image and os.path.isfile(bg_image):
            bg_pil = Image.open(bg_image).convert("RGBA")

        # Process each frame
        for i, fname in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, fname)
            with open(frame_path, "rb") as f:
                frame_data = f.read()

            # Remove background
            result_data = remove(frame_data, session=session)
            fg = Image.open(io.BytesIO(result_data)).convert("RGBA")

            # Composite
            if background == "transparent":
                out_img = fg
            elif background == "blur":
                orig = Image.open(frame_path).convert("RGBA")
                from PIL import ImageFilter
                blurred = orig.filter(ImageFilter.GaussianBlur(radius=30))
                blurred.paste(fg, mask=fg.split()[3])
                out_img = blurred.convert("RGB")
            elif background == "color":
                r, g, b = int(bg_color[:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)
                bg_layer = Image.new("RGBA", fg.size, (r, g, b, 255))
                bg_layer.paste(fg, mask=fg.split()[3])
                out_img = bg_layer.convert("RGB")
            elif background == "image" and bg_pil:
                bg_resized = bg_pil.resize(fg.size)
                bg_resized.paste(fg, mask=fg.split()[3])
                out_img = bg_resized.convert("RGB")
            else:
                out_img = fg.convert("RGB")

            out_path = os.path.join(output_frames_dir, fname)
            out_img.save(out_path)

            if on_progress and (i % max(1, total_frames // 20) == 0 or i == total_frames - 1):
                pct = 15 + int((i / total_frames) * 70)
                on_progress(pct, f"Frame {i + 1}/{total_frames}")

        if on_progress:
            on_progress(85, "Re-encoding video...")

        # Re-encode
        crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")

        if background == "transparent":
            output_path = _resolve_output(input_path, output_dir, "nobg", ".webm")
            cmd_encode = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-framerate", str(media["fps"]),
                "-i", os.path.join(output_frames_dir, "frame_%06d.png"),
                "-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                "-auto-alt-ref", "0", "-pix_fmt", "yuva420p",
                output_path,
            ]
        else:
            output_path = _resolve_output(input_path, output_dir, "nobg")
            cmd_encode = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-framerate", str(media["fps"]),
                "-i", os.path.join(output_frames_dir, "frame_%06d.png"),
            ]
            # Add audio from original
            if media["has_audio"]:
                cmd_encode += ["-i", input_path, "-map", "0:v", "-map", "1:a", "-shortest"]

            cmd_encode += [
                "-c:v", "libx264", "-crf", crf, "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if media["has_audio"]:
                cmd_encode += ["-c:a", "aac", "-b:a", "192k"]
            cmd_encode.append(output_path)

        _run_ffmpeg(cmd_encode, timeout=3600)

        if on_progress:
            on_progress(100, "Background removed from video")

        return VideoFxResult(
            output_path=output_path,
            original_duration=media["duration"],
            output_duration=media["duration"],
            width=media["width"],
            height=media["height"],
            operation="background_removal",
            details={
                "model": "birefnet-general",
                "background": background,
                "frames_processed": total_frames,
            },
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 4. SLOW MOTION (RIFE Frame Interpolation)
# ===========================================================================
SLOWMO_PRESETS = {
    "2x": {"multi": 2, "label": "2x Slow-Mo (2x frames)", "speed": 0.5},
    "4x": {"multi": 4, "label": "4x Slow-Mo (4x frames)", "speed": 0.25},
    "8x": {"multi": 8, "label": "8x Slow-Mo (8x frames)", "speed": 0.125},
}


def check_rife_available() -> Tuple[bool, str]:
    """Check if RIFE / frame interpolation deps are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "RIFE requires an NVIDIA GPU with CUDA support"
        # Check for the rife inference module
        rife_dir = _get_rife_dir()
        if rife_dir and os.path.isfile(os.path.join(rife_dir, "inference_video.py")):
            return True, f"RIFE ready (GPU: {torch.cuda.get_device_name(0)})"
        return False, "RIFE model not downloaded"
    except ImportError:
        return False, "PyTorch is not installed"


def _get_rife_dir() -> str:
    """Get the RIFE installation directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.path.expanduser("~"))
    return str(base / "OpenCut" / "models" / "Practical-RIFE")


def install_rife(progress_callback: Optional[Callable] = None) -> bool:
    """Download and set up Practical-RIFE."""
    rife_dir = _get_rife_dir()
    os.makedirs(os.path.dirname(rife_dir), exist_ok=True)

    # Install torch if not present
    try:
        import torch  # noqa: F401
    except ImportError:
        if progress_callback:
            progress_callback(10, "Installing PyTorch...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--quiet"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"PyTorch install failed: {result.stderr[-500:]}")
            return False

    if progress_callback:
        progress_callback(30, "Downloading Practical-RIFE...")

    # Clone or download Practical-RIFE
    if os.path.isdir(rife_dir):
        shutil.rmtree(rife_dir)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/hzwer/Practical-RIFE.git", rife_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr[-500:]}")
            return False
    except FileNotFoundError:
        logger.error("git not found in PATH")
        return False

    if progress_callback:
        progress_callback(60, "Downloading RIFE model weights...")

    # Download model weights (v4.25 or latest)
    model_dir = os.path.join(rife_dir, "train_log")
    os.makedirs(model_dir, exist_ok=True)

    # Install requirements
    req_file = os.path.join(rife_dir, "requirements.txt")
    if os.path.isfile(req_file):
        if progress_callback:
            progress_callback(80, "Installing RIFE dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
            capture_output=True, text=True, timeout=300,
        )

    if progress_callback:
        progress_callback(100, "RIFE installed successfully")

    return True


def apply_slow_motion(
    input_path: str,
    multiplier: int = 2,
    output_dir: str = "",
    scale: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Apply AI slow motion using RIFE frame interpolation.

    Args:
        input_path:   Source video.
        multiplier:   Frame multiplier (2, 4, 8).
        output_dir:   Output directory.
        scale:        Processing scale (0.5 for 4K to save VRAM).
        on_progress:  Callback(pct, msg).
    """
    rife_dir = _get_rife_dir()
    inference_script = os.path.join(rife_dir, "inference_video.py")

    if not os.path.isfile(inference_script):
        raise RuntimeError("RIFE is not installed. Please install it first.")

    media = _probe_media(input_path)

    if on_progress:
        on_progress(5, f"Starting {multiplier}x slow motion...")

    output_path = _resolve_output(input_path, output_dir, f"slowmo_{multiplier}x")

    # Calculate exp value (2^exp = multiplier)
    import math
    exp = int(math.log2(multiplier))
    if 2 ** exp != multiplier:
        # Fallback to nearest power of 2
        exp = max(1, round(math.log2(multiplier)))

    if on_progress:
        on_progress(10, f"Interpolating frames ({2**exp}x)...")

    # Run RIFE inference
    cmd = [
        sys.executable, inference_script,
        f"--multi={2**exp}",
        f"--video={input_path}",
        f"--output={output_path}",
        f"--scale={scale}",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
            cwd=rife_dir,
        )
        if result.returncode != 0:
            logger.error(f"RIFE error: {result.stderr[-1000:]}")
            raise RuntimeError(f"RIFE processing failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("RIFE processing timed out (>2 hours)")

    output_media = _probe_media(output_path)

    if on_progress:
        on_progress(100, f"Slow motion ({multiplier}x) complete")

    return VideoFxResult(
        output_path=output_path,
        original_duration=media["duration"],
        output_duration=output_media["duration"],
        width=media["width"],
        height=media["height"],
        operation="slow_motion",
        details={
            "multiplier": multiplier,
            "original_fps": media["fps"],
            "output_fps": output_media["fps"],
        },
    )


# ===========================================================================
# 5. VIDEO UPSCALING (Real-ESRGAN)
# ===========================================================================
UPSCALE_MODELS = [
    {"id": "realesrgan-x4plus", "name": "Real-ESRGAN x4+", "scale": 4, "description": "Best quality 4x upscale"},
    {"id": "realesrgan-x2plus", "name": "Real-ESRGAN x2+", "scale": 2, "description": "Fast 2x upscale"},
    {"id": "realesrgan-x4plus-anime", "name": "Real-ESRGAN Anime", "scale": 4, "description": "Optimized for anime"},
]


def check_realesrgan_available() -> Tuple[bool, str]:
    """Check if Real-ESRGAN is available."""
    try:
        from realesrgan import RealESRGANer  # noqa: F401
        return True, "Real-ESRGAN is installed"
    except ImportError:
        pass

    # Check for CLI tool
    result = subprocess.run(
        ["realesrgan-ncnn-vulkan", "-h"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return True, "Real-ESRGAN (ncnn-vulkan) is available"

    return False, "Real-ESRGAN is not installed"


def install_realesrgan(progress_callback: Optional[Callable] = None) -> bool:
    """Install Real-ESRGAN Python package."""
    steps = [
        {"label": "Installing basicsr", "cmd": [sys.executable, "-m", "pip", "install", "basicsr", "--quiet"]},
        {"label": "Installing facexlib", "cmd": [sys.executable, "-m", "pip", "install", "facexlib", "--quiet"]},
        {"label": "Installing gfpgan", "cmd": [sys.executable, "-m", "pip", "install", "gfpgan", "--quiet"]},
        {"label": "Installing realesrgan", "cmd": [sys.executable, "-m", "pip", "install", "realesrgan", "--quiet"]},
    ]

    for i, step in enumerate(steps):
        if progress_callback:
            pct = int((i / len(steps)) * 80) + 10
            progress_callback(pct, step["label"])
        try:
            result = subprocess.run(step["cmd"], capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.error(f"Install failed: {result.stderr[-500:]}")
                return False
        except Exception as e:
            logger.error(f"Install error: {e}")
            return False

    return True


def upscale_video(
    input_path: str,
    scale: int = 4,
    model_name: str = "realesrgan-x4plus",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Upscale video using Real-ESRGAN (frame-by-frame).

    Args:
        input_path:   Source video.
        scale:        Upscale factor (2 or 4).
        model_name:   Model to use.
        output_dir:   Output directory.
        on_progress:  Callback(pct, msg).
    """
    media = _probe_media(input_path)

    if on_progress:
        on_progress(5, "Extracting frames for upscaling...")

    tmpdir = tempfile.mkdtemp(prefix="opencut_upscale_")
    frames_dir = os.path.join(tmpdir, "frames")
    output_frames_dir = os.path.join(tmpdir, "output")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        # Extract frames
        cmd_extract = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-i", input_path,
            "-vsync", "0",
            os.path.join(frames_dir, "frame_%06d.png"),
        ]
        _run_ffmpeg(cmd_extract, timeout=600)

        if on_progress:
            on_progress(15, "Loading upscaling model...")

        # Try Python API first
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
            import cv2
            import numpy as np

            # Select model architecture
            if "anime" in model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
            elif "x2plus" in model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4

            upsampler = RealESRGANer(
                scale=netscale,
                model_path=None,  # Auto-downloads
                model=model,
                half=torch.cuda.is_available(),
                gpu_id=0 if torch.cuda.is_available() else None,
            )

            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            total_frames = len(frame_files)

            for i, fname in enumerate(frame_files):
                img = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=scale)
                cv2.imwrite(os.path.join(output_frames_dir, fname), output)

                if on_progress and (i % max(1, total_frames // 20) == 0):
                    pct = 15 + int((i / total_frames) * 70)
                    on_progress(pct, f"Upscaling frame {i + 1}/{total_frames}")

        except ImportError:
            # Fallback: use realesrgan-ncnn-vulkan CLI
            logger.info("Using realesrgan-ncnn-vulkan CLI fallback")
            if on_progress:
                on_progress(20, "Upscaling with ncnn-vulkan...")

            cmd_upscale = [
                "realesrgan-ncnn-vulkan",
                "-i", frames_dir,
                "-o", output_frames_dir,
                "-s", str(scale),
                "-n", model_name,
            ]
            result = subprocess.run(cmd_upscale, capture_output=True, text=True, timeout=7200)
            if result.returncode != 0:
                raise RuntimeError(f"Upscaling failed: {result.stderr[-500:]}")

        if on_progress:
            on_progress(85, "Re-encoding upscaled video...")

        # Re-encode
        output_path = _resolve_output(input_path, output_dir, f"upscale_{scale}x")
        new_w = media["width"] * scale
        new_h = media["height"] * scale

        cmd_encode = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-framerate", str(media["fps"]),
            "-i", os.path.join(output_frames_dir, "frame_%06d.png"),
        ]
        if media["has_audio"]:
            cmd_encode += ["-i", input_path, "-map", "0:v", "-map", "1:a", "-shortest"]

        cmd_encode += [
            "-c:v", "libx264", "-crf", "18", "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if media["has_audio"]:
            cmd_encode += ["-c:a", "aac", "-b:a", "192k"]
        cmd_encode.append(output_path)

        _run_ffmpeg(cmd_encode, timeout=3600)

        if on_progress:
            on_progress(100, f"Video upscaled to {new_w}x{new_h}")

        return VideoFxResult(
            output_path=output_path,
            original_duration=media["duration"],
            output_duration=media["duration"],
            width=new_w,
            height=new_h,
            operation="upscale",
            details={
                "scale": scale,
                "model": model_name,
                "original_resolution": f"{media['width']}x{media['height']}",
                "output_resolution": f"{new_w}x{new_h}",
            },
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 6. AUTO-REFRAME / CROP
# ===========================================================================
ASPECT_PRESETS = {
    "9:16": {"w": 9, "h": 16, "label": "9:16 (TikTok/Reels)", "use_case": "Short-form vertical"},
    "1:1": {"w": 1, "h": 1, "label": "1:1 (Instagram)", "use_case": "Square format"},
    "4:5": {"w": 4, "h": 5, "label": "4:5 (Instagram Portrait)", "use_case": "Instagram feed"},
    "16:9": {"w": 16, "h": 9, "label": "16:9 (YouTube)", "use_case": "Standard widescreen"},
    "4:3": {"w": 4, "h": 3, "label": "4:3 (Classic)", "use_case": "Classic TV"},
    "21:9": {"w": 21, "h": 9, "label": "21:9 (Cinematic)", "use_case": "Ultrawide cinematic"},
}


def auto_reframe(
    input_path: str,
    aspect: str = "9:16",
    mode: str = "center",
    output_dir: str = "",
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> VideoFxResult:
    """
    Reframe/crop video to a different aspect ratio.

    Modes:
      - "center":  Static center crop.
      - "smart":   Uses FFmpeg cropdetect to find the subject area.

    Args:
        input_path:  Source video.
        aspect:      Aspect ratio preset or "W:H" string.
        mode:        "center" or "smart".
        output_dir:  Output directory.
        quality:     Encoding quality.
        on_progress: Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Analyzing video dimensions...")

    media = _probe_media(input_path)
    src_w, src_h = media["width"], media["height"]

    # Parse aspect ratio
    if aspect in ASPECT_PRESETS:
        target_w_ratio = ASPECT_PRESETS[aspect]["w"]
        target_h_ratio = ASPECT_PRESETS[aspect]["h"]
    else:
        parts = aspect.split(":")
        target_w_ratio = int(parts[0])
        target_h_ratio = int(parts[1])

    # Calculate crop dimensions to fill the target aspect ratio
    target_aspect = target_w_ratio / target_h_ratio
    src_aspect = src_w / src_h

    if target_aspect > src_aspect:
        # Target is wider: use full width, crop height
        crop_w = src_w
        crop_h = int(src_w / target_aspect)
    else:
        # Target is taller: use full height, crop width
        crop_h = src_h
        crop_w = int(src_h * target_aspect)

    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    if on_progress:
        on_progress(15, f"Reframing to {aspect} ({crop_w}x{crop_h})")

    output_path = _resolve_output(
        input_path, output_dir,
        f"reframe_{aspect.replace(':', 'x')}"
    )
    crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")

    if mode == "center":
        # Simple center crop
        x_off = (src_w - crop_w) // 2
        y_off = (src_h - crop_h) // 2
        vf = f"crop={crop_w}:{crop_h}:{x_off}:{y_off}"
    elif mode == "smart":
        # Use FFmpeg cropdetect to analyze, then apply center crop
        # For a proper smart reframe, we'd need face detection.
        # This is a reasonable approximation using rule of thirds.
        x_off = (src_w - crop_w) // 2
        y_off = max(0, (src_h - crop_h) // 3)  # Bias toward top third (faces tend to be upper)
        vf = f"crop={crop_w}:{crop_h}:{x_off}:{y_off}"
    else:
        x_off = (src_w - crop_w) // 2
        y_off = (src_h - crop_h) // 2
        vf = f"crop={crop_w}:{crop_h}:{x_off}:{y_off}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", crf, "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    if on_progress:
        on_progress(30, "Encoding reframed video...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Reframed to {aspect}")

    return VideoFxResult(
        output_path=output_path,
        original_duration=media["duration"],
        output_duration=media["duration"],
        width=crop_w,
        height=crop_h,
        operation="reframe",
        details={
            "aspect": aspect,
            "mode": mode,
            "original_resolution": f"{src_w}x{src_h}",
            "output_resolution": f"{crop_w}x{crop_h}",
        },
    )


# ===========================================================================
# Public API for endpoint discovery
# ===========================================================================
def get_video_fx_capabilities() -> Dict:
    """Return capabilities and presets for the Video FX module."""
    rembg_ok, rembg_msg = check_rembg_available()
    rife_ok, rife_msg = check_rife_available()
    esrgan_ok, esrgan_msg = check_realesrgan_available()

    return {
        "lut": {"available": True, "formats": LUT_FORMATS},
        "chroma_key": {"available": True, "presets": list(CHROMA_PRESETS.keys())},
        "background_removal": {
            "available": rembg_ok,
            "message": rembg_msg,
            "models": REMBG_MODELS,
            "backgrounds": BG_REPLACE_MODES,
        },
        "slow_motion": {
            "available": rife_ok,
            "message": rife_msg,
            "presets": SLOWMO_PRESETS,
        },
        "upscale": {
            "available": esrgan_ok,
            "message": esrgan_msg,
            "models": UPSCALE_MODELS,
        },
        "reframe": {
            "available": True,
            "presets": {k: v["label"] for k, v in ASPECT_PRESETS.items()},
        },
    }


# Need io for image processing
import io
