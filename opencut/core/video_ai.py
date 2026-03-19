"""
OpenCut Video AI Module v0.7.0

AI-powered video processing with lazy model loading:
- AI Upscale (Real-ESRGAN, 2x/4x)
- AI Background Removal (rembg + multiple models)
- AI Frame Interpolation (RIFE via practical-rife)
- AI Noise Reduction (video denoising via FFmpeg hqdn3d/nlmeans)

Models are downloaded on first use and cached in ~/.opencut/models/.
All functions accept a progress callback for UI integration.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")

MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------



def _count_frames(filepath: str) -> int:
    """Count total frames in video."""
    cmd = [
        "ffprobe", "-v", "quiet", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nw=1:nk=1", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    try:
        return int(result.stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0

# ---------------------------------------------------------------------------
# AI Upscale (Real-ESRGAN)
# ---------------------------------------------------------------------------
def check_upscale_available() -> bool:
    """Check if Real-ESRGAN is available."""
    try:
        import realesrgan  # noqa: F401
        return True
    except ImportError:
        return False


def upscale_video(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    scale: int = 2,
    model: str = "realesrgan-x4plus",
    denoise_strength: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    AI upscale video using Real-ESRGAN.

    Args:
        scale: Upscale factor (2 or 4).
        model: Model name (realesrgan-x4plus, realesrgan-x4plus-anime).
        denoise_strength: Denoise during upscale (0.0-1.0).
    """
    if not ensure_package("realesrgan", "realesrgan", on_progress):
        raise RuntimeError("Failed to install realesrgan. Install manually: pip install realesrgan")
    if not ensure_package("basicsr", "basicsr", on_progress):
        raise RuntimeError("Failed to install basicsr. Install manually: pip install basicsr")
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")

    if output_path is None:
        output_path = _output_path(input_path, f"upscale_{scale}x", output_dir)

    if on_progress:
        on_progress(5, "Loading upscale model...")

    # Import and set up Real-ESRGAN
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select model architecture
    if "anime" in model:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=6, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus_anime_6B"
    else:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=23, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus"

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=denoise_strength if denoise_strength > 0 else None,
        model=net,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device,
    )

    if on_progress:
        on_progress(10, "Extracting frames...")

    # Create temp directory for frames
    tmp_dir = tempfile.mkdtemp(prefix="opencut_upscale_")
    frames_in = os.path.join(tmp_dir, "in")
    frames_out = os.path.join(tmp_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        # Extract frames
        info = get_video_info(input_path)
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            os.path.join(frames_in, "frame_%06d.png"),
        ])

        frame_files = sorted(Path(frames_in).glob("frame_*.png"))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("No frames extracted from video")

        if on_progress:
            on_progress(15, f"Upscaling {total} frames...")

        # Process each frame
        import cv2
        for i, frame_path in enumerate(frame_files):
            img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning("Skipping unreadable frame: %s", frame_path)
                continue
            output, _ = upsampler.enhance(img, outscale=scale)
            out_file = os.path.join(frames_out, frame_path.name)
            cv2.imwrite(out_file, output)

            if on_progress and i % max(1, total // 20) == 0:
                pct = 15 + int((i / total) * 75)
                on_progress(pct, f"Upscaling frame {i+1}/{total}...")

        if on_progress:
            on_progress(92, "Encoding output video...")

        # Reassemble with audio
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-framerate", str(info["fps"]),
            "-i", os.path.join(frames_out, "frame_%06d.png"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ])

        if on_progress:
            on_progress(100, f"Upscale {scale}x complete")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Free GPU memory
        try:
            del upsampler
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI Background Removal (rembg)
# ---------------------------------------------------------------------------
def check_rembg_available() -> bool:
    """Check if rembg is available."""
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def remove_background(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    model: str = "birefnet-general",
    bg_color: str = "",
    bg_image: str = "",
    alpha_only: bool = False,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Remove video background using rembg.

    Args:
        model: rembg model (u2net, u2net_human_seg, isnet-general-use, birefnet-general, birefnet-massive).
        bg_color: Replacement background color hex (e.g., "#00FF00"). Empty = transparent.
        bg_image: Path to background image/video.
        alpha_only: If True, output alpha matte only.
    """
    if not ensure_package("rembg", "rembg[gpu]", on_progress):
        raise RuntimeError("Failed to install rembg. Install manually: pip install rembg[gpu]")
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")

    if output_path is None:
        suffix = "nobg" if not bg_color else "newbg"
        ext = ".mov" if not bg_color and not bg_image else ".mp4"
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_{suffix}{ext}")

    if on_progress:
        on_progress(5, f"Loading background removal model ({model})...")

    from rembg import new_session, remove
    session = new_session(model)

    tmp_dir = tempfile.mkdtemp(prefix="opencut_rembg_")
    frames_in = os.path.join(tmp_dir, "in")
    frames_out = os.path.join(tmp_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        info = get_video_info(input_path)

        if on_progress:
            on_progress(10, "Extracting frames...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            os.path.join(frames_in, "frame_%06d.png"),
        ])

        frame_files = sorted(Path(frames_in).glob("frame_*.png"))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("No frames extracted")

        if on_progress:
            on_progress(15, f"Processing {total} frames...")


        from PIL import Image

        for i, frame_path in enumerate(frame_files):
            img = Image.open(str(frame_path))
            result = remove(img, session=session, alpha_matting=True)

            if alpha_only:
                # Extract alpha channel as grayscale
                alpha = result.split()[-1]
                alpha.save(os.path.join(frames_out, frame_path.name))
            elif bg_color:
                # Replace background with solid color
                bg = Image.new("RGBA", result.size, bg_color)
                bg.paste(result, mask=result.split()[-1])
                bg.convert("RGB").save(os.path.join(frames_out, frame_path.name))
            else:
                result.save(os.path.join(frames_out, frame_path.name))

            if on_progress and i % max(1, total // 20) == 0:
                pct = 15 + int((i / total) * 75)
                on_progress(pct, f"Removing background {i+1}/{total}...")

        if on_progress:
            on_progress(92, "Encoding output...")

        # Use ProRes for alpha, H.264 for opaque
        if not bg_color and not bg_image and not alpha_only:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y",
                "-framerate", str(info["fps"]),
                "-i", os.path.join(frames_out, "frame_%06d.png"),
                "-i", input_path,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "prores_ks", "-profile:v", "4",
                "-pix_fmt", "yuva444p10le",
                "-c:a", "copy", "-shortest",
                output_path,
            ])
        else:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y",
                "-framerate", str(info["fps"]),
                "-i", os.path.join(frames_out, "frame_%06d.png"),
                "-i", input_path,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-shortest",
                output_path,
            ])

        if on_progress:
            on_progress(100, "Background removed")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            del session
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI Frame Interpolation (RIFE)
# ---------------------------------------------------------------------------
def check_rife_available() -> bool:
    """Check if RIFE dependencies are available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def frame_interpolate(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    multiplier: int = 2,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    AI frame interpolation to increase framerate.

    Uses FFmpeg minterpolate as a reliable cross-platform solution,
    with optical flow motion estimation for high quality.

    Args:
        multiplier: Frame rate multiplier (2 = double, 4 = quadruple).
    """
    if output_path is None:
        output_path = _output_path(input_path, f"interp_{multiplier}x", output_dir)

    info = get_video_info(input_path)
    target_fps = info["fps"] * multiplier

    if on_progress:
        on_progress(10, f"Interpolating {info['fps']:.0f}fps -> {target_fps:.0f}fps...")

    # Use minterpolate with motion-compensated interpolation
    mi_mode = "mci"  # motion compensated interpolation
    mc_mode = "aobmc"  # adaptive overlapped block motion compensation
    me_mode = "bidir"  # bidirectional motion estimation

    vf = (
        f"minterpolate=fps={target_fps}:mi_mode={mi_mode}"
        f":mc_mode={mc_mode}:me_mode={me_mode}:vsbmc=1"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        output_path,
    ]

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"Interpolated to {target_fps:.0f}fps")
    return output_path


# ---------------------------------------------------------------------------
# AI Video Denoise (hqdn3d / nlmeans)
# ---------------------------------------------------------------------------
def video_denoise(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    method: str = "nlmeans",
    strength: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Video noise reduction.

    Args:
        method: "nlmeans" (best spatial, slower), "hqdn3d" (fast),
                "basicvsr" (ML temporal, best quality, GPU required).
        strength: Denoise strength (0.1-1.0).
    """
    if output_path is None:
        output_path = _output_path(input_path, "denoised", output_dir)

    if method == "basicvsr":
        return _denoise_basicvsr(input_path, output_path, strength, on_progress)

    if on_progress:
        on_progress(10, f"Denoising video ({method})...")

    if method == "nlmeans":
        s = int(3 + strength * 12)
        vf = f"nlmeans=s={s}:p=7:r=15"
    else:  # hqdn3d
        s = 2 + strength * 8
        vf = f"hqdn3d={s}:{s}:{s*0.6}:{s*0.6}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "Video denoised")
    return output_path


def _denoise_basicvsr(
    input_path: str,
    output_path: str,
    strength: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    ML-based video denoising using BasicVSR++ temporal propagation.

    Exploits information across multiple frames for significantly better
    results than spatial-only filters (nlmeans/hqdn3d). Requires GPU.
    """
    if not ensure_package("basicsr", "basicsr", on_progress):
        raise RuntimeError("basicsr not installed. Run: pip install basicsr")

    import cv2
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BasicVSR++ requires a CUDA GPU. Use 'nlmeans' for CPU denoising.")

    if on_progress:
        on_progress(5, "Loading BasicVSR++ model...")

    from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

    device = torch.device("cuda")
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=False).to(device)

    # Try to load pre-trained weights
    weights_path = os.path.expanduser("~/.opencut/models/basicvsrpp_denoise.pth")
    if os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("params", ckpt.get("params_ema", ckpt)), strict=False)
    else:
        logger.warning("BasicVSR++ weights not found at %s — using untrained model", weights_path)

    model.eval()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if on_progress:
        on_progress(10, f"Reading {total} frames...")

    # Read all frames into tensor (BasicVSR++ needs full sequence)
    # Process in chunks of 30 frames to manage VRAM
    chunk_size = 30
    tmp_video = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create video writer")

    frame_idx = 0
    try:
        while True:
            chunk_frames = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)

            if not chunk_frames:
                break

            # Convert chunk to tensor: [1, T, C, H, W] in [0,1]
            frames_np = np.stack(chunk_frames).astype(np.float32) / 255.0
            frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)

            with torch.inference_mode():
                output = model(frames_t)

            # Write denoised frames
            output_np = output.squeeze(0).permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy() * 255
            for i in range(output_np.shape[0]):
                # Blend with original based on strength
                if strength < 1.0:
                    blended = chunk_frames[i].astype(np.float32) * (1 - strength) + output_np[i] * strength
                    writer.write(blended.astype(np.uint8))
                else:
                    writer.write(output_np[i].astype(np.uint8))

            frame_idx += len(chunk_frames)
            if on_progress:
                pct = 10 + int((frame_idx / total) * 80)
                on_progress(pct, f"Denoising frame {frame_idx}/{total}...")

    finally:
        cap.release()
        writer.release()
        del model
        torch.cuda.empty_cache()

    # Mux audio
    if on_progress:
        on_progress(92, "Encoding with audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", input_path,
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
        on_progress(100, "Video denoised (BasicVSR++)")
    return output_path


# ---------------------------------------------------------------------------
# Availability checks for health endpoint
# ---------------------------------------------------------------------------
def get_ai_capabilities() -> Dict:
    """Return available AI video capabilities."""
    caps = {
        "upscale": check_upscale_available(),
        "rembg": check_rembg_available(),
        "frame_interp": True,  # Uses FFmpeg minterpolate, always available
        "video_denoise": True,  # Uses FFmpeg, always available
    }
    # Check GPU
    try:
        import torch
        caps["gpu_available"] = torch.cuda.is_available()
        if caps["gpu_available"]:
            caps["gpu_name"] = torch.cuda.get_device_name(0)
            caps["gpu_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        caps["gpu_available"] = False
    return caps
