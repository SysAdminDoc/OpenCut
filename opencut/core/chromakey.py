"""
OpenCut Chromakey & Compositing Module v0.9.0

Green/blue screen removal and video compositing:
- Chromakey via HSV thresholding (green, blue, custom color)
- Spill suppression and edge refinement
- Alpha blending / overlay compositing
- Blend modes (multiply, screen, overlay, soft light, etc.)
- Picture-in-picture positioning

All via OpenCV + FFmpeg - zero ML dependencies.
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
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg,
                        "--break-system-packages", "-q"], capture_output=True, timeout=300)


def _run_ffmpeg(cmd, timeout=7200):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {r.stderr.decode(errors='replace')[-500:]}")


def _get_video_info(fp):
    import json
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration",
                        "-of", "json", fp], capture_output=True, timeout=30)
    try:
        s = json.loads(r.stdout.decode())["streams"][0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_p[0]) / float(fps_p[1]) if len(fps_p) == 2 else 30.0
        return {"width": int(s.get("width", 1920)), "height": int(s.get("height", 1080)),
                "fps": fps, "duration": float(s.get("duration", 0))}
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0}


# ---------------------------------------------------------------------------
# Chromakey Presets
# ---------------------------------------------------------------------------
CHROMA_PRESETS = {
    "green": {"lower": (35, 80, 40), "upper": (85, 255, 255), "label": "Green Screen"},
    "blue": {"lower": (90, 80, 40), "upper": (130, 255, 255), "label": "Blue Screen"},
    "red": {"lower": (0, 80, 40), "upper": (15, 255, 255), "label": "Red Screen"},
}


def chromakey_video(
    fg_path: str,
    bg_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    color: str = "green",
    tolerance: float = 0.5,
    spill_suppress: float = 0.5,
    edge_blur: int = 3,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Remove chroma key background and composite over another video/image.

    Args:
        fg_path: Foreground video (with green/blue screen).
        bg_path: Background video or image.
        color: Preset name or "green", "blue", "red".
        tolerance: Key tolerance 0-1 (wider range = more removed).
        spill_suppress: Spill suppression strength 0-1.
        edge_blur: Edge feathering radius in pixels.
    """
    _ensure_package("cv2", "opencv-python-headless", on_progress)
    import cv2
    import numpy as np

    if output_path is None:
        base = os.path.splitext(os.path.basename(fg_path))[0]
        directory = output_dir or os.path.dirname(fg_path)
        output_path = os.path.join(directory, f"{base}_keyed.mp4")

    if on_progress:
        on_progress(5, "Setting up chromakey...")

    fg_info = _get_video_info(fg_path)
    w, h, fps = fg_info["width"], fg_info["height"], fg_info["fps"]

    # Get HSV range
    preset = CHROMA_PRESETS.get(color, CHROMA_PRESETS["green"])
    lower = list(preset["lower"])
    upper = list(preset["upper"])
    # Adjust by tolerance
    tol_adj = int(tolerance * 30)
    lower[1] = max(0, lower[1] - tol_adj)
    lower[2] = max(0, lower[2] - tol_adj)
    upper[1] = min(255, upper[1] + tol_adj)
    upper[2] = min(255, upper[2] + tol_adj)
    lower_hsv = np.array(lower, dtype=np.uint8)
    upper_hsv = np.array(upper, dtype=np.uint8)

    fg_cap = cv2.VideoCapture(fg_path)
    bg_is_video = bg_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    bg_cap = cv2.VideoCapture(bg_path) if bg_is_video else None
    bg_img = None if bg_is_video else cv2.imread(bg_path)
    if bg_img is not None:
        bg_img = cv2.resize(bg_img, (w, h))

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    total_frames = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_idx = 0

    try:
        while True:
            ret, fg_frame = fg_cap.read()
            if not ret:
                break

            # Get background frame
            if bg_cap is not None:
                ret_bg, bg_frame = bg_cap.read()
                if not ret_bg:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, bg_frame = bg_cap.read()
                bg_frame = cv2.resize(bg_frame, (w, h))
            else:
                bg_frame = bg_img.copy()

            # Create mask
            hsv = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask = cv2.bitwise_not(mask)

            # Edge refinement
            if edge_blur > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_blur * 2 + 1,) * 2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask, (edge_blur * 2 + 1,) * 2, 0)

            # Spill suppression
            if spill_suppress > 0 and color == "green":
                fg_float = fg_frame.astype(np.float32)
                g_spill = fg_float[:, :, 1] - np.maximum(fg_float[:, :, 0], fg_float[:, :, 2])
                g_spill = np.clip(g_spill * spill_suppress, 0, 255)
                fg_frame[:, :, 1] = np.clip(fg_float[:, :, 1] - g_spill, 0, 255).astype(np.uint8)

            # Composite
            mask_3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            result = (fg_frame.astype(np.float32) * mask_3 +
                      bg_frame.astype(np.float32) * (1 - mask_3))
            writer.write(result.astype(np.uint8))

            frame_idx += 1
            if on_progress and frame_idx % 30 == 0:
                pct = 5 + int((frame_idx / total_frames) * 85)
                on_progress(pct, f"Keying frame {frame_idx}/{total_frames}...")

    finally:
        fg_cap.release()
        if bg_cap:
            bg_cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding final video with audio...")

    # Mux with audio from foreground
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_video, "-i", fg_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])
    os.unlink(tmp_video)

    if on_progress:
        on_progress(100, "Chromakey complete!")
    return output_path


# ---------------------------------------------------------------------------
# Picture-in-Picture
# ---------------------------------------------------------------------------
def picture_in_picture(
    main_path: str,
    pip_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    position: str = "bottom_right",
    scale: float = 0.25,
    margin: int = 20,
    border: int = 2,
    on_progress: Optional[Callable] = None,
) -> str:
    """Overlay a PiP video on the main video via FFmpeg."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(main_path))[0]
        directory = output_dir or os.path.dirname(main_path)
        output_path = os.path.join(directory, f"{base}_pip.mp4")

    info = _get_video_info(main_path)
    pip_w = int(info["width"] * scale)

    pos_map = {
        "top_left": f"{margin}:{margin}",
        "top_right": f"W-w-{margin}:{margin}",
        "bottom_left": f"{margin}:H-h-{margin}",
        "bottom_right": f"W-w-{margin}:H-h-{margin}",
        "center": "W/2-w/2:H/2-h/2",
    }
    pos = pos_map.get(position, pos_map["bottom_right"])

    if on_progress:
        on_progress(10, "Creating picture-in-picture...")

    fc = f"[1:v]scale={pip_w}:-1[pip];[0:v][pip]overlay={pos}:shortest=1"
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", main_path, "-i", pip_path,
        "-filter_complex", fc,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ])

    if on_progress:
        on_progress(100, "PiP complete!")
    return output_path


# ---------------------------------------------------------------------------
# Blend Modes (FFmpeg blend filter)
# ---------------------------------------------------------------------------
BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "darken", "lighten",
    "softlight", "hardlight", "difference", "exclusion", "addition",
    "dodge", "burn", "average",
]


def blend_videos(
    base_path: str,
    overlay_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    mode: str = "overlay",
    opacity: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """Blend two videos using a blend mode."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(base_path))[0]
        directory = output_dir or os.path.dirname(base_path)
        output_path = os.path.join(directory, f"{base}_blend_{mode}.mp4")

    if on_progress:
        on_progress(10, f"Blending with {mode} mode...")

    info = _get_video_info(base_path)
    fc = (
        f"[1:v]scale={info['width']}:{info['height']}[ov];"
        f"[0:v][ov]blend=all_mode={mode}:all_opacity={opacity}"
    )
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", base_path, "-i", overlay_path,
        "-filter_complex", fc,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", output_path,
    ])

    if on_progress:
        on_progress(100, f"Blend ({mode}) complete!")
    return output_path
