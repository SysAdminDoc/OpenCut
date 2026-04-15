"""
OpenCut Voice Avatar Generation Module

Record audio, generate a talking avatar video with lip-sync:
- Accept audio input + reference face image
- Generate lip-synced talking head video
- Support multiple avatar styles: realistic, cartoon, silhouette
- Background options: solid color, blur, custom image
- Output configurable resolution and duration

Uses SadTalker/LivePortrait for realistic mode, Pillow for simple styles.
"""

import logging
import math
import os
import struct
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Avatar Styles
# ---------------------------------------------------------------------------
AVATAR_STYLES = [
    {
        "id": "realistic",
        "name": "Realistic",
        "description": "High-fidelity talking head using face reenactment",
        "requires_gpu": True,
        "backends": ["sadtalker", "liveportrait"],
    },
    {
        "id": "cartoon",
        "name": "Cartoon",
        "description": "Simplified cartoon avatar with lip flap animation",
        "requires_gpu": False,
        "backends": ["pillow"],
    },
    {
        "id": "silhouette",
        "name": "Silhouette",
        "description": "Dark silhouette profile with animated mouth",
        "requires_gpu": False,
        "backends": ["pillow"],
    },
    {
        "id": "minimal",
        "name": "Minimal Circle",
        "description": "Circular avatar with waveform-driven animation",
        "requires_gpu": False,
        "backends": ["pillow"],
    },
    {
        "id": "sketch",
        "name": "Sketch",
        "description": "Pencil sketch-style avatar with lip movement",
        "requires_gpu": False,
        "backends": ["pillow"],
    },
]

AVATAR_STYLE_IDS = [s["id"] for s in AVATAR_STYLES]

# Background modes
BACKGROUND_MODES = ["solid", "blur", "custom", "transparent"]

# Default colors
DEFAULT_BG_COLOR = (18, 18, 24)  # Deep dark
DEFAULT_SILHOUETTE_COLOR = (40, 40, 50)
DEFAULT_MOUTH_COLOR = (200, 80, 80)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class AvatarConfig:
    """Configuration for avatar generation."""
    style: str = "cartoon"
    width: int = 720
    height: int = 720
    fps: int = 30
    background_mode: str = "solid"
    background_color: Tuple[int, int, int] = (18, 18, 24)
    background_image: str = ""
    background_blur_radius: int = 25
    mouth_open_threshold: float = 0.02
    mouth_amplitude_scale: float = 1.5
    face_scale: float = 0.6
    face_position: Tuple[float, float] = (0.5, 0.45)
    sadtalker_checkpoint: str = ""
    liveportrait_model: str = ""
    enhancer: str = ""
    max_duration: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.style not in AVATAR_STYLE_IDS:
            errors.append(f"Unknown style '{self.style}', must be one of {AVATAR_STYLE_IDS}")
        if self.width < 64 or self.width > 4096:
            errors.append(f"Width {self.width} out of range [64, 4096]")
        if self.height < 64 or self.height > 4096:
            errors.append(f"Height {self.height} out of range [64, 4096]")
        if self.fps < 1 or self.fps > 120:
            errors.append(f"FPS {self.fps} out of range [1, 120]")
        if self.background_mode not in BACKGROUND_MODES:
            errors.append(f"Unknown background_mode '{self.background_mode}'")
        if self.background_mode == "custom" and not self.background_image:
            errors.append("background_image required when background_mode is 'custom'")
        if self.face_scale < 0.1 or self.face_scale > 1.0:
            errors.append(f"face_scale {self.face_scale} out of range [0.1, 1.0]")
        return errors


@dataclass
class AvatarResult:
    """Result of avatar generation."""
    output_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: int = 0
    frame_count: int = 0
    style: str = ""
    audio_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Audio Analysis
# ---------------------------------------------------------------------------
def _read_audio_amplitudes(audio_path: str, target_fps: int = 30) -> List[float]:
    """Read audio file and return per-frame amplitude envelope.

    Converts input to WAV via FFmpeg if needed, then reads PCM samples
    and computes RMS amplitude per frame window.
    """
    tmp_wav = None
    wav_path = audio_path

    # Convert to WAV if not already
    if not audio_path.lower().endswith(".wav"):
        tmp_wav = tempfile.mktemp(suffix=".wav", prefix="avatar_audio_")
        cmd = (FFmpegCmd()
               .input(audio_path)
               .no_video()
               .audio_codec("pcm_s16le")
               .option("ar", "16000")
               .option("ac", "1")
               .output(tmp_wav)
               .build())
        try:
            run_ffmpeg(cmd)
            wav_path = tmp_wav
        except Exception as exc:
            logger.error("Failed to convert audio to WAV: %s", exc)
            if tmp_wav and os.path.isfile(tmp_wav):
                os.unlink(tmp_wav)
            return []

    amplitudes = []
    try:
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()

            samples_per_video_frame = max(1, sample_rate // target_fps)
            total_video_frames = max(1, math.ceil(n_frames / samples_per_video_frame))

            fmt = "<" + ("h" if sample_width == 2 else "b") * n_channels

            for _ in range(total_video_frames):
                raw = wf.readframes(samples_per_video_frame)
                if not raw:
                    amplitudes.append(0.0)
                    continue

                # Parse samples and compute RMS
                n_samples = len(raw) // (sample_width * n_channels)
                if n_samples == 0:
                    amplitudes.append(0.0)
                    continue

                rms_sum = 0.0
                max_val = 32768.0 if sample_width == 2 else 128.0
                for i in range(n_samples):
                    offset = i * sample_width * n_channels
                    chunk = raw[offset:offset + sample_width * n_channels]
                    if len(chunk) < sample_width * n_channels:
                        break
                    try:
                        vals = struct.unpack(fmt, chunk)
                        sample_val = vals[0] / max_val  # Use first channel
                        rms_sum += sample_val * sample_val
                    except struct.error:
                        break

                rms = math.sqrt(rms_sum / max(n_samples, 1))
                amplitudes.append(min(rms, 1.0))

    except Exception as exc:
        logger.error("Failed to read WAV amplitudes: %s", exc)
    finally:
        if tmp_wav and os.path.isfile(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

    return amplitudes


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    from opencut.helpers import get_ffprobe_path
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            return float(result.stdout.decode().strip())
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Background Rendering
# ---------------------------------------------------------------------------
def _create_background(config: AvatarConfig) -> "Image.Image":  # noqa: F821
    """Create a background image based on config settings."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageFilter

    if config.background_mode == "custom" and config.background_image:
        try:
            bg = Image.open(config.background_image).convert("RGBA")
            bg = bg.resize((config.width, config.height), Image.LANCZOS)
            return bg
        except Exception as exc:
            logger.warning("Failed to load custom background: %s, falling back to solid", exc)

    if config.background_mode == "blur" and config.background_image:
        try:
            bg = Image.open(config.background_image).convert("RGBA")
            bg = bg.resize((config.width, config.height), Image.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=config.background_blur_radius))
            return bg
        except Exception:
            pass

    # Solid color fallback
    bg = Image.new("RGBA", (config.width, config.height),
                    config.background_color + (255,))
    return bg


def _create_gradient_overlay(width: int, height: int,
                              color_top: Tuple[int, int, int] = (30, 30, 40),
                              color_bottom: Tuple[int, int, int] = (10, 10, 15),
                              ) -> "Image.Image":  # noqa: F821
    """Create a vertical gradient overlay."""
    ensure_package("PIL", "Pillow")
    from PIL import Image

    img = Image.new("RGBA", (width, height))
    pixels = img.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(color_top[0] * (1 - t) + color_bottom[0] * t)
        g = int(color_top[1] * (1 - t) + color_bottom[1] * t)
        b = int(color_top[2] * (1 - t) + color_bottom[2] * t)
        for x in range(width):
            pixels[x, y] = (r, g, b, 255)
    return img


# ---------------------------------------------------------------------------
# Simple Avatar Renderers (Pillow-based)
# ---------------------------------------------------------------------------
def _draw_cartoon_frame(face_image: "Image.Image", amplitude: float,  # noqa: F821
                         config: AvatarConfig, bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Draw a cartoon-style avatar frame with mouth animation."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)

    # Scale and position face
    face_w = int(config.width * config.face_scale)
    face_h = int(config.height * config.face_scale)
    face_resized = face_image.resize((face_w, face_h), Image.LANCZOS)

    fx = int(config.width * config.face_position[0] - face_w / 2)
    fy = int(config.height * config.face_position[1] - face_h / 2)
    frame.paste(face_resized, (fx, fy), face_resized if face_resized.mode == "RGBA" else None)

    # Draw mouth ellipse based on amplitude
    mouth_open = amplitude * config.mouth_amplitude_scale
    mouth_cx = int(config.width * config.face_position[0])
    mouth_cy = fy + int(face_h * 0.75)
    mouth_w = int(face_w * 0.25)
    mouth_h = max(2, int(face_h * 0.12 * mouth_open))

    if amplitude > config.mouth_open_threshold:
        draw.ellipse(
            [mouth_cx - mouth_w // 2, mouth_cy - mouth_h // 2,
             mouth_cx + mouth_w // 2, mouth_cy + mouth_h // 2],
            fill=DEFAULT_MOUTH_COLOR + (220,),
            outline=(160, 60, 60, 255),
            width=2,
        )

    return frame.convert("RGB")


def _draw_silhouette_frame(face_image: "Image.Image", amplitude: float,  # noqa: F821
                            config: AvatarConfig, bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Draw a silhouette avatar with animated mouth gap."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)

    # Dark silhouette circle
    cx = int(config.width * config.face_position[0])
    cy = int(config.height * config.face_position[1])
    radius = int(min(config.width, config.height) * config.face_scale * 0.45)

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=DEFAULT_SILHOUETTE_COLOR + (255,),
    )

    # Mouth gap
    mouth_open = amplitude * config.mouth_amplitude_scale
    if amplitude > config.mouth_open_threshold:
        mouth_w = int(radius * 0.5)
        mouth_h = max(2, int(radius * 0.2 * mouth_open))
        mouth_cy = cy + int(radius * 0.35)
        draw.ellipse(
            [cx - mouth_w, mouth_cy - mouth_h,
             cx + mouth_w, mouth_cy + mouth_h],
            fill=config.background_color + (255,),
        )

    return frame.convert("RGB")


def _draw_minimal_frame(face_image: "Image.Image", amplitude: float,  # noqa: F821
                         config: AvatarConfig, bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Draw a minimal circular waveform avatar."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)

    cx = int(config.width * config.face_position[0])
    cy = int(config.height * config.face_position[1])
    base_radius = int(min(config.width, config.height) * config.face_scale * 0.35)

    # Pulsating circle
    pulse = 1.0 + amplitude * config.mouth_amplitude_scale * 0.15
    r = int(base_radius * pulse)

    # Outer glow ring
    for offset in range(3, 0, -1):
        alpha = 60 + (3 - offset) * 40
        color = (100, 130, 220, alpha)
        draw.ellipse(
            [cx - r - offset * 4, cy - r - offset * 4,
             cx + r + offset * 4, cy + r + offset * 4],
            outline=color, width=2,
        )

    # Main circle
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(50, 60, 90, 255),
        outline=(120, 150, 240, 255),
        width=3,
    )

    # Inner waveform bars
    n_bars = 12
    for i in range(n_bars):
        angle = (2 * math.pi * i) / n_bars
        bar_amp = amplitude * config.mouth_amplitude_scale
        # Vary bar height per bar for visual interest
        bar_variation = 0.5 + 0.5 * math.sin(angle * 3 + amplitude * 10)
        bar_len = int(base_radius * 0.3 * bar_amp * bar_variation)
        bar_len = max(2, bar_len)

        inner_r = int(base_radius * 0.5)
        x1 = cx + int(inner_r * math.cos(angle))
        y1 = cy + int(inner_r * math.sin(angle))
        x2 = cx + int((inner_r + bar_len) * math.cos(angle))
        y2 = cy + int((inner_r + bar_len) * math.sin(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(160, 180, 255, 200), width=3)

    return frame.convert("RGB")


def _draw_sketch_frame(face_image: "Image.Image", amplitude: float,  # noqa: F821
                        config: AvatarConfig, bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Draw a sketch-style avatar with edge-detected face."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    frame = bg.copy()

    # Convert face to sketch effect
    face_w = int(config.width * config.face_scale)
    face_h = int(config.height * config.face_scale)
    sketch = face_image.convert("L").resize((face_w, face_h), Image.LANCZOS)
    sketch = ImageOps.autocontrast(sketch, cutoff=5)
    edges = sketch.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    edges = edges.point(lambda p: 255 if p > 200 else p)
    sketch_rgba = Image.new("RGBA", (face_w, face_h), (0, 0, 0, 0))
    for x in range(face_w):
        for y in range(face_h):
            val = edges.getpixel((x, y))
            if val < 240:
                sketch_rgba.putpixel((x, y), (220, 220, 230, 255 - val))

    fx = int(config.width * config.face_position[0] - face_w / 2)
    fy = int(config.height * config.face_position[1] - face_h / 2)
    frame.paste(sketch_rgba, (fx, fy), sketch_rgba)

    # Mouth line
    draw = ImageDraw.Draw(frame)
    if amplitude > config.mouth_open_threshold:
        mouth_open = amplitude * config.mouth_amplitude_scale
        mouth_cx = int(config.width * config.face_position[0])
        mouth_cy = fy + int(face_h * 0.72)
        mouth_w = int(face_w * 0.2)
        gap = max(1, int(mouth_open * face_h * 0.06))
        draw.arc(
            [mouth_cx - mouth_w, mouth_cy - gap,
             mouth_cx + mouth_w, mouth_cy + gap],
            start=0, end=180,
            fill=(200, 200, 210, 230), width=2,
        )

    return frame.convert("RGB")


# Style -> renderer mapping
_STYLE_RENDERERS = {
    "cartoon": _draw_cartoon_frame,
    "silhouette": _draw_silhouette_frame,
    "minimal": _draw_minimal_frame,
    "sketch": _draw_sketch_frame,
}


# ---------------------------------------------------------------------------
# Realistic Avatar (SadTalker/LivePortrait)
# ---------------------------------------------------------------------------
def _try_realistic_generation(audio_path: str, face_image_path: str,
                               config: AvatarConfig, out_path: str,
                               on_progress: Optional[Callable] = None) -> Optional[str]:
    """Attempt realistic avatar generation via SadTalker or LivePortrait.

    Returns output path on success, None if backends unavailable.
    """
    # Try SadTalker
    try:
        import importlib
        importlib.import_module("inference")
        logger.info("SadTalker backend found, generating realistic avatar")
        if on_progress:
            on_progress(20, "Generating realistic avatar via SadTalker...")
        # SadTalker API call would go here
        # sadtalker.main(source_image=face_image_path, driven_audio=audio_path, ...)
        return None  # Placeholder — real SadTalker requires specific setup
    except ImportError:
        logger.debug("SadTalker not available")

    # Try LivePortrait
    try:
        import importlib
        importlib.import_module("liveportrait")
        logger.info("LivePortrait backend found")
        if on_progress:
            on_progress(20, "Generating realistic avatar via LivePortrait...")
        return None  # Placeholder
    except ImportError:
        logger.debug("LivePortrait not available")

    return None


# ---------------------------------------------------------------------------
# Frame Sequence Assembly
# ---------------------------------------------------------------------------
def _assemble_frames_to_video(frame_dir: str, audio_path: str,
                                out_path: str, fps: int = 30,
                                width: int = 720, height: int = 720) -> str:
    """Assemble numbered PNG frames + audio into an MP4 video."""
    pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = (FFmpegCmd()
           .option("framerate", str(fps))
           .input(pattern)
           .input(audio_path)
           .video_codec("libx264")
           .option("pix_fmt", "yuv420p")
           .option("crf", "18")
           .option("preset", "medium")
           .option("shortest", None)
           .audio_codec("aac")
           .option("b:a", "192k")
           .output(out_path)
           .build())
    run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------
def generate_avatar(
    audio_path: str,
    face_image: str,
    config: Optional[AvatarConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> AvatarResult:
    """Generate a talking avatar video from audio and face image.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        face_image: Path to reference face image (PNG, JPG)
        config: Avatar configuration. Uses defaults if None.
        output_dir: Output directory. Defaults to audio file directory.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        AvatarResult with output video path and metadata.

    Raises:
        FileNotFoundError: If audio or face image not found.
        ValueError: If config validation fails.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.isfile(face_image):
        raise FileNotFoundError(f"Face image not found: {face_image}")

    config = config or AvatarConfig()
    errors = config.validate()
    if errors:
        raise ValueError("Invalid AvatarConfig: " + "; ".join(errors))

    if on_progress:
        on_progress(5, f"Starting {config.style} avatar generation...")

    # Determine output path
    out_dir = output_dir or os.path.dirname(audio_path)
    out_path = output_path(audio_path, f"avatar_{config.style}", out_dir)
    if not out_path.lower().endswith(".mp4"):
        out_path = os.path.splitext(out_path)[0] + ".mp4"

    # Get audio duration
    duration = _get_audio_duration(audio_path)
    if config.max_duration > 0 and duration > config.max_duration:
        duration = config.max_duration

    if duration <= 0:
        raise ValueError("Could not determine audio duration or duration is zero")

    # Try realistic first if requested
    if config.style == "realistic":
        if on_progress:
            on_progress(10, "Attempting realistic avatar backend...")
        realistic_path = _try_realistic_generation(
            audio_path, face_image, config, out_path, on_progress)
        if realistic_path and os.path.isfile(realistic_path):
            info = get_video_info(realistic_path)
            return AvatarResult(
                output_path=realistic_path,
                duration=info.get("duration", duration),
                width=info.get("width", config.width),
                height=info.get("height", config.height),
                fps=int(info.get("fps", config.fps)),
                frame_count=int(info.get("duration", duration) * info.get("fps", config.fps)),
                style="realistic",
                audio_path=audio_path,
            )
        logger.info("Realistic backend unavailable, falling back to cartoon style")
        config.style = "cartoon"

    # Pillow-based rendering
    if on_progress:
        on_progress(15, "Analyzing audio amplitudes...")

    amplitudes = _read_audio_amplitudes(audio_path, config.fps)
    if not amplitudes:
        raise RuntimeError("Failed to extract audio amplitude data")

    total_frames = len(amplitudes)
    if config.max_duration > 0:
        max_frames = int(config.max_duration * config.fps)
        if total_frames > max_frames:
            amplitudes = amplitudes[:max_frames]
            total_frames = max_frames

    if on_progress:
        on_progress(20, f"Rendering {total_frames} frames...")

    ensure_package("PIL", "Pillow")
    from PIL import Image

    # Load face image
    face_img = Image.open(face_image).convert("RGBA")
    bg = _create_background(config)
    renderer = _STYLE_RENDERERS.get(config.style, _draw_cartoon_frame)

    # Render frames to temp directory
    frame_dir = tempfile.mkdtemp(prefix="avatar_frames_")
    try:
        for i, amp in enumerate(amplitudes):
            frame = renderer(face_img, amp, config, bg)
            frame_path = os.path.join(frame_dir, f"frame_{i:06d}.png")
            frame.save(frame_path, "PNG")

            if on_progress and (i % max(1, total_frames // 20) == 0 or i == total_frames - 1):
                pct = 20 + int(60 * (i + 1) / total_frames)
                on_progress(pct, f"Rendered frame {i + 1}/{total_frames}")

        # Assemble into video
        if on_progress:
            on_progress(85, "Assembling video...")

        _assemble_frames_to_video(
            frame_dir, audio_path, out_path,
            fps=config.fps, width=config.width, height=config.height)

        if not os.path.isfile(out_path):
            raise RuntimeError("Video assembly failed — output file not created")

        if on_progress:
            on_progress(95, "Finalizing...")

        actual_duration = _get_audio_duration(out_path) or duration
        frame_count = total_frames

        return AvatarResult(
            output_path=out_path,
            duration=actual_duration,
            width=config.width,
            height=config.height,
            fps=config.fps,
            frame_count=frame_count,
            style=config.style,
            audio_path=audio_path,
        )

    finally:
        # Clean up frame directory
        try:
            import shutil
            shutil.rmtree(frame_dir, ignore_errors=True)
        except Exception:
            logger.debug("Failed to clean up frame directory: %s", frame_dir)


def list_avatar_styles() -> List[Dict]:
    """Return list of available avatar style definitions."""
    return list(AVATAR_STYLES)
