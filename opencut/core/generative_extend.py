"""
OpenCut Generative Extend Module

AI extends clips beyond their recorded length. Adobe Firefly's flagship feature.

Methods (in priority order):
    1. Video generation model (Wan 2.1 / LTX-2 / CogVideoX) conditioned on last frame
    2. Optical flow extrapolation (RIFE-style interpolation to predict next frames)
    3. Freeze-frame with subtle zoom/pan to simulate motion

Audio extension: loop detected room tone or pad with silence.

Functions:
    extend_clip - Extend a video clip forward, backward, or both directions
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_DIRECTIONS = ("forward", "backward", "both")
VALID_MODELS = ("auto", "wan2.1", "ltx2", "cogvideox", "flow", "freeze")

# Default frame count extracted for model conditioning
_CONDITIONING_FRAMES = 8
# Maximum extend duration per direction (seconds)
_MAX_EXTEND_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ExtendResult:
    """Result of a generative extend operation."""
    output_path: str = ""
    original_duration: float = 0.0
    extended_duration: float = 0.0
    frames_generated: int = 0
    model_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _detect_available_model() -> str:
    """Probe for the best available video generation model.

    Returns one of: "wan2.1", "ltx2", "cogvideox", "flow", "freeze".
    """
    # Try diffusers-based models first
    try:
        import importlib
        importlib.import_module("diffusers")
        # If diffusers is available, prefer Wan 2.1 (best quality/speed ratio)
        return "wan2.1"
    except ImportError:
        pass

    # Try optical flow (RIFE) via torch
    try:
        import importlib
        importlib.import_module("torch")
        return "flow"
    except ImportError:
        pass

    # Fallback: freeze-frame with motion
    return "freeze"


def _extract_boundary_frames(
    video_path: str,
    direction: str,
    count: int,
    tmp_dir: str,
) -> list:
    """Extract the first or last N frames from a video.

    Args:
        video_path: Source video file.
        direction: "forward" extracts last frames, "backward" extracts first frames.
        count: Number of frames to extract.
        tmp_dir: Directory to store extracted frames.

    Returns:
        List of extracted frame file paths (sorted).
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    fps = info.get("fps", 30.0)

    pattern = os.path.join(tmp_dir, "frame_%05d.png")

    if direction == "backward":
        # Extract first N frames
        cmd = (
            FFmpegCmd()
            .input(video_path)
            .option("frames:v", str(count))
            .output(pattern)
            .build()
        )
    else:
        # Extract last N frames -- seek near the end
        start_time = max(0, duration - (count / fps) - 0.5)
        cmd = (
            FFmpegCmd()
            .input(video_path, ss=f"{start_time:.3f}")
            .option("frames:v", str(count))
            .output(pattern)
            .build()
        )

    run_ffmpeg(cmd)

    frames = sorted(
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    return frames[-count:] if len(frames) > count else frames


def _generate_via_model(
    frames: list,
    extend_seconds: float,
    fps: float,
    width: int,
    height: int,
    model: str,
    out_dir: str,
    on_progress: Optional[Callable] = None,
) -> list:
    """Generate continuation frames using a video generation model.

    This is a structured placeholder -- actual model inference requires
    diffusers/torch with significant GPU resources.  The function sets up
    the pipeline correctly so it works when models are available and falls
    back gracefully when they are not.

    Returns:
        List of generated frame file paths.
    """
    num_frames = int(extend_seconds * fps)
    generated = []

    if model in ("wan2.1", "ltx2", "cogvideox"):
        if not ensure_package("diffusers", "diffusers"):
            logger.warning("diffusers not available, cannot run %s model", model)
            return []
        if not ensure_package("torch", "torch"):
            logger.warning("torch not available, cannot run %s model", model)
            return []

        # Lazy imports after ensure_package
        import torch  # noqa: F811

        try:
            from diffusers import DiffusionPipeline

            model_ids = {
                "wan2.1": "Wan-AI/Wan2.1-T2V-14B",
                "ltx2": "Lightricks/LTX-Video",
                "cogvideox": "THUDM/CogVideoX-5b",
            }
            model_id = model_ids.get(model, model_ids["wan2.1"])

            if on_progress:
                on_progress(30, f"Loading {model} pipeline...")

            # Load the last conditioning frame as PIL image
            from PIL import Image
            cond_image = Image.open(frames[-1]).convert("RGB")

            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            if on_progress:
                on_progress(50, f"Generating {num_frames} frames via {model}...")

            result = pipe(
                image=cond_image,
                num_frames=num_frames,
                width=width,
                height=height,
            )

            # Save generated frames
            for i, frame in enumerate(result.frames[0]):
                path = os.path.join(out_dir, f"gen_{i:05d}.png")
                frame.save(path)
                generated.append(path)

        except Exception as e:
            logger.warning("Model %s inference failed: %s", model, e)
            return []

    return generated


def _generate_via_flow(
    frames: list,
    extend_seconds: float,
    fps: float,
    out_dir: str,
    on_progress: Optional[Callable] = None,
) -> list:
    """Generate continuation frames via optical flow extrapolation.

    Uses frame interpolation in reverse -- extrapolating motion vectors
    from the last few frames to predict the next ones, with diminishing
    confidence applied as fade-to-last-frame.

    Returns:
        List of generated frame file paths.
    """
    num_frames = int(extend_seconds * fps)
    if len(frames) < 2:
        return []

    if not ensure_package("cv2", "opencv-python-headless"):
        return []
    if not ensure_package("numpy", "numpy"):
        return []

    import cv2
    import numpy as np

    if on_progress:
        on_progress(30, "Computing optical flow extrapolation...")

    # Read the last two frames
    frame_prev = cv2.imread(frames[-2])
    frame_last = cv2.imread(frames[-1])

    if frame_prev is None or frame_last is None:
        return []

    # Compute dense optical flow (Farneback)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_last = cv2.cvtColor(frame_last, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_last, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    h, w = frame_last.shape[:2]
    generated = []

    for i in range(num_frames):
        # Diminishing confidence: flow strength decreases with distance
        confidence = max(0.1, 1.0 - (i / max(num_frames, 1)) * 0.8)
        scaled_flow = flow * (i + 1) * confidence

        # Build remap coordinates
        map_x = np.float32(np.tile(np.arange(w), (h, 1))) + scaled_flow[..., 0]
        map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + scaled_flow[..., 1]

        warped = cv2.remap(
            frame_last, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Blend with last frame as confidence drops
        alpha = confidence
        blended = cv2.addWeighted(warped, alpha, frame_last, 1.0 - alpha, 0)

        path = os.path.join(out_dir, f"flow_{i:05d}.png")
        cv2.imwrite(path, blended)
        generated.append(path)

        if on_progress and i % max(1, num_frames // 5) == 0:
            pct = 30 + int(50 * (i + 1) / num_frames)
            on_progress(pct, f"Flow frame {i + 1}/{num_frames}")

    return generated


def _generate_via_freeze(
    video_path: str,
    extend_seconds: float,
    direction: str,
    fps: float,
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate extension via freeze-frame with subtle zoom/pan.

    Extracts the boundary frame and applies a slow Ken Burns effect
    (gentle zoom + slight pan) to simulate motion.

    Returns:
        Path to the generated video segment.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    if on_progress:
        on_progress(30, "Generating freeze-frame extension...")

    if direction == "backward":
        # Freeze first frame
        ss_time = "0"
    else:
        # Freeze last frame
        ss_time = f"{max(0, duration - 0.1):.3f}"

    # Extract single frame
    fd, frame_path = tempfile.mkstemp(suffix=".png", prefix="freeze_")
    os.close(fd)

    extract_cmd = (
        FFmpegCmd()
        .input(video_path, ss=ss_time)
        .option("frames:v", "1")
        .output(frame_path)
        .build()
    )
    run_ffmpeg(extract_cmd)

    # Apply subtle zoom/pan (Ken Burns) on the frozen frame
    # Zoom from 1.0x to 1.03x over the duration -- barely perceptible
    zoom_expr = f"min(1.03,1+0.03*on/{max(1, extend_seconds * fps)})"
    # Slight horizontal drift
    x_expr = f"iw/2-(iw/zoom/2)+{width * 0.002}*on/{max(1, extend_seconds * fps)}"
    y_expr = "ih/2-(ih/zoom/2)"

    zoompan_filter = (
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={int(extend_seconds * fps)}:s={width}x{height}:fps={fps}"
    )

    gen_cmd = (
        FFmpegCmd()
        .input(frame_path, loop="1", framerate=str(fps))
        .option("t", f"{extend_seconds:.3f}")
        .option("vf", zoompan_filter)
        .video_codec("libx264", crf=18, preset="fast")
        .option("an", "")
        .output(out_path)
        .build()
    )
    run_ffmpeg(gen_cmd)

    # Cleanup temp frame
    try:
        os.unlink(frame_path)
    except OSError:
        pass

    if on_progress:
        on_progress(70, "Freeze-frame extension generated")

    return out_path


def _extend_audio(
    video_path: str,
    extend_seconds: float,
    direction: str,
    out_path: str,
) -> str:
    """Extend audio by looping room tone or padding with silence.

    Detects a quiet section in the original audio to use as room tone.
    If detection fails, pads with silence.

    Returns:
        Path to the extended audio file.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    # Try to extract a short room-tone segment (last 0.5s of audio)
    fd, tone_path = tempfile.mkstemp(suffix=".wav", prefix="roomtone_")
    os.close(fd)

    # Extract audio from a likely quiet segment near the end
    tone_start = max(0, duration - 1.0)
    tone_dur = min(0.5, duration * 0.1) if duration > 0 else 0.5

    try:
        extract_cmd = (
            FFmpegCmd()
            .input(video_path, ss=f"{tone_start:.3f}")
            .option("t", f"{tone_dur:.3f}")
            .option("vn", "")
            .audio_codec("pcm_s16le")
            .output(tone_path)
            .build()
        )
        run_ffmpeg(extract_cmd)

        # Loop the room tone to fill the extension duration
        loop_count = int(extend_seconds / tone_dur) + 2
        loop_cmd = (
            FFmpegCmd()
            .input(tone_path, stream_loop=str(loop_count))
            .option("t", f"{extend_seconds:.3f}")
            .audio_codec("pcm_s16le")
            .output(out_path)
            .build()
        )
        run_ffmpeg(loop_cmd)
    except Exception:
        # Fallback: generate silence
        logger.info("Room tone extraction failed, padding with silence")
        silence_cmd = (
            FFmpegCmd()
            .filter_complex(
                f"anullsrc=r=44100:cl=stereo:d={extend_seconds}[out]",
                maps=["[out]"],
            )
            .audio_codec("pcm_s16le")
            .output(out_path)
            .build()
        )
        run_ffmpeg(silence_cmd)
    finally:
        try:
            os.unlink(tone_path)
        except OSError:
            pass

    return out_path


def _frames_to_video(
    frame_dir: str,
    pattern: str,
    fps: float,
    duration: float,
    out_path: str,
) -> str:
    """Encode a sequence of frames into a video segment."""
    frame_input = os.path.join(frame_dir, pattern)
    cmd = (
        FFmpegCmd()
        .input(frame_input, framerate=str(fps))
        .option("t", f"{duration:.3f}")
        .video_codec("libx264", crf=18, preset="fast")
        .option("an", "")
        .output(out_path)
        .build()
    )
    run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extend_clip(
    video_path: str,
    extend_seconds: float = 2.0,
    direction: str = "forward",
    output: Optional[str] = None,
    model: str = "auto",
    on_progress: Optional[Callable] = None,
) -> ExtendResult:
    """Extend a video clip beyond its recorded length using AI generation.

    Tries the best available method in order:
    1. Video generation model (Wan 2.1 / LTX-2 / CogVideoX)
    2. Optical flow extrapolation (RIFE-style)
    3. Freeze-frame with Ken Burns motion

    Args:
        video_path: Path to the input video file.
        extend_seconds: Duration to extend in seconds (per direction). Max 30s.
        direction: "forward" (after end), "backward" (before start), or "both".
        output: Output video path. Auto-generated if None.
        model: Model to use: "auto", "wan2.1", "ltx2", "cogvideox", "flow", "freeze".
        on_progress: Optional progress callback(pct, msg).

    Returns:
        ExtendResult with output path and metadata.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If direction or model is invalid, or extend_seconds is out of range.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    direction = direction.lower().strip()
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of: {VALID_DIRECTIONS}")

    model = model.lower().strip()
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")

    extend_seconds = max(0.1, min(float(extend_seconds), _MAX_EXTEND_SECONDS))

    if on_progress:
        on_progress(5, f"Preparing to extend clip {direction} by {extend_seconds:.1f}s...")

    info = get_video_info(video_path)
    original_duration = info.get("duration", 0)
    fps = info.get("fps", 30.0)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    if output is None:
        output = output_path(video_path, f"extended_{direction}")

    # Resolve model
    selected_model = model if model != "auto" else _detect_available_model()

    tmp_dir = tempfile.mkdtemp(prefix="opencut_extend_")
    generated_segments = []
    total_frames_generated = 0

    try:
        directions = (
            ["backward", "forward"] if direction == "both" else [direction]
        )

        for dir_idx, d in enumerate(directions):
            if on_progress:
                base_pct = 10 + dir_idx * 40
                on_progress(base_pct, f"Extending {d}...")

            seg_dir = os.path.join(tmp_dir, d)
            os.makedirs(seg_dir, exist_ok=True)

            fd, seg_path = tempfile.mkstemp(
                suffix=".mp4", prefix=f"seg_{d}_", dir=tmp_dir,
            )
            os.close(fd)

            frames_generated = 0

            if selected_model in ("wan2.1", "ltx2", "cogvideox"):
                # Extract conditioning frames
                cond_frames = _extract_boundary_frames(
                    video_path, d, _CONDITIONING_FRAMES, seg_dir,
                )
                gen_frames = _generate_via_model(
                    cond_frames, extend_seconds, fps, width, height,
                    selected_model, seg_dir, on_progress,
                )
                if gen_frames:
                    _frames_to_video(
                        seg_dir, "gen_%05d.png", fps, extend_seconds, seg_path,
                    )
                    frames_generated = len(gen_frames)
                else:
                    # Fall through to flow
                    selected_model = "flow"

            if selected_model == "flow" and frames_generated == 0:
                cond_frames = _extract_boundary_frames(
                    video_path, d, _CONDITIONING_FRAMES, seg_dir,
                )
                gen_frames = _generate_via_flow(
                    cond_frames, extend_seconds, fps, seg_dir, on_progress,
                )
                if gen_frames:
                    _frames_to_video(
                        seg_dir, "flow_%05d.png", fps, extend_seconds, seg_path,
                    )
                    frames_generated = len(gen_frames)
                else:
                    selected_model = "freeze"

            if selected_model == "freeze" and frames_generated == 0:
                _generate_via_freeze(
                    video_path, extend_seconds, d, fps, seg_path, on_progress,
                )
                frames_generated = int(extend_seconds * fps)

            generated_segments.append((d, seg_path))
            total_frames_generated += frames_generated

        # Generate extended audio
        if on_progress:
            on_progress(80, "Extending audio...")

        fd, ext_audio = tempfile.mkstemp(suffix=".wav", prefix="ext_audio_", dir=tmp_dir)
        os.close(fd)
        total_extend = extend_seconds * len(directions)
        _extend_audio(video_path, total_extend, direction, ext_audio)

        # Concatenate: [backward_segment] + original + [forward_segment]
        if on_progress:
            on_progress(85, "Concatenating segments...")

        # Build concat file
        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w") as f:
            for d, seg_path in generated_segments:
                if d == "backward":
                    f.write(f"file '{seg_path}'\n")
            f.write(f"file '{video_path}'\n")
            for d, seg_path in generated_segments:
                if d == "forward":
                    f.write(f"file '{seg_path}'\n")

        # Concat video segments
        fd, concat_video = tempfile.mkstemp(suffix=".mp4", prefix="concat_", dir=tmp_dir)
        os.close(fd)

        concat_cmd = (
            FFmpegCmd()
            .option("f", "concat")
            .option("safe", "0")
            .input(concat_path)
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("aac", bitrate="192k")
            .output(output)
            .build()
        )
        run_ffmpeg(concat_cmd)

        if on_progress:
            on_progress(95, "Extension complete")

        extended_duration = original_duration + (extend_seconds * len(directions))

        return ExtendResult(
            output_path=output,
            original_duration=round(original_duration, 3),
            extended_duration=round(extended_duration, 3),
            frames_generated=total_frames_generated,
            model_used=selected_model,
        )

    finally:
        # Cleanup temp directory
        import shutil as _shutil
        try:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
