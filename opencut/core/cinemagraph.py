"""
OpenCut Cinemagraph Creator Module

Create cinemagraphs -- still images with a single looping motion region.

Pipeline:
  1. Extract a reference (still) frame from the video.
  2. Apply a user-defined motion mask to isolate the moving region.
  3. Loop the masked motion over the static base with crossfade at
     the loop boundary for seamless playback.

Uses OpenCV for frame-level compositing and FFmpeg for encoding.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class CinemagraphResult:
    """Result from cinemagraph creation."""
    output_path: str = ""
    frames_written: int = 0
    loop_duration: float = 0.0
    crossfade_frames: int = 0
    resolution: Tuple[int, int] = (0, 0)


@dataclass
class ReferenceFrameResult:
    """Result from reference frame extraction."""
    frame_path: str = ""
    timestamp: float = 0.0
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Reference Frame Extraction
# ---------------------------------------------------------------------------
def extract_reference_frame(
    video_path: str,
    timestamp: float = 0.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ReferenceFrameResult:
    """
    Extract a single frame from a video to use as the cinemagraph base.

    Args:
        video_path: Source video.
        timestamp: Time in seconds to extract the frame.
        output_path_str: Output image path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReferenceFrameResult with the frame path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    timestamp = max(0.0, min(float(timestamp), max(0.0, info["duration"] - 0.1)))

    if output_path_str is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = os.path.dirname(video_path)
        output_path_str = os.path.join(directory, f"{base}_ref_frame.png")

    if on_progress:
        on_progress(10, f"Extracting frame at {timestamp:.2f}s...")

    cmd = (FFmpegCmd()
           .pre_input("ss", str(timestamp))
           .input(video_path)
           .frames(1)
           .option("q:v", "2")
           .output(output_path_str)
           .build())
    run_ffmpeg(cmd, timeout=60)

    if on_progress:
        on_progress(100, "Reference frame extracted")

    return ReferenceFrameResult(
        frame_path=output_path_str,
        timestamp=timestamp,
        width=info["width"],
        height=info["height"],
    )


# ---------------------------------------------------------------------------
# Mask Parsing
# ---------------------------------------------------------------------------
def _parse_mask_data(mask_data: Dict, width: int, height: int):
    """
    Parse mask specification into an OpenCV mask image.

    Supports:
      - Rectangle: {"type": "rect", "x": 0, "y": 0, "w": 100, "h": 100}
      - Ellipse:   {"type": "ellipse", "cx": 50, "cy": 50, "rx": 30, "ry": 20}
      - Polygon:   {"type": "polygon", "points": [[x1,y1], [x2,y2], ...]}
      - Full mask: {"type": "image", "path": "/path/to/mask.png"}
      - Feather:   {"feather": 10}  (optional, in pixels)

    Returns:
        numpy array (H, W) float32 0-1 alpha mask.
    """
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    mask_type = mask_data.get("type", "rect")
    feather = int(mask_data.get("feather", 10))

    if mask_type == "rect":
        x = int(mask_data.get("x", 0))
        y = int(mask_data.get("y", 0))
        w = int(mask_data.get("w", width // 3))
        h = int(mask_data.get("h", height // 3))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    elif mask_type == "ellipse":
        cx = int(mask_data.get("cx", width // 2))
        cy = int(mask_data.get("cy", height // 2))
        rx = int(mask_data.get("rx", width // 4))
        ry = int(mask_data.get("ry", height // 4))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    elif mask_type == "polygon":
        points = mask_data.get("points", [])
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        else:
            raise ValueError("Polygon mask requires at least 3 points")

    elif mask_type == "image":
        mask_path = mask_data.get("path", "")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        loaded = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise RuntimeError(f"Could not read mask image: {mask_path}")
        mask = cv2.resize(loaded, (width, height))

    else:
        raise ValueError(f"Unknown mask type: {mask_type}. "
                         f"Supported: rect, ellipse, polygon, image")

    # Apply feathering
    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Cinemagraph Creation
# ---------------------------------------------------------------------------
def create_cinemagraph(
    video_path: str,
    mask_data: Dict,
    loop_duration: float = 3.0,
    start_time: float = 0.0,
    crossfade: float = 0.5,
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    output_format: str = "mp4",
    ref_timestamp: Optional[float] = None,
    on_progress: Optional[Callable] = None,
) -> CinemagraphResult:
    """
    Create a cinemagraph from a video clip.

    The pipeline:
      1. Extract a reference still frame.
      2. Parse the motion mask to identify the looping region.
      3. For each frame in the loop window, composite the masked region
         over the still frame.
      4. Apply crossfade at the loop boundary for seamless looping.
      5. Encode as a looping video (or GIF).

    Args:
        video_path: Source video.
        mask_data: Dict describing the motion region mask.
        loop_duration: Duration of the loop in seconds.
        start_time: Start time in the source video for the loop window.
        crossfade: Crossfade duration at loop boundary (seconds).
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        output_format: "mp4" or "gif".
        ref_timestamp: Timestamp for the reference frame. Defaults to start_time.
        on_progress: Progress callback(pct, msg).

    Returns:
        CinemagraphResult.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    vid_duration = info["duration"]

    loop_duration = max(0.5, min(float(loop_duration), vid_duration))
    start_time = max(0.0, float(start_time))
    crossfade = max(0.0, min(float(crossfade), loop_duration / 3.0))

    if ref_timestamp is None:
        ref_timestamp = start_time

    if on_progress:
        on_progress(5, "Extracting reference frame...")

    # Step 1: Extract reference frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Seek to reference frame
    cap.set(cv2.CAP_PROP_POS_MSEC, ref_timestamp * 1000)
    ret, ref_frame = cap.read()
    if not ret or ref_frame is None:
        cap.release()
        raise RuntimeError("Failed to extract reference frame")
    ref_frame = cv2.resize(ref_frame, (w, h))

    if on_progress:
        on_progress(15, "Parsing motion mask...")

    # Step 2: Parse mask
    alpha_mask = _parse_mask_data(mask_data, w, h)
    alpha_3 = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)

    if on_progress:
        on_progress(20, "Reading loop frames...")

    # Step 3: Read frames for the loop window
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    total_loop_frames = max(1, int(loop_duration * fps))
    crossfade_frames = max(0, int(crossfade * fps))

    frames: List = []
    for i in range(total_loop_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (w, h)))

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video for loop window")

    if on_progress:
        on_progress(40, f"Compositing {len(frames)} frames...")

    # Step 4: Composite and crossfade
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError("Failed to create video writer")

    n_frames = len(frames)
    frames_written = 0

    try:
        for i in range(n_frames):
            motion_frame = frames[i].astype(np.float32)
            still_frame = ref_frame.astype(np.float32)

            # Composite: motion region from video, rest from still
            composite = still_frame * (1.0 - alpha_3) + motion_frame * alpha_3

            # Crossfade near loop boundary
            if crossfade_frames > 0 and i >= n_frames - crossfade_frames:
                # Blend towards the first frame for seamless loop
                fade_progress = (i - (n_frames - crossfade_frames)) / max(1, crossfade_frames)
                first_motion = frames[0].astype(np.float32)
                first_composite = still_frame * (1.0 - alpha_3) + first_motion * alpha_3
                composite = composite * (1.0 - fade_progress) + first_composite * fade_progress

            writer.write(composite.astype(np.uint8))
            frames_written += 1

            if on_progress and i % 10 == 0:
                pct = 40 + int((i / n_frames) * 45)
                on_progress(min(85, pct), f"Writing frame {i + 1}/{n_frames}")

    finally:
        writer.release()

    if on_progress:
        on_progress(88, "Encoding final output...")

    # Step 5: Encode final output
    if output_path_str is None:
        directory = output_dir or os.path.dirname(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = ".gif" if output_format == "gif" else ".mp4"
        output_path_str = os.path.join(directory, f"{base}_cinemagraph{ext}")

    try:
        if output_format == "gif":
            # GIF encoding via FFmpeg palette generation
            palette_path = os.path.join(tempfile.gettempdir(),
                                        f"cine_palette_{os.getpid()}.png")
            try:
                # Pass 1: generate palette
                cmd_palette = (FFmpegCmd()
                               .input(tmp_path)
                               .video_filter(f"fps={min(fps, 15)},scale={min(w, 480)}:-1:flags=lanczos,palettegen")
                               .output(palette_path)
                               .build())
                run_ffmpeg(cmd_palette, timeout=120)

                # Pass 2: encode GIF with palette
                cmd_gif = (FFmpegCmd()
                           .input(tmp_path)
                           .input(palette_path)
                           .filter_complex(
                               f"[0:v]fps={min(fps, 15)},scale={min(w, 480)}:-1:flags=lanczos[v];"
                               f"[v][1:v]paletteuse=dither=bayer",
                           )
                           .output(output_path_str)
                           .build())
                run_ffmpeg(cmd_gif, timeout=120)
            finally:
                try:
                    os.unlink(palette_path)
                except OSError:
                    pass
        else:
            cmd = (FFmpegCmd()
                   .input(tmp_path)
                   .video_codec("libx264", crf=18, preset="medium")
                   .audio_codec("aac", bitrate="128k")
                   .faststart()
                   .output(output_path_str)
                   .build())
            run_ffmpeg(cmd, timeout=300)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Cinemagraph created!")

    return CinemagraphResult(
        output_path=output_path_str,
        frames_written=frames_written,
        loop_duration=loop_duration,
        crossfade_frames=crossfade_frames,
        resolution=(w, h),
    )
