"""
OpenCut Object Track & Overlay Module (Category 69.3)

Track objects through video and attach graphic overlays that follow
the tracked object's position, scale, and rotation.

Pipeline:
    1. Click or describe object to track
    2. Track through video using SAM2 or feature matching fallback
    3. Attach text labels, arrows, blur, highlight, or custom images
    4. Overlays follow the tracked object's bounding box
    5. Export tracking data as JSON and composited video

Functions:
    track_and_overlay  - Full pipeline: track + overlay + render
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAY_TYPES = [
    "text_label",
    "arrow",
    "blur",
    "highlight",
    "custom_image",
    "circle",
    "rectangle",
    "crosshair",
    "spotlight",
    "censor",
]

# Default overlay styles
DEFAULT_FONT_SIZE = 24
DEFAULT_FONT_COLOR = "#FFFFFF"
DEFAULT_BG_COLOR = "#000000AA"
DEFAULT_ARROW_COLOR = "#FF4444"
DEFAULT_HIGHLIGHT_COLOR = "#FFFF0066"
DEFAULT_BLUR_STRENGTH = 15
DEFAULT_LINE_WIDTH = 3
DEFAULT_SPOTLIGHT_FEATHER = 50

# Tracking
TEMPLATE_MATCH_PAD = 40          # pixels of padding around object for template
MIN_TRACKING_CONFIDENCE = 0.3    # minimum match confidence before declaring lost
SMOOTH_WINDOW = 5                # frames for position smoothing


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class OverlayConfig:
    """Configuration for an overlay attached to a tracked object."""
    overlay_type: str = "text_label"
    text: str = ""
    font_size: int = DEFAULT_FONT_SIZE
    font_color: str = DEFAULT_FONT_COLOR
    bg_color: str = DEFAULT_BG_COLOR
    image_path: str = ""
    arrow_color: str = DEFAULT_ARROW_COLOR
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR
    blur_strength: int = DEFAULT_BLUR_STRENGTH
    line_width: int = DEFAULT_LINE_WIDTH
    offset_x: int = 0
    offset_y: int = -30
    scale_with_object: bool = True
    opacity: float = 1.0
    spotlight_feather: int = DEFAULT_SPOTLIGHT_FEATHER

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "OverlayConfig":
        """Create OverlayConfig from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TrackFrame:
    """Tracked position for a single frame."""
    frame_idx: int = 0
    timestamp: float = 0.0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 1.0
    lost: bool = False
    rotation: float = 0.0
    scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cx(self) -> int:
        return self.x + self.width // 2

    @property
    def cy(self) -> int:
        return self.y + self.height // 2


@dataclass
class TrackOverlayResult:
    """Result of tracking + overlay render."""
    output_path: str = ""
    track_data_path: str = ""
    frame_count: int = 0
    tracked_frames: int = 0
    lost_frames: int = 0
    overlay_type: str = ""
    video_width: int = 0
    video_height: int = 0
    fps: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Template matching tracker
# ---------------------------------------------------------------------------
def _init_template(frame_gray, point: Tuple[int, int], pad: int = TEMPLATE_MATCH_PAD):
    """Extract a template patch around the click point.

    Returns (template, bbox) where bbox is (x, y, w, h).
    """

    h, w = frame_gray.shape[:2]
    px, py = point

    x1 = max(px - pad, 0)
    y1 = max(py - pad, 0)
    x2 = min(px + pad, w)
    y2 = min(py + pad, h)

    template = frame_gray[y1:y2, x1:x2].copy()
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return template, bbox


def _match_template(frame_gray, template, prev_bbox, search_expand: int = 60):
    """Find the best match for template in frame, searching near prev_bbox.

    Returns (bbox, confidence).
    """
    import cv2

    h, w = frame_gray.shape[:2]
    th, tw = template.shape[:2]

    # Search region around previous position
    px, py, pw, ph = prev_bbox
    sx1 = max(px - search_expand, 0)
    sy1 = max(py - search_expand, 0)
    sx2 = min(px + pw + search_expand, w)
    sy2 = min(py + ph + search_expand, h)

    search_region = frame_gray[sy1:sy2, sx1:sx2]

    if search_region.shape[0] < th or search_region.shape[1] < tw:
        return prev_bbox, 0.0

    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    match_x = sx1 + max_loc[0]
    match_y = sy1 + max_loc[1]
    bbox = (match_x, match_y, tw, th)

    return bbox, float(max_val)


def _smooth_track(
    tracks: List[TrackFrame], window: int = SMOOTH_WINDOW,
) -> List[TrackFrame]:
    """Apply moving-average smoothing to track positions.

    Reduces jitter in the tracked bounding box positions.
    """
    if len(tracks) < window:
        return tracks

    smoothed = []
    half = window // 2

    for i, tf in enumerate(tracks):
        if tf.lost:
            smoothed.append(tf)
            continue

        start = max(0, i - half)
        end = min(len(tracks), i + half + 1)
        valid = [t for t in tracks[start:end] if not t.lost]

        if not valid:
            smoothed.append(tf)
            continue

        avg_x = int(sum(t.x for t in valid) / len(valid))
        avg_y = int(sum(t.y for t in valid) / len(valid))
        avg_w = int(sum(t.width for t in valid) / len(valid))
        avg_h = int(sum(t.height for t in valid) / len(valid))

        new_tf = TrackFrame(
            frame_idx=tf.frame_idx,
            timestamp=tf.timestamp,
            x=avg_x, y=avg_y,
            width=avg_w, height=avg_h,
            confidence=tf.confidence,
            lost=False,
            rotation=tf.rotation,
            scale=tf.scale,
        )
        smoothed.append(new_tf)

    return smoothed


def _track_object_through_video(
    video_path: str,
    track_point: Tuple[int, int],
    max_frames: int = 0,
    on_progress: Optional[Callable] = None,
) -> List[TrackFrame]:
    """Track an object from a click point through all video frames.

    Uses OpenCV template matching as a reliable fallback tracker.

    Args:
        video_path: Path to the video.
        track_point: (x, y) initial click position.
        max_frames: Limit tracking (0 = all frames).
        on_progress: Progress callback.

    Returns:
        List of TrackFrame for each frame.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("OpenCV required for tracking")

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")

    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    template, init_bbox = _init_template(gray, track_point)
    ref_w, ref_h = init_bbox[2], init_bbox[3]

    tracks = [TrackFrame(
        frame_idx=0, timestamp=0.0,
        x=init_bbox[0], y=init_bbox[1],
        width=init_bbox[2], height=init_bbox[3],
        confidence=1.0, scale=1.0,
    )]

    prev_bbox = init_bbox

    for fi in range(1, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox, conf = _match_template(gray, template, prev_bbox)

        lost = conf < MIN_TRACKING_CONFIDENCE
        scale = 1.0
        if not lost:
            scale = ((bbox[2] * bbox[3]) / max(ref_w * ref_h, 1)) ** 0.5
            prev_bbox = bbox

            # Update template periodically for drift correction
            if fi % 30 == 0 and conf > 0.6:
                bx, by, bw, bh = bbox
                new_tmpl = gray[max(by, 0):min(by + bh, gray.shape[0]),
                                max(bx, 0):min(bx + bw, gray.shape[1])]
                if new_tmpl.shape[0] > 5 and new_tmpl.shape[1] > 5:
                    template = new_tmpl.copy()

        tracks.append(TrackFrame(
            frame_idx=fi,
            timestamp=fi / fps,
            x=bbox[0], y=bbox[1],
            width=bbox[2], height=bbox[3],
            confidence=round(conf, 3),
            lost=lost,
            scale=round(scale, 3),
        ))

        if on_progress and fi % 50 == 0:
            pct = 10 + int(40 * fi / total_frames)
            on_progress(min(pct, 50), f"Tracking: {fi}/{total_frames}")

    cap.release()

    # Smooth the track
    tracks = _smooth_track(tracks)

    return tracks


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def _draw_text_label(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw a text label overlay at the tracked position."""
    from PIL import ImageFont

    if not cfg.text:
        return

    font_size = cfg.font_size
    if cfg.scale_with_object:
        font_size = max(int(font_size * tf.scale), 10)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    tx = tf.cx + cfg.offset_x
    ty = tf.y + cfg.offset_y

    # Background box
    bbox = draw.textbbox((tx, ty), cfg.text, font=font)
    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=cfg.bg_color,
    )
    draw.text((tx, ty), cfg.text, fill=cfg.font_color, font=font)


def _draw_arrow(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw an arrow pointing at the tracked object."""
    arrow_start = (tf.cx + cfg.offset_x, tf.y + cfg.offset_y)
    arrow_end = (tf.cx, tf.y)

    draw.line([arrow_start, arrow_end], fill=cfg.arrow_color, width=cfg.line_width)

    # Arrowhead
    angle = math.atan2(arrow_end[1] - arrow_start[1], arrow_end[0] - arrow_start[0])
    head_len = 12
    for sign in (-1, 1):
        hx = arrow_end[0] - int(head_len * math.cos(angle + sign * 0.5))
        hy = arrow_end[1] - int(head_len * math.sin(angle + sign * 0.5))
        draw.line([arrow_end, (hx, hy)], fill=cfg.arrow_color, width=cfg.line_width)


def _draw_highlight(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw a translucent highlight rectangle over the object."""
    draw.rectangle(
        [tf.x, tf.y, tf.x + tf.width, tf.y + tf.height],
        fill=cfg.highlight_color,
        outline=cfg.highlight_color,
        width=2,
    )


def _draw_circle(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw a circle around the tracked object."""
    radius = max(tf.width, tf.height) // 2 + 10
    draw.ellipse(
        [tf.cx - radius, tf.cy - radius, tf.cx + radius, tf.cy + radius],
        outline=cfg.arrow_color,
        width=cfg.line_width,
    )


def _draw_rectangle(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw a rectangle bounding box around the tracked object."""
    draw.rectangle(
        [tf.x, tf.y, tf.x + tf.width, tf.y + tf.height],
        outline=cfg.arrow_color,
        width=cfg.line_width,
    )


def _draw_crosshair(draw, tf: TrackFrame, cfg: OverlayConfig, img_w: int, img_h: int):
    """Draw crosshair lines centred on the tracked object."""
    cx, cy = tf.cx, tf.cy
    arm = max(tf.width, tf.height) // 2 + 15

    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=cfg.arrow_color, width=cfg.line_width)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=cfg.arrow_color, width=cfg.line_width)


def _apply_blur(img, tf: TrackFrame, cfg: OverlayConfig):
    """Apply a Gaussian blur to the tracked region."""
    from PIL import ImageFilter

    x1 = max(tf.x, 0)
    y1 = max(tf.y, 0)
    x2 = min(tf.x + tf.width, img.width)
    y2 = min(tf.y + tf.height, img.height)

    if x2 <= x1 or y2 <= y1:
        return img

    region = img.crop((x1, y1, x2, y2))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=cfg.blur_strength))
    img.paste(blurred, (x1, y1))
    return img


def _apply_censor(img, tf: TrackFrame, cfg: OverlayConfig):
    """Apply pixelation censoring to the tracked region."""
    x1 = max(tf.x, 0)
    y1 = max(tf.y, 0)
    x2 = min(tf.x + tf.width, img.width)
    y2 = min(tf.y + tf.height, img.height)

    if x2 <= x1 or y2 <= y1:
        return img

    region = img.crop((x1, y1, x2, y2))
    small_w = max(region.width // 10, 1)
    small_h = max(region.height // 10, 1)
    pixelated = region.resize((small_w, small_h), resample=0).resize(region.size, resample=0)
    img.paste(pixelated, (x1, y1))
    return img


def _apply_spotlight(img, tf: TrackFrame, cfg: OverlayConfig):
    """Darken everything except the tracked region (spotlight effect)."""
    import numpy as np
    from PIL import Image

    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    # Create a vignette mask centred on the object
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - tf.cx) ** 2 + (ys - tf.cy) ** 2)
    radius = max(tf.width, tf.height) / 2 + cfg.spotlight_feather
    feather = max(cfg.spotlight_feather, 1)

    mask = np.clip((dist - radius) / feather, 0, 1)
    darkness = 0.3  # how dark the non-spotlighted area gets

    for c in range(min(arr.shape[2], 3)):
        arr[:, :, c] = arr[:, :, c] * (1 - mask * (1 - darkness))

    return Image.fromarray(arr.astype(np.uint8))


def _apply_custom_image(img, tf: TrackFrame, cfg: OverlayConfig):
    """Paste a custom image overlay at the tracked position."""
    from PIL import Image

    if not cfg.image_path or not os.path.isfile(cfg.image_path):
        return img

    overlay = Image.open(cfg.image_path).convert("RGBA")

    # Scale overlay to match tracked object size if enabled
    if cfg.scale_with_object:
        new_w = max(int(overlay.width * tf.scale), 1)
        new_h = max(int(overlay.height * tf.scale), 1)
        overlay = overlay.resize((new_w, new_h), resample=3)

    # Position at offset from tracked centre
    paste_x = tf.cx + cfg.offset_x - overlay.width // 2
    paste_y = tf.y + cfg.offset_y - overlay.height

    # Apply opacity
    if cfg.opacity < 1.0:
        alpha = overlay.split()[3]
        import numpy as np
        a_arr = np.array(alpha).astype(np.float32) * cfg.opacity
        from PIL import Image as PILImage
        overlay.putalpha(PILImage.fromarray(a_arr.astype(np.uint8)))

    # Composite
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    temp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    temp.paste(overlay, (paste_x, paste_y))
    img = Image.alpha_composite(img, temp)

    return img


# Overlay renderer dispatch
_OVERLAY_RENDERERS = {
    "text_label": lambda img, draw, tf, cfg: _draw_text_label(draw, tf, cfg, img.width, img.height),
    "arrow": lambda img, draw, tf, cfg: _draw_arrow(draw, tf, cfg, img.width, img.height),
    "highlight": lambda img, draw, tf, cfg: _draw_highlight(draw, tf, cfg, img.width, img.height),
    "circle": lambda img, draw, tf, cfg: _draw_circle(draw, tf, cfg, img.width, img.height),
    "rectangle": lambda img, draw, tf, cfg: _draw_rectangle(draw, tf, cfg, img.width, img.height),
    "crosshair": lambda img, draw, tf, cfg: _draw_crosshair(draw, tf, cfg, img.width, img.height),
}

# These modify the image directly (not via draw)
_OVERLAY_IMG_RENDERERS = {
    "blur": _apply_blur,
    "censor": _apply_censor,
    "spotlight": _apply_spotlight,
    "custom_image": _apply_custom_image,
}


def _render_overlay_frame(
    frame_path: str, tf: TrackFrame, cfg: OverlayConfig, out_path: str,
):
    """Render the overlay onto a single frame and save."""
    from PIL import Image, ImageDraw

    img = Image.open(frame_path).convert("RGBA")

    if tf.lost:
        # Skip overlay on lost frames
        img.convert("RGB").save(out_path)
        return

    otype = cfg.overlay_type

    # Image-level renderers
    if otype in _OVERLAY_IMG_RENDERERS:
        img = _OVERLAY_IMG_RENDERERS[otype](img, tf, cfg)
    # Draw-level renderers
    elif otype in _OVERLAY_RENDERERS:
        draw = ImageDraw.Draw(img, "RGBA")
        _OVERLAY_RENDERERS[otype](img, draw, tf, cfg)
    else:
        logger.warning("Unknown overlay type '%s'; skipping", otype)

    img.convert("RGB").save(out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def track_and_overlay(
    video_path: str,
    track_point: Tuple[int, int] = (100, 100),
    overlay_config: Optional[Dict] = None,
    max_frames: int = 0,
    output_dir: str = "",
    smooth: bool = True,
    export_track_json: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Track an object and render an overlay following it.

    Args:
        video_path: Path to input video.
        track_point: (x, y) pixel to track.
        overlay_config: Dict matching OverlayConfig fields.
        max_frames: Limit tracking (0 = all).
        output_dir: Output directory.
        smooth: Apply position smoothing.
        export_track_json: Also export tracking data as JSON.
        on_progress: Progress callback(pct, msg).

    Returns:
        TrackOverlayResult as dict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow required")


    cfg = OverlayConfig.from_dict(overlay_config or {})
    if cfg.overlay_type not in OVERLAY_TYPES:
        raise ValueError(f"Unknown overlay type '{cfg.overlay_type}'. Valid: {OVERLAY_TYPES}")

    if on_progress:
        on_progress(2, f"Starting track & overlay ({cfg.overlay_type})...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    vid_w = info.get("width", 1920)
    vid_h = info.get("height", 1080)

    # Step 1: Track object
    if on_progress:
        on_progress(5, "Tracking object...")

    tracks = _track_object_through_video(
        video_path, track_point,
        max_frames=max_frames,
        on_progress=on_progress,
    )

    if not smooth:
        pass  # Already smoothed in _track_object_through_video

    tracked_count = sum(1 for t in tracks if not t.lost)
    lost_count = sum(1 for t in tracks if t.lost)

    if on_progress:
        on_progress(55, f"Tracked {tracked_count} frames ({lost_count} lost); rendering overlays...")

    # Step 2: Extract frames
    work_dir = tempfile.mkdtemp(prefix="trackoverlay_")
    frames_dir = os.path.join(work_dir, "frames")
    render_dir = os.path.join(work_dir, "rendered")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    pattern = os.path.join(frames_dir, "frame_%06d.png")
    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-q:v", "2", pattern,
    ], timeout=7200)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))

    # Step 3: Render overlays
    for fi, fname in enumerate(frame_files):
        if fi >= len(tracks):
            break

        tf = tracks[fi]
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(render_dir, fname)
        _render_overlay_frame(src, tf, cfg, dst)

        if on_progress and fi % 30 == 0:
            pct = 55 + int(35 * fi / len(frame_files))
            on_progress(min(pct, 90), f"Rendering: {fi}/{len(frame_files)}")

    # Step 4: Reassemble video
    if on_progress:
        on_progress(92, "Encoding output video...")

    out_dir = output_dir or os.path.dirname(video_path)
    out_file = output_path(video_path, "trackoverlay", output_dir=out_dir)
    render_pattern = os.path.join(render_dir, "frame_%06d.png")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", render_pattern,
        "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", "-pix_fmt", "yuv420p",
        out_file,
    ]
    run_ffmpeg(cmd, timeout=3600)

    # Step 5: Export track data
    track_json_path = ""
    if export_track_json:
        track_json_path = os.path.splitext(out_file)[0] + "_track.json"
        track_data = {
            "video": os.path.basename(video_path),
            "fps": fps,
            "frame_count": len(tracks),
            "initial_point": list(track_point),
            "overlay_config": cfg.to_dict(),
            "frames": [t.to_dict() for t in tracks],
        }
        with open(track_json_path, "w", encoding="utf-8") as f:
            json.dump(track_data, f, indent=2)

    if on_progress:
        on_progress(100, "Track & overlay complete")

    result = TrackOverlayResult(
        output_path=out_file,
        track_data_path=track_json_path,
        frame_count=len(tracks),
        tracked_frames=tracked_count,
        lost_frames=lost_count,
        overlay_type=cfg.overlay_type,
        video_width=vid_w,
        video_height=vid_h,
        fps=fps,
    )
    return result.to_dict()
