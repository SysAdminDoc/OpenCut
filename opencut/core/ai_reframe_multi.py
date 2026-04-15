"""
OpenCut Multi-Subject Intelligent Reframe Module (Category 69.5)

Reframe video for different aspect ratios while keeping all important
subjects visible. Detects faces, objects, and text, scores importance,
and computes optimal crop paths with smooth transitions.

Pipeline:
    1. Detect all subjects per frame (faces, objects, text regions)
    2. Score importance of each subject
    3. Compute optimal crop window per frame
    4. Fall back to split-screen when subjects too far apart
    5. Smooth crop transitions across frames
    6. Render reframed video via FFmpeg

Functions:
    reframe_multi_subject - Full pipeline: detect + score + crop + render
"""

import logging
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
ASPECT_RATIOS = {
    "9:16":  (9, 16),    # Vertical / TikTok / Reels
    "1:1":   (1, 1),     # Instagram square
    "4:5":   (4, 5),     # Instagram portrait
    "16:9":  (16, 9),    # Standard widescreen
    "4:3":   (4, 3),     # Classic TV
    "21:9":  (21, 9),    # Ultra-wide cinema
    "2.35:1": (235, 100), # Cinemascope
    "3:4":   (3, 4),     # Portrait 3:4
    "2:3":   (2, 3),     # Portrait 2:3
}

# Subject detection weights
SUBJECT_WEIGHTS = {
    "face":   1.0,    # Faces are highest priority
    "person": 0.85,   # Full body / person
    "text":   0.7,    # On-screen text
    "object": 0.5,    # Generic objects
    "motion": 0.4,    # Motion hotspots
}

# Crop smoothing
SMOOTH_KERNEL_SIZE = 15          # frames for temporal smoothing
SPLIT_SCREEN_THRESHOLD = 0.65   # if optimal crop covers < this fraction of subjects, split
MAX_CROP_VELOCITY = 0.05        # max crop movement per frame (fraction of frame width)

# Face detection cascade
FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# Subject sampling - analyse every Nth frame then interpolate
ANALYSIS_INTERVAL = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SubjectInfo:
    """Detected subject in a frame."""
    subject_type: str = "object"  # face, person, text, object, motion
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    importance: float = 0.5
    label: str = ""
    frame_idx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cx(self) -> int:
        return self.x + self.width // 2

    @property
    def cy(self) -> int:
        return self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class CropWindow:
    """A crop window for a single frame."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    frame_idx: int = 0
    coverage: float = 1.0     # fraction of important subjects covered
    use_split: bool = False   # whether split-screen is needed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReframeResult:
    """Result of multi-subject reframe."""
    output_path: str = ""
    input_ratio: str = ""
    target_ratio: str = ""
    frame_count: int = 0
    subjects_detected: int = 0
    split_screen_frames: int = 0
    avg_coverage: float = 0.0
    video_width: int = 0
    video_height: int = 0
    output_width: int = 0
    output_height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------
def _parse_ratio(ratio_str: str) -> Tuple[int, int]:
    """Parse an aspect ratio string into (w, h) integers."""
    if ratio_str in ASPECT_RATIOS:
        return ASPECT_RATIOS[ratio_str]

    # Try parsing "W:H" format
    parts = ratio_str.replace("/", ":").split(":")
    if len(parts) == 2:
        try:
            return (int(float(parts[0]) * 100), int(float(parts[1]) * 100))
        except ValueError:
            pass

    raise ValueError(f"Unknown aspect ratio: {ratio_str}. Valid: {list(ASPECT_RATIOS.keys())}")


def _compute_output_dims(
    src_w: int, src_h: int, ratio_w: int, ratio_h: int,
) -> Tuple[int, int]:
    """Compute output dimensions that fit within the source while matching the target ratio.

    Output is always <= source dimensions. Ensures dimensions are even.
    """
    target_aspect = ratio_w / ratio_h

    # Try fitting by width first
    out_w = src_w
    out_h = int(src_w / target_aspect)

    if out_h > src_h:
        # Fit by height instead
        out_h = src_h
        out_w = int(src_h * target_aspect)

    # Ensure even dimensions (required by most codecs)
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    return max(out_w, 2), max(out_h, 2)


# ---------------------------------------------------------------------------
# Subject detection
# ---------------------------------------------------------------------------
def _detect_faces(frame_gray, frame_idx: int) -> List[SubjectInfo]:
    """Detect faces using OpenCV Haar cascade."""
    import cv2

    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__), "data", FACE_CASCADE_FILE,
    )
    if not os.path.isfile(cascade_path):
        # Try alternative location
        cascade_path = cv2.data.haarcascades + FACE_CASCADE_FILE

    try:
        cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        return []

    faces = cascade.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )

    subjects = []
    for (fx, fy, fw, fh) in faces:
        area = fw * fh
        # Larger faces get higher importance
        importance = SUBJECT_WEIGHTS["face"] * min(area / 10000, 1.5)
        subjects.append(SubjectInfo(
            subject_type="face",
            x=int(fx), y=int(fy),
            width=int(fw), height=int(fh),
            importance=round(min(importance, 1.0), 3),
            label="face",
            frame_idx=frame_idx,
        ))

    return subjects


def _detect_text_regions(frame_gray, frame_idx: int) -> List[SubjectInfo]:
    """Detect text regions using edge density and morphological operations.

    This is a lightweight heuristic (no OCR/deep learning required).
    """
    import cv2
    import numpy as np

    h, w = frame_gray.shape[:2]

    # Edge detection
    edges = cv2.Canny(frame_gray, 50, 150)

    # Dilate to merge nearby edges (text tends to cluster)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    subjects = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        # Filter: text regions are typically wider than tall and not too big
        aspect = cw / max(ch, 1)
        area_ratio = (cw * ch) / (w * h)

        if aspect < 1.5 or aspect > 20:
            continue
        if area_ratio < 0.005 or area_ratio > 0.3:
            continue
        if ch < 10 or cw < 30:
            continue

        # Check edge density within region
        region_edges = edges[y:y+ch, x:x+cw]
        density = np.mean(region_edges > 0)
        if density < 0.1:
            continue

        importance = SUBJECT_WEIGHTS["text"] * min(density * 2, 1.0)
        subjects.append(SubjectInfo(
            subject_type="text",
            x=x, y=y, width=cw, height=ch,
            importance=round(importance, 3),
            label="text",
            frame_idx=frame_idx,
        ))

    return subjects


def _detect_motion_hotspots(
    prev_gray, curr_gray, frame_idx: int,
) -> List[SubjectInfo]:
    """Detect regions with significant motion between consecutive frames."""
    import cv2
    import numpy as np

    if prev_gray is None:
        return []

    # Ensure same size
    if prev_gray.shape != curr_gray.shape:
        return []

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = curr_gray.shape[:2]
    subjects = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area_ratio = (cw * ch) / (w * h)

        if area_ratio < 0.01 or area_ratio > 0.5:
            continue

        motion_strength = np.mean(diff[y:y+ch, x:x+cw]) / 255.0
        importance = SUBJECT_WEIGHTS["motion"] * motion_strength
        subjects.append(SubjectInfo(
            subject_type="motion",
            x=x, y=y, width=cw, height=ch,
            importance=round(importance, 3),
            label="motion",
            frame_idx=frame_idx,
        ))

    return subjects


def _detect_all_subjects(
    frame_gray, prev_gray, frame_idx: int,
) -> List[SubjectInfo]:
    """Run all subject detectors on a frame."""
    subjects = []
    subjects.extend(_detect_faces(frame_gray, frame_idx))
    subjects.extend(_detect_text_regions(frame_gray, frame_idx))
    subjects.extend(_detect_motion_hotspots(prev_gray, frame_gray, frame_idx))

    # Sort by importance descending
    subjects.sort(key=lambda s: s.importance, reverse=True)

    return subjects


# ---------------------------------------------------------------------------
# Optimal crop computation
# ---------------------------------------------------------------------------
def _compute_crop_for_subjects(
    subjects: List[SubjectInfo],
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
    frame_idx: int,
) -> CropWindow:
    """Compute the optimal crop window that maximises subject coverage.

    Uses a weighted centroid approach: crop centre is the importance-weighted
    average of subject centres, then clamped to frame bounds.
    """
    if not subjects:
        # Centre crop fallback
        cx = frame_w // 2
        cy = frame_h // 2
        x = max(cx - crop_w // 2, 0)
        y = max(cy - crop_h // 2, 0)
        x = min(x, frame_w - crop_w)
        y = min(y, frame_h - crop_h)
        return CropWindow(
            x=max(x, 0), y=max(y, 0),
            width=crop_w, height=crop_h,
            frame_idx=frame_idx, coverage=0.0,
        )

    # Importance-weighted centroid
    total_weight = sum(s.importance for s in subjects)
    if total_weight < 1e-6:
        total_weight = 1.0

    wcx = sum(s.cx * s.importance for s in subjects) / total_weight
    wcy = sum(s.cy * s.importance for s in subjects) / total_weight

    # Crop position
    x = int(wcx - crop_w / 2)
    y = int(wcy - crop_h / 2)

    # Clamp to frame bounds
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))

    # Compute coverage: what fraction of important subjects are within the crop
    covered_weight = 0.0
    for s in subjects:
        # Check if subject centre is within crop
        if x <= s.cx <= x + crop_w and y <= s.cy <= y + crop_h:
            covered_weight += s.importance

    coverage = covered_weight / total_weight if total_weight > 0 else 0.0

    return CropWindow(
        x=x, y=y, width=crop_w, height=crop_h,
        frame_idx=frame_idx, coverage=round(coverage, 3),
        use_split=coverage < SPLIT_SCREEN_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Crop path smoothing
# ---------------------------------------------------------------------------
def _smooth_crop_path(
    crops: List[CropWindow],
    kernel_size: int = SMOOTH_KERNEL_SIZE,
    max_velocity: float = MAX_CROP_VELOCITY,
    frame_w: int = 1920,
) -> List[CropWindow]:
    """Smooth the crop path to avoid jarring jumps.

    1. Moving-average filter on (x, y) positions
    2. Velocity clamping to limit frame-to-frame movement
    """
    if len(crops) < 3:
        return crops

    half = kernel_size // 2

    # Step 1: Moving average
    smoothed_x = []
    smoothed_y = []

    for i in range(len(crops)):
        start = max(0, i - half)
        end = min(len(crops), i + half + 1)
        window = crops[start:end]
        avg_x = sum(c.x for c in window) / len(window)
        avg_y = sum(c.y for c in window) / len(window)
        smoothed_x.append(avg_x)
        smoothed_y.append(avg_y)

    # Step 2: Velocity clamping
    max_delta = int(frame_w * max_velocity)

    for i in range(1, len(smoothed_x)):
        dx = smoothed_x[i] - smoothed_x[i - 1]
        dy = smoothed_y[i] - smoothed_y[i - 1]

        if abs(dx) > max_delta:
            smoothed_x[i] = smoothed_x[i - 1] + max_delta * (1 if dx > 0 else -1)
        if abs(dy) > max_delta:
            smoothed_y[i] = smoothed_y[i - 1] + max_delta * (1 if dy > 0 else -1)

    # Apply smoothed positions
    result = []
    for i, crop in enumerate(crops):
        new_x = max(0, min(int(smoothed_x[i]), frame_w - crop.width))
        new_y = max(0, int(smoothed_y[i]))
        result.append(CropWindow(
            x=new_x, y=new_y,
            width=crop.width, height=crop.height,
            frame_idx=crop.frame_idx,
            coverage=crop.coverage,
            use_split=crop.use_split,
        ))

    return result


# ---------------------------------------------------------------------------
# Interpolate crops between analysed frames
# ---------------------------------------------------------------------------
def _interpolate_crops(
    analysed_crops: Dict[int, CropWindow],
    total_frames: int,
    crop_w: int,
    crop_h: int,
) -> List[CropWindow]:
    """Linearly interpolate crop positions between analysed key frames."""
    if not analysed_crops:
        return [CropWindow(x=0, y=0, width=crop_w, height=crop_h, frame_idx=i)
                for i in range(total_frames)]

    sorted_indices = sorted(analysed_crops.keys())
    result = []

    for fi in range(total_frames):
        # Find surrounding analysed frames
        prev_idx = None
        next_idx = None

        for idx in sorted_indices:
            if idx <= fi:
                prev_idx = idx
            if idx >= fi and next_idx is None:
                next_idx = idx

        if prev_idx is None:
            prev_idx = sorted_indices[0]
        if next_idx is None:
            next_idx = sorted_indices[-1]

        if prev_idx == next_idx or prev_idx == fi:
            crop = analysed_crops.get(prev_idx, analysed_crops[sorted_indices[0]])
            result.append(CropWindow(
                x=crop.x, y=crop.y,
                width=crop_w, height=crop_h,
                frame_idx=fi,
                coverage=crop.coverage,
                use_split=crop.use_split,
            ))
        else:
            # Linear interpolation
            t = (fi - prev_idx) / (next_idx - prev_idx)
            c1 = analysed_crops[prev_idx]
            c2 = analysed_crops[next_idx]
            ix = int(c1.x + (c2.x - c1.x) * t)
            iy = int(c1.y + (c2.y - c1.y) * t)
            icov = c1.coverage + (c2.coverage - c1.coverage) * t

            result.append(CropWindow(
                x=ix, y=iy,
                width=crop_w, height=crop_h,
                frame_idx=fi,
                coverage=round(icov, 3),
                use_split=c1.use_split or c2.use_split,
            ))

    return result


# ---------------------------------------------------------------------------
# Split-screen rendering
# ---------------------------------------------------------------------------
def _render_split_screen(
    frame, subjects: List[SubjectInfo],
    out_w: int, out_h: int,
) -> "numpy.ndarray":  # noqa: F821
    """Render a split-screen frame showing the two most important subject groups.

    Divides subjects into left/right groups by x-position, renders each
    half of the output from the corresponding region.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    if not subjects or len(subjects) < 2:
        # Just centre-crop
        cx, cy = w // 2, h // 2
        x1 = max(cx - out_w // 2, 0)
        y1 = max(cy - out_h // 2, 0)
        x2 = min(x1 + out_w, w)
        y2 = min(y1 + out_h, h)
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (out_w, out_h))

    # Sort subjects by x-position
    sorted_subj = sorted(subjects, key=lambda s: s.cx)
    mid = len(sorted_subj) // 2

    left_group = sorted_subj[:mid]
    right_group = sorted_subj[mid:]

    half_w = out_w // 2

    def _crop_region(group: List[SubjectInfo], target_w: int, target_h: int):
        gcx = int(sum(s.cx for s in group) / len(group))
        gcy = int(sum(s.cy for s in group) / len(group))
        x1 = max(gcx - target_w // 2, 0)
        y1 = max(gcy - target_h // 2, 0)
        x2 = min(x1 + target_w, w)
        y2 = min(y1 + target_h, h)
        # Adjust if we hit boundaries
        if x2 - x1 < target_w:
            x1 = max(x2 - target_w, 0)
        if y2 - y1 < target_h:
            y1 = max(y2 - target_h, 0)
        return frame[y1:y2, x1:x2]

    left_crop = _crop_region(left_group, half_w, out_h)
    right_crop = _crop_region(right_group, half_w, out_h)

    # Resize to exact half-width
    left_resized = cv2.resize(left_crop, (half_w, out_h))
    right_resized = cv2.resize(right_crop, (out_w - half_w, out_h))

    # Combine with a thin divider line
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    output[:, :half_w] = left_resized
    output[:, half_w:] = right_resized

    # Draw 2px dark divider
    if half_w > 0 and half_w < out_w:
        output[:, max(half_w - 1, 0):half_w + 1] = [40, 40, 40]

    return output


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def reframe_multi_subject(
    video_path: str,
    target_ratio: str = "9:16",
    output_dir: str = "",
    enable_split_screen: bool = True,
    analysis_interval: int = ANALYSIS_INTERVAL,
    smooth_strength: int = SMOOTH_KERNEL_SIZE,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Reframe video for a target aspect ratio with multi-subject awareness.

    Args:
        video_path: Path to input video.
        target_ratio: Target aspect ratio (e.g., "9:16", "1:1").
        output_dir: Output directory.
        enable_split_screen: Use split-screen when subjects are far apart.
        analysis_interval: Analyse every Nth frame (interpolate between).
        smooth_strength: Temporal smoothing kernel size.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReframeResult as dict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("OpenCV required for reframing")

    import cv2

    ratio_w, ratio_h = _parse_ratio(target_ratio)

    if on_progress:
        on_progress(2, f"Reframing to {target_ratio}...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    src_w = info.get("width", 1920)
    src_h = info.get("height", 1080)
    duration = info.get("duration", 0)

    input_ratio = f"{src_w}:{src_h}"
    out_w, out_h = _compute_output_dims(src_w, src_h, ratio_w, ratio_h)

    logger.info("Reframe: %dx%d (%s) -> %dx%d (%s)",
                src_w, src_h, input_ratio, out_w, out_h, target_ratio)

    # Step 1: Analyse frames for subjects
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = int(duration * fps)

    if on_progress:
        on_progress(5, f"Analysing {total_frames} frames for subjects...")

    analysed_crops = {}
    total_subjects = 0
    prev_gray = None

    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if fi % analysis_interval == 0:
            subjects = _detect_all_subjects(gray, prev_gray, fi)
            total_subjects += len(subjects)

            crop = _compute_crop_for_subjects(
                subjects, out_w, out_h, src_w, src_h, fi,
            )
            analysed_crops[fi] = crop

            if on_progress and fi % (analysis_interval * 10) == 0:
                pct = 5 + int(30 * fi / max(total_frames, 1))
                on_progress(min(pct, 35), f"Analysing frame {fi}/{total_frames}")

        prev_gray = gray

    cap.release()

    if on_progress:
        on_progress(40, "Computing smooth crop path...")

    # Step 2: Interpolate and smooth crop path
    all_crops = _interpolate_crops(analysed_crops, total_frames, out_w, out_h)
    all_crops = _smooth_crop_path(all_crops, kernel_size=smooth_strength,
                                   frame_w=src_w)

    split_count = sum(1 for c in all_crops if c.use_split)
    avg_coverage = sum(c.coverage for c in all_crops) / max(len(all_crops), 1)

    # Step 3: Render reframed video
    if on_progress:
        on_progress(45, "Rendering reframed video...")

    work_dir = tempfile.mkdtemp(prefix="reframe_")
    render_dir = os.path.join(work_dir, "rendered")
    os.makedirs(render_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    prev_gray = None

    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if fi >= len(all_crops):
            break

        crop = all_crops[fi]

        if crop.use_split and enable_split_screen:
            # Need subjects for split-screen; re-detect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = _detect_all_subjects(gray, prev_gray, fi)
            prev_gray = gray
            output_frame = _render_split_screen(frame, subjects, out_w, out_h)
        else:
            # Standard crop
            x1 = max(crop.x, 0)
            y1 = max(crop.y, 0)
            x2 = min(x1 + out_w, src_w)
            y2 = min(y1 + out_h, src_h)
            cropped = frame[y1:y2, x1:x2]
            output_frame = cv2.resize(cropped, (out_w, out_h))
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out_frame_path = os.path.join(render_dir, f"frame_{fi:06d}.png")
        cv2.imwrite(out_frame_path, output_frame)

        if on_progress and fi % 50 == 0:
            pct = 45 + int(45 * fi / max(total_frames, 1))
            on_progress(min(pct, 90), f"Rendering: {fi}/{total_frames}")

    cap.release()

    # Step 4: Encode with FFmpeg
    if on_progress:
        on_progress(92, "Encoding final video...")

    out_dir = output_dir or os.path.dirname(video_path)
    out_file = output_path(video_path, f"reframe_{target_ratio.replace(':', 'x')}",
                           output_dir=out_dir)
    frame_pattern = os.path.join(render_dir, "frame_%06d.png")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", "-pix_fmt", "yuv420p",
        out_file,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Multi-subject reframe complete")

    result = ReframeResult(
        output_path=out_file,
        input_ratio=input_ratio,
        target_ratio=target_ratio,
        frame_count=total_frames,
        subjects_detected=total_subjects,
        split_screen_frames=split_count,
        avg_coverage=round(avg_coverage, 3),
        video_width=src_w,
        video_height=src_h,
        output_width=out_w,
        output_height=out_h,
    )
    return result.to_dict()
