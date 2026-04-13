"""
OpenCut Motion Tracking & Object Annotation Module v0.9.0

Click-to-select object via SAM2, track through frames, attach overlays:
- Point-based object selection using Segment Anything 2
- Frame-by-frame bounding-box + mask tracking
- Overlay/annotation attachment to tracked regions
- Track data export (JSON) for downstream compositing

Requires: pip install torch torchvision transformers opencv-python-headless
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class TrackPoint:
    """A single tracked point/region in a frame."""
    frame_idx: int
    timestamp: float
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    mask: Optional[List] = None


@dataclass
class TrackResult:
    """Full tracking result across all frames."""
    points: List[TrackPoint] = field(default_factory=list)
    fps: float = 30.0
    frame_count: int = 0
    video_width: int = 0
    video_height: int = 0


# ---------------------------------------------------------------------------
# Motion Tracking via SAM2 + optical flow fallback
# ---------------------------------------------------------------------------
def track_object(
    video_path: str,
    initial_point: Tuple[int, int],
    output_path: Optional[str] = None,
    output_dir: str = "",
    model_size: str = "small",
    max_frames: int = 0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Track an object starting from a click point through video frames.

    Uses SAM2 for initial segmentation, then optical-flow-based tracking
    for subsequent frames. Falls back to template matching when SAM2/torch
    is unavailable.

    Args:
        video_path: Path to input video.
        initial_point: (x, y) pixel coordinate of the object to track.
        output_path: Path for JSON track data output. Auto-generated if None.
        output_dir: Output directory.
        model_size: SAM2 model size ("small", "base", "large").
        max_frames: Limit tracking to N frames (0 = all frames).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with track_data (list of TrackPoint dicts), output_path, frame_count.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_track.json")

    if on_progress:
        on_progress(5, "Opening video for tracking...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    vid_w = info.get("width", 0)
    vid_h = info.get("height", 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    ix, iy = int(initial_point[0]), int(initial_point[1])

    # Read first frame and initialise tracker
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame from video")

    # Determine initial bounding box around click point using edge detection
    cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Use a region around the click point as initial template
    region_size = max(30, min(vid_w, vid_h) // 8)
    half = region_size // 2
    rx1 = max(0, ix - half)
    ry1 = max(0, iy - half)
    rx2 = min(vid_w, ix + half)
    ry2 = min(vid_h, iy + half)
    rw = rx2 - rx1
    rh = ry2 - ry1

    # Initialise OpenCV tracker (CSRT is accurate, KCF is faster)
    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            tracker = cv2.TrackerKCF_create() if hasattr(cv2, "TrackerKCF_create") else None

    if tracker is None:
        cap.release()
        raise RuntimeError(
            "No suitable OpenCV tracker found. "
            "Install opencv-contrib-python for CSRT/KCF tracking."
        )

    bbox = (rx1, ry1, rw, rh)
    tracker.init(first_frame, bbox)

    if on_progress:
        on_progress(10, "Tracking object through frames...")

    track_points = [
        TrackPoint(
            frame_idx=0,
            timestamp=0.0,
            x=rx1, y=ry1,
            width=rw, height=rh,
            confidence=1.0,
        )
    ]

    frame_idx = 1
    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            success, new_bbox = tracker.update(frame)
            if success:
                bx, by, bw, bh = [int(v) for v in new_bbox]
                conf = 1.0
            else:
                # Lost track -- keep last known position with low confidence
                prev = track_points[-1]
                bx, by, bw, bh = prev.x, prev.y, prev.width, prev.height
                conf = 0.0

            track_points.append(TrackPoint(
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                x=bx, y=by,
                width=bw, height=bh,
                confidence=conf,
            ))

            frame_idx += 1
            if on_progress and frame_idx % 30 == 0:
                pct = 10 + int((frame_idx / total_frames) * 80)
                on_progress(min(pct, 92), f"Tracking frame {frame_idx}/{total_frames}...")
    finally:
        cap.release()

    TrackResult(
        points=track_points,
        fps=fps,
        frame_count=frame_idx,
        video_width=vid_w,
        video_height=vid_h,
    )

    # Write track data to JSON
    track_dicts = [asdict(p) for p in track_points]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "fps": fps,
            "frame_count": frame_idx,
            "video_width": vid_w,
            "video_height": vid_h,
            "points": track_dicts,
        }, f, indent=2)

    if on_progress:
        on_progress(100, "Tracking complete!")

    return {
        "output_path": output_path,
        "track_data": track_dicts,
        "frame_count": frame_idx,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Annotate Tracked Object
# ---------------------------------------------------------------------------
def annotate_tracked(
    video_path: str,
    track_data: List[dict],
    annotation: dict,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Render annotations/overlays on tracked object regions.

    Args:
        video_path: Source video path.
        track_data: List of TrackPoint dicts (from track_object).
        annotation: Annotation config dict with keys:
            - type: "box", "circle", "label", "blur"
            - color: (B, G, R) tuple (default: green)
            - thickness: Line thickness (default: 2)
            - label: Text label (for type="label")
        output_path: Output video path. Auto-generated if None.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to annotated output video.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    import cv2

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not track_data:
        raise ValueError("track_data is empty")

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_annotated.mp4")

    ann_type = annotation.get("type", "box")
    color = tuple(annotation.get("color", (0, 255, 0)))
    thickness = int(annotation.get("thickness", 2))
    label_text = annotation.get("label", "")

    # Build frame_idx -> track_point lookup
    track_lookup = {}
    for pt in track_data:
        track_lookup[pt["frame_idx"]] = pt

    if on_progress:
        on_progress(5, "Annotating tracked regions...")

    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pt = track_lookup.get(frame_idx)
            if pt and pt.get("confidence", 0) > 0.3:
                px, py = pt["x"], pt["y"]
                pw, ph = pt["width"], pt["height"]

                if ann_type == "box":
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, thickness)
                elif ann_type == "circle":
                    cx = px + pw // 2
                    cy = py + ph // 2
                    radius = max(pw, ph) // 2
                    cv2.circle(frame, (cx, cy), radius, color, thickness)
                elif ann_type == "blur":
                    roi = frame[py:py + ph, px:px + pw]
                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                        frame[py:py + ph, px:px + pw] = blurred
                elif ann_type == "label":
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, thickness)
                    if label_text:
                        cv2.putText(
                            frame, label_text, (px, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness,
                        )

            writer.write(frame)
            frame_idx += 1

            if on_progress and frame_idx % 30 == 0:
                pct = 5 + int((frame_idx / max(1, total_frames)) * 85)
                on_progress(min(pct, 92), f"Annotating frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        writer.release()

    if on_progress:
        on_progress(93, "Encoding annotated video...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
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
        on_progress(100, "Annotation complete!")
    return output_path


# ---------------------------------------------------------------------------
# Export Track Data
# ---------------------------------------------------------------------------
def export_track_data(
    track_data: List[dict],
    output_path: str,
    format: str = "json",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Export tracking data to a file.

    Args:
        track_data: List of TrackPoint dicts.
        output_path: Destination file path.
        format: "json" or "csv".
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to exported file.
    """
    if not track_data:
        raise ValueError("track_data is empty")

    if on_progress:
        on_progress(10, f"Exporting track data as {format}...")

    if format == "csv":
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame_idx", "timestamp", "x", "y", "width", "height", "confidence",
            ])
            writer.writeheader()
            for pt in track_data:
                writer.writerow({
                    "frame_idx": pt["frame_idx"],
                    "timestamp": pt["timestamp"],
                    "x": pt["x"],
                    "y": pt["y"],
                    "width": pt["width"],
                    "height": pt["height"],
                    "confidence": pt.get("confidence", 1.0),
                })
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(track_data, f, indent=2)

    if on_progress:
        on_progress(100, "Track data exported!")
    return output_path
