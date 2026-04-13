"""
OpenCut Planar Tracking Module v1.0.0

Track flat surfaces through camera movement and perspective changes,
outputting corner-pin data for inserting replacement content.

Uses OpenCV ORB feature matching + findHomography for robust
perspective tracking. Falls back to a clear error when OpenCV
is not installed.

Pipeline: user selects 4 corners -> ORB tracks features per frame
          -> homography warps corners -> replacement composited.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackRegion:
    """Four corner points defining a planar region (TL, TR, BR, BL)."""
    corners: List[Tuple[float, float]]  # exactly 4 (x, y) tuples

    def __post_init__(self):
        if len(self.corners) != 4:
            raise ValueError(
                f"TrackRegion requires exactly 4 corners, got {len(self.corners)}"
            )
        self.corners = [(float(x), float(y)) for x, y in self.corners]

    @property
    def center(self) -> Tuple[float, float]:
        """Return the centroid of the four corners."""
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    @property
    def area(self) -> float:
        """Approximate area using the shoelace formula."""
        n = len(self.corners)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += self.corners[i][0] * self.corners[j][1]
            a -= self.corners[j][0] * self.corners[i][1]
        return abs(a) / 2.0

    def as_list(self) -> List[List[float]]:
        """Return corners as [[x, y], ...] for JSON serialization."""
        return [[c[0], c[1]] for c in self.corners]


@dataclass
class TrackResult:
    """Result of planar surface tracking across multiple frames."""
    frames: List[TrackRegion]
    confidence_per_frame: List[float]
    fps: float
    total_frames: int

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_per_frame:
            return 0.0
        return sum(self.confidence_per_frame) / len(self.confidence_per_frame)

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.total_frames / self.fps


@dataclass
class PlanarInsert:
    """Configuration for inserting replacement content onto tracked surface."""
    track_data: TrackResult
    replacement_path: str
    output_path: str


# ---------------------------------------------------------------------------
# OpenCV availability check
# ---------------------------------------------------------------------------

def _require_cv2():
    """Ensure OpenCV is available, raise clear error if not."""
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError(
            "OpenCV is required for planar tracking. "
            "Install with: pip install opencv-python-headless"
        )
    import cv2
    return cv2


# ---------------------------------------------------------------------------
# Planar surface tracking
# ---------------------------------------------------------------------------

def track_planar_surface(
    video_path: str,
    initial_corners: List[Tuple[float, float]],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> TrackResult:
    """
    Track 4 corner points of a planar surface through video frames.

    Uses ORB feature detection + BFMatcher + findHomography to compute
    per-frame perspective transforms of the initial region.

    Args:
        video_path:      Source video path.
        initial_corners: 4 (x, y) tuples defining the region on start_frame
                         (TL, TR, BR, BL order).
        start_frame:     Frame index to begin tracking (default 0).
        end_frame:       Frame index to stop (None = end of video).
        on_progress:     Callback(pct: int, msg: str).

    Returns:
        TrackResult with per-frame corner data and confidence scores.
    """
    cv2 = _require_cv2()
    import numpy as np

    if len(initial_corners) != 4:
        raise ValueError(f"Exactly 4 corners required, got {len(initial_corners)}")

    if on_progress:
        on_progress(5, "Opening video for tracking...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None or end_frame > total:
        end_frame = total
    if start_frame < 0:
        start_frame = 0
    if start_frame >= end_frame:
        cap.release()
        raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read frame {start_frame}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Detect features in initial frame
    prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
    if prev_desc is None or len(prev_kp) < 4:
        cap.release()
        raise RuntimeError(
            "Not enough features detected in initial frame. "
            "Try selecting a more textured region."
        )

    # Current corners (will be updated each frame)
    current_corners = np.array(initial_corners, dtype=np.float64)

    tracked_frames: List[TrackRegion] = [
        TrackRegion(corners=[tuple(c) for c in current_corners.tolist()])
    ]
    confidences: List[float] = [1.0]

    if on_progress:
        on_progress(10, f"Tracking frames {start_frame} to {end_frame}...")

    frame_count = end_frame - start_frame
    np.eye(3, dtype=np.float64)

    for i in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)

        confidence = 0.0
        if desc is not None and prev_desc is not None and len(kp) >= 4:
            matches = bf.match(prev_desc, desc)
            matches = sorted(matches, key=lambda m: m.distance)

            # Use top matches (at least 4 for homography)
            good_matches = matches[:min(100, len(matches))]

            if len(good_matches) >= 4:
                src_pts = np.float32(
                    [prev_kp[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                H, inlier_mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0
                )

                if H is not None:
                    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
                    confidence = inliers / max(len(good_matches), 1)

                    # Validate homography (reject degenerate transforms)
                    det = abs(np.linalg.det(H[:2, :2]))
                    if 0.1 < det < 10.0 and confidence > 0.2:
                        # Apply homography to corners
                        corners_arr = current_corners.reshape(-1, 1, 2).astype(np.float64)
                        new_corners = cv2.perspectiveTransform(corners_arr, H)
                        current_corners = new_corners.reshape(-1, 2)
                    else:
                        confidence *= 0.3  # Low confidence, keep previous corners
                else:
                    confidence = 0.0

        tracked_frames.append(
            TrackRegion(corners=[tuple(c) for c in current_corners.tolist()])
        )
        confidences.append(confidence)

        prev_gray = gray
        prev_kp = kp
        prev_desc = desc

        if on_progress and i % 30 == 0:
            pct = 10 + int((i / frame_count) * 85)
            on_progress(pct, f"Tracking frame {i}/{frame_count} (confidence: {confidence:.1%})...")

    cap.release()

    if on_progress:
        on_progress(100, f"Tracking complete: {len(tracked_frames)} frames")

    return TrackResult(
        frames=tracked_frames,
        confidence_per_frame=confidences,
        fps=fps,
        total_frames=len(tracked_frames),
    )


# ---------------------------------------------------------------------------
# Content insertion
# ---------------------------------------------------------------------------

def insert_replacement(
    video_path: str,
    track_result: TrackResult,
    replacement_image_or_video: str,
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Composite replacement content onto a tracked planar surface.

    Warps the replacement image/video to fit the tracked corners for each
    frame and overlays it on the source video.

    Args:
        video_path:                Source video.
        track_result:              TrackResult from track_planar_surface().
        replacement_image_or_video: Path to replacement image or video.
        out_path:                   Output path (auto-generated if None).
        on_progress:               Callback(pct: int, msg: str).

    Returns:
        Output video path.
    """
    cv2 = _require_cv2()
    import numpy as np

    if not os.path.isfile(replacement_image_or_video):
        raise FileNotFoundError(f"Replacement file not found: {replacement_image_or_video}")

    if on_progress:
        on_progress(5, "Preparing replacement content...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0) or 30.0
    vid_w = info.get("width", 0)
    vid_h = info.get("height", 0)
    if vid_w <= 0 or vid_h <= 0:
        raise RuntimeError(f"Cannot determine video dimensions: {vid_w}x{vid_h}")

    if out_path is None:
        out_path = _output_path(video_path, "planar_insert")

    # Load replacement content
    is_video_replacement = replacement_image_or_video.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    )
    if is_video_replacement:
        rep_cap = cv2.VideoCapture(replacement_image_or_video)
        if not rep_cap.isOpened():
            raise RuntimeError(f"Cannot open replacement video: {replacement_image_or_video}")
    else:
        rep_image = cv2.imread(replacement_image_or_video)
        if rep_image is None:
            raise RuntimeError(f"Cannot read replacement image: {replacement_image_or_video}")
        rep_cap = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (vid_w, vid_h))
    if not writer.isOpened():
        cap.release()
        if rep_cap:
            rep_cap.release()
        raise RuntimeError("Failed to create video writer")

    if on_progress:
        on_progress(10, "Compositing replacement content...")

    total = len(track_result.frames)
    try:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # Get replacement frame
            if rep_cap is not None:
                rret, rep_frame = rep_cap.read()
                if not rret:
                    rep_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    rret, rep_frame = rep_cap.read()
                    if not rret:
                        rep_frame = rep_image if rep_image is not None else frame
            else:
                rep_frame = rep_image

            if frame_idx < len(track_result.frames):
                region = track_result.frames[frame_idx]
                dst_corners = np.array(region.corners, dtype=np.float32)

                rep_h, rep_w = rep_frame.shape[:2]
                src_corners = np.array([
                    [0, 0],
                    [rep_w - 1, 0],
                    [rep_w - 1, rep_h - 1],
                    [0, rep_h - 1],
                ], dtype=np.float32)

                # Compute perspective transform
                M = cv2.getPerspectiveTransform(src_corners, dst_corners)
                warped = cv2.warpPerspective(
                    rep_frame, M, (vid_w, vid_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                # Create mask from warped region
                mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
                pts = dst_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)

                # Composite
                mask_3ch = (mask > 127).astype(np.float32)[:, :, None]
                frame = (
                    warped * mask_3ch + frame * (1.0 - mask_3ch)
                ).astype(np.uint8)

            writer.write(frame)

            if on_progress and frame_idx % 30 == 0:
                pct = 10 + int((frame_idx / max(total, 1)) * 80)
                on_progress(pct, f"Compositing frame {frame_idx}/{total}...")

        # Write remaining source frames beyond tracked region
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        if rep_cap is not None:
            rep_cap.release()

    if on_progress:
        on_progress(92, "Encoding final output with audio...")

    # Mux audio from source
    try:
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Replacement inserted!")

    return out_path


# ---------------------------------------------------------------------------
# Track data export
# ---------------------------------------------------------------------------

def export_track_data(track_result: TrackResult, format: str = "json") -> str:
    """
    Export corner-pin tracking data.

    Args:
        track_result: TrackResult from track_planar_surface().
        format:       Export format ("json", "csv", "nuke").

    Returns:
        Serialized tracking data as a string.
    """
    if format == "json":
        data = {
            "fps": track_result.fps,
            "total_frames": track_result.total_frames,
            "avg_confidence": round(track_result.avg_confidence, 4),
            "frames": [],
        }
        for i, region in enumerate(track_result.frames):
            data["frames"].append({
                "frame": i,
                "corners": region.as_list(),
                "confidence": round(track_result.confidence_per_frame[i], 4)
                if i < len(track_result.confidence_per_frame) else 0.0,
            })
        return json.dumps(data, indent=2)

    elif format == "csv":
        lines = ["frame,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y,confidence"]
        for i, region in enumerate(track_result.frames):
            c = region.corners
            conf = (track_result.confidence_per_frame[i]
                    if i < len(track_result.confidence_per_frame) else 0.0)
            lines.append(
                f"{i},{c[0][0]:.2f},{c[0][1]:.2f},"
                f"{c[1][0]:.2f},{c[1][1]:.2f},"
                f"{c[2][0]:.2f},{c[2][1]:.2f},"
                f"{c[3][0]:.2f},{c[3][1]:.2f},"
                f"{conf:.4f}"
            )
        return "\n".join(lines)

    elif format == "nuke":
        # Nuke CornerPin2D format
        lines = [
            "# Nuke CornerPin2D tracking data",
            f"# Exported from OpenCut ({track_result.total_frames} frames @ {track_result.fps:.2f}fps)",
            "",
        ]
        corner_names = ["to1", "to2", "to3", "to4"]
        for ci, name in enumerate(corner_names):
            lines.append(f"{name} {{curve")
            values = []
            for i, region in enumerate(track_result.frames):
                x, y = region.corners[ci]
                values.append(f"x{i} {{{x:.2f} {y:.2f}}}")
            lines.append(" ".join(values))
            lines.append("}")
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown export format: '{format}'. Use 'json', 'csv', or 'nuke'.")


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_track_frame(
    video_path: str,
    track_result: TrackResult,
    frame_number: int = 0,
) -> Dict:
    """
    Visualize tracking on a single frame by drawing the tracked region.

    Args:
        video_path:    Source video.
        track_result:  TrackResult with per-frame corner data.
        frame_number:  Frame index to visualize.

    Returns:
        Dict with preview_path (PNG), frame_number, and confidence.
    """
    cv2 = _require_cv2()
    import numpy as np

    if frame_number < 0 or frame_number >= len(track_result.frames):
        raise ValueError(
            f"frame_number {frame_number} out of range "
            f"(0-{len(track_result.frames) - 1})"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number}")

    region = track_result.frames[frame_number]
    corners = np.array(region.corners, dtype=np.int32)

    # Draw the tracked quadrilateral
    pts = corners.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw corner markers
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            frame, str(i + 1), (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Draw confidence
    conf = (track_result.confidence_per_frame[frame_number]
            if frame_number < len(track_result.confidence_per_frame) else 0.0)
    cv2.putText(
        frame, f"Confidence: {conf:.1%}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    preview_path = tempfile.NamedTemporaryFile(
        suffix=".png", prefix="planar_preview_", delete=False
    ).name
    cv2.imwrite(preview_path, frame)

    return {
        "preview_path": preview_path,
        "frame_number": frame_number,
        "confidence": conf,
        "corners": region.as_list(),
    }
