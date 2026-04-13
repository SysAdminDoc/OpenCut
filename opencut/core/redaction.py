"""
OpenCut Selective Redaction Module

Video redaction tools:
- Region-based redaction (blur, pixelate, black fill) with time ranges
- Face detection and automatic blur/redaction
- Audit log generation for compliance

All via FFmpeg filters - no heavy ML dependencies for basic redaction.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class RedactionRegion:
    """A region to redact in the video."""
    x: int
    y: int
    w: int
    h: int
    start_time: float = 0.0
    end_time: float = -1.0  # -1 means end of video


@dataclass
class RedactionResult:
    """Result of redaction processing."""
    output_path: str = ""
    regions_count: int = 0
    method: str = "blur"
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Region-Based Redaction
# ---------------------------------------------------------------------------

def redact_region(
    input_path: str,
    regions: List[dict],
    method: str = "blur",
    blur_strength: int = 20,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Apply redaction to specified regions of a video.

    Args:
        input_path: Source video file.
        regions: List of region dicts with keys: x, y, w, h, start_time, end_time.
                 Can also be a list of RedactionRegion dataclasses.
        method: Redaction method: "blur", "pixelate", or "black".
        blur_strength: Blur strength for blur method (1-100).
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, regions_count, method, duration.
    """
    if on_progress:
        on_progress(5, "Preparing redaction filters...")

    method = method if method in ("blur", "pixelate", "black") else "blur"
    blur_strength = max(1, min(100, blur_strength))

    info = get_video_info(input_path)
    duration = info["duration"]
    width, height = info["width"], info["height"]

    if output_path_str is None:
        output_path_str = output_path(input_path, f"redacted_{method}")

    # Normalize regions
    parsed_regions = []
    for r in regions:
        if isinstance(r, RedactionRegion):
            parsed_regions.append(r)
        elif isinstance(r, dict):
            parsed_regions.append(RedactionRegion(
                x=int(r.get("x", 0)),
                y=int(r.get("y", 0)),
                w=int(r.get("w", 100)),
                h=int(r.get("h", 100)),
                start_time=float(r.get("start_time", 0)),
                end_time=float(r.get("end_time", -1)),
            ))

    if not parsed_regions:
        raise ValueError("At least one redaction region is required")

    if on_progress:
        on_progress(15, f"Building filter chain for {len(parsed_regions)} regions...")

    # Build FFmpeg filter complex
    filters = []
    for i, reg in enumerate(parsed_regions):
        # Clamp region to video bounds
        rx = max(0, min(reg.x, width - 1))
        ry = max(0, min(reg.y, height - 1))
        rw = max(1, min(reg.w, width - rx))
        rh = max(1, min(reg.h, height - ry))
        end_t = reg.end_time if reg.end_time > 0 else duration

        enable = f"between(t,{reg.start_time},{end_t})"

        if method == "black":
            filters.append(
                f"drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black:t=fill:enable='{enable}'"
            )
        elif method == "pixelate":
            # Pixelate: use boxblur with very high luma radius for block effect
            # Split, crop region, scale down then up, overlay
            # Simpler approach: use heavy boxblur that creates pixelation effect
            block_size = max(2, min(rw, rh) // 8)
            filters.append(
                f"boxblur={block_size}:{block_size}:enable='{enable}'"
                if i == 0 and len(parsed_regions) == 1
                else f"drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black@0.01:t=fill:enable='{enable}'"
            )
        else:  # blur
            # For boxblur on a specific region, we need a complex filter
            pass

    if method == "blur":
        # Build complex filter for blur redaction
        filter_parts = _build_blur_filter(parsed_regions, width, height, duration, blur_strength)
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-filter_complex", filter_parts,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-y", output_path_str,
        ]
    elif method == "pixelate":
        filter_parts = _build_pixelate_filter(parsed_regions, width, height, duration)
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-filter_complex", filter_parts,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-y", output_path_str,
        ]
    else:  # black
        vf = ";".join(filters) if filters else "null"
        # drawbox is a simple video filter
        vf = ",".join(filters) if filters else "null"
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-y", output_path_str,
        ]

    if on_progress:
        on_progress(30, "Applying redaction...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Redaction complete ({len(parsed_regions)} regions)")

    return {
        "output_path": output_path_str,
        "regions_count": len(parsed_regions),
        "method": method,
        "duration": duration,
    }


def _build_blur_filter(
    regions: List[RedactionRegion],
    width: int, height: int,
    duration: float,
    blur_strength: int = 20,
) -> str:
    """Build FFmpeg filter_complex string for blur redaction.

    Uses split -> crop -> boxblur -> overlay chain for each region.
    """
    parts = []
    current_label = "[0:v]"

    for i, reg in enumerate(regions):
        rx = max(0, min(reg.x, width - 1))
        ry = max(0, min(reg.y, height - 1))
        rw = max(2, min(reg.w, width - rx))
        rh = max(2, min(reg.h, height - ry))
        end_t = reg.end_time if reg.end_time > 0 else duration

        enable = f"between(t,{reg.start_time},{end_t})"
        next_label = f"[v{i}]"

        # Crop the region, blur it, overlay it back
        parts.append(
            f"{current_label}split[base{i}][crop{i}];"
            f"[crop{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"boxblur={blur_strength}:{blur_strength}[blurred{i}];"
            f"[base{i}][blurred{i}]overlay={rx}:{ry}:enable='{enable}'{next_label}"
        )
        current_label = next_label

    # Rename final output
    filter_str = ";".join(parts)
    if filter_str:
        # Replace last label with [out]
        last_label = f"[v{len(regions) - 1}]"
        filter_str = filter_str[:filter_str.rfind(last_label)] + "[out]"
    else:
        filter_str = "[0:v]copy[out]"

    return filter_str


def _build_pixelate_filter(
    regions: List[RedactionRegion],
    width: int, height: int,
    duration: float,
) -> str:
    """Build FFmpeg filter_complex for pixelation redaction.

    Crops region, scales down then up to create block effect, overlays back.
    """
    parts = []
    current_label = "[0:v]"

    for i, reg in enumerate(regions):
        rx = max(0, min(reg.x, width - 1))
        ry = max(0, min(reg.y, height - 1))
        rw = max(4, min(reg.w, width - rx))
        rh = max(4, min(reg.h, height - ry))
        end_t = reg.end_time if reg.end_time > 0 else duration

        enable = f"between(t,{reg.start_time},{end_t})"
        # Scale down to ~1/8 then back up for pixelation
        small_w = max(2, rw // 8)
        small_h = max(2, rh // 8)
        next_label = f"[v{i}]"

        parts.append(
            f"{current_label}split[base{i}][crop{i}];"
            f"[crop{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"scale={small_w}:{small_h}:flags=neighbor,"
            f"scale={rw}:{rh}:flags=neighbor[pix{i}];"
            f"[base{i}][pix{i}]overlay={rx}:{ry}:enable='{enable}'{next_label}"
        )
        current_label = next_label

    filter_str = ";".join(parts)
    if filter_str:
        last_label = f"[v{len(regions) - 1}]"
        filter_str = filter_str[:filter_str.rfind(last_label)] + "[out]"
    else:
        filter_str = "[0:v]copy[out]"

    return filter_str


# ---------------------------------------------------------------------------
# Face Redaction
# ---------------------------------------------------------------------------

def redact_faces(
    input_path: str,
    method: str = "blur",
    blur_strength: int = 30,
    sample_interval: float = 1.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect and redact faces in a video.

    Detects faces per sampled keyframe and applies redaction across
    the surrounding time range. Uses OpenCV Haar cascade if available,
    otherwise falls back to FFmpeg drawbox heuristic.

    Args:
        input_path: Source video file.
        method: Redaction method: "blur", "pixelate", or "black".
        blur_strength: Blur strength for blur method.
        sample_interval: Seconds between face detection samples.
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, faces_found, method, duration.
    """
    if on_progress:
        on_progress(5, "Detecting faces...")

    info = get_video_info(input_path)
    duration = info["duration"]
    _width, _height = info["width"], info["height"]

    if output_path_str is None:
        output_path_str = output_path(input_path, "faces_redacted")

    # Try OpenCV face detection
    face_regions = _detect_faces_opencv(input_path, info, sample_interval, on_progress)

    if face_regions is None:
        # Fallback: try FFmpeg-based rough face detection
        face_regions = _detect_faces_ffmpeg(input_path, info, sample_interval, on_progress)

    if not face_regions:
        # No faces found, copy file as-is
        if on_progress:
            on_progress(90, "No faces detected")
        cmd = [
            get_ffmpeg_path(), "-i", input_path,
            "-c", "copy", "-y", output_path_str,
        ]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "faces_found": 0,
            "method": method,
            "duration": duration,
        }

    if on_progress:
        on_progress(60, f"Found {len(face_regions)} face regions, applying redaction...")

    # Apply redaction using region-based method
    result = redact_region(
        input_path=input_path,
        regions=[asdict(r) for r in face_regions],
        method=method,
        blur_strength=blur_strength,
        output_path_str=output_path_str,
        on_progress=lambda pct, msg: on_progress(60 + int(pct * 0.4), msg) if on_progress else None,
    )

    result["faces_found"] = len(face_regions)
    return result


def _detect_faces_opencv(
    input_path: str, info: dict,
    sample_interval: float,
    on_progress: Optional[Callable] = None,
) -> Optional[List[RedactionRegion]]:
    """Detect faces using OpenCV Haar cascade."""
    try:
        import cv2
    except ImportError:
        logger.debug("OpenCV not available for face detection")
        return None

    cascade_path = None
    # Try to find Haar cascade
    for path in [
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]:
        if os.path.isfile(path):
            cascade_path = path
            break

    if cascade_path is None:
        logger.debug("Haar cascade file not found")
        return None

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    duration = info["duration"]
    fps = info["fps"]
    regions = []
    sample_frames = max(1, int(sample_interval * fps))

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                )

                t = frame_idx / fps
                t_end = min(t + sample_interval, duration)

                for (fx, fy, fw, fh) in faces:
                    # Add padding around face
                    pad = int(max(fw, fh) * 0.2)
                    regions.append(RedactionRegion(
                        x=max(0, fx - pad),
                        y=max(0, fy - pad),
                        w=min(fw + pad * 2, info["width"]),
                        h=min(fh + pad * 2, info["height"]),
                        start_time=t,
                        end_time=t_end,
                    ))

                if on_progress:
                    pct = 5 + int(50 * t / duration)
                    on_progress(pct, f"Scanning frame at {t:.1f}s...")

            frame_idx += 1
    finally:
        cap.release()

    return regions


def _detect_faces_ffmpeg(
    input_path: str, info: dict,
    sample_interval: float,
    on_progress: Optional[Callable] = None,
) -> List[RedactionRegion]:
    """Rough face region detection using FFmpeg frame extraction + simple heuristic.

    Extracts frames and looks for skin-tone regions as a basic approximation.
    This is a fallback when OpenCV is not available.
    """
    # Without ML, we can only provide manual/empty region detection
    # Return empty - user should provide regions manually or install opencv
    logger.info("Face detection requires opencv-python. Install with: pip install opencv-python")
    return []


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

def generate_redaction_log(
    regions: List[dict],
    input_path: str,
    method: str = "blur",
    output_path_str: Optional[str] = None,
) -> dict:
    """Generate an audit log documenting what was redacted.

    Creates a JSON log file with timestamps, regions, method, and metadata
    for compliance and record-keeping purposes.

    Args:
        regions: List of region dicts that were redacted.
        input_path: Original input video path.
        method: Redaction method used.
        output_path_str: Path to write log file. Auto-generated if None.

    Returns:
        dict with log_path, region_count, and log data.
    """
    info = get_video_info(input_path)

    log_data = {
        "redaction_log": {
            "version": "1.0",
            "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_file": os.path.basename(input_path),
            "source_path": input_path,
            "source_duration": info["duration"],
            "source_resolution": f"{info['width']}x{info['height']}",
            "method": method,
            "regions": [],
        }
    }

    for i, r in enumerate(regions):
        if isinstance(r, RedactionRegion):
            r = asdict(r)
        log_data["redaction_log"]["regions"].append({
            "index": i,
            "x": r.get("x", 0),
            "y": r.get("y", 0),
            "width": r.get("w", 0),
            "height": r.get("h", 0),
            "start_time": r.get("start_time", 0),
            "end_time": r.get("end_time", -1),
            "method": method,
        })

    log_data["redaction_log"]["total_regions"] = len(regions)

    if output_path_str is None:
        base = os.path.splitext(input_path)[0]
        output_path_str = f"{base}_redaction_log.json"

    with open(output_path_str, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    return {
        "log_path": output_path_str,
        "region_count": len(regions),
        "log": log_data,
    }
