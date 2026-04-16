"""
OpenCut OCR-Based PII Redaction

Extract frames, run OCR (pytesseract), detect PII via regex patterns
(SSN, phone, email, credit card, etc.), track text regions, apply blur
mask, and generate a PII report.

Uses FFmpeg for frame I/O and pytesseract for OCR when available.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "date_of_birth": re.compile(r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "drivers_license": re.compile(r"\b[A-Z]\d{7,14}\b"),
}


@dataclass
class PIIDetection:
    """A detected PII region in a frame."""
    x: int
    y: int
    w: int
    h: int
    frame: int = 0
    timestamp: float = 0.0
    pii_type: str = ""
    matched_text: str = ""
    confidence: float = 0.0


@dataclass
class PIIRedactResult:
    """Result of PII redaction processing."""
    output_path: str = ""
    pii_found: int = 0
    pii_types: List[str] = field(default_factory=list)
    report_path: str = ""


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def _extract_frames(input_path: str, output_dir: str, sample_fps: float = 1.0) -> None:
    """Extract frames for OCR processing."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"fps={sample_fps}",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    run_ffmpeg(cmd)


def _ocr_frame(frame_path: str) -> List[dict]:
    """Run OCR on a single frame. Returns list of {text, x, y, w, h, conf}."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.info("pytesseract/Pillow not available; install with: pip install pytesseract Pillow")
        return []

    try:
        img = Image.open(frame_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        logger.warning("OCR failed on %s: %s", frame_path, e)
        return []

    results = []
    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        text = str(data["text"][i]).strip()
        conf = float(data["conf"][i]) if data["conf"][i] != -1 else 0.0
        if text and conf > 30:
            results.append({
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf,
            })
    return results


def _scan_for_pii(
    ocr_results: List[dict],
    frame_idx: int,
    timestamp: float,
    pii_types: Optional[List[str]] = None,
) -> List[PIIDetection]:
    """Scan OCR text for PII patterns."""
    detections = []
    patterns = PII_PATTERNS
    if pii_types:
        patterns = {k: v for k, v in PII_PATTERNS.items() if k in pii_types}

    # Concatenate nearby text boxes for multi-word matching
    full_text = " ".join(r["text"] for r in ocr_results)

    for pii_type, pattern in patterns.items():
        matches = pattern.finditer(full_text)
        for match in matches:
            matched = match.group()
            # Find the OCR box(es) that contain this match
            for r in ocr_results:
                if r["text"] in matched or matched in r["text"]:
                    detections.append(PIIDetection(
                        x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                        frame=frame_idx, timestamp=timestamp,
                        pii_type=pii_type, matched_text=matched,
                        confidence=r["conf"],
                    ))
                    break
            else:
                # No exact box match, use first OCR result as approximate
                if ocr_results:
                    r = ocr_results[0]
                    detections.append(PIIDetection(
                        x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                        frame=frame_idx, timestamp=timestamp,
                        pii_type=pii_type, matched_text=matched,
                        confidence=r["conf"],
                    ))

    return detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_pii(
    input_path: str,
    sample_fps: float = 1.0,
    pii_types: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect PII (personally identifiable information) in video frames via OCR.

    Args:
        input_path: Source video file.
        sample_fps: Frames per second for OCR sampling.
        pii_types: List of PII types to detect (None = all).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with detections, pii_found count, types found.
    """
    sample_fps = max(0.25, min(5.0, float(sample_fps)))
    get_video_info(input_path)

    if on_progress:
        on_progress(10, "Extracting frames for OCR...")

    tmpdir = tempfile.mkdtemp(prefix="opencut_pii_")
    try:
        _extract_frames(input_path, tmpdir, sample_fps)

        frame_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        all_detections: List[PIIDetection] = []

        for idx, fname in enumerate(frame_files):
            fpath = os.path.join(tmpdir, fname)
            t = idx / sample_fps

            if on_progress:
                pct = 10 + int(70 * idx / max(len(frame_files), 1))
                on_progress(pct, f"OCR scanning frame {idx + 1}/{len(frame_files)}...")

            ocr_results = _ocr_frame(fpath)
            pii_dets = _scan_for_pii(ocr_results, idx, t, pii_types)
            all_detections.extend(pii_dets)

        types_found = sorted(set(d.pii_type for d in all_detections))

        if on_progress:
            on_progress(100, f"Found {len(all_detections)} PII instances ({', '.join(types_found) or 'none'})")

        return {
            "detections": [asdict(d) for d in all_detections],
            "pii_found": len(all_detections),
            "pii_types": types_found,
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def redact_pii(
    input_path: str,
    detections: Optional[List[dict]] = None,
    pii_types: Optional[List[str]] = None,
    blur_strength: int = 30,
    sample_fps: float = 1.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect and redact PII in video frames.

    Args:
        input_path: Source video file.
        detections: Pre-computed PII detections (optional; auto-detected if None).
        pii_types: PII types to scan for.
        blur_strength: Blur kernel size (1-100).
        sample_fps: Sampling rate for detection.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, pii_found, pii_types, report_path.
    """
    blur_strength = max(1, min(100, int(blur_strength)))

    if output_path_str is None:
        output_path_str = output_path(input_path, "pii_redacted")

    info = get_video_info(input_path)
    duration = info["duration"]
    width, height = info["width"], info["height"]

    if detections is None:
        if on_progress:
            on_progress(5, "Detecting PII...")
        det_result = detect_pii(
            input_path, sample_fps=sample_fps, pii_types=pii_types,
            on_progress=lambda p, m: on_progress(5 + int(p * 0.4), m) if on_progress else None,
        )
        detections = det_result["detections"]
        pii_types_found = det_result["pii_types"]
    else:
        pii_types_found = sorted(set(d.get("pii_type", "") for d in detections))

    if not detections:
        if on_progress:
            on_progress(90, "No PII detected, copying original...")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "pii_found": 0,
            "pii_types": [],
            "report_path": "",
        }

    if on_progress:
        on_progress(50, f"Redacting {len(detections)} PII regions...")

    # Build blur filter chain
    filter_parts = []
    current_label = "[0:v]"

    for i, det in enumerate(detections):
        rx = max(0, min(int(det.get("x", 0)), width - 1))
        ry = max(0, min(int(det.get("y", 0)), height - 1))
        rw = max(2, min(int(det.get("w", 50)), width - rx))
        rh = max(2, min(int(det.get("h", 20)), height - ry))
        t = float(det.get("timestamp", 0))
        t_end = min(t + (1.0 / sample_fps) + 0.1, duration)

        enable = f"between(t,{t},{t_end})"
        next_label = f"[v{i}]"

        filter_parts.append(
            f"{current_label}split[base{i}][crop{i}];"
            f"[crop{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"boxblur={blur_strength}:{blur_strength}[blurred{i}];"
            f"[base{i}][blurred{i}]overlay={rx}:{ry}:enable='{enable}'{next_label}"
        )
        current_label = next_label

    filter_str = ";".join(filter_parts)
    if filter_str:
        last_label = f"[v{len(detections) - 1}]"
        filter_str = filter_str[:filter_str.rfind(last_label)] + "[out]"
    else:
        filter_str = "[0:v]copy[out]"

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-i", input_path,
        "-filter_complex", filter_str,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    # Generate PII report
    report_path = os.path.splitext(output_path_str)[0] + "_pii_report.json"
    report = {
        "pii_redaction_report": {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_file": os.path.basename(input_path),
            "pii_types_found": pii_types_found,
            "total_detections": len(detections),
            "blur_strength": blur_strength,
            "detections": [
                {
                    "pii_type": d.get("pii_type", ""),
                    "timestamp": d.get("timestamp", 0),
                    "region": {"x": d.get("x", 0), "y": d.get("y", 0),
                               "w": d.get("w", 0), "h": d.get("h", 0)},
                    "matched_text": "***REDACTED***",
                }
                for d in detections
            ],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if on_progress:
        on_progress(100, "PII redaction complete")

    return {
        "output_path": output_path_str,
        "pii_found": len(detections),
        "pii_types": pii_types_found,
        "report_path": report_path,
    }
