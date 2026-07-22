"""
OpenCut Deepfake Detection Module v0.9.0

Analyse video for AI-generated manipulation artifacts:
- Face consistency analysis (temporal jitter, blending edges)
- Frequency-domain artifact detection (GAN fingerprints)
- Per-segment confidence scoring
- Authenticity report generation

Requires: pip install opencv-python numpy
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import ensure_package, get_video_info

logger = logging.getLogger("opencut")

DETECTOR_VERSION = "1.0"
ANALYSIS_METHODS = [
    "face_temporal_consistency",
    "face_boundary_blending",
    "frequency_artifact_scan",
]
SIGNAL_WEIGHTS = {
    "face_inconsistency": 0.5,
    "blending": 0.25,
    "frequency": 0.25,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class DeepfakeResult:
    """Detection result for a video segment."""
    start_time: float
    end_time: float
    confidence: float  # 0-1, higher = more likely manipulated
    face_consistency: float = 0.0
    frequency_score: float = 0.0
    blending_score: float = 0.0
    faces_detected: int = 0
    evidence: List[str] = field(default_factory=list)
    label: str = "unknown"  # "authentic", "suspicious", "likely_fake"


# ---------------------------------------------------------------------------
# Face Consistency Analysis
# ---------------------------------------------------------------------------
def analyze_face_consistency(
    video_path: str,
    segment_duration: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """
    Analyse temporal face consistency across video segments.

    Detects face regions and measures:
    - Landmark jitter (unnatural micro-movements)
    - Face boundary sharpness (blending artifacts)
    - Skin texture consistency (GAN noise patterns)

    Args:
        video_path: Input video path.
        segment_duration: Analysis window in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of per-segment analysis dicts with consistency metrics.
    """
    if not ensure_package("cv2", "opencv-python"):
        raise RuntimeError("Failed to install opencv-python")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    duration = info.get("duration", 0.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if on_progress:
        on_progress(5, "Loading face detector...")

    # Use Haar cascade for face detection (always available in OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = max(1, int(duration / segment_duration)) if duration > 0 else 1
    sample_step = max(1, int(fps / 3))  # ~3 fps sample rate

    segments = []
    for seg_idx in range(num_segments):
        segments.append({
            "start_time": seg_idx * segment_duration,
            "end_time": min((seg_idx + 1) * segment_duration, duration),
            "face_sizes": [],
            "face_positions": [],
            "edge_sharpness": [],
            "texture_scores": [],
        })

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                t = frame_idx / fps
                seg_idx = min(int(t / segment_duration), num_segments - 1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                )

                for (fx, fy, fw, fh) in faces:
                    segments[seg_idx]["face_sizes"].append(fw * fh)
                    segments[seg_idx]["face_positions"].append((fx + fw // 2, fy + fh // 2))

                    # Edge sharpness around face boundary
                    face_roi = gray[fy:fy + fh, fx:fx + fw]
                    if face_roi.size > 0:
                        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
                        edge_var = float(laplacian.var())
                        segments[seg_idx]["edge_sharpness"].append(edge_var)

                        # Texture analysis: high-frequency content via DCT
                        resized = cv2.resize(face_roi, (64, 64)).astype(np.float32)
                        dct = cv2.dct(resized)
                        # High-frequency energy ratio
                        hf_energy = float(np.sum(np.abs(dct[32:, 32:])))
                        total_energy = float(np.sum(np.abs(dct))) + 1e-8
                        segments[seg_idx]["texture_scores"].append(hf_energy / total_energy)

            frame_idx += 1

            if on_progress and frame_idx % (sample_step * 30) == 0:
                pct = 5 + int((frame_idx / max(1, total_frames)) * 85)
                on_progress(min(pct, 92), f"Analysing frame {frame_idx}/{total_frames}...")
    finally:
        cap.release()

    # Compute per-segment consistency scores
    results = []
    for seg in segments:
        consistency = 1.0  # Start at "consistent" (authentic)

        # Face size variance (deepfakes often have unstable face scaling)
        if len(seg["face_sizes"]) > 1:
            size_std = float(np.std(seg["face_sizes"]))
            size_mean = float(np.mean(seg["face_sizes"])) + 1e-8
            size_cv = size_std / size_mean
            if size_cv > 0.3:
                consistency -= min(0.3, size_cv - 0.3)

        # Position jitter
        if len(seg["face_positions"]) > 1:
            pos = np.array(seg["face_positions"], dtype=np.float32)
            pos_std = float(np.std(pos, axis=0).mean())
            if pos_std > 15:
                consistency -= min(0.2, (pos_std - 15) / 100)

        # Edge sharpness anomaly (deepfakes often have unnaturally smooth edges)
        edge_score = 0.0
        if seg["edge_sharpness"]:
            mean_edge = float(np.mean(seg["edge_sharpness"]))
            if mean_edge < 50:  # Suspiciously smooth
                edge_score = 0.3
                consistency -= 0.2

        # Texture regularity (GAN artifacts in high frequencies)
        texture_score = 0.0
        if seg["texture_scores"]:
            mean_tex = float(np.mean(seg["texture_scores"]))
            if mean_tex > 0.15:  # Abnormal high-frequency content
                texture_score = min(1.0, mean_tex / 0.3)
                consistency -= min(0.3, texture_score * 0.3)

        consistency = max(0.0, consistency)

        results.append({
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "face_consistency": round(consistency, 4),
            "edge_sharpness": round(edge_score, 4),
            "texture_anomaly": round(texture_score, 4),
            "faces_detected": len(seg["face_sizes"]),
        })

    if on_progress:
        on_progress(100, "Face consistency analysis complete!")

    return results


# ---------------------------------------------------------------------------
# Full Deepfake Detection
# ---------------------------------------------------------------------------
def detect_deepfake(
    video_path: str,
    segment_duration: float = 5.0,
    threshold: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Run full deepfake detection analysis on a video.

    Combines face consistency, frequency analysis, and blending
    detection to produce per-segment confidence scores.

    Args:
        video_path: Input video path.
        segment_duration: Analysis segment duration in seconds.
        threshold: Confidence threshold for "suspicious" label.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with results (list of DeepfakeResult dicts),
        overall_score, verdict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    face_results = analyze_face_consistency(
        video_path,
        segment_duration=segment_duration,
        on_progress=on_progress,
    )

    deepfake_results = []
    for seg in face_results:
        # Invert face_consistency (1.0 = authentic) to manipulation confidence
        manipulation_conf = 1.0 - seg["face_consistency"]

        # Combine signals
        combined = (
            SIGNAL_WEIGHTS["face_inconsistency"] * manipulation_conf
            + SIGNAL_WEIGHTS["blending"] * seg["edge_sharpness"]
            + SIGNAL_WEIGHTS["frequency"] * seg["texture_anomaly"]
        )
        combined = min(1.0, combined)

        if combined >= 0.7:
            label = "likely_fake"
        elif combined >= threshold:
            label = "suspicious"
        else:
            label = "authentic"

        evidence = _build_evidence(seg, manipulation_conf)
        deepfake_results.append(DeepfakeResult(
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            confidence=round(combined, 4),
            face_consistency=round(seg["face_consistency"], 4),
            frequency_score=round(seg["texture_anomaly"], 4),
            blending_score=round(seg["edge_sharpness"], 4),
            faces_detected=int(seg.get("faces_detected", 0)),
            evidence=evidence,
            label=label,
        ))

    # Overall verdict
    if deepfake_results:
        max_conf = max(r.confidence for r in deepfake_results)
        avg_conf = sum(r.confidence for r in deepfake_results) / len(deepfake_results)
    else:
        max_conf = 0.0
        avg_conf = 0.0

    if max_conf >= 0.7:
        verdict = "likely_fake"
    elif max_conf >= threshold:
        verdict = "suspicious"
    else:
        verdict = "likely_authentic"

    flagged_count = sum(
        1 for r in deepfake_results if r.label in ("suspicious", "likely_fake")
    )
    return {
        "detector_version": DETECTOR_VERSION,
        "analysis_methods": ANALYSIS_METHODS,
        "threshold": threshold,
        "segment_duration": segment_duration,
        "results": [asdict(r) for r in deepfake_results],
        "flagged_segments": flagged_count,
        "overall_score": round(avg_conf, 4),
        "max_score": round(max_conf, 4),
        "verdict": verdict,
        "review_recommendation": _review_recommendation(verdict, flagged_count),
    }


def _build_evidence(seg: dict, manipulation_conf: float) -> List[str]:
    """Build stable evidence tags from segment-level detector signals."""
    evidence = []
    if manipulation_conf >= 0.3:
        evidence.append("face_temporal_inconsistency")
    if seg.get("edge_sharpness", 0.0) >= 0.3:
        evidence.append("face_boundary_blending")
    if seg.get("texture_anomaly", 0.0) >= 0.3:
        evidence.append("frequency_artifact_pattern")
    if int(seg.get("faces_detected", 0)) == 0:
        evidence.append("no_face_detected")
    return evidence


def _review_recommendation(verdict: str, flagged_segments: int) -> str:
    """Return concise guidance for editorial review workflows."""
    if verdict == "likely_fake":
        return "Block publication until source authenticity is manually verified."
    if verdict == "suspicious" or flagged_segments:
        return "Review flagged segments frame-by-frame before publishing."
    return "No deepfake-specific artifacts exceeded the configured threshold."


# ---------------------------------------------------------------------------
# Authenticity Report
# ---------------------------------------------------------------------------
def generate_authenticity_report(
    results: dict,
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a JSON authenticity report from detection results.

    Args:
        results: Output from detect_deepfake().
        output_path: Path for the report file.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the generated report file.
    """
    if not results:
        raise ValueError("No results to generate report from")

    if on_progress:
        on_progress(10, "Generating authenticity report...")

    report = {
        "version": "1.0",
        "detector_version": results.get("detector_version", DETECTOR_VERSION),
        "analysis_methods": results.get("analysis_methods", ANALYSIS_METHODS),
        "verdict": results.get("verdict", "unknown"),
        "overall_score": results.get("overall_score", 0.0),
        "max_score": results.get("max_score", 0.0),
        "flagged_segments": results.get("flagged_segments", 0),
        "review_recommendation": results.get(
            "review_recommendation",
            _review_recommendation(results.get("verdict", "unknown"), 0),
        ),
        "segments": results.get("results", []),
        "summary": _generate_summary(results),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if on_progress:
        on_progress(100, "Report generated!")

    return output_path


def _generate_summary(results: dict) -> str:
    """Generate a human-readable summary from detection results."""
    verdict = results.get("verdict", "unknown")
    overall = results.get("overall_score", 0)
    segments = results.get("results", [])
    suspicious = sum(1 for s in segments if s.get("label") in ("suspicious", "likely_fake"))

    if verdict == "likely_authentic":
        return (
            f"Analysis indicates the video is likely authentic. "
            f"Average manipulation confidence: {overall:.1%}. "
            f"{suspicious} of {len(segments)} segments flagged."
        )
    elif verdict == "suspicious":
        return (
            f"Analysis found potentially suspicious artifacts. "
            f"Average manipulation confidence: {overall:.1%}. "
            f"{suspicious} of {len(segments)} segments flagged. "
            f"Manual review recommended."
        )
    else:
        return (
            f"Analysis indicates likely AI manipulation. "
            f"Average manipulation confidence: {overall:.1%}. "
            f"{suspicious} of {len(segments)} segments flagged. "
            f"Significant artifacts detected in face regions."
        )
