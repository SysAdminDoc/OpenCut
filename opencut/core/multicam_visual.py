"""
Multicam Visual Speech Cues

Analyzes video frames for lip movement and shot type to improve multicam
cut decisions beyond audio-only diarization. Resolves cross-mic bleed
by checking which on-camera face is actually speaking.

Uses MediaPipe Face Mesh for lip movement scoring and face-to-frame ratio
for shot type classification (wide / medium / close).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

MOUTH_OUTER_TOP = 13
MOUTH_OUTER_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


@dataclass
class VisualCue:
    """Visual analysis for a single sampled frame."""
    time: float = 0.0
    faces: int = 0
    lip_scores: Dict[int, float] = field(default_factory=dict)
    shot_types: Dict[int, str] = field(default_factory=dict)
    face_bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    active_face: int = -1


@dataclass
class VisualAnalysisResult:
    """Result of visual speech cue analysis."""
    cues: List[VisualCue] = field(default_factory=list)
    total_frames_sampled: int = 0
    faces_detected: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("cues", "total_frames_sampled", "faces_detected", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_visual_multicam_available() -> bool:
    """MediaPipe + OpenCV required for visual multicam cues."""
    return (_try_import("mediapipe") is not None
            and _try_import("cv2") is not None)


def _classify_shot(face_h: int, frame_h: int) -> str:
    """Classify shot type by face-to-frame height ratio."""
    if frame_h <= 0:
        return "unknown"
    ratio = face_h / frame_h
    if ratio > 0.45:
        return "close"
    if ratio > 0.20:
        return "medium"
    return "wide"


def _lip_openness(landmarks, frame_h: int) -> float:
    """Compute mouth openness as a 0-1 ratio from face mesh landmarks."""
    top = landmarks[MOUTH_OUTER_TOP]
    bottom = landmarks[MOUTH_OUTER_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    mouth_height = abs(bottom.y - top.y) * frame_h
    mouth_width = abs(right.x - left.x) * frame_h
    if mouth_width < 1.0:
        return 0.0
    return min(1.0, mouth_height / mouth_width)


def analyze_frames(
    video_path: str,
    sample_times: List[float],
    on_progress: Optional[Callable] = None,
) -> VisualAnalysisResult:
    """Sample video frames and extract lip movement + shot type cues.

    Args:
        video_path: Path to video file.
        sample_times: List of timestamps (seconds) to sample.
        on_progress: Optional callback(pct, msg).

    Returns:
        VisualAnalysisResult with per-frame cues.
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not sample_times:
        return VisualAnalysisResult(notes=["No sample times provided"])

    if not check_visual_multicam_available():
        raise RuntimeError(
            "Visual multicam requires mediapipe and opencv-python. "
            "Install with: pip install mediapipe opencv-python-headless"
        )

    import cv2
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=8,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cues: List[VisualCue] = []
    total_faces = 0

    try:
        for i, t in enumerate(sample_times):
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                cues.append(VisualCue(time=t))
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            cue = VisualCue(time=t)

            if results.multi_face_landmarks:
                cue.faces = len(results.multi_face_landmarks)
                total_faces += cue.faces

                best_lip = -1.0
                best_face_id = -1

                for face_id, face_lm in enumerate(results.multi_face_landmarks):
                    lm = face_lm.landmark
                    lip = _lip_openness(lm, h)
                    cue.lip_scores[face_id] = round(lip, 4)

                    ys = [pt.y * h for pt in lm]
                    xs = [pt.x * w for pt in lm]
                    face_top = int(min(ys))
                    face_bottom = int(max(ys))
                    face_left = int(min(xs))
                    face_right = int(max(xs))
                    face_h_px = face_bottom - face_top
                    face_w_px = face_right - face_left

                    cue.face_bboxes[face_id] = (face_left, face_top, face_w_px, face_h_px)
                    cue.shot_types[face_id] = _classify_shot(face_h_px, h)

                    if lip > best_lip:
                        best_lip = lip
                        best_face_id = face_id

                if best_lip > 0.05:
                    cue.active_face = best_face_id

            cues.append(cue)

            if on_progress and len(sample_times) > 1:
                pct = int(10 + 80 * (i + 1) / len(sample_times))
                on_progress(pct, f"Analyzed frame {i + 1}/{len(sample_times)}")
    finally:
        cap.release()
        face_mesh.close()

    if on_progress:
        on_progress(100, "Visual analysis complete")

    return VisualAnalysisResult(
        cues=cues,
        total_frames_sampled=len(cues),
        faces_detected=total_faces,
        notes=[f"Sampled {len(cues)} frames, detected {total_faces} total face instances"],
    )


def refine_cuts_with_visual(
    cuts: List[dict],
    cues: List[VisualCue],
    speaker_face_map: Optional[Dict[str, int]] = None,
    lip_threshold: float = 0.05,
) -> List[dict]:
    """Refine audio-only multicam cuts using visual speech cues.

    For each cut, checks whether the assigned speaker's face is actively
    speaking (lip movement above threshold). If another face is speaking
    more actively, reassigns the cut to prefer the visually active speaker.

    Args:
        cuts: Audio-only cut list from generate_multicam_cuts().
        cues: Visual cues from analyze_frames(), one per cut timestamp.
        speaker_face_map: Optional mapping of speaker label → face_id.
            If None, uses order-based heuristic (SPEAKER_00 → face 0).
        lip_threshold: Minimum lip openness to consider a face "speaking".

    Returns:
        Refined cut list with visual_confidence and shot_type annotations.
    """
    if not cues or not cuts:
        return cuts

    cue_by_time = {}
    for cue in cues:
        cue_by_time[round(cue.time, 2)] = cue

    refined = []
    visual_overrides = 0

    for cut in cuts:
        new_cut = dict(cut)
        t = round(cut.get("time", 0.0), 2)
        cue = cue_by_time.get(t)

        if not cue or cue.faces == 0:
            new_cut["visual_confidence"] = 0.0
            new_cut["shot_type"] = "unknown"
            refined.append(new_cut)
            continue

        speaker = cut.get("speaker", "")
        assigned_face = None
        if speaker_face_map:
            assigned_face = speaker_face_map.get(speaker)
        if assigned_face is None:
            idx = 0
            try:
                idx = int(speaker.split("_")[-1]) if "_" in speaker else 0
            except (ValueError, IndexError):
                pass
            assigned_face = idx

        assigned_lip = cue.lip_scores.get(assigned_face, 0.0)
        new_cut["shot_type"] = cue.shot_types.get(assigned_face, "unknown")

        if assigned_lip >= lip_threshold:
            new_cut["visual_confidence"] = round(assigned_lip, 3)
        elif cue.active_face >= 0 and cue.active_face != assigned_face:
            active_lip = cue.lip_scores.get(cue.active_face, 0.0)
            if active_lip > assigned_lip + 0.03:
                if speaker_face_map:
                    face_to_speaker = {v: k for k, v in speaker_face_map.items()}
                    active_speaker = face_to_speaker.get(cue.active_face)
                else:
                    active_speaker = f"SPEAKER_{cue.active_face:02d}"

                if active_speaker:
                    new_cut["speaker"] = active_speaker
                    new_cut["visual_override"] = True
                    new_cut["shot_type"] = cue.shot_types.get(cue.active_face, "unknown")
                    visual_overrides += 1

            new_cut["visual_confidence"] = round(max(assigned_lip, active_lip), 3)
        else:
            new_cut["visual_confidence"] = round(assigned_lip, 3)

        refined.append(new_cut)

    if visual_overrides:
        logger.info(
            "Visual cues overrode %d/%d cuts (cross-mic bleed correction)",
            visual_overrides, len(cuts),
        )

    return refined


__all__ = [
    "VisualCue",
    "VisualAnalysisResult",
    "check_visual_multicam_available",
    "analyze_frames",
    "refine_cuts_with_visual",
]
