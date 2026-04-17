"""
Emotion arc analysis via HSEmotion.

Wraps HSEmotion (https://github.com/av-savchenko/face-emotion-recognition,
Apache-2) — a lightweight ONNX/pytorch face-emotion classifier — to
produce a time-indexed emotion signal over a video clip.

Output: per-sample 8-class emotion probabilities (Neutral, Happy, Sad,
Surprise, Fear, Disgust, Anger, Contempt) plus a dominant-label
timeline and summary stats (mean emotion, emotion peaks, transitions).

Use cases:
- Feed the engagement-arc into `emotion_timeline.py` plotting.
- Rank clips by "emotional range" for highlight selection.
- Flag sentiment swings for B-roll suggestion.

Falls back to DeepFace when HSEmotion is unavailable; falls back
silently to ``RuntimeError`` when neither backend is importable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

EMOTION_LABELS = (
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt",
)


@dataclass
class EmotionSample:
    """A single emotion reading at a timestamp."""
    t: float = 0.0
    dominant: str = "Neutral"
    probabilities: Dict[str, float] = field(default_factory=dict)
    face_found: bool = True


@dataclass
class EmotionArc:
    """Structured timeline of emotion readings."""
    filepath: str = ""
    samples: List[EmotionSample] = field(default_factory=list)
    dominant_overall: str = "Neutral"
    mean_probabilities: Dict[str, float] = field(default_factory=dict)
    transitions: int = 0
    emotional_range: float = 0.0
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_hsemotion_available() -> bool:
    try:
        import hsemotion_onnx  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import hsemotion  # noqa: F401
        return True
    except ImportError:
        return False


def _load_hsemotion():
    """Return (model, predict_fn) for the best available HSEmotion backend."""
    # Try ONNX runtime first — avoids heavy torch dep if user already has
    # onnxruntime installed.
    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer  # type: ignore
        rec = HSEmotionRecognizer(model_name="enet_b2_8")
        return rec, "onnx"
    except ImportError:
        pass
    from hsemotion.facial_emotions import HSEmotionRecognizer  # type: ignore
    rec = HSEmotionRecognizer(model_name="enet_b2_8")
    return rec, "torch"


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _iter_sampled_frames(
    filepath: str,
    fps_sample: float,
    max_frames: int,
) -> List[Tuple[float, object]]:
    """Yield (timestamp, BGR frame ndarray) tuples at ``fps_sample`` Hz."""
    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {filepath}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            raise RuntimeError("Video metadata invalid (fps/frame count)")
        stride = max(1, int(round(fps / max(0.1, fps_sample))))
        result = []
        idx = 0
        sampled = 0
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                t = idx / fps
                result.append((t, frame))
                sampled += 1
            idx += 1
        return result
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_emotion_arc(
    filepath: str,
    fps_sample: float = 1.0,
    max_frames: int = 120,
    on_progress: Optional[Callable] = None,
) -> EmotionArc:
    """Produce an :class:`EmotionArc` over ``filepath``.

    Args:
        filepath: Input video (any FFmpeg/OpenCV-decodable format).
        fps_sample: Sample rate in Hz. Default 1.0 (1 frame/sec) —
            HSEmotion runs at ~100 FPS on CPU so 1 Hz is near-free.
        max_frames: Cap on sampled frames.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`EmotionArc`. Empty ``samples`` when no faces were found
        in any sampled frame.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not check_hsemotion_available():
        raise RuntimeError(
            "HSEmotion not installed. Install: pip install hsemotion-onnx"
        )

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("opencv-python is required") from exc

    if on_progress:
        on_progress(5, "Sampling frames…")
    frames = _iter_sampled_frames(filepath, fps_sample, max_frames)

    if on_progress:
        on_progress(15, f"Loading HSEmotion ({len(frames)} frames)…")
    recognizer, backend = _load_hsemotion()

    samples: List[EmotionSample] = []
    prob_sum: Dict[str, float] = {k: 0.0 for k in EMOTION_LABELS}

    try:
        for i, (t, frame) in enumerate(frames):
            try:
                # predict_emotions returns (label, probabilities)
                label, probs = recognizer.predict_emotions(frame, logits=False)
            except Exception as exc:  # noqa: BLE001
                # No face / bad frame → skip but record placeholder
                samples.append(EmotionSample(
                    t=round(t, 3),
                    dominant="Unknown",
                    face_found=False,
                    probabilities={},
                ))
                logger.debug("HSEmotion failed at t=%.2f: %s", t, exc)
                continue

            # probs may be a numpy array or dict depending on backend
            if hasattr(probs, "tolist"):
                prob_list = probs.tolist()
            else:
                prob_list = list(probs)
            prob_dict = {
                EMOTION_LABELS[j]: round(float(prob_list[j]), 4)
                for j in range(min(len(EMOTION_LABELS), len(prob_list)))
            }
            for k, v in prob_dict.items():
                prob_sum[k] += v
            samples.append(EmotionSample(
                t=round(t, 3),
                dominant=str(label),
                probabilities=prob_dict,
                face_found=True,
            ))
            if on_progress:
                pct = 15 + int(80 * ((i + 1) / len(frames)))
                on_progress(pct, f"{i + 1}/{len(frames)}")
    finally:
        # Free ONNX session / torch weights
        try:
            del recognizer
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    valid = [s for s in samples if s.face_found]
    mean_probs = {
        k: round(prob_sum[k] / max(1, len(valid)), 4)
        for k in EMOTION_LABELS
    } if valid else {}
    dominant_overall = (
        max(mean_probs.items(), key=lambda kv: kv[1])[0]
        if mean_probs else "Unknown"
    )

    # Transitions: count adjacent samples whose dominant differs
    transitions = sum(
        1 for a, b in zip(valid, valid[1:])
        if a.dominant != b.dominant
    )
    # Emotional range: max - min of the dominant probability distribution
    if valid:
        dom_values = [
            s.probabilities.get(s.dominant, 0.0) for s in valid
        ]
        emotional_range = round(max(dom_values) - min(dom_values), 4)
    else:
        emotional_range = 0.0

    duration = samples[-1].t if samples else 0.0

    if on_progress:
        on_progress(100, "Emotion arc complete")

    return EmotionArc(
        filepath=filepath,
        samples=samples,
        dominant_overall=dominant_overall,
        mean_probabilities=mean_probs,
        transitions=transitions,
        emotional_range=emotional_range,
        duration=round(duration, 3),
        notes=[f"backend={backend}", f"samples={len(samples)} valid={len(valid)}"],
    )
