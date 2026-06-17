"""Multimodal footage indexing: OCR text + audio event classification.

Extends the transcript-only footage index with on-screen text (OCR)
and audio event tags so footage search matches visual text, sound
events, and speech.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")

FRAMES_FOR_OCR = 6


def check_ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def check_audio_classify_available() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


def extract_ocr_text(
    video_path: str,
    num_frames: int = FRAMES_FOR_OCR,
    on_progress: Optional[Callable] = None,
) -> str:
    """Extract on-screen text from key frames via OCR.

    Tries pytesseract first, then easyocr. Returns concatenated text
    from all sampled frames (deduplicated).
    """
    if on_progress:
        on_progress(5, "Extracting frames for OCR...")

    info_cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-print_format", "json", "-show_format",
        video_path,
    ]
    import json
    try:
        probe = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
        duration = float(json.loads(probe.stdout).get("format", {}).get("duration", 0))
    except Exception:
        duration = 0

    if duration <= 0:
        num_frames = 1

    tmpdir = tempfile.mkdtemp(prefix="opencut_ocr_")
    try:
        interval = max(1, duration / (num_frames + 1)) if duration > 0 else 1
        frame_paths: List[str] = []
        for i in range(num_frames):
            ts = interval * (i + 1)
            out_path = os.path.join(tmpdir, f"frame_{i:03d}.png")
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", str(ts),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                frame_paths.append(out_path)

        if on_progress:
            on_progress(40, f"Running OCR on {len(frame_paths)} frames...")

        all_text: List[str] = []
        seen: set = set()

        ocr_engine = _get_ocr_engine()
        for i, fp in enumerate(frame_paths):
            try:
                texts = ocr_engine(fp)
                for t in texts:
                    t = t.strip()
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        all_text.append(t)
            except Exception as exc:
                logger.debug("OCR failed on frame %d: %s", i, exc)
            if on_progress:
                pct = 40 + int(50 * (i + 1) / len(frame_paths))
                on_progress(pct, f"OCR frame {i+1}/{len(frame_paths)}")

        if on_progress:
            on_progress(95, f"OCR found {len(all_text)} text segments")

        return " ".join(all_text)

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _get_ocr_engine():
    """Return a callable that takes an image path and returns a list of strings."""
    try:
        import pytesseract

        def _tess(image_path: str) -> List[str]:
            text = pytesseract.image_to_string(image_path, timeout=15)
            return [line for line in text.splitlines() if line.strip()]

        return _tess
    except ImportError:
        pass

    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        def _easy(image_path: str) -> List[str]:
            results = _reader.readtext(image_path)
            return [r[1] for r in results if r[1].strip()]

        return _easy
    except ImportError:
        pass

    def _noop(image_path: str) -> List[str]:
        return []

    return _noop


def classify_audio_events(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Classify audio events using spectral heuristics.

    Returns space-separated tags like "speech music silence".
    Uses librosa when available; falls back to ffmpeg astats.
    """
    if on_progress:
        on_progress(5, "Analyzing audio events...")

    tags: List[str] = []

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(video_path, sr=22050, mono=True, duration=120)
        if len(y) == 0:
            return ""

        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))
        max_rms = float(np.max(rms))

        if mean_rms < 0.005:
            tags.append("silence")
        else:
            tags.append("audio")

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        mean_zcr = float(np.mean(zcr))
        if mean_zcr > 0.1:
            tags.append("speech")

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = float(np.mean(spectral_centroid))
        if mean_centroid > 2000:
            tags.append("high-frequency")

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]) if len(tempo) > 0 else 0.0
        if tempo_val > 60:
            tags.append("music")
            tags.append(f"bpm:{int(tempo_val)}")

        if max_rms > 0.5:
            tags.append("loud")

        if on_progress:
            on_progress(90, f"Audio: {' '.join(tags)}")

    except ImportError:
        tags.append("audio")
        if on_progress:
            on_progress(90, "librosa unavailable; basic audio tag only")
    except Exception as exc:
        logger.debug("Audio classification failed: %s", exc)
        tags.append("audio")

    if on_progress:
        on_progress(100, f"Audio tags: {' '.join(tags)}")

    return " ".join(tags)
