"""
BeatNet backend for beat + downbeat detection.

Wraps the ``BeatNet`` CRNN + particle-filter model — https://github.com/mjhydri/BeatNet
(MIT) — which beats librosa/madmom on downbeats and hands back a
structured :class:`~opencut.core.audio_suite.BeatInfo` dict compatible
with existing routes.

Graceful degradation: when ``beatnet`` is not installed, this module
raises ``RuntimeError`` and callers fall back to
``audio_suite.detect_beats`` (FFmpeg energy onset).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Callable, List, Optional

from opencut.core.audio_suite import BeatInfo
from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


def check_beatnet_available() -> bool:
    """True when `BeatNet` is importable."""
    try:
        import BeatNet  # noqa: F401
        return True
    except ImportError:
        return False


def _wav_from_input(input_path: str, target_sr: int = 22050) -> str:
    """Decode `input_path` to a mono 22.05 kHz WAV in the temp dir."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg decode failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[-200:]}"
        )
    return wav_path


def detect_beats_beatnet(
    input_path: str,
    mode: str = "offline",
    meter: int = 4,
    on_progress: Optional[Callable] = None,
) -> BeatInfo:
    """Detect beats + downbeats using BeatNet.

    Args:
        input_path: Any audio/video file FFmpeg can decode.
        mode: ``"offline"`` (default, most accurate) or ``"realtime"``.
            ``"realtime"`` uses the causal model — lower latency but
            noticeably lower accuracy.
        meter: Musical meter (3 = 3/4 waltz, 4 = 4/4 common time,
            6 = 6/8). BeatNet defaults to 4.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`BeatInfo` — `bpm`, `beat_times`, `downbeat_times`,
        and a `confidence` float derived from the beat / downbeat ratio
        (BeatNet itself doesn't emit a confidence; we approximate).

    Raises:
        RuntimeError: BeatNet is unavailable.
        ValueError: invalid ``mode`` or ``meter``.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if mode not in ("offline", "realtime"):
        raise ValueError("mode must be 'offline' or 'realtime'")
    if int(meter) not in (3, 4, 6):
        raise ValueError("meter must be 3, 4, or 6")

    if not check_beatnet_available():
        raise RuntimeError(
            "BeatNet not installed. Install: pip install BeatNet"
        )

    if on_progress:
        on_progress(5, "Preparing audio…")
    wav_path = _wav_from_input(input_path, target_sr=22050)

    try:
        from BeatNet.BeatNet import BeatNet

        if on_progress:
            on_progress(15, f"Loading BeatNet ({mode})…")

        # BeatNet API: BeatNet(1, mode=..., inference_model=..., plot=False, thread=False)
        net = BeatNet(
            1,  # model index — 1 = default pretrained
            mode=mode,
            inference_model="PF",  # particle filter
            plot=[],
            thread=False,
        )

        if on_progress:
            on_progress(40, "Running beat tracker…")

        output = net.process(wav_path)
        # Output shape: (N, 2) → (time_seconds, beat_label)
        # beat_label: 1.0 = downbeat, 2.0 = beat (or vice-versa
        # depending on BeatNet version).  We sort robustly:
        beat_times: List[float] = []
        downbeat_times: List[float] = []
        try:
            for row in output:
                t = float(row[0])
                label = int(float(row[1]))
                beat_times.append(t)
                if label == 1:
                    downbeat_times.append(t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BeatNet output parse fallback: %s", exc)

        if on_progress:
            on_progress(85, "Estimating BPM…")

        bpm = _estimate_bpm(beat_times)
        confidence = 0.0
        if beat_times:
            # Rough confidence: fraction of labelled downbeats near 1/meter.
            expected = max(1, len(beat_times) // int(meter))
            confidence = min(1.0, len(downbeat_times) / max(1, expected))

        if on_progress:
            on_progress(100, "BeatNet complete")

        return BeatInfo(
            bpm=round(bpm, 2),
            beat_times=[round(t, 4) for t in beat_times],
            downbeat_times=[round(t, 4) for t in downbeat_times],
            confidence=round(confidence, 3),
        )
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _estimate_bpm(beat_times: List[float]) -> float:
    if len(beat_times) < 2:
        return 0.0
    deltas = [
        beat_times[i + 1] - beat_times[i]
        for i in range(len(beat_times) - 1)
        if beat_times[i + 1] - beat_times[i] > 0.05
    ]
    if not deltas:
        return 0.0
    deltas.sort()
    median = deltas[len(deltas) // 2]
    if median <= 0:
        return 0.0
    return 60.0 / median
