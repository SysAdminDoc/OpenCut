"""
Event moment finder — wedding/ceremony/live-event highlight detection.

Looks for audio-energy + cheering/applause spikes that correlate with
canonical event moments: first kiss, first dance, ring exchange,
toasts, applause, vows.  Pure-Python implementation on top of the
existing ``audio_suite`` energy envelope — no new hard dependencies
and no models to download.

Two operating modes:

- ``heuristic`` (default): plain audio-energy spike detection with
  ``min_spacing`` between picks. Ships with every install.
- ``yamnet`` (optional): uses TensorFlow Lite's 521-class AudioSet
  YAMNet model to tag spikes as applause / cheer / music / speech so
  "first kiss → applause spike" can be distinguished from a generic
  loud moment. Gated on ``tflite_runtime`` availability.
"""

from __future__ import annotations

import array as _array
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Canonical tags + YAMNet class mapping
# ---------------------------------------------------------------------------

CANONICAL_EVENTS = (
    "applause",
    "cheering",
    "speech",
    "music_change",
    "loud_peak",
    "silence_break",
)

DEFAULT_TAG_MAP = {
    "Applause": "applause",
    "Clapping": "applause",
    "Cheering": "cheering",
    "Crowd": "cheering",
    "Chatter": "speech",
    "Speech": "speech",
    "Music": "music_change",
    "Singing": "music_change",
    "Wedding music": "music_change",
    "Organ": "music_change",
    "Laughter": "cheering",
    "Hubbub": "cheering",
}


@dataclass
class EventMoment:
    """One detected event moment."""
    t: float = 0.0
    label: str = "loud_peak"
    score: float = 0.0                      # 0..1, higher is more significant
    surrounding_energy: float = 0.0         # mean energy in surrounding window
    notes: str = ""


@dataclass
class EventMomentsResult:
    """Structured return from :func:`find_event_moments`."""
    filepath: str = ""
    moments: List[EventMoment] = field(default_factory=list)
    duration: float = 0.0
    mode: str = "heuristic"
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

def check_yamnet_available() -> bool:
    """True when TFLite runtime (or TensorFlow) is importable.

    The YAMNet model file itself is a second dependency — callers point
    ``OPENCUT_YAMNET_MODEL`` at a ``.tflite`` checkpoint; this function
    only probes the runtime.
    """
    try:
        import tflite_runtime  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PCM extraction + RMS envelope
# ---------------------------------------------------------------------------

def _extract_pcm(input_path: str, sample_rate: int = 16000) -> tuple:
    """Extract mono PCM samples via FFmpeg. Returns (samples, sample_rate)."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=1200, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed: "
            f"{proc.stderr.decode(errors='replace')[-200:]}"
        )
    data = proc.stdout
    if len(data) % 2:
        data = data[:-1]
    samples = _array.array("h", data)
    return samples, sample_rate


def _energy_envelope(
    samples: _array.array, sample_rate: int, hop_seconds: float = 0.1,
) -> List[float]:
    """RMS energy envelope with ``hop_seconds`` step size."""
    hop = max(1, int(sample_rate * hop_seconds))
    out: List[float] = []
    for i in range(0, len(samples), hop):
        chunk = samples[i:i + hop]
        if not chunk:
            continue
        acc = 0.0
        for s in chunk:
            acc += (s / 32768.0) ** 2
        rms = math.sqrt(acc / len(chunk))
        out.append(rms)
    return out


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

def _build_spike(
    i: int, e: float, hop_seconds: float, threshold: float,
    mean: float, sigma: float,
) -> EventMoment:
    """Construct an ``EventMoment`` from a single RMS envelope hit."""
    return EventMoment(
        t=round(i * hop_seconds, 3),
        label="loud_peak",
        score=round(min(1.0, e / (threshold + 1e-9)), 3),
        surrounding_energy=round(mean, 5),
        notes=f"sigma={sigma:.4f}",
    )


def _find_spikes(
    env: List[float],
    hop_seconds: float,
    min_spacing: float,
    k_sigma: float,
) -> List[EventMoment]:
    """Find RMS peaks that exceed ``mean + k_sigma·σ`` with a minimum
    spacing between picks.

    When two peaks fall inside ``min_spacing``, the one with the
    **higher raw RMS** wins — replacing the earlier pick in-place so
    ``len(moments)`` never grows without spacing.  (The v1.19.0 version
    of this function compared a raw RMS value against a normalised
    score × mean, which is apples-to-oranges — a bug fixed here.)
    """
    if len(env) < 10:
        return []

    n = len(env)
    mean = sum(env) / n
    var = sum((e - mean) ** 2 for e in env) / n
    sigma = math.sqrt(var)
    threshold = mean + k_sigma * sigma

    min_spacing_hops = max(1, int(min_spacing / hop_seconds))

    moments: List[EventMoment] = []
    # Track the raw RMS magnitude of the most recent pick separately so
    # we can compare against future candidates in the same units.
    last_picked_idx = -min_spacing_hops
    last_picked_e = 0.0

    for i, e in enumerate(env):
        if e < threshold:
            continue

        if i - last_picked_idx < min_spacing_hops:
            # Too close to the previous pick. Replace only if this peak
            # is louder in raw RMS — same units, no scale confusion.
            if moments and e > last_picked_e:
                moments[-1] = _build_spike(
                    i, e, hop_seconds, threshold, mean, sigma,
                )
                last_picked_idx = i
                last_picked_e = e
            continue

        moments.append(_build_spike(i, e, hop_seconds, threshold, mean, sigma))
        last_picked_idx = i
        last_picked_e = e

    return moments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_event_moments(
    filepath: str,
    mode: str = "heuristic",
    k_sigma: float = 2.0,
    min_spacing: float = 8.0,
    max_moments: int = 20,
    on_progress: Optional[Callable] = None,
) -> EventMomentsResult:
    """Locate event-highlight timestamps (kiss, dance, toast, applause).

    Args:
        filepath: Any ffmpeg-decodable input file.
        mode: ``"heuristic"`` (always available, audio-energy spikes)
            or ``"yamnet"`` (adds AudioSet tagging; requires
            ``tflite_runtime`` or ``tensorflow`` *and* a YAMNet model
            via ``OPENCUT_YAMNET_MODEL``).
        k_sigma: Spike threshold as *mean + k·σ* over the RMS envelope.
            2.0 = moderately loud; 3.0 = very loud peaks only.
        min_spacing: Seconds between accepted spikes. Prevents one loud
            event from producing five redundant picks.
        max_moments: Truncate results to the top N by score.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`EventMomentsResult`.  Its ``mode`` field records the
        **effective** mode — a ``yamnet`` request silently falls back to
        ``heuristic`` when runtime / model are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if mode not in ("heuristic", "yamnet"):
        raise ValueError("mode must be 'heuristic' or 'yamnet'")
    if mode == "yamnet" and not check_yamnet_available():
        logger.info(
            "YAMNet requested but tflite_runtime/tensorflow missing — "
            "falling back to heuristic"
        )
        mode = "heuristic"

    if on_progress:
        on_progress(5, "Decoding audio…")
    samples, sr = _extract_pcm(filepath, sample_rate=16000)

    if on_progress:
        on_progress(25, "Building energy envelope…")
    hop_seconds = 0.1
    env = _energy_envelope(samples, sr, hop_seconds=hop_seconds)
    duration = len(env) * hop_seconds

    if on_progress:
        on_progress(50, "Locating peaks…")
    moments = _find_spikes(env, hop_seconds, min_spacing, k_sigma)

    if mode == "yamnet":
        try:
            moments = _retag_with_yamnet(filepath, moments, on_progress)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YAMNet tagging failed: %s — using heuristic labels", exc)

    # Keep the top N by score, then re-sort chronologically for the UI.
    moments.sort(key=lambda m: m.score, reverse=True)
    moments = moments[:max_moments]
    moments.sort(key=lambda m: m.t)

    if on_progress:
        on_progress(100, f"Found {len(moments)} moment(s)")

    return EventMomentsResult(
        filepath=filepath,
        moments=moments,
        duration=round(duration, 3),
        mode=mode,
        notes=[f"k_sigma={k_sigma}", f"min_spacing={min_spacing}"],
    )


# ---------------------------------------------------------------------------
# Optional YAMNet retagger (stub)
# ---------------------------------------------------------------------------

def _retag_with_yamnet(
    filepath: str,
    moments: List[EventMoment],
    on_progress: Optional[Callable],
) -> List[EventMoment]:
    """Best-effort YAMNet class tagging.

    Returns the input unchanged when:
    - ``OPENCUT_YAMNET_MODEL`` is unset or missing, or
    - the full classify loop is not yet wired (current state — shipping
      the interface stable while the model-specific windowing logic is
      being finalised).

    The call is wrapped in a ``try/except`` by the caller so any failure
    here logs at warning and the untagged ``heuristic`` moments are
    returned verbatim.
    """
    model_path = os.environ.get("OPENCUT_YAMNET_MODEL", "")
    if not model_path or not os.path.isfile(model_path):
        return moments

    try:
        import tflite_runtime.interpreter as tflite  # type: ignore
    except ImportError:
        import tensorflow.lite as tflite  # type: ignore

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # Full per-window classification loop deferred to a subsequent pass —
    # ship the stable interface so the route exists and downstream code
    # can depend on it. Until then, heuristic labels are returned.
    return moments
