"""
OpenCut Virality Score (Wave H1.1, v1.25.0)

Multimodal 0-100 scoring to rank short-form clip candidates *before*
handing them to the shorts pipeline. Blends three heuristic signals:

1. **Audio-energy peaks** — via the existing ``silence.py`` infrastructure.
   Clips with wide dynamic range + moments of loudness score higher than
   flat drone clips.
2. **Transcript sentiment / hook density** — if ``core/llm.py`` reports
   a reachable provider it is asked for a 0-100 "hook strength". Falls
   back to a keyword lexicon (curiosity-gap words, power words, numerics,
   questions, imperatives) so the score is always computable.
3. **Visual salience** — mean optical-flow magnitude via FFmpeg's
   ``-vf select=... ,signalstats`` or a lightweight cv2 frame-diff.
   Opt-in — the route skips this step if cv2/ffmpeg can't profile the
   clip in under ``max_probe_seconds``.

The final score is a weighted blend clamped to 0-100. Design principles
match Wave A–G:
- Subscriptable dataclass result (``ViralityResult``) for Flask jsonify.
- Single entry-point ``score(filepath, segments=None, llm_config=None,
  on_progress=None)`` with ``on_progress(pct, msg="")`` default.
- ``rank(candidates, ...)`` returns a sorted list of dicts, stable on
  ties, deterministic across runs for the same input.
- Heuristic by design — the absolute number is **not** comparable
  across video types. Use for relative ranking only.

No new required pip deps.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Hook lexicon — curated from viral-video research
# ---------------------------------------------------------------------------

_HOOK_WORDS = {
    # curiosity gap
    "why", "how", "what", "secret", "truth", "nobody", "hidden", "revealed",
    "actually", "really", "finally", "turns out", "here's",
    # power words
    "never", "always", "best", "worst", "ultimate", "instantly", "proven",
    "shocking", "mistake", "warning", "stop", "wait",
    # imperative
    "watch", "listen", "look", "imagine", "think", "believe", "remember",
    # numeric cues
    "one", "two", "three", "five", "ten", "first", "last",
}

_NUMERIC_RE = re.compile(r"\b\d+\b")
_QUESTION_RE = re.compile(r"\?")
_EMPHASIS_RE = re.compile(r"[!]{1,}|[A-Z]{4,}")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ViralitySignals:
    audio_energy: float = 0.0      # 0-100
    transcript_hook: float = 0.0   # 0-100
    visual_salience: float = 0.0   # 0-100

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)

    def keys(self):
        return ("audio_energy", "transcript_hook", "visual_salience")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


@dataclass
class ViralityResult:
    score: float = 0.0               # 0-100 composite
    signals: ViralitySignals = field(default_factory=ViralitySignals)
    weights: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    duration: float = 0.0
    hook_phrase: str = ""            # best detected hook line, if any

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return ("score", "signals", "weights", "notes",
                "duration", "hook_phrase")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_virality_score_available() -> bool:
    """Virality scoring is stdlib + FFmpeg; always True when FFmpeg present."""
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "audio_energy": 0.30,
    "transcript_hook": 0.45,
    "visual_salience": 0.25,
}


def _normalise_weights(w: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Normalise user-supplied weights to sum to 1.0, falling back to defaults."""
    if not w:
        return dict(_DEFAULT_WEIGHTS)
    cleaned = {}
    for key, default in _DEFAULT_WEIGHTS.items():
        try:
            v = float(w.get(key, default))
            if not math.isfinite(v) or v < 0:
                v = default
        except (TypeError, ValueError):
            v = default
        cleaned[key] = v
    total = sum(cleaned.values()) or 1.0
    return {k: v / total for k, v in cleaned.items()}


# ---------------------------------------------------------------------------
# Audio energy probe
# ---------------------------------------------------------------------------

def _audio_energy_score(filepath: str, max_probe_seconds: float = 180.0) -> Tuple[float, List[str]]:
    """Probe peak-to-RMS ratio + loudness range via FFmpeg astats.

    Returns (score 0-100, notes).
    """
    notes: List[str] = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return 0.0, ["ffmpeg not found — audio_energy skipped"]

    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
        "-t", f"{max(5.0, min(max_probe_seconds, 600.0)):.1f}",
        "-i", filepath,
        "-af", "astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except (_sp.TimeoutExpired, OSError, FileNotFoundError) as exc:
        notes.append(f"astats probe failed: {exc}")
        return 0.0, notes

    text = (proc.stderr or "") + (proc.stdout or "")
    rms_vals: List[float] = []
    for line in text.splitlines():
        if "RMS_level=" in line:
            try:
                v = float(line.split("RMS_level=")[-1].strip())
                if math.isfinite(v):
                    rms_vals.append(v)
            except ValueError:
                continue

    if len(rms_vals) < 4:
        notes.append("insufficient RMS samples from astats")
        return 0.0, notes

    # dBFS range maps roughly to 0-40 dB dynamic range — richer audio ranks higher.
    spread = max(rms_vals) - min(rms_vals)
    # Peaks above -20 dBFS indicate "loud moments"; more peaks → more score.
    peak_frac = sum(1 for v in rms_vals if v > -20.0) / float(len(rms_vals))

    spread_score = min(100.0, (spread / 30.0) * 100.0)
    peak_score = min(100.0, peak_frac * 200.0)   # 50 % peaks → full score
    score = 0.6 * spread_score + 0.4 * peak_score
    return float(max(0.0, min(100.0, score))), notes


# ---------------------------------------------------------------------------
# Transcript hook score
# ---------------------------------------------------------------------------

def _hook_lexicon_score(text: str) -> Tuple[float, str]:
    """Pure-Python hook-density heuristic. Returns (0-100 score, best phrase)."""
    if not text:
        return 0.0, ""
    words = [w for w in re.split(r"\s+", text.lower().strip()) if w]
    if not words:
        return 0.0, ""

    hook_hits = sum(1 for w in words if w.strip(".,!?;:") in _HOOK_WORDS)
    numeric_hits = len(_NUMERIC_RE.findall(text))
    question_hits = len(_QUESTION_RE.findall(text))
    emphasis_hits = len(_EMPHASIS_RE.findall(text))

    density = hook_hits / max(10, len(words))       # per-word hook density
    numeric_density = numeric_hits / max(1, len(words))
    question_density = question_hits / max(1, len(words)) * 10.0

    raw = (density * 60.0
           + numeric_density * 40.0
           + question_density * 30.0
           + min(20.0, emphasis_hits * 2.0))
    score = max(0.0, min(100.0, raw))

    # Best phrase = the first sentence containing the most hook words.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    best_phrase, best_hits = "", 0
    for sent in sentences[:20]:
        lower = sent.lower()
        hits = sum(1 for w in _HOOK_WORDS if w in lower)
        if hits > best_hits:
            best_phrase, best_hits = sent.strip(), hits
    return score, best_phrase


def _transcript_hook_score(
    segments: Optional[Sequence[Dict[str, Any]]],
    transcript: str = "",
    llm_config: Optional[Any] = None,
) -> Tuple[float, str, List[str]]:
    """Blend LLM and lexicon hook scores when both are available."""
    notes: List[str] = []
    if not transcript and segments:
        transcript = " ".join(str(s.get("text", "")) for s in segments)
    transcript = transcript.strip()

    if not transcript:
        return 0.0, "", ["no transcript supplied — transcript_hook skipped"]

    lex_score, best_phrase = _hook_lexicon_score(transcript)

    if llm_config is None:
        notes.append("using keyword-lexicon fallback (no LLM configured)")
        return lex_score, best_phrase, notes

    # Try LLM — if it fails, stick with the lexicon score.
    try:
        from opencut.core.llm import query_llm
        prompt = (
            "Rate the following video transcript for short-form (TikTok/"
            "Reel/Short) 'hook strength' on a 0 to 100 integer scale. "
            "Higher means a more arresting opening, clearer curiosity "
            "gap, and quotable punchlines. Respond with ONLY the integer.\n\n"
            f"TRANSCRIPT:\n{transcript[:4000]}"
        )
        resp = query_llm(llm_config, prompt, max_tokens=8)
        raw = str(getattr(resp, "text", resp) or "").strip()
        m = re.search(r"\b\d{1,3}\b", raw)
        if m:
            llm_score = float(m.group(0))
            llm_score = max(0.0, min(100.0, llm_score))
            # Weighted blend: trust LLM 65%, lexicon 35%.
            blended = 0.65 * llm_score + 0.35 * lex_score
            notes.append(f"llm hook score: {int(llm_score)}, lexicon: {int(lex_score)}")
            return blended, best_phrase, notes
        notes.append("llm response had no parseable integer — falling back to lexicon")
    except Exception as exc:
        notes.append(f"llm call failed ({type(exc).__name__}) — lexicon fallback")

    return lex_score, best_phrase, notes


# ---------------------------------------------------------------------------
# Visual salience probe
# ---------------------------------------------------------------------------

def _visual_salience_score(
    filepath: str,
    max_probe_seconds: float = 30.0,
    sample_fps: float = 4.0,
) -> Tuple[float, List[str]]:
    """Estimate motion activity via FFmpeg ``signalstats`` YDIF.

    Higher YDIF → more inter-frame change → more visually kinetic clip.
    """
    notes: List[str] = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return 0.0, ["ffmpeg not found — visual_salience skipped"]

    probe_duration = max(5.0, min(max_probe_seconds, 90.0))
    sample_fps = max(1.0, min(sample_fps, 10.0))

    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info",
        "-t", f"{probe_duration:.1f}",
        "-i", filepath,
        "-vf", f"fps={sample_fps},signalstats,metadata=print:key=lavfi.signalstats.YDIF",
        "-f", "null", "-",
    ]
    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except (_sp.TimeoutExpired, OSError, FileNotFoundError) as exc:
        notes.append(f"signalstats probe failed: {exc}")
        return 0.0, notes

    text = (proc.stderr or "") + (proc.stdout or "")
    vals: List[float] = []
    for line in text.splitlines():
        if "YDIF=" in line:
            try:
                v = float(line.split("YDIF=")[-1].strip())
                if math.isfinite(v):
                    vals.append(v)
            except ValueError:
                continue

    if len(vals) < 3:
        notes.append("insufficient YDIF samples")
        return 0.0, notes

    mean_ydif = sum(vals) / len(vals)
    # Empirically YDIF 0-30 spans still-frame to action-footage.
    score = min(100.0, (mean_ydif / 25.0) * 100.0)
    return float(max(0.0, score)), notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(
    filepath: str,
    segments: Optional[Sequence[Dict[str, Any]]] = None,
    transcript: str = "",
    llm_config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
    skip_visual: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> ViralityResult:
    """Compute a 0-100 virality score for a single clip.

    Args:
        filepath: Absolute path to the video/audio file.
        segments: Optional whisper-style segment list for transcript-hook scoring.
        transcript: Raw transcript text. If omitted, concatenates ``segments``.
        llm_config: Optional ``LLMConfig``. Lexicon fallback used if absent.
        weights: Optional override for ``{audio_energy, transcript_hook,
            visual_salience}``. Normalised to sum 1.0.
        skip_visual: If True, skip the visual salience probe (faster).
        on_progress: ``on_progress(pct, msg="")`` default-arg callback.

    Returns:
        A ``ViralityResult`` subscriptable for Flask jsonify.
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("filepath must be a non-empty string")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    notes: List[str] = []
    w = _normalise_weights(weights)

    def _progress(pct: int, msg: str = "") -> None:
        if on_progress is not None:
            try:
                on_progress(int(pct), str(msg))
            except Exception:  # noqa: BLE001
                pass

    _progress(5, "probing audio energy")
    audio_score, a_notes = _audio_energy_score(filepath)
    notes.extend(a_notes)

    _progress(40, "scoring transcript hook")
    hook_score, best_phrase, h_notes = _transcript_hook_score(
        segments, transcript, llm_config,
    )
    notes.extend(h_notes)

    if skip_visual:
        visual_score, v_notes = 0.0, ["visual_salience skipped (skip_visual=True)"]
        # Re-normalise the weights so skipping one signal doesn't cap
        # the final composite.
        w = dict(w)
        dropped = w.pop("visual_salience", 0.0)
        total = sum(w.values()) or 1.0
        w = {k: v / total for k, v in w.items()}
        if dropped:
            notes.append(f"weights renormalised (dropped visual: {dropped:.2f})")
    else:
        _progress(70, "probing visual salience")
        visual_score, v_notes = _visual_salience_score(filepath)
    notes.extend(v_notes)

    signals = ViralitySignals(
        audio_energy=audio_score,
        transcript_hook=hook_score,
        visual_salience=visual_score,
    )
    composite = (
        signals.audio_energy * w.get("audio_energy", 0.0)
        + signals.transcript_hook * w.get("transcript_hook", 0.0)
        + signals.visual_salience * w.get("visual_salience", 0.0)
    )
    composite = max(0.0, min(100.0, composite))

    # Probe duration for the record.
    duration = 0.0
    try:
        from opencut.helpers import get_video_info
        info = get_video_info(filepath) or {}
        duration = float(info.get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        pass

    _progress(100, "done")
    return ViralityResult(
        score=round(composite, 2),
        signals=signals,
        weights=w,
        notes=notes,
        duration=duration,
        hook_phrase=best_phrase,
    )


def rank(
    candidates: Sequence[Dict[str, Any]],
    llm_config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
    skip_visual: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Score a batch of candidates and return them sorted score-desc.

    Each candidate must include ``filepath``; optional fields
    (``segments``, ``transcript``, ``label``) are passed through into
    the output dict. Scoring is deterministic for identical inputs;
    ties break by the original list order (stable sort).
    """
    if not candidates:
        return []

    n = len(candidates)
    results: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        fp = str(cand.get("filepath") or "").strip()
        if not fp:
            results.append({
                "filepath": "", "score": 0.0,
                "error": "missing filepath",
                "input_index": i,
            })
            continue

        def _inner(pct: int, msg: str = "", _i=i) -> None:
            if on_progress is None:
                return
            outer_pct = int((100.0 * _i + pct) / n)
            on_progress(outer_pct, msg)

        try:
            r = score(
                fp,
                segments=cand.get("segments"),
                transcript=cand.get("transcript", ""),
                llm_config=llm_config,
                weights=weights,
                skip_visual=skip_visual,
                on_progress=_inner,
            )
            results.append({
                "filepath": fp,
                "score": r.score,
                "signals": dict(
                    audio_energy=r.signals.audio_energy,
                    transcript_hook=r.signals.transcript_hook,
                    visual_salience=r.signals.visual_salience,
                ),
                "weights": dict(r.weights),
                "duration": r.duration,
                "hook_phrase": r.hook_phrase,
                "notes": list(r.notes),
                "label": cand.get("label", ""),
                "input_index": i,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "filepath": fp,
                "score": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "input_index": i,
            })

    # Stable sort by score desc, preserving insertion order on ties.
    results.sort(key=lambda r: -float(r.get("score") or 0.0))
    return results


__all__ = [
    "ViralitySignals",
    "ViralityResult",
    "check_virality_score_available",
    "score",
    "rank",
]
