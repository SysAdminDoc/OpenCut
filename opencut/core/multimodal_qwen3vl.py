"""Local Ollama-backed Qwen3-VL video analysis.

The adapter samples a small number of decision-point frames for each
transcript segment. Ollama scores the transcript and pixels separately, then
OpenCut applies a fixed transcript-first blend so visual novelty cannot drown
out what was actually said.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")

CancellationCallback = Optional[Callable[[], bool]]

DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
TRANSCRIPT_WEIGHT = 0.65
VISUAL_WEIGHT = 0.35
# Each segment costs up to six 20s ffmpeg frame grabs plus one Ollama call, so
# the segment count is the only thing standing between a request and hours of
# work. Callers above this are refused by the route; the no-transcript fallback
# coarsens its windows to stay under it rather than refusing.
MAX_SEGMENTS = 2000
# Notes are one string per skipped segment and land in the persisted job
# result, so they need their own ceiling.
MAX_NOTES = 100
INSTALL_HINT = "Install Ollama, then run: ollama pull qwen3-vl:8b"
_VISION_SYSTEM_PROMPT = (
    "You are a careful video editor. Score the transcript and sampled frames "
    "separately. Return only one JSON object with numeric scores from 0 to 1."
)


@dataclass
class Qwen3VLResult:
    query: str = ""
    response: str = ""
    structured_data: List = field(default_factory=list)
    model: str = ""
    processing_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    frames_analyzed: int = 0
    transcript_weight: float = TRANSCRIPT_WEIGHT
    visual_weight: float = VISUAL_WEIGHT

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return (
            "query",
            "response",
            "structured_data",
            "model",
            "processing_seconds",
            "notes",
            "frames_analyzed",
            "transcript_weight",
            "visual_weight",
        )

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def _default_model() -> str:
    return str(os.environ.get("OPENCUT_QWEN3VL_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _default_base_url() -> str:
    return str(os.environ.get("OPENCUT_OLLAMA_BASE_URL") or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp_score(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value)))


def _normalise_transcript_segments(segments: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert transcript-like mappings into bounded scoring windows."""
    normalised = []
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        start = max(0.0, _finite_float(item.get("start"), 0.0))
        end = max(start, _finite_float(item.get("end"), start))
        text = str(item.get("text") or item.get("transcript") or "").strip()
        if end <= start and not text:
            continue
        normalised.append({"start": start, "end": end, "text": text})
    return normalised


def _raise_if_cancelled(is_cancelled: CancellationCallback) -> None:
    if is_cancelled and is_cancelled():
        raise InterruptedError("Qwen3-VL analysis cancelled")


def _probe_duration(video_path: str) -> float:
    """Read a video duration without adding a media-library dependency."""
    try:
        result = subprocess.run(
            [
                get_ffprobe_path(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            duration = _finite_float(result.stdout.strip())
            if duration > 0:
                return duration
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        logger.warning("Qwen3-VL duration probe failed for %s: %s", video_path, exc)
    return 0.0


def _decision_points(start: float, end: float, count: int) -> List[float]:
    """Return evenly spaced frame timestamps inside a segment."""
    count = max(1, min(6, int(count)))
    if end <= start:
        return [start]
    if count == 1:
        return [start + (end - start) / 2.0]
    return [start + ((end - start) * index / (count - 1)) for index in range(count)]


def _sample_video_frames(video_path: str, timestamps: Sequence[float]) -> List[Dict[str, Any]]:
    """Extract small JPEG frames directly into memory for Ollama."""
    frames: List[Dict[str, Any]] = []
    try:
        ffmpeg = get_ffmpeg_path()
    except (OSError, RuntimeError) as exc:
        logger.warning("Qwen3-VL frame extraction is unavailable: %s", exc)
        return frames

    for timestamp in timestamps:
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-ss",
                    f"{max(0.0, timestamp):.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=640:-1",
                    "-q:v",
                    "5",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=20,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("Qwen3-VL frame extraction failed at %.3fs: %s", timestamp, exc)
            continue
        if result.returncode != 0 or len(result.stdout or b"") < 100:
            continue
        frames.append({
            "timestamp": round(max(0.0, timestamp), 3),
            "base64": base64.b64encode(result.stdout).decode("ascii"),
        })
    return frames


def _fallback_segments(duration: float, segment_duration: float) -> List[Dict[str, Any]]:
    duration = max(0.0, duration)
    segment_duration = max(1.0, segment_duration)
    if duration <= 0:
        return [{"start": 0.0, "end": segment_duration, "text": ""}]
    # A long file at a fine segment_duration would otherwise produce thousands
    # of windows. Coarsen rather than refuse: the caller asked to analyse the
    # whole video, so cover it at the finest granularity the cap allows.
    if duration / segment_duration > MAX_SEGMENTS:
        segment_duration = duration / MAX_SEGMENTS
    segments = []
    start = 0.0
    while start < duration:
        end = min(duration, start + segment_duration)
        segments.append({"start": start, "end": end, "text": ""})
        start = end
    return segments


def _build_segment_prompt(
    prompt: str,
    start: float,
    end: float,
    transcript: str,
    frame_count: int,
) -> str:
    return (
        f"{prompt.strip() or 'Find the most relevant moments in this video segment.'}\n\n"
        f"SEGMENT: {start:.3f}-{end:.3f} seconds\n"
        f"TRANSCRIPT (primary signal): {transcript or '[no transcript available]'}\n"
        f"SAMPLED PIXELS: {frame_count} decision-point frames\n\n"
        "Return JSON only with these fields: "
        "transcript_score, visual_score, relevance, reason, summary. "
        "Each score must be between 0 and 1. Score transcript relevance first "
        "and visual relevance second. The final relevance is calculated by "
        f"OpenCut as {TRANSCRIPT_WEIGHT:.2f} * transcript_score + "
        f"{VISUAL_WEIGHT:.2f} * visual_score."
    )


def _parse_score_payload(text: str) -> Dict[str, Any]:
    """Parse JSON from a model response, including fenced or chatty output."""
    value = str(text or "").strip()
    if not value:
        return {}
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    candidates = [value]
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        candidates.append(value[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
    return {}


def _score_value(payload: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in payload:
            return _clamp_score(payload[key])
    return 0.0


def check_qwen3vl_available(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """Return whether a local Ollama server exposes the requested Qwen model."""
    from opencut.core.llm import list_ollama_models

    requested = str(model or _default_model()).strip().lower()
    requested_base = requested.split(":", 1)[0]
    requested_has_tag = ":" in requested
    try:
        models = list_ollama_models(base_url or _default_base_url(), timeout=1.5)
    except Exception:  # noqa: BLE001 - availability probes must never break the route
        return False
    for available in models:
        name = str(available).strip().lower()
        if name == requested or (not requested_has_tag and name.split(":", 1)[0] == requested_base):
            return True
    return False


def check_transformers_available() -> bool:
    """Backward-compatible probe name for the now Ollama-backed lane."""
    return check_qwen3vl_available()


def analyze(
    video_path: str,
    prompt: str = "Summarise the video.",
    max_tokens: int = 1024,
    on_progress=None,
    transcript_segments: Optional[Iterable[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    segment_duration: float = 30.0,
    frames_per_segment: int = 3,
    frame_interval: float = 10.0,
    base_url: Optional[str] = None,
    is_cancelled: CancellationCallback = None,
    **kwargs,
) -> Qwen3VLResult:
    """Score transcript windows with local Qwen3-VL decision-point frames.

    *is_cancelled* is polled once per segment. It must be an explicit
    parameter: this function ends in ``**kwargs`` with ``del kwargs``, so a
    callback passed by keyword alone would be silently discarded.
    """
    del kwargs
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    started = time.monotonic()
    selected_model = str(model or _default_model()).strip() or DEFAULT_MODEL
    selected_url = str(base_url or _default_base_url()).strip() or DEFAULT_BASE_URL
    segment_duration = max(1.0, _finite_float(segment_duration, 30.0))
    frames_per_segment = max(1, min(6, int(_finite_float(frames_per_segment, 3))))
    frame_interval = max(1.0, _finite_float(frame_interval, 10.0))
    segments = _normalise_transcript_segments(transcript_segments)
    duration = _probe_duration(video_path)
    if not segments:
        segments = _fallback_segments(duration, segment_duration)

    from opencut.core.llm import LLMConfig, query_ollama_vision

    config = LLMConfig(
        provider="ollama",
        model=selected_model,
        base_url=selected_url,
        temperature=0.1,
        max_tokens=max(1, min(8192, int(_finite_float(max_tokens, 1024)))),
    )
    scored_segments: List[Dict[str, Any]] = []
    notes = [
        f"Transcript-first blend: {TRANSCRIPT_WEIGHT:.0%} transcript, {VISUAL_WEIGHT:.0%} visual.",
        f"Frames sampled at up to {frame_interval:g}s decision points per segment.",
    ]
    frames_analyzed = 0

    def _note(message: str) -> None:
        """Record a per-segment note, keeping the persisted list bounded."""
        if len(notes) < MAX_NOTES:
            notes.append(message)
        elif len(notes) == MAX_NOTES:
            notes.append("Further per-segment notes were suppressed.")

    for index, segment in enumerate(segments):
        # Polled per segment so a cancelled job stops here instead of running
        # the whole list out. Nothing else can interrupt this worker: the frame
        # grabs use subprocess.run, so no process is registered for the job
        # runner to kill.
        _raise_if_cancelled(is_cancelled)
        start = max(0.0, _finite_float(segment.get("start")))
        end = max(start, _finite_float(segment.get("end"), start))
        timestamps = _decision_points(start, end, frames_per_segment)
        frames = _sample_video_frames(video_path, timestamps)
        frames_analyzed += len(frames)
        if not frames:
            _note(f"Segment {index + 1} had no readable decision-point frames.")
            continue

        if on_progress:
            on_progress(
                15 + int(70 * index / max(1, len(segments))),
                f"Scoring Qwen3-VL segment {index + 1} of {len(segments)}...",
            )

        segment_prompt = _build_segment_prompt(
            prompt,
            start,
            end,
            str(segment.get("text") or ""),
            len(frames),
        )
        try:
            response = query_ollama_vision(
                prompt=segment_prompt,
                images=[frame["base64"] for frame in frames],
                config=config,
                system_prompt=_VISION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001
            _note(f"Segment {index + 1} vision query failed: {exc}")
            continue

        response_text = getattr(response, "text", response)
        if not response_text or str(response_text).startswith("LLM error:"):
            _note(f"Segment {index + 1} returned no usable vision response.")
            continue
        payload = _parse_score_payload(str(response_text))
        transcript_score = _score_value(
            payload,
            "transcript_score",
            "transcript_relevance",
        )
        visual_score = _score_value(
            payload,
            "visual_score",
            "visual_relevance",
            "pixel_score",
        )
        segment_result = {
            "start": round(start, 3),
            "end": round(end, 3),
            "transcript": str(segment.get("text") or ""),
            "frame_timestamps": [frame["timestamp"] for frame in frames],
            "frames": len(frames),
            "transcript_score": round(transcript_score, 4),
            "visual_score": round(visual_score, 4),
            "relevance": round(
                TRANSCRIPT_WEIGHT * transcript_score + VISUAL_WEIGHT * visual_score,
                4,
            ),
            "reason": str(payload.get("reason") or "").strip(),
            "summary": str(payload.get("summary") or "").strip(),
        }
        scored_segments.append(segment_result)

    if on_progress:
        on_progress(100, f"Qwen3-VL scored {len(scored_segments)} segments")
    if not scored_segments:
        raise RuntimeError(
            "Qwen3-VL produced no segment scores. Verify Ollama is running and "
            "the qwen3-vl model is installed."
        )

    scored_segments.sort(key=lambda item: item["relevance"], reverse=True)
    elapsed = time.monotonic() - started
    return Qwen3VLResult(
        query=prompt,
        response=json.dumps({"segments": scored_segments}, ensure_ascii=False),
        structured_data=scored_segments,
        model=selected_model,
        processing_seconds=round(elapsed, 3),
        notes=notes,
        frames_analyzed=frames_analyzed,
    )


__all__ = [
    "Qwen3VLResult",
    "check_qwen3vl_available",
    "check_transformers_available",
    "INSTALL_HINT",
    "analyze",
]
