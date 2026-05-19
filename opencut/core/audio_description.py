"""
OpenCut Audio Description Track Generator v1.0.0

Generate audio description (AD) tracks for accessibility:
- Detect dialogue pauses/gaps suitable for descriptions
- Describe visual content at timestamps
- Synthesize description speech
- Mix descriptions into an AD track alongside original audio

Uses FFmpeg for audio processing and gap detection.
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg
from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class DescriptionGap:
    """A gap in dialogue suitable for audio description."""
    start: float
    end: float
    duration: float = 0.0
    suitable: bool = True
    max_words: int = 0  # Estimated words that fit in this gap


@dataclass
class VisualDescription:
    """A description of visual content at a timestamp."""
    timestamp: float
    description: str
    duration: float = 0.0
    importance: str = "normal"  # low, normal, high


@dataclass
@openapi_response_schema("/audio/description/generate")
class ADResult:
    """Result from audio description generation."""
    output_path: str = ""
    gaps_found: int = 0
    descriptions_added: int = 0
    total_description_duration: float = 0.0
    original_duration: float = 0.0


@dataclass
class AudioDescriptionCue:
    """Human-reviewable AD cue compatible with Microsoft AD authoring flow."""

    cue_id: str
    scene_index: int
    scene_start: float
    scene_end: float
    target_start: float
    target_end: float
    available_gap_seconds: float
    max_words: int
    estimated_duration: float
    fits_gap: bool
    script: str
    source_description: str
    dialogue_context: str = ""
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    priority: str = "normal"
    needs_review: bool = True
    review_reason: str = "draft-ai-description"
    tts_backend_hint: str = "indextts2"
    timing_mode: str = "standard"
    extended_pause_seconds: float = 0.0
    descriptive_transcript_text: str = ""


@dataclass
@openapi_response_schema(
    "/audio/description/microsoft-draft",
    extra_properties={
        "cue_count": {"type": "integer"},
        "transcript_segment_count": {"type": "integer"},
        "gap_count": {"type": "integer"},
        "workflow": {"type": "array", "items": {"type": "string"}},
        "wcag3_compatibility": {"type": "object"},
        "descriptive_transcript": {"type": "array", "items": {"type": "object"}},
        "extended_timing_plan": {"type": "array", "items": {"type": "object"}},
    },
)
class AudioDescriptionReviewDraft:
    """Draft AD package for review before TTS insertion/rendering."""

    cues: List[AudioDescriptionCue] = field(default_factory=list)
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[DescriptionGap] = field(default_factory=list)
    source: str = "microsoft/ai-audio-descriptions"
    source_url: str = "https://github.com/microsoft/ai-audio-descriptions"
    draft_format: str = "opencut.microsoft-ad-draft.v1"
    method: str = "scene-description-plus-transcript"
    tts_backend_hint: str = "indextts2"
    review_required: bool = True
    notes: List[str] = field(default_factory=list)
    wcag3_compatibility: Dict[str, Any] = field(default_factory=dict)
    descriptive_transcript: List[Dict[str, Any]] = field(default_factory=list)
    extended_timing_plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-safe response body."""
        return {
            "draft_format": self.draft_format,
            "source": self.source,
            "source_url": self.source_url,
            "method": self.method,
            "tts_backend_hint": self.tts_backend_hint,
            "review_required": self.review_required,
            "wcag3_compatibility": dict(self.wcag3_compatibility),
            "cue_count": len(self.cues),
            "transcript_segment_count": len(self.transcript_segments),
            "gap_count": len(self.gaps),
            "notes": list(self.notes),
            "cues": [asdict(c) for c in self.cues],
            "transcript_segments": list(self.transcript_segments),
            "gaps": [asdict(g) for g in self.gaps],
            "descriptive_transcript": list(self.descriptive_transcript),
            "extended_timing_plan": list(self.extended_timing_plan),
            "workflow": [
                "per-scene descriptions",
                "dialogue transcript alignment",
                "silence-gap assignment",
                "human AD editor review",
                "TTS insertion",
            ],
        }


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------
def find_description_gaps(
    audio_path: str,
    transcript: Optional[List[Dict]] = None,
    min_gap_seconds: float = 2.0,
    max_gap_seconds: float = 15.0,
    silence_threshold_db: float = -35.0,
    on_progress: Optional[Callable] = None,
) -> List[DescriptionGap]:
    """
    Find gaps in dialogue suitable for audio descriptions.

    Uses FFmpeg silencedetect to find pauses, optionally cross-referenced
    with a transcript for more accurate gap identification.

    Args:
        audio_path: Source audio/video file.
        transcript: Optional transcript (list of dicts with start, end, text).
        min_gap_seconds: Minimum gap duration to consider.
        max_gap_seconds: Maximum gap duration to consider.
        silence_threshold_db: Silence detection threshold in dB.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of DescriptionGap objects.
    """
    if on_progress:
        on_progress(5, "Detecting dialogue gaps...")

    ffmpeg = get_ffmpeg_path()

    # Use transcript gaps if available
    if transcript:
        gaps = _gaps_from_transcript(transcript, min_gap_seconds, max_gap_seconds)
        if on_progress:
            on_progress(100, f"Found {len(gaps)} gaps from transcript")
        return gaps

    # Otherwise use silence detection
    cmd = [
        ffmpeg, "-hide_banner", "-i", audio_path,
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_gap_seconds}",
        "-f", "null", "-",
    ]

    try:
        stderr_output = run_ffmpeg(cmd, timeout=600, stderr_cap=1)
    except RuntimeError:
        stderr_output = ""

    if on_progress:
        on_progress(50, "Parsing silence regions...")

    gaps = []
    lines = stderr_output.split("\n") if stderr_output else []

    silence_start = None
    for line in lines:
        if "silence_start:" in line:
            try:
                silence_start = float(
                    line.split("silence_start:")[1].strip().split()[0]
                )
            except (ValueError, IndexError):
                silence_start = None
        elif "silence_end:" in line and silence_start is not None:
            try:
                silence_end = float(
                    line.split("silence_end:")[1].strip().split()[0]
                )
                duration = silence_end - silence_start
                if min_gap_seconds <= duration <= max_gap_seconds:
                    # Estimate words: ~3 words per second for AD narration
                    max_words = int(duration * 3)
                    gaps.append(DescriptionGap(
                        start=silence_start,
                        end=silence_end,
                        duration=duration,
                        suitable=True,
                        max_words=max_words,
                    ))
            except (ValueError, IndexError):
                pass
            silence_start = None

    if on_progress:
        on_progress(100, f"Found {len(gaps)} suitable gaps")

    return gaps


def _gaps_from_transcript(
    transcript: List[Dict],
    min_gap: float,
    max_gap: float,
) -> List[DescriptionGap]:
    """Extract gaps from transcript timing data."""
    gaps = []
    sorted_entries = sorted(transcript, key=lambda x: float(x.get("start", 0)))

    for i in range(len(sorted_entries) - 1):
        current_end = float(sorted_entries[i].get("end", 0))
        next_start = float(sorted_entries[i + 1].get("start", 0))
        duration = next_start - current_end

        if min_gap <= duration <= max_gap:
            gaps.append(DescriptionGap(
                start=current_end,
                end=next_start,
                duration=duration,
                suitable=True,
                max_words=int(duration * 3),
            ))

    return gaps


# ---------------------------------------------------------------------------
# Visual Content Description
# ---------------------------------------------------------------------------
def describe_visual_content(
    video_path: str,
    timestamp: float,
    on_progress: Optional[Callable] = None,
) -> VisualDescription:
    """
    Describe visual content at a specific timestamp.

    Extracts a frame at the given timestamp and generates a text
    description. Uses basic frame analysis for scene type detection.

    Args:
        video_path: Source video file.
        timestamp: Timestamp in seconds to describe.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualDescription with text description of the scene.
    """
    if on_progress:
        on_progress(10, f"Analyzing frame at {timestamp:.1f}s...")

    ffmpeg = get_ffmpeg_path()

    # Extract frame for analysis
    temp_frame = None
    try:
        ntf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_frame = ntf.name
        ntf.close()

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            temp_frame,
        ]
        run_ffmpeg(cmd, timeout=60)

        # Basic frame analysis using FFmpeg signalstats
        stats_cmd = [
            ffmpeg, "-hide_banner",
            "-i", temp_frame,
            "-vf", "signalstats",
            "-f", "null", "-",
        ]
        try:
            stats_output = run_ffmpeg(stats_cmd, timeout=30, stderr_cap=1)
        except RuntimeError:
            stats_output = ""

        # Generate description based on available analysis
        description = _generate_scene_description(stats_output, timestamp)

        if on_progress:
            on_progress(100, "Frame analysis complete")

        return VisualDescription(
            timestamp=timestamp,
            description=description,
            importance="normal",
        )

    finally:
        if temp_frame and os.path.exists(temp_frame):
            try:
                os.unlink(temp_frame)
            except OSError:
                pass


def _generate_scene_description(stats_output: str, timestamp: float) -> str:
    """Generate a scene description from frame statistics."""
    # Basic analysis from signal stats
    brightness = "normal"
    if "YAVG" in stats_output:
        try:
            for line in stats_output.split("\n"):
                if "YAVG" in line:
                    val = float(line.split("YAVG:")[1].strip().split()[0])
                    if val < 50:
                        brightness = "dark"
                    elif val > 200:
                        brightness = "bright"
                    break
        except (ValueError, IndexError):
            pass

    # Return a placeholder description indicating the scene characteristics
    time_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
    return (
        f"Scene at {time_str}: "
        f"{'Dark' if brightness == 'dark' else 'Bright' if brightness == 'bright' else 'Normal lighting'} scene. "
        f"[Visual description placeholder — connect an AI vision model for detailed descriptions.]"
    )


def describe_scene_llm(
    timestamp: float,
    transcript_context: str,
    duration: float = 3.0,
    llm_config=None,
    brightness_hint: str = "normal",
) -> str:
    """Generate an AD line via LLM using transcript context.

    Real-world audio description scripts are written by humans who read
    the screenplay / transcript and fill in *what's happening visually*
    between lines of dialogue. This function asks the configured LLM
    to do the same — taking the surrounding dialogue as context plus
    an optional brightness hint — and outputs a single short
    present-tense description that fits in ``duration`` seconds at a
    typical AD reading speed (~3 words per second).

    Unlike :func:`describe_visual_content`, this path does not do
    frame analysis — there's no vision model requirement. The LLM
    fills in plausible scene text from transcript cues alone. Best
    paired with :func:`describe_visual_content` to anchor the AD line
    with a brightness hint from the actual frame.

    Args:
        timestamp: Timestamp in seconds the description targets.
        transcript_context: Surrounding dialogue (e.g. the last 60 s
            of captions preceding ``timestamp`` + the first 30 s
            following). Empty string is tolerated but yields weaker
            results.
        duration: Available gap length in seconds. Used to cap the
            word-count target (~3 words/sec).
        llm_config: :class:`opencut.core.llm.LLMConfig` instance. When
            ``None``, falls back to the heuristic description — no
            error is raised so callers can pass ``None`` unconditionally
            and let the configured provider (or its absence) decide.
        brightness_hint: ``"dark"`` / ``"normal"`` / ``"bright"`` — a
            short cue the LLM can weave into the output.

    Returns:
        A single-line AD string. Falls back to the heuristic template
        when the LLM provider isn't available or the call fails — this
        path must never raise since AD rendering proceeds line-by-line
        and one missing line shouldn't kill the whole job.
    """
    if llm_config is None:
        return f"[{brightness_hint.title()} scene at {int(timestamp)}s.]"

    max_words = max(4, min(30, int(duration * 3.0)))
    prompt = (
        "You are writing audio-description copy for a video. "
        f"At timestamp {int(timestamp)} s there is a {duration:.1f}-second "
        "pause in dialogue. Write ONE short present-tense sentence "
        f"(maximum {max_words} words) describing what's likely happening "
        "visually, based on the surrounding dialogue below. No speaker "
        "names, no line breaks, no quotation marks.\n\n"
        f"Brightness cue from the frame: {brightness_hint}\n"
        f"Surrounding dialogue:\n{transcript_context or '(no transcript available)'}"
    )

    try:
        from opencut.core.llm import query_llm
        resp = query_llm(prompt, config=llm_config)
        text = (getattr(resp, "text", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("describe_scene_llm failed: %s", exc)
        return f"[{brightness_hint.title()} scene at {int(timestamp)}s.]"

    # Strip surrounding quotes the model sometimes wraps its output in
    text = text.strip(" \t\r\n\"'")
    # Collapse whitespace and cap length defensively
    text = " ".join(text.split())
    if not text:
        return f"[{brightness_hint.title()} scene at {int(timestamp)}s.]"
    # Hard cap — AD lines shouldn't overflow the gap
    words = text.split()
    if len(words) > max_words + 4:
        text = " ".join(words[:max_words + 4])
    return text


# ---------------------------------------------------------------------------
# Microsoft ai-audio-descriptions Review Draft
# ---------------------------------------------------------------------------

def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_ad_text(text: str) -> str:
    cleaned = " ".join(str(text or "").replace("\n", " ").split())
    return cleaned.strip(" \t\r\n\"'")


def _cap_words(text: str, max_words: int) -> str:
    """Cap AD copy to a word budget without returning an unterminated phrase."""
    cleaned = _clean_ad_text(text)
    words = cleaned.split()
    if not words:
        return ""
    if max_words <= 0 or len(words) <= max_words:
        return cleaned
    clipped = " ".join(words[:max(1, max_words)])
    return clipped.rstrip(" ,;:") + "."


def _normalize_transcript_segments(transcript) -> List[Dict[str, Any]]:
    """Normalize common caption/Whisper transcript shapes."""
    if not transcript:
        return []

    if isinstance(transcript, dict):
        if isinstance(transcript.get("segments"), list):
            raw_segments = transcript["segments"]
        elif isinstance(transcript.get("words"), list):
            raw_segments = transcript["words"]
        else:
            raw_segments = [transcript]
    elif isinstance(transcript, list):
        raw_segments = transcript
    else:
        return []

    segments: List[Dict[str, Any]] = []
    for entry in raw_segments:
        if not isinstance(entry, dict):
            continue
        start = _as_float(
            entry.get("start", entry.get("start_time", entry.get("time", 0.0))),
            0.0,
        )
        end = _as_float(
            entry.get("end", entry.get("end_time", entry.get("stop", start))),
            start,
        )
        if end < start:
            start, end = end, start
        text = _clean_ad_text(
            entry.get("text")
            or entry.get("word")
            or entry.get("line")
            or entry.get("caption")
            or ""
        )
        speaker = _clean_ad_text(entry.get("speaker") or entry.get("speaker_id") or "")
        if not text and end <= start:
            continue
        item = {
            "start": round(max(0.0, start), 3),
            "end": round(max(0.0, end), 3),
            "text": text,
        }
        if speaker:
            item["speaker"] = speaker
        segments.append(item)

    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def _normalize_description_gaps(gaps) -> List[DescriptionGap]:
    if not gaps:
        return []
    normalized: List[DescriptionGap] = []
    for entry in gaps:
        if isinstance(entry, DescriptionGap):
            gap = entry
        elif isinstance(entry, dict):
            start = _as_float(entry.get("start", entry.get("start_time", 0.0)), 0.0)
            end_default = start + _as_float(entry.get("duration", 0.0), 0.0)
            end = _as_float(entry.get("end", entry.get("end_time", end_default)), end_default)
            if end < start:
                start, end = end, start
            duration = max(0.0, _as_float(entry.get("duration", end - start), end - start))
            gap = DescriptionGap(
                start=round(max(0.0, start), 3),
                end=round(max(0.0, end), 3),
                duration=round(duration, 3),
                suitable=bool(entry.get("suitable", True)),
                max_words=int(entry.get("max_words", max(0, int(duration * 3.0)))),
            )
        else:
            continue
        if gap.duration <= 0:
            gap.duration = max(0.0, gap.end - gap.start)
        normalized.append(gap)
    return sorted(normalized, key=lambda g: (g.start, g.end))


def _normalize_scene_descriptions(scene_descriptions) -> List[Dict[str, Any]]:
    if not scene_descriptions:
        return []
    if isinstance(scene_descriptions, dict):
        if isinstance(scene_descriptions.get("descriptions"), list):
            raw_scenes = scene_descriptions["descriptions"]
        elif isinstance(scene_descriptions.get("scenes"), list):
            raw_scenes = scene_descriptions["scenes"]
        else:
            raw_scenes = [scene_descriptions]
    else:
        raw_scenes = scene_descriptions

    scenes: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_scenes):
        if isinstance(entry, VisualDescription):
            timestamp = entry.timestamp
            description = entry.description
            importance = entry.importance
        elif hasattr(entry, "timestamp") and hasattr(entry, "description"):
            timestamp = _as_float(getattr(entry, "timestamp"), float(idx) * 5.0)
            description = _clean_ad_text(getattr(entry, "description", ""))
            importance = _clean_ad_text(getattr(entry, "importance", "normal")) or "normal"
        elif isinstance(entry, dict):
            timestamp = _as_float(
                entry.get("timestamp", entry.get("time", entry.get("start", float(idx) * 5.0))),
                float(idx) * 5.0,
            )
            description = _clean_ad_text(
                entry.get("description")
                or entry.get("text")
                or entry.get("alt_text")
                or entry.get("script")
                or ""
            )
            importance = _clean_ad_text(entry.get("importance") or entry.get("priority") or "normal")
        else:
            continue
        if not description:
            description = f"Visual scene at {timestamp:.1f} seconds."
        scenes.append({
            "timestamp": round(max(0.0, timestamp), 3),
            "description": description,
            "importance": importance or "normal",
        })
    return sorted(scenes, key=lambda s: s["timestamp"])


def _scene_descriptions_from_video(
    video_path: str,
    scene_timestamps: Optional[List[float]],
    llm_config,
    on_progress: Optional[Callable],
) -> List[Dict[str, Any]]:
    from opencut.core.scene_description import describe_all_scenes

    result = describe_all_scenes(
        video_path,
        scene_timestamps=scene_timestamps,
        llm_config=llm_config,
        on_progress=on_progress,
    )
    return [
        {
            "timestamp": desc.timestamp,
            "description": desc.description or desc.alt_text,
            "importance": "normal",
        }
        for desc in result.descriptions
    ]


def _dialogue_context(
    transcript_segments: List[Dict[str, Any]],
    start: float,
    end: float,
    context_seconds: float,
) -> tuple[str, List[Dict[str, Any]]]:
    window_start = max(0.0, start - context_seconds)
    window_end = end + context_seconds
    selected = [
        seg for seg in transcript_segments
        if seg["end"] >= window_start and seg["start"] <= window_end
    ]
    parts = []
    for seg in selected[:12]:
        speaker = f"{seg.get('speaker')}: " if seg.get("speaker") else ""
        parts.append(
            f"[{seg['start']:.1f}-{seg['end']:.1f}] "
            f"{speaker}{seg.get('text', '')}"
        )
    return " ".join(parts)[:800], selected[:12]


def _format_transcript_time(seconds: float) -> str:
    value = max(0.0, float(seconds))
    millis = int(round((value - int(value)) * 1000))
    total_seconds = int(value)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _build_descriptive_transcript(
    cues: List[AudioDescriptionCue],
    transcript_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for seg in transcript_segments:
        speaker = str(seg.get("speaker") or "").strip()
        text = _clean_ad_text(seg.get("text") or "")
        if not text:
            continue
        events.append(
            {
                "type": "dialogue",
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "text": text,
            }
        )
    for cue in cues:
        text = cue.descriptive_transcript_text or cue.script
        if not text:
            continue
        events.append(
            {
                "type": "visual_description",
                "cue_id": cue.cue_id,
                "start": cue.scene_start,
                "end": cue.scene_end,
                "target_start": cue.target_start,
                "target_end": cue.target_end,
                "timing_mode": cue.timing_mode,
                "text": text,
            }
        )
    events.sort(key=lambda item: (float(item.get("start", 0.0)), item.get("type", ""), item.get("cue_id", "")))
    for event in events:
        start = _format_transcript_time(float(event.get("start", 0.0)))
        end = _format_transcript_time(float(event.get("end", event.get("start", 0.0))))
        label = "Visual description" if event["type"] == "visual_description" else "Dialogue"
        speaker = f" {event.get('speaker')}:" if event.get("speaker") else ":"
        event["display_text"] = f"[{start} - {end}] {label}{speaker} {event['text']}"
    return events


def _build_extended_timing_plan(cues: List[AudioDescriptionCue]) -> List[Dict[str, Any]]:
    plan = []
    for cue in cues:
        if cue.timing_mode != "extended":
            continue
        plan.append(
            {
                "cue_id": cue.cue_id,
                "pause_at": cue.target_start,
                "resume_at": cue.target_end,
                "pause_seconds": cue.extended_pause_seconds,
                "reason": cue.review_reason,
            }
        )
    return plan


def _choose_gap_for_scene(
    scene_start: float,
    scene_end: float,
    gaps: List[DescriptionGap],
    used_indices: set[int],
) -> tuple[Optional[int], Optional[DescriptionGap]]:
    candidates: List[tuple[float, int, DescriptionGap]] = []
    for idx, gap in enumerate(gaps):
        if idx in used_indices or not gap.suitable:
            continue
        overlap = min(scene_end, gap.end) - max(scene_start, gap.start)
        if overlap > 0:
            score = -overlap
        elif gap.start >= scene_start:
            score = gap.start - scene_start
        else:
            score = scene_start - gap.end + 30.0
        candidates.append((score, idx, gap))
    if not candidates:
        return None, None
    _, idx, gap = min(candidates, key=lambda item: item[0])
    return idx, gap


def build_microsoft_audio_description_draft(
    video_path: str = "",
    *,
    scene_descriptions: Optional[List[Dict]] = None,
    scene_timestamps: Optional[List[float]] = None,
    transcript: Optional[Any] = None,
    gaps: Optional[List[Dict]] = None,
    min_gap_seconds: float = 1.0,
    max_gap_seconds: float = 15.0,
    context_seconds: float = 6.0,
    words_per_second: float = 3.0,
    tts_backend_hint: str = "indextts2",
    include_wcag3_hooks: bool = False,
    include_descriptive_transcript: bool = False,
    extended_timing: bool = False,
    llm_config=None,
    on_progress: Optional[Callable] = None,
) -> AudioDescriptionReviewDraft:
    """Build a Microsoft-style AD draft for human review.

    Microsoft `ai-audio-descriptions` documents an authoring flow that
    generates per-scene visual descriptions, aligns them with dialogue
    transcript and silence gaps, presents a draft to a human AD editor, and
    only then inserts TTS into the video. This function implements that local
    review contract without requiring Azure credentials or model downloads.
    """
    min_gap_seconds = max(0.1, float(min_gap_seconds))
    max_gap_seconds = max(min_gap_seconds, float(max_gap_seconds))
    context_seconds = max(0.0, float(context_seconds))
    words_per_second = max(1.0, min(5.0, float(words_per_second)))
    tts_backend_hint = _clean_ad_text(tts_backend_hint or "indextts2").lower()
    include_wcag3_hooks = bool(include_wcag3_hooks)
    include_descriptive_transcript = bool(include_descriptive_transcript or include_wcag3_hooks)
    extended_timing = bool(extended_timing or include_wcag3_hooks)

    if on_progress:
        on_progress(5, "Preparing transcript and scene descriptions...")

    transcript_segments = _normalize_transcript_segments(transcript)
    scenes = _normalize_scene_descriptions(scene_descriptions)

    if not scenes:
        if not video_path or not os.path.isfile(video_path):
            raise ValueError("scene_descriptions or a valid video_path is required")
        scenes = _scene_descriptions_from_video(video_path, scene_timestamps, llm_config, on_progress)

    if on_progress:
        on_progress(35, "Finding dialogue gaps for AD placement...")

    normalized_gaps = _normalize_description_gaps(gaps)
    if not normalized_gaps and transcript_segments:
        normalized_gaps = _gaps_from_transcript(
            transcript_segments,
            min_gap_seconds,
            max_gap_seconds,
        )
    if not normalized_gaps and video_path and os.path.isfile(video_path):
        normalized_gaps = find_description_gaps(
            video_path,
            transcript=transcript_segments,
            min_gap_seconds=min_gap_seconds,
            max_gap_seconds=max_gap_seconds,
            on_progress=None,
        )

    notes: List[str] = []
    if not normalized_gaps:
        notes.append("No dialogue gaps were available; cues need editor timing.")

    if on_progress:
        on_progress(60, "Assigning scene descriptions to review cues...")

    cues: List[AudioDescriptionCue] = []
    used_gaps: set[int] = set()
    for idx, scene in enumerate(scenes):
        scene_start = float(scene["timestamp"])
        if idx + 1 < len(scenes):
            scene_end = max(scene_start, float(scenes[idx + 1]["timestamp"]))
        else:
            scene_end = scene_start + 5.0

        gap_idx, gap = _choose_gap_for_scene(scene_start, scene_end, normalized_gaps, used_gaps)
        if gap_idx is not None:
            used_gaps.add(gap_idx)

        description = _clean_ad_text(scene["description"])
        if gap is not None:
            target_start = gap.start
            target_end = gap.end
            available = max(0.0, gap.duration or gap.end - gap.start)
            max_words = max(1, int(available * words_per_second))
        else:
            target_start = scene_start
            target_end = scene_start
            available = 0.0
            max_words = max(3, int(words_per_second * 2))

        script_budget = len(description.split()) if extended_timing else max_words
        script = _cap_words(description, script_budget)
        estimated_duration = round(len(script.split()) / words_per_second, 3) if script else 0.0
        fits_gap = bool(gap is not None and estimated_duration <= available)
        timing_mode = "standard"
        extended_pause_seconds = 0.0
        if gap is None:
            priority = "needs-timing"
            review_reason = "no-dialogue-gap"
        elif fits_gap:
            priority = _clean_ad_text(scene.get("importance", "normal")) or "normal"
            review_reason = "draft-ai-description"
        else:
            priority = "needs-rewrite"
            review_reason = "description-may-overrun-gap"
        if extended_timing and (gap is None or not fits_gap):
            timing_mode = "extended"
            extended_pause_seconds = max(0.0, estimated_duration - available)
            target_end = round(target_start + max(estimated_duration, available), 3)
            priority = "extended-ad-review"
            review_reason = "extended-audio-description"

        context, context_segments = _dialogue_context(
            transcript_segments,
            scene_start,
            scene_end,
            context_seconds,
        )
        cues.append(AudioDescriptionCue(
            cue_id=f"AD-{idx + 1:04d}",
            scene_index=idx,
            scene_start=round(scene_start, 3),
            scene_end=round(scene_end, 3),
            target_start=round(target_start, 3),
            target_end=round(target_end, 3),
            available_gap_seconds=round(available, 3),
            max_words=max_words,
            estimated_duration=estimated_duration,
            fits_gap=fits_gap,
            script=script,
            source_description=description,
            dialogue_context=context,
            transcript_segments=context_segments,
            priority=priority,
            needs_review=True,
            review_reason=review_reason,
            tts_backend_hint=tts_backend_hint,
            timing_mode=timing_mode,
            extended_pause_seconds=round(extended_pause_seconds, 3),
            descriptive_transcript_text=script,
        ))

    unplaced_gap_count = max(0, len(normalized_gaps) - len(used_gaps))
    if unplaced_gap_count:
        notes.append(f"{unplaced_gap_count} dialogue gaps were left unused.")

    if on_progress:
        on_progress(100, f"Prepared {len(cues)} audio-description cues")

    descriptive_transcript = (
        _build_descriptive_transcript(cues, transcript_segments)
        if include_descriptive_transcript
        else []
    )
    extended_timing_plan = _build_extended_timing_plan(cues) if extended_timing else []
    wcag3_compatibility = {}
    if include_wcag3_hooks:
        wcag3_compatibility = {
            "draft": "WCAG 3.0 Working Draft media alternatives hooks",
            "status": "draft-not-normative",
            "descriptive_transcript_available": bool(descriptive_transcript),
            "extended_audio_description_available": bool(extended_timing_plan),
            "review_required": True,
            "source_urls": [
                "https://www.w3.org/TR/wcag-3.0/",
                "https://www.w3.org/WAI/media/av/transcripts/",
                "https://www.w3.org/WAI/media/av/description/",
            ],
        }

    return AudioDescriptionReviewDraft(
        cues=cues,
        transcript_segments=transcript_segments,
        gaps=normalized_gaps,
        tts_backend_hint=tts_backend_hint,
        notes=notes,
        wcag3_compatibility=wcag3_compatibility,
        descriptive_transcript=descriptive_transcript,
        extended_timing_plan=extended_timing_plan,
    )


# ---------------------------------------------------------------------------
# Speech Synthesis
# ---------------------------------------------------------------------------
def synthesize_description(
    text: str,
    voice: str = "default",
    output_path_val: Optional[str] = None,
    speed: float = 1.1,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Synthesize audio description speech from text.

    Uses available TTS engine (pyttsx3 as fallback, or system TTS).
    Audio descriptions are typically spoken slightly faster than normal
    to fit within dialogue pauses.

    Args:
        text: Description text to synthesize.
        voice: Voice identifier (engine-dependent).
        output_path_val: Output audio path (auto-generated if None).
        speed: Speech rate multiplier (1.1 = slightly fast for AD).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to synthesized audio file.
    """
    if on_progress:
        on_progress(10, "Synthesizing description audio...")

    if output_path_val is None:
        output_path_val = os.path.join(
            tempfile.gettempdir(),
            f"ad_synth_{hash(text) & 0xFFFFFF:06x}.wav",
        )

    ffmpeg = get_ffmpeg_path()

    # Try pyttsx3 first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", int(150 * speed))

        if voice != "default":
            voices = engine.getProperty("voices")
            for v in voices:
                if voice.lower() in v.name.lower() or voice == v.id:
                    engine.setProperty("voice", v.id)
                    break

        engine.save_to_file(text, output_path_val)
        engine.runAndWait()

        if on_progress:
            on_progress(100, "Description audio synthesized")
        return output_path_val

    except (ImportError, Exception) as e:
        logger.debug("pyttsx3 not available: %s, using FFmpeg sine fallback", e)

    # Fallback: generate a placeholder tone (sine wave beep)
    # In production this would use a proper TTS API
    duration = max(1.0, len(text.split()) / (3 * speed))

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y",
        "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={duration:.2f}",
        "-af", f"volume=0.3,atempo={speed}",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=60)

    if on_progress:
        on_progress(100, "Description audio synthesized (placeholder)")

    return output_path_val


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def generate_audio_description(
    video_path: str,
    output_path_val: Optional[str] = None,
    descriptions: Optional[List[Dict]] = None,
    transcript: Optional[List[Dict]] = None,
    voice: str = "default",
    min_gap_seconds: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> ADResult:
    """
    Generate a complete audio description track for a video.

    Pipeline:
    1. Find gaps in dialogue
    2. Generate/use provided descriptions for visual content
    3. Synthesize description speech
    4. Mix descriptions into AD audio track

    Args:
        video_path: Source video file.
        output_path_val: Output path (auto-generated if None).
        descriptions: Pre-written descriptions (list of dicts with
            timestamp, text, duration).
        transcript: Transcript for gap detection.
        voice: TTS voice identifier.
        min_gap_seconds: Minimum gap for descriptions.
        on_progress: Progress callback(pct, msg).

    Returns:
        ADResult with output path and statistics.
    """
    if output_path_val is None:
        output_path_val = output_path(video_path, "_audio_described")

    ffmpeg = get_ffmpeg_path()
    temp_files = []
    result = ADResult()

    try:
        # Step 1: Find gaps
        if on_progress:
            on_progress(5, "Finding description gaps...")

        gaps = find_description_gaps(
            video_path,
            transcript=transcript,
            min_gap_seconds=min_gap_seconds,
            on_progress=None,
        )
        result.gaps_found = len(gaps)

        # Step 2: Match descriptions to gaps
        if on_progress:
            on_progress(20, "Preparing descriptions...")

        desc_audio_pairs = []  # (timestamp, audio_path)

        if descriptions:
            # Use provided descriptions
            for i, desc in enumerate(descriptions):
                ts = float(desc.get("timestamp", 0))
                text = desc.get("text", "")
                if not text:
                    continue

                # Descriptions are scheduled by their explicit ``timestamp``
                # regardless of whether they line up with a detected silent
                # gap — the previous implementation searched ``gaps`` for a
                # match but then guarded the synthesis with
                # ``if best_gap or True:``, which always entered the branch.
                # Skipping the lookup keeps the same effective behavior with
                # less noise.
                synth_path = os.path.join(
                    tempfile.gettempdir(), f"ad_desc_{i:04d}.wav",
                )
                temp_files.append(synth_path)

                synthesize_description(
                    text, voice=voice,
                    output_path_val=synth_path,
                    on_progress=None,
                )
                desc_audio_pairs.append((ts, synth_path))

                if on_progress:
                    pct = 20 + int(40 * (i + 1) / len(descriptions))
                    on_progress(pct, f"Synthesized {i + 1}/{len(descriptions)}")
        else:
            # Auto-describe at gap positions
            for i, gap in enumerate(gaps[:20]):  # Limit to 20 descriptions
                desc = describe_visual_content(video_path, gap.start, on_progress=None)
                synth_path = os.path.join(
                    tempfile.gettempdir(), f"ad_auto_{i:04d}.wav",
                )
                temp_files.append(synth_path)

                synthesize_description(
                    desc.description, voice=voice,
                    output_path_val=synth_path,
                    on_progress=None,
                )
                desc_audio_pairs.append((gap.start, synth_path))

        result.descriptions_added = len(desc_audio_pairs)

        if not desc_audio_pairs:
            # No descriptions to add, just copy original
            if on_progress:
                on_progress(90, "No descriptions to add, copying original...")

            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-y", "-i", video_path, "-c", "copy",
                output_path_val,
            ]
            run_ffmpeg(cmd, timeout=600)
            result.output_path = output_path_val
            return result

        # Step 3: Mix descriptions into audio
        if on_progress:
            on_progress(65, "Mixing descriptions into audio track...")

        result = _mix_descriptions(
            video_path, desc_audio_pairs, output_path_val, result, temp_files,
        )

        if on_progress:
            on_progress(100, f"Audio description complete: {result.descriptions_added} descriptions")

        return result

    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except OSError:
                    pass


def _mix_descriptions(
    video_path: str,
    desc_pairs: List,
    output_path_val: str,
    result: ADResult,
    temp_files: List[str],
) -> ADResult:
    """Mix description audio clips into the video's audio track."""
    ffmpeg = get_ffmpeg_path()

    # Build FFmpeg command with multiple inputs and amix
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    cmd.extend(["-i", video_path])

    for _, audio_path in desc_pairs:
        cmd.extend(["-i", audio_path])

    # Build filter complex to delay and mix each description
    n_desc = len(desc_pairs)
    filter_parts = []
    mix_inputs = ["[0:a]"]

    for i, (timestamp, _) in enumerate(desc_pairs):
        delay_ms = int(timestamp * 1000)
        filter_parts.append(
            f"[{i + 1}:a]adelay={delay_ms}|{delay_ms},"
            f"volume=0.9[d{i}]"
        )
        mix_inputs.append(f"[d{i}]")

    # Mix all streams
    mix_str = "".join(mix_inputs)
    filter_parts.append(
        f"{mix_str}amix=inputs={n_desc + 1}:duration=longest:dropout_transition=2[outa]"
    )

    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path_val,
    ])

    run_ffmpeg(cmd, timeout=3600)

    result.output_path = output_path_val
    result.total_description_duration = sum(
        3.0 for _ in desc_pairs  # approximate
    )
    return result
