"""
OpenCut SDH (Subtitles for the Deaf and Hard-of-Hearing) Formatter

Auto-apply SDH conventions: speaker identification, non-speech sound
descriptions, music notation, and emotional tone markers. Classifies
audio segments based on energy profile and spectral characteristics.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Sound event library
# ---------------------------------------------------------------------------
SOUND_EVENTS: Dict[str, str] = {
    "door_slam": "[door slams]",
    "door_knock": "[knocking on door]",
    "door_close": "[door closes]",
    "phone_ring": "[phone rings]",
    "phone_buzz": "[phone buzzing]",
    "phone_notification": "[phone notification]",
    "glass_break": "[glass shatters]",
    "glass_clink": "[glasses clink]",
    "gunshot": "[gunshot]",
    "explosion": "[explosion]",
    "car_horn": "[car horn honking]",
    "car_engine": "[engine revving]",
    "car_screech": "[tires screeching]",
    "siren": "[sirens wailing]",
    "thunder": "[thunder rumbling]",
    "rain": "[rain falling]",
    "wind": "[wind howling]",
    "footsteps": "[footsteps]",
    "applause": "[applause]",
    "laughter": "[laughter]",
    "crying": "[crying]",
    "scream": "[screaming]",
    "whisper": "[whispering]",
    "gasp": "[gasps]",
    "sigh": "[sighs]",
    "cough": "[coughs]",
    "sneeze": "[sneezes]",
    "heartbeat": "[heartbeat]",
    "bell": "[bell rings]",
    "alarm": "[alarm blaring]",
    "keyboard": "[keyboard typing]",
    "water": "[water running]",
    "splash": "[splash]",
    "fire": "[fire crackling]",
    "crowd": "[crowd murmuring]",
    "birds": "[birds chirping]",
    "dog_bark": "[dog barking]",
    "music_playing": "[music playing]",
    "silence": "[silence]",
}

# Emotional tone markers
TONE_MARKERS: Dict[str, str] = {
    "whispering": "[whispering]",
    "shouting": "[shouting]",
    "crying": "[tearfully]",
    "laughing": "[laughing]",
    "angry": "[angrily]",
    "sarcastic": "[sarcastically]",
    "nervous": "[nervously]",
    "excited": "[excitedly]",
    "sad": "[sadly]",
    "fearful": "[fearfully]",
    "surprised": "[surprised]",
    "disgusted": "[with disgust]",
    "confused": "[confused]",
    "relieved": "[with relief]",
    "muffled": "[muffled]",
    "echoing": "[echoing]",
    "distorted": "[distorted]",
    "on_phone": "[over phone]",
    "on_radio": "[over radio]",
    "on_tv": "[on TV]",
    "voiceover": "[voiceover]",
    "narrating": "[narrating]",
}


@dataclass
class SDHSegment:
    """A subtitle segment with SDH formatting applied."""

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    original_text: str = ""
    formatted_text: str = ""
    speaker: str = ""
    sound_events: List[str] = field(default_factory=list)
    is_music: bool = False
    tone: str = ""


@dataclass
class SDHResult:
    """Result of SDH formatting."""

    formatted_subtitles: List[SDHSegment] = field(default_factory=list)
    speaker_labels_added: int = 0
    sound_events_added: int = 0
    music_segments_marked: int = 0
    tone_markers_added: int = 0
    total_segments: int = 0


@dataclass
class SDHConfig:
    """Configuration for SDH formatting."""

    uppercase_speakers: bool = True
    bracket_style: str = "square"  # square, round, angle
    music_symbol: str = "\u266a"  # Music note
    speaker_separator: str = ":"
    include_sound_events: bool = True
    include_music_notation: bool = True
    include_tone_markers: bool = True
    max_speaker_length: int = 30
    sound_event_position: str = "before"  # before, after, inline


# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------
def _format_speaker_label(
    speaker: str,
    config: SDHConfig,
) -> str:
    """Format a speaker identification label.

    Args:
        speaker: Speaker name or identifier.
        config: SDH formatting configuration.

    Returns:
        Formatted speaker label string.
    """
    if not speaker:
        return ""
    speaker = speaker.strip()
    if len(speaker) > config.max_speaker_length:
        speaker = speaker[:config.max_speaker_length]
    if config.uppercase_speakers:
        speaker = speaker.upper()
    return f"{speaker}{config.speaker_separator} "


def _apply_speaker_labels(
    segments: List[SDHSegment],
    diarization: Optional[List[Dict]],
    config: SDHConfig,
) -> int:
    """Apply speaker identification labels to segments.

    Args:
        segments: SDH segments to modify in place.
        diarization: List of diarization entries with 'start', 'end',
            'speaker' keys.
        config: SDH configuration.

    Returns:
        Number of speaker labels added.
    """
    count = 0
    for seg in segments:
        if seg.speaker:
            # Speaker already set from subtitle data
            label = _format_speaker_label(seg.speaker, config)
            if label:
                seg.formatted_text = f"{label}{seg.formatted_text}"
                count += 1
            continue

        if not diarization:
            continue

        # Find matching diarization entry by overlap
        best_speaker = ""
        best_overlap = 0.0
        for diar in diarization:
            d_start = float(diar.get("start", 0))
            d_end = float(diar.get("end", 0))
            overlap_start = max(seg.start, d_start)
            overlap_end = min(seg.end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(diar.get("speaker", ""))

        if best_speaker:
            seg.speaker = best_speaker
            label = _format_speaker_label(best_speaker, config)
            if label:
                seg.formatted_text = f"{label}{seg.formatted_text}"
                count += 1

    return count


# ---------------------------------------------------------------------------
# Sound event detection and insertion
# ---------------------------------------------------------------------------
def _classify_audio_segment(
    energy: float,
    spectral_centroid: float,
    spectral_bandwidth: float,
    zero_crossing_rate: float,
) -> str:
    """Classify an audio segment based on spectral characteristics.

    Uses simplified heuristics based on energy profile and spectral
    features to distinguish speech, music, and sound effects.

    Args:
        energy: RMS energy of the segment (0-1 normalized).
        spectral_centroid: Frequency centroid in Hz.
        spectral_bandwidth: Spectral bandwidth in Hz.
        zero_crossing_rate: Zero crossing rate (0-1).

    Returns:
        Classification: 'speech', 'music', 'sound_effect', or 'silence'.
    """
    if energy < 0.01:
        return "silence"

    # High ZCR + moderate energy + moderate bandwidth = speech
    if zero_crossing_rate > 0.1 and 500 < spectral_centroid < 4000:
        if spectral_bandwidth < 3000:
            return "speech"

    # Moderate energy + low ZCR + wide bandwidth = music
    if zero_crossing_rate < 0.15 and spectral_bandwidth > 2000:
        if 200 < spectral_centroid < 8000:
            return "music"

    # High energy + high centroid + narrow bandwidth = sound effect
    if energy > 0.3 and spectral_centroid > 3000:
        return "sound_effect"

    # Default: if speech-like centroid range, assume speech
    if 300 < spectral_centroid < 4000:
        return "speech"

    return "sound_effect"


def _detect_sound_events(
    audio_events: Optional[List[Dict]],
) -> Dict[float, List[str]]:
    """Map timestamps to detected sound event descriptions.

    Args:
        audio_events: List of audio event dicts with 'start', 'end',
            'type' keys where type is a key from SOUND_EVENTS.

    Returns:
        Dict mapping start timestamps to event description lists.
    """
    if not audio_events:
        return {}

    result: Dict[float, List[str]] = {}
    for event in audio_events:
        start = float(event.get("start", 0))
        event_type = str(event.get("type", "")).lower().strip()
        custom_desc = str(event.get("description", "")).strip()

        if custom_desc:
            desc = f"[{custom_desc}]"
        elif event_type in SOUND_EVENTS:
            desc = SOUND_EVENTS[event_type]
        else:
            desc = f"[{event_type}]" if event_type else None

        if desc:
            result.setdefault(start, []).append(desc)

    return result


def _apply_sound_events(
    segments: List[SDHSegment],
    audio_events: Optional[List[Dict]],
    config: SDHConfig,
) -> int:
    """Insert sound event descriptions into segments.

    Returns number of sound events added.
    """
    if not config.include_sound_events or not audio_events:
        return 0

    event_map = _detect_sound_events(audio_events)
    if not event_map:
        return 0

    count = 0
    for seg in segments:
        matching_events: List[str] = []
        for ts, descs in event_map.items():
            if seg.start <= ts < seg.end:
                matching_events.extend(descs)

        if not matching_events:
            continue

        seg.sound_events = matching_events
        events_str = " ".join(matching_events)

        if config.sound_event_position == "before":
            seg.formatted_text = f"{events_str}\n{seg.formatted_text}"
        elif config.sound_event_position == "after":
            seg.formatted_text = f"{seg.formatted_text}\n{events_str}"
        else:  # inline
            seg.formatted_text = f"{events_str} {seg.formatted_text}"

        count += len(matching_events)

    return count


# ---------------------------------------------------------------------------
# Music notation
# ---------------------------------------------------------------------------
def _detect_music_segments(
    segments: List[SDHSegment],
    stem_metadata: Optional[List[Dict]],
) -> int:
    """Mark segments as music based on stem separation metadata.

    If vocals are absent in a segment's time range, it is considered
    instrumental music.

    Args:
        segments: SDH segments to check.
        stem_metadata: List of stem analysis entries with 'start', 'end',
            'has_vocals', 'has_music' keys.

    Returns:
        Number of music segments marked.
    """
    if not stem_metadata:
        return 0

    count = 0
    for seg in segments:
        for stem in stem_metadata:
            s_start = float(stem.get("start", 0))
            s_end = float(stem.get("end", 0))
            overlap_start = max(seg.start, s_start)
            overlap_end = min(seg.end, s_end)
            overlap = overlap_end - overlap_start
            if overlap <= 0:
                continue
            has_vocals = bool(stem.get("has_vocals", True))
            has_music = bool(stem.get("has_music", False))
            if has_music and not has_vocals:
                seg.is_music = True
                count += 1
                break

    return count


def _apply_music_notation(
    segments: List[SDHSegment],
    config: SDHConfig,
) -> int:
    """Apply music notation symbols to music segments.

    Returns number of segments formatted with music notation.
    """
    if not config.include_music_notation:
        return 0

    symbol = config.music_symbol
    count = 0
    for seg in segments:
        if seg.is_music:
            text = seg.formatted_text.strip()
            if text:
                seg.formatted_text = f"{symbol} {text} {symbol}"
            else:
                seg.formatted_text = f"{symbol} [music playing] {symbol}"
            count += 1

    return count


# ---------------------------------------------------------------------------
# Tone markers
# ---------------------------------------------------------------------------
def _apply_tone_markers(
    segments: List[SDHSegment],
    tone_annotations: Optional[List[Dict]],
    config: SDHConfig,
) -> int:
    """Apply emotional tone markers to segments.

    Args:
        segments: SDH segments to modify.
        tone_annotations: List of tone annotation dicts with 'start',
            'end', 'tone' keys.
        config: SDH configuration.

    Returns:
        Number of tone markers added.
    """
    if not config.include_tone_markers or not tone_annotations:
        return 0

    count = 0
    for seg in segments:
        for ann in tone_annotations:
            a_start = float(ann.get("start", 0))
            a_end = float(ann.get("end", 0))
            overlap_start = max(seg.start, a_start)
            overlap_end = min(seg.end, a_end)
            if overlap_end - overlap_start <= 0:
                continue

            tone_key = str(ann.get("tone", "")).lower().strip()
            if tone_key in TONE_MARKERS:
                marker = TONE_MARKERS[tone_key]
            elif tone_key:
                marker = f"[{tone_key}]"
            else:
                continue

            seg.tone = tone_key
            seg.formatted_text = f"{marker} {seg.formatted_text}"
            count += 1
            break  # One tone marker per segment

    return count


# ---------------------------------------------------------------------------
# Bracket style normalization
# ---------------------------------------------------------------------------
def _normalize_brackets(text: str, style: str) -> str:
    """Normalize bracket style in formatted text.

    Args:
        text: Text containing brackets.
        style: Bracket style (square, round, angle).

    Returns:
        Text with normalized brackets.
    """
    if style == "round":
        text = text.replace("[", "(").replace("]", ")")
    elif style == "angle":
        text = text.replace("[", "<").replace("]", ">")
    return text


# ---------------------------------------------------------------------------
# Main formatting function
# ---------------------------------------------------------------------------
def format_sdh(
    subtitles: List[Dict],
    diarization: Optional[List[Dict]] = None,
    audio_events: Optional[List[Dict]] = None,
    stem_metadata: Optional[List[Dict]] = None,
    tone_annotations: Optional[List[Dict]] = None,
    config: Optional[SDHConfig] = None,
    on_progress: Optional[Callable] = None,
) -> SDHResult:
    """Apply SDH formatting to subtitle segments.

    Args:
        subtitles: List of subtitle dicts with 'start', 'end', 'text' keys.
        diarization: Speaker diarization data with 'start', 'end', 'speaker'.
        audio_events: Detected audio events with 'start', 'end', 'type'.
        stem_metadata: Stem separation results with 'start', 'end',
            'has_vocals', 'has_music'.
        tone_annotations: Emotional tone annotations with 'start', 'end', 'tone'.
        config: SDH formatting configuration (uses defaults if None).
        on_progress: Progress callback (percentage int).

    Returns:
        SDHResult with formatted subtitles and statistics.
    """
    if config is None:
        config = SDHConfig()

    if not subtitles:
        return SDHResult()

    if on_progress:
        on_progress(5)

    # Build SDH segments
    segments: List[SDHSegment] = []
    for i, sub in enumerate(subtitles):
        speaker = str(sub.get("speaker", "")).strip()
        segments.append(SDHSegment(
            index=i + 1,
            start=float(sub.get("start", 0)),
            end=float(sub.get("end", 0)),
            original_text=str(sub.get("text", "")),
            formatted_text=str(sub.get("text", "")),
            speaker=speaker,
        ))

    if on_progress:
        on_progress(15)

    # Step 1: Detect music segments
    music_count = _detect_music_segments(segments, stem_metadata)
    if on_progress:
        on_progress(30)

    # Step 2: Apply speaker labels
    speaker_count = _apply_speaker_labels(segments, diarization, config)
    if on_progress:
        on_progress(50)

    # Step 3: Apply sound events
    event_count = _apply_sound_events(segments, audio_events, config)
    if on_progress:
        on_progress(65)

    # Step 4: Apply music notation
    music_formatted = _apply_music_notation(segments, config)
    if on_progress:
        on_progress(75)

    # Step 5: Apply tone markers
    tone_count = _apply_tone_markers(segments, tone_annotations, config)
    if on_progress:
        on_progress(85)

    # Step 6: Normalize bracket style
    if config.bracket_style != "square":
        for seg in segments:
            seg.formatted_text = _normalize_brackets(
                seg.formatted_text, config.bracket_style,
            )
    if on_progress:
        on_progress(95)

    # Clean up whitespace
    for seg in segments:
        seg.formatted_text = re.sub(
            r"\n{3,}", "\n\n", seg.formatted_text.strip(),
        )

    if on_progress:
        on_progress(100)

    return SDHResult(
        formatted_subtitles=segments,
        speaker_labels_added=speaker_count,
        sound_events_added=event_count,
        music_segments_marked=music_count + music_formatted,
        tone_markers_added=tone_count,
        total_segments=len(segments),
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_sdh_srt(result: SDHResult) -> str:
    """Export SDH-formatted subtitles as SRT string."""
    lines: List[str] = []
    for seg in result.formatted_subtitles:
        lines.append(str(seg.index))
        start_ts = _seconds_to_srt(seg.start)
        end_ts = _seconds_to_srt(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.formatted_text)
        lines.append("")
    return "\n".join(lines)


def export_sdh_vtt(result: SDHResult) -> str:
    """Export SDH-formatted subtitles as WebVTT string."""
    lines: List[str] = ["WEBVTT", ""]
    for seg in result.formatted_subtitles:
        start_ts = _seconds_to_vtt(seg.start)
        end_ts = _seconds_to_vtt(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.formatted_text)
        lines.append("")
    return "\n".join(lines)


def _seconds_to_srt(s: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def _seconds_to_vtt(s: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def list_sound_events() -> List[Dict]:
    """Return list of known sound event types and descriptions."""
    return [
        {"type": k, "description": v}
        for k, v in sorted(SOUND_EVENTS.items())
    ]


def list_tone_markers() -> List[Dict]:
    """Return list of emotional tone marker types."""
    return [
        {"tone": k, "marker": v}
        for k, v in sorted(TONE_MARKERS.items())
    ]
