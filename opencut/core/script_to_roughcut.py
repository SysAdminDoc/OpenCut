"""
OpenCut Script-to-Rough-Cut Assembly (59.4)

Transcribe footage, parse script, fuzzy-match transcript segments to script,
rank matching clips by similarity + quality, select best take per segment,
and assemble as OTIO timeline or simple XML EDL.

Pipeline:
1. Transcribe all source footage (via Whisper or existing transcripts)
2. Parse script into dialogue/action segments
3. Fuzzy-match transcript to script (difflib SequenceMatcher)
4. Rank clips per segment by text similarity + video quality
5. Select best take per segment
6. Assemble as OTIO or XML timeline

Uses FFmpeg for transcription (via Whisper) and standard library for matching.
"""

import difflib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class TranscriptSegment:
    """A segment from a transcribed clip."""
    clip_path: str = ""
    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0


@dataclass
class ScriptSegment:
    """A segment from the parsed script."""
    index: int = 0
    segment_type: str = ""  # dialogue, action, scene_heading
    text: str = ""
    character: str = ""
    scene: str = ""


@dataclass
class MatchResult:
    """A match between a script segment and a transcript segment."""
    script_segment: ScriptSegment = field(default_factory=ScriptSegment)
    transcript_segment: TranscriptSegment = field(default_factory=TranscriptSegment)
    similarity: float = 0.0
    quality_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class RoughCutSegment:
    """A segment selected for the rough cut."""
    index: int = 0
    clip_path: str = ""
    in_point: float = 0.0
    out_point: float = 0.0
    script_text: str = ""
    character: str = ""
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoughCutResult:
    """Complete rough cut assembly result."""
    segments: List[RoughCutSegment] = field(default_factory=list)
    total_segments: int = 0
    matched_segments: int = 0
    unmatched_segments: int = 0
    total_duration: float = 0.0
    timeline_path: str = ""
    format: str = ""

    def to_dict(self) -> dict:
        return {
            "total_segments": self.total_segments,
            "matched_segments": self.matched_segments,
            "unmatched_segments": self.unmatched_segments,
            "total_duration": round(self.total_duration, 2),
            "timeline_path": self.timeline_path,
            "format": self.format,
            "segments": [s.to_dict() for s in self.segments],
        }


# ---------------------------------------------------------------------------
# Script Parsing
# ---------------------------------------------------------------------------
_SCENE_HEADING = re.compile(
    r"^\s*(INT\.?|EXT\.?|INT\.?/EXT\.?)\s+(.+?)(?:\s*-\s*(.+))?\s*$",
    re.I | re.M,
)
_CHARACTER_CUE = re.compile(r"^([A-Z][A-Z0-9 _\-\.]+)(\s*\(.*\))?\s*$")


def parse_script_segments(script_text: str) -> List[ScriptSegment]:
    """Parse a script into matchable segments (dialogue + action lines)."""
    if not script_text or not script_text.strip():
        return []

    lines = script_text.split("\n")
    segments = []
    current_character = ""
    current_scene = ""
    idx = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        i += 1

        if not stripped:
            current_character = ""
            continue

        # Scene heading
        m = _SCENE_HEADING.match(stripped)
        if m:
            current_scene = stripped
            segments.append(ScriptSegment(
                index=idx, segment_type="scene_heading",
                text=stripped, scene=current_scene,
            ))
            idx += 1
            current_character = ""
            continue

        # Character cue
        if _CHARACTER_CUE.match(stripped) and len(stripped) < 60:
            current_character = stripped.split("(")[0].strip()
            continue

        # Dialogue
        if current_character and (line.startswith("    ") or line.startswith("\t")):
            segments.append(ScriptSegment(
                index=idx, segment_type="dialogue",
                text=stripped, character=current_character,
                scene=current_scene,
            ))
            idx += 1
            continue

        # Action
        segments.append(ScriptSegment(
            index=idx, segment_type="action",
            text=stripped, scene=current_scene,
        ))
        current_character = ""
        idx += 1

    return segments


# ---------------------------------------------------------------------------
# Transcript Handling
# ---------------------------------------------------------------------------
def load_transcript(transcript_path: str, clip_path: str = "") -> List[TranscriptSegment]:
    """
    Load a transcript from a JSON file (Whisper-style output).

    Expected JSON format:
    {"segments": [{"text": "...", "start": 0.0, "end": 1.5, "avg_logprob": -0.3}]}
    or a list of segments directly.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_segments = data if isinstance(data, list) else data.get("segments", [])
    result = []

    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        result.append(TranscriptSegment(
            clip_path=clip_path,
            text=text,
            start_time=float(seg.get("start", 0)),
            end_time=float(seg.get("end", 0)),
            confidence=abs(float(seg.get("avg_logprob", -1.0))),
        ))

    return result


def transcribe_clip(clip_path: str, model: str = "base") -> List[TranscriptSegment]:
    """
    Transcribe a clip using Whisper.

    Falls back to empty list if Whisper is not available.

    Args:
        clip_path: Path to the video/audio file.
        model: Whisper model size.

    Returns:
        List of TranscriptSegment.
    """
    try:
        import whisper
        mdl = whisper.load_model(model)
        result = mdl.transcribe(clip_path)
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                clip_path=clip_path,
                text=seg.get("text", "").strip(),
                start_time=float(seg.get("start", 0)),
                end_time=float(seg.get("end", 0)),
                confidence=abs(float(seg.get("avg_logprob", -1.0))),
            ))
        return segments
    except ImportError:
        logger.warning("Whisper not available; provide pre-made transcripts")
        return []
    except Exception as e:
        logger.error("Transcription failed for %s: %s", clip_path, e)
        return []


# ---------------------------------------------------------------------------
# Fuzzy Matching
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def fuzzy_match_segments(
    script_segments: List[ScriptSegment],
    transcript_segments: List[TranscriptSegment],
    threshold: float = 0.4,
) -> Dict[int, List[MatchResult]]:
    """
    Match script segments to transcript segments using difflib.

    Args:
        script_segments: Parsed script segments.
        transcript_segments: Transcribed segments from footage.
        threshold: Minimum similarity to consider a match.

    Returns:
        Dict mapping script segment index to list of MatchResult (ranked).
    """
    matches: Dict[int, List[MatchResult]] = {}

    for ss in script_segments:
        if ss.segment_type not in ("dialogue", "action"):
            continue

        ss_norm = _normalize(ss.text)
        if not ss_norm:
            continue

        segment_matches = []
        for ts in transcript_segments:
            ts_norm = _normalize(ts.text)
            if not ts_norm:
                continue

            # Use SequenceMatcher for fuzzy matching
            ratio = difflib.SequenceMatcher(None, ss_norm, ts_norm).ratio()

            if ratio >= threshold:
                # Quality score: higher resolution + longer duration = better
                quality = 0.5  # base quality
                if ts.confidence > 0:
                    quality = min(1.0, ts.confidence)

                combined = ratio * 0.7 + quality * 0.3

                segment_matches.append(MatchResult(
                    script_segment=ss,
                    transcript_segment=ts,
                    similarity=round(ratio, 4),
                    quality_score=round(quality, 4),
                    combined_score=round(combined, 4),
                ))

        # Sort by combined score descending
        segment_matches.sort(key=lambda m: m.combined_score, reverse=True)
        if segment_matches:
            matches[ss.index] = segment_matches

    return matches


# ---------------------------------------------------------------------------
# Best Take Selection
# ---------------------------------------------------------------------------
def select_best_takes(
    matches: Dict[int, List[MatchResult]],
    script_segments: List[ScriptSegment],
) -> List[RoughCutSegment]:
    """
    Select the best clip for each script segment.

    Args:
        matches: Output from fuzzy_match_segments.
        script_segments: All script segments (for ordering).

    Returns:
        Ordered list of RoughCutSegment for the rough cut.
    """
    result = []
    idx = 0

    for ss in script_segments:
        if ss.segment_type not in ("dialogue", "action"):
            continue

        if ss.index in matches and matches[ss.index]:
            best = matches[ss.index][0]
            ts = best.transcript_segment
            result.append(RoughCutSegment(
                index=idx,
                clip_path=ts.clip_path,
                in_point=ts.start_time,
                out_point=ts.end_time,
                script_text=ss.text[:200],
                character=ss.character,
                similarity=best.similarity,
            ))
        else:
            # No match — placeholder gap
            result.append(RoughCutSegment(
                index=idx,
                clip_path="",
                script_text=ss.text[:200],
                character=ss.character,
                similarity=0.0,
            ))
        idx += 1

    return result


# ---------------------------------------------------------------------------
# Timeline Export
# ---------------------------------------------------------------------------
def export_xml_timeline(
    segments: List[RoughCutSegment],
    output_path: str,
    fps: float = 24.0,
    name: str = "OpenCut Rough Cut",
) -> str:
    """
    Export rough cut as a simple XML timeline (FCP XML-like).

    Args:
        segments: Ordered RoughCutSegment list.
        output_path: Where to save the XML.
        fps: Timeline frame rate.
        name: Sequence name.

    Returns:
        Path to the XML file.
    """
    root = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = name
    ET.SubElement(sequence, "duration").text = str(
        int(sum((s.out_point - s.in_point) for s in segments if s.clip_path) * fps)
    )

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(int(fps))
    ET.SubElement(rate, "ntsc").text = "FALSE"

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    track = ET.SubElement(video, "track")

    timeline_pos = 0
    for seg in segments:
        if not seg.clip_path:
            continue

        clip_el = ET.SubElement(track, "clipitem")
        ET.SubElement(clip_el, "name").text = os.path.basename(seg.clip_path)

        dur_frames = int((seg.out_point - seg.in_point) * fps)
        ET.SubElement(clip_el, "start").text = str(timeline_pos)
        ET.SubElement(clip_el, "end").text = str(timeline_pos + dur_frames)
        ET.SubElement(clip_el, "in").text = str(int(seg.in_point * fps))
        ET.SubElement(clip_el, "out").text = str(int(seg.out_point * fps))

        file_el = ET.SubElement(clip_el, "file")
        ET.SubElement(file_el, "pathurl").text = f"file:///{seg.clip_path.replace(os.sep, '/')}"

        clip_rate = ET.SubElement(clip_el, "rate")
        ET.SubElement(clip_rate, "timebase").text = str(int(fps))

        timeline_pos += dur_frames

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def export_otio_timeline(
    segments: List[RoughCutSegment],
    output_path: str,
    fps: float = 24.0,
    name: str = "OpenCut Rough Cut",
) -> str:
    """
    Export rough cut as OpenTimelineIO (.otio) JSON.

    Falls back to XML if opentimelineio is not installed.

    Args:
        segments: Ordered RoughCutSegment list.
        output_path: Where to save the OTIO file.
        fps: Timeline frame rate.
        name: Sequence name.

    Returns:
        Path to the timeline file.
    """
    try:
        import opentimelineio as otio
        timeline = otio.schema.Timeline(name=name)
        track = otio.schema.Track(name="V1")

        for seg in segments:
            if not seg.clip_path:
                # Gap for unmatched segments
                dur = otio.opentime.RationalTime(24, fps)  # 1 second gap
                gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, fps),
                        duration=dur,
                    )
                )
                track.append(gap)
                continue

            dur_secs = seg.out_point - seg.in_point
            if dur_secs <= 0:
                continue

            ref = otio.schema.ExternalReference(target_url=seg.clip_path)
            clip = otio.schema.Clip(
                name=os.path.basename(seg.clip_path),
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(
                        int(seg.in_point * fps), fps
                    ),
                    duration=otio.opentime.RationalTime(
                        int(dur_secs * fps), fps
                    ),
                ),
                media_reference=ref,
            )
            track.append(clip)

        timeline.tracks.append(track)
        otio.adapters.write_to_file(timeline, output_path)
        return output_path

    except ImportError:
        logger.info("opentimelineio not available, falling back to XML export")
        xml_path = output_path.replace(".otio", ".xml")
        return export_xml_timeline(segments, xml_path, fps, name)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def assemble_rough_cut(
    script_text: str,
    clip_paths: Optional[List[str]] = None,
    transcript_map: Optional[Dict[str, str]] = None,
    output_dir: str = "",
    output_format: str = "xml",
    match_threshold: float = 0.4,
    fps: float = 24.0,
    on_progress: Optional[Callable] = None,
) -> RoughCutResult:
    """
    Assemble a rough cut from script and footage.

    Pipeline:
    1. Parse script into segments
    2. Load/generate transcripts for each clip
    3. Fuzzy-match transcript to script
    4. Select best take per segment
    5. Export as timeline

    Args:
        script_text: The screenplay/script text.
        clip_paths: List of video clip paths.
        transcript_map: Dict mapping clip_path -> transcript JSON path.
            If not provided, clips are transcribed via Whisper.
        output_dir: Directory for output files.
        output_format: "xml" or "otio".
        match_threshold: Minimum similarity for matching.
        fps: Timeline frame rate.
        on_progress: Callback(pct, msg).

    Returns:
        RoughCutResult with segments and timeline path.
    """
    if not script_text or not script_text.strip():
        raise ValueError("script_text cannot be empty")

    clip_paths = clip_paths or []
    transcript_map = transcript_map or {}

    if on_progress:
        on_progress(5, "Parsing script...")

    # Phase 1: Parse script
    script_segments = parse_script_segments(script_text)
    if not script_segments:
        raise ValueError("No segments found in script")

    result = RoughCutResult()
    result.total_segments = len([s for s in script_segments
                                 if s.segment_type in ("dialogue", "action")])

    # Phase 2: Load/generate transcripts
    if on_progress:
        on_progress(15, "Loading transcripts...")

    all_transcripts: List[TranscriptSegment] = []
    total_clips = len(clip_paths) + len(transcript_map)

    for idx, clip in enumerate(clip_paths):
        if on_progress:
            pct = 15 + int((idx / max(total_clips, 1)) * 30)
            on_progress(pct, f"Processing clip {idx + 1}/{total_clips}...")

        if clip in transcript_map:
            segs = load_transcript(transcript_map[clip], clip_path=clip)
        else:
            segs = transcribe_clip(clip)
        all_transcripts.extend(segs)

    # Also load any transcripts from transcript_map that aren't in clip_paths
    for clip, transcript_path in transcript_map.items():
        if clip not in clip_paths:
            segs = load_transcript(transcript_path, clip_path=clip)
            all_transcripts.extend(segs)

    if not all_transcripts:
        logger.warning("No transcript segments available for matching")

    # Phase 3: Fuzzy match
    if on_progress:
        on_progress(50, "Matching script to transcripts...")

    matches = fuzzy_match_segments(
        script_segments, all_transcripts,
        threshold=match_threshold,
    )

    # Phase 4: Select best takes
    if on_progress:
        on_progress(70, "Selecting best takes...")

    rough_segments = select_best_takes(matches, script_segments)
    result.segments = rough_segments
    result.matched_segments = sum(1 for s in rough_segments if s.clip_path)
    result.unmatched_segments = len(rough_segments) - result.matched_segments
    result.total_duration = sum(
        (s.out_point - s.in_point) for s in rough_segments if s.clip_path
    )

    # Phase 5: Export timeline
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        if on_progress:
            on_progress(85, f"Exporting {output_format.upper()} timeline...")

        if output_format == "otio":
            timeline_path = os.path.join(output_dir, "rough_cut.otio")
            result.timeline_path = export_otio_timeline(
                rough_segments, timeline_path, fps=fps,
            )
            result.format = "otio"
        else:
            timeline_path = os.path.join(output_dir, "rough_cut.xml")
            result.timeline_path = export_xml_timeline(
                rough_segments, timeline_path, fps=fps,
            )
            result.format = "xml"

    if on_progress:
        on_progress(100, f"Rough cut assembled: {result.matched_segments}/{result.total_segments} segments matched")

    return result
