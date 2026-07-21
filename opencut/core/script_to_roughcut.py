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
from typing import Callable, Dict, List, Optional

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
    """A segment selected for the rough cut.

    ``alternates`` holds the runner-up takes for this same script line (distinct
    clips, ranked). They are laid out on separate video tracks (V2, V3, ...) in
    the exported timeline so an editor can audition them against the primary V1
    pick without re-running the match.
    """
    index: int = 0
    clip_path: str = ""
    in_point: float = 0.0
    out_point: float = 0.0
    script_text: str = ""
    character: str = ""
    similarity: float = 0.0
    alternates: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoughCutResult:
    """Complete rough cut assembly result.

    When ``timeline_path`` is empty the result is a *preview plan* (no files
    written, no host write-back). ``journal_entry_id`` is set once the plan is
    written back through the reversible operation journal.
    """
    segments: List[RoughCutSegment] = field(default_factory=list)
    total_segments: int = 0
    matched_segments: int = 0
    unmatched_segments: int = 0
    total_duration: float = 0.0
    timeline_path: str = ""
    format: str = ""
    preview: bool = False
    alternate_takes: int = 0
    journal_entry_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "total_segments": self.total_segments,
            "matched_segments": self.matched_segments,
            "unmatched_segments": self.unmatched_segments,
            "total_duration": round(self.total_duration, 2),
            "timeline_path": self.timeline_path,
            "format": self.format,
            "preview": self.preview,
            "alternate_takes": self.alternate_takes,
            "journal_entry_id": self.journal_entry_id,
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


def _alternate_from_match(match: "MatchResult") -> dict:
    ts = match.transcript_segment
    return {
        "clip_path": ts.clip_path,
        "in_point": ts.start_time,
        "out_point": ts.end_time,
        "similarity": match.similarity,
        "combined_score": match.combined_score,
    }


def select_takes_with_alternates(
    matches: Dict[int, List[MatchResult]],
    script_segments: List[ScriptSegment],
    max_alternates: int = 2,
) -> List[RoughCutSegment]:
    """Select the best take per script segment plus ranked alternate takes.

    Identical primary selection to :func:`select_best_takes`, but each segment
    also carries up to ``max_alternates`` runner-up takes drawn from *distinct*
    clips (so V2/V3 never just repeat V1's clip). Alternates are exported on
    separate video tracks for review.
    """
    result = select_best_takes(matches, script_segments)
    if max_alternates <= 0:
        return result

    dialogue_action = [
        ss for ss in script_segments if ss.segment_type in ("dialogue", "action")
    ]
    for seg, ss in zip(result, dialogue_action):
        ranked = matches.get(ss.index) or []
        if not seg.clip_path or len(ranked) < 2:
            continue
        seen_clips = {seg.clip_path}
        alternates: List[dict] = []
        for match in ranked[1:]:
            clip = match.transcript_segment.clip_path
            if clip in seen_clips:
                continue
            seen_clips.add(clip)
            alternates.append(_alternate_from_match(match))
            if len(alternates) >= max_alternates:
                break
        seg.alternates = alternates

    return result


# ---------------------------------------------------------------------------
# Timeline Export
# ---------------------------------------------------------------------------
def _alternate_tracks(segments: List[RoughCutSegment]) -> int:
    """How many alternate video tracks the segment list needs (V2, V3, ...)."""
    return max((len(s.alternates) for s in segments if s.clip_path), default=0)
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

    def _add_clipitem(track_el, clip_path, in_point, out_point, timeline_pos):
        clip_el = ET.SubElement(track_el, "clipitem")
        ET.SubElement(clip_el, "name").text = os.path.basename(clip_path)
        dur_frames = int((out_point - in_point) * fps)
        ET.SubElement(clip_el, "start").text = str(timeline_pos)
        ET.SubElement(clip_el, "end").text = str(timeline_pos + dur_frames)
        ET.SubElement(clip_el, "in").text = str(int(in_point * fps))
        ET.SubElement(clip_el, "out").text = str(int(out_point * fps))
        file_el = ET.SubElement(clip_el, "file")
        ET.SubElement(file_el, "pathurl").text = f"file:///{clip_path.replace(os.sep, '/')}"
        clip_rate = ET.SubElement(clip_el, "rate")
        ET.SubElement(clip_rate, "timebase").text = str(int(fps))
        return dur_frames

    # Primary track (V1). Record each segment's timeline start so alternate
    # tracks can align their takes to the same position.
    primary_track = ET.SubElement(video, "track")
    starts: List[int] = []
    timeline_pos = 0
    for seg in segments:
        starts.append(timeline_pos)
        if not seg.clip_path:
            continue
        timeline_pos += _add_clipitem(
            primary_track, seg.clip_path, seg.in_point, seg.out_point, timeline_pos
        )

    # Alternate tracks (V2, V3, ...): each ranked alternate on its own track,
    # aligned to the primary segment's start position.
    for rank in range(_alternate_tracks(segments)):
        alt_track = ET.SubElement(video, "track")
        for seg, start in zip(segments, starts):
            if not seg.clip_path or rank >= len(seg.alternates):
                continue
            alt = seg.alternates[rank]
            _add_clipitem(
                alt_track, alt["clip_path"], alt["in_point"], alt["out_point"], start
            )

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

        def _slot_frames(seg) -> int:
            """Timeline width (frames) a segment occupies on V1."""
            if not seg.clip_path:
                return int(fps)  # 1-second gap for unmatched segments
            return max(0, int((seg.out_point - seg.in_point) * fps))

        def _gap(frames):
            return otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, fps),
                    duration=otio.opentime.RationalTime(frames, fps),
                )
            )

        def _clip(clip_path, in_point, frames):
            return otio.schema.Clip(
                name=os.path.basename(clip_path),
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(int(in_point * fps), fps),
                    duration=otio.opentime.RationalTime(frames, fps),
                ),
                media_reference=otio.schema.ExternalReference(target_url=clip_path),
            )

        timeline = otio.schema.Timeline(name=name)

        # Primary track (V1).
        track = otio.schema.Track(name="V1")
        for seg in segments:
            frames = _slot_frames(seg)
            if not seg.clip_path or frames <= 0:
                track.append(_gap(max(frames, int(fps))))
                continue
            track.append(_clip(seg.clip_path, seg.in_point, frames))
        timeline.tracks.append(track)

        # Alternate tracks (V2, V3, ...): each rank on its own track, every slot
        # padded to the primary segment's width so alternates stay time-aligned.
        for rank in range(_alternate_tracks(segments)):
            alt_track = otio.schema.Track(name=f"V{rank + 2}")
            for seg in segments:
                slot = _slot_frames(seg)
                if not seg.clip_path or rank >= len(seg.alternates):
                    alt_track.append(_gap(max(slot, 1)))
                    continue
                alt = seg.alternates[rank]
                alt_frames = max(0, int((alt["out_point"] - alt["in_point"]) * fps))
                take = min(alt_frames, slot) if slot > 0 else alt_frames
                if take <= 0:
                    alt_track.append(_gap(max(slot, 1)))
                    continue
                alt_track.append(_clip(alt["clip_path"], alt["in_point"], take))
                if slot > take:
                    alt_track.append(_gap(slot - take))
            timeline.tracks.append(alt_track)

        otio.adapters.write_to_file(timeline, output_path)
        return output_path

    except ImportError:
        logger.info("opentimelineio not available, falling back to XML export")
        xml_path = output_path.replace(".otio", ".xml")
        return export_xml_timeline(segments, xml_path, fps, name)


# ---------------------------------------------------------------------------
# Reversible write-back (operation journal)
# ---------------------------------------------------------------------------
def record_write_back(result: RoughCutResult, sequence_name: str = "OpenCut Rough Cut") -> Optional[int]:
    """Record the rough-cut write-back in the reversible operation journal.

    The rough cut is imported into the host as a sequence, which the panel can
    revert (``import_sequence`` is a revertible action). We store the timeline
    path and a plan summary as the inverse payload so the panel can dispatch the
    ExtendScript inverse. Best-effort: a journal failure never breaks assembly.

    Returns the journal entry id, or ``None`` if nothing was recorded.
    """
    if not result.timeline_path:
        return None
    try:
        from opencut import journal

        entry = journal.record(
            action="import_sequence",
            label=f"Rough cut: {sequence_name} ({result.matched_segments}/{result.total_segments} matched)",
            inverse_payload={
                "sequence_name": sequence_name,
                "timeline_path": result.timeline_path,
                "format": result.format,
                "clip_count": result.matched_segments,
                "alternate_takes": result.alternate_takes,
            },
            forward_payload={
                "endpoint": "/rough-cut/from-script",
                "timeline_path": result.timeline_path,
            },
        )
        return entry.get("id") if isinstance(entry, dict) else None
    except Exception as exc:  # pragma: no cover - durability is best-effort
        logger.warning("Could not record rough-cut journal entry: %s", exc)
        return None


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
    max_alternates: int = 2,
    write_back: bool = True,
    record_journal: bool = True,
) -> RoughCutResult:
    """
    Assemble a rough cut from script and footage.

    Pipeline:
    1. Parse script into segments
    2. Load/generate transcripts for each clip
    3. Fuzzy-match transcript to script
    4. Select best take per segment (plus ranked alternates)
    5. Export as timeline (unless preview) and record a reversible journal entry

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
        max_alternates: How many runner-up takes to attach per segment (0 = none).
            Alternates are laid out on separate video tracks in the export.
        write_back: When False, produce a previewable plan only — no timeline is
            written and no journal entry is recorded, so the caller can review
            (and the user approve) before committing.
        record_journal: When True (and writing back), record the write-back in
            the reversible operation journal.

    Returns:
        RoughCutResult with segments, alternates, and (when written back) the
        timeline path and journal entry id.
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

    # Phase 4: Select best takes (with ranked alternates)
    if on_progress:
        on_progress(70, "Selecting best takes...")

    rough_segments = select_takes_with_alternates(
        matches, script_segments, max_alternates=max_alternates
    )
    result.segments = rough_segments
    result.matched_segments = sum(1 for s in rough_segments if s.clip_path)
    result.unmatched_segments = len(rough_segments) - result.matched_segments
    result.total_duration = sum(
        (s.out_point - s.in_point) for s in rough_segments if s.clip_path
    )
    result.alternate_takes = sum(len(s.alternates) for s in rough_segments)

    # Phase 5: Preview or write back. A preview returns the plan without writing
    # any timeline or touching the reversible journal, so it can be reviewed and
    # approved before committing.
    if not write_back or not output_dir:
        result.preview = True
        if on_progress:
            on_progress(100, f"Rough cut plan ready (preview): "
                             f"{result.matched_segments}/{result.total_segments} segments matched")
        return result

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

    # Record the reversible write-back so the panel can undo the import.
    if record_journal:
        if on_progress:
            on_progress(95, "Recording reversible journal entry...")
        result.journal_entry_id = record_write_back(result)

    if on_progress:
        on_progress(100, f"Rough cut assembled: {result.matched_segments}/{result.total_segments} segments matched")

    return result
