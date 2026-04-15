"""
Advanced AI Rough Cut Assembly (Category 74)

Input: script text (or creative brief) + list of media file paths.
Pipeline: (1) transcribe each clip via get_video_info, (2) LLM matches
transcript segments to script sections, (3) score and rank candidate clips
per section, (4) select best takes, (5) arrange in script order,
(6) generate EDL / cut list with timecodes.

Supports assembly modes: strict (exact script match), loose (thematic),
highlight (best moments). Configurable target duration.
"""

import json
import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSEMBLY_MODES = ("strict", "loose", "highlight")
DEFAULT_TARGET_DURATION = 120.0  # seconds
MAX_CLIPS = 500
MAX_SCRIPT_SECTIONS = 200
KEYWORD_MATCH_THRESHOLD = 0.15
HIGHLIGHT_MIN_SCORE = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ScriptSection:
    """A section parsed from the input script."""
    index: int = 0
    text: str = ""
    keywords: List[str] = field(default_factory=list)
    duration_hint: float = 0.0
    required: bool = True

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "keywords": self.keywords,
            "duration_hint": self.duration_hint,
            "required": self.required,
        }


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio from a clip."""
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "duration": self.duration,
        }


@dataclass
class ClipAnalysis:
    """Analysis of a single media clip for rough cut selection."""
    file_path: str = ""
    filename: str = ""
    duration: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    has_audio: bool = True
    transcript: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    loudness_lufs: float = -23.0
    scene_changes: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_audio": self.has_audio,
            "transcript": self.transcript[:500],
            "segment_count": len(self.segments),
            "keywords_found": self.keywords_found,
            "loudness_lufs": self.loudness_lufs,
        }


@dataclass
class CandidateClip:
    """A candidate clip matched to a script section."""
    source_file: str = ""
    section_index: int = 0
    start: float = 0.0
    end: float = 0.0
    score: float = 0.0
    match_reason: str = ""
    transcript_excerpt: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "section_index": self.section_index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "score": self.score,
            "match_reason": self.match_reason,
            "transcript_excerpt": self.transcript_excerpt[:200],
        }


@dataclass
class CutEntry:
    """An entry in the final cut list / EDL."""
    order: int = 0
    source_file: str = ""
    source_in: float = 0.0
    source_out: float = 0.0
    record_in: float = 0.0
    record_out: float = 0.0
    section_index: int = -1
    transition: str = "cut"
    transition_duration: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.source_out - self.source_in)

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "source_file": self.source_file,
            "source_in": self.source_in,
            "source_out": self.source_out,
            "record_in": self.record_in,
            "record_out": self.record_out,
            "section_index": self.section_index,
            "duration": self.duration,
            "transition": self.transition,
            "transition_duration": self.transition_duration,
        }


@dataclass
class RoughCutResult:
    """Result of the rough cut assembly pipeline."""
    cuts: List[CutEntry] = field(default_factory=list)
    unmatched_script_sections: List[int] = field(default_factory=list)
    unused_clips: List[str] = field(default_factory=list)
    total_duration: float = 0.0
    mode: str = "strict"
    edl_text: str = ""
    clip_analyses: List[ClipAnalysis] = field(default_factory=list)
    llm_used: bool = False

    def to_dict(self) -> dict:
        return {
            "cuts": [c.to_dict() for c in self.cuts],
            "unmatched_script_sections": self.unmatched_script_sections,
            "unused_clips": self.unused_clips,
            "total_duration": self.total_duration,
            "mode": self.mode,
            "edl_text": self.edl_text,
            "clip_count": len(self.cuts),
            "llm_used": self.llm_used,
        }


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------
def parse_script(script_text: str) -> List[ScriptSection]:
    """Parse script text into sections.

    Splits on double newlines, numbered lines, or section markers.
    Returns a list of ScriptSection objects.
    """
    if not script_text or not script_text.strip():
        return []

    sections: List[ScriptSection] = []
    # Try splitting on common section markers
    # Pattern: numbered sections like "1.", "1)", "Section 1:", or "## Heading"
    pattern = r"(?:^|\n)(?=(?:\d+[\.\)]\s|#{1,3}\s|SECTION\s+\d|INT\.|EXT\.))"
    parts = re.split(pattern, script_text.strip(), flags=re.IGNORECASE)

    if len(parts) <= 1:
        # Fallback: split on double newlines
        parts = [p.strip() for p in script_text.strip().split("\n\n") if p.strip()]

    if not parts:
        parts = [script_text.strip()]

    for i, part in enumerate(parts[:MAX_SCRIPT_SECTIONS]):
        text = part.strip()
        if not text:
            continue
        keywords = _extract_keywords(text)
        sections.append(ScriptSection(
            index=i,
            text=text,
            keywords=keywords,
            duration_hint=max(3.0, len(text.split()) / 3.0),  # ~3 words/sec
            required=True,
        ))

    return sections


def _extract_keywords(text: str) -> List[str]:
    """Extract significant keywords from text for matching."""
    stop_words = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "then", "than", "too", "very", "just",
        "about", "up", "out", "off", "over", "under", "again", "further",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such",
        "only", "own", "same", "it", "its", "he", "she", "they", "them",
        "his", "her", "their", "this", "that", "these", "those", "i", "we",
        "you", "my", "your", "our", "what", "which", "who",
    })
    words = re.findall(r"[a-z]{3,}", text.lower())
    seen = set()
    keywords = []
    for w in words:
        if w not in stop_words and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords[:50]


# ---------------------------------------------------------------------------
# Clip analysis
# ---------------------------------------------------------------------------
def _analyze_clip(file_path: str) -> ClipAnalysis:
    """Analyze a single clip: probe metadata, detect audio presence."""
    filename = os.path.basename(file_path)
    info = get_video_info(file_path)

    analysis = ClipAnalysis(
        file_path=file_path,
        filename=filename,
        duration=info.get("duration", 0.0),
        width=info.get("width", 1920),
        height=info.get("height", 1080),
        fps=info.get("fps", 30.0),
    )

    # Check for audio stream
    try:
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "json", file_path,
        ]
        result = _sp.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            data = json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            analysis.has_audio = len(streams) > 0
    except Exception as exc:
        logger.debug("Audio probe failed for %s: %s", filename, exc)

    return analysis


def _transcribe_clip_mock(analysis: ClipAnalysis) -> ClipAnalysis:
    """Mock transcription using filename and duration.

    In production, this would call Whisper or similar. For now, we generate
    placeholder transcript segments based on clip metadata.
    """
    if analysis.duration <= 0:
        return analysis

    # Generate mock segments every ~5 seconds
    seg_duration = 5.0
    t = 0.0
    segments = []
    base_name = os.path.splitext(analysis.filename)[0].replace("_", " ").replace("-", " ")
    while t < analysis.duration:
        end = min(t + seg_duration, analysis.duration)
        segments.append(TranscriptSegment(
            text=f"[{base_name}] segment at {t:.1f}s",
            start=t,
            end=end,
            confidence=0.85,
        ))
        t = end

    analysis.segments = segments
    analysis.transcript = " ".join(s.text for s in segments)
    return analysis


def analyze_clips(
    media_paths: List[str],
    on_progress: Optional[Callable] = None,
) -> List[ClipAnalysis]:
    """Analyze all input clips: probe metadata, mock-transcribe."""
    analyses = []
    total = len(media_paths)
    for i, path in enumerate(media_paths):
        if on_progress:
            pct = int((i / max(total, 1)) * 30)
            on_progress(pct)

        try:
            clip = _analyze_clip(path)
            clip = _transcribe_clip_mock(clip)
            analyses.append(clip)
        except Exception as exc:
            logger.warning("Failed to analyze clip %s: %s", path, exc)
            analyses.append(ClipAnalysis(file_path=path, filename=os.path.basename(path)))

    return analyses


# ---------------------------------------------------------------------------
# LLM-based matching
# ---------------------------------------------------------------------------
def _match_sections_llm(
    sections: List[ScriptSection],
    analyses: List[ClipAnalysis],
    config: object = None,
) -> Dict[int, List[CandidateClip]]:
    """Use LLM to match clip transcripts to script sections."""
    try:
        from opencut.core.llm import LLMConfig, query_llm  # noqa: F401
    except ImportError:
        logger.info("LLM module not available, falling back to keyword matching")
        return {}

    if config is None:
        config = LLMConfig()

    # Build prompt
    section_descs = []
    for s in sections:
        section_descs.append(f"Section {s.index}: {s.text[:200]}")

    clip_descs = []
    for a in analyses:
        clip_descs.append(
            f"Clip '{a.filename}' ({a.duration:.1f}s): {a.transcript[:300]}"
        )

    prompt = (
        "Match the following clips to script sections. For each section, "
        "list the best matching clips with start/end timecodes and a score 0-1.\n\n"
        "SECTIONS:\n" + "\n".join(section_descs) + "\n\n"
        "CLIPS:\n" + "\n".join(clip_descs) + "\n\n"
        "Respond in JSON format:\n"
        '{"matches": [{"section_index": 0, "clip_filename": "...", '
        '"start": 0.0, "end": 10.0, "score": 0.8, "reason": "..."}]}'
    )

    try:
        response = query_llm(prompt, config=config, system_prompt=(
            "You are a professional video editor. Match footage clips to "
            "script sections based on transcript content relevance."
        ))
        if not response or not response.text:
            return {}

        # Parse JSON from response
        text = response.text.strip()
        # Extract JSON block if wrapped in markdown
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return {}

        result = json.loads(json_match.group())
        matches_data = result.get("matches", [])

        # Build filename -> analysis lookup
        name_to_analysis = {a.filename: a for a in analyses}

        candidates: Dict[int, List[CandidateClip]] = {}
        for m in matches_data:
            sec_idx = int(m.get("section_index", -1))
            clip_fn = m.get("clip_filename", "")
            analysis = name_to_analysis.get(clip_fn)
            if analysis is None or sec_idx < 0:
                continue

            candidate = CandidateClip(
                source_file=analysis.file_path,
                section_index=sec_idx,
                start=float(m.get("start", 0)),
                end=float(m.get("end", analysis.duration)),
                score=min(1.0, max(0.0, float(m.get("score", 0.5)))),
                match_reason=str(m.get("reason", "LLM match")),
                transcript_excerpt=analysis.transcript[:200],
            )
            candidates.setdefault(sec_idx, []).append(candidate)

        return candidates

    except Exception as exc:
        logger.warning("LLM matching failed: %s — falling back to keywords", exc)
        return {}


# ---------------------------------------------------------------------------
# Keyword-based matching (fallback)
# ---------------------------------------------------------------------------
def _compute_keyword_score(
    section: ScriptSection,
    analysis: ClipAnalysis,
) -> float:
    """Compute keyword overlap score between section and clip transcript."""
    if not section.keywords or not analysis.transcript:
        return 0.0

    transcript_lower = analysis.transcript.lower()
    transcript_words = set(re.findall(r"[a-z]{3,}", transcript_lower))

    matches = 0
    for kw in section.keywords:
        if kw in transcript_words or kw in transcript_lower:
            matches += 1

    if not section.keywords:
        return 0.0
    return matches / len(section.keywords)


def _match_sections_keywords(
    sections: List[ScriptSection],
    analyses: List[ClipAnalysis],
) -> Dict[int, List[CandidateClip]]:
    """Match sections to clips using keyword overlap."""
    candidates: Dict[int, List[CandidateClip]] = {}

    for section in sections:
        section_candidates = []
        for analysis in analyses:
            if analysis.duration <= 0:
                continue

            score = _compute_keyword_score(section, analysis)
            if score >= KEYWORD_MATCH_THRESHOLD or not section.keywords:
                # If no keywords, assign a base score proportional to duration fit
                if not section.keywords:
                    dur_ratio = min(analysis.duration, section.duration_hint) / max(
                        section.duration_hint, 1.0
                    )
                    score = 0.3 * dur_ratio

                section_candidates.append(CandidateClip(
                    source_file=analysis.file_path,
                    section_index=section.index,
                    start=0.0,
                    end=min(analysis.duration, section.duration_hint * 1.5),
                    score=score,
                    match_reason="keyword_match",
                    transcript_excerpt=analysis.transcript[:200],
                ))

        # Sort by score descending
        section_candidates.sort(key=lambda c: c.score, reverse=True)
        if section_candidates:
            candidates[section.index] = section_candidates

    return candidates


# ---------------------------------------------------------------------------
# Clip selection
# ---------------------------------------------------------------------------
def _select_best_clips(
    sections: List[ScriptSection],
    candidates: Dict[int, List[CandidateClip]],
    mode: str = "strict",
    target_duration: float = DEFAULT_TARGET_DURATION,
) -> tuple:
    """Select the best clip for each section.

    Returns (selected_cuts, unmatched_section_indices, unused_clip_paths).
    """
    selected: List[CutEntry] = []
    used_files = set()
    unmatched = []
    record_pos = 0.0

    if mode == "highlight":
        # Highlight mode: pick top-scoring candidates regardless of section
        all_candidates = []
        for sec_idx, cands in candidates.items():
            for c in cands:
                if c.score >= HIGHLIGHT_MIN_SCORE:
                    all_candidates.append(c)
        all_candidates.sort(key=lambda c: c.score, reverse=True)

        for c in all_candidates:
            if record_pos >= target_duration:
                break
            clip_dur = c.duration
            if clip_dur <= 0:
                continue
            # Trim to fit target
            if record_pos + clip_dur > target_duration:
                clip_dur = target_duration - record_pos

            entry = CutEntry(
                order=len(selected),
                source_file=c.source_file,
                source_in=c.start,
                source_out=c.start + clip_dur,
                record_in=record_pos,
                record_out=record_pos + clip_dur,
                section_index=c.section_index,
                transition="cut" if not selected else "dissolve",
                transition_duration=0.5 if selected else 0.0,
            )
            selected.append(entry)
            used_files.add(c.source_file)
            record_pos += clip_dur

    else:
        # Strict and loose modes: assign best clip per section in order
        for section in sections:
            sec_cands = candidates.get(section.index, [])

            if mode == "strict" and not sec_cands:
                unmatched.append(section.index)
                continue

            if not sec_cands:
                # Loose mode: skip unmatched sections silently
                unmatched.append(section.index)
                continue

            # Pick the highest-scoring candidate not yet used (prefer variety)
            best = None
            for c in sec_cands:
                if mode == "strict" or c.source_file not in used_files:
                    best = c
                    break
            if best is None and sec_cands:
                best = sec_cands[0]

            if best is None:
                unmatched.append(section.index)
                continue

            clip_dur = best.duration
            # Trim to section duration hint if available
            if section.duration_hint > 0 and clip_dur > section.duration_hint * 1.5:
                clip_dur = section.duration_hint

            # Check target duration
            if record_pos + clip_dur > target_duration and target_duration > 0:
                remaining = target_duration - record_pos
                if remaining < 1.0:
                    break
                clip_dur = remaining

            entry = CutEntry(
                order=len(selected),
                source_file=best.source_file,
                source_in=best.start,
                source_out=best.start + clip_dur,
                record_in=record_pos,
                record_out=record_pos + clip_dur,
                section_index=section.index,
                transition="cut",
            )
            selected.append(entry)
            used_files.add(best.source_file)
            record_pos += clip_dur

    return selected, unmatched, used_files


# ---------------------------------------------------------------------------
# EDL generation
# ---------------------------------------------------------------------------
def _format_timecode(seconds: float, fps: float = 30.0) -> str:
    """Format seconds as SMPTE timecode HH:MM:SS:FF."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1.0) * fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def generate_edl(
    cuts: List[CutEntry],
    title: str = "OpenCut Rough Cut",
    fps: float = 30.0,
) -> str:
    """Generate EDL (Edit Decision List) text from cut entries."""
    lines = [
        f"TITLE: {title}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    for i, cut in enumerate(cuts):
        edit_num = f"{i + 1:03d}"
        reel = os.path.splitext(os.path.basename(cut.source_file))[0][:8].upper()
        if not reel:
            reel = f"REEL{i + 1:03d}"

        # Transition type
        if cut.transition == "dissolve" and cut.transition_duration > 0:
            trans = f"D    {int(cut.transition_duration * fps):03d}"
        else:
            trans = "C   "

        src_in = _format_timecode(cut.source_in, fps)
        src_out = _format_timecode(cut.source_out, fps)
        rec_in = _format_timecode(cut.record_in, fps)
        rec_out = _format_timecode(cut.record_out, fps)

        lines.append(f"{edit_num}  {reel:8s}  V     {trans}  {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {os.path.basename(cut.source_file)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def assemble_rough_cut(
    script_text: str,
    media_paths: List[str],
    mode: str = "strict",
    target_duration: float = DEFAULT_TARGET_DURATION,
    llm_config: object = None,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> RoughCutResult:
    """Assemble a rough cut from script and media clips.

    Args:
        script_text: Script or creative brief text.
        media_paths: List of media file paths.
        mode: Assembly mode — 'strict', 'loose', or 'highlight'.
        target_duration: Target output duration in seconds.
        llm_config: Optional LLMConfig for AI matching.
        fps: Timeline frame rate.
        on_progress: Callback(pct) for progress updates.

    Returns:
        RoughCutResult with cuts, EDL, and metadata.
    """
    if mode not in ASSEMBLY_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(ASSEMBLY_MODES)}")

    if not media_paths:
        raise ValueError("No media files provided")

    if len(media_paths) > MAX_CLIPS:
        raise ValueError(f"Too many clips ({len(media_paths)}). Max: {MAX_CLIPS}")

    if on_progress:
        on_progress(5)

    # Step 1: Parse script
    sections = parse_script(script_text)
    if not sections and mode == "strict":
        raise ValueError("No sections found in script text")

    # For highlight mode without script, create a single catch-all section
    if not sections:
        sections = [ScriptSection(index=0, text="Highlights", required=False)]

    if on_progress:
        on_progress(10)

    # Step 2: Analyze and transcribe clips
    analyses = analyze_clips(media_paths, on_progress=on_progress)

    if on_progress:
        on_progress(40)

    # Step 3: Match sections to clips
    llm_used = False
    candidates: Dict[int, List[CandidateClip]] = {}

    if llm_config is not None:
        candidates = _match_sections_llm(sections, analyses, config=llm_config)
        if candidates:
            llm_used = True

    # Fallback to keyword matching if LLM produced no results
    if not candidates:
        candidates = _match_sections_keywords(sections, analyses)

    if on_progress:
        on_progress(60)

    # Step 4: Select best clips
    cuts, unmatched, used_files = _select_best_clips(
        sections, candidates, mode=mode, target_duration=target_duration,
    )

    if on_progress:
        on_progress(80)

    # Step 5: Determine unused clips
    all_paths = set(media_paths)
    unused = sorted(all_paths - used_files)

    # Step 6: Generate EDL
    total_dur = sum(c.duration for c in cuts)
    edl_text = generate_edl(cuts, fps=fps)

    if on_progress:
        on_progress(95)

    result = RoughCutResult(
        cuts=cuts,
        unmatched_script_sections=unmatched,
        unused_clips=unused,
        total_duration=total_dur,
        mode=mode,
        edl_text=edl_text,
        clip_analyses=analyses,
        llm_used=llm_used,
    )

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# Convenience: export EDL to file
# ---------------------------------------------------------------------------
def export_edl(
    result: RoughCutResult,
    output_dir: str = "",
    filename: str = "rough_cut.edl",
) -> str:
    """Write the EDL from a RoughCutResult to a file."""
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    edl_path = os.path.join(output_dir, filename) if output_dir else filename
    with open(edl_path, "w", encoding="utf-8") as f:
        f.write(result.edl_text)
    logger.info("EDL exported to %s", edl_path)
    return edl_path


# ---------------------------------------------------------------------------
# Convenience: export cut list as JSON
# ---------------------------------------------------------------------------
def export_cut_list_json(
    result: RoughCutResult,
    output_dir: str = "",
    filename: str = "rough_cut.json",
) -> str:
    """Write the cut list from a RoughCutResult as JSON."""
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, filename) if output_dir else filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Cut list exported to %s", json_path)
    return json_path
