"""
AI Rough Cut Assembly (21.3)

Multi-stage pipeline: transcribe all footage -> LLM matches to brief ->
score+select clips -> arrange narratively -> apply audio -> add captions ->
add music -> export.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RoughCutBrief:
    """Creative brief for rough cut assembly."""
    goal: str = ""
    style: str = "narrative"  # narrative, documentary, highlight, tutorial
    duration: float = 60.0  # target duration in seconds
    keywords: List[str] = field(default_factory=list)
    tone: str = "neutral"  # upbeat, dramatic, calm, humorous, neutral
    pacing: str = "medium"  # slow, medium, fast
    include_captions: bool = False
    include_music: bool = False
    music_style: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "style": self.style,
            "duration": self.duration,
            "keywords": self.keywords,
            "tone": self.tone,
            "pacing": self.pacing,
            "include_captions": self.include_captions,
            "include_music": self.include_music,
            "music_style": self.music_style,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoughCutBrief":
        return cls(
            goal=data.get("goal", ""),
            style=data.get("style", "narrative"),
            duration=float(data.get("duration", 60)),
            keywords=data.get("keywords", []),
            tone=data.get("tone", "neutral"),
            pacing=data.get("pacing", "medium"),
            include_captions=bool(data.get("include_captions", False)),
            include_music=bool(data.get("include_music", False)),
            music_style=data.get("music_style", ""),
        )


@dataclass
class AnalyzedClip:
    """A clip analyzed for rough cut selection."""
    file_path: str = ""
    duration: float = 0.0
    transcript_text: str = ""
    transcript_segments: List[dict] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    highlights: List[dict] = field(default_factory=list)
    speech_segments: List[dict] = field(default_factory=list)
    has_speech: bool = False
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "duration": self.duration,
            "transcript_text": self.transcript_text[:500],
            "keywords_found": self.keywords_found,
            "has_speech": self.has_speech,
            "quality_score": self.quality_score,
            "highlight_count": len(self.highlights),
        }


@dataclass
class PlannedClip:
    """A clip selected for the rough cut with justification."""
    source_file: str = ""
    start: float = 0.0
    end: float = 0.0
    order: int = 0
    justification: str = ""
    score: float = 0.0
    clip_type: str = "content"  # content, broll, transition
    transcript_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "order": self.order,
            "justification": self.justification,
            "score": self.score,
            "clip_type": self.clip_type,
            "transcript_text": self.transcript_text[:200],
        }


@dataclass
class RoughCutPlan:
    """Ordered list of selected clips with justification."""
    clips: List[PlannedClip] = field(default_factory=list)
    brief: Optional[RoughCutBrief] = None
    total_duration: float = 0.0
    narrative_summary: str = ""
    llm_provider: str = ""
    llm_model: str = ""

    def to_dict(self) -> dict:
        return {
            "clips": [c.to_dict() for c in self.clips],
            "brief": self.brief.to_dict() if self.brief else None,
            "total_duration": self.total_duration,
            "narrative_summary": self.narrative_summary,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }


@dataclass
class RoughCutResult:
    """Result of rough cut assembly."""
    output_path: str = ""
    duration: float = 0.0
    clip_count: int = 0
    plan: Optional[RoughCutPlan] = None
    caption_path: str = ""
    music_path: str = ""
    edl_path: str = ""

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "duration": self.duration,
            "clip_count": self.clip_count,
            "plan": self.plan.to_dict() if self.plan else None,
            "caption_path": self.caption_path,
            "music_path": self.music_path,
            "edl_path": self.edl_path,
        }


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------
_PLAN_SYSTEM_PROMPT = """You are a professional video editor creating a rough cut assembly.
Given a creative brief and analyzed footage, select and arrange clips
to tell a compelling story. Return your plan as JSON.

Your response MUST be a valid JSON object with this structure:
{
  "clips": [
    {
      "source_file": "filename",
      "start": 0.0,
      "end": 10.0,
      "justification": "why this clip",
      "clip_type": "content",
      "score": 0.9
    }
  ],
  "narrative_summary": "Brief description of the story arc"
}

Guidelines:
- Select clips that match the brief's goal, style, and keywords
- Order clips for narrative flow
- Respect the target duration
- Prefer clips with clear speech and good quality
- Include variety in shot types when possible
- Clip types: "content" (main footage), "broll" (supporting visuals), "transition" (bridge clips)
"""


# ---------------------------------------------------------------------------
# Analyze Footage
# ---------------------------------------------------------------------------
def analyze_footage(
    file_paths: List[str],
    keywords: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[AnalyzedClip]:
    """Transcribe and analyze all footage clips.

    Args:
        file_paths: List of video/audio file paths.
        keywords: Optional keywords to look for in transcripts.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of AnalyzedClip with transcript, highlights, and analysis.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    # Validate all files exist
    for fp in file_paths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"File not found: {fp}")

    keywords = keywords or []
    analyzed = []
    total = len(file_paths)

    for i, fp in enumerate(file_paths):
        if on_progress:
            pct = int((i / total) * 90)
            on_progress(pct, f"Analyzing clip {i + 1}/{total}: {os.path.basename(fp)}")

        clip = _analyze_single_clip(fp, keywords)
        analyzed.append(clip)

    if on_progress:
        on_progress(100, f"Analyzed {len(analyzed)} clips")

    return analyzed


def _analyze_single_clip(
    file_path: str,
    keywords: List[str],
) -> AnalyzedClip:
    """Analyze a single clip: transcribe, detect speech, find keywords."""
    clip = AnalyzedClip(file_path=file_path)

    # Get duration
    try:
        info = get_video_info(file_path)
        clip.duration = float(info.get("duration", 0))
    except Exception as e:
        logger.warning("Could not probe %s: %s", file_path, e)

    # Transcribe
    try:
        from opencut.core.captions import transcribe
        result = transcribe(file_path)
        if result and result.segments:
            clip.transcript_text = result.text
            clip.transcript_segments = [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result.segments
            ]
            clip.has_speech = bool(result.text.strip())
    except Exception as e:
        logger.warning("Transcription failed for %s: %s", file_path, e)

    # Detect speech segments
    try:
        from opencut.core.silence import detect_speech
        speech_segs = detect_speech(file_path)
        clip.speech_segments = [
            {"start": s.start, "end": s.end}
            for s in speech_segs
        ]
    except Exception as e:
        logger.warning("Speech detection failed for %s: %s", file_path, e)

    # Find keywords
    if keywords and clip.transcript_text:
        text_lower = clip.transcript_text.lower()
        clip.keywords_found = [
            kw for kw in keywords if kw.lower() in text_lower
        ]

    # Quality score (heuristic)
    clip.quality_score = _score_clip_quality(clip)

    # Extract highlights
    if clip.transcript_segments:
        try:
            from opencut.core.highlights import extract_highlights
            hl_result = extract_highlights(
                clip.transcript_segments,
                max_highlights=3,
                min_duration=5.0,
                max_duration=30.0,
            )
            clip.highlights = [
                {
                    "start": h.start,
                    "end": h.end,
                    "title": h.title,
                    "score": h.score,
                }
                for h in hl_result.highlights
            ]
        except Exception as e:
            logger.warning("Highlight extraction failed for %s: %s", file_path, e)

    return clip


def _score_clip_quality(clip: AnalyzedClip) -> float:
    """Score clip quality from 0-1 based on available signals."""
    score = 0.0

    # Has speech is valuable
    if clip.has_speech:
        score += 0.3

    # Longer clips have more content
    if clip.duration > 10:
        score += 0.1
    if clip.duration > 30:
        score += 0.1

    # Keywords found
    if clip.keywords_found:
        score += min(0.3, len(clip.keywords_found) * 0.1)

    # Has speech segments (clear audio)
    if clip.speech_segments:
        speech_ratio = sum(
            s.get("end", 0) - s.get("start", 0)
            for s in clip.speech_segments
        ) / max(1.0, clip.duration)
        score += min(0.2, speech_ratio * 0.3)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Generate Plan (LLM)
# ---------------------------------------------------------------------------
def generate_plan(
    brief: RoughCutBrief,
    analyzed_footage: List[AnalyzedClip],
    llm_config=None,
    on_progress: Optional[Callable] = None,
) -> RoughCutPlan:
    """Use LLM to generate an ordered rough cut plan from analyzed footage.

    Args:
        brief: Creative brief describing the desired rough cut.
        analyzed_footage: List of AnalyzedClip from analyze_footage().
        llm_config: LLMConfig for LLM query. Uses defaults if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        RoughCutPlan with ordered clips and justifications.
    """
    from opencut.core.llm import LLMConfig, query_llm

    if llm_config is None:
        llm_config = LLMConfig()

    if not analyzed_footage:
        raise ValueError("No analyzed footage provided")

    if on_progress:
        on_progress(10, "Formatting footage analysis for LLM...")

    # Build footage summary for LLM
    footage_summary = _format_footage_for_llm(analyzed_footage)

    prompt = (
        f"Create a rough cut plan based on this brief and available footage.\n\n"
        f"BRIEF:\n"
        f"- Goal: {brief.goal}\n"
        f"- Style: {brief.style}\n"
        f"- Target duration: {brief.duration:.0f} seconds\n"
        f"- Keywords: {', '.join(brief.keywords) if brief.keywords else 'none'}\n"
        f"- Tone: {brief.tone}\n"
        f"- Pacing: {brief.pacing}\n\n"
        f"AVAILABLE FOOTAGE:\n{footage_summary}\n\n"
        f"Select clips and arrange them to tell a compelling story. "
        f"Total duration should be close to {brief.duration:.0f} seconds."
    )

    if on_progress:
        on_progress(30, "Querying LLM for rough cut plan...")

    response = query_llm(
        prompt=prompt,
        config=llm_config,
        system_prompt=_PLAN_SYSTEM_PROMPT,
    )

    if on_progress:
        on_progress(70, "Parsing LLM response...")

    if not response or not response.text or response.text.startswith("LLM error:"):
        logger.error("LLM query failed: %s", getattr(response, "text", None))
        # Fallback: create a simple plan from highest-quality clips
        return _fallback_plan(brief, analyzed_footage)

    # Parse LLM response
    plan = _parse_plan_response(response.text, analyzed_footage, brief)
    plan.llm_provider = response.provider
    plan.llm_model = response.model

    if on_progress:
        on_progress(100, f"Plan ready: {len(plan.clips)} clips, {plan.total_duration:.1f}s")

    return plan


def _format_footage_for_llm(footage: List[AnalyzedClip]) -> str:
    """Format analyzed footage into a text summary for LLM."""
    parts = []
    for i, clip in enumerate(footage):
        name = os.path.basename(clip.file_path)
        parts.append(f"\nClip {i + 1}: {name}")
        parts.append(f"  Duration: {clip.duration:.1f}s")
        parts.append(f"  Has speech: {clip.has_speech}")
        if clip.keywords_found:
            parts.append(f"  Keywords found: {', '.join(clip.keywords_found)}")
        if clip.transcript_text:
            text = clip.transcript_text[:300]
            parts.append(f"  Transcript: {text}")
        if clip.highlights:
            for j, hl in enumerate(clip.highlights[:3]):
                parts.append(
                    f"  Highlight {j + 1}: [{hl['start']:.1f}-{hl['end']:.1f}s] "
                    f"{hl.get('title', '')}"
                )
        parts.append(f"  Quality score: {clip.quality_score:.2f}")
    return "\n".join(parts)


def _parse_plan_response(
    response_text: str,
    footage: List[AnalyzedClip],
    brief: RoughCutBrief,
) -> RoughCutPlan:
    """Parse LLM JSON response into a RoughCutPlan."""
    # Extract JSON from response
    json_data = _extract_json(response_text)
    if not json_data:
        logger.warning("Could not parse LLM response as JSON, using fallback")
        return _fallback_plan(brief, footage)

    clips_data = json_data.get("clips", [])
    if not clips_data:
        return _fallback_plan(brief, footage)

    # Build filename -> AnalyzedClip lookup
    file_lookup = {}
    for clip in footage:
        name = os.path.basename(clip.file_path)
        file_lookup[name] = clip
        file_lookup[clip.file_path] = clip

    planned_clips = []
    for i, cd in enumerate(clips_data):
        source = cd.get("source_file", "")
        # Resolve source file
        ac = file_lookup.get(source)
        if ac is None:
            # Try matching by partial name
            for key, val in file_lookup.items():
                if source in key or key in source:
                    ac = val
                    break
        if ac is None and footage:
            ac = footage[i % len(footage)]

        source_path = ac.file_path if ac else source
        start = float(cd.get("start", 0))
        end = float(cd.get("end", min(start + 10, ac.duration if ac else 10)))

        # Clamp to clip duration
        if ac and end > ac.duration:
            end = ac.duration

        planned_clips.append(PlannedClip(
            source_file=source_path,
            start=start,
            end=end,
            order=i,
            justification=cd.get("justification", ""),
            score=float(cd.get("score", 0.5)),
            clip_type=cd.get("clip_type", "content"),
            transcript_text=cd.get("transcript_text", ""),
        ))

    total_duration = sum(c.duration for c in planned_clips)
    narrative = json_data.get("narrative_summary", "")

    return RoughCutPlan(
        clips=planned_clips,
        brief=brief,
        total_duration=total_duration,
        narrative_summary=narrative,
    )


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response text (may contain markdown fences)."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fence
    import re
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError, IndexError):
                continue

    return None


def _fallback_plan(
    brief: RoughCutBrief,
    footage: List[AnalyzedClip],
) -> RoughCutPlan:
    """Create a simple plan when LLM is unavailable.

    Selects clips by quality score, fits to target duration.
    """
    if not footage:
        return RoughCutPlan(brief=brief)

    # Sort by quality score descending
    sorted_footage = sorted(footage, key=lambda c: c.quality_score, reverse=True)

    clips = []
    remaining = brief.duration
    order = 0

    for clip in sorted_footage:
        if remaining <= 0:
            break

        # Use highlights if available
        if clip.highlights:
            for hl in clip.highlights:
                if remaining <= 0:
                    break
                dur = hl.get("end", 0) - hl.get("start", 0)
                if dur > 0:
                    clips.append(PlannedClip(
                        source_file=clip.file_path,
                        start=hl["start"],
                        end=hl["end"],
                        order=order,
                        justification=f"Highlight: {hl.get('title', 'top moment')}",
                        score=clip.quality_score,
                        clip_type="content",
                    ))
                    remaining -= dur
                    order += 1
        else:
            # Use first portion of clip
            use_dur = min(clip.duration, remaining, 30.0)
            if use_dur > 0:
                clips.append(PlannedClip(
                    source_file=clip.file_path,
                    start=0.0,
                    end=use_dur,
                    order=order,
                    justification="Selected by quality score",
                    score=clip.quality_score,
                    clip_type="content",
                ))
                remaining -= use_dur
                order += 1

    total_duration = sum(c.duration for c in clips)

    return RoughCutPlan(
        clips=clips,
        brief=brief,
        total_duration=total_duration,
        narrative_summary="Fallback plan: clips ordered by quality score",
    )


# ---------------------------------------------------------------------------
# Execute Plan
# ---------------------------------------------------------------------------
def execute_plan(
    plan: RoughCutPlan,
    out_path: str = "",
    on_progress: Optional[Callable] = None,
) -> RoughCutResult:
    """Assemble the rough cut from a plan.

    Extracts each clip segment and concatenates them via FFmpeg.
    Optionally adds captions and music if specified in the brief.

    Args:
        plan: RoughCutPlan from generate_plan().
        out_path: Output file path (auto-generated if empty).
        on_progress: Progress callback(pct, msg).

    Returns:
        RoughCutResult with output path and stats.
    """
    if not plan.clips:
        raise ValueError("Plan has no clips")

    # Validate all source files exist
    for clip in plan.clips:
        if not os.path.isfile(clip.source_file):
            raise FileNotFoundError(
                f"Source file not found: {clip.source_file}"
            )

    if on_progress:
        on_progress(5, f"Assembling {len(plan.clips)} clips...")

    # Determine output path
    if not out_path:
        first_file = plan.clips[0].source_file
        out_dir = os.path.dirname(os.path.abspath(first_file))
        out_path = output_path(first_file, "rough_cut", out_dir)

    tmp_dir = tempfile.mkdtemp(prefix="opencut_roughcut_")
    segment_files = []
    total = len(plan.clips)

    try:
        # Extract each clip segment
        for i, clip in enumerate(plan.clips):
            if on_progress:
                pct = 5 + int((i / total) * 60)
                on_progress(pct, f"Extracting clip {i + 1}/{total}...")

            seg_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
            cmd = (FFmpegCmd()
                   .input(clip.source_file, ss=clip.start, to=clip.end)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .faststart()
                   .output(seg_path)
                   .build())
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if on_progress:
            on_progress(70, "Concatenating clips...")

        # Concatenate
        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg_path in segment_files:
                safe = seg_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_path)
               .copy_streams()
               .faststart()
               .output(out_path)
               .build())
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(85, "Finalizing...")

        result = RoughCutResult(
            output_path=out_path,
            duration=plan.total_duration,
            clip_count=len(plan.clips),
            plan=plan,
        )

        # Optional: add captions
        if plan.brief and plan.brief.include_captions:
            try:
                caption_path = _add_captions(out_path, tmp_dir, on_progress)
                result.caption_path = caption_path
            except Exception as e:
                logger.warning("Caption generation failed: %s", e)

        # Generate EDL
        try:
            edl_path = os.path.join(tmp_dir, "rough_cut.edl")
            _export_plan_edl(plan, edl_path)
            # Copy to persistent location
            persist_edl = out_path.rsplit(".", 1)[0] + ".edl"
            import shutil
            shutil.copy2(edl_path, persist_edl)
            result.edl_path = persist_edl
        except Exception as e:
            logger.warning("EDL export failed: %s", e)

        if on_progress:
            on_progress(100, f"Rough cut complete: {result.duration:.1f}s, {result.clip_count} clips")

        return result

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _add_captions(
    video_path: str,
    tmp_dir: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Add captions to the rough cut."""
    from opencut.core.captions import transcribe

    if on_progress:
        on_progress(88, "Generating captions...")

    result = transcribe(video_path)
    if not result or not result.segments:
        return ""

    # Export as SRT
    srt_path = os.path.join(tmp_dir, "rough_cut.srt")
    lines = []
    for i, seg in enumerate(result.segments):
        lines.append(str(i + 1))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Copy to persistent location next to video
    persist_srt = video_path.rsplit(".", 1)[0] + ".srt"
    import shutil
    shutil.copy2(srt_path, persist_srt)
    return persist_srt


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _export_plan_edl(plan: RoughCutPlan, edl_path: str) -> None:
    """Export rough cut plan as CMX 3600 EDL."""
    lines = ["TITLE: AI Rough Cut\nFCM: NON-DROP FRAME\n\n"]
    record_in = 0.0

    for i, clip in enumerate(plan.clips):
        record_out = record_in + clip.duration
        filename = os.path.basename(clip.source_file)
        lines.append(
            f"{i + 1:03d}  AX  V  C  "
            f"{_fmt_edl_tc(clip.start)} {_fmt_edl_tc(clip.end)} "
            f"{_fmt_edl_tc(record_in)} {_fmt_edl_tc(record_out)}\n"
        )
        lines.append(f"* FROM CLIP NAME: {filename}\n")
        if clip.justification:
            lines.append(f"* COMMENT: {clip.justification}\n")
        lines.append("\n")
        record_in = record_out

    with open(edl_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _fmt_edl_tc(seconds: float) -> str:
    """Format seconds as EDL timecode HH:MM:SS:FF (30fps)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1) * 30)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def rough_cut_from_brief(
    file_paths: List[str],
    brief_text: str,
    out_path: str = "",
    llm_config=None,
    on_progress: Optional[Callable] = None,
) -> RoughCutResult:
    """Full automatic rough cut pipeline from brief text.

    Stages:
    1. Parse brief
    2. Analyze all footage (transcribe, detect speech, find keywords)
    3. Generate plan via LLM
    4. Execute plan (assemble video)

    Args:
        file_paths: List of source video/audio files.
        brief_text: Natural language brief describing the desired output.
        out_path: Output file path (auto-generated if empty).
        llm_config: LLMConfig for LLM queries.
        on_progress: Progress callback(pct, msg).

    Returns:
        RoughCutResult with output video and metadata.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    if on_progress:
        on_progress(2, "Parsing creative brief...")

    # Stage 1: Parse brief via LLM
    brief = _parse_brief(brief_text, llm_config)

    if on_progress:
        on_progress(5, f"Analyzing {len(file_paths)} clips...")

    # Stage 2: Analyze footage
    def _analysis_progress(pct, msg=""):
        if on_progress:
            scaled = 5 + int(pct * 0.35)
            on_progress(scaled, msg)

    analyzed = analyze_footage(
        file_paths,
        keywords=brief.keywords,
        on_progress=_analysis_progress,
    )

    if on_progress:
        on_progress(40, "Generating rough cut plan...")

    # Stage 3: Generate plan
    def _plan_progress(pct, msg=""):
        if on_progress:
            scaled = 40 + int(pct * 0.20)
            on_progress(scaled, msg)

    plan = generate_plan(
        brief, analyzed,
        llm_config=llm_config,
        on_progress=_plan_progress,
    )

    if not plan.clips:
        raise ValueError("LLM could not generate a valid plan from the footage")

    if on_progress:
        on_progress(60, f"Executing plan: {len(plan.clips)} clips...")

    # Stage 4: Execute plan
    def _exec_progress(pct, msg=""):
        if on_progress:
            scaled = 60 + int(pct * 0.40)
            on_progress(scaled, msg)

    result = execute_plan(plan, out_path=out_path, on_progress=_exec_progress)

    if on_progress:
        on_progress(100, f"Rough cut complete: {result.duration:.1f}s")

    return result


def _parse_brief(brief_text: str, llm_config=None) -> RoughCutBrief:
    """Parse natural language brief into structured RoughCutBrief.

    Uses LLM to extract structured data; falls back to keyword parsing.
    """
    brief = RoughCutBrief(goal=brief_text)

    # Try LLM parsing
    try:
        from opencut.core.llm import LLMConfig, query_llm

        if llm_config is None:
            llm_config = LLMConfig()

        prompt = (
            f"Parse this video editing brief into structured JSON:\n\n"
            f"\"{brief_text}\"\n\n"
            f"Return JSON with: goal, style (narrative/documentary/highlight/tutorial), "
            f"duration (seconds), keywords (list), tone (upbeat/dramatic/calm/humorous/neutral), "
            f"pacing (slow/medium/fast)"
        )
        response = query_llm(prompt=prompt, config=llm_config)
        if response and response.text and not response.text.startswith("LLM error:"):
            parsed = _extract_json(response.text)
            if parsed:
                brief = RoughCutBrief.from_dict(parsed)
                brief.goal = brief.goal or brief_text
                return brief
    except Exception as e:
        logger.warning("LLM brief parsing failed: %s", e)

    # Fallback: simple keyword extraction
    text_lower = brief_text.lower()

    # Detect style
    for style in ("documentary", "highlight", "tutorial", "narrative"):
        if style in text_lower:
            brief.style = style
            break

    # Detect tone
    for tone in ("upbeat", "dramatic", "calm", "humorous"):
        if tone in text_lower:
            brief.tone = tone
            break

    # Detect pacing
    for pacing in ("slow", "fast"):
        if pacing in text_lower:
            brief.pacing = pacing
            break

    # Extract duration hints
    import re
    dur_match = re.search(r"(\d+)\s*(?:second|sec|s\b)", text_lower)
    if dur_match:
        brief.duration = float(dur_match.group(1))
    else:
        dur_match = re.search(r"(\d+)\s*(?:minute|min|m\b)", text_lower)
        if dur_match:
            brief.duration = float(dur_match.group(1)) * 60

    # Extract keywords (nouns after common prepositions)
    kw_match = re.findall(r"(?:about|featuring|with|showing)\s+(\w+)", text_lower)
    if kw_match:
        brief.keywords = kw_match

    return brief
