"""
OpenCut Transcript Edit & Rough Cut Routes

Transcript-based editing (1.1) and AI rough cut assembly (21.3).
"""

import logging

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

transcript_edit_bp = Blueprint("transcript_edit", __name__)


# ---------------------------------------------------------------------------
# Transcript-Based Editing Routes
# ---------------------------------------------------------------------------

@transcript_edit_bp.route("/transcript-edit/build-map", methods=["POST"])
@require_csrf
@async_job("transcript_edit_map", filepath_required=False)
def build_transcript_map(job_id, filepath, data):
    """Build a bidirectional transcript<->timeline map.

    Expects JSON body:
      transcript_json: Transcript data (WhisperX format or segment list)
      source_file: (optional) Source video file path
    """
    from opencut.core.transcript_edit import build_transcript_map

    transcript_json = data.get("transcript_json")
    if not transcript_json:
        raise ValueError("transcript_json is required")

    source_file = data.get("source_file", filepath or "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = build_transcript_map(
        transcript_json=transcript_json,
        source_file=source_file,
        on_progress=_on_progress,
    )

    return {"transcript_map": result.to_dict()}


@transcript_edit_bp.route("/transcript-edit/apply-edits", methods=["POST"])
@require_csrf
@async_job("transcript_edit_apply")
def apply_text_edits(job_id, filepath, data):
    """Apply text-based edits to a video file.

    Expects JSON body:
      filepath: Source video file path
      transcript_map: Transcript map data (from build-map)
      edits: List of edit operations [{edit_type, word_indices?, new_order?}]
      output_dir: (optional) Output directory
    """
    from opencut.core.transcript_edit import (
        TextEdit,
    )
    from opencut.core.transcript_edit import (
        apply_text_edits as _apply_text_edits,
    )

    map_data = data.get("transcript_map")
    if not map_data:
        raise ValueError("transcript_map is required")

    edits_data = data.get("edits", [])
    if not edits_data:
        raise ValueError("At least one edit is required")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    # Reconstruct TranscriptMap from dict
    tmap = _reconstruct_map(map_data)

    # Build TextEdit objects
    edits = []
    for ed in edits_data:
        edits.append(TextEdit(
            edit_type=ed.get("edit_type", "delete"),
            word_indices=ed.get("word_indices", []),
            paragraph_indices=ed.get("paragraph_indices", []),
            new_order=ed.get("new_order", []),
        ))

    out_path = ""
    if output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _output_path
        out_path = _output_path(filepath, "transcript_edit", effective_dir)

    result = _apply_text_edits(
        video_path=filepath,
        transcript_map=tmap,
        edits=edits,
        out_path=out_path,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "duration": result.duration,
        "cut_count": result.cut_count,
        "removed_duration": result.removed_duration,
    }


@transcript_edit_bp.route("/transcript-edit/delete-words", methods=["POST"])
@require_csrf
@async_job("transcript_edit_delete", filepath_required=False)
def delete_words_route(job_id, filepath, data):
    """Delete words from transcript map and return resulting cut segments.

    Expects JSON body:
      transcript_map: Transcript map data
      word_indices: List of word indices to delete
    """
    from opencut.core.transcript_edit import (
        delete_words,
    )

    map_data = data.get("transcript_map")
    if not map_data:
        raise ValueError("transcript_map is required")

    word_indices = data.get("word_indices", [])
    if not word_indices:
        raise ValueError("word_indices is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    tmap = _reconstruct_map(map_data)
    segments = delete_words(tmap, word_indices, on_progress=_on_progress)

    return {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "source_word_start": s.source_word_start,
                "source_word_end": s.source_word_end,
            }
            for s in segments
        ],
        "updated_map": tmap.to_dict(),
    }


@transcript_edit_bp.route("/transcript-edit/rearrange", methods=["POST"])
@require_csrf
@async_job("transcript_edit_rearrange", filepath_required=False)
def rearrange_route(job_id, filepath, data):
    """Rearrange paragraphs in transcript and return new cut segments.

    Expects JSON body:
      transcript_map: Transcript map data
      new_order: List of paragraph indices in new order
    """
    from opencut.core.transcript_edit import rearrange_paragraphs

    map_data = data.get("transcript_map")
    if not map_data:
        raise ValueError("transcript_map is required")

    new_order = data.get("new_order", [])
    if not new_order:
        raise ValueError("new_order is required")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    tmap = _reconstruct_map(map_data)
    segments = rearrange_paragraphs(tmap, new_order, on_progress=_on_progress)

    return {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "source_word_start": s.source_word_start,
                "source_word_end": s.source_word_end,
            }
            for s in segments
        ],
    }


@transcript_edit_bp.route("/transcript-edit/export", methods=["POST"])
@require_csrf
@async_job("transcript_edit_export", filepath_required=False)
def export_route(job_id, filepath, data):
    """Export edited sequence as EDL, OTIO, or JSON.

    Expects JSON body:
      video_path: (optional) Source video path for OTIO references
      transcript_map: Transcript map data
      edits: List of edit operations
      format: Export format (otio, edl, json)
      output_dir: (optional) Output directory
    """
    from opencut.core.transcript_edit import (
        TextEdit,
        export_edited_sequence,
    )

    map_data = data.get("transcript_map")
    if not map_data:
        raise ValueError("transcript_map is required")

    edits_data = data.get("edits", [])
    fmt = data.get("format", "otio")
    video_path = data.get("video_path", filepath or "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    tmap = _reconstruct_map(map_data)

    edits = []
    for ed in edits_data:
        edits.append(TextEdit(
            edit_type=ed.get("edit_type", "delete"),
            word_indices=ed.get("word_indices", []),
            paragraph_indices=ed.get("paragraph_indices", []),
            new_order=ed.get("new_order", []),
        ))

    result = export_edited_sequence(
        video_path=video_path,
        transcript_map=tmap,
        edits=edits,
        format=fmt,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Rough Cut Routes
# ---------------------------------------------------------------------------

@transcript_edit_bp.route("/rough-cut/analyze", methods=["POST"])
@require_csrf
@async_job("rough_cut_analyze", filepath_required=False)
def analyze_footage_route(job_id, filepath, data):
    """Analyze footage files for rough cut assembly.

    Expects JSON body:
      file_paths: List of video/audio file paths
      keywords: (optional) Keywords to look for
    """
    from opencut.core.rough_cut import analyze_footage

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("file_paths is required (list of file paths)")

    # Validate all paths
    validated = []
    for fp in file_paths:
        validated.append(validate_filepath(fp))

    keywords = data.get("keywords", [])

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    analyzed = analyze_footage(
        validated,
        keywords=keywords,
        on_progress=_on_progress,
    )

    return {
        "clips": [c.to_dict() for c in analyzed],
        "total_clips": len(analyzed),
    }


@transcript_edit_bp.route("/rough-cut/plan", methods=["POST"])
@require_csrf
@async_job("rough_cut_plan", filepath_required=False)
def generate_plan_route(job_id, filepath, data):
    """Generate a rough cut plan from analyzed footage and brief.

    Expects JSON body:
      brief: Creative brief {goal, style, duration, keywords, tone, pacing}
      analyzed_footage: List of analyzed clip dicts (from /rough-cut/analyze)
      llm_config: (optional) LLM configuration
    """
    from opencut.core.rough_cut import (
        AnalyzedClip,
        RoughCutBrief,
        generate_plan,
    )

    brief_data = data.get("brief", {})
    if not brief_data or not brief_data.get("goal"):
        raise ValueError("brief with goal is required")

    footage_data = data.get("analyzed_footage", [])
    if not footage_data:
        raise ValueError("analyzed_footage is required")

    brief = RoughCutBrief.from_dict(brief_data)

    # Reconstruct AnalyzedClip objects
    analyzed = []
    for fd in footage_data:
        clip = AnalyzedClip(
            file_path=fd.get("file_path", ""),
            duration=float(fd.get("duration", 0)),
            transcript_text=fd.get("transcript_text", ""),
            transcript_segments=fd.get("transcript_segments", []),
            keywords_found=fd.get("keywords_found", []),
            highlights=fd.get("highlights", []),
            has_speech=bool(fd.get("has_speech", False)),
            quality_score=float(fd.get("quality_score", 0)),
        )
        analyzed.append(clip)

    llm_config = None
    llm_data = data.get("llm_config")
    if llm_data:
        from opencut.core.llm import LLMConfig
        llm_config = LLMConfig(
            provider=llm_data.get("provider", "ollama"),
            model=llm_data.get("model", "llama3.2"),
            api_key=llm_data.get("api_key", ""),
            base_url=llm_data.get("base_url", "http://localhost:11434"),
            temperature=float(llm_data.get("temperature", 0.3)),
            max_tokens=int(llm_data.get("max_tokens", 2000)),
        )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    plan = generate_plan(
        brief, analyzed,
        llm_config=llm_config,
        on_progress=_on_progress,
    )

    return plan.to_dict()


@transcript_edit_bp.route("/rough-cut/execute", methods=["POST"])
@require_csrf
@async_job("rough_cut_execute", filepath_required=False)
def execute_plan_route(job_id, filepath, data):
    """Execute a rough cut plan to assemble the final video.

    Expects JSON body:
      plan: RoughCutPlan dict (from /rough-cut/plan)
      output_dir: (optional) Output directory
    """
    from opencut.core.rough_cut import (
        PlannedClip,
        RoughCutBrief,
        RoughCutPlan,
        execute_plan,
    )

    plan_data = data.get("plan", {})
    if not plan_data or not plan_data.get("clips"):
        raise ValueError("plan with clips is required")

    # Reconstruct plan
    clips = []
    for i, cd in enumerate(plan_data["clips"]):
        source = cd.get("source_file", "")
        if source:
            source = validate_filepath(source)
        clips.append(PlannedClip(
            source_file=source,
            start=float(cd.get("start", 0)),
            end=float(cd.get("end", 0)),
            order=i,
            justification=cd.get("justification", ""),
            score=float(cd.get("score", 0.5)),
            clip_type=cd.get("clip_type", "content"),
        ))

    brief = None
    if plan_data.get("brief"):
        brief = RoughCutBrief.from_dict(plan_data["brief"])

    plan = RoughCutPlan(
        clips=clips,
        brief=brief,
        total_duration=sum(c.duration for c in clips),
        narrative_summary=plan_data.get("narrative_summary", ""),
    )

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = ""
    if output_dir and clips:
        effective_dir = _resolve_output_dir(clips[0].source_file, output_dir)
        from opencut.helpers import output_path as _output_path
        out_path = _output_path(clips[0].source_file, "rough_cut", effective_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = execute_plan(plan, out_path=out_path, on_progress=_on_progress)
    return result.to_dict()


@transcript_edit_bp.route("/rough-cut/auto", methods=["POST"])
@require_csrf
@async_job("rough_cut_auto", filepath_required=False)
def rough_cut_auto_route(job_id, filepath, data):
    """Full automatic rough cut pipeline.

    Expects JSON body:
      file_paths: List of source video/audio file paths
      brief: Natural language brief (string) or structured brief dict
      output_dir: (optional) Output directory
      llm_config: (optional) LLM configuration
    """
    from opencut.core.rough_cut import rough_cut_from_brief

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("file_paths is required (list of file paths)")

    validated = []
    for fp in file_paths:
        validated.append(validate_filepath(fp))

    brief = data.get("brief", "")
    if isinstance(brief, dict):
        brief = brief.get("goal", "Create a rough cut from the footage")
    if not brief:
        brief = "Create a rough cut from the footage"

    llm_config = None
    llm_data = data.get("llm_config")
    if llm_data:
        from opencut.core.llm import LLMConfig
        llm_config = LLMConfig(
            provider=llm_data.get("provider", "ollama"),
            model=llm_data.get("model", "llama3.2"),
            api_key=llm_data.get("api_key", ""),
            base_url=llm_data.get("base_url", "http://localhost:11434"),
        )

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out_path = ""
    if output_dir and validated:
        effective_dir = _resolve_output_dir(validated[0], output_dir)
        from opencut.helpers import output_path as _output_path
        out_path = _output_path(validated[0], "rough_cut", effective_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = rough_cut_from_brief(
        file_paths=validated,
        brief_text=brief,
        out_path=out_path,
        llm_config=llm_config,
        on_progress=_on_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reconstruct_map(map_data: dict):
    """Reconstruct a TranscriptMap from a serialized dict."""
    from opencut.core.transcript_edit import (
        ParagraphMapping,
        TranscriptMap,
        WordMapping,
    )

    words = []
    for wd in map_data.get("words", []):
        words.append(WordMapping(
            index=int(wd.get("index", 0)),
            text=wd.get("text", ""),
            start=float(wd.get("start", 0)),
            end=float(wd.get("end", 0)),
            confidence=float(wd.get("confidence", 1.0)),
            speaker=wd.get("speaker", ""),
            paragraph_index=int(wd.get("paragraph_index", 0)),
            is_deleted=bool(wd.get("is_deleted", False)),
        ))

    paragraphs = []
    for pd in map_data.get("paragraphs", []):
        paragraphs.append(ParagraphMapping(
            index=int(pd.get("index", 0)),
            text=pd.get("text", ""),
            start=float(pd.get("start", 0)),
            end=float(pd.get("end", 0)),
            word_start_index=int(pd.get("word_start_index", 0)),
            word_end_index=int(pd.get("word_end_index", 0)),
            speaker=pd.get("speaker", ""),
        ))

    return TranscriptMap(
        words=words,
        paragraphs=paragraphs,
        total_duration=float(map_data.get("total_duration", 0)),
        language=map_data.get("language", "en"),
        source_file=map_data.get("source_file", ""),
    )
