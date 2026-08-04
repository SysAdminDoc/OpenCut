"""
OpenCut Transcript Edit & Rough Cut Routes

Transcript-based editing (1.1) and AI rough cut assembly (21.3).
"""

import logging
import uuid

from flask import Blueprint, jsonify, request

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    require_csrf,
    safe_bool,
    validate_filepath,
    validate_path,
    verify_destructive_confirm_token,
)

logger = logging.getLogger("opencut")

transcript_edit_bp = Blueprint("transcript_edit", __name__)


# ---------------------------------------------------------------------------
# Transcript-Based Editing Routes
# ---------------------------------------------------------------------------


def _correction_request_data() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _correction_segments(data: dict) -> list:
    source = data.get("segments")
    if source is None:
        source = data.get("transcript", data.get("transcript_json"))
    if isinstance(source, dict):
        source = source.get("segments")
    if not isinstance(source, list):
        raise ValueError("segments or transcript.segments is required")
    return source


def _correction_project_path(data: dict) -> str:
    project_path = data.get("project_path") or data.get("filepath") or "default"
    if not isinstance(project_path, str) or len(project_path) > 2_000:
        raise ValueError("project_path must be a string")
    return project_path.strip() or "default"


def _correction_preview(data: dict) -> tuple[dict, str, dict]:
    from opencut.core.transcript_corrections import (
        normalize_correction_rules,
        preview_transcript_corrections,
        project_identity,
    )

    project_path = _correction_project_path(data)
    rules = normalize_correction_rules(
        data.get("rules"),
        find=data.get("find"),
        replace=data.get("replace"),
        case_sensitive=safe_bool(data.get("case_sensitive", False), False),
        whole_word=safe_bool(data.get("whole_word", False), False),
    )
    if not rules:
        raise ValueError("At least one correction rule is required")
    preview = preview_transcript_corrections(_correction_segments(data), rules)
    save_to_glossary = safe_bool(data.get("save_to_glossary", False), False)
    plan = build_destructive_plan(
        "transcript_bulk_correction",
        targets=[project_identity(project_path)],
        records=[preview["summary"]],
        metadata={
            "project_id": project_identity(project_path),
            "source_hash": preview["summary"]["source_hash"],
            "rules": preview["rules"],
            "save_to_glossary": save_to_glossary,
        },
        reversible=True,
    )
    return preview, project_path, plan


def _correction_preview_response(preview: dict, plan: dict) -> dict:
    return {
        "dry_run": True,
        "preview": True,
        "segments": preview["segments"],
        "corrected_segments": preview["segments"],
        "changes": preview["changes"],
        "summary": preview["summary"],
        "rules": preview["rules"],
        "plan": plan,
        "confirm_token": plan["confirm_token"],
    }


@transcript_edit_bp.route("/transcript-edit/corrections/preview", methods=["POST"])
@require_csrf
def preview_transcript_corrections_route():
    """Preview literal bulk transcript corrections without mutating state."""
    try:
        preview, _project_path, plan = _correction_preview(_correction_request_data())
        return jsonify(_correction_preview_response(preview, plan))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcript correction preview failed")
        return jsonify({"error": str(exc), "code": "CORRECTION_PREVIEW_FAILED"}), 500


@transcript_edit_bp.route("/transcript-edit/corrections/apply", methods=["POST"])
@require_csrf
def apply_transcript_corrections_route():
    """Apply a reviewed transcript correction and persist an undo record."""
    try:
        data = _correction_request_data()
        preview, project_path, plan = _correction_preview(data)
        if safe_bool(data.get("dry_run", data.get("preview", False)), False):
            return jsonify(_correction_preview_response(preview, plan))
        if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409

        from opencut.user_data import (
            load_transcript_glossary,
            save_transcript_correction_revision,
            save_transcript_glossary,
        )

        revision = {
            "id": uuid.uuid4().hex,
            "before_segments": preview["original_segments"],
            "after_segments": preview["segments"],
            "changes": preview["changes"],
            "summary": preview["summary"],
            "rules": preview["rules"],
        }
        save_transcript_correction_revision(project_path, revision)

        glossary = load_transcript_glossary(project_path)
        if safe_bool(data.get("save_to_glossary", False), False):
            from opencut.core.transcript_corrections import merge_glossary_rules

            glossary = save_transcript_glossary(
                project_path,
                merge_glossary_rules(glossary, preview["rules"]),
            )
        return jsonify(
            {
                "applied": True,
                "segments": preview["segments"],
                "corrected_segments": preview["segments"],
                "changes": preview["changes"],
                "summary": preview["summary"],
                "rules": preview["rules"],
                "undo_token": revision["id"],
                "glossary": glossary,
                "plan": plan,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcript correction apply failed")
        return jsonify({"error": str(exc), "code": "CORRECTION_APPLY_FAILED"}), 500


@transcript_edit_bp.route("/transcript-edit/corrections/undo", methods=["POST"])
@require_csrf
def undo_transcript_corrections_route():
    """Return the prior transcript snapshot for a previously applied correction."""
    try:
        data = _correction_request_data()
        project_path = _correction_project_path(data)
        undo_token = data.get("undo_token")
        if not isinstance(undo_token, str) or not undo_token:
            raise ValueError("undo_token is required")
        from opencut.user_data import (
            get_transcript_correction_revision,
            mark_transcript_correction_undone,
        )

        revision = get_transcript_correction_revision(project_path, undo_token)
        if revision is None:
            return jsonify({"error": "Correction undo record not found", "code": "UNDO_NOT_FOUND"}), 404
        restored = mark_transcript_correction_undone(project_path, undo_token) or revision
        return jsonify(
            {
                "undone": True,
                "segments": restored.get("before_segments", []),
                "summary": restored.get("summary", {}),
                "undo_token": undo_token,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcript correction undo failed")
        return jsonify({"error": str(exc), "code": "CORRECTION_UNDO_FAILED"}), 500


@transcript_edit_bp.route("/transcript-edit/glossary", methods=["GET"])
def get_transcript_glossary_route():
    """Return the persisted correction glossary for one project."""
    try:
        project_path = _correction_project_path(request.args.to_dict())
        from opencut.user_data import load_transcript_glossary

        return jsonify({"project_id": project_path, "rules": load_transcript_glossary(project_path)})
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400


@transcript_edit_bp.route("/transcript-edit/glossary", methods=["POST"])
@require_csrf
def save_transcript_glossary_route():
    """Add or replace persisted correction terms for one project."""
    try:
        data = _correction_request_data()
        project_path = _correction_project_path(data)
        from opencut.core.transcript_corrections import (
            merge_glossary_rules,
            normalize_correction_rules,
        )
        from opencut.user_data import load_transcript_glossary, save_transcript_glossary

        rules = normalize_correction_rules(
            data.get("rules"),
            find=data.get("find"),
            replace=data.get("replace"),
            case_sensitive=safe_bool(data.get("case_sensitive", False), False),
            whole_word=safe_bool(data.get("whole_word", False), False),
        )
        if not rules:
            raise ValueError("At least one glossary rule is required")
        if safe_bool(data.get("replace_all", False), False):
            glossary = save_transcript_glossary(project_path, rules)
        else:
            glossary = save_transcript_glossary(
                project_path,
                merge_glossary_rules(load_transcript_glossary(project_path), rules),
            )
        return jsonify({"saved": True, "rules": glossary})
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcript glossary save failed")
        return jsonify({"error": str(exc), "code": "GLOSSARY_SAVE_FAILED"}), 500

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
            has_speech=safe_bool(fd.get("has_speech"), False),
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
            is_deleted=safe_bool(wd.get("is_deleted"), False),
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
