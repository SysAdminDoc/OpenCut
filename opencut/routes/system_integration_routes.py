"""System integration routes registered on the shared system blueprint."""

from .system import (
    _is_cancelled,
    _update_job,
    async_job,
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    jsonify,
    logger,
    require_csrf,
    safe_bool,
    safe_error,
    safe_float,
    safe_int,
    system_bp,
    validate_filepath,
    validate_path,
    verify_destructive_confirm_token,
)


# ---------------------------------------------------------------------------
# DaVinci Resolve Integration
# ---------------------------------------------------------------------------
@system_bp.route("/resolve/status", methods=["GET"])
def resolve_status():
    """Check DaVinci Resolve connection status and project info."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"connected": False, "message": "DaVinci Resolve not running or not accessible"})
        info = bridge.get_project_info()
        return jsonify({"connected": True, "project": info})
    except ImportError:
        return jsonify({"connected": False, "message": "Resolve scripting module not found"})
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)})


@system_bp.route("/resolve/media", methods=["GET"])
def resolve_media():
    """Get all clips from the Resolve media pool."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        clips = bridge.get_media_pool_clips()
        return jsonify({"media": clips, "count": len(clips)})
    except Exception as e:
        return safe_error(e, "resolve_media")


@system_bp.route("/resolve/import", methods=["POST"])
@require_csrf
def resolve_import():
    """Import one file, or a list of files, into the Resolve media pool."""
    data = get_json_dict()
    filepath = str(data.get("filepath", "") or "").strip()
    raw_paths = data.get("paths", [])
    bin_name = data.get("bin_name", "OpenCut Output")

    if raw_paths not in (None, []) and not isinstance(raw_paths, list):
        return jsonify({"error": "paths must be a list"}), 400

    if isinstance(raw_paths, list) and raw_paths:
        cleaned_paths = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                return jsonify({"error": "paths must contain only non-empty strings"}), 400
            try:
                cleaned_paths.append(validate_filepath(raw_path.strip()))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        try:
            from opencut.core.resolve_integration import resolve_import_media

            return jsonify(resolve_import_media(cleaned_paths))
        except Exception as exc:
            return safe_error(exc, "resolve_import")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        success = bridge.import_file(filepath, bin_name=bin_name)
        if success:
            return jsonify({"success": True, "message": f"Imported to {bin_name}"})
        from opencut.errors import error_response
        return error_response("OPERATION_FAILED", "Resolve import failed",
                              status=500, suggestion="Check that the file format is supported by DaVinci Resolve.")
    except Exception as e:
        return safe_error(e, "resolve_import")


@system_bp.route("/resolve/markers", methods=["POST"])
@require_csrf
def resolve_markers():
    """Add markers to the current Resolve timeline."""
    data = get_json_dict()
    markers = data.get("markers", [])
    if not markers:
        return jsonify({"error": "No markers provided"}), 400
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        added = bridge.add_markers(markers)
        return jsonify({"added": added, "total": len(markers)})
    except Exception as e:
        return safe_error(e, "resolve_markers")


@system_bp.route("/resolve/timeline", methods=["GET"])
def resolve_timeline():
    """Get current Resolve timeline info."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        info = bridge.get_timeline_info()
        if info:
            return jsonify(info)
        return jsonify({"error": "No timeline open"}), 404
    except Exception as e:
        return safe_error(e, "resolve_timeline")


# ---------------------------------------------------------------------------
# Chat-Driven Editing Assistant
# ---------------------------------------------------------------------------
@system_bp.route("/chat", methods=["POST"])
@require_csrf
def chat_message():
    """Send a message to the AI editing assistant.

    Maintains conversation context across messages. Returns the assistant's
    response with any editing actions to execute.
    """
    data = get_json_dict()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    filepath = data.get("filepath", "")
    clip_info = data.get("clip_info", {})

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long (max 2000 chars)"}), 400

    try:
        from opencut.core.chat_editor import chat
        from opencut.core.llm import LLMConfig

        # Build LLM config from settings or request
        provider = data.get("llm_provider", "ollama")
        if provider not in ("ollama", "openai", "anthropic", "gemini"):
            return jsonify({"error": "Invalid provider", "code": "INVALID_INPUT", "suggestion": "Use ollama, openai, anthropic, or gemini"}), 400
        model = data.get("llm_model", "")
        api_key = data.get("llm_api_key", "")

        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
        )

        result = chat(
            session_id=session_id,
            user_message=message,
            filepath=filepath,
            clip_info=clip_info,
            llm_config=llm_config,
        )

        return jsonify(result)

    except ImportError as e:
        return jsonify({"error": f"Chat requires LLM module: {e}"}), 400
    except Exception as e:
        logger.exception("Chat error")
        return safe_error(e, "chat_message")


@system_bp.route("/chat/clear", methods=["POST"])
@require_csrf
def chat_clear():
    """Clear a chat session's history."""
    data = get_json_dict()
    session_id = data.get("session_id", "default")
    try:
        from opencut.core.chat_editor import clear_session, list_sessions
        session = next((item for item in list_sessions() if item.get("session_id") == session_id), None)
        records = []
        if session and int(session.get("message_count", 0) or 0) > 0:
            records.append({
                "id": str(session_id),
                "category": "chat-session",
                "type": "session",
                "message_count": int(session.get("message_count", 0) or 0),
                "filepath": str(session.get("filepath", "")),
                "bytes": 0,
                "reversible": False,
            })
        plan = build_destructive_plan(
            "chat.clear_session",
            records=records,
            metadata={"route": "/chat/clear", "session_id": str(session_id)},
            reversible=False,
        )
        dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
        if dry_run:
            return jsonify({
                "success": True,
                "dry_run": True,
                "message": "Session clear previewed",
                "would_clear": records[0]["message_count"] if records else 0,
                "destructive_plan": plan,
                "confirm_token": plan["confirm_token"],
            })
        if records and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409
        clear_session(session_id)
        return jsonify({"success": True, "message": "Session cleared", "destructive_plan": plan})
    except Exception as e:
        return safe_error(e, "chat_clear")


@system_bp.route("/chat/sessions", methods=["GET"])
def chat_sessions():
    """List all active chat sessions."""
    try:
        from opencut.core.chat_editor import list_sessions
        return jsonify({"sessions": list_sessions()})
    except Exception as e:
        return safe_error(e, "chat_sessions")


# ---------------------------------------------------------------------------
# Multimodal Diarization (Audio + Face)
# ---------------------------------------------------------------------------
@system_bp.route("/video/multimodal-diarize", methods=["POST"])
@require_csrf
@async_job("multimodal-diarize", rate_limit_key="ai_gpu")
def video_multimodal_diarize(job_id, filepath, data):
    """Run multimodal speaker diarization combining audio and face recognition.

    Returns speaker segments enriched with face IDs for accurate multicam switching.
    """
    num_speakers = data.get("num_speakers")
    if num_speakers is not None:
        num_speakers = safe_int(num_speakers, None, min_val=1, max_val=20)
    sample_fps = safe_float(data.get("sample_fps", 2.0), 2.0, min_val=0.5, max_val=10.0)
    min_face_confidence = safe_float(data.get("min_face_confidence", 0.5), 0.5, min_val=0.1, max_val=1.0)

    from opencut.core.multimodal_diarize import multimodal_diarize

    def _on_progress(pct, msg=""):
        if _is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        _update_job(job_id, progress=pct, message=msg)

    result = multimodal_diarize(
        filepath,
        num_speakers=num_speakers,
        sample_fps=sample_fps,
        min_face_confidence=min_face_confidence,
        on_progress=_on_progress,
    )

    # Build enriched cuts
    cuts = result.to_enriched_cuts()

    return {
        "speaker_segments": result.speaker_segments,
        "face_segments": [
            {"face_id": f.face_id, "start": f.start, "end": f.end,
             "confidence": f.confidence, "bbox": list(f.bbox)}
            for f in result.face_segments
        ],
        "mappings": [
            {"speaker": m.speaker, "face_id": m.face_id,
             "confidence": m.confidence, "overlap_seconds": m.overlap_seconds}
            for m in result.mappings
        ],
        "cuts": cuts,
        "num_speakers": result.num_speakers,
        "num_faces": result.num_faces,
    }


# ---------------------------------------------------------------------------
# AI B-Roll Generation (Text-to-Video)
# ---------------------------------------------------------------------------
def _validate_broll_prompt(data):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return "No prompt provided"
    if len(prompt) > 500:
        return "Prompt too long (max 500 chars)"
    return None


@system_bp.route("/video/broll-generate", methods=["POST"])
@require_csrf
@async_job("broll-generate", filepath_required=False, pre_validate=_validate_broll_prompt, rate_limit_key="ai_gpu")
def video_broll_generate(job_id, filepath, data):
    """Generate a B-roll video clip from a text description using AI."""
    prompt = data.get("prompt", "").strip()

    if not prompt:
        raise ValueError("No prompt provided")
    if len(prompt) > 500:
        raise ValueError("Prompt too long (max 500 chars)")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    backend = data.get("backend", "auto")
    _valid_backends = {"auto", "stable_diffusion", "dall_e", "replicate"}
    if backend not in _valid_backends:
        backend = "auto"
    seed = data.get("seed")
    if seed is not None:
        seed = safe_int(seed, None, min_val=0, max_val=2**31)
    reference_image = data.get("reference_image", "")
    if reference_image:
        reference_image = validate_filepath(reference_image)

    from opencut.core.broll_generate import generate_broll

    def _on_progress(pct, msg=""):
        if _is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = None
    if output_dir:
        try:
            effective_dir = validate_path(output_dir)
        except ValueError:
            effective_dir = None

    result = generate_broll(
        prompt=prompt,
        output_dir=effective_dir,
        backend=backend,
        seed=seed,
        reference_image=reference_image if reference_image else None,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "prompt": result.prompt,
        "duration": result.duration,
        "resolution": result.resolution,
        "backend": result.backend,
        "generation_time": result.generation_time,
        "seed": result.seed,
    }


@system_bp.route("/video/broll-backends", methods=["GET"])
def video_broll_backends():
    """List available text-to-video backends."""
    try:
        from opencut.core.broll_generate import get_available_backends
        return jsonify({"backends": get_available_backends()})
    except Exception:
        return jsonify({"backends": []})
