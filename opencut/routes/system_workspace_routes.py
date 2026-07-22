"""System workspace routes registered on the shared system blueprint."""

from .system import (
    _OPEN_PATH_ALLOWED_EXTS,
    _list_jobs_copy,
    _sp,
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_ffmpeg_path,
    get_json_dict,
    is_path_within_any,
    jsonify,
    logger,
    os,
    request,
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    send_file,
    sys,
    system_bp,
    validate_filepath,
    verify_destructive_confirm_token,
)


# ---------------------------------------------------------------------------
# Project Media (stub for non-Premiere dev mode)
# ---------------------------------------------------------------------------
@system_bp.route("/project/media", methods=["GET"])
def project_media():
    """Return project media list. Stub for dev mode (outside Premiere).

    Inside Premiere the CEP panel uses ExtendScript directly; this endpoint
    exists so the panel doesn't 404 when running in a browser for development.
    """
    return jsonify({"media": [], "projectFolder": "", "rootChildren": 0})


# ---------------------------------------------------------------------------
# Open / Reveal Path (for session context "Open Output" / "Reveal in Folder")
# ---------------------------------------------------------------------------
@system_bp.route("/system/open-path", methods=["POST"])
@require_csrf
def open_path():
    """Open a file (``mode=open``) or reveal it in the OS file manager
    (``mode=reveal``). Used by the session-context overlay's job-result
    quick actions.

    The path must pass ``validate_filepath`` — no traversal, null bytes,
    or UNC paths. Non-existent files are rejected early.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw_path = data.get("path", "")
    mode = str(data.get("mode", "open")).strip().lower()
    if mode not in ("open", "reveal"):
        return jsonify({"error": "mode must be 'open' or 'reveal'"}), 400
    if not raw_path:
        return jsonify({"error": "path is required"}), 400

    try:
        filepath = validate_filepath(raw_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ext = os.path.splitext(filepath)[1].lower()
    if mode == "open" and ext not in _OPEN_PATH_ALLOWED_EXTS:
        return jsonify({"error": f"Cannot open unsupported file type: {ext or '(none)'}"}), 403

    try:
        if sys.platform == "win32":
            if mode == "reveal":
                _sp.Popen(["explorer", "/select,", filepath],
                          creationflags=_sp.CREATE_NEW_PROCESS_GROUP
                          if hasattr(_sp, "CREATE_NEW_PROCESS_GROUP") else 0)
            else:
                os.startfile(filepath)  # noqa: S606 — validated path, open-mode extension allowlisted above
        elif sys.platform == "darwin":
            if mode == "reveal":
                _sp.Popen(["open", "-R", filepath], start_new_session=True)
            else:
                _sp.Popen(["open", filepath], start_new_session=True)
        else:
            # Linux / BSD — xdg-open has no "reveal" equivalent; open the
            # containing directory when reveal is requested.
            target = os.path.dirname(filepath) if mode == "reveal" else filepath
            _sp.Popen(["xdg-open", target], start_new_session=True)
    except Exception as e:
        logger.exception("open_path failed for %s", filepath)
        return jsonify({"error": f"Could not open path: {e}"}), 500

    return jsonify({"ok": True, "path": filepath, "mode": mode})


# ---------------------------------------------------------------------------
# Sequence Assistant (v1.10.0, feature E)
# ---------------------------------------------------------------------------
@system_bp.route("/assistant/suggest", methods=["POST"])
@require_csrf
def assistant_suggest():
    """Analyze a sequence snapshot and return ranked editing suggestions.

    Body::

        {
          "sequence": {...},               # shape of ocGetSequenceInfo JSON
          "dismissed": ["silence-dead-air"],  # optional, session-scoped
          "sequence_key": "project.prproj"    # optional, v1.10.2 persistent
        }

    v1.10.2: persisted dismissals are keyed by *sequence_key* (typically
    the Premiere project path). The server unions session dismissals
    with the stored per-sequence set so either flavor hides the card.
    """
    from opencut.core.assistant import analyze_sequence
    from opencut.user_data import load_assistant_dismissed

    data = request.get_json(force=True, silent=True) or {}
    sequence = data.get("sequence") or {}
    dismissed = data.get("dismissed") or []
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    if not isinstance(sequence, dict):
        return jsonify({"error": "sequence must be an object"}), 400
    if not isinstance(dismissed, list):
        return jsonify({"error": "dismissed must be a list"}), 400

    persisted = load_assistant_dismissed(sequence_key)
    all_dismissed = list({str(d) for d in dismissed} | set(persisted))

    try:
        suggestions = analyze_sequence(sequence, dismissed_ids=all_dismissed)
    except Exception as e:
        logger.exception("assistant_suggest failed")
        return jsonify({"error": f"Analyze failed: {e}"}), 500
    return jsonify({
        "suggestions": suggestions,
        "count": len(suggestions),
        "persisted_dismissed": persisted,
    })


@system_bp.route("/assistant/dismiss", methods=["POST"])
@require_csrf
def assistant_dismiss():
    """Persistently dismiss a suggestion for a sequence.

    Body: ``{"sequence_key": "...", "id": "silence-dead-air"}``
    Returns the updated list so the panel can reconcile state.
    """
    from opencut.user_data import load_assistant_dismissed, save_assistant_dismissed

    data = request.get_json(force=True, silent=True) or {}
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    sug_id = (data.get("id") or "").strip()
    if not sug_id:
        return jsonify({"error": "id is required"}), 400

    current = load_assistant_dismissed(sequence_key)
    if sug_id not in current:
        current.append(sug_id)
        save_assistant_dismissed(sequence_key, current)
    return jsonify({"sequence_key": sequence_key, "dismissed": current})


@system_bp.route("/assistant/dismiss-clear", methods=["POST"])
@require_csrf
def assistant_dismiss_clear():
    """Clear every persisted dismissal for a sequence so they reappear."""
    from opencut.user_data import (
        build_user_data_destructive_record,
        create_user_tombstone,
        load_assistant_dismissed,
        save_assistant_dismissed,
        summarize_user_tombstone,
    )

    data = get_json_dict() if request.data else {}
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    current = load_assistant_dismissed(sequence_key)
    records = []
    if current:
        records.append(build_user_data_destructive_record(
            "assistant_dismissed",
            sequence_key,
            current,
            source_file="assistant_dismissed.json",
            route="/assistant/dismiss-clear",
            action="clear",
        ))
    plan = build_destructive_plan(
        "assistant.dismiss_clear",
        records=records,
        metadata={"route": "/assistant/dismiss-clear", "sequence_key": sequence_key},
        reversible=True,
    )
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "sequence_key": sequence_key,
            "dismissed": current,
            "would_clear": len(current),
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
        })
    if records and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409
    tombstone = create_user_tombstone(
        "assistant_dismissed",
        sequence_key,
        current,
        source_file="assistant_dismissed.json",
        action="clear",
        metadata={"route": "/assistant/dismiss-clear"},
    )
    save_assistant_dismissed(sequence_key, [])
    return jsonify({
        "sequence_key": sequence_key,
        "dismissed": [],
        "tombstone": summarize_user_tombstone(tombstone),
    })


# ---------------------------------------------------------------------------
# Live audio preview (v1.9.36, feature C)
# ---------------------------------------------------------------------------
# Renders a short slice of a clip with an effect applied and returns the raw
# WAV bytes so the panel can A/B-compare settings in ~1s instead of waiting
# for a full-file job. Budget: 15s slice + FFmpeg-native filters only. Heavy
# neural processors (Resemble Enhance, Demucs) fall back to full-file jobs.
_PREVIEW_MAX_SECONDS = 15
_PREVIEW_FILTERS = {"raw", "denoise", "normalize", "compress", "eq", "silence"}


@system_bp.route("/preview/audio", methods=["POST"])
@require_csrf
def preview_audio():
    """Render a short audio slice with a filter applied.

    Body::

        {
          "filepath": "...",
          "start": 0,                   # seconds into the file
          "duration": 10,               # seconds of preview (max 15)
          "filter": "raw|denoise|normalize|compress|eq|silence",
          "params": {...}               # filter-specific
        }

    Returns ``audio/wav`` on success, 400 on bad input.
    """
    import tempfile

    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()
    start = safe_float(data.get("start", 0), 0.0, min_val=0.0)
    duration = safe_float(data.get("duration", 10), 10.0,
                          min_val=1.0, max_val=float(_PREVIEW_MAX_SECONDS))
    flt = str(data.get("filter", "denoise")).lower()
    params = data.get("params") or {}

    if flt not in _PREVIEW_FILTERS:
        return jsonify({
            "error": f"Unsupported preview filter: {flt}",
            "supported": sorted(_PREVIEW_FILTERS),
        }), 400
    if not filepath:
        return jsonify({"error": "filepath is required"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Build an FFmpeg filter chain for the requested filter type. All are
    # lightweight built-in filters that run in near-realtime on a short
    # slice — no ML model loads, no model downloads.
    afilter = None
    if flt == "denoise":
        # afftdn is an FFT-based denoiser — maps 0..1 strength to noise
        # reduction level in dB (0..24).
        strength = safe_float(params.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0)
        nr = round(strength * 24, 1)
        afilter = f"afftdn=nr={nr}"
    elif flt == "normalize":
        target = safe_float(params.get("target_lufs", -16.0), -16.0,
                            min_val=-40.0, max_val=0.0)
        afilter = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    elif flt == "compress":
        threshold = safe_float(params.get("threshold_db", -20), -20.0,
                               min_val=-40.0, max_val=0.0)
        ratio = safe_float(params.get("ratio", 4.0), 4.0, min_val=1.0, max_val=20.0)
        # acompressor threshold is linear (0..1); approximate from dB.
        thr_lin = 10 ** (threshold / 20.0)
        afilter = f"acompressor=threshold={thr_lin}:ratio={ratio}:attack=5:release=50"
    elif flt == "eq":
        low = safe_float(params.get("low_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        mid = safe_float(params.get("mid_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        high = safe_float(params.get("high_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        afilter = (
            f"equalizer=f=120:t=h:w=100:g={low},"
            f"equalizer=f=1000:t=h:w=800:g={mid},"
            f"equalizer=f=8000:t=h:w=4000:g={high}"
        )
    elif flt == "silence":
        # Preview the impact of a silence threshold by *muting* detected
        # silent regions in the slice so users can hear what gets cut.
        thr_db = safe_float(params.get("threshold_db", -30), -30.0,
                            min_val=-60.0, max_val=0.0)
        min_dur = safe_float(params.get("min_silence", 0.4), 0.4,
                             min_val=0.05, max_val=5.0)
        afilter = f"silenceremove=start_periods=0:stop_periods=-1:stop_threshold={thr_db}dB:stop_duration={min_dur}"

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="opencut_preview_", delete=False
    )
    out_path = tmp.name
    tmp.close()

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", filepath,
        "-vn",
    ]
    if afilter:
        cmd.extend(["-af", afilter])
    cmd.extend([
        "-ac", "2",
        "-ar", "44100",
        "-f", "wav",
        out_path,
    ])
    try:
        proc = _sp.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0:
            os.unlink(out_path)
            return jsonify({
                "error": "Preview render failed",
                "detail": proc.stderr.decode("utf-8", "ignore")[-300:],
            }), 500
    except _sp.TimeoutExpired:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return jsonify({"error": "Preview render timed out"}), 504
    except Exception as e:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return jsonify({"error": f"Preview render failed: {e}"}), 500

    # Read back + return as audio/wav. Schedule deletion of the temp file.
    try:
        with open(out_path, "rb") as f:
            data_bytes = f.read()
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass

    from flask import Response
    return Response(data_bytes, mimetype="audio/wav",
                    headers={"Content-Length": str(len(data_bytes))})


# ---------------------------------------------------------------------------
# Preflight — "can this pipeline succeed?" check (v1.9.33, feature G)
# ---------------------------------------------------------------------------
@system_bp.route("/preflight/<pipeline>", methods=["POST"])
@require_csrf
def preflight_check(pipeline: str):
    """Run the preflight checklist for *pipeline* before kicking off the job.

    Body::

        {"filepath": "...", "output_dir": "..."}

    Response carries blocking issues (must fix), soft warnings (might
    degrade but won't abort), and a top-level pass:true/false.
    """
    from opencut.preflight import run_preflight

    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()
    output_dir = (data.get("output_dir") or "").strip()

    try:
        report = run_preflight(pipeline, filepath=filepath, output_dir=output_dir)
    except Exception as e:
        logger.exception("preflight for %s failed", pipeline)
        return jsonify({"error": f"Preflight failed: {e}"}), 500
    if "error" in report:
        return jsonify(report), 400
    return jsonify(report)


# ---------------------------------------------------------------------------
# File Serving (for audio preview player)
# ---------------------------------------------------------------------------
@system_bp.route("/file", methods=["GET"])
def serve_file():
    """Serve a validated local media file for preview.

    Defence-in-depth:

    1. ``validate_filepath`` rejects null bytes, traversal, and UNC paths.
    2. The resolved realpath must live under an approved root — either the
       per-user ``~/.opencut`` directory (job outputs) or the system temp
       directory (short-lived work files). Arbitrary local media files
       elsewhere on disk are NOT served even if they look like audio/video.
       This prevents any same-browser origin from enumerating personal
       media via ``<img>``/``<audio>`` side-channels.
    3. The guessed MIME type must be audio/video/image so this endpoint
       can never act as a generic local-file reader.
    """
    import mimetypes
    import tempfile

    from opencut.helpers import OPENCUT_DIR

    filepath = request.args.get("path", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 404

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Approved roots: OpenCut user data dir + system temp dir. All legitimate
    # preview files (TTS/SFX/music outputs, intermediate renders) land here.
    allowed_roots = [
        tempfile.gettempdir(),
        OPENCUT_DIR,
    ]
    if not is_path_within_any(filepath, allowed_roots):
        return jsonify({"error": "Access denied"}), 403

    mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    if not mime_type.startswith(("audio/", "video/", "image/")):
        return jsonify({"error": "Unsupported preview file type"}), 403

    return send_file(filepath, mimetype=mime_type)


# ---------------------------------------------------------------------------
# Output Browser (recent outputs)
# ---------------------------------------------------------------------------
@system_bp.route("/outputs/recent", methods=["GET"])
def recent_outputs():
    """List recent output files from completed jobs."""
    limit = safe_int(request.args.get("limit", 20), default=20, min_val=1, max_val=100)
    outputs = []
    all_jobs = _list_jobs_copy()
    sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created", 0), reverse=True)
    for job in sorted_jobs[:limit * 2]:
        if job.get("status") != "complete":
            continue
        result = job.get("result", {})
        path = result.get("output_path", result.get("output", ""))
        if isinstance(path, list):
            for p in path:
                if os.path.isfile(p):
                    try:
                        stat = os.stat(p)
                        outputs.append({
                            "path": p,
                            "name": os.path.basename(p),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": stat.st_mtime,
                            "type": job.get("type", "unknown"),
                        })
                    except OSError:
                        pass
        elif isinstance(path, str) and path and os.path.isfile(path):
            try:
                stat = os.stat(path)
                outputs.append({
                    "path": path,
                    "name": os.path.basename(path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "type": job.get("type", "unknown"),
                })
            except OSError:
                pass
        if len(outputs) >= limit:
            break
    return jsonify(outputs)

