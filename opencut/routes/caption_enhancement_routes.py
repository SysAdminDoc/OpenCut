"""WhisperX, translation, karaoke, and caption conversion routes."""

from .captions import (
    _VALID_SUBTITLE_FORMATS,
    VALID_WHISPER_MODELS,
    _resolve_output_dir,
    _sanitize_font_name,
    _update_job,
    async_job,
    captions_bp,
    get_json_dict,
    jsonify,
    os,
    require_csrf,
    safe_bool,
    safe_error,
    safe_int,
    safe_pip_install,
    tempfile,
    validate_filepath,
    validate_path,
)


# Enhanced Captions (WhisperX, Translation, Karaoke)
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/enhanced/capabilities", methods=["GET"])
def captions_enhanced_capabilities():
    """Return enhanced caption capabilities."""
    try:
        from opencut.core.captions_enhanced import (
            TRANSLATION_LANGUAGES,
            check_nllb_available,
            check_pysubs2_available,
            check_whisperx_available,
            get_translation_backend_info,
            translation_backends_manifest,
        )
        return jsonify({
            "whisperx": check_whisperx_available(),
            "nllb": check_nllb_available(),
            "pysubs2": check_pysubs2_available(),
            "languages": TRANSLATION_LANGUAGES,
            "translation_backends": translation_backends_manifest(),
            "translation_default_backend": "",
            "translation_default_notice": (
                "No commercial-safe local translation backend is configured. "
                "NLLB and SeamlessM4T require explicit non-commercial-license opt-in."
            ),
            "nllb_license": get_translation_backend_info("nllb").get("license", ""),
        })
    except Exception as e:
        return safe_error(e, "captions_enhanced_capabilities")


@captions_bp.route("/captions/whisperx", methods=["POST"])
@require_csrf
@async_job("whisperx", disk_operation="transcribe", resumable=True)
def captions_whisperx(job_id, filepath, data):
    """Transcribe with WhisperX for word-level timestamps."""
    model_size = data.get("model", "base")
    if model_size not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model_size}")
    language = data.get("language", "")
    diarize = safe_bool(data.get("diarize", False), False)
    hf_token = data.get("hf_token", "")

    from opencut.core.captions_enhanced import whisperx_transcribe

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = whisperx_transcribe(
        filepath, model_size=model_size,
        model_revision=str(data.get("model_revision") or ""),
        language=language, diarize=diarize,
        hf_token=hf_token, on_progress=_on_progress,
    )
    return result


def _validate_segments(data, *, max_segments=10000):
    """Shared sync validator for routes that operate on a list of segments."""
    segs = data.get("segments")
    if not segs:
        return "No segments provided"
    if not isinstance(segs, list):
        return "segments must be a list"
    if len(segs) > max_segments:
        return f"Too many segments (max {max_segments})"
    return None


def _validate_translate_input(data, *, max_segments=10000):
    """Translate-specific validator: accept segments OR SRT (path/content)."""
    segs = data.get("segments")
    srt_path = (data.get("srt_path") or "").strip()
    srt_content = data.get("srt_content")
    if isinstance(srt_content, str):
        srt_content = srt_content.strip()

    def _license_error() -> str | None:
        try:
            from opencut.core.captions_enhanced import (
                TranslationLicenseError,
                resolve_translation_backend,
                translation_license_opted_in,
            )

            resolve_translation_backend(
                data.get("backend", "auto"),
                allow_restricted=translation_license_opted_in(data),
            )
        except TranslationLicenseError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - keep pre-validation user-facing.
            return f"Translation backend validation failed: {exc}"
        return None

    if segs:
        if not isinstance(segs, list):
            return "segments must be a list"
        if len(segs) > max_segments:
            return f"Too many segments (max {max_segments})"
        return _license_error()
    if srt_path or srt_content:
        return _license_error()
    return "Provide `segments`, `srt_path`, or `srt_content`."


def _validate_enhanced_install_input(data):
    """Require explicit license acceptance before restricted translation installs."""
    component = (data.get("component") or "whisperx").strip().lower()
    if component not in {"nllb", "seamless"}:
        return None
    try:
        from opencut.core.captions_enhanced import (
            TranslationLicenseError,
            resolve_translation_backend,
            translation_license_opted_in,
        )

        resolve_translation_backend(
            component,
            allow_restricted=translation_license_opted_in(data),
        )
    except TranslationLicenseError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - keep pre-validation user-facing.
        return f"Enhanced caption install validation failed: {exc}"
    return None


def _srt_to_translate_segments(srt_text: str, *, max_segments: int = 10000):
    """Parse an SRT blob into the {start, end, text} segments shape."""
    from opencut.core.multilang_subtitle import _parse_srt_content

    parsed = _parse_srt_content(srt_text or "")
    if not parsed:
        raise ValueError("No SRT cues found")
    if len(parsed) > max_segments:
        raise ValueError(f"Too many SRT cues (max {max_segments})")
    return parsed


def _segments_to_srt_text(segments) -> str:
    """Render translated segments back as SRT text. Self-contained so we
    don't have to instantiate a full TranscriptionResult / CaptionSegment
    chain for the translate path."""
    from opencut.export.srt import _format_srt_time

    lines = []
    for i, seg in enumerate(segments, 1):
        if not isinstance(seg, dict):
            continue
        start = seg.get("start", 0.0) or 0.0
        end = seg.get("end", 0.0) or 0.0
        text = (seg.get("text") or "").strip()
        if not text:
            # Preserve cue index spacing even for empty cues so timing stays sane.
            text = ""
        lines.append(f"{i}")
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


@captions_bp.route("/captions/translate", methods=["POST"])
@require_csrf
@async_job(
    "translate",
    filepath_required=False,
    pre_validate=_validate_translate_input,
)
def captions_translate(job_id, filepath, data):
    """Translate caption segments.

    Backend ``auto`` is fail-closed until a commercial-safe local translation
    engine is registered. ``nllb`` and ``seamless`` are non-commercial-license
    engines and require ``accept_restricted_license=true``.

    Input accepted (F139):

    * ``segments``: list of ``{start, end, text}`` dicts (legacy path).
    * ``srt_path``: absolute path to an SRT file on disk; parsed into
      segments before translation. Useful for "SRT in / SRT out"
      workflows.
    * ``srt_content``: raw SRT text. Useful when the panel already
      holds the SRT blob in memory.

    Output options:

    * ``output_srt=true`` to also receive the translated SRT text in
      the response.
    * ``srt_output_path`` (absolute) to also write the translated SRT
      to disk; the resolved path is echoed back in ``srt_output_path``.
    * ``srt_legacy_bom``/``legacy_bom``/``windows_legacy_bom``: opt
      into the legacy Windows BOM (F243 toggle) on the written SRT.
    """
    segments = data.get("segments") or []
    srt_path = (data.get("srt_path") or "").strip()
    srt_content = data.get("srt_content")
    if isinstance(srt_content, str):
        srt_content = srt_content.strip()
    else:
        srt_content = None

    srt_input_used = False
    if not segments:
        if srt_path:
            srt_path = validate_filepath(srt_path)
            # SRT files are line-oriented timecodes + plain text. A real
            # caption file rarely exceeds a few hundred KB; cap the read
            # at 16 MB so a crafted multi-GB blob can't OOM the worker.
            _max_srt_bytes = 16 * 1024 * 1024
            try:
                size = os.path.getsize(srt_path)
            except OSError as exc:
                raise ValueError(f"could not stat srt_path: {exc}") from exc
            if size > _max_srt_bytes:
                raise ValueError(
                    f"srt_path is {size} bytes (>{_max_srt_bytes}); "
                    "refusing to read. Pre-trim the file or chunk it."
                )
            with open(srt_path, "rb") as f:
                raw_bytes = f.read(_max_srt_bytes + 1)
            if len(raw_bytes) > _max_srt_bytes:
                raise ValueError(
                    "srt_path grew past size cap during read; "
                    "refusing to parse a truncated SRT."
                )
            # Tolerate UTF-8 with or without BOM (F243 legacy path).
            srt_content = raw_bytes.decode("utf-8-sig", errors="replace")
            srt_input_used = True
        if srt_content:
            segments = _srt_to_translate_segments(srt_content)
            srt_input_used = True

    if not segments:
        raise ValueError("No segments provided")
    if len(segments) > 10000:
        raise ValueError("Too many segments (max 10000)")

    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")
    from opencut.core.captions_enhanced import (
        get_translation_backend_info,
        resolve_translation_backend,
        translation_license_opted_in,
    )

    restricted_accepted = translation_license_opted_in(data)
    backend = resolve_translation_backend(
        data.get("backend", "auto"),
        allow_restricted=restricted_accepted,
    )
    backend_info = get_translation_backend_info(backend)

    output_srt = safe_bool(data.get("output_srt", False), False)
    if srt_input_used:
        # If the caller fed us SRT, default to also emitting SRT.
        output_srt = output_srt or True
    srt_output_path = (data.get("srt_output_path") or "").strip()
    if srt_output_path:
        srt_output_path = validate_path(srt_output_path)
        output_srt = True
    legacy_bom = (
        safe_bool(data.get("srt_legacy_bom", False), False)
        or safe_bool(data.get("legacy_bom", False), False)
        or safe_bool(data.get("windows_legacy_bom", False), False)
    )

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if backend == "auto":
        # Unified: tries SeamlessM4T first, falls back to NLLB
        from opencut.core.captions_enhanced import translate_segments_auto
        translated = translate_segments_auto(
            segments, source_lang=source_lang,
            target_lang=target_lang, on_progress=_on_progress,
        )
    elif backend == "seamless":
        # SeamlessM4T v2 -- load model ONCE, translate all segments
        _on_progress(10, "Loading SeamlessM4T v2 model...")
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "facebook/seamless-m4t-v2-large"
        processor = AutoProcessor.from_pretrained(model_id)
        s_model = SeamlessM4Tv2ForTextToText.from_pretrained(model_id).to(device)
        try:
            translated = []
            total = len(segments)
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                if not text:
                    translated.append(seg.copy())
                    continue
                inputs = processor(text=text, src_lang=source_lang, return_tensors="pt").to(device)
                with torch.inference_mode():
                    tokens = s_model.generate(**inputs, tgt_lang=target_lang, max_new_tokens=512)
                t = processor.decode(tokens[0].tolist(), skip_special_tokens=True)
                new_seg = seg.copy()
                new_seg["text"] = t
                translated.append(new_seg)
                if i % 10 == 0:
                    _on_progress(10 + int((i / total) * 85), f"Translating {i+1}/{total}...")
        finally:
            del s_model
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        from opencut.core.captions_enhanced import translate_segments
        translated = translate_segments(
            segments, source_lang=source_lang,
            target_lang=target_lang, on_progress=_on_progress,
        )

    result = {
        "segments": translated,
        "target_lang": target_lang,
        "source_lang": source_lang,
        "backend": backend,
        "backend_label": backend_info.get("label", backend),
        "backend_license": backend_info.get("license", ""),
        "backend_commercial_safe": safe_bool(backend_info.get("commercial_safe"), False),
        "restricted_license_accepted": restricted_accepted,
        "license_notice": backend_info.get("notice", ""),
        "redistribution": backend_info.get("redistribution", ""),
        "count": len(translated),
        "input_format": "srt" if srt_input_used else "segments",
    }
    if output_srt:
        srt_text = _segments_to_srt_text(translated)
        result["srt"] = srt_text
        result["srt_encoding"] = "utf-8-sig" if legacy_bom else "utf-8"
        if srt_output_path:
            from opencut.export.srt import write_srt_text

            write_srt_text(
                srt_output_path, srt_text, legacy_windows_bom=legacy_bom
            )
            result["srt_output_path"] = srt_output_path
    return result


@captions_bp.route("/captions/karaoke", methods=["POST"])
@require_csrf
@async_job("karaoke", filepath_required=False, pre_validate=_validate_segments)
def captions_karaoke(job_id, filepath, data):
    """Export segments as ASS karaoke subtitles."""
    segments = data.get("segments", [])
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not segments:
        raise ValueError("No segments provided")
    if len(segments) > 10000:
        raise ValueError("Too many segments (max 10000)")

    from opencut.core.captions_enhanced import segments_to_ass_karaoke

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir) if filepath else tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(filepath))[0] if filepath else "captions"
    out_path = os.path.join(effective_dir, f"{base}_karaoke.ass")

    segments_to_ass_karaoke(
        segments, out_path,
        font_name=_sanitize_font_name(data.get("font", "Arial")),
        font_size=safe_int(data.get("font_size", 48), default=48),
        on_progress=_on_progress,
    )
    return {"output_path": out_path}


@captions_bp.route("/captions/convert", methods=["POST"])
@require_csrf
def captions_convert():
    """Convert subtitle file between formats."""
    data = get_json_dict()
    filepath = data.get("filepath", "").strip()
    target_format = data.get("format", "srt")
    if target_format not in _VALID_SUBTITLE_FORMATS:
        return jsonify({"error": f"Unsupported format: {target_format}"}), 400

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        from opencut.core.captions_enhanced import convert_subtitle_format
        out = convert_subtitle_format(filepath, output_format=target_format)
        return jsonify({"output_path": out})
    except Exception as e:
        return safe_error(e, "captions_convert")


@captions_bp.route("/captions/enhanced/install", methods=["POST"])
@require_csrf
@async_job(
    "install",
    filepath_required=False,
    rate_limit_key="model_install",
    pre_validate=_validate_enhanced_install_input,
)
def captions_enhanced_install(job_id, filepath, data):
    """Install enhanced caption dependencies."""
    component = (data.get("component") or "whisperx").strip().lower()

    packages = {
        "whisperx": ["whisperx"],
        "pysubs2": ["pysubs2"],
        "nllb": ["ctranslate2", "sentencepiece", "huggingface-hub"],
        "seamless": ["transformers", "torch", "sentencepiece"],
    }

    if component in {"nllb", "seamless"}:
        from opencut.core.captions_enhanced import (
            TranslationLicenseError,
            resolve_translation_backend,
            translation_license_opted_in,
        )

        try:
            resolve_translation_backend(
                component,
                allow_restricted=translation_license_opted_in(data),
            )
        except TranslationLicenseError as exc:
            raise ValueError(str(exc)) from exc

    pkgs = packages.get(component, [])
    if not pkgs:
        raise ValueError(f"Unknown component: {component}")

    for i, pkg in enumerate(pkgs):
        pct = int((i / len(pkgs)) * 90)
        _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
        safe_pip_install(pkg)
    return {"component": component}



# ---------------------------------------------------------------------------
