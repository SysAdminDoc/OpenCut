"""
OpenCut Audio Expansion Routes

AI SFX generation, spatial audio, stem remix, SFX library search,
transition SFX, podcast audio extraction, and Google Fonts browser.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.helpers import _resolve_output_dir
from opencut.helpers import output_path as _op
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
    validate_path,
)

logger = logging.getLogger("opencut")

audio_expand_bp = Blueprint("audio_expand", __name__)


# ---------------------------------------------------------------------------
# AI Sound Effects Generation (2.4)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/audio/generate-sfx", methods=["POST"])
@require_csrf
@async_job("generate_sfx")
def generate_sfx(job_id, filepath, data):
    """Analyze video and generate AI-driven sound effects."""
    from opencut.core.ai_sfx import (
        apply_sfx_to_video,
        detect_sfx_cues,
    )
    from opencut.core.ai_sfx import (
        generate_sfx as _generate,
    )

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    scene_data = data.get("scenes", None)
    keywords = data.get("keywords", None)
    use_ai = safe_bool(data.get("use_ai", False))
    auto_apply = safe_bool(data.get("auto_apply", True))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Detecting SFX cue points...")

    # Detect cues
    cues = detect_sfx_cues(
        video_path=filepath,
        scene_data=scene_data,
        keywords=keywords,
        on_progress=_progress,
    )

    if not cues:
        return {"cues": [], "message": "No SFX cues detected"}

    _progress(30, f"Generating {len(cues)} sound effects...")

    # Generate SFX for each cue
    sfx_list = []
    for i, cue in enumerate(cues):
        pct = 30 + int((i / len(cues)) * 40)
        _progress(pct, f"Generating SFX {i + 1}/{len(cues)}: {cue.category}")

        result = _generate(cue=cue, use_ai=use_ai, on_progress=None)
        sfx_list.append({
            "audio_path": result.output_path,
            "timestamp": cue.timestamp,
            "volume": cue.volume,
            "category": result.category,
            "duration": result.duration,
        })

    # Optionally apply SFX to video
    if auto_apply and sfx_list:
        _progress(75, "Applying SFX to video...")

        _resolve_output_dir(filepath, output_dir)
        out = _op(filepath, "sfx")

        apply_result = apply_sfx_to_video(
            video_path=filepath,
            sfx_list=sfx_list,
            output=out,
            on_progress=_progress,
        )

        return {
            "output_path": apply_result.output_path,
            "sfx_count": apply_result.sfx_count,
            "cues": [
                {"timestamp": s["timestamp"], "category": s["category"],
                 "duration": s["duration"]}
                for s in sfx_list
            ],
        }

    return {
        "sfx_count": len(sfx_list),
        "cues": [
            {"timestamp": s["timestamp"], "category": s["category"],
             "duration": s["duration"], "audio_path": s["audio_path"]}
            for s in sfx_list
        ],
    }


# ---------------------------------------------------------------------------
# Spatial Audio (2.5)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/audio/spatial", methods=["POST"])
@require_csrf
@async_job("spatial_audio")
def spatial_audio(job_id, filepath, data):
    """Convert audio to binaural or surround sound."""
    from opencut.core.spatial_audio import (
        detect_channel_layout,
        to_binaural,
        to_surround,
    )

    mode = data.get("mode", "binaural").strip().lower()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    sofa_path = data.get("sofa_path", None)
    channels = safe_int(data.get("channels", 6), 6, min_val=6, max_val=8)
    gain = safe_float(data.get("gain", 0.0), 0.0, min_val=-20.0, max_val=20.0)
    lfe_cutoff = safe_int(data.get("lfe_cutoff", 120), 120, min_val=40, max_val=300)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if mode == "binaural":
        _progress(5, "Converting to binaural audio...")
        _resolve_output_dir(filepath, output_dir)
        out = _op(filepath, "binaural")

        result = to_binaural(
            audio_path=filepath,
            output=out,
            sofa_path=sofa_path,
            gain=gain,
            on_progress=_progress,
        )
    elif mode in ("surround", "5.1", "7.1"):
        if mode == "7.1":
            channels = 8
        _progress(5, f"Upmixing to {channels}-channel surround...")
        _resolve_output_dir(filepath, output_dir)
        suffix = "51" if channels == 6 else "71"
        out = _op(filepath, f"surround_{suffix}")

        result = to_surround(
            audio_path=filepath,
            channels=channels,
            output=out,
            lfe_cutoff=lfe_cutoff,
            on_progress=_progress,
        )
    elif mode == "detect":
        _progress(10, "Detecting channel layout...")
        layout = detect_channel_layout(filepath, on_progress=_progress)
        return layout
    else:
        raise ValueError(f"Unknown spatial mode: {mode}. Use 'binaural', 'surround', '5.1', '7.1', or 'detect'.")

    return {
        "output_path": result.output_path,
        "input_layout": result.input_layout,
        "output_layout": result.output_layout,
        "channels": result.channels,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# Stem Remix (2.7)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/audio/stem-remix", methods=["POST"])
@require_csrf
@async_job("stem_remix", filepath_required=False)
def stem_remix(job_id, filepath, data):
    """Remix separated audio stems with effects, volume, and pan."""
    from opencut.core.stem_remix import mix_stems, remix_stems

    stem_paths = data.get("stem_paths", [])
    effects_config = data.get("effects_config", [])
    mix_config = data.get("mix_config", [])
    data.get("output_dir", "")
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    if not stem_paths:
        raise ValueError("No stem paths provided. Pass 'stem_paths' as a list.")

    # Validate all stem paths
    validated_paths = []
    for sp in stem_paths:
        if isinstance(sp, str) and sp.strip():
            validated_paths.append(validate_filepath(sp.strip()))
    if not validated_paths:
        raise ValueError("No valid stem file paths found")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if effects_config:
        # Full remix with effects
        _progress(5, "Starting stem remix with effects...")
        result = remix_stems(
            stem_paths=validated_paths,
            effects_config=effects_config,
            output=output,
            on_progress=_progress,
        )
    else:
        # Simple mix with volume/pan
        _progress(5, "Mixing stems...")
        result = mix_stems(
            stem_paths=validated_paths,
            mix_config=mix_config,
            output_file=output or "",
            on_progress=_progress,
        )

    return {
        "output_path": result.output_path,
        "stems_processed": result.stems_processed,
        "effects_applied": result.effects_applied,
    }


# ---------------------------------------------------------------------------
# Free SFX Library Search (19.2)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/sfx/search", methods=["POST"])
@require_csrf
@async_job("sfx_search", filepath_required=False)
def sfx_search(job_id, filepath, data):
    """Search the SFX library (Freesound or builtin)."""
    from opencut.core.sfx_library import list_cached_sfx, search_sfx

    query = data.get("query", "").strip()
    if not query:
        # If no query, list cached SFX
        cached = list_cached_sfx()
        return {"results": cached, "source": "cache", "total": len(cached)}

    api_key = data.get("api_key", "")
    use_freesound = safe_bool(data.get("use_freesound", True))
    filters = data.get("filters", None)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    results = search_sfx(
        query=query,
        filters=filters,
        api_key=api_key,
        use_freesound=use_freesound,
        on_progress=_progress,
    )

    return {
        "results": [
            {
                "id": r.id,
                "name": r.name,
                "tags": r.tags,
                "duration": r.duration,
                "category": r.category,
                "source": r.source,
                "preview_url": r.preview_url,
                "license": r.license,
            }
            for r in results
        ],
        "total": len(results),
        "query": query,
    }


# ---------------------------------------------------------------------------
# SFX Download (19.2)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/sfx/download", methods=["POST"])
@require_csrf
@async_job("sfx_download", filepath_required=False)
def sfx_download(job_id, filepath, data):
    """Download an SFX file by ID."""
    from opencut.core.sfx_library import download_sfx

    sfx_id = data.get("sfx_id", "").strip()
    if not sfx_id:
        raise ValueError("No sfx_id provided")

    api_key = data.get("api_key", "")
    output_dir = data.get("output_dir", None)
    if output_dir:
        output_dir = validate_path(output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = download_sfx(
        sfx_id=sfx_id,
        output_dir=output_dir,
        api_key=api_key,
        on_progress=_progress,
    )

    return {
        "id": result.id,
        "name": result.name,
        "local_path": result.local_path,
        "duration": result.duration,
        "source": result.source,
    }


# ---------------------------------------------------------------------------
# Transition Sound Effects (18.7)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/audio/transition-sfx", methods=["POST"])
@require_csrf
@async_job("transition_sfx")
def transition_sfx(job_id, filepath, data):
    """Apply sound effects at transition points in a video."""
    from opencut.core.transition_sfx import apply_transition_sfx, list_available_sfx

    transitions = data.get("transitions", [])
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    master_volume = safe_float(data.get("master_volume", 0.7), 0.7, min_val=0.0, max_val=1.0)

    if not transitions:
        # Return available SFX list if no transitions provided
        available = list_available_sfx()
        return available

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _resolve_output_dir(filepath, output_dir)
    out = _op(filepath, "transition_sfx")

    result = apply_transition_sfx(
        video_path=filepath,
        transitions=transitions,
        output=out,
        master_volume=master_volume,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "transitions_processed": result.transitions_processed,
        "total_sfx_applied": result.total_sfx_applied,
        "video_duration": result.video_duration,
    }


# ---------------------------------------------------------------------------
# Video Podcast to Audio-Only (40.3)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/podcast/extract-audio", methods=["POST"])
@require_csrf
@async_job("podcast_extract")
def podcast_extract_audio(job_id, filepath, data):
    """Extract audio from video podcast with normalization and metadata."""
    from opencut.core.podcast_extract import add_id3_metadata, extract_podcast_audio

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    audio_format = data.get("format", "mp3").strip().lower()
    bitrate = data.get("bitrate", "192k").strip()
    sample_rate = safe_int(data.get("sample_rate", 44100), 44100, min_val=8000, max_val=96000)
    mono = safe_bool(data.get("mono", True))
    target_lufs = safe_float(data.get("target_lufs", -16.0), -16.0, min_val=-30.0, max_val=-5.0)
    noise_reduce = safe_bool(data.get("noise_reduce", False))
    noise_strength = safe_float(data.get("noise_strength", 0.3), 0.3, min_val=0.0, max_val=1.0)
    trim_silence = safe_bool(data.get("trim_silence", False))
    metadata = data.get("metadata", {})
    artwork_path = data.get("artwork_path", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Starting podcast audio extraction...")

    _resolve_output_dir(filepath, output_dir)
    out = _op(filepath, "podcast")
    # Fix extension
    import os
    base = os.path.splitext(out)[0]
    ext_map = {"m4a": ".m4a", "aac": ".m4a", "ogg": ".ogg", "opus": ".opus",
                "flac": ".flac", "wav": ".wav", "mp3": ".mp3"}
    out = base + ext_map.get(audio_format, f".{audio_format}")

    result = extract_podcast_audio(
        video_path=filepath,
        output=out,
        audio_format=audio_format,
        bitrate=bitrate,
        sample_rate=sample_rate,
        mono=mono,
        target_lufs=target_lufs,
        noise_reduce=noise_reduce,
        noise_reduce_strength=noise_strength,
        trim_silence=trim_silence,
        on_progress=_progress,
    )

    # Apply metadata if provided
    meta_result = None
    if metadata:
        _progress(92, "Embedding metadata...")
        artwork = ""
        if artwork_path:
            try:
                artwork = validate_filepath(artwork_path)
            except ValueError:
                artwork = ""

        meta_result = add_id3_metadata(
            audio_path=result.output_path,
            metadata=metadata,
            artwork_path=artwork or None,
            on_progress=_progress,
        )

    response = {
        "output_path": result.output_path,
        "format": result.format,
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "loudness_lufs": result.loudness_lufs,
        "noise_reduced": result.noise_reduced,
        "file_size_bytes": result.file_size_bytes,
    }
    if meta_result:
        response["metadata_applied"] = True
        response["metadata_fields"] = meta_result.metadata_fields
        response["has_artwork"] = meta_result.has_artwork

    return response


# ---------------------------------------------------------------------------
# Google Fonts Browser (19.3)
# ---------------------------------------------------------------------------
@audio_expand_bp.route("/api/fonts/list", methods=["GET"])
def fonts_list():
    """List available Google Fonts."""
    try:
        from opencut.core.google_fonts import list_fonts, search_fonts

        category = request.args.get("category", "")
        query = request.args.get("query", "")
        api_key = request.args.get("api_key", "")
        refresh = safe_bool(request.args.get("refresh", "false"))

        if query:
            results = search_fonts(query=query, category=category, api_key=api_key)
        else:
            results = list_fonts(category=category, api_key=api_key, refresh=refresh)

        return jsonify({
            "fonts": [
                {
                    "family": f.family,
                    "category": f.category,
                    "variants": f.variants,
                    "subsets": f.subsets,
                    "is_downloaded": f.is_downloaded,
                    "local_path": f.local_path,
                }
                for f in results
            ],
            "total": len(results),
            "categories": ["serif", "sans-serif", "display", "handwriting", "monospace"],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Font list error")
        return jsonify({"error": str(e)}), 500


@audio_expand_bp.route("/api/fonts/download", methods=["POST"])
@require_csrf
@async_job("font_download", filepath_required=False)
def font_download(job_id, filepath, data):
    """Download a Google Font to local cache."""
    from opencut.core.google_fonts import download_font

    font_name = data.get("font_name", "").strip()
    if not font_name:
        raise ValueError("No font_name provided")

    variant = data.get("variant", "regular").strip()
    api_key = data.get("api_key", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = download_font(
        font_name=font_name,
        variant=variant,
        api_key=api_key,
        on_progress=_progress,
    )

    return {
        "family": result.family,
        "local_dir": result.local_dir,
        "files": result.files,
        "total_size_bytes": result.total_size_bytes,
    }
