"""
OpenCut Captions Routes

Transcription, caption styles, burn-in, animated captions, whisperx,
translation, karaoke, format conversion.
"""

import logging
import os
import tempfile
import threading

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _make_sequence_name, _resolve_output_dir
from opencut.jobs import _is_cancelled, _update_job, async_job
from opencut.security import (
    VALID_WHISPER_MODELS,
    rate_limit,
    rate_limit_release,
    require_csrf,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
)

logger = logging.getLogger("opencut")

# Core imports used by multiple routes (try relative/absolute, tolerate missing)
detect_speech = get_edit_summary = generate_zoom_events = None
export_premiere_xml = export_ass = export_json = export_srt = export_vtt = None
CaptionConfig = ExportConfig = get_preset = _probe_media = LLMConfig = None
try:
    from ..core.llm import LLMConfig  # noqa: F811
    from ..core.silence import detect_speech, get_edit_summary  # noqa: F811
    from ..core.zoom import generate_zoom_events  # noqa: F811
    from ..export.premiere import export_premiere_xml  # noqa: F811
    from ..export.srt import export_ass, export_json, export_srt, export_vtt  # noqa: F811
    from ..utils.config import CaptionConfig, ExportConfig, get_preset  # noqa: F811
    from ..utils.media import probe as _probe_media  # noqa: F811
except ImportError:
    try:
        from opencut.core.llm import LLMConfig  # noqa: F811
        from opencut.core.silence import detect_speech, get_edit_summary  # noqa: F811
        from opencut.core.zoom import generate_zoom_events  # noqa: F811
        from opencut.export.premiere import export_premiere_xml  # noqa: F811
        from opencut.export.srt import export_ass, export_json, export_srt, export_vtt  # noqa: F811
        from opencut.utils.config import CaptionConfig, ExportConfig, get_preset  # noqa: F811
        from opencut.utils.media import probe as _probe_media  # noqa: F811
    except ImportError:
        LLMConfig = None
        logger.warning("Some caption dependencies could not be imported")

captions_bp = Blueprint("captions", __name__)

import re as _re  # noqa: E402

# ---------------------------------------------------------------------------
# Transcript Cache (transcribe once, reuse across highlights/shorts/translate)
# ---------------------------------------------------------------------------
_transcript_cache = {}  # {filepath_hash: {"segments": [...], "language": str, "model": str, "mtime": float}}
_transcript_cache_lock = threading.Lock()
_TRANSCRIPT_CACHE_MAX = 20


def _cache_key(filepath):
    """Generate cache key from filepath + file modification time."""
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        mtime = 0
    return f"{os.path.normcase(os.path.realpath(filepath))}:{mtime}"


def cache_transcript(filepath, segments, language="", model=""):
    """Cache transcription results for reuse."""
    key = _cache_key(filepath)
    with _transcript_cache_lock:
        if len(_transcript_cache) >= _TRANSCRIPT_CACHE_MAX:
            oldest = next(iter(_transcript_cache))
            del _transcript_cache[oldest]
        _transcript_cache[key] = {
            "segments": segments,
            "language": language,
            "model": model,
        }


def get_cached_transcript(filepath, model=""):
    """Get cached transcript if available and model matches."""
    key = _cache_key(filepath)
    with _transcript_cache_lock:
        cached = _transcript_cache.get(key)
        if cached and (not model or cached.get("model") == model):
            return cached["segments"]
    return None

_VALID_SUBTITLE_FORMATS = {"srt", "ass", "vtt", "sub", "json"}
_VALID_ANIMATIONS = {"pop", "slide_up", "fade", "bounce", "typewriter", "highlight_box", "glow"}
_VALID_BURNIN_STYLES = {"default", "bold_yellow", "boxed_dark", "neon_cyan", "cinematic_serif", "top_center"}


def _sanitize_force_style(s: str) -> str:
    """Allow only safe ASS style directives (alphanumeric, commas, equals, parens, dots, spaces)."""
    if s and _re.match(r'^[a-zA-Z0-9,=.() -]+$', s):
        return s
    return ""


def _sanitize_font_name(s: str) -> str:
    """Allow only safe font names (alphanumeric, spaces, hyphens, underscores)."""
    if s and _re.match(r'^[a-zA-Z0-9 _-]+$', s):
        return s
    return "Arial"


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions", methods=["POST"])
@require_csrf
@async_job("captions")
def generate_captions(job_id, filepath, data):
    """Generate captions/subtitles."""
    output_dir = data.get("output_dir", "")

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)
    sub_format = data.get("format", "srt")
    if sub_format not in ("srt", "vtt", "json", "ass"):
        sub_format = "srt"
    word_timestamps = data.get("word_timestamps", True)

    from opencut.core.captions import check_whisper_available, transcribe

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    _update_job(job_id, progress=10, message=f"Loading {model} model ({backend})...")

    config = CaptionConfig(
        model=model,
        language=language,
        word_timestamps=word_timestamps,
    )

    # Check transcript cache (only for SRT/VTT export -- styled captions need Segment objects)
    _use_cache = sub_format in ("srt", "vtt", "json") and not data.get("force_retranscribe", False)
    cached = get_cached_transcript(filepath, model=model) if _use_cache else None
    if cached:
        _update_job(job_id, progress=20, message="Using cached transcript...")
        # Build minimal objects with .start/.end/.text for export functions
        _CachedSeg = type("CachedSeg", (), {})
        cached_segs = []
        for cs in cached:
            seg_obj = _CachedSeg()
            seg_obj.start = cs.get("start", 0)
            seg_obj.end = cs.get("end", 0)
            seg_obj.text = cs.get("text", "")
            seg_obj.words = []
            cached_segs.append(seg_obj)
        result = type("TranscriptResult", (), {"segments": cached_segs, "language": language or "en"})()
    else:
        _update_job(job_id, progress=20, message="Transcribing audio (this takes a while for long files)...")
        result = transcribe(filepath, config=config)
        # Cache for reuse by highlights/shorts/translate
        if hasattr(result, "segments"):
            segs = [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments] if hasattr(result.segments[0], "start") else result.segments
            cache_transcript(filepath, segs, language=getattr(result, "language", ""), model=model)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    _update_job(job_id, progress=80, message="Writing subtitle file...")

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base_name}.{sub_format}")

    if sub_format == "vtt":
        export_vtt(result, out_path)
    elif sub_format == "json":
        export_json(result, out_path)
    elif sub_format == "ass":
        info = _probe_media(filepath)
        vid_w = info.video.width if info.video else 1920
        vid_h = info.video.height if info.video else 1080
        export_ass(
            result, out_path,
            video_width=vid_w, video_height=vid_h,
            karaoke=word_timestamps,
        )
    else:
        export_srt(result, out_path)

    return {
        "output_path": out_path,
        "language": result.language,
        "segments": len(result.segments),
        "caption_segments": len(result.segments),
        "words": result.word_count,
    }


# ---------------------------------------------------------------------------
# Styled Captions (video overlay)
# ---------------------------------------------------------------------------
@captions_bp.route("/caption-styles", methods=["GET"])
def get_caption_styles():
    """Return available caption style presets for the panel preview."""
    try:
        from opencut.core.styled_captions import get_style_info
        return jsonify({"styles": get_style_info()})
    except Exception as e:
        return safe_error(e, "caption styles")


@captions_bp.route("/styled-captions", methods=["POST"])
@require_csrf
@async_job("styled-captions")
def styled_captions_route(job_id, filepath, data):
    """Generate a transparent video overlay with styled, animated captions."""
    output_dir = data.get("output_dir", "")

    style_name = data.get("style", "youtube_bold")
    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)
    custom_action_words = data.get("action_words", [])
    auto_detect_energy = data.get("auto_detect_energy", True)
    # Optional: pre-existing speech segments for remapping
    remap_segments_raw = data.get("remap_segments", None)

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.core.styled_captions import (
        detect_action_words_by_energy,
        get_action_word_indices,
        render_styled_caption_video,
    )

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper is required for styled captions. Install from the Captions tab.")

    # Get video resolution
    info = _probe_media(filepath)
    vid_w = info.video.width if info.video else 1920
    vid_h = info.video.height if info.video else 1080
    vid_fps = info.video.fps if info.video else 30.0

    # Step 1: Transcribe
    _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
    config = CaptionConfig(model=model, language=language, word_timestamps=True)
    transcription = transcribe(filepath, config=config)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Optional: remap captions to edited timeline
    if remap_segments_raw and isinstance(remap_segments_raw, list):
        from opencut.core.captions import remap_captions_to_segments
        from opencut.core.silence import TimeSegment
        remap_segs = []
        for s in remap_segments_raw:
            if isinstance(s, dict) and "start" in s and "end" in s:
                remap_segs.append(TimeSegment(
                    safe_float(s["start"], 0.0, min_val=0.0),
                    safe_float(s["end"], 0.0, min_val=0.0),
                ))
        if remap_segs:
            transcription = remap_captions_to_segments(transcription, remap_segs)

    # Step 2: Detect action words
    _update_job(job_id, progress=50, message="Detecting action words...")
    all_words = []
    for seg in transcription.segments:
        all_words.extend(seg.words)

    energy_indices = set()
    if auto_detect_energy and all_words:
        try:
            energy_indices = detect_action_words_by_energy(filepath, all_words)
        except Exception as e:
            logger.warning(f"Energy detection failed: {e}")

    action_indices = get_action_word_indices(
        all_words,
        custom_words=custom_action_words if custom_action_words else None,
        use_keywords=True,
        energy_indices=energy_indices,
    )

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step 3: Render overlay video
    _update_job(job_id, progress=60, message="Rendering styled captions...")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    overlay_path = os.path.join(effective_dir, f"{base_name}_captions.mov")

    def _on_render_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_styled_caption_video(
        transcription,
        overlay_path,
        style_name=style_name,
        video_width=vid_w,
        video_height=vid_h,
        fps=vid_fps,
        action_indices=action_indices,
        total_duration=transcription.duration or info.duration,
        on_progress=_on_render_progress,
    )

    return {
        "output_path": overlay_path,
        "overlay_path": overlay_path,
        "style": style_name,
        "frames_rendered": result["frames_rendered"],
        "action_words_found": len(action_indices),
        "caption_segments": len(transcription.segments),
        "words": sum(len(s.words) for s in transcription.segments),
        "language": transcription.language,
    }


# ---------------------------------------------------------------------------
# Transcript Editor
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript", methods=["POST"])
@require_csrf
@async_job("transcript")
def get_transcript(job_id, filepath, data):
    """Transcribe and return full word-level transcript for editing."""
    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)

    from opencut.core.captions import check_whisper_available, transcribe

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper is required for transcription.")

    _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
    config = CaptionConfig(model=model, language=language, word_timestamps=True)
    result = transcribe(filepath, config=config)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Build editable transcript structure
    return {
        "language": result.language,
        "duration": result.duration,
        "word_count": result.word_count,
        "full_text": result.text,
        "segments": [
            {
                "id": i,
                "text": seg.text.strip(),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.speaker,
                "words": [
                    {
                        "text": w.text.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "confidence": round(w.confidence, 3),
                    }
                    for w in seg.words
                ],
            }
            for i, seg in enumerate(result.segments)
        ],
    }


@captions_bp.route("/transcript/export", methods=["POST"])
@require_csrf
def export_edited_transcript():
    """Export an edited transcript to SRT/VTT/ASS/JSON."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    segments_data = data.get("segments", [])
    sub_format = data.get("format", "srt")
    if sub_format not in ("srt", "vtt", "json", "ass"):
        sub_format = "srt"
    language = data.get("language", "en")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not segments_data:
        return jsonify({"error": "No transcript segments provided"}), 400

    try:
        from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

        # Reconstruct TranscriptionResult from edited segments
        caption_segments = []
        for seg in segments_data:
            words = [
                Word(
                    text=w.get("text", ""),
                    start=safe_float(w.get("start", 0.0)),
                    end=safe_float(w.get("end", 0.0)),
                    confidence=safe_float(w.get("confidence", 1.0), default=1.0),
                )
                for w in seg.get("words", [])
            ]
            caption_segments.append(CaptionSegment(
                text=seg.get("text", ""),
                start=safe_float(seg.get("start", 0.0)),
                end=safe_float(seg.get("end", 0.0)),
                words=words,
                speaker=seg.get("speaker"),
            ))

        result = TranscriptionResult(
            segments=caption_segments,
            language=language,
            duration=caption_segments[-1].end if caption_segments else 0.0,
        )

        effective_dir = _resolve_output_dir(filepath, output_dir)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base_name}_edited.{sub_format}")

        if sub_format == "vtt":
            export_vtt(result, out_path)
        elif sub_format == "json":
            export_json(result, out_path)
        elif sub_format == "ass":
            info = _probe_media(filepath) if os.path.isfile(filepath) else None
            vid_w = info.video.width if info and info.video else 1920
            vid_h = info.video.height if info and info.video else 1080
            export_ass(result, out_path, video_width=vid_w, video_height=vid_h, karaoke=True)
        else:
            export_srt(result, out_path)

        return jsonify({"output_path": out_path, "format": sub_format})
    except Exception as e:
        return safe_error(e, "export_edited_transcript")


# ---------------------------------------------------------------------------
# Emoji Map for Captions
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "laugh": "\U0001f602", "lol": "\U0001f602", "haha": "\U0001f602", "funny": "\U0001f602",
    "love": "\u2764\ufe0f", "heart": "\u2764\ufe0f", "beautiful": "\u2764\ufe0f",
    "fire": "\U0001f525", "hot": "\U0001f525", "lit": "\U0001f525", "amazing": "\U0001f525",
    "cry": "\U0001f622", "sad": "\U0001f622", "tears": "\U0001f622",
    "wow": "\U0001f62e", "shocked": "\U0001f62e", "omg": "\U0001f62e", "unbelievable": "\U0001f62e",
    "think": "\U0001f914", "hmm": "\U0001f914", "wondering": "\U0001f914",
    "thumbs": "\U0001f44d", "great": "\U0001f44d", "good": "\U0001f44d", "nice": "\U0001f44d",
    "clap": "\U0001f44f", "bravo": "\U0001f44f", "congrats": "\U0001f44f",
    "money": "\U0001f4b0", "rich": "\U0001f4b0", "dollar": "\U0001f4b0", "expensive": "\U0001f4b0",
    "star": "\u2b50", "perfect": "\u2b50", "excellent": "\u2b50",
    "rocket": "\U0001f680", "launch": "\U0001f680", "skyrocket": "\U0001f680", "moon": "\U0001f680",
    "brain": "\U0001f9e0", "smart": "\U0001f9e0", "genius": "\U0001f9e0", "intelligent": "\U0001f9e0",
    "muscle": "\U0001f4aa", "strong": "\U0001f4aa", "power": "\U0001f4aa", "strength": "\U0001f4aa",
    "warning": "\u26a0\ufe0f", "danger": "\u26a0\ufe0f", "careful": "\u26a0\ufe0f",
    "check": "\u2705", "correct": "\u2705", "yes": "\u2705", "done": "\u2705",
    "wrong": "\u274c", "no": "\u274c", "false": "\u274c", "bad": "\u274c",
    "question": "\u2753", "why": "\u2753", "how": "\u2753",
    "eyes": "\U0001f440", "look": "\U0001f440", "see": "\U0001f440", "watch": "\U0001f440",
    "celebrate": "\U0001f389", "party": "\U0001f389", "winner": "\U0001f389", "victory": "\U0001f389",
    "sleep": "\U0001f634", "tired": "\U0001f634", "boring": "\U0001f634",
    "music": "\U0001f3b5", "song": "\U0001f3b5", "singing": "\U0001f3b5",
    "camera": "\U0001f4f8", "photo": "\U0001f4f8", "picture": "\U0001f4f8",
    "time": "\u23f0", "clock": "\u23f0", "hurry": "\u23f0", "fast": "\u23f0",
    "earth": "\U0001f30d", "world": "\U0001f30d", "global": "\U0001f30d",
    "sun": "\u2600\ufe0f", "sunny": "\u2600\ufe0f", "bright": "\u2600\ufe0f",
    "rain": "\U0001f327\ufe0f", "storm": "\U0001f327\ufe0f", "weather": "\U0001f327\ufe0f",
    "food": "\U0001f355", "eat": "\U0001f355", "hungry": "\U0001f355", "delicious": "\U0001f355",
    "coffee": "\u2615", "morning": "\u2615", "energy": "\u2615",
    "book": "\U0001f4da", "read": "\U0001f4da", "study": "\U0001f4da", "learn": "\U0001f4da",
    "computer": "\U0001f4bb", "code": "\U0001f4bb", "tech": "\U0001f4bb", "programming": "\U0001f4bb",
    "phone": "\U0001f4f1", "call": "\U0001f4f1", "mobile": "\U0001f4f1",
    "key": "\U0001f511", "secret": "\U0001f511", "unlock": "\U0001f511",
    "light": "\U0001f4a1", "idea": "\U0001f4a1", "tip": "\U0001f4a1", "insight": "\U0001f4a1",
    "target": "\U0001f3af", "goal": "\U0001f3af", "focus": "\U0001f3af", "aim": "\U0001f3af",
    "gem": "\U0001f48e", "diamond": "\U0001f48e", "valuable": "\U0001f48e", "premium": "\U0001f48e",
    "crown": "\U0001f451", "king": "\U0001f451", "queen": "\U0001f451", "best": "\U0001f451",
    "ghost": "\U0001f47b", "scary": "\U0001f47b", "spooky": "\U0001f47b", "halloween": "\U0001f47b",
    "skull": "\U0001f480", "dead": "\U0001f480", "kill": "\U0001f480", "destroy": "\U0001f480",
    "hundred": "\U0001f4af", "percent": "\U0001f4af",
}


@captions_bp.route("/captions/emoji-map", methods=["GET"])
def get_emoji_map():
    """Return the keyword->emoji mapping for auto-emoji insertion."""
    return jsonify({"emoji_map": EMOJI_MAP})


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
@captions_bp.route("/full", methods=["POST"])
@require_csrf
@async_job("full")
def full_pipeline(job_id, filepath, data):
    """Run silence removal + zoom + optional captions."""
    output_dir = data.get("output_dir", "")

    preset = data.get("preset", "youtube")
    skip_captions = data.get("skip_captions", False)
    skip_zoom = data.get("skip_zoom", False)
    remove_fillers = data.get("remove_fillers", False)
    seq_name = data.get("sequence_name", "")

    cfg = get_preset(preset)
    effective_name = seq_name or _make_sequence_name(filepath, "Full Edit")
    ecfg = ExportConfig(sequence_name=effective_name)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")
    srt_path = os.path.join(effective_dir, f"{base_name}_opencut.srt")

    # Probe once for all pipeline steps
    _finfo = _probe_media(filepath)
    _fdur = _finfo.duration if _finfo else 0.0

    # Calculate total steps
    total_steps = 2  # silence + export always
    if not skip_zoom:
        total_steps += 1
    if not skip_captions:
        total_steps += 1
    if remove_fillers:
        total_steps += 1
    step = [0]

    def next_step(msg):
        step[0] += 1
        pct = int(5 + (step[0] / total_steps) * 85)
        _update_job(job_id, progress=pct, message=f"Step {step[0]}/{total_steps}: {msg}")

    # Step: Silence detection
    next_step("Detecting silences...")
    segments = detect_speech(filepath, config=cfg.silence, file_duration=_fdur)
    summary = get_edit_summary(filepath, segments, file_duration=_fdur)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Filler word removal (requires Whisper)
    filler_stats = None
    transcription_result = None
    if remove_fillers:
        from opencut.core.captions import check_whisper_available, transcribe
        from opencut.core.fillers import detect_fillers, remove_fillers_from_segments

        available, backend = check_whisper_available()
        if available:
            next_step(f"Detecting filler words ({backend})...")
            filler_cfg = CaptionConfig(
                model=cfg.captions.model,
                language=cfg.captions.language,
                word_timestamps=True,
            )
            # Use timeout to prevent hanging (10 min max)
            try:
                transcription_result = transcribe(filepath, config=filler_cfg, timeout=600)
            except TimeoutError as te:
                logger.warning(f"Filler detection timed out: {te}")
                # Continue without filler removal
                _update_job(job_id, message="Filler detection timed out, continuing without it...")
                transcription_result = None
            except Exception as te:
                logger.warning(f"Filler detection failed: {te}")
                _update_job(job_id, message="Filler detection failed, continuing without it...")
                transcription_result = None

            if transcription_result:
                analysis = detect_fillers(transcription_result, include_context_fillers=True)

                if analysis.hits:
                    segments = remove_fillers_from_segments(segments, analysis.hits)
                    # Recalculate summary with filler-cleaned segments
                    summary = get_edit_summary(filepath, segments, file_duration=_fdur)
                    logger.info(
                        f"Fillers removed: {len(analysis.hits)} instances, "
                        f"{analysis.total_filler_time:.1f}s"
                    )

                filler_stats = {
                    "total_fillers": len(analysis.hits),
                    "removed_fillers": len(analysis.hits),
                    "total_filler_time": analysis.total_filler_time,
                    "filler_percentage": analysis.filler_percentage,
                    "total_words": analysis.total_words,
                    "breakdown": [
                        {"word": k, "count": c,
                         "time": round(sum(h.duration for h in analysis.hits if h.filler_key == k), 2),
                         "removed": True}
                        for k, c in sorted(analysis.filler_counts.items(), key=lambda x: -x[1])
                    ],
                }
        else:
            logger.warning("Filler removal requested but Whisper not installed, skipping")

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Zoom
    zoom_events = None
    if not skip_zoom:
        next_step("Analyzing emphasis points for zoom...")
        zoom_events = generate_zoom_events(filepath, config=cfg.zoom, speech_segments=segments)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Captions
    captions_result = None
    if not skip_captions:
        from opencut.core.captions import check_whisper_available
        available, backend = check_whisper_available()
        if available:
            next_step("Generating captions...")
            # Reuse transcription from filler step if available
            if transcription_result:
                captions_result = transcription_result
            else:
                from opencut.core.captions import transcribe
                captions_result = transcribe(filepath, config=cfg.captions)

            # Remap caption timestamps to the condensed timeline
            from opencut.core.captions import remap_captions_to_segments
            captions_result = remap_captions_to_segments(captions_result, segments)
            logger.info(
                f"Captions remapped: {len(captions_result.segments)} segments, "
                f"condensed duration {captions_result.duration:.1f}s"
            )

            export_srt(captions_result, srt_path)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Export XML
    _update_job(job_id, progress=92, message="Exporting Premiere XML...")
    export_premiere_xml(filepath, segments, xml_path, config=ecfg, zoom_events=zoom_events)

    result_data = {
        "xml_path": xml_path,
        "summary": summary,
        "segments": len(segments),
        "zoom_events": len(zoom_events) if zoom_events else 0,
        "segments_data": [
            {"start": round(s.start, 4), "end": round(s.end, 4)}
            for s in segments
        ],
    }
    if captions_result:
        result_data["srt_path"] = srt_path
        result_data["caption_segments"] = len(captions_result.segments)
        result_data["words"] = captions_result.word_count
        result_data["language"] = captions_result.language
    if filler_stats:
        result_data["filler_stats"] = filler_stats

    return result_data


# ---------------------------------------------------------------------------
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
        )
        return jsonify({
            "whisperx": check_whisperx_available(),
            "nllb": check_nllb_available(),
            "pysubs2": check_pysubs2_available(),
            "languages": TRANSLATION_LANGUAGES,
        })
    except Exception as e:
        return safe_error(e, "captions_enhanced_capabilities")


@captions_bp.route("/captions/whisperx", methods=["POST"])
@require_csrf
@async_job("whisperx")
def captions_whisperx(job_id, filepath, data):
    """Transcribe with WhisperX for word-level timestamps."""
    model_size = data.get("model", "base")
    if model_size not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model_size}")
    language = data.get("language", "")
    diarize = data.get("diarize", False)
    hf_token = data.get("hf_token", "")

    from opencut.core.captions_enhanced import whisperx_transcribe

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = whisperx_transcribe(
        filepath, model_size=model_size,
        language=language, diarize=diarize,
        hf_token=hf_token, on_progress=_on_progress,
    )
    return result


@captions_bp.route("/captions/translate", methods=["POST"])
@require_csrf
@async_job("translate", filepath_required=False)
def captions_translate(job_id, filepath, data):
    """Translate caption segments using NLLB or SeamlessM4T v2."""
    segments = data.get("segments", [])
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")
    backend = data.get("backend", "nllb")
    if backend not in ("nllb", "seamless"):
        backend = "nllb"

    if not segments:
        raise ValueError("No segments provided")
    if len(segments) > 10000:
        raise ValueError("Too many segments (max 10000)")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if backend == "seamless":
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

    return {"segments": translated, "target_lang": target_lang}


@captions_bp.route("/captions/karaoke", methods=["POST"])
@require_csrf
@async_job("karaoke", filepath_required=False)
def captions_karaoke(job_id, filepath, data):
    """Export segments as ASS karaoke subtitles."""
    segments = data.get("segments", [])
    output_dir = data.get("output_dir", "")

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
    data = request.get_json(force=True)
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
@async_job("install", filepath_required=False)
def captions_enhanced_install(job_id, filepath, data):
    """Install enhanced caption dependencies."""
    acquired = rate_limit("model_install")
    if not acquired:
        raise ValueError("Another model_install operation is already running. Please wait.")

    try:
        component = data.get("component", "whisperx")

        packages = {
            "whisperx": ["whisperx"],
            "pysubs2": ["pysubs2"],
            "nllb": ["ctranslate2", "sentencepiece", "huggingface-hub"],
        }

        pkgs = packages.get(component, [])
        if not pkgs:
            raise ValueError(f"Unknown component: {component}")

        for i, pkg in enumerate(pkgs):
            pct = int((i / len(pkgs)) * 90)
            _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
            safe_pip_install(pkg)
        return {"component": component}
    finally:
        if acquired:
            rate_limit_release("model_install")


# ---------------------------------------------------------------------------
# Caption Burn-in
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/burnin/styles", methods=["GET"])
def burnin_styles():
    """Return available burn-in styles."""
    try:
        from opencut.core.caption_burnin import get_burnin_styles
        return jsonify({"styles": get_burnin_styles()})
    except Exception as e:
        return safe_error(e, "burnin_styles")


@captions_bp.route("/captions/burnin/file", methods=["POST"])
@require_csrf
@async_job("burnin")
def burnin_from_file(job_id, filepath, data):
    """Burn a subtitle file into video."""
    video_path = filepath
    subtitle_path = data.get("subtitle_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not subtitle_path:
        raise ValueError("Subtitle file not found")
    subtitle_path = validate_filepath(subtitle_path)

    from opencut.core.caption_burnin import burnin_subtitles

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(video_path, output_dir)
    out = burnin_subtitles(
        video_path, subtitle_path,
        output_dir=effective_dir,
        font_size=safe_int(data.get("font_size", 0)),
        margin_bottom=safe_int(data.get("margin_bottom", 0)),
        force_style=_sanitize_force_style(data.get("force_style", "")),
        on_progress=_on_progress,
    )
    return {"output_path": out}


@captions_bp.route("/captions/burnin/segments", methods=["POST"])
@require_csrf
@async_job("burnin-seg")
def burnin_from_segments(job_id, filepath, data):
    """Burn caption segments directly into video."""
    video_path = filepath
    segments = data.get("segments", [])
    style = data.get("style", "default")
    if style not in _VALID_BURNIN_STYLES:
        style = "default"
    output_dir = data.get("output_dir", "")

    if not segments:
        raise ValueError("No segments provided")
    if len(segments) > 10000:
        raise ValueError("Too many segments (max 10000)")

    from opencut.core.caption_burnin import burnin_segments

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(video_path, output_dir)
    out = burnin_segments(
        video_path, segments, output_dir=effective_dir,
        style=style, on_progress=_on_progress,
    )
    return {"output_path": out, "style": style}


# ---------------------------------------------------------------------------
# Animated Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/animated/presets", methods=["GET"])
def animated_caption_presets():
    try:
        from opencut.core.animated_captions import get_animation_presets
        return jsonify({"presets": get_animation_presets()})
    except Exception as e:
        return safe_error(e, "animated_caption_presets")


@captions_bp.route("/captions/animated/render", methods=["POST"])
@require_csrf
@async_job("anim-cap")
def animated_caption_render(job_id, filepath, data):
    """Render animated word-by-word captions onto video."""
    words = data.get("word_segments", [])

    if not words:
        raise ValueError("No word segments")
    if len(words) > 50000:
        raise ValueError("Too many word segments (max 50000)")

    from opencut.core.animated_captions import render_animated_captions

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    _anim = data.get("animation", "pop")
    if _anim not in _VALID_ANIMATIONS:
        _anim = "pop"
    out = render_animated_captions(filepath, words, output_dir=d,
                                    animation=_anim,
                                    font_size=safe_int(data.get("font_size", 56), default=56),
                                    max_words_per_line=safe_int(data.get("max_words", 6), default=6),
                                    on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Transcript Summarization (LLM)
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript/summarize", methods=["POST"])
@require_csrf
@async_job("summarize")
def transcript_summarize(job_id, filepath, data):
    """Summarize a video transcript using LLM."""
    style = data.get("style", "bullets").strip()
    if style not in ("bullets", "paragraph", "detailed"):
        style = "bullets"
    transcript = data.get("transcript", None)

    # LLM config from request
    llm_provider = data.get("llm_provider", "ollama")
    if llm_provider not in ("ollama", "openai", "anthropic", "gemini"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "")
    llm_api_key = data.get("llm_api_key", "")
    llm_base_url = data.get("llm_base_url", "")

    from opencut.core.highlights import summarize_video
    from opencut.core.llm import LLMConfig

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
    )

    # If no transcript provided, transcribe first
    transcript_segments = transcript
    if not transcript_segments:
        _update_job(job_id, progress=5, message="Transcribing video first...")
        from opencut.core.captions import transcribe as _transcribe
        t_result = _transcribe(filepath)
        transcript_segments = [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in t_result.segments
        ]

    summary = summarize_video(
        transcript_segments,
        style=style,
        llm_config=llm_config,
        on_progress=_on_progress,
    )

    return {
        "text": summary.text,
        "bullet_points": summary.bullet_points,
        "topics": summary.topics,
        "word_count": summary.word_count,
    }


# ---------------------------------------------------------------------------
# Captions: Chapter Generation
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/chapters", methods=["POST"])
@require_csrf
@async_job("chapters", filepath_required=False)
def captions_chapters(job_id, filepath, data):
    """Generate YouTube-style chapters from a transcript using an LLM."""
    # filepath may come as "filepath" or "file" -- decorator gets "filepath",
    # but also check "file" fallback
    if not filepath:
        filepath = data.get("file", "").strip()
        if filepath:
            filepath = validate_filepath(filepath)

    segments = data.get("segments", None)
    llm_provider = data.get("llm_provider", "ollama")
    if llm_provider not in ("ollama", "openai", "anthropic", "gemini"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "llama3")
    api_key = data.get("api_key", "")
    max_chapters = safe_int(data.get("max_chapters", 15), 15, min_val=1, max_val=100)
    transcribe_model = data.get("model", "base")

    # Validate segments if provided directly
    if segments is not None:
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        if len(segments) > 50000:
            raise ValueError("Too many segments (max 50000)")

    if transcribe_model not in VALID_WHISPER_MODELS:
        transcribe_model = "base"

    effective_segments = segments

    # Transcribe if segments not provided
    if effective_segments is None:
        if not filepath:
            raise ValueError("file or segments required")

        from opencut.core.captions import check_whisper_available, transcribe

        available, backend = check_whisper_available()
        if not available:
            raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

        _update_job(job_id, progress=10, message="Transcribing for chapter generation...")
        config = CaptionConfig(model=transcribe_model, word_timestamps=False)

        # Check cache first
        cached = get_cached_transcript(filepath, model=transcribe_model)
        if cached:
            effective_segments = cached
            _update_job(job_id, progress=30, message="Using cached transcript...")
        else:
            result = transcribe(filepath, config=config)
            if hasattr(result, "segments"):
                effective_segments = [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in result.segments
                ]
                cache_transcript(filepath, effective_segments, model=transcribe_model)
            else:
                effective_segments = []

    if not effective_segments:
        raise ValueError("No transcript segments available")

    _update_job(job_id, progress=50, message="Generating chapters with LLM...")

    llm_config = None
    if LLMConfig is not None:
        llm_config = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key or "",
        )

    from opencut.core import chapter_gen
    result = chapter_gen.generate_chapters(
        effective_segments,
        llm_config=llm_config,
        max_chapters=max_chapters,
    )

    chapters = result.get("chapters", []) if isinstance(result, dict) else []
    description_block = result.get("description_block", "") if isinstance(result, dict) else str(result)

    return {"chapters": chapters, "description_block": description_block}


# ---------------------------------------------------------------------------
# Captions: Repeat / Duplicate Take Detection
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/repeat-detect", methods=["POST"])
@require_csrf
@async_job("repeat-detect")
def captions_repeat_detect(job_id, filepath, data):
    """Detect repeated takes in a recording and identify clean ranges."""
    model = data.get("model", "base")
    threshold = safe_float(data.get("threshold", 0.6), 0.6, min_val=0.0, max_val=1.0)
    gap_tolerance = safe_float(data.get("gap_tolerance", 2.0), 2.0, min_val=0.0, max_val=30.0)

    if model not in VALID_WHISPER_MODELS:
        model = "base"

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.utils.config import CaptionConfig as _CC

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    _update_job(job_id, progress=10, message="Transcribing with word-level timestamps...")

    config = _CC(model=model, word_timestamps=True)

    # Check cache
    cached = get_cached_transcript(filepath, model=model)
    if cached:
        segments = cached
        _update_job(job_id, progress=30, message="Using cached transcript...")
    else:
        result = transcribe(filepath, config=config)
        if hasattr(result, "segments"):
            segments = [
                {"start": s.start, "end": s.end, "text": s.text,
                 "words": [{"word": w.word, "start": w.start, "end": w.end}
                           for w in getattr(s, "words", [])] if hasattr(s, "words") else []}
                for s in result.segments
            ]
            cache_transcript(filepath, segments, model=model)
        else:
            segments = []

    if not segments:
        return {"repeats": [], "clean_ranges": [], "total_removed_seconds": 0.0}

    _update_job(job_id, progress=60, message="Detecting repeated takes...")

    from opencut.core import repeat_detect
    detection = repeat_detect.detect_repeated_takes(
        segments, threshold=threshold, gap_tolerance=gap_tolerance
    )

    repeats = detection.get("repeats", []) if isinstance(detection, dict) else []
    clean_ranges = detection.get("clean_ranges", []) if isinstance(detection, dict) else []
    total_removed = float(detection.get("total_removed_seconds", 0.0)) if isinstance(detection, dict) else 0.0

    return {
        "repeats": repeats,
        "clean_ranges": clean_ranges,
        "total_removed_seconds": total_removed,
    }
