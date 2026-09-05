"""
OpenCut Video Editing Routes

Reframe, auto-edit, auto-zoom, color match, multicam, emotion, highlights.
"""

import json
import logging
import os
import subprocess as _sp

from flask import Blueprint, jsonify

from opencut.core.workflow import workflow_step
from opencut.helpers import (
    _resolve_output_dir,
    _unique_output_path,
    get_ffmpeg_path,
    get_ffprobe_path,
)
from opencut.jobs import (
    _is_cancelled,
    _update_job,
    async_job,
)
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

video_editing_bp = Blueprint("video_editing", __name__)


# ---------------------------------------------------------------------------
# Video Reframe (resize/crop for phone / social media aspect ratios)
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/reframe/presets", methods=["GET"])
def reframe_presets():
    """Return available reframe presets with target dimensions."""
    return jsonify([
        {"id": "tiktok",          "name": "TikTok / Shorts",        "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "instagram_reel",  "name": "Instagram Reel / Story",  "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "instagram_post",  "name": "Instagram Post",          "width": 1080, "height": 1080, "aspect": "1:1"},
        {"id": "instagram_land",  "name": "Instagram Landscape",     "width": 1080, "height": 566,  "aspect": "1.91:1"},
        {"id": "youtube_short",   "name": "YouTube Short",           "width": 1080, "height": 1920, "aspect": "9:16"},
        {"id": "youtube",         "name": "YouTube (1080p)",         "width": 1920, "height": 1080, "aspect": "16:9"},
        {"id": "youtube_4k",      "name": "YouTube (4K)",            "width": 3840, "height": 2160, "aspect": "16:9"},
        {"id": "twitter",         "name": "Twitter / X",             "width": 1920, "height": 1080, "aspect": "16:9"},
        {"id": "square",          "name": "Square",                  "width": 1080, "height": 1080, "aspect": "1:1"},
        {"id": "custom",          "name": "Custom",                  "width": 0,    "height": 0,    "aspect": "custom"},
    ])


@video_editing_bp.route("/video/reframe", methods=["POST"])
@require_csrf
@workflow_step("Reframing video")
@async_job("reframe")
def video_reframe(job_id, filepath, data):
    """Reframe/resize video for a target aspect ratio using crop, pad, or scale."""
    target_w = safe_int(data.get("width", 1080), 1080, min_val=16, max_val=7680)
    target_h = safe_int(data.get("height", 1920), 1920, min_val=16, max_val=7680)
    _VALID_REFRAME_MODES = {"crop", "pad", "stretch"}
    mode = data.get("mode", "crop")
    if mode not in _VALID_REFRAME_MODES:
        mode = "crop"
    _VALID_POSITIONS = {"center", "top", "bottom", "left", "right", "auto"}
    position = data.get("position", "center")
    if position not in _VALID_POSITIONS:
        position = "center"
    bg_color = data.get("bg_color", "black")  # for pad mode
    import re as _re
    if not _re.match(r'^[a-zA-Z0-9#]+$', bg_color):
        bg_color = "black"
    quality = data.get("quality", "high")  # low, medium, high
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    if target_w < 16 or target_h < 16:
        raise ValueError("Target dimensions too small (min 16x16)")

    _update_job(job_id, progress=5, message="Probing video...")

    # Get source dimensions
    probe_cmd = [
        get_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", filepath
    ]
    probe_result = _sp.run(probe_cmd, capture_output=True, text=True, timeout=15)
    if probe_result.returncode != 0 or not probe_result.stdout.strip():
        raise ValueError("ffprobe failed on video file")
    try:
        probe_data = json.loads(probe_result.stdout)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("Could not read video metadata")
    streams = probe_data.get("streams", [])
    if not streams:
        raise ValueError("No video stream found")
    src_w = int(streams[0].get("width", 0))
    src_h = int(streams[0].get("height", 0))
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Could not determine video dimensions")

    _update_job(job_id, progress=10, message=f"Source: {src_w}x{src_h} -> Target: {target_w}x{target_h}")

    # Build the video filter chain based on mode
    if mode == "stretch":
        # Simple scale to exact target (distorts if aspect differs)
        vf = f"scale={target_w}:{target_h}"

    elif mode == "pad":
        # Scale to fit inside target, then pad with bg_color
        vf = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color={bg_color}"
        )

    else:
        # Crop mode (default) -- scale to cover target, then crop
        src_aspect = src_w / src_h
        tgt_aspect = target_w / target_h

        if abs(src_aspect - tgt_aspect) < 0.01:
            # Aspects match -- just scale
            vf = f"scale={target_w}:{target_h}"
        else:
            # Scale to cover target dimensions, then crop
            vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"

            # Determine crop position
            _effective_pos = position
            if position == "auto":
                # Auto-detect content region via cropdetect
                try:
                    import re as _re
                    cd_cmd = [
                        get_ffmpeg_path(), "-i", filepath,
                        "-vf", "cropdetect=24:16:0",
                        "-frames:v", "120", "-f", "null", "-"
                    ]
                    cd_result = _sp.run(cd_cmd, capture_output=True, text=True, timeout=30)
                    crops = _re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", cd_result.stderr)
                    if crops:
                        # Use the last (most stable) detection
                        cw, ch, cx, cy = [int(v) for v in crops[-1]]
                        # Center of the detected content region
                        content_cx = cx + cw // 2
                        content_cy = cy + ch // 2
                        prop_x = content_cx / src_w if src_w else 0.5
                        prop_y = content_cy / src_h if src_h else 0.5
                        # Clamp so the crop window stays inside the frame
                        crop_x = f"max(0,min(iw-{target_w},int(iw*{prop_x:.4f}-{target_w}/2)))"
                        crop_y = f"max(0,min(ih-{target_h},int(ih*{prop_y:.4f}-{target_h}/2)))"
                        _effective_pos = None  # skip named-position branch
                except Exception:
                    pass
                if _effective_pos == "auto":
                    _effective_pos = "center"  # fallback

            if _effective_pos == "top":
                crop_x, crop_y = "(iw-ow)/2", "0"
            elif _effective_pos == "bottom":
                crop_x, crop_y = "(iw-ow)/2", "(ih-oh)"
            elif _effective_pos == "left":
                crop_x, crop_y = "0", "(ih-oh)/2"
            elif _effective_pos == "right":
                crop_x, crop_y = "(iw-ow)", "(ih-oh)/2"
            elif _effective_pos is None:
                pass  # crop_x/crop_y already set by auto-detect
            else:
                # center (default)
                crop_x, crop_y = "(iw-ow)/2", "(ih-oh)/2"

            vf += f"crop={target_w}:{target_h}:{crop_x}:{crop_y}"

    # Quality mapping
    crf_map = {"low": "28", "medium": "23", "high": "18"}
    crf = crf_map.get(quality, "18")

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = _unique_output_path(
        os.path.join(effective_dir, f"{base_name}_reframed_{target_w}x{target_h}.mp4")
    )

    _update_job(job_id, progress=20, message=f"Reframing ({mode})...")

    cmd = [
        get_ffmpeg_path(), "-i", filepath,
        "-vf", vf,
        "-c:v", "libx264", "-crf", crf, "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", output_path
    ]
    proc = _sp.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        err_msg = (proc.stderr or "")[-500:]
        raise ValueError(f"FFmpeg failed: {err_msg}")

    if not os.path.isfile(output_path):
        raise ValueError("Output file not created")

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    return {
                    "output_path": output_path,
                    "summary": f"{src_w}x{src_h} -> {target_w}x{target_h} ({mode}), {size_mb} MB",
                    "width": target_w,
                    "height": target_h,
                    "size_mb": size_mb,
                    "mode": mode,
                }


# ---------------------------------------------------------------------------
# Auto-Edit (Motion-Based Editing via auto-editor)
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/auto-edit", methods=["POST"])
@require_csrf
@workflow_step("Auto-editing")
@async_job("auto_edit")
def video_auto_edit(job_id, filepath, data):
    """Run auto-editor to detect interesting segments by motion/audio."""
    from opencut.checks import check_auto_editor_available
    if not check_auto_editor_available():
        from opencut.core.auto_edit import INSTALL_HINT
        raise ValueError(f"auto-editor not installed. {INSTALL_HINT}")
    method = data.get("method", "motion").strip()
    if method not in ("motion", "audio", "both"):
        method = "motion"
    threshold = safe_float(data.get("threshold", 0.02), 0.02, min_val=0.001, max_val=1.0)
    margin = safe_float(data.get("margin", 0.2), 0.2, min_val=0.0, max_val=5.0)
    min_clip = safe_float(data.get("min_clip_length", 0.5), 0.5, min_val=0.1, max_val=10.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    from opencut.core.auto_edit import auto_edit

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_edit(
        filepath,
        method=method,
        threshold=threshold,
        margin=margin,
        min_clip_length=min_clip,
        export_xml=True,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return {
            "total_duration": result.total_duration,
            "kept_duration": result.kept_duration,
            "removed_duration": result.removed_duration,
            "reduction_percent": result.reduction_percent,
            "segments_count": len([s for s in result.segments if s.action == "keep"]),
            "xml_path": result.xml_path,
            "segments": [
                {"start": s.start, "end": s.end, "action": s.action}
                for s in result.segments
            ],
        }


# ---------------------------------------------------------------------------
# Face-Tracking Auto-Reframe
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/reframe/face", methods=["POST"])
@require_csrf
@workflow_step("Reframing to face")
@async_job("reframe_face")
def video_reframe_face(job_id, filepath, data):
    """Auto-reframe video to keep face centered using MediaPipe face detection."""
    from opencut.checks import check_mediapipe_available
    if not check_mediapipe_available():
        raise ValueError("mediapipe not installed. Install with: pip install mediapipe")
    target_w = safe_int(data.get("width", 1080), 1080, min_val=100, max_val=7680)
    target_h = safe_int(data.get("height", 1920), 1920, min_val=100, max_val=7680)
    smoothing = safe_float(data.get("smoothing", 0.15), 0.15, min_val=0.01, max_val=1.0)
    face_padding = safe_float(data.get("face_padding", 0.3), 0.3, min_val=0.0, max_val=1.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    from opencut.core.face_reframe import face_reframe

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    output_path = face_reframe(
        filepath,
        target_w=target_w,
        target_h=target_h,
        smoothing=smoothing,
        face_padding=face_padding,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return {"output_path": output_path}


# ---------------------------------------------------------------------------
# LLM Highlight Extraction
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/highlights", methods=["POST"])
@require_csrf
@workflow_step("Extracting highlights")
@async_job("highlights")
def video_highlights(job_id, filepath, data):
    """Extract highlight clips using LLM analysis of transcript."""
    max_highlights = safe_int(data.get("max_highlights", 5), 5, min_val=1, max_val=20)
    min_duration = safe_float(data.get("min_duration", 15.0), 15.0, min_val=5.0, max_val=300.0)
    max_duration = safe_float(data.get("max_duration", 60.0), 60.0, min_val=10.0, max_val=600.0)
    transcript = data.get("transcript", None)
    llm_provider = data.get("llm_provider", "ollama")
    if llm_provider not in ("ollama", "openai", "anthropic", "gemini"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "")
    llm_api_key = data.get("llm_api_key", "")
    llm_base_url = data.get("llm_base_url", "")

    from opencut.core.highlights import extract_highlights
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
        try:
            from opencut.core.captions import transcribe as _transcribe
            t_result = _transcribe(filepath)
            transcript_segments = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in t_result.segments
            ]
        except Exception as te:
            raise RuntimeError(f"Transcription failed: {te}")

    use_vision = safe_bool(data.get("use_vision", False), False)
    if use_vision:
        from opencut.core.highlights import extract_highlights_with_vision
        result = extract_highlights_with_vision(
            filepath,
            transcript_segments,
            max_highlights=max_highlights,
            min_duration=min_duration,
            max_duration=max_duration,
            llm_config=llm_config,
            on_progress=_on_progress,
        )
    else:
        result = extract_highlights(
            transcript_segments,
            max_highlights=max_highlights,
            min_duration=min_duration,
            max_duration=max_duration,
            llm_config=llm_config,
            on_progress=_on_progress,
        )

    return {
            "total_found": result.total_found,
            "highlights": [
                {
                    "start": h.start,
                    "end": h.end,
                    "score": h.score,
                    "reason": h.reason,
                    "title": h.title,
                    # Normalized 0-100 hook-forward virality score (heuristic) so
                    # creators can sort/threshold clips at a glance.
                    "virality": getattr(h.engagement, "virality", 0) if h.engagement else 0,
                    "engagement": {
                        "hook_strength": h.engagement.hook_strength,
                        "emotional_peak": h.engagement.emotional_peak,
                        "pacing": h.engagement.pacing,
                        "quotability": h.engagement.quotability,
                        "overall": h.engagement.overall,
                        "virality": h.engagement.virality,
                        "virality_weights": dict(h.engagement.virality_weights),
                        "virality_contributions": dict(h.engagement.virality_contributions),
                        "virality_breakdown": h.engagement.virality_breakdown(),
                    } if h.engagement else None,
                    "virality_breakdown": h.engagement.virality_breakdown() if h.engagement else [],
                }
                for h in result.highlights
            ],
        }


# ---------------------------------------------------------------------------
# Video: Color Match
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/color-match", methods=["POST"])
@require_csrf
@workflow_step("Matching colors")
@async_job("color_match", filepath_param="source")
def video_color_match(job_id, filepath, data):
    """Match the color grading of a source video to a reference clip."""
    source = data.get("source", "").strip()
    reference = data.get("reference", "").strip()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=1.0)
    if not source:
        raise ValueError("source is required")
    if not reference:
        raise ValueError("reference is required")
    try:
        source = validate_filepath(source)
    except ValueError as e:
        raise ValueError(str(e))
    try:
        reference = validate_filepath(reference)
    except ValueError as e:
        raise ValueError(str(e))
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            raise ValueError(str(e))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(source)

    from opencut.core import color_match

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    base_name = os.path.splitext(os.path.basename(source))[0]
    ext = os.path.splitext(source)[1]
    out_path = os.path.join(output_dir, f"{base_name}_color_matched{ext}")

    result = color_match.color_match_video(
        source, reference, out_path,
        strength=strength,
        on_progress=_on_progress,
    )

    out = result.get("output", out_path) if isinstance(result, dict) else out_path
    return {"output": out}


# ---------------------------------------------------------------------------
# Video: Auto Zoom
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/auto-zoom", methods=["POST"])
@require_csrf
@workflow_step("Applying auto-zoom")
@async_job("auto_zoom")
def video_auto_zoom(job_id, filepath, data):
    """Generate zoom keyframes for a clip, optionally baking them in via FFmpeg."""
    zoom_amount = safe_float(data.get("zoom_amount", 1.15), 1.15, min_val=1.0, max_val=4.0)
    easing = data.get("easing", "ease_in_out").strip()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    apply_to_file = safe_bool(data.get("apply_to_file", False), False)
    _VALID_EASINGS = {"ease_in", "ease_out", "ease_in_out", "linear"}
    if easing not in _VALID_EASINGS:
        easing = "ease_in_out"
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            raise ValueError(str(e))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(filepath)

    from opencut.core import auto_zoom

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    keyframes = auto_zoom.generate_zoom_keyframes(
        filepath, zoom_amount=zoom_amount, easing=easing,
        on_progress=_on_progress,
    )

    output_path = None
    if apply_to_file:
        _update_job(job_id, progress=60, message="Applying zoom via FFmpeg...")
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1]
        out_path = os.path.join(output_dir, f"{base_name}_autozoom{ext}")

        # Build FFmpeg zoompan filter from keyframes
        kf_data = keyframes.get("keyframes", []) if isinstance(keyframes, dict) else keyframes if isinstance(keyframes, list) else []
        if kf_data:
            # Get source dimensions via probe
            src_w, src_h = 1920, 1080
            source_fps = keyframes.get("fps", 30.0) if isinstance(keyframes, dict) else 30.0
            try:
                from opencut.helpers import get_video_info
                info = get_video_info(filepath)
                if info and info.get("width"):
                    src_w = int(info["width"])
                    src_h = int(info["height"])
                if not isinstance(keyframes, dict) and info and info.get("fps"):
                    source_fps = info["fps"]
            except Exception:
                pass
            zoompan_filter = auto_zoom.build_zoompan_filter(
                kf_data, src_w, src_h, source_fps
            )
            cmd = [
                get_ffmpeg_path(), "-y", "-i", filepath,
                "-vf", zoompan_filter,
                "-c:a", "copy",
                out_path,
            ]
            ffmpeg_result = _sp.run(cmd, capture_output=True, timeout=600)
            if ffmpeg_result.returncode == 0:
                output_path = out_path
            else:
                logger.warning("FFmpeg auto-zoom failed: %s", ffmpeg_result.stderr.decode(errors="replace")[:200])

    kf_list = keyframes.get("keyframes", []) if isinstance(keyframes, dict) else keyframes if isinstance(keyframes, list) else []
    result_dict = {"keyframes": kf_list}
    if output_path:
        result_dict["output"] = output_path

    return result_dict


# ---------------------------------------------------------------------------
# Video: Multicam Cuts
# ---------------------------------------------------------------------------
def _validate_multicam_cuts_request(data):
    segments = data.get("segments")
    if segments is not None:
        if not isinstance(segments, list):
            return "segments must be a list"
        if len(segments) > 50000:
            return "Too many segments (max 50000)"

    filepath = data.get("filepath", data.get("file", "")) or ""
    diarization_file = data.get("diarization_file", "") or ""
    if not isinstance(filepath, str) or not isinstance(diarization_file, str):
        return "filepath, diarization_file, and file must be strings"
    if not segments and not diarization_file.strip() and not filepath.strip():
        return "filepath, diarization_file, or segments required"
    return None


def _diarization_result_to_multicam_segments(result):
    """Convert a diarization result into the route's cut-segment shape."""
    raw_segments = getattr(result, "segments", None) or []
    segments = []
    for segment in raw_segments:
        if isinstance(segment, dict):
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            speaker = segment.get("speaker", "SPEAKER_00")
            text = segment.get("text", "")
        else:
            start = getattr(segment, "start", 0.0)
            end = getattr(segment, "end", 0.0)
            speaker = getattr(segment, "speaker", "SPEAKER_00")
            text = getattr(segment, "text", "")
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError, OverflowError):
            continue
        if end_value <= start_value:
            continue
        segments.append({
            "start": start_value,
            "end": end_value,
            "text": str(text or ""),
            "speaker": str(speaker or "SPEAKER_00"),
        })
    return segments, str(
        getattr(result, "boundary_source", "speaker_diarization")
        or "speaker_diarization"
    )


def _pyannote_segments_for_multicam(filepath, data):
    """Run optional pyannote diarization and return route-ready segments."""
    from opencut.core.diarize import check_pyannote_available, diarize

    if not check_pyannote_available():
        logger.info("pyannote.audio unavailable; falling back to ASR segments")
        return None, None

    from opencut.utils.config import DiarizeConfig

    raw_num_speakers = data.get("num_speakers")
    num_speakers = None
    if raw_num_speakers not in (None, ""):
        parsed_num_speakers = safe_int(raw_num_speakers, 0, min_val=1, max_val=10)
        if parsed_num_speakers > 0:
            num_speakers = parsed_num_speakers
    diarization_result = diarize(
        filepath,
        config=DiarizeConfig(num_speakers=num_speakers),
    )
    return _diarization_result_to_multicam_segments(diarization_result)


def _saved_multicam_config() -> dict:
    """Return the operator's saved multicam defaults, falling back in place."""
    fallback = {"min_cut_duration": 1.0, "gap_tolerance": 0.5, "default_speaker_count": 2}
    try:
        from opencut.user_data import load_multicam_config

        saved = load_multicam_config()
    except Exception as exc:  # pragma: no cover - unreadable settings
        logger.debug("Could not read saved multicam defaults: %s", exc)
        return fallback
    return {**fallback, **(saved if isinstance(saved, dict) else {})}


@video_editing_bp.route("/video/multicam-cuts", methods=["POST"])
@require_csrf
@async_job(
    "multicam-cuts",
    filepath_required=False,
    pre_validate=_validate_multicam_cuts_request,
)
def video_multicam_cuts(job_id, filepath, data):
    """Generate multicam cut points from speaker diarization data."""
    filepath = str(filepath or data.get("file", "") or "").strip()
    diarization_file = str(data.get("diarization_file", "") or "").strip()
    segments = data.get("segments", None)
    speaker_map = data.get("speaker_map", None)
    track_map = data.get("track_map", None)
    # The saved multicam defaults were stored and served but never applied, so
    # a minimum cut length set in Settings had no effect on a generated cut.
    multicam_defaults = _saved_multicam_config()
    min_cut_duration = safe_float(
        data.get("min_cut_duration", multicam_defaults["min_cut_duration"]),
        multicam_defaults["min_cut_duration"],
        min_val=0.1,
        max_val=60.0,
    )
    mode = str(data.get("mode") or "audio").strip().lower()
    strategy = str(data.get("strategy") or "speaker").strip().lower()
    boundary_source = "provided_segments" if segments is not None else "unknown"

    if filepath and not diarization_file and not segments:
        try:
            filepath = validate_filepath(filepath)
        except ValueError as e:
            raise ValueError(str(e)) from e

    if diarization_file:
        try:
            diarization_file = validate_filepath(diarization_file)
        except ValueError as e:
            raise ValueError(str(e)) from e

    # Load diarization file if segments not given directly
    effective_segments = segments
    if effective_segments is None and diarization_file:
        try:
            file_size = os.path.getsize(diarization_file)
            if file_size > 50_000_000:
                raise ValueError("Diarization file too large (max 50 MB)")
            import json as _json
            with open(diarization_file, encoding="utf-8") as _f:
                loaded = _json.load(_f)
            if isinstance(loaded, dict) and isinstance(loaded.get("segments"), list):
                effective_segments = loaded["segments"]
                boundary_source = str(loaded.get("boundary_source") or "diarization_file")
            else:
                effective_segments = loaded
                boundary_source = "diarization_file"
        except Exception as exc:
            raise ValueError(f"Could not read diarization_file: {exc}") from exc

    # Prefer pyannote 4's exclusive speaker annotation when the caller asks for
    # speaker boundaries (the default). It reconciles overlapping diarization
    # turns before cuts are generated. If the optional model or its token is
    # unavailable, retain the existing ASR-based path as an explicit fallback.
    if effective_segments is None and not diarization_file and filepath and strategy in {
        "auto", "diarization", "exclusive", "speaker"
    }:
        try:
            effective_segments, boundary_source = _pyannote_segments_for_multicam(
                filepath, data
            )
            if not effective_segments:
                effective_segments = None
                logger.info("Pyannote returned no speaker turns; falling back to ASR segments")
        except Exception as exc:
            logger.info("Speaker diarization unavailable; falling back to ASR segments: %s", exc)

    # If filepath given but no segments, attempt transcription as a deterministic
    # fallback for installations without pyannote or an accepted HF model.
    if effective_segments is None and not diarization_file and filepath:
        try:
            from opencut.core.captions import check_whisper_available, transcribe
            from opencut.utils.config import CaptionConfig
            available, _backend = check_whisper_available()
            if not available:
                raise RuntimeError(
                    "Whisper not installed. Cannot transcribe for multicam. "
                    "Provide segments or diarization_file instead."
                )
            config = CaptionConfig(model="base", word_timestamps=True)
            result = transcribe(filepath, config=config)
            effective_segments = []
            if hasattr(result, "segments"):
                for seg in result.segments:
                    effective_segments.append({
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", ""),
                        "speaker": getattr(seg, "speaker", "SPEAKER_00"),
                    })
            boundary_source = "asr_segments"
        except ImportError:
            raise RuntimeError(
                "Transcription modules not available. Provide segments directly."
            )
        except Exception as exc:
            raise RuntimeError(f"Multicam transcription failed: {exc}") from exc

    if not isinstance(effective_segments, list) or not effective_segments:
        raise ValueError("No valid segments found")

    try:
        from opencut.core import multicam

        # Auto-assign speakers if no map provided. The CEP panel sends its
        # ordered track choices as a list, so translate that order into the
        # speaker labels discovered by the diarization pass.
        effective_speaker_map = speaker_map if isinstance(speaker_map, dict) else None
        if not effective_speaker_map:
            effective_speaker_map = multicam.auto_assign_speakers(effective_segments)
            if isinstance(track_map, list):
                for speaker, appearance_index in list(effective_speaker_map.items()):
                    if appearance_index < len(track_map):
                        effective_speaker_map[speaker] = safe_int(
                            track_map[appearance_index],
                            appearance_index,
                            min_val=0,
                            max_val=100,
                        )

        # Cutting grammar. Every option is off by default, so an existing
        # client keeps the previous behaviour without sending anything new.
        _wide_track = data.get("wide_track")
        result = multicam.generate_multicam_cuts(
            effective_segments,
            speaker_to_track=effective_speaker_map,
            min_cut_duration=min_cut_duration,
            wide_track=(
                safe_int(_wide_track, 0, min_val=0, max_val=100)
                if _wide_track is not None
                else None
            ),
            wide_every_n_cuts=safe_int(
                data.get("wide_every_n_cuts", 0), 0, min_val=0, max_val=100
            ),
            wide_every_seconds=safe_float(
                data.get("wide_every_seconds", 0.0), 0.0, min_val=0.0, max_val=3600.0
            ),
            wide_duration=safe_float(
                data.get("wide_duration", 2.0), 2.0, min_val=0.0, max_val=60.0
            ),
            cut_on_interruption=safe_bool(data.get("cut_on_interruption", True), True),
            speaker_min_duration=(
                data.get("speaker_min_duration")
                if isinstance(data.get("speaker_min_duration"), dict)
                else None
            ),
        )
        cuts = result.get("cuts", []) if isinstance(result, dict) else result
        total_cuts = result.get("total_cuts", len(cuts)) if isinstance(result, dict) else len(cuts)

        visual_applied = False
        if mode == "audio+visual" and filepath and cuts:
            try:
                from opencut.core.multicam_visual import (
                    analyze_frames,
                    check_visual_multicam_available,
                    refine_cuts_with_visual,
                )
                if check_visual_multicam_available():
                    sample_times = [c["time"] for c in cuts]
                    cues = analyze_frames(filepath, sample_times)
                    speaker_face_map = data.get("speaker_face_map")
                    cuts = refine_cuts_with_visual(
                        cuts, cues.cues,
                        speaker_face_map=speaker_face_map,
                    )
                    total_cuts = len(cuts)
                    visual_applied = True
            except Exception as exc:
                logger.warning("Visual multicam refinement failed, using audio-only: %s", exc)

        speakers = sorted({str(s.get("speaker", "")) for s in effective_segments if s.get("speaker")})
        resp = {
            "cuts": cuts,
            "total_cuts": total_cuts,
            "speakers": speakers,
            "speaker_map": effective_speaker_map,
            "boundary_source": boundary_source,
            "uses_exclusive_speaker_diarization": boundary_source == "exclusive_speaker_diarization",
        }
        if mode == "audio+visual":
            resp["mode"] = "audio+visual" if visual_applied else "audio_fallback"
        return resp
    except ImportError:
        raise RuntimeError(
            "Multicam module not available. Install the multicam dependencies from the Settings tab."
        )
    except Exception as exc:
        raise RuntimeError(f"Multicam cuts failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Video: Multicam XML Export
# ---------------------------------------------------------------------------
def _validate_multicam_xml_request(data):
    cuts = data.get("cuts")
    if not cuts or not isinstance(cuts, list):
        return "cuts must be a non-empty list"
    return None


@video_editing_bp.route("/video/multicam-xml", methods=["POST"])
@require_csrf
@async_job(
    "multicam-xml",
    filepath_required=False,
    pre_validate=_validate_multicam_xml_request,
)
def multicam_xml_export(job_id, filepath, data):
    """Export multicam cuts as Premiere-compatible FCP XML.

    Expects JSON body:
    {
        "cuts": [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00", "track": 1}, ...],
        "source_files": {"SPEAKER_00": "/path/to/cam1.mp4", ...},
        "sequence_name": "My Multicam Edit",
        "fps": 29.97,
        "width": 1920,
        "height": 1080,
        "output_dir": ""
    }
    """
    cuts = data.get("cuts", [])

    source_files = data.get("source_files", {})
    sequence_name = str(data.get("sequence_name", "OpenCut Multicam"))
    fps_val = safe_float(data.get("fps", 29.97), 29.97, min_val=1, max_val=120)
    width_val = safe_int(data.get("width", 1920), 1920, min_val=1, max_val=7680)
    height_val = safe_int(data.get("height", 1080), 1080, min_val=1, max_val=4320)

    # Determine output path
    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            raise ValueError(str(e)) from e
    else:
        output_dir = os.path.join(os.path.expanduser("~"), ".opencut", "exports")
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize sequence name for filename
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in sequence_name).strip()
    if not safe_name:
        safe_name = "multicam"
    output_path = os.path.join(output_dir, f"{safe_name}.xml")

    try:
        from opencut.core.multicam_xml import generate_multicam_xml

        result = generate_multicam_xml(
            cuts=cuts,
            source_files=source_files,
            sequence_name=sequence_name,
            fps=fps_val,
            width=width_val,
            height=height_val,
            output_path=output_path,
        )

        return {
            "success": True,
            "output": result["output"],
            "cuts_count": result["cuts_count"],
            "duration": result["duration"],
        }

    except Exception as e:
        raise RuntimeError(f"Multicam XML export failed: {e}") from e


# ---------------------------------------------------------------------------
# Video: Emotion-Based Highlight Detection
# ---------------------------------------------------------------------------
@video_editing_bp.route("/video/emotion-highlights", methods=["POST"])
@require_csrf
@async_job("emotion-highlights", rate_limit_key="ai_gpu")
def video_emotion_highlights(job_id, filepath, data):
    """Detect emotional peaks in a video for highlight extraction.

    Analyzes facial expressions across frames to build an emotion curve,
    then identifies peaks as potential highlight moments.
    """
    sample_interval = safe_float(data.get("sample_interval", 1.0), 1.0, min_val=0.25, max_val=10.0)
    min_intensity = safe_float(data.get("min_intensity", 0.6), 0.6, min_val=0.1, max_val=1.0)
    min_duration = safe_float(data.get("min_duration", 2.0), 2.0, min_val=0.5, max_val=30.0)

    from opencut.core.emotion_highlights import (
        analyze_video_emotions,
        emotion_peaks_to_highlights,
    )

    def _on_progress(pct, msg=""):
        if _is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        _update_job(job_id, progress=pct, message=msg)

    curve = analyze_video_emotions(
        filepath,
        sample_interval=sample_interval,
        min_peak_intensity=min_intensity,
        min_peak_duration=min_duration,
        on_progress=_on_progress,
    )

    highlights = emotion_peaks_to_highlights(curve.peaks)

    return {
        "peaks": [
            {
                "time": p.time,
                "start": p.start,
                "end": p.end,
                "duration": p.duration,
                "emotion": p.emotion,
                "intensity": p.intensity,
            }
            for p in curve.peaks
        ],
        "highlights": highlights,
        "avg_intensity": curve.avg_intensity,
        "dominant_emotion": curve.dominant_emotion,
        "samples_analyzed": len(curve.samples),
    }
