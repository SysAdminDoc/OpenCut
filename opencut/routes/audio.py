"""
OpenCut Audio Routes

Silence removal, filler removal, denoise, isolate, separate, normalize,
effects, TTS, music AI, waveform preview.
"""

import array
import json
import logging
import os
import subprocess as _sp
import sys
import tempfile

from flask import Blueprint, jsonify, request

from opencut.checks import check_demucs_available
from opencut.errors import safe_error
from opencut.helpers import (
    _make_sequence_name,
    _resolve_output_dir,
    get_ffmpeg_path,
    get_ffprobe_path,
)
from opencut.jobs import (
    _is_cancelled,
    _register_job_process,
    _unregister_job_process,
    _update_job,
    async_job,
    make_install_route,
)
from opencut.security import (
    VALID_WHISPER_MODELS,
    rate_limit,
    rate_limit_release,
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
    validate_path,
)

# Core imports that many routes need
try:
    from opencut.core.silence import detect_speech, get_edit_summary
    from opencut.export.premiere import export_premiere_xml
    from opencut.utils.config import CaptionConfig, ExportConfig, SilenceConfig, get_preset
    from opencut.utils.media import probe as _probe_media
except ImportError:
    _probe_media = None
    SilenceConfig = ExportConfig = CaptionConfig = get_preset = None
    detect_speech = get_edit_summary = export_premiere_xml = None

logger = logging.getLogger("opencut")

audio_bp = Blueprint("audio", __name__)



# ---------------------------------------------------------------------------
# Silence Removal
# ---------------------------------------------------------------------------
@audio_bp.route("/silence", methods=["POST"])
@require_csrf
@async_job("silence")
def silence_remove(job_id, filepath, data):
    """Remove silences and export Premiere XML."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if detect_speech is None:
        raise ValueError("Core audio modules not available. Reinstall opencut.")

    threshold = safe_float(data.get("threshold", -30.0), -30.0, min_val=-60.0, max_val=0.0)
    min_duration = safe_float(data.get("min_duration", 0.5), 0.5, min_val=0.05, max_val=30.0)
    padding_before = safe_float(data.get("padding_before", 0.1), 0.1, min_val=0.0, max_val=5.0)
    padding_after = safe_float(data.get("padding_after", 0.1), 0.1, min_val=0.0, max_val=5.0)
    min_speech = safe_float(data.get("min_speech", 0.25), 0.25, min_val=0.05, max_val=10.0)
    preset = data.get("preset", None)
    seq_name = data.get("sequence_name", "")
    # Detection method: "energy" (FFmpeg threshold), "vad" (Silero VAD), "auto" (try VAD, fallback to energy)
    detection_method = data.get("method", "auto")
    if detection_method not in ("energy", "vad", "auto"):
        detection_method = "auto"

    _update_job(job_id, progress=5, message="Analyzing media file...")

    if preset:
        cfg = get_preset(preset)
        scfg = cfg.silence
    else:
        scfg = SilenceConfig(
            threshold_db=threshold,
            min_duration=min_duration,
            padding_before=padding_before,
            padding_after=padding_after,
            min_speech_duration=min_speech,
        )

    effective_name = seq_name or _make_sequence_name(filepath)
    ecfg = ExportConfig(sequence_name=effective_name)

    _update_job(job_id, progress=10, message="Probing media file...")
    _file_info = _probe_media(filepath)
    _file_dur = _file_info.duration if _file_info else 0.0

    method_label = "Silero VAD" if detection_method == "vad" else ("auto (VAD → energy fallback)" if detection_method == "auto" else "energy threshold")
    _update_job(job_id, progress=15, message=f"Detecting silences via {method_label}...")
    segments = detect_speech(filepath, config=scfg, file_duration=_file_dur, method=detection_method)

    if _is_cancelled(job_id):
        return

    _update_job(job_id, progress=70, message="Building edit summary...")
    summary = get_edit_summary(filepath, segments, file_duration=_file_dur)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")

    _update_job(job_id, progress=85, message="Exporting Premiere XML...")
    export_premiere_xml(filepath, segments, xml_path, config=ecfg)

    return {
        "xml_path": xml_path,
        "summary": summary,
        "segments": len(segments),
        "segments_data": [
            {"start": round(s.start, 4), "end": round(s.end, 4)}
            for s in segments
        ],
    }


# ---------------------------------------------------------------------------
# Filler Word Removal
# ---------------------------------------------------------------------------
@audio_bp.route("/fillers", methods=["POST"])
@require_csrf
@async_job("fillers")
def filler_removal(job_id, filepath, data):
    """Detect and remove filler words (um, uh, like, etc.)."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if detect_speech is None:
        raise ValueError("Core audio modules not available. Reinstall opencut.")

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        model = "base"
    language = data.get("language", None)
    include_context = safe_bool(data.get("include_context_fillers", True), True)
    custom_words = data.get("custom_words", [])
    # Which filler types to actually remove (empty = all detected)
    remove_keys = data.get("remove_fillers", [])
    # Also remove silences?
    remove_silence = safe_bool(data.get("remove_silence", True), True)
    silence_preset = data.get("silence_preset", "youtube")
    seq_name = data.get("sequence_name", "")
    # Filler detection backend: "whisper" (default) or "crisper" (CrisperWhisper verbatim)
    filler_backend = data.get("filler_backend", "whisper")
    if filler_backend not in ("whisper", "crisper"):
        filler_backend = "whisper"

    # CrisperWhisper path: use dedicated verbatim filler detector
    if filler_backend == "crisper":
        try:
            from opencut.core.crisper_whisper import detect_fillers_crisper

            def _on_progress(pct, msg=""):
                _update_job(job_id, progress=pct, message=msg)

            _update_job(job_id, progress=5, message="Detecting fillers with CrisperWhisper (verbatim mode)...")
            result = detect_fillers_crisper(
                filepath,
                language=language,
                custom_words=custom_words if custom_words else None,
                on_progress=_on_progress,
            )

            cuts = result.get("cuts", [])
            actual_method = result.get("method", "crisper_whisper")

            return {
                "cuts": cuts,
                "count": result["count"],
                "total_filler_time": result["total_filler_time"],
                "fillers": result["fillers"],
                "method": actual_method,
                "filler_stats": {
                    "removed_fillers": result["count"],
                    "total_filler_time": result["total_filler_time"],
                },
            }
        except ImportError as e:
            logger.warning("CrisperWhisper unavailable (%s), falling back to Whisper", e)
            _update_job(job_id, progress=5, message="CrisperWhisper unavailable, falling back to Whisper...")
            # Fall through to standard Whisper path
        except Exception as e:
            logger.exception("CrisperWhisper failed, falling back to Whisper")
            _update_job(job_id, progress=5, message=f"CrisperWhisper error ({e}), falling back to Whisper...")
            # Fall through to standard Whisper path

    # Standard Whisper path
    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.core.fillers import detect_fillers, remove_fillers_from_segments

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper is required for filler word detection. Install it from the Captions tab.")

    # Probe once for all steps
    _finfo = _probe_media(filepath)
    _fdur = _finfo.duration if _finfo else 0.0

    # Step 1: Silence detection (optional)
    segments = None
    if remove_silence:
        _update_job(job_id, progress=5, message="Step 1/4: Detecting silences...")
        cfg = get_preset(silence_preset)
        segments = detect_speech(filepath, config=cfg.silence, file_duration=_fdur)
        get_edit_summary(filepath, segments, file_duration=_fdur)
    else:
        # Use the whole file as one segment
        from opencut.core.silence import TimeSegment
        segments = [TimeSegment(start=0.0, end=_fdur or 0.0, label="speech")]

    if _is_cancelled(job_id):
        return

    # Step 2: Transcribe with word timestamps
    _update_job(job_id, progress=20, message=f"Step 2/4: Transcribing with {backend} ({model})...")
    config = CaptionConfig(
        model=model,
        language=language,
        word_timestamps=True,
    )
    transcription = transcribe(filepath, config=config)

    if _is_cancelled(job_id):
        return

    # Step 3: Detect fillers
    _update_job(job_id, progress=65, message="Step 3/4: Detecting filler words...")
    analysis = detect_fillers(
        transcription,
        include_context_fillers=include_context,
        custom_words=custom_words if custom_words else None,
    )

    logger.info(
        f"Filler analysis: {len(analysis.hits)} fillers found "
        f"({analysis.filler_percentage}% of words), "
        f"{analysis.total_filler_time:.1f}s total filler time"
    )

    if _is_cancelled(job_id):
        return

    # Step 4: Remove selected fillers from segments
    _update_job(job_id, progress=75, message="Step 4/4: Removing fillers from segments...")

    # Filter hits to only the ones the user wants removed
    hits_to_remove = analysis.hits
    if remove_keys:
        key_set = set(remove_keys)
        hits_to_remove = [h for h in analysis.hits if h.filler_key in key_set]

    refined_segments = remove_fillers_from_segments(segments, hits_to_remove)

    # Build summary for refined segments
    refined_summary = get_edit_summary(filepath, refined_segments, file_duration=_fdur)

    # Export XML
    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_path = os.path.join(effective_dir, f"{base_name}_cleancut.xml")

    effective_name = seq_name or _make_sequence_name(filepath, "Clean Cut")
    ecfg = ExportConfig(sequence_name=effective_name)

    _update_job(job_id, progress=90, message="Exporting Premiere XML...")
    export_premiere_xml(filepath, refined_segments, xml_path, config=ecfg)

    # Build per-filler breakdown for UI
    filler_breakdown = []
    for key, count in sorted(analysis.filler_counts.items(), key=lambda x: -x[1]):
        time_for_key = sum(
            h.duration for h in analysis.hits if h.filler_key == key
        )
        filler_breakdown.append({
            "word": key,
            "count": count,
            "time": round(time_for_key, 2),
            "removed": (not remove_keys) or (key in set(remove_keys)),
        })

    return {
        "xml_path": xml_path,
        "summary": refined_summary,
        "segments": len(refined_segments),
        "segments_data": [
            {"start": round(s.start, 4), "end": round(s.end, 4)}
            for s in refined_segments
        ],
        "filler_stats": {
            "total_fillers": len(analysis.hits),
            "removed_fillers": len(hits_to_remove),
            "total_filler_time": analysis.total_filler_time,
            "filler_percentage": analysis.filler_percentage,
            "total_words": analysis.total_words,
            "breakdown": filler_breakdown,
        },
        "language": transcription.language,
    }


# ---------------------------------------------------------------------------
# Audio Suite: Noise Reduction
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/denoise", methods=["POST"])
@require_csrf
@async_job("denoise")
def audio_denoise(job_id, filepath, data):
    """Remove background noise from audio/video."""
    from opencut.core.audio_suite import denoise_audio

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    method = data.get("method", "afftdn")
    if method not in ("afftdn", "highpass", "gate"):
        method = "afftdn"
    strength = safe_float(data.get("strength", 0.7), 0.7, min_val=0.0, max_val=1.0)
    noise_floor = safe_float(data.get("noise_floor", -30.0), -30.0, min_val=-80.0, max_val=0.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    out_path = os.path.join(effective_dir, f"{base_name}_denoised{ext}")

    denoise_audio(
        filepath, out_path,
        method=method, strength=strength,
        noise_floor=noise_floor,
        on_progress=_on_progress,
    )

    return {
        "output_path": out_path,
        "method": method,
        "strength": strength,
    }


# ---------------------------------------------------------------------------
# Audio Suite: Voice Isolation
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/isolate", methods=["POST"])
@require_csrf
@async_job("isolate")
def audio_isolate(job_id, filepath, data):
    """Isolate vocals by emphasizing voice frequencies."""
    from opencut.core.audio_suite import isolate_voice

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    out_path = os.path.join(effective_dir, f"{base_name}_voice{ext}")

    isolate_voice(filepath, out_path, on_progress=_on_progress)

    return {"output_path": out_path}


# ---------------------------------------------------------------------------
# Audio Suite: AI Stem Separation (Demucs)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/separate", methods=["POST"])
@require_csrf
@async_job("separate")
def audio_separate(job_id, filepath, data):
    """Separate audio into stems using AI (Demucs or audio-separator with RoFormer models)."""
    import shutil

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    backend = data.get("backend", "demucs")
    if backend not in ("demucs", "audio-separator"):
        backend = "demucs"
    model = data.get("model", "htdemucs" if backend == "demucs" else "mel_band_roformer")
    allowed_demucs = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"}
    allowed_separator = {"mel_band_roformer", "bs_roformer", "scnet", "mdx23c", "htdemucs"}
    allowed_models = allowed_demucs if backend == "demucs" else allowed_separator
    if model not in allowed_models:
        raise ValueError(f"Unknown model: {model}. Allowed: {', '.join(sorted(allowed_models))}")
    stems = data.get("stems", ["vocals", "no_vocals"])
    valid_stems = {"vocals", "drums", "bass", "other", "no_vocals", "guitar", "piano"}
    stems = [s for s in stems if isinstance(s, str) and s in valid_stems] or ["vocals", "no_vocals"]
    output_format = data.get("format", "wav")
    if output_format not in ("wav", "mp3", "flac"):
        output_format = "wav"
    auto_import = safe_bool(data.get("auto_import", True), True)

    if backend == "demucs" and not check_demucs_available():
        raise ValueError("Demucs not installed. Please install it first.")

    temp_audio = None
    try:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        base_name = os.path.splitext(os.path.basename(filepath))[0]

        _update_job(job_id, progress=5, message="Extracting audio...")

        # Extract audio if video file
        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
        file_ext = os.path.splitext(filepath)[1].lower()

        if file_ext in video_exts:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_audio = tmp.name

            ffmpeg_cmd = [
                get_ffmpeg_path(), '-y', '-i', filepath,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                temp_audio
            ]
            _sp.run(ffmpeg_cmd, check=True, capture_output=True, timeout=300)
            input_audio = temp_audio
        else:
            input_audio = filepath

        if backend == "audio-separator":
            # Use python-audio-separator with RoFormer/MDX models
            try:
                from audio_separator.separator import Separator
            except ImportError:
                raise RuntimeError("audio-separator not installed. Run: pip install audio-separator[cpu]")

            _update_job(job_id, progress=15, message=f"Running audio-separator ({model})...")
            separator = Separator(output_dir=effective_dir, output_format=output_format)
            separator.load_model(model_filename=model)
            output_files = separator.separate(input_audio)

            output_paths = []
            # Filter to requested stems only
            requested = set(stems)
            for f in output_files:
                if os.path.isfile(f):
                    fname = os.path.splitext(os.path.basename(f))[0].lower()
                    # Match if any requested stem appears in filename
                    if not requested or any(s in fname for s in requested):
                        output_paths.append(f)

            return {
                "output_paths": output_paths,
                "auto_import": auto_import,
            }

        _update_job(job_id, progress=15, message=f"Running Demucs ({model})...")

        # Create temp output directory for demucs
        with tempfile.TemporaryDirectory() as temp_out:
            # Run demucs
            demucs_cmd = [
                sys.executable, '-m', 'demucs',
                '--out', temp_out,
                '-n', model,
                '--two-stems=vocals' if set(stems) <= {'vocals', 'no_vocals'} else '',
                input_audio
            ]
            # Remove empty strings from command
            demucs_cmd = [c for c in demucs_cmd if c]

            process = _sp.Popen(
                demucs_cmd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                universal_newlines=True
            )
            _register_job_process(job_id, process)

            # Monitor progress
            for line in process.stdout:
                if _is_cancelled(job_id):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except Exception:
                        logger.warning("Process did not exit after terminate, killing")
                        process.kill()
                    _unregister_job_process(job_id)
                    return

                if '%' in line:
                    try:
                        pct_str = line.split('%')[0].strip().split()[-1]
                        pct = int(float(pct_str))
                        _update_job(job_id, progress=15 + int(pct * 0.7), message=f"Separating audio... {pct}%")
                    except (ValueError, IndexError):
                        pass

            process.wait()
            _unregister_job_process(job_id)

            if process.returncode != 0:
                raise Exception("Demucs separation failed")

            _update_job(job_id, progress=90, message="Saving stems...")

            # Find output files
            # Demucs outputs to: temp_out/model_name/track_name/stem.wav
            demucs_out_dir = None
            for root, dirs, files in os.walk(temp_out):
                if any(f.endswith('.wav') for f in files):
                    demucs_out_dir = root
                    break

            if not demucs_out_dir:
                raise Exception("Could not find Demucs output files")

            output_paths = []
            stem_map = {
                'vocals': 'vocals',
                'no_vocals': 'no_vocals',
                'drums': 'drums',
                'bass': 'bass',
                'other': 'other',
                'piano': 'piano',
                'guitar': 'guitar'
            }

            for stem in stems:
                stem_file = stem_map.get(stem, stem)
                src_path = os.path.join(demucs_out_dir, f"{stem_file}.wav")

                # Check alternate naming (instrumental vs no_vocals)
                if not os.path.exists(src_path) and stem == 'no_vocals':
                    src_path = os.path.join(demucs_out_dir, "instrumental.wav")

                if os.path.exists(src_path):
                    # Convert to desired format
                    out_filename = f"{base_name}_{stem}.{output_format}"
                    out_path = os.path.join(effective_dir, out_filename)

                    if output_format == 'wav':
                        shutil.copy2(src_path, out_path)
                    elif output_format == 'mp3':
                        _sp.run([
                            get_ffmpeg_path(), '-y', '-i', src_path,
                            '-codec:a', 'libmp3lame', '-b:a', '320k',
                            out_path
                        ], check=True, capture_output=True, timeout=300)
                    elif output_format == 'flac':
                        _sp.run([
                            get_ffmpeg_path(), '-y', '-i', src_path,
                            '-codec:a', 'flac',
                            out_path
                        ], check=True, capture_output=True, timeout=300)

                    output_paths.append(out_path)

        return {
            "output_paths": output_paths,
            "auto_import": auto_import,
        }
    finally:
        _unregister_job_process(job_id)
        # Cleanup temp audio extracted from video
        try:
            if temp_audio and os.path.exists(temp_audio):
                os.unlink(temp_audio)
        except (OSError, NameError):
            pass


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Normalization
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/normalize", methods=["POST"])
@require_csrf
@async_job("normalize")
def audio_normalize(job_id, filepath, data):
    """Normalize audio loudness to broadcast/platform standards."""
    from opencut.core.audio_suite import LOUDNESS_PRESETS, normalize_loudness

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    preset = data.get("preset", "youtube")
    target_lufs = data.get("target_lufs", None)
    if target_lufs is not None:
        target_lufs = safe_float(target_lufs, -16.0, min_val=-70.0, max_val=0.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    out_path = os.path.join(effective_dir, f"{base_name}_normalized{ext}")

    targets = LOUDNESS_PRESETS.get(preset, LOUDNESS_PRESETS["youtube"])

    out_path, info = normalize_loudness(
        filepath, out_path,
        preset=preset,
        target_lufs=target_lufs,
        on_progress=_on_progress,
    )

    return {
        "output_path": out_path,
        "preset": preset,
        "input_loudness": info.input_i,
        "target_loudness": targets["i"],
        "input_peak": info.input_tp,
        "loudness_range": info.input_lra,
    }


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Measurement
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/measure", methods=["POST"])
@require_csrf
def audio_measure():
    """Measure audio loudness (EBU R128) without processing."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.audio_suite import measure_loudness

        info = measure_loudness(filepath)
        return jsonify({
            "integrated_lufs": info.input_i,
            "true_peak_dbtp": info.input_tp,
            "loudness_range_lu": info.input_lra,
        })
    except Exception as e:
        return safe_error(e, "audio_measure")


# ---------------------------------------------------------------------------
# Audio Suite: Beat Detection
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/beats", methods=["POST"])
@require_csrf
@async_job("beats")
def audio_beats(job_id, filepath, data):
    """Detect beats and estimate BPM."""
    from opencut.core.audio_suite import detect_beats

    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5, min_val=0.0, max_val=1.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    info = detect_beats(filepath, sensitivity=sensitivity, on_progress=_on_progress)

    return {
        "bpm": info.bpm,
        "total_beats": len(info.beat_times),
        "beat_times": [round(t, 3) for t in info.beat_times[:500]],
        "downbeat_times": [round(t, 3) for t in info.downbeat_times[:125]],
        "confidence": round(info.confidence, 3),
    }


# ---------------------------------------------------------------------------
# Audio Suite: Audio Effects
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/effects", methods=["GET"])
def audio_effects_list():
    """Return available audio effects."""
    try:
        from opencut.core.audio_suite import get_available_effects
        return jsonify({"effects": get_available_effects()})
    except Exception as e:
        return safe_error(e, "audio_effects_list")


@audio_bp.route("/audio/effects/apply", methods=["POST"])
@require_csrf
@async_job("audio-effect")
def audio_effects_apply(job_id, filepath, data):
    """Apply an audio effect."""
    from opencut.core.audio_suite import apply_audio_effect

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    effect = data.get("effect", "")

    if not effect:
        raise ValueError("No effect specified")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1]
    out_path = os.path.join(effective_dir, f"{base_name}_{effect}{ext}")

    apply_audio_effect(filepath, effect, out_path, on_progress=_on_progress)

    return {"output_path": out_path, "effect": effect}


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Presets
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/loudness-presets", methods=["GET"])
def loudness_presets():
    """Return available loudness normalization presets."""
    try:
        from opencut.core.audio_suite import LOUDNESS_PRESETS
        presets = []
        for name, vals in LOUDNESS_PRESETS.items():
            presets.append({
                "name": name,
                "label": name.replace("_", " ").title(),
                "target_lufs": vals["i"],
                "target_tp": vals["tp"],
                "target_lra": vals["lra"],
            })
        return jsonify({"presets": presets})
    except Exception as e:
        return safe_error(e, "loudness_presets")


# ---------------------------------------------------------------------------
# Audio Pro (Pedalboard studio effects, DeepFilterNet)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/pro/effects", methods=["GET"])
def audio_pro_effects():
    """Return available Pedalboard effects."""
    try:
        from opencut.core.audio_pro import check_pedalboard_available, get_pedalboard_effects
        return jsonify({
            "effects": get_pedalboard_effects(),
            "available": check_pedalboard_available(),
        })
    except Exception as e:
        return safe_error(e, "audio_pro_effects")


@audio_bp.route("/audio/pro/apply", methods=["POST"])
@require_csrf
@async_job("audio-pro")
def audio_pro_apply(job_id, filepath, data):
    """Apply a Pedalboard studio effect."""
    from opencut.core.audio_pro import apply_pedalboard_effect

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    effect = data.get("effect", "").strip()
    params = data.get("params", {})

    if not effect:
        raise ValueError("No effect specified")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = apply_pedalboard_effect(
        filepath, effect, output_dir=effective_dir,
        params=params, on_progress=_on_progress,
    )

    return {"output_path": out, "effect": effect}


@audio_bp.route("/audio/pro/deepfilter", methods=["POST"])
@require_csrf
@async_job("deepfilter")
def audio_pro_deepfilter(job_id, filepath, data):
    """AI noise reduction using DeepFilterNet."""
    from opencut.core.audio_pro import deepfilter_denoise

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = deepfilter_denoise(
        filepath, output_dir=effective_dir,
        on_progress=_on_progress,
    )

    return {"output_path": out}


@audio_bp.route("/audio/pro/install", methods=["POST"])
@require_csrf
@async_job("install", filepath_required=False)
def audio_pro_install(job_id, filepath, data):
    """Install audio pro dependencies."""
    acquired = rate_limit("model_install")
    if not acquired:
        raise ValueError("Another model_install operation is already running. Please wait.")

    try:
        component = data.get("component", "pedalboard")

        packages = {
            "pedalboard": ["pedalboard"],
            "deepfilter": ["deepfilternet"],
        }

        pkgs = packages.get(component, [])
        if not pkgs:
            raise ValueError(f"Unknown component: {component}")

        for i, pkg in enumerate(pkgs):
            pct = int((i / len(pkgs)) * 90)
            _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
            safe_pip_install(pkg, timeout=600)

        return {"component": component}
    finally:
        if acquired:
            rate_limit_release("model_install")


# ---------------------------------------------------------------------------
# Voice Generation (TTS)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/tts/voices", methods=["GET"])
def tts_voices():
    """Return available TTS voices."""
    try:
        from opencut.core.voice_gen import (
            KOKORO_VOICES,
            check_edge_tts_available,
            check_kokoro_available,
            get_voice_list,
        )
        return jsonify({
            "edge_tts": {
                "available": check_edge_tts_available(),
                "voices": get_voice_list("edge"),
            },
            "kokoro": {
                "available": check_kokoro_available(),
                "voices": KOKORO_VOICES,
            },
        })
    except Exception as e:
        return safe_error(e, "tts_voices")


def _validate_tts_text(data):
    text = (data.get("text") or "").strip()
    if not text:
        return "No text provided"
    if len(text) > 50000:
        return "Text too long (max 50000 chars)"
    return None


@audio_bp.route("/audio/tts/generate", methods=["POST"])
@require_csrf
@async_job("tts", filepath_required=False, pre_validate=_validate_tts_text)
def tts_generate(job_id, filepath, data):
    """Generate speech from text."""
    import re as _re_tts

    text = data.get("text", "").strip()
    engine = data.get("engine", "edge")
    if engine not in ("edge", "kokoro", "chatterbox"):
        engine = "edge"
    voice = data.get("voice", "en-US-AriaNeural")
    rate = data.get("rate", "+0%")
    if not isinstance(rate, str) or not _re_tts.match(r'^[+-]?\d{1,3}%$', rate):
        rate = "+0%"
    pitch = data.get("pitch", "+0Hz")
    if not isinstance(pitch, str) or not _re_tts.match(r'^[+-]?\d{1,4}Hz$', pitch):
        pitch = "+0Hz"
    speed = safe_float(data.get("speed", 1.0), 1.0, min_val=0.25, max_val=4.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not text:
        raise ValueError("No text provided")
    if len(text) > 50000:
        raise ValueError("Text too long (max 50000 chars)")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir or tempfile.gettempdir()

    if engine == "chatterbox":
        from opencut.core.voice_gen import chatterbox_generate
        voice_ref = data.get("voice_ref", "")
        if voice_ref:
            try:
                voice_ref = validate_filepath(voice_ref)
            except ValueError:
                voice_ref = None
        else:
            voice_ref = None
        exaggeration = safe_float(data.get("exaggeration", 0.5), 0.5, min_val=0.0, max_val=1.0)
        out = chatterbox_generate(
            text, voice_ref=voice_ref, output_dir=effective_dir,
            exaggeration=exaggeration, on_progress=_on_progress,
        )
    elif engine == "kokoro":
        from opencut.core.voice_gen import kokoro_generate
        out = kokoro_generate(
            text, voice=voice, output_dir=effective_dir,
            speed=speed, on_progress=_on_progress,
        )
    else:
        from opencut.core.voice_gen import edge_tts_generate
        out = edge_tts_generate(
            text, voice=voice, output_dir=effective_dir,
            rate=rate, pitch=pitch, on_progress=_on_progress,
        )

    return {"output_path": out, "engine": engine, "voice": voice}


@audio_bp.route("/audio/tts/subtitled", methods=["POST"])
@require_csrf
@async_job("tts-sub", filepath_required=False, pre_validate=_validate_tts_text)
def tts_subtitled(job_id, filepath, data):
    """Generate speech + word-level subtitles."""
    from opencut.core.voice_gen import edge_tts_with_subtitles

    text = data.get("text", "").strip()
    voice = data.get("voice", "en-US-AriaNeural")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not text:
        raise ValueError("No text provided")
    if len(text) > 50000:
        raise ValueError("Text too long (max 50000 chars)")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir or tempfile.gettempdir()
    result = edge_tts_with_subtitles(
        text, voice=voice, output_dir=effective_dir,
        on_progress=_on_progress,
    )

    return result


@audio_bp.route("/audio/tts/install", methods=["POST"])
@require_csrf
@async_job("install", filepath_required=False)
def tts_install(job_id, filepath, data):
    """Install TTS engine dependencies."""
    acquired = rate_limit("model_install")
    if not acquired:
        raise ValueError("Another model_install operation is already running. Please wait.")

    try:
        component = data.get("component", "edge_tts")

        packages = {
            "edge_tts": ["edge-tts"],
            "kokoro": ["kokoro>=0.3", "soundfile"],
        }
        pkgs = packages.get(component, [])
        if not pkgs:
            raise ValueError(f"Unknown component: {component}")

        for pkg in pkgs:
            _update_job(job_id, progress=30, message=f"Installing {pkg}...")
            safe_pip_install(pkg, timeout=600)

        return {"component": component}
    finally:
        if acquired:
            rate_limit_release("model_install")

make_install_route(audio_bp, "/audio/crisper-whisper/install", "crisper_whisper",
                   ["torch", "transformers"],
                   doc="Install CrisperWhisper dependencies.")


# ---------------------------------------------------------------------------
# Music & SFX Generation
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/gen/capabilities", methods=["GET"])
def audio_gen_capabilities():
    """Return available audio generation capabilities."""
    try:
        from opencut.core.music_gen import get_audio_generators
        return jsonify(get_audio_generators())
    except Exception as e:
        return safe_error(e, "audio_gen_capabilities")


@audio_bp.route("/audio/gen/tone", methods=["POST"])
@require_csrf
@async_job("tone", filepath_required=False)
def audio_gen_tone(job_id, filepath, data):
    """Generate a tone."""
    from opencut.core.music_gen import generate_tone

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir or tempfile.gettempdir()
    out = generate_tone(
        output_dir=effective_dir,
        frequency=safe_float(data.get("frequency", 440), 440.0, min_val=20.0, max_val=20000.0),
        duration=safe_float(data.get("duration", 3.0), 3.0, min_val=0.1, max_val=300.0),
        waveform=data.get("waveform", "sine"),
        volume=safe_float(data.get("volume", 0.5), 0.5, min_val=0.0, max_val=1.0),
        fade_in=safe_float(data.get("fade_in", 0.1), 0.1, min_val=0.0, max_val=30.0),
        fade_out=safe_float(data.get("fade_out", 0.5), 0.5, min_val=0.0, max_val=30.0),
        on_progress=_on_progress,
    )

    return {"output_path": out}


@audio_bp.route("/audio/gen/sfx", methods=["POST"])
@require_csrf
@async_job("sfx", filepath_required=False)
def audio_gen_sfx(job_id, filepath, data):
    """Generate a synthesized sound effect."""
    from opencut.core.music_gen import generate_sfx

    preset = data.get("preset", "swoosh")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir or tempfile.gettempdir()
    out = generate_sfx(
        preset=preset,
        output_dir=effective_dir,
        duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=30.0),
        on_progress=_on_progress,
    )

    return {"output_path": out, "preset": preset}


@audio_bp.route("/audio/gen/silence", methods=["POST"])
@require_csrf
def audio_gen_silence():
    """Generate silence/padding audio."""
    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    duration = safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=3600.0)

    try:
        from opencut.core.music_gen import generate_silence
        out = generate_silence(duration=duration, output_dir=output_dir or tempfile.gettempdir())
        return jsonify({"output_path": out})
    except Exception as e:
        return safe_error(e, "audio_gen_silence")


# ---------------------------------------------------------------------------
# Audio Ducking
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/duck", methods=["POST"])
@require_csrf
@async_job("duck", filepath_param="music_path")
def audio_duck_route(job_id, filepath, data):
    """Duck music under voice using sidechain compression."""
    from opencut.core.audio_duck import sidechain_duck

    music_path = filepath  # validated by decorator via filepath_param
    voice_path = data.get("voice_path", "").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not voice_path:
        raise ValueError("Voice file not found")
    voice_path = validate_filepath(voice_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(music_path, output_dir)
    out = sidechain_duck(
        music_path, voice_path, output_dir=effective_dir,
        duck_amount=safe_float(data.get("duck_amount", 0.7), 0.7, min_val=0.0, max_val=1.0),
        threshold=safe_float(data.get("threshold", 0.015), 0.015, min_val=0.001, max_val=1.0),
        attack=safe_float(data.get("attack", 20), 20.0, min_val=1.0, max_val=1000.0),
        release=safe_float(data.get("release", 300), 300.0, min_val=10.0, max_val=5000.0),
        on_progress=_on_progress,
    )

    return {"output_path": out}


@audio_bp.route("/audio/mix-duck", methods=["POST"])
@require_csrf
@async_job("mix-duck", filepath_param="voice_path")
def audio_mix_duck_route(job_id, filepath, data):
    """Mix voice + music with auto-ducking."""
    from opencut.core.audio_duck import mix_with_duck

    voice_path = filepath  # validated by decorator via filepath_param
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not music_path:
        raise ValueError("Music file not found")
    music_path = validate_filepath(music_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(voice_path, output_dir)
    out = mix_with_duck(
        voice_path, music_path, output_dir=effective_dir,
        music_volume=safe_float(data.get("music_volume", 0.3), 0.3, min_val=0.0, max_val=2.0),
        duck_amount=safe_float(data.get("duck_amount", 0.6), 0.6, min_val=0.0, max_val=1.0),
        on_progress=_on_progress,
    )

    return {"output_path": out}


@audio_bp.route("/audio/duck-video", methods=["POST"])
@require_csrf
@async_job("duck-video")
def audio_duck_video_route(job_id, filepath, data):
    """Add music to video with auto-ducking under speech."""
    from opencut.core.audio_duck import auto_duck_video

    video_path = filepath  # validated by decorator
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not music_path:
        raise ValueError("Music file not found")
    music_path = validate_filepath(music_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(video_path, output_dir)
    out = auto_duck_video(
        video_path, music_path, output_dir=effective_dir,
        music_volume=safe_float(data.get("music_volume", 0.25), 0.25, min_val=0.0, max_val=2.0),
        duck_amount=safe_float(data.get("duck_amount", 0.7), 0.7, min_val=0.0, max_val=1.0),
        on_progress=_on_progress,
    )

    return {"output_path": out}


def _validate_mix_tracks(data):
    tracks = data.get("tracks")
    if not tracks:
        return "No tracks to mix"
    if not isinstance(tracks, list):
        return "tracks must be a list"
    if len(tracks) > 32:
        return "Too many tracks (max 32)"
    return None


@audio_bp.route("/audio/mix", methods=["POST"])
@require_csrf
@async_job("mix", filepath_required=False, pre_validate=_validate_mix_tracks)
def audio_mix_route(job_id, filepath, data):
    """Mix multiple audio tracks."""
    from opencut.core.audio_duck import mix_audio_tracks

    tracks = data.get("tracks", [])
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    if not tracks:
        raise ValueError("No tracks to mix")
    if len(tracks) > 32:
        raise ValueError("Too many tracks (max 32)")

    # Validate each track path
    for i, track in enumerate(tracks):
        track_path = track.get("path", "") if isinstance(track, dict) else str(track)
        if track_path:
            validate_filepath(track_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir or tempfile.gettempdir()
    _mix_mode = data.get("duration_mode", "longest")
    if _mix_mode not in ("longest", "shortest", "first"):
        _mix_mode = "longest"
    out = mix_audio_tracks(
        tracks, output_dir=effective_dir,
        duration_mode=_mix_mode,
        on_progress=_on_progress,
    )

    return {"output_path": out}


# ---------------------------------------------------------------------------
# AI Music Generation
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/music-ai/capabilities", methods=["GET"])
def music_ai_capabilities():
    try:
        from opencut.core.music_ai import get_music_ai_capabilities
        return jsonify(get_music_ai_capabilities())
    except Exception as e:
        return safe_error(e, "music_ai_capabilities")


@audio_bp.route("/audio/music-ai/generate", methods=["POST"])
@require_csrf
@async_job("musicgen", filepath_required=False)
def music_ai_generate(job_id, filepath, data):
    """Generate music from text prompt via MusicGen."""
    from opencut.core.music_ai import generate_music

    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("No prompt")

    acquired = rate_limit("ai_gpu")
    if not acquired:
        raise ValueError("Another AI GPU operation is already running. Please wait.")

    try:
        def _p(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        d = data.get("output_dir", "")
        if d:
            d = validate_path(d)
        else:
            d = tempfile.gettempdir()
        _mg_model = data.get("model", "small")
        if _mg_model not in ("small", "medium", "large"):
            _mg_model = "small"
        out = generate_music(prompt, output_dir=d,
                              duration=safe_float(data.get("duration", 10), 10.0, min_val=1.0, max_val=120.0),
                              model_size=_mg_model,
                              temperature=safe_float(data.get("temperature", 1.0), 1.0, min_val=0.1, max_val=2.0),
                              on_progress=_p)

        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("ai_gpu")


@audio_bp.route("/audio/music-ai/ace-step", methods=["POST"])
@require_csrf
@async_job("ace-step", filepath_required=False)
def music_ai_ace_step(job_id, filepath, data):
    """Generate music with vocals + lyrics using ACE-Step 1.5."""
    from opencut.core.music_ai import generate_music_ace_step

    prompt = data.get("prompt", "").strip()
    lyrics = data.get("lyrics", "").strip()
    if not prompt:
        raise ValueError("No prompt")
    if len(lyrics) > 10000:
        raise ValueError("Lyrics too long (max 10000 chars)")

    acquired = rate_limit("ai_gpu")
    if not acquired:
        raise ValueError("Another AI GPU operation is already running. Please wait.")

    try:
        def _p(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        d = data.get("output_dir", "")
        if d:
            d = validate_path(d)
        else:
            d = tempfile.gettempdir()
        out = generate_music_ace_step(
            prompt, lyrics=lyrics, output_dir=d,
            duration=safe_float(data.get("duration", 30), 30.0, min_val=10.0, max_val=600.0),
            on_progress=_p,
        )

        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("ai_gpu")


@audio_bp.route("/audio/music-ai/stable-audio", methods=["POST"])
@require_csrf
@async_job("stable-audio", filepath_required=False)
def music_ai_stable_audio(job_id, filepath, data):
    """Generate music using Stable Audio Open (Stability AI)."""
    from opencut.core.music_ai import generate_music_stable_audio

    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("No prompt")

    acquired = rate_limit("ai_gpu")
    if not acquired:
        raise ValueError("Another AI GPU operation is already running. Please wait.")

    try:
        def _p(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        d = data.get("output_dir", "")
        if d:
            d = validate_path(d)
        else:
            d = tempfile.gettempdir()
        result = generate_music_stable_audio(
            prompt, output_dir=d,
            duration=safe_float(data.get("duration", 30), 30.0, min_val=1.0, max_val=47.0),
            on_progress=_p,
        )

        return {"output_path": result["output"], "duration": result["duration"], "backend": result["backend"]}
    finally:
        if acquired:
            rate_limit_release("ai_gpu")


@audio_bp.route("/audio/music-ai/melody", methods=["POST"])
@require_csrf
@async_job("musicgen-melody", filepath_param="melody_path")
def music_ai_melody(job_id, filepath, data):
    """Generate music with melody conditioning."""
    from opencut.core.music_ai import generate_music_with_melody

    prompt = data.get("prompt", "").strip()
    melody = filepath  # validated by decorator via filepath_param
    if not prompt:
        raise ValueError("No prompt")

    acquired = rate_limit("ai_gpu")
    if not acquired:
        raise ValueError("Another AI GPU operation is already running. Please wait.")

    try:
        def _p(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        d = data.get("output_dir", "")
        if d:
            d = validate_path(d)
        else:
            d = tempfile.gettempdir()
        out = generate_music_with_melody(prompt, melody, output_dir=d,
                                          duration=safe_float(data.get("duration", 10), 10.0, min_val=1.0, max_val=120.0),
                                          on_progress=_p)

        return {"output_path": out}
    finally:
        if acquired:
            rate_limit_release("ai_gpu")


# ---------------------------------------------------------------------------
# Waveform Data (for silence threshold preview) - Job-based async
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/waveform", methods=["POST"])
@require_csrf
@async_job("waveform", filepath_param="file")
def audio_waveform(job_id, filepath, data):
    """Extract waveform amplitude data from audio/video file for visualization."""
    file_path = filepath  # validated by decorator via filepath_param="file"

    samples = safe_int(data.get("samples", 500), 500)
    samples = max(100, min(samples, 2000))

    _update_job(job_id, progress=10, message="Probing file duration...")

    # Use FFmpeg to extract peak amplitude per chunk
    cmd = [
        get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
        "-of", "json", file_path
    ]
    dur_result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
    if dur_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {dur_result.stderr[:200]}")
    try:
        dur_data = json.loads(dur_result.stdout)
        duration = float(dur_data["format"]["duration"])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        raise RuntimeError("Could not determine audio duration from ffprobe output")

    _update_job(job_id, progress=30, message="Extracting audio samples...")

    # Extract raw audio samples as 16-bit PCM
    cmd = [
        get_ffmpeg_path(), "-i", file_path, "-ac", "1", "-ar", "8000",
        "-f", "s16le", "-acodec", "pcm_s16le", "-v", "error", "-"
    ]
    result = _sp.run(cmd, capture_output=True, timeout=120)
    raw = result.stdout
    # Bulk unpack all samples at once using array module (much faster than per-sample struct)
    sample_count = len(raw) // 2
    if sample_count == 0:
        raise ValueError("No audio data found in file")

    _update_job(job_id, progress=70, message="Computing peaks...")

    all_samples = array.array('h')
    all_samples.frombytes(raw[:sample_count * 2])

    chunk_size = max(1, sample_count // samples)
    peaks = []
    for i in range(0, sample_count, chunk_size):
        chunk = all_samples[i:i + chunk_size]
        chunk_max = max(abs(s) for s in chunk)
        peaks.append(round(chunk_max / 32768.0, 4))

    return {
        "peaks": peaks[:samples],
        "duration": duration,
        "sample_count": len(peaks[:samples]),
    }


# ---------------------------------------------------------------------------
# Speed-Up-Silence Mode
# ---------------------------------------------------------------------------
@audio_bp.route("/silence/speed-up", methods=["POST"])
@require_csrf
@async_job("speed-silence")
def silence_speed_up(job_id, filepath, data):
    """Speed up silent segments instead of removing them."""
    from opencut.core.silence import speed_up_silences

    speed_factor = safe_float(data.get("speed_factor", 4.0), 4.0, min_val=1.5, max_val=8.0)
    threshold_db = safe_float(data.get("threshold_db", -30.0), -30.0, min_val=-60.0, max_val=-10.0)
    min_duration = safe_float(data.get("min_duration", 0.5), 0.5, min_val=0.1, max_val=5.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = speed_up_silences(
        filepath,
        speed_factor=speed_factor,
        threshold_db=threshold_db,
        min_duration=min_duration,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Audio Enhancement (Resemble Enhance)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/enhance", methods=["POST"])
@require_csrf
@async_job("audio-enhance")
def audio_enhance(job_id, filepath, data):
    """Enhance speech audio using ClearerVoice-Studio or Resemble Enhance."""
    backend = data.get("backend", "clearvoice")
    if backend not in ("clearvoice", "resemble"):
        backend = "clearvoice"

    if backend == "resemble":
        from opencut.checks import check_resemble_enhance_available
        if not check_resemble_enhance_available():
            raise ValueError("resemble-enhance not installed. Install with: pip install resemble-enhance")

    denoise = safe_bool(data.get("denoise", True), True)
    enhance = safe_bool(data.get("enhance", True), True)
    cv_model = data.get("model", "MossFormer2_SE_48K")
    _allowed_cv_models = {"MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"}
    if cv_model not in _allowed_cv_models:
        cv_model = "MossFormer2_SE_48K"
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    if backend == "resemble" and not denoise and not enhance:
        raise ValueError("At least one of 'denoise' or 'enhance' must be True")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if backend == "clearvoice":
        from opencut.core.audio_enhance import enhance_speech_clearvoice
        output_path = enhance_speech_clearvoice(
            filepath,
            output_dir=output_dir,
            model=cv_model,
            on_progress=_on_progress,
        )
    else:
        from opencut.core.audio_enhance import enhance_speech
        output_path = enhance_speech(
            filepath,
            output_dir=output_dir,
            denoise=denoise,
            enhance=enhance,
            on_progress=_on_progress,
        )

    return {"output_path": output_path}


# ---------------------------------------------------------------------------
# Audio: Beat Markers (for ExtendScript timeline marker insertion)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/beat-markers", methods=["POST"])
@require_csrf
@async_job("beat-markers")
def audio_beat_markers(job_id, filepath, data):
    """Extract beat timestamps for use as Premiere timeline markers."""
    from opencut.core.audio_suite import detect_beats

    subdivisions = safe_int(data.get("subdivisions", 1), 1, min_val=1, max_val=8)
    offset = safe_float(data.get("offset", 0.0), 0.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    info = detect_beats(filepath, on_progress=_on_progress)

    raw_beats = info.beat_times

    # Apply offset
    if offset != 0.0:
        raw_beats = [max(0.0, t + offset) for t in raw_beats]

    # Apply subdivisions: insert evenly-spaced sub-beats between each pair
    if subdivisions > 1 and len(raw_beats) >= 2:
        subdivided = []
        for i in range(len(raw_beats) - 1):
            subdivided.append(raw_beats[i])
            interval = (raw_beats[i + 1] - raw_beats[i]) / subdivisions
            for s in range(1, subdivisions):
                subdivided.append(round(raw_beats[i] + s * interval, 4))
        subdivided.append(raw_beats[-1])
        raw_beats = subdivided

    beats = [round(t, 4) for t in raw_beats]

    return {"beats": beats, "bpm": round(info.bpm, 2), "count": len(beats)}


# ---------------------------------------------------------------------------
# Audio: Loudness Match (batch)
# ---------------------------------------------------------------------------
def _validate_loudness_files(data):
    files = data.get("files")
    if not isinstance(files, list) or not files:
        return "files must be a non-empty list"
    if len(files) > 100:
        return "Too many files (max 100)"
    return None


@audio_bp.route("/audio/loudness-match", methods=["POST"])
@require_csrf
@async_job("loudness-match", filepath_required=False, pre_validate=_validate_loudness_files)
def audio_loudness_match(job_id, filepath, data):
    """Batch-normalize a list of audio/video files to a target LUFS level."""
    from opencut.core import loudness_match

    files = data.get("files", [])
    target_lufs = safe_float(data.get("target_lufs", -14.0), -14.0, min_val=-70.0, max_val=0.0)
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if not isinstance(files, list) or not files:
        raise ValueError("files must be a non-empty list")

    if len(files) > 100:
        raise ValueError("Too many files (max 100)")

    validated_files = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            raise ValueError("Each file entry must be a non-empty string")
        validated_files.append(validate_filepath(f.strip()))

    if output_dir:
        output_dir = validate_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = output_dir if output_dir else None
    result = loudness_match.batch_loudness_match(
        validated_files,
        output_dir=effective_dir,
        target_lufs=target_lufs,
        on_progress=_on_progress,
    )

    # batch_loudness_match returns a list, not a dict
    outputs = result if isinstance(result, list) else result.get("outputs", []) if isinstance(result, dict) else []

    return {"outputs": outputs}
