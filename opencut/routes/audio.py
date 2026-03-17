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
import tempfile
import threading

from flask import Blueprint, jsonify, request

from opencut.checks import check_demucs_available
from opencut.helpers import (
    _make_sequence_name,
    _resolve_output_dir,
)
from opencut.jobs import (
    _is_cancelled,
    _new_job,
    _register_job_process,
    _safe_error,
    _unregister_job_process,
    _update_job,
    job_lock,
    jobs,
)
from opencut.security import (
    require_csrf,
    require_rate_limit,
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
def silence_remove():
    """Remove silences and export Premiere XML."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    threshold = safe_float(data.get("threshold", -30.0), -30.0)
    min_duration = safe_float(data.get("min_duration", 0.5), 0.5)
    padding_before = safe_float(data.get("padding_before", 0.1), 0.1)
    padding_after = safe_float(data.get("padding_after", 0.1), 0.1)
    min_speech = safe_float(data.get("min_speech", 0.25), 0.25)
    preset = data.get("preset", None)
    seq_name = data.get("sequence_name", "")

    job_id = _new_job("silence", filepath)

    def _process():
        try:
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

            _update_job(job_id, progress=15, message="Detecting silences (this may take a moment for long files)...")
            segments = detect_speech(filepath, config=scfg, file_duration=_file_dur)

            if _is_cancelled(job_id):
                return

            _update_job(job_id, progress=70, message="Building edit summary...")
            summary = get_edit_summary(filepath, segments, file_duration=_file_dur)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")

            _update_job(job_id, progress=85, message="Exporting Premiere XML...")
            export_premiere_xml(filepath, segments, xml_path, config=ecfg)

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
                    "xml_path": xml_path,
                    "summary": summary,
                    "segments": len(segments),
                    "segments_data": [
                        {"start": round(s.start, 4), "end": round(s.end, 4)}
                        for s in segments
                    ],
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Silence removal error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Filler Word Removal
# ---------------------------------------------------------------------------
@audio_bp.route("/fillers", methods=["POST"])
@require_csrf
def filler_removal():
    """Detect and remove filler words (um, uh, like, etc.)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    model = data.get("model", "base")
    language = data.get("language", None)
    include_context = data.get("include_context_fillers", True)
    custom_words = data.get("custom_words", [])
    # Which filler types to actually remove (empty = all detected)
    remove_keys = data.get("remove_fillers", [])
    # Also remove silences?
    remove_silence = data.get("remove_silence", True)
    silence_preset = data.get("silence_preset", "youtube")
    seq_name = data.get("sequence_name", "")

    job_id = _new_job("fillers", filepath)

    def _process():
        try:
            # Check Whisper
            from opencut.core.captions import check_whisper_available, transcribe
            from opencut.core.fillers import detect_fillers, remove_fillers_from_segments

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="Whisper is required for filler word detection. Install it from the Captions tab.",
                    message="Whisper not installed",
                )
                return

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
                info = _finfo
                from opencut.core.silence import TimeSegment
                segments = [TimeSegment(start=0.0, end=info.duration, label="speech")]

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

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
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
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Filler removal error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Noise Reduction
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/denoise", methods=["POST"])
@require_csrf
def audio_denoise():
    """Remove background noise from audio/video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    method = data.get("method", "afftdn")
    strength = safe_float(data.get("strength", 0.7), 0.7, min_val=0.0, max_val=1.0)
    noise_floor = safe_float(data.get("noise_floor", -30.0), -30.0, min_val=-80.0, max_val=0.0)

    job_id = _new_job("denoise", filepath)

    def _process():
        try:
            from opencut.core.audio_suite import denoise_audio

            def _on_progress(pct, msg):
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

            _update_job(
                job_id, status="complete", progress=100,
                message="Noise reduction complete!",
                result={
                    "output_path": out_path,
                    "method": method,
                    "strength": strength,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Denoise error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Voice Isolation
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/isolate", methods=["POST"])
@require_csrf
def audio_isolate():
    """Isolate vocals by emphasizing voice frequencies."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("isolate", filepath)

    def _process():
        try:
            from opencut.core.audio_suite import isolate_voice

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_voice{ext}")

            isolate_voice(filepath, out_path, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message="Voice isolation complete!",
                result={"output_path": out_path},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Voice isolation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: AI Stem Separation (Demucs)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/separate", methods=["POST"])
@require_csrf
def audio_separate():
    """Separate audio into stems using Meta's Demucs AI model."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    model = data.get("model", "htdemucs")
    allowed_models = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"}
    if model not in allowed_models:
        return jsonify({"error": f"Unknown model: {model}. Allowed: {', '.join(sorted(allowed_models))}"}), 400
    stems = data.get("stems", ["vocals", "no_vocals"])
    output_format = data.get("format", "wav")
    auto_import = data.get("auto_import", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not check_demucs_available():
        return jsonify({"error": "Demucs not installed. Please install it first."}), 400

    job_id = _new_job("separate", filepath)

    def _process():
        try:
            import shutil

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]

            _update_job(job_id, progress=5, message="Extracting audio...")

            # Extract audio if video file
            video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
            file_ext = os.path.splitext(filepath)[1].lower()
            temp_audio = None

            if file_ext in video_exts:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio = tmp.name

                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', filepath,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                    temp_audio
                ]
                _sp.run(ffmpeg_cmd, check=True, capture_output=True, timeout=300)
                input_audio = temp_audio
            else:
                input_audio = filepath

            _update_job(job_id, progress=15, message=f"Running Demucs ({model})...")

            # Create temp output directory for demucs
            with tempfile.TemporaryDirectory() as temp_out:
                # Run demucs
                demucs_cmd = [
                    'python', '-m', 'demucs',
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
                                'ffmpeg', '-y', '-i', src_path,
                                '-codec:a', 'libmp3lame', '-b:a', '320k',
                                out_path
                            ], check=True, capture_output=True, timeout=300)
                        elif output_format == 'flac':
                            _sp.run([
                                'ffmpeg', '-y', '-i', src_path,
                                '-codec:a', 'flac',
                                out_path
                            ], check=True, capture_output=True, timeout=300)

                        output_paths.append(out_path)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Separation complete! {len(output_paths)} stems exported.",
                result={
                    "output_paths": output_paths,
                    "auto_import": auto_import
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio separation error")
        finally:
            _unregister_job_process(job_id)
            # Cleanup temp audio extracted from video
            try:
                if file_ext in video_exts and temp_audio and os.path.exists(temp_audio):
                    os.unlink(temp_audio)
            except OSError:
                pass

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Normalization
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/normalize", methods=["POST"])
@require_csrf
def audio_normalize():
    """Normalize audio loudness to broadcast/platform standards."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    preset = data.get("preset", "youtube")
    target_lufs = data.get("target_lufs", None)
    if target_lufs is not None:
        target_lufs = safe_float(target_lufs, -16.0, min_val=-70.0, max_val=0.0)

    job_id = _new_job("normalize", filepath)

    def _process():
        try:
            from opencut.core.audio_suite import LOUDNESS_PRESETS, normalize_loudness

            def _on_progress(pct, msg):
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

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Normalized to {targets['i']:.1f} LUFS",
                result={
                    "output_path": out_path,
                    "preset": preset,
                    "input_loudness": info.input_i,
                    "target_loudness": targets["i"],
                    "input_peak": info.input_tp,
                    "loudness_range": info.input_lra,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Normalize error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(e)


# ---------------------------------------------------------------------------
# Audio Suite: Beat Detection
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/beats", methods=["POST"])
@require_csrf
def audio_beats():
    """Detect beats and estimate BPM."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5, min_val=0.0, max_val=1.0)

    job_id = _new_job("beats", filepath)

    def _process():
        try:
            from opencut.core.audio_suite import detect_beats

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            info = detect_beats(filepath, sensitivity=sensitivity, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Found {len(info.beat_times)} beats at {info.bpm:.0f} BPM",
                result={
                    "bpm": info.bpm,
                    "total_beats": len(info.beat_times),
                    "beat_times": [round(t, 3) for t in info.beat_times[:500]],
                    "downbeat_times": [round(t, 3) for t in info.downbeat_times[:125]],
                    "confidence": round(info.confidence, 3),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Beat detection error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(e)


@audio_bp.route("/audio/effects/apply", methods=["POST"])
@require_csrf
def audio_effects_apply():
    """Apply an audio effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("audio-effect", filepath)

    def _process():
        try:
            from opencut.core.audio_suite import apply_audio_effect

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_{effect}{ext}")

            apply_audio_effect(filepath, effect, out_path, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Effect '{effect}' applied!",
                result={"output_path": out_path, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio effect error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(e)


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
        return _safe_error(e)


@audio_bp.route("/audio/pro/apply", methods=["POST"])
@require_csrf
def audio_pro_apply():
    """Apply a Pedalboard studio effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "").strip()
    params = data.get("params", {})

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("audio-pro", filepath)

    def _process():
        try:
            from opencut.core.audio_pro import apply_pedalboard_effect

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = apply_pedalboard_effect(
                filepath, effect, output_dir=effective_dir,
                params=params, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Effect applied!",
                result={"output_path": out, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio pro error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/pro/deepfilter", methods=["POST"])
@require_csrf
def audio_pro_deepfilter():
    """AI noise reduction using DeepFilterNet."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("deepfilter", filepath)

    def _process():
        try:
            from opencut.core.audio_pro import deepfilter_denoise

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = deepfilter_denoise(
                filepath, output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="AI noise reduction complete!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("DeepFilterNet error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/pro/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def audio_pro_install():
    """Install audio pro dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "pedalboard")

    packages = {
        "pedalboard": ["pedalboard"],
        "deepfilter": ["deepfilternet"],
    }

    pkgs = packages.get(component, [])
    if not pkgs:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    job_id = _new_job("install", component)

    def _process():
        try:
            for i, pkg in enumerate(pkgs):
                pct = int((i / len(pkgs)) * 90)
                _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
                safe_pip_install(pkg, timeout=600)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio pro install error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(e)


@audio_bp.route("/audio/tts/generate", methods=["POST"])
@require_csrf
def tts_generate():
    """Generate speech from text."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    engine = data.get("engine", "edge")
    voice = data.get("voice", "en-US-AriaNeural")
    rate = data.get("rate", "+0%")
    pitch = data.get("pitch", "+0Hz")
    speed = safe_float(data.get("speed", 1.0), 1.0, min_val=0.25, max_val=4.0)
    output_dir = data.get("output_dir", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    job_id = _new_job("tts", text[:50])

    def _process():
        try:
            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()

            if engine == "kokoro":
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

            _update_job(
                job_id, status="complete", progress=100,
                message="Speech generated!",
                result={"output_path": out, "engine": engine, "voice": voice},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("TTS error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/tts/subtitled", methods=["POST"])
@require_csrf
def tts_subtitled():
    """Generate speech + word-level subtitles."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    voice = data.get("voice", "en-US-AriaNeural")
    output_dir = data.get("output_dir", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    job_id = _new_job("tts-sub", text[:50])

    def _process():
        try:
            from opencut.core.voice_gen import edge_tts_with_subtitles

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            result = edge_tts_with_subtitles(
                text, voice=voice, output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Speech + subtitles generated!",
                result=result,
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("TTS subtitled error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/tts/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def tts_install():
    """Install TTS engine dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "edge_tts")

    packages = {
        "edge_tts": ["edge-tts"],
        "kokoro": ["kokoro>=0.3", "soundfile"],
    }
    pkgs = packages.get(component, [])
    if not pkgs:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    job_id = _new_job("install", component)

    def _process():
        try:
            for pkg in pkgs:
                _update_job(job_id, progress=30, message=f"Installing {pkg}...")
                safe_pip_install(pkg, timeout=600)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
        return _safe_error(e)


@audio_bp.route("/audio/gen/tone", methods=["POST"])
@require_csrf
def audio_gen_tone():
    """Generate a tone."""
    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "")

    job_id = _new_job("tone", f"{data.get('frequency', 440)}Hz")

    def _process():
        try:
            from opencut.core.music_gen import generate_tone

            def _on_progress(pct, msg):
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
            _update_job(
                job_id, status="complete", progress=100,
                message="Tone generated!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/gen/sfx", methods=["POST"])
@require_csrf
def audio_gen_sfx():
    """Generate a synthesized sound effect."""
    data = request.get_json(force=True)
    preset = data.get("preset", "swoosh")
    output_dir = data.get("output_dir", "")

    job_id = _new_job("sfx", preset)

    def _process():
        try:
            from opencut.core.music_gen import generate_sfx

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            out = generate_sfx(
                preset=preset,
                output_dir=effective_dir,
                duration=safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=30.0),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"SFX '{preset}' generated!",
                result={"output_path": out, "preset": preset},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/gen/silence", methods=["POST"])
@require_csrf
def audio_gen_silence():
    """Generate silence/padding audio."""
    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "")
    duration = safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=3600.0)

    try:
        from opencut.core.music_gen import generate_silence
        out = generate_silence(duration=duration, output_dir=output_dir or tempfile.gettempdir())
        return jsonify({"output_path": out})
    except Exception as e:
        return _safe_error(e)


# ---------------------------------------------------------------------------
# Audio Ducking
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/duck", methods=["POST"])
@require_csrf
def audio_duck_route():
    """Duck music under voice using sidechain compression."""
    data = request.get_json(force=True)
    music_path = data.get("music_path", "").strip()
    voice_path = data.get("voice_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not music_path:
        return jsonify({"error": "Music file not found"}), 400
    if not voice_path:
        return jsonify({"error": "Voice file not found"}), 400

    try:
        music_path = validate_filepath(music_path)
    except ValueError as e:
        return jsonify({"error": f"Music file: {e}"}), 400

    try:
        voice_path = validate_filepath(voice_path)
    except ValueError as e:
        return jsonify({"error": f"Voice file: {e}"}), 400

    job_id = _new_job("duck", music_path)

    def _process():
        try:
            from opencut.core.audio_duck import sidechain_duck

            def _on_progress(pct, msg):
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
            _update_job(
                job_id, status="complete", progress=100,
                message="Audio ducking applied!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/mix-duck", methods=["POST"])
@require_csrf
def audio_mix_duck_route():
    """Mix voice + music with auto-ducking."""
    data = request.get_json(force=True)
    voice_path = data.get("voice_path", "").strip()
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not voice_path:
        return jsonify({"error": "Voice file not found"}), 400
    if not music_path:
        return jsonify({"error": "Music file not found"}), 400

    try:
        voice_path = validate_filepath(voice_path)
    except ValueError as e:
        return jsonify({"error": f"Voice file: {e}"}), 400

    try:
        music_path = validate_filepath(music_path)
    except ValueError as e:
        return jsonify({"error": f"Music file: {e}"}), 400

    job_id = _new_job("mix-duck", voice_path)

    def _process():
        try:
            from opencut.core.audio_duck import mix_with_duck

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(voice_path, output_dir)
            out = mix_with_duck(
                voice_path, music_path, output_dir=effective_dir,
                music_volume=safe_float(data.get("music_volume", 0.3), 0.3, min_val=0.0, max_val=2.0),
                duck_amount=safe_float(data.get("duck_amount", 0.6), 0.6, min_val=0.0, max_val=1.0),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Audio mixed with ducking!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/duck-video", methods=["POST"])
@require_csrf
def audio_duck_video_route():
    """Add music to video with auto-ducking under speech."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not video_path:
        return jsonify({"error": "Video file not found"}), 400
    if not music_path:
        return jsonify({"error": "Music file not found"}), 400

    try:
        video_path = validate_filepath(video_path)
    except ValueError as e:
        return jsonify({"error": f"Video file: {e}"}), 400

    try:
        music_path = validate_filepath(music_path)
    except ValueError as e:
        return jsonify({"error": f"Music file: {e}"}), 400

    job_id = _new_job("duck-video", video_path)

    def _process():
        try:
            from opencut.core.audio_duck import auto_duck_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(video_path, output_dir)
            out = auto_duck_video(
                video_path, music_path, output_dir=effective_dir,
                music_volume=safe_float(data.get("music_volume", 0.25), 0.25, min_val=0.0, max_val=2.0),
                duck_amount=safe_float(data.get("duck_amount", 0.7), 0.7, min_val=0.0, max_val=1.0),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Music added with ducking!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Duck video error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/mix", methods=["POST"])
@require_csrf
def audio_mix_route():
    """Mix multiple audio tracks."""
    data = request.get_json(force=True)
    tracks = data.get("tracks", [])
    output_dir = data.get("output_dir", "")

    if not tracks:
        return jsonify({"error": "No tracks to mix"}), 400

    # Validate each track path
    for i, track in enumerate(tracks):
        track_path = track.get("path", "") if isinstance(track, dict) else str(track)
        if track_path:
            try:
                validate_filepath(track_path)
            except ValueError as e:
                return jsonify({"error": f"Track {i + 1}: {e}"}), 400

    job_id = _new_job("mix", f"{len(tracks)} tracks")

    def _process():
        try:
            from opencut.core.audio_duck import mix_audio_tracks

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            out = mix_audio_tracks(
                tracks, output_dir=effective_dir,
                duration_mode=data.get("duration_mode", "longest"),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Mixed {len(tracks)} tracks!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI Music Generation
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/music-ai/capabilities", methods=["GET"])
def music_ai_capabilities():
    try:
        from opencut.core.music_ai import get_music_ai_capabilities
        return jsonify(get_music_ai_capabilities())
    except Exception as e:
        return _safe_error(e)


@audio_bp.route("/audio/music-ai/generate", methods=["POST"])
@require_csrf
def music_ai_generate():
    """Generate music from text prompt via MusicGen."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    job_id = _new_job("musicgen", prompt[:40])

    def _process():
        try:
            from opencut.core.music_ai import generate_music

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            d = data.get("output_dir", "")
            if d:
                valid, msg = validate_path(d)
                if not valid:
                    _update_job(job_id, status="error", message=msg)
                    return
            else:
                d = tempfile.gettempdir()
            out = generate_music(prompt, output_dir=d,
                                  duration=safe_float(data.get("duration", 10), 10.0, min_val=1.0, max_val=120.0),
                                  model_size=data.get("model", "small"),
                                  temperature=safe_float(data.get("temperature", 1.0), 1.0, min_val=0.1, max_val=2.0),
                                  on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@audio_bp.route("/audio/music-ai/melody", methods=["POST"])
@require_csrf
def music_ai_melody():
    """Generate music with melody conditioning."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    melody = data.get("melody_path", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400
    if not melody:
        return jsonify({"error": "Melody not found"}), 400

    try:
        melody = validate_filepath(melody)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    job_id = _new_job("musicgen-melody", prompt[:40])

    def _process():
        try:
            from opencut.core.music_ai import generate_music_with_melody

            def _p(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            d = data.get("output_dir", "")
            if d:
                valid, msg = validate_path(d)
                if not valid:
                    _update_job(job_id, status="error", message=msg)
                    return
            else:
                d = tempfile.gettempdir()
            out = generate_music_with_melody(prompt, melody, output_dir=d,
                                              duration=safe_float(data.get("duration", 10), 10.0, min_val=1.0, max_val=120.0),
                                              on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Waveform Data (for silence threshold preview) - Job-based async
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/waveform", methods=["POST"])
@require_csrf
def audio_waveform():
    """Extract waveform amplitude data from audio/video file for visualization."""
    data = request.get_json(force=True)
    file_path = data.get("file", "")

    if not file_path:
        return jsonify({"error": "File not found"}), 400

    try:
        file_path = validate_filepath(file_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    samples = safe_int(data.get("samples", 500), 500)
    samples = max(100, min(samples, 2000))

    job_id = _new_job("waveform", file_path)

    def _process():
        try:
            _update_job(job_id, progress=10, message="Probing file duration...")

            # Use FFmpeg to extract peak amplitude per chunk
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", file_path
            ]
            dur_result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
            dur_data = json.loads(dur_result.stdout)
            duration = float(dur_data["format"]["duration"])

            _update_job(job_id, progress=30, message="Extracting audio samples...")

            # Extract raw audio samples as 16-bit PCM
            cmd = [
                "ffmpeg", "-i", file_path, "-ac", "1", "-ar", "8000",
                "-f", "s16le", "-acodec", "pcm_s16le", "-v", "error", "-"
            ]
            result = _sp.run(cmd, capture_output=True, timeout=120)
            raw = result.stdout
            # Bulk unpack all samples at once using array module (much faster than per-sample struct)
            sample_count = len(raw) // 2
            if sample_count == 0:
                _update_job(job_id, status="error", error="No audio data", message="No audio data found in file")
                return

            _update_job(job_id, progress=70, message="Computing peaks...")

            all_samples = array.array('h')
            all_samples.frombytes(raw[:sample_count * 2])

            chunk_size = max(1, sample_count // samples)
            peaks = []
            for i in range(0, sample_count, chunk_size):
                chunk = all_samples[i:i + chunk_size]
                chunk_max = max(abs(s) for s in chunk)
                peaks.append(round(chunk_max / 32768.0, 4))

            _update_job(
                job_id, status="complete", progress=100,
                message="Waveform extracted!",
                result={
                    "peaks": peaks[:samples],
                    "duration": duration,
                    "sample_count": len(peaks[:samples]),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Waveform extraction error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Speed-Up-Silence Mode
# ---------------------------------------------------------------------------
@audio_bp.route("/silence/speed-up", methods=["POST"])
@require_csrf
def silence_speed_up():
    """Speed up silent segments instead of removing them."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    speed_factor = safe_float(data.get("speed_factor", 4.0), 4.0, min_val=1.5, max_val=8.0)
    threshold_db = safe_float(data.get("threshold_db", -30.0), -30.0, min_val=-60.0, max_val=-10.0)
    min_duration = safe_float(data.get("min_duration", 0.5), 0.5, min_val=0.1, max_val=5.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    job_id = _new_job("speed-silence", filepath)

    def _process():
        try:
            from opencut.core.silence import speed_up_silences

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            result = speed_up_silences(
                filepath,
                speed_factor=speed_factor,
                threshold_db=threshold_db,
                min_duration=min_duration,
                output_dir=output_dir,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Done: {result['reduction_percent']:.0f}% shorter",
                result=result,
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Speed-up-silence error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Enhancement (Resemble Enhance)
# ---------------------------------------------------------------------------
@audio_bp.route("/audio/enhance", methods=["POST"])
@require_csrf
def audio_enhance():
    """Enhance speech audio using Resemble Enhance (super-resolution + denoising)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    from opencut.checks import check_resemble_enhance_available
    if not check_resemble_enhance_available():
        return jsonify({"error": "resemble-enhance not installed. Install with: pip install resemble-enhance"}), 400

    denoise = bool(data.get("denoise", True))
    enhance = bool(data.get("enhance", True))
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    if not denoise and not enhance:
        return jsonify({"error": "At least one of 'denoise' or 'enhance' must be True"}), 400

    job_id = _new_job("audio-enhance", filepath)

    def _process():
        try:
            from opencut.core.audio_enhance import enhance_speech

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            output_path = enhance_speech(
                filepath,
                output_dir=output_dir,
                denoise=denoise,
                enhance=enhance,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message="Audio enhanced!",
                result={"output_path": output_path},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio enhance error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})
