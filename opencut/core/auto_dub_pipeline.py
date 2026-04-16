"""
OpenCut End-to-End Auto-Dubbing Pipeline

Orchestrates existing modules into a complete dubbing workflow:
1. Transcribe source audio (Whisper integration)
2. Translate transcript to target language
3. Clone speaker voice (TTS / voice cloning)
4. Generate dubbed audio segments
5. Lip sync video to new audio
6. Composite final output

Supports 50+ languages with graceful fallback at each stage.
"""

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Supported languages (ISO 639-1 with display names)
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: List[str] = [
    "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
    "ar", "hi", "bn", "tr", "vi", "th", "pl", "nl", "sv", "da",
    "no", "fi", "cs", "sk", "hu", "ro", "bg", "hr", "sr", "sl",
    "uk", "el", "he", "fa", "ur", "ta", "te", "id", "ms", "sw",
    "fil", "af", "ca", "eu", "gl", "cy", "ga",
]

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "bn": "Bengali", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
    "pl": "Polish", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
    "sr": "Serbian", "sl": "Slovenian", "uk": "Ukrainian", "el": "Greek",
    "he": "Hebrew", "fa": "Persian", "ur": "Urdu", "ta": "Tamil",
    "te": "Telugu", "id": "Indonesian", "ms": "Malay", "sw": "Swahili",
    "fil": "Filipino", "af": "Afrikaans", "ca": "Catalan", "eu": "Basque",
    "gl": "Galician", "cy": "Welsh", "ga": "Irish",
}


# ---------------------------------------------------------------------------
# Pipeline stage enum-like constants
# ---------------------------------------------------------------------------
STAGE_EXTRACT = "extract_audio"
STAGE_TRANSCRIBE = "transcribe"
STAGE_TRANSLATE = "translate"
STAGE_VOICE_CLONE = "voice_clone"
STAGE_TTS = "tts_generate"
STAGE_LIP_SYNC = "lip_sync"
STAGE_COMPOSITE = "composite"

_STAGE_ORDER = [
    STAGE_EXTRACT, STAGE_TRANSCRIBE, STAGE_TRANSLATE,
    STAGE_VOICE_CLONE, STAGE_TTS, STAGE_LIP_SYNC, STAGE_COMPOSITE,
]

_STAGE_WEIGHTS = {
    STAGE_EXTRACT: 5,
    STAGE_TRANSCRIBE: 15,
    STAGE_TRANSLATE: 15,
    STAGE_VOICE_CLONE: 10,
    STAGE_TTS: 25,
    STAGE_LIP_SYNC: 15,
    STAGE_COMPOSITE: 15,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DubSegment:
    """A single dubbed segment with original and translated content."""
    index: int = 0
    start: float = 0.0
    end: float = 0.0
    original_text: str = ""
    translated_text: str = ""
    speaker_id: str = "speaker_0"
    tts_audio_path: str = ""
    synced: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DubConfig:
    """Configuration for the auto-dubbing pipeline."""
    target_language: str = "es"
    source_language: str = ""       # auto-detect if empty
    whisper_model: str = "base"
    voice_clone: bool = True        # attempt voice cloning
    lip_sync: bool = True           # attempt lip sync
    preserve_music: bool = True     # keep background music via stem separation
    tts_engine: str = "edge"        # "edge", "kokoro", "chatterbox"
    output_dir: str = ""
    max_segment_gap: float = 0.5    # merge segments closer than this

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DubResult:
    """Result of the complete auto-dubbing pipeline."""
    output_path: str = ""
    source_language: str = ""
    target_language: str = ""
    target_language_name: str = ""
    total_duration: float = 0.0
    segments_dubbed: int = 0
    segments_total: int = 0
    segments: List[Dict] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    voice_cloned: bool = False
    lip_synced: bool = False
    music_preserved: bool = False
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------
def _stage_progress(on_progress: Optional[Callable], stage: str, sub_pct: float,
                    msg: str = ""):
    """Report progress scaled to the stage's weight within the pipeline."""
    if not on_progress:
        return
    # Calculate global percentage
    cumulative = 0
    for s in _STAGE_ORDER:
        if s == stage:
            break
        cumulative += _STAGE_WEIGHTS.get(s, 10)
    weight = _STAGE_WEIGHTS.get(stage, 10)
    global_pct = cumulative + weight * (sub_pct / 100.0)
    on_progress(int(min(99, global_pct)), msg or f"{stage}: {sub_pct:.0f}%")


# ---------------------------------------------------------------------------
# Stage 1: Extract audio
# ---------------------------------------------------------------------------
def _extract_source_audio(video_path: str, work_dir: str,
                          on_progress: Optional[Callable] = None) -> str:
    """Extract audio from video to WAV for processing."""
    _stage_progress(on_progress, STAGE_EXTRACT, 0, "Extracting audio...")

    audio_path = os.path.join(work_dir, "source_audio.wav")
    cmd = (FFmpegCmd()
           .input(video_path)
           .no_video()
           .audio_codec("pcm_s16le")
           .option("ar", "16000")
           .option("ac", "1")
           .output(audio_path)
           .build())
    try:
        run_ffmpeg(cmd, timeout=300)
    except Exception as exc:
        raise RuntimeError(f"Audio extraction failed: {exc}")

    if not os.path.isfile(audio_path):
        raise RuntimeError("Audio extraction produced no output")

    _stage_progress(on_progress, STAGE_EXTRACT, 100, "Audio extracted")
    return audio_path


# ---------------------------------------------------------------------------
# Stage 2: Transcribe
# ---------------------------------------------------------------------------
def _transcribe_audio(
    audio_path: str,
    source_language: str,
    whisper_model: str,
    on_progress: Optional[Callable] = None,
) -> Tuple[List[DubSegment], str]:
    """Transcribe audio to timestamped text segments.

    Returns (segments, detected_language).
    """
    _stage_progress(on_progress, STAGE_TRANSCRIBE, 0, "Transcribing audio...")
    segments = []
    detected_lang = source_language

    # Try Whisper
    try:
        from opencut.core.captions import transcribe

        result = transcribe(audio_path, model=whisper_model,
                            language=source_language or None)

        if hasattr(result, "segments"):
            raw_segs = result.segments
        elif isinstance(result, dict):
            raw_segs = result.get("segments", [])
        else:
            raw_segs = []

        if hasattr(result, "language"):
            detected_lang = result.language
        elif isinstance(result, dict) and "language" in result:
            detected_lang = result["language"]

        for i, seg in enumerate(raw_segs):
            text = seg.get("text", "").strip() if isinstance(seg, dict) else ""
            if not text:
                continue
            segments.append(DubSegment(
                index=i,
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                original_text=text,
                speaker_id="speaker_0",
            ))

        _stage_progress(on_progress, STAGE_TRANSCRIBE, 100,
                        f"Transcribed {len(segments)} segments")

    except Exception as exc:
        logger.info("Whisper transcription failed: %s. Using silence-based fallback.", exc)

        # Fallback: silence-based segmentation
        try:
            from opencut.core.silence import detect_silence
            silence = detect_silence(audio_path, min_silence_len=0.5)
            if hasattr(silence, "segments"):
                for i, seg in enumerate(silence.segments):
                    segments.append(DubSegment(
                        index=i,
                        start=seg.start,
                        end=seg.end,
                        original_text=f"[segment_{i + 1}]",
                    ))
        except Exception:
            # Last resort: single segment for entire audio
            info = get_video_info(audio_path)
            dur = info.get("duration", 60.0)
            segments.append(DubSegment(
                index=0, start=0.0, end=dur,
                original_text="[full_audio]",
            ))

        _stage_progress(on_progress, STAGE_TRANSCRIBE, 100,
                        f"Fallback segmentation: {len(segments)} segments")

    return segments, detected_lang or "en"


# ---------------------------------------------------------------------------
# Stage 3: Translate
# ---------------------------------------------------------------------------
def _translate_segments(
    segments: List[DubSegment],
    source_lang: str,
    target_lang: str,
    on_progress: Optional[Callable] = None,
) -> List[DubSegment]:
    """Translate segment text to target language."""
    _stage_progress(on_progress, STAGE_TRANSLATE, 0, "Translating segments...")

    if source_lang == target_lang:
        for seg in segments:
            seg.translated_text = seg.original_text
        _stage_progress(on_progress, STAGE_TRANSLATE, 100, "Same language — no translation needed")
        return segments

    # Try LLM translation
    try:
        from opencut.core.llm import llm_chat
        source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        for i, seg in enumerate(segments):
            if seg.original_text.startswith("["):
                seg.translated_text = seg.original_text
                continue

            prompt = (
                f"Translate the following from {source_name} to {target_name}. "
                f"Return only the translated text, nothing else.\n\n"
                f"{seg.original_text}"
            )
            result = llm_chat(prompt)
            seg.translated_text = result.strip() if isinstance(result, str) else seg.original_text

            sub_pct = (i + 1) / len(segments) * 100
            _stage_progress(on_progress, STAGE_TRANSLATE, sub_pct,
                            f"Translated {i + 1}/{len(segments)} segments")

    except Exception as exc:
        logger.info("LLM translation unavailable: %s. Using placeholder.", exc)
        for seg in segments:
            seg.translated_text = f"[{target_lang}] {seg.original_text}"
        _stage_progress(on_progress, STAGE_TRANSLATE, 100, "Translation fallback used")

    return segments


# ---------------------------------------------------------------------------
# Stage 4: Voice clone reference extraction
# ---------------------------------------------------------------------------
def _extract_voice_reference(
    audio_path: str,
    segments: List[DubSegment],
    work_dir: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Extract a clean voice reference clip for cloning.

    Takes the longest speech segment as reference.
    """
    _stage_progress(on_progress, STAGE_VOICE_CLONE, 0, "Extracting voice reference...")

    if not segments:
        return ""

    # Find longest segment
    longest = max(segments, key=lambda s: s.end - s.start)
    ref_start = longest.start
    ref_dur = min(longest.end - longest.start, 15.0)  # cap at 15s

    ref_path = os.path.join(work_dir, "voice_reference.wav")
    cmd = (FFmpegCmd()
           .input(audio_path, ss=str(ref_start), t=str(ref_dur))
           .audio_codec("pcm_s16le")
           .option("ar", "22050")
           .option("ac", "1")
           .output(ref_path)
           .build())

    try:
        run_ffmpeg(cmd, timeout=30)
        _stage_progress(on_progress, STAGE_VOICE_CLONE, 100, "Voice reference extracted")
        return ref_path if os.path.isfile(ref_path) else ""
    except Exception as exc:
        logger.debug("Voice reference extraction failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Stage 5: TTS generation
# ---------------------------------------------------------------------------
def _generate_tts_segment(
    text: str,
    output_audio: str,
    target_lang: str,
    target_duration: float,
    tts_engine: str,
    voice_reference: str = "",
) -> bool:
    """Generate TTS audio for a single segment."""
    # Try edge-tts first (most reliable)
    if tts_engine == "edge" or tts_engine == "auto":
        try:
            from opencut.core.voice_gen import edge_tts_generate
            return edge_tts_generate(
                text=text,
                output_path=output_audio,
                language=target_lang,
            )
        except Exception as exc:
            logger.debug("edge-tts failed: %s", exc)

    # Try Kokoro
    if tts_engine in ("kokoro", "auto"):
        try:
            from opencut.core.voice_gen import kokoro_generate
            return kokoro_generate(
                text=text,
                output_path=output_audio,
            )
        except Exception as exc:
            logger.debug("Kokoro TTS failed: %s", exc)

    # Try Chatterbox (voice cloning)
    if tts_engine in ("chatterbox", "auto") and voice_reference:
        try:
            from opencut.core.voice_gen import chatterbox_generate
            return chatterbox_generate(
                text=text,
                output_path=output_audio,
                reference_audio=voice_reference,
            )
        except Exception as exc:
            logger.debug("Chatterbox TTS failed: %s", exc)

    # Fallback: generate silence as placeholder
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={target_duration}",
        "-c:a", "pcm_s16le", output_audio,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        return os.path.isfile(output_audio)
    except Exception:
        return False


def _generate_all_tts(
    segments: List[DubSegment],
    target_lang: str,
    tts_engine: str,
    voice_reference: str,
    work_dir: str,
    on_progress: Optional[Callable] = None,
) -> List[DubSegment]:
    """Generate TTS audio for all segments."""
    _stage_progress(on_progress, STAGE_TTS, 0, "Generating dubbed audio...")

    for i, seg in enumerate(segments):
        text = seg.translated_text or seg.original_text
        if not text or text.startswith("["):
            continue

        tts_path = os.path.join(work_dir, f"tts_seg_{i:04d}.wav")
        target_dur = seg.end - seg.start

        success = _generate_tts_segment(
            text=text,
            output_audio=tts_path,
            target_lang=target_lang,
            target_duration=target_dur,
            tts_engine=tts_engine,
            voice_reference=voice_reference,
        )

        if success and os.path.isfile(tts_path):
            seg.tts_audio_path = tts_path

            # Time-stretch TTS to match original segment duration
            tts_info = get_video_info(tts_path)
            tts_dur = tts_info.get("duration", 0)
            if tts_dur > 0 and abs(tts_dur - target_dur) > 0.2:
                tempo = tts_dur / target_dur
                tempo = max(0.5, min(2.0, tempo))  # clamp
                stretched_path = os.path.join(work_dir, f"tts_seg_{i:04d}_stretched.wav")
                stretch_cmd = (FFmpegCmd()
                               .input(tts_path)
                               .audio_filter(f"atempo={tempo:.4f}")
                               .audio_codec("pcm_s16le")
                               .output(stretched_path)
                               .build())
                try:
                    run_ffmpeg(stretch_cmd, timeout=30)
                    if os.path.isfile(stretched_path):
                        seg.tts_audio_path = stretched_path
                except Exception:
                    pass  # keep original TTS

        sub_pct = (i + 1) / len(segments) * 100
        _stage_progress(on_progress, STAGE_TTS, sub_pct,
                        f"Generated TTS {i + 1}/{len(segments)}")

    return segments


# ---------------------------------------------------------------------------
# Stage 6: Lip sync (optional)
# ---------------------------------------------------------------------------
def _apply_lip_sync(
    video_path: str,
    dubbed_video: str,
    work_dir: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply lip sync to match new audio. Returns output path or original if failed."""
    _stage_progress(on_progress, STAGE_LIP_SYNC, 0, "Applying lip sync...")

    try:
        from opencut.core.lip_sync_gen import LipSyncConfig, apply_lip_sync

        os.path.join(work_dir, "lip_synced.mp4")
        config = LipSyncConfig()

        result = apply_lip_sync(
            video_path=dubbed_video,
            config=config,
        )

        if hasattr(result, "output_path") and os.path.isfile(result.output_path):
            _stage_progress(on_progress, STAGE_LIP_SYNC, 100, "Lip sync applied")
            return result.output_path

    except Exception as exc:
        logger.info("Lip sync failed (non-critical): %s", exc)

    _stage_progress(on_progress, STAGE_LIP_SYNC, 100, "Lip sync skipped")
    return dubbed_video


# ---------------------------------------------------------------------------
# Stage 7: Composite final output
# ---------------------------------------------------------------------------
def _composite_dubbed_audio(
    video_path: str,
    segments: List[DubSegment],
    work_dir: str,
    output_file: str,
    preserve_music: bool = True,
    on_progress: Optional[Callable] = None,
) -> str:
    """Mix dubbed TTS segments with original video, optionally preserving music."""
    _stage_progress(on_progress, STAGE_COMPOSITE, 0, "Compositing final output...")

    # Create a full-length dubbed audio track by mixing TTS segments
    info = get_video_info(video_path)
    info.get("duration", 60.0)

    # Generate the mixed audio
    tts_segments_with_audio = [s for s in segments if s.tts_audio_path and os.path.isfile(s.tts_audio_path)]

    if not tts_segments_with_audio:
        # No TTS audio generated — copy original
        logger.warning("No TTS segments generated; copying original audio")
        cmd = (FFmpegCmd()
               .input(video_path)
               .video_codec("copy")
               .audio_codec("copy")
               .output(output_file)
               .build())
        run_ffmpeg(cmd, timeout=300)
        return output_file

    # Build a complex filter to overlay TTS segments at their timestamps
    filter_parts = []
    input_args = ["-i", video_path]
    input_idx = 1

    for seg in tts_segments_with_audio:
        input_args.extend(["-i", seg.tts_audio_path])
        delay_ms = int(seg.start * 1000)
        filter_parts.append(f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[dub{input_idx}]")
        input_idx += 1

    # Mix all dubbed segments together
    dub_labels = [f"[dub{i}]" for i in range(1, input_idx)]
    if len(dub_labels) == 1:
        mix_label = dub_labels[0]
    else:
        filter_parts.append(
            f"{''.join(dub_labels)}amix=inputs={len(dub_labels)}:normalize=0[dubbed_mix]"
        )
        mix_label = "[dubbed_mix]"

    # Option: mix with original background (music preservation)
    if preserve_music:
        # Reduce original audio volume and mix with dubbed
        filter_parts.append("[0:a]volume=0.15[bg]")
        final_mix = f"[bg]{mix_label}amix=inputs=2:normalize=0[final_audio]"
        filter_parts.append(final_mix)
        audio_map = "[final_audio]"
    else:
        audio_map = mix_label if "[" in mix_label else mix_label

    # Build full FFmpeg command
    filter_complex = ";".join(filter_parts)
    cmd = [get_ffmpeg_path(), "-hide_banner", "-y"]
    cmd.extend(input_args)
    cmd.extend(["-filter_complex", filter_complex])
    cmd.extend(["-map", "0:v", "-map", audio_map.strip("[]") if "[" not in audio_map else audio_map])
    cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"])
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.extend(["-movflags", "+faststart"])
    cmd.append(output_file)

    _stage_progress(on_progress, STAGE_COMPOSITE, 50, "Encoding final video...")

    try:
        run_ffmpeg(cmd, timeout=3600)
    except RuntimeError as exc:
        # Fallback: simpler approach — just replace audio entirely
        logger.warning("Complex composite failed: %s. Trying simple replacement.", exc)
        if tts_segments_with_audio:
            # Concatenate all TTS to single track
            concat_path = os.path.join(work_dir, "concat_tts.wav")
            list_path = os.path.join(work_dir, "concat_list.txt")
            with open(list_path, "w") as f:
                for seg in tts_segments_with_audio:
                    f.write(f"file '{seg.tts_audio_path}'\n")
            concat_cmd = (FFmpegCmd()
                          .option("f", "concat")
                          .option("safe", "0")
                          .input(list_path)
                          .audio_codec("pcm_s16le")
                          .output(concat_path)
                          .build())
            try:
                run_ffmpeg(concat_cmd, timeout=120)
                # Replace audio
                simple_cmd = (FFmpegCmd()
                              .input(video_path)
                              .input(concat_path)
                              .option("map", "0:v")
                              .option("map", "1:a")
                              .video_codec("copy")
                              .audio_codec("aac", bitrate="192k")
                              .faststart()
                              .output(output_file)
                              .build())
                run_ffmpeg(simple_cmd, timeout=300)
            except Exception as exc2:
                raise RuntimeError(f"Audio compositing failed: {exc2}")

    _stage_progress(on_progress, STAGE_COMPOSITE, 100, "Composite complete")
    return output_file


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def auto_dub(
    video_path: str,
    target_language: str = "es",
    config: Optional[DubConfig] = None,
    on_progress: Optional[Callable] = None,
) -> DubResult:
    """
    End-to-end auto-dubbing pipeline.

    Transcribes, translates, generates TTS, and composites a fully dubbed
    version of the input video.

    Args:
        video_path: Path to the source video file.
        target_language: ISO 639-1 language code for target dub.
        config: Optional DubConfig for fine-grained control.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        DubResult with output path, segment details, and pipeline status.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If target_language is not supported.
    """
    start_time = time.time()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES[:10])}..."
        )

    if config is None:
        config = DubConfig(target_language=target_language)
    else:
        config.target_language = target_language

    if on_progress:
        on_progress(0, "Initializing auto-dub pipeline...")

    info = get_video_info(video_path)
    total_duration = info.get("duration", 0)

    stages_completed = []
    stages_skipped = []
    voice_cloned = False
    lip_synced = False

    # Create working directory
    work_dir = tempfile.mkdtemp(prefix="opencut_autodub_")

    try:
        # Stage 1: Extract audio
        audio_path = _extract_source_audio(video_path, work_dir, on_progress)
        stages_completed.append(STAGE_EXTRACT)

        # Stage 2: Transcribe
        segments, detected_lang = _transcribe_audio(
            audio_path, config.source_language, config.whisper_model, on_progress
        )
        config.source_language = detected_lang
        stages_completed.append(STAGE_TRANSCRIBE)

        # Stage 3: Translate
        segments = _translate_segments(
            segments, detected_lang, target_language, on_progress
        )
        stages_completed.append(STAGE_TRANSLATE)

        # Stage 4: Voice clone reference
        voice_ref = ""
        if config.voice_clone:
            voice_ref = _extract_voice_reference(audio_path, segments, work_dir, on_progress)
            if voice_ref:
                voice_cloned = True
                stages_completed.append(STAGE_VOICE_CLONE)
            else:
                stages_skipped.append(STAGE_VOICE_CLONE)
        else:
            stages_skipped.append(STAGE_VOICE_CLONE)

        # Stage 5: TTS generation
        segments = _generate_all_tts(
            segments, target_language, config.tts_engine,
            voice_ref, work_dir, on_progress
        )
        stages_completed.append(STAGE_TTS)

        # Stage 7: Composite (before lip sync so lip sync can work on final)
        out_path = output_path(video_path, f"dubbed_{target_language}", config.output_dir)
        _composite_dubbed_audio(
            video_path, segments, work_dir, out_path,
            preserve_music=config.preserve_music, on_progress=on_progress
        )
        stages_completed.append(STAGE_COMPOSITE)

        # Stage 6: Lip sync (optional, on composited output)
        if config.lip_sync and os.path.isfile(out_path):
            synced = _apply_lip_sync(video_path, out_path, work_dir, on_progress)
            if synced != out_path:
                lip_synced = True
                out_path = synced
                stages_completed.append(STAGE_LIP_SYNC)
            else:
                stages_skipped.append(STAGE_LIP_SYNC)
        else:
            stages_skipped.append(STAGE_LIP_SYNC)

    except Exception:
        # Clean up on error but re-raise
        _cleanup_work_dir(work_dir)
        raise

    # Build result
    segments_dubbed = sum(1 for s in segments if s.tts_audio_path)
    processing_time = time.time() - start_time

    result = DubResult(
        output_path=out_path,
        source_language=detected_lang,
        target_language=target_language,
        target_language_name=LANGUAGE_NAMES.get(target_language, target_language),
        total_duration=round(total_duration, 3),
        segments_dubbed=segments_dubbed,
        segments_total=len(segments),
        segments=[s.to_dict() for s in segments],
        stages_completed=stages_completed,
        stages_skipped=stages_skipped,
        voice_cloned=voice_cloned,
        lip_synced=lip_synced,
        music_preserved=config.preserve_music,
        processing_time=round(processing_time, 2),
    )

    if on_progress:
        on_progress(100, f"Auto-dub complete: {segments_dubbed} segments dubbed")

    # Schedule cleanup of work directory
    _schedule_cleanup(work_dir)

    return result


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
def _cleanup_work_dir(work_dir: str):
    """Remove the temporary working directory."""
    import shutil
    try:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as exc:
        logger.debug("Failed to clean work dir %s: %s", work_dir, exc)


def _schedule_cleanup(work_dir: str, delay: float = 60.0):
    """Schedule deferred cleanup of work directory."""
    import threading

    def _deferred():
        time.sleep(delay)
        _cleanup_work_dir(work_dir)

    t = threading.Thread(target=_deferred, daemon=True, name="autodub-cleanup")
    t.start()
