"""
OpenCut Automated Dialogue Premix

Per-speaker dialogue processing chains driven by diarization data.

Pipeline per speaker:
1. Separate dialogue by speaker from diarization segments
2. Analyze: frequency profile, sibilance detection, dynamic range
3. Process: de-ess (4-10 kHz), content-type EQ, compress dynamic range
4. Level-match all speakers to target LUFS

Content-type EQ presets:
- interview: presence boost 2-4 kHz, rumble cut <80 Hz
- podcast:   warm boost 200-400 Hz, presence 3-5 kHz
- broadcast: flat with HPF at 80 Hz, presence 3 kHz
- film:      gentle warmth, wide presence, natural dynamics
- voiceover: strong presence, deep low cut, tight compression

All processing via FFmpeg audio filters.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# EQ Presets
# ---------------------------------------------------------------------------
EQ_PRESETS: Dict[str, Dict] = {
    "interview": {
        "label": "Interview",
        "description": "Presence boost 2-4 kHz, rumble cut below 80 Hz",
        "filters": [
            "highpass=f=80:poles=2",
            "equalizer=f=3000:t=q:w=2.0:g=3",
            "equalizer=f=200:t=q:w=1.0:g=-2",
        ],
        "target_lufs": -16.0,
    },
    "podcast": {
        "label": "Podcast",
        "description": "Warm boost 200-400 Hz, presence 3-5 kHz",
        "filters": [
            "highpass=f=60:poles=2",
            "equalizer=f=300:t=q:w=1.5:g=2",
            "equalizer=f=4000:t=q:w=2.0:g=3",
            "lowpass=f=16000",
        ],
        "target_lufs": -16.0,
    },
    "broadcast": {
        "label": "Broadcast",
        "description": "Flat response, HPF at 80 Hz, presence at 3 kHz",
        "filters": [
            "highpass=f=80:poles=2",
            "equalizer=f=3000:t=q:w=1.5:g=2",
        ],
        "target_lufs": -24.0,
    },
    "film": {
        "label": "Film Dialogue",
        "description": "Gentle warmth, wide presence, natural dynamics",
        "filters": [
            "highpass=f=60:poles=2",
            "equalizer=f=250:t=q:w=1.0:g=1.5",
            "equalizer=f=3500:t=q:w=2.5:g=2",
            "equalizer=f=8000:t=q:w=2.0:g=-1",
        ],
        "target_lufs": -24.0,
    },
    "voiceover": {
        "label": "Voiceover",
        "description": "Strong presence, deep low cut, tight compression",
        "filters": [
            "highpass=f=100:poles=2",
            "equalizer=f=2500:t=q:w=1.5:g=4",
            "equalizer=f=5000:t=q:w=2.0:g=2",
            "equalizer=f=150:t=q:w=1.0:g=-3",
        ],
        "target_lufs": -16.0,
    },
}

# Compressor presets per content type
COMPRESSOR_PRESETS: Dict[str, str] = {
    "interview": "acompressor=threshold=-20dB:ratio=3:attack=15:release=200:makeup=3dB:knee=5dB",
    "podcast": "acompressor=threshold=-18dB:ratio=4:attack=10:release=150:makeup=4dB:knee=4dB",
    "broadcast": "acompressor=threshold=-22dB:ratio=2.5:attack=20:release=250:makeup=2dB:knee=6dB",
    "film": "acompressor=threshold=-24dB:ratio=2:attack=25:release=300:makeup=1.5dB:knee=8dB",
    "voiceover": "acompressor=threshold=-16dB:ratio=4:attack=5:release=100:makeup=5dB:knee=3dB",
}

# De-esser frequency ranges
DEESS_FREQS = {
    "gentle": {"freq": 7000, "gain": -3, "width": 2.0},
    "moderate": {"freq": 6000, "gain": -5, "width": 2.5},
    "aggressive": {"freq": 5500, "gain": -8, "width": 3.0},
}


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------
@dataclass
class SpeakerStats:
    """Analysis and processing stats for a single speaker."""

    speaker_id: str = ""
    speaker_label: str = ""
    segment_count: int = 0
    total_duration: float = 0.0
    original_lufs: float = -23.0
    output_lufs: float = -23.0
    sibilance_detected: bool = False
    deess_applied: bool = False
    eq_preset: str = ""
    dynamic_range_db: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PremixResult:
    """Result from dialogue premix processing.

    Supports both attribute access (``result.target_lufs``) and dict-style
    subscript access (``result["target_lufs"]``) so the route layer and
    test suites with different conventions can both consume it.
    """

    output_path: str = ""
    speakers_processed: int = 0
    per_speaker_stats: List[SpeakerStats] = field(default_factory=list)
    target_lufs: float = -16.0
    content_type: str = ""
    processing_chain: List[str] = field(default_factory=list)
    segments_processed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_speaker_stats"] = [s.to_dict() for s in self.per_speaker_stats]
        return d

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.to_dict().keys()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def premix_dialogue(
    input_path: str,
    target_lufs: float = -23.0,
    content_type: str = "podcast",
    deess_strength: str = "moderate",
    diarization_segments: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Automated dialogue premix with per-speaker processing.

    Args:
        input_path: Source audio/video file.
        target_lufs: Target loudness in LUFS. Defaults to broadcast-friendly
            ``-23.0``. Clamped to ``[-36.0, -10.0]`` so out-of-range values
            from clients can never produce a useless output file.
        content_type: EQ preset to use (interview/podcast/broadcast/film/voiceover).
        deess_strength: De-esser strength (gentle/moderate/aggressive).
        diarization_segments: Speaker segments from diarization, each dict with
            keys: start (float), end (float), speaker (str). When provided
            the multi-speaker pipeline is used; otherwise the single-speaker
            chain is applied to the whole file.
        output_path: Output file path (auto-generated if None).
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        dict with output_path, target_lufs, processing_chain, content_type,
        and per-speaker stats. JSON-ready so route handlers can return it
        verbatim.
    """
    if content_type not in EQ_PRESETS:
        content_type = "podcast"

    if target_lufs is None:
        target_lufs = EQ_PRESETS[content_type]["target_lufs"]
    target_lufs = max(-36.0, min(-10.0, float(target_lufs)))

    if deess_strength not in DEESS_FREQS:
        deess_strength = "moderate"

    if output_path is None:
        output_path = _output_path(input_path, "premix")

    # Internal helpers historically call ``on_progress(int)`` (one arg).
    # Newer callers expect ``on_progress(pct, msg)`` (two args). Wrap so
    # both calling conventions work without forcing a refactor of every
    # internal call site.
    wrapped_progress = _wrap_progress_callback(on_progress)

    if wrapped_progress is not None:
        wrapped_progress(5)

    # If we have diarization segments, do per-speaker processing
    if diarization_segments and len(diarization_segments) > 0:
        result = _premix_multi_speaker(
            input_path, output_path, content_type, target_lufs,
            deess_strength, diarization_segments, wrapped_progress,
        )
    else:
        # Single-speaker fallback: process entire file
        result = _premix_single_speaker(
            input_path, output_path, content_type, target_lufs,
            deess_strength, wrapped_progress,
        )
    return result


def _wrap_progress_callback(callback):
    """Adapt a 1- or 2-arg progress callback to a uniform ``(pct,)`` shape.

    Internal pipeline functions invoke the callback with a single ``int``
    percentage; public callers (routes, CLI) commonly pass ``(pct, msg)``
    closures. The wrapper inspects the supplied callable and adds a
    blank message argument when needed so neither side has to care.
    """
    if callback is None:
        return None
    import inspect as _inspect
    try:
        sig = _inspect.signature(callback)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (_inspect.Parameter.POSITIONAL_OR_KEYWORD, _inspect.Parameter.POSITIONAL_ONLY)
        ]
        # Required positional parameters (no default).
        required = sum(1 for p in params if p.default is _inspect.Parameter.empty)
        wants_two = required >= 2 or any(p.kind is _inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()) is False and required >= 2
    except (TypeError, ValueError):
        wants_two = False

    if not wants_two:
        return callback

    def _wrapped(pct, *extra):
        if extra:
            return callback(pct, *extra)
        return callback(pct, "")

    return _wrapped


def premix_multi_speaker(
    input_path: str,
    diarization_segments: Optional[List[Dict]] = None,
    target_lufs: float = -23.0,
    content_type: str = "podcast",
    deess_strength: str = "moderate",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Public alias for the multi-speaker premix pipeline.

    Falls back to :func:`premix_dialogue`'s single-speaker chain when no
    segments are supplied — the route layer always wants a usable output
    file even when diarization fails.
    """
    if not diarization_segments:
        result = premix_dialogue(
            input_path=input_path,
            target_lufs=target_lufs,
            content_type=content_type,
            deess_strength=deess_strength,
            output_path=output_path,
            on_progress=on_progress,
        )
        # Surface that we ran the single-speaker fallback so callers can
        # show "no diarization data — applied flat chain" in the UI.
        if hasattr(result, "segments_processed"):
            result.segments_processed = 0
        elif isinstance(result, dict):
            result.setdefault("segments_processed", 0)
        return result

    return premix_dialogue(
        input_path=input_path,
        target_lufs=target_lufs,
        content_type=content_type,
        deess_strength=deess_strength,
        diarization_segments=diarization_segments,
        output_path=output_path,
        on_progress=on_progress,
    )


def list_presets() -> List[dict]:
    """List available EQ/processing presets for dialogue premix."""
    result = []
    for key, preset in EQ_PRESETS.items():
        result.append({
            "id": key,
            "label": preset["label"],
            "description": preset["description"],
            "target_lufs": preset["target_lufs"],
            "filter_count": len(preset["filters"]),
        })
    return result


def list_deess_strengths() -> List[dict]:
    """List available de-esser strength options."""
    return [
        {"id": k, "freq": v["freq"], "gain_db": v["gain"]}
        for k, v in DEESS_FREQS.items()
    ]


# ---------------------------------------------------------------------------
# Single Speaker Processing
# ---------------------------------------------------------------------------
def _premix_single_speaker(
    input_path: str,
    output_path: str,
    content_type: str,
    target_lufs: float,
    deess_strength: str,
    on_progress: Optional[Callable] = None,
) -> PremixResult:
    """Apply full premix chain to a single speaker (whole file)."""
    if on_progress:
        on_progress(10)

    # Analyze input
    analysis = _analyze_audio(input_path)

    if on_progress:
        on_progress(20)

    # Build processing chain
    chain, chain_desc = _build_processing_chain(
        content_type, deess_strength, analysis,
    )

    # Add loudness normalization as final step
    chain.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11")
    chain_desc.append(f"loudnorm to {target_lufs} LUFS")

    if on_progress:
        on_progress(40)

    # Apply chain
    af_str = ",".join(chain)
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_str,
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(90)

    stats = SpeakerStats(
        speaker_id="all",
        speaker_label="Full Mix",
        segment_count=1,
        total_duration=analysis.get("duration", 0),
        original_lufs=analysis.get("lufs", -23.0),
        output_lufs=target_lufs,
        sibilance_detected=analysis.get("sibilance", False),
        deess_applied=analysis.get("sibilance", False),
        eq_preset=content_type,
        dynamic_range_db=analysis.get("dynamic_range", 0),
    )

    if on_progress:
        on_progress(100)

    logger.info("Dialogue premix (single speaker, %s): %s -> %s",
                content_type, input_path, output_path)

    return PremixResult(
        output_path=output_path,
        speakers_processed=1,
        per_speaker_stats=[stats],
        target_lufs=target_lufs,
        content_type=content_type,
        processing_chain=chain_desc,
    )


# ---------------------------------------------------------------------------
# Multi-Speaker Processing
# ---------------------------------------------------------------------------
def _premix_multi_speaker(
    input_path: str,
    output_path: str,
    content_type: str,
    target_lufs: float,
    deess_strength: str,
    segments: List[Dict],
    on_progress: Optional[Callable] = None,
) -> PremixResult:
    """Process each speaker segment independently, then recombine."""
    ffmpeg = get_ffmpeg_path()

    # Group segments by speaker
    speakers: Dict[str, List[Dict]] = {}
    for seg in segments:
        spk = str(seg.get("speaker", "unknown"))
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append({
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
        })

    if on_progress:
        on_progress(10)

    total_speakers = len(speakers)
    speaker_stats = []
    processed_segments = []

    try:
        for spk_idx, (speaker_label, spk_segments) in enumerate(speakers.items()):
            if on_progress:
                base_pct = 10 + int(60 * spk_idx / max(1, total_speakers))
                on_progress(base_pct)

            # Extract all segments for this speaker and concatenate
            spk_temp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix=f"spk_{speaker_label}_",
            )
            spk_temp.close()

            seg_files = []
            total_dur = 0.0

            for seg in spk_segments:
                start = seg["start"]
                end = seg["end"]
                if end <= start:
                    continue

                seg_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="dseg_",
                )
                seg_file.close()

                cmd = [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(start), "-to", str(end),
                    "-i", input_path,
                    "-vn", "-c:a", "pcm_s16le",
                    seg_file.name,
                ]
                try:
                    run_ffmpeg(cmd, timeout=120)
                    seg_files.append({
                        "path": seg_file.name,
                        "start": start,
                        "end": end,
                    })
                    total_dur += end - start
                except Exception as exc:
                    logger.debug("Failed to extract segment %.1f-%.1f: %s", start, end, exc)
                    try:
                        os.unlink(seg_file.name)
                    except OSError:
                        pass

            if not seg_files:
                continue

            # Analyze combined speaker audio (use first segment as sample)
            analysis = _analyze_audio(seg_files[0]["path"])

            # Build per-speaker processing chain
            chain, chain_desc = _build_processing_chain(
                content_type, deess_strength, analysis,
            )
            chain.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11")

            af_str = ",".join(chain)

            # Process each segment with the speaker chain
            for sf in seg_files:
                premixed = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="pmx_",
                )
                premixed.close()

                cmd = [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", sf["path"],
                    "-af", af_str,
                    "-c:a", "pcm_s16le",
                    premixed.name,
                ]
                try:
                    run_ffmpeg(cmd, timeout=300)
                    processed_segments.append({
                        "path": premixed.name,
                        "start": sf["start"],
                        "end": sf["end"],
                        "speaker": speaker_label,
                    })
                except Exception as exc:
                    logger.debug("Failed to premix segment: %s", exc)
                    try:
                        os.unlink(premixed.name)
                    except OSError:
                        pass

            # Clean up extracted segment files
            for sf in seg_files:
                try:
                    os.unlink(sf["path"])
                except OSError:
                    pass

            try:
                os.unlink(spk_temp.name)
            except OSError:
                pass

            stats = SpeakerStats(
                speaker_id=f"spk_{spk_idx}",
                speaker_label=speaker_label,
                segment_count=len(seg_files),
                total_duration=total_dur,
                original_lufs=analysis.get("lufs", -23.0),
                output_lufs=target_lufs,
                sibilance_detected=analysis.get("sibilance", False),
                deess_applied=analysis.get("sibilance", False),
                eq_preset=content_type,
                dynamic_range_db=analysis.get("dynamic_range", 0),
            )
            speaker_stats.append(stats)

        if on_progress:
            on_progress(75)

        if not processed_segments:
            # No segments processed, fall back to single-speaker
            return _premix_single_speaker(
                input_path, output_path, content_type, target_lufs,
                deess_strength, on_progress,
            )

        # Reassemble: overlay processed segments onto original timeline
        _reassemble_segments(input_path, processed_segments, output_path)

        if on_progress:
            on_progress(95)

    finally:
        # Clean up all temp files
        for ps in processed_segments:
            try:
                os.unlink(ps["path"])
            except OSError:
                pass

    if on_progress:
        on_progress(100)

    chain_desc_final = []
    if speaker_stats:
        chain_desc_final.append(f"Processed {len(speaker_stats)} speakers")
    chain_desc_final.extend(
        _build_processing_chain(content_type, deess_strength, {})[1]
    )
    chain_desc_final.append(f"loudnorm to {target_lufs} LUFS")

    logger.info("Dialogue premix (%d speakers, %s): %s -> %s",
                len(speaker_stats), content_type, input_path, output_path)

    return PremixResult(
        output_path=output_path,
        speakers_processed=len(speaker_stats),
        per_speaker_stats=speaker_stats,
        target_lufs=target_lufs,
        content_type=content_type,
        processing_chain=chain_desc_final,
    )


# ---------------------------------------------------------------------------
# Audio Analysis
# ---------------------------------------------------------------------------
def _analyze_audio(filepath: str) -> dict:
    """Analyze audio for frequency profile, sibilance, dynamic range.

    Returns dict with keys: lufs, sibilance, dynamic_range, duration, peak_db.
    """
    result = {
        "lufs": -23.0,
        "sibilance": False,
        "dynamic_range": 12.0,
        "duration": 0.0,
        "peak_db": 0.0,
    }

    ffmpeg = get_ffmpeg_path()

    # Measure loudness with loudnorm analysis pass
    cmd_loudness = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", filepath,
        "-af", "loudnorm=I=-23:print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd_loudness, capture_output=True, text=True, timeout=60)
        stderr = proc.stderr

        # Parse loudnorm JSON from stderr
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            ld = json.loads(stderr[json_start:json_end])
            try:
                result["lufs"] = float(ld.get("input_i", -23.0))
            except (ValueError, TypeError):
                pass
            try:
                result["peak_db"] = float(ld.get("input_tp", 0.0))
            except (ValueError, TypeError):
                pass
            try:
                lra = float(ld.get("input_lra", 12.0))
                result["dynamic_range"] = lra
            except (ValueError, TypeError):
                pass
    except Exception as exc:
        logger.debug("Loudness analysis failed: %s", exc)

    # Check sibilance: measure energy in 5-10 kHz band vs overall
    cmd_sibl = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", filepath,
        "-af", "bandpass=f=7000:t=h:w=5000,volumedetect",
        "-t", "10",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd_sibl, capture_output=True, text=True, timeout=30)
        for line in proc.stderr.split("\n"):
            if "mean_volume" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    val = parts[-1].strip().replace("dB", "").strip()
                    sibl_db = float(val)
                    # If sibilant band is louder than -30 dB, flag it
                    result["sibilance"] = sibl_db > -30.0
                break
    except Exception:
        pass

    # Get duration
    result["duration"] = _get_duration(filepath)

    return result


def _get_duration(filepath: str) -> float:
    """Get audio duration in seconds."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        filepath,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0:
            return 0.0
        data = json.loads(proc.stdout.decode())
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Processing Chain Builder
# ---------------------------------------------------------------------------
def _build_processing_chain(
    content_type: str,
    deess_strength: str,
    analysis: dict,
) -> tuple:
    """Build FFmpeg audio filter chain based on content type and analysis.

    Returns (filter_list, description_list) tuple.
    """
    chain = []
    desc = []

    preset = EQ_PRESETS.get(content_type, EQ_PRESETS["podcast"])
    deess = DEESS_FREQS.get(deess_strength, DEESS_FREQS["moderate"])

    # 1. De-ess: attenuate sibilant peaks
    sibilant = analysis.get("sibilance", True)  # Apply de-ess by default if no analysis
    if sibilant:
        deess_filter = f"equalizer=f={deess['freq']}:t=q:w={deess['width']}:g={deess['gain']}"
        chain.append(deess_filter)
        desc.append(f"de-ess ({deess_strength}: {deess['freq']} Hz, {deess['gain']} dB)")

    # 2. EQ preset filters
    for f in preset["filters"]:
        chain.append(f)
    desc.append(f"EQ preset: {preset['label']}")

    # 3. Compressor
    compressor = COMPRESSOR_PRESETS.get(content_type, COMPRESSOR_PRESETS["podcast"])
    chain.append(compressor)
    desc.append(f"compressor ({content_type})")

    # 4. Dynamic normalization for consistent levels
    chain.append("dynaudnorm=p=0.9:s=5")
    desc.append("dynamic normalization")

    return chain, desc


# ---------------------------------------------------------------------------
# Segment Reassembly
# ---------------------------------------------------------------------------
def _reassemble_segments(
    original_path: str,
    segments: List[Dict],
    output_path: str,
) -> None:
    """Reassemble processed segments into a full timeline.

    Overlays processed dialogue segments onto the original audio,
    replacing the corresponding time ranges.
    """
    ffmpeg = get_ffmpeg_path()

    if not segments:
        # Just copy original
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", original_path,
            "-c", "copy",
            output_path,
        ]
        run_ffmpeg(cmd, timeout=600)
        return

    # Sort segments by start time
    segments_sorted = sorted(segments, key=lambda s: s["start"])

    # Build a complex filter to overlay processed segments
    # Strategy: use the original as base, then overlay each processed segment
    # at its correct position

    # For simplicity with large segment counts, concatenate with silence padding
    concat_list = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="reassemble_",
    )
    try:
        # Create silence fills for gaps between segments
        gap_files = []
        prev_end = 0.0

        for seg in segments_sorted:
            gap = seg["start"] - prev_end
            if gap > 0.01:
                # Extract original audio for the gap
                gap_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="gap_",
                )
                gap_file.close()
                cmd = [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(prev_end), "-t", str(gap),
                    "-i", original_path,
                    "-vn", "-c:a", "pcm_s16le",
                    gap_file.name,
                ]
                try:
                    run_ffmpeg(cmd, timeout=120)
                    safe_path = gap_file.name.replace("'", "'\\''")
                    concat_list.write(f"file '{safe_path}'\n")
                    gap_files.append(gap_file.name)
                except Exception:
                    try:
                        os.unlink(gap_file.name)
                    except OSError:
                        pass

            # Add the processed segment
            safe_path = seg["path"].replace("'", "'\\''")
            concat_list.write(f"file '{safe_path}'\n")
            prev_end = seg["end"]

        # Add trailing original audio if any
        total_duration = _get_duration(original_path)
        if prev_end < total_duration - 0.01:
            trail_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="trail_",
            )
            trail_file.close()
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(prev_end),
                "-i", original_path,
                "-vn", "-c:a", "pcm_s16le",
                trail_file.name,
            ]
            try:
                run_ffmpeg(cmd, timeout=120)
                safe_path = trail_file.name.replace("'", "'\\''")
                concat_list.write(f"file '{safe_path}'\n")
                gap_files.append(trail_file.name)
            except Exception:
                try:
                    os.unlink(trail_file.name)
                except OSError:
                    pass

        concat_list.close()

        # Concatenate everything
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list.name,
            "-c:a", "pcm_s16le",
            output_path,
        ]
        run_ffmpeg(cmd, timeout=1800)

    finally:
        try:
            os.unlink(concat_list.name)
        except OSError:
            pass
        for gf in gap_files:
            try:
                os.unlink(gf)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Legacy compatibility: simple single-pass premix
# ---------------------------------------------------------------------------
def premix_simple(
    input_path: str,
    target_lufs: float = -23.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Simple single-pass dialogue premix (legacy compatibility).

    Applies de-ess, EQ, compression, and loudness normalization.
    """
    output = _output_path(input_path, "premix")
    target_lufs = max(-36.0, min(-10.0, float(target_lufs)))

    if on_progress:
        on_progress(5)

    chain_parts = [
        "equalizer=f=6000:t=q:w=2:g=-3",
        "highpass=f=80:poles=2",
        "equalizer=f=3000:t=q:w=1.5:g=3",
        "acompressor=threshold=-18dB:ratio=3:attack=10:release=150:makeup=2dB:knee=4dB",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=summary",
    ]
    chain_description = [
        "de-ess (6kHz -3dB)",
        "HPF 80Hz, presence +3dB at 3kHz",
        "compress (3:1, -18dB threshold)",
        f"loudnorm to {target_lufs} LUFS",
    ]

    if on_progress:
        on_progress(30)

    af_chain = ",".join(chain_parts)
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_chain,
        "-c:v", "copy",
        output,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100)

    return {
        "output_path": output,
        "target_lufs": target_lufs,
        "processing_chain": chain_description,
    }
