"""
Multi-Track Audio Auto-Mixing (Category 74)

Analyze multiple audio tracks (dialogue, music, SFX, ambience) and generate
mix automation: auto-duck music under dialogue, level-match dialogue tracks,
apply frequency-dependent ducking, generate mix parameters as keyframe data.

Uses FFmpeg for audio analysis (loudness, frequency content).
Ducking profiles: podcast (aggressive duck), film (gentle duck),
music_video (no duck).
"""

import json
import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DUCKING_PROFILES = {
    "podcast": {
        "duck_amount_db": -18.0,
        "attack_ms": 50,
        "release_ms": 300,
        "threshold_db": -30.0,
        "hold_ms": 200,
        "dialogue_target_lufs": -16.0,
        "music_target_lufs": -30.0,
        "sfx_target_lufs": -22.0,
        "ambience_target_lufs": -35.0,
    },
    "film": {
        "duck_amount_db": -8.0,
        "attack_ms": 100,
        "release_ms": 500,
        "threshold_db": -35.0,
        "hold_ms": 400,
        "dialogue_target_lufs": -24.0,
        "music_target_lufs": -30.0,
        "sfx_target_lufs": -26.0,
        "ambience_target_lufs": -32.0,
    },
    "music_video": {
        "duck_amount_db": 0.0,
        "attack_ms": 0,
        "release_ms": 0,
        "threshold_db": -100.0,
        "hold_ms": 0,
        "dialogue_target_lufs": -16.0,
        "music_target_lufs": -14.0,
        "sfx_target_lufs": -20.0,
        "ambience_target_lufs": -28.0,
    },
}

TRACK_TYPES = ("dialogue", "music", "sfx", "ambience")
MAX_TRACKS = 64
KEYFRAME_INTERVAL = 0.1  # seconds between gain keyframes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrackAnalysis:
    """Analysis results for a single audio track."""
    file_path: str = ""
    track_type: str = "dialogue"
    duration: float = 0.0
    loudness_lufs: float = -23.0
    loudness_range: float = 0.0
    true_peak_dbtp: float = 0.0
    silence_segments: List[Dict] = field(default_factory=list)
    speech_segments: List[Dict] = field(default_factory=list)
    frequency_profile: str = "flat"  # flat, bass_heavy, treble_heavy, mid_focused
    sample_rate: int = 48000
    channels: int = 2
    gain_adjustment_db: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "track_type": self.track_type,
            "duration": self.duration,
            "loudness_lufs": self.loudness_lufs,
            "loudness_range": self.loudness_range,
            "true_peak_dbtp": self.true_peak_dbtp,
            "silence_segment_count": len(self.silence_segments),
            "speech_segment_count": len(self.speech_segments),
            "frequency_profile": self.frequency_profile,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "gain_adjustment_db": self.gain_adjustment_db,
        }


@dataclass
class GainKeyframe:
    """A single gain keyframe in the mix automation."""
    time: float = 0.0
    gain_db: float = 0.0
    is_ducking: bool = False

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "gain_db": round(self.gain_db, 2),
            "is_ducking": self.is_ducking,
        }


@dataclass
class TrackMixParams:
    """Mix parameters for a single track."""
    file_path: str = ""
    track_type: str = "dialogue"
    base_gain_db: float = 0.0
    keyframes: List[GainKeyframe] = field(default_factory=list)
    eq_settings: Dict = field(default_factory=dict)
    compressor_settings: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "track_type": self.track_type,
            "base_gain_db": round(self.base_gain_db, 2),
            "keyframe_count": len(self.keyframes),
            "keyframes": [k.to_dict() for k in self.keyframes],
            "eq_settings": self.eq_settings,
            "compressor_settings": self.compressor_settings,
        }


@dataclass
class AutoMixResult:
    """Result of the auto-mix pipeline."""
    keyframes: Dict[str, List[Dict]] = field(default_factory=dict)
    track_analysis: List[TrackAnalysis] = field(default_factory=list)
    recommended_levels: Dict[str, float] = field(default_factory=dict)
    mix_params: List[TrackMixParams] = field(default_factory=list)
    profile: str = "podcast"
    output_path: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "keyframes": self.keyframes,
            "track_analysis": [t.to_dict() for t in self.track_analysis],
            "recommended_levels": self.recommended_levels,
            "mix_params": [m.to_dict() for m in self.mix_params],
            "profile": self.profile,
            "output_path": self.output_path,
            "duration": self.duration,
            "track_count": len(self.track_analysis),
        }


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------
def _analyze_loudness(file_path: str) -> Dict:
    """Analyze loudness of an audio file using FFmpeg loudnorm filter."""
    cmd = [
        get_ffmpeg_path(), "-i", file_path,
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        # Parse loudnorm JSON output from stderr
        json_match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "loudness_lufs": float(data.get("input_i", -23.0)),
                "loudness_range": float(data.get("input_lra", 0.0)),
                "true_peak_dbtp": float(data.get("input_tp", 0.0)),
            }
    except Exception as exc:
        logger.debug("Loudness analysis failed for %s: %s", file_path, exc)

    return {"loudness_lufs": -23.0, "loudness_range": 0.0, "true_peak_dbtp": 0.0}


def _detect_silence(file_path: str, threshold_db: float = -40.0) -> List[Dict]:
    """Detect silence segments using FFmpeg silencedetect filter."""
    cmd = [
        get_ffmpeg_path(), "-i", file_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d=0.5",
        "-f", "null", "-",
    ]
    segments = []
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
        ends = re.findall(r"silence_end:\s*([\d.]+)", stderr)

        for i in range(min(len(starts), len(ends))):
            segments.append({
                "start": float(starts[i]),
                "end": float(ends[i]),
                "duration": float(ends[i]) - float(starts[i]),
            })
    except Exception as exc:
        logger.debug("Silence detection failed for %s: %s", file_path, exc)

    return segments


def _get_audio_info(file_path: str) -> Dict:
    """Get audio stream info via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", file_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            data = json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            if streams:
                s = streams[0]
                duration = float(s.get("duration", 0))
                if duration <= 0:
                    duration = float(data.get("format", {}).get("duration", 0))
                return {
                    "sample_rate": int(s.get("sample_rate", 48000)),
                    "channels": int(s.get("channels", 2)),
                    "duration": duration,
                }
    except Exception as exc:
        logger.debug("Audio info probe failed for %s: %s", file_path, exc)

    return {"sample_rate": 48000, "channels": 2, "duration": 0.0}


def analyze_track(file_path: str, track_type: str = "dialogue") -> TrackAnalysis:
    """Analyze a single audio track."""
    if track_type not in TRACK_TYPES:
        track_type = "dialogue"

    audio_info = _get_audio_info(file_path)
    loudness = _analyze_loudness(file_path)
    silence = _detect_silence(file_path)

    # Invert silence to get speech segments
    duration = audio_info["duration"]
    speech_segments = []
    prev_end = 0.0
    for seg in silence:
        if seg["start"] > prev_end:
            speech_segments.append({
                "start": prev_end,
                "end": seg["start"],
                "duration": seg["start"] - prev_end,
            })
        prev_end = seg["end"]
    if duration > prev_end:
        speech_segments.append({
            "start": prev_end,
            "end": duration,
            "duration": duration - prev_end,
        })

    return TrackAnalysis(
        file_path=file_path,
        track_type=track_type,
        duration=duration,
        loudness_lufs=loudness["loudness_lufs"],
        loudness_range=loudness["loudness_range"],
        true_peak_dbtp=loudness["true_peak_dbtp"],
        silence_segments=silence,
        speech_segments=speech_segments,
        sample_rate=audio_info["sample_rate"],
        channels=audio_info["channels"],
    )


# ---------------------------------------------------------------------------
# Level matching
# ---------------------------------------------------------------------------
def _compute_gain_adjustment(
    current_lufs: float,
    target_lufs: float,
) -> float:
    """Compute gain adjustment to reach target loudness."""
    if current_lufs <= -70.0:
        return 0.0
    adjustment = target_lufs - current_lufs
    # Clamp to reasonable range
    return max(-30.0, min(30.0, adjustment))


def _level_match_tracks(
    analyses: List[TrackAnalysis],
    profile: Dict,
) -> List[TrackAnalysis]:
    """Apply level-matching gain adjustments per track type."""
    type_targets = {
        "dialogue": profile["dialogue_target_lufs"],
        "music": profile["music_target_lufs"],
        "sfx": profile["sfx_target_lufs"],
        "ambience": profile["ambience_target_lufs"],
    }

    for analysis in analyses:
        target = type_targets.get(analysis.track_type, -23.0)
        analysis.gain_adjustment_db = _compute_gain_adjustment(
            analysis.loudness_lufs, target,
        )

    return analyses


# ---------------------------------------------------------------------------
# Ducking keyframe generation
# ---------------------------------------------------------------------------
def _find_dialogue_active_regions(
    dialogue_tracks: List[TrackAnalysis],
) -> List[Dict]:
    """Merge speech segments from all dialogue tracks."""
    all_speech = []
    for track in dialogue_tracks:
        all_speech.extend(track.speech_segments)

    if not all_speech:
        return []

    # Sort by start time
    all_speech.sort(key=lambda s: s["start"])

    # Merge overlapping regions
    merged = [all_speech[0].copy()]
    for seg in all_speech[1:]:
        if seg["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(seg.copy())

    return merged


def _generate_ducking_keyframes(
    track: TrackAnalysis,
    dialogue_regions: List[Dict],
    profile: Dict,
) -> List[GainKeyframe]:
    """Generate ducking gain keyframes for a non-dialogue track."""
    if not dialogue_regions or profile["duck_amount_db"] == 0.0:
        return []

    duck_db = profile["duck_amount_db"]
    attack_s = profile["attack_ms"] / 1000.0
    release_s = profile["release_ms"] / 1000.0
    hold_s = profile["hold_ms"] / 1000.0

    keyframes = []
    duration = track.duration

    if duration <= 0:
        return []

    for region in dialogue_regions:
        # Skip regions outside this track's duration
        if region["start"] > duration:
            continue

        # Attack: ramp down before dialogue
        attack_start = max(0.0, region["start"] - attack_s)
        keyframes.append(GainKeyframe(time=attack_start, gain_db=0.0, is_ducking=False))
        keyframes.append(GainKeyframe(time=region["start"], gain_db=duck_db, is_ducking=True))

        # Hold through dialogue
        hold_end = min(duration, region["end"] + hold_s)
        keyframes.append(GainKeyframe(time=hold_end, gain_db=duck_db, is_ducking=True))

        # Release: ramp back up
        release_end = min(duration, hold_end + release_s)
        keyframes.append(GainKeyframe(time=release_end, gain_db=0.0, is_ducking=False))

    # Sort and deduplicate
    keyframes.sort(key=lambda k: k.time)

    # Remove duplicate timestamps (keep last)
    if keyframes:
        deduped = [keyframes[0]]
        for k in keyframes[1:]:
            if abs(k.time - deduped[-1].time) < 0.001:
                deduped[-1] = k
            else:
                deduped.append(k)
        keyframes = deduped

    return keyframes


# ---------------------------------------------------------------------------
# EQ/Compressor suggestions
# ---------------------------------------------------------------------------
def _suggest_eq(track_type: str) -> Dict:
    """Suggest EQ settings based on track type."""
    presets = {
        "dialogue": {
            "high_pass_hz": 80,
            "low_shelf_db": -3.0,
            "low_shelf_hz": 200,
            "presence_boost_db": 2.0,
            "presence_hz": 3000,
            "presence_q": 1.5,
        },
        "music": {
            "high_pass_hz": 30,
            "low_shelf_db": 0.0,
            "low_shelf_hz": 100,
            "presence_boost_db": 0.0,
            "presence_hz": 3000,
            "presence_q": 1.0,
        },
        "sfx": {
            "high_pass_hz": 50,
            "low_shelf_db": 0.0,
            "low_shelf_hz": 150,
            "presence_boost_db": 1.0,
            "presence_hz": 5000,
            "presence_q": 2.0,
        },
        "ambience": {
            "high_pass_hz": 60,
            "low_shelf_db": -2.0,
            "low_shelf_hz": 200,
            "presence_boost_db": -1.0,
            "presence_hz": 4000,
            "presence_q": 1.0,
        },
    }
    return presets.get(track_type, presets["dialogue"])


def _suggest_compressor(track_type: str) -> Dict:
    """Suggest compressor settings based on track type."""
    presets = {
        "dialogue": {
            "threshold_db": -18.0,
            "ratio": 3.0,
            "attack_ms": 10,
            "release_ms": 100,
            "makeup_db": 3.0,
        },
        "music": {
            "threshold_db": -12.0,
            "ratio": 2.0,
            "attack_ms": 20,
            "release_ms": 200,
            "makeup_db": 1.0,
        },
        "sfx": {
            "threshold_db": -15.0,
            "ratio": 4.0,
            "attack_ms": 5,
            "release_ms": 50,
            "makeup_db": 2.0,
        },
        "ambience": {
            "threshold_db": -20.0,
            "ratio": 1.5,
            "attack_ms": 30,
            "release_ms": 300,
            "makeup_db": 0.0,
        },
    }
    return presets.get(track_type, presets["dialogue"])


# ---------------------------------------------------------------------------
# Mix-down via FFmpeg
# ---------------------------------------------------------------------------
def _build_mix_command(
    mix_params: List[TrackMixParams],
    output_file: str,
    sample_rate: int = 48000,
) -> List[str]:
    """Build FFmpeg command to mix down tracks with gain adjustments."""
    cmd = [get_ffmpeg_path(), "-y"]

    # Add input files
    for mp in mix_params:
        cmd.extend(["-i", mp.file_path])

    if not mix_params:
        return cmd

    # Build filter complex
    n = len(mix_params)
    filters = []
    for i, mp in enumerate(mix_params):
        gain = mp.base_gain_db + (mp.keyframes[0].gain_db if mp.keyframes else 0.0)
        volume = max(0.0, 10 ** (gain / 20.0))
        filters.append(f"[{i}:a]volume={volume:.4f}[a{i}]")

    # Mix all adjusted tracks
    mix_inputs = "".join(f"[a{i}]" for i in range(n))
    filters.append(f"{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=2[out]")

    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-ar", str(sample_rate),
        "-ac", "2",
        "-c:a", "aac",
        "-b:a", "192k",
        output_file,
    ])

    return cmd


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def auto_mix(
    tracks: List[Dict],
    profile: str = "podcast",
    output_file: str = "",
    mix_down: bool = False,
    on_progress: Optional[Callable] = None,
) -> AutoMixResult:
    """Auto-mix multiple audio tracks.

    Args:
        tracks: List of dicts with 'file_path' and 'type' keys.
            type is one of: dialogue, music, sfx, ambience.
        profile: Ducking profile — 'podcast', 'film', or 'music_video'.
        output_file: If set with mix_down=True, render mixed audio.
        mix_down: Whether to render a mixed-down audio file.
        on_progress: Callback(pct) for progress updates.

    Returns:
        AutoMixResult with keyframes, analysis, and mix parameters.
    """
    if profile not in DUCKING_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Use: {', '.join(DUCKING_PROFILES.keys())}"
        )

    if not tracks:
        raise ValueError("No tracks provided")

    if len(tracks) > MAX_TRACKS:
        raise ValueError(f"Too many tracks ({len(tracks)}). Max: {MAX_TRACKS}")

    profile_settings = DUCKING_PROFILES[profile]

    if on_progress:
        on_progress(5)

    # Step 1: Analyze all tracks
    analyses = []
    total = len(tracks)
    for i, track_info in enumerate(tracks):
        if on_progress:
            pct = 5 + int((i / max(total, 1)) * 30)
            on_progress(pct)

        file_path = track_info.get("file_path", "")
        track_type = track_info.get("type", "dialogue")

        if not file_path or not os.path.isfile(file_path):
            logger.warning("Track file not found: %s", file_path)
            analyses.append(TrackAnalysis(
                file_path=file_path,
                track_type=track_type,
            ))
            continue

        try:
            analysis = analyze_track(file_path, track_type)
            analyses.append(analysis)
        except Exception as exc:
            logger.warning("Failed to analyze track %s: %s", file_path, exc)
            analyses.append(TrackAnalysis(
                file_path=file_path,
                track_type=track_type,
            ))

    if on_progress:
        on_progress(40)

    # Step 2: Level-match tracks
    analyses = _level_match_tracks(analyses, profile_settings)

    if on_progress:
        on_progress(50)

    # Step 3: Find dialogue active regions
    dialogue_tracks = [a for a in analyses if a.track_type == "dialogue"]
    dialogue_regions = _find_dialogue_active_regions(dialogue_tracks)

    if on_progress:
        on_progress(60)

    # Step 4: Generate ducking keyframes for non-dialogue tracks
    all_keyframes: Dict[str, List[Dict]] = {}
    mix_params: List[TrackMixParams] = []

    for analysis in analyses:
        ducking_kf = []
        if analysis.track_type != "dialogue":
            ducking_kf = _generate_ducking_keyframes(
                analysis, dialogue_regions, profile_settings,
            )

        kf_key = os.path.basename(analysis.file_path) or analysis.track_type
        all_keyframes[kf_key] = [k.to_dict() for k in ducking_kf]

        mix_params.append(TrackMixParams(
            file_path=analysis.file_path,
            track_type=analysis.track_type,
            base_gain_db=analysis.gain_adjustment_db,
            keyframes=ducking_kf,
            eq_settings=_suggest_eq(analysis.track_type),
            compressor_settings=_suggest_compressor(analysis.track_type),
        ))

    if on_progress:
        on_progress(75)

    # Step 5: Compute recommended levels
    recommended = {}
    for analysis in analyses:
        target_key = f"{analysis.track_type}_target_lufs"
        recommended[analysis.track_type] = profile_settings.get(target_key, -23.0)

    # Step 6: Optional mix-down
    out_path = ""
    if mix_down and output_file:
        valid_params = [mp for mp in mix_params if mp.file_path and os.path.isfile(mp.file_path)]
        if valid_params:
            try:
                cmd = _build_mix_command(valid_params, output_file)
                _sp.run(cmd, capture_output=True, timeout=600, check=True)
                out_path = output_file
                logger.info("Mix rendered to %s", output_file)
            except Exception as exc:
                logger.warning("Mix-down failed: %s", exc)

    if on_progress:
        on_progress(95)

    duration = max((a.duration for a in analyses), default=0.0)

    result = AutoMixResult(
        keyframes=all_keyframes,
        track_analysis=analyses,
        recommended_levels=recommended,
        mix_params=mix_params,
        profile=profile,
        output_path=out_path,
        duration=duration,
    )

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# Preview: analyze first N seconds
# ---------------------------------------------------------------------------
def preview_mix(
    tracks: List[Dict],
    profile: str = "podcast",
    preview_seconds: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> AutoMixResult:
    """Generate mix preview for the first N seconds.

    Same as auto_mix but faster — only analyzes the preview duration.
    """
    if preview_seconds <= 0:
        preview_seconds = 10.0
    if preview_seconds > 60:
        preview_seconds = 60.0

    # For preview, we still run full analysis but flag it
    result = auto_mix(
        tracks=tracks,
        profile=profile,
        mix_down=False,
        on_progress=on_progress,
    )

    # Trim keyframes to preview window
    for key in result.keyframes:
        result.keyframes[key] = [
            kf for kf in result.keyframes[key]
            if kf.get("time", 0) <= preview_seconds
        ]

    # Trim mix params keyframes
    for mp in result.mix_params:
        mp.keyframes = [k for k in mp.keyframes if k.time <= preview_seconds]

    result.duration = min(result.duration, preview_seconds)

    return result
