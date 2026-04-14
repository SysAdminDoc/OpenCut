"""
OpenCut AI Foley Sound Generation Module

Generate sound effects from video content analysis.  Detect visual cues
(footsteps, impacts, doors) and synthesize matching foley sounds,
then mix them onto the original audio track.

Uses FFmpeg for motion/scene-change detection and sine/noise synthesis
with envelope shaping for foley generation.  Final mix via amix.

Functions:
    detect_foley_cues - Analyze video for moments needing foley
    generate_foley    - Generate and mix foley sounds onto video
"""

import logging
import math
import os
import re
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class FoleySound:
    """A single synthesized foley sound event."""
    sound_type: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    confidence: float = 0.0


@dataclass
class FoleyResult:
    """Result of foley generation and mixing."""
    output_path: str = ""
    sounds_generated: list = field(default_factory=list)
    duration: float = 0.0
    mix_level: float = 0.3


# ---------------------------------------------------------------------------
# Foley categories -- visual-cue-to-sound mapping
# ---------------------------------------------------------------------------
FOLEY_CATEGORIES: Dict[str, Dict] = {
    "footstep": {
        "description": "Walking/running detected via periodic lower-body motion",
        "freq_range": (80, 250),
        "duration_range": (0.05, 0.15),
        "envelope": "percussive",
    },
    "door": {
        "description": "Scene boundary with hard cut suggesting door open/close",
        "freq_range": (100, 400),
        "duration_range": (0.2, 0.5),
        "envelope": "thud",
    },
    "glass": {
        "description": "High-frequency transient with bright scene change",
        "freq_range": (2000, 6000),
        "duration_range": (0.1, 0.4),
        "envelope": "percussive",
    },
    "water": {
        "description": "Sustained motion with blue-dominant colour",
        "freq_range": (200, 1200),
        "duration_range": (0.5, 2.0),
        "envelope": "noise_wash",
    },
    "impact": {
        "description": "Sudden motion spike -- collision or hit",
        "freq_range": (60, 200),
        "duration_range": (0.05, 0.2),
        "envelope": "percussive",
    },
    "whoosh": {
        "description": "Fast lateral/pan motion across frame",
        "freq_range": (300, 3000),
        "duration_range": (0.2, 0.6),
        "envelope": "sweep",
    },
    "ambient": {
        "description": "Low-motion scene -- background room tone",
        "freq_range": (100, 600),
        "duration_range": (1.0, 3.0),
        "envelope": "noise_wash",
    },
    "crowd": {
        "description": "Multiple motion sources detected simultaneously",
        "freq_range": (200, 2000),
        "duration_range": (1.0, 3.0),
        "envelope": "noise_wash",
    },
    "vehicle": {
        "description": "Sustained directional motion at steady speed",
        "freq_range": (80, 500),
        "duration_range": (0.5, 2.0),
        "envelope": "sustained",
    },
    "nature": {
        "description": "Slow ambient motion -- wind, leaves, outdoors",
        "freq_range": (150, 1500),
        "duration_range": (1.0, 4.0),
        "envelope": "noise_wash",
    },
}


# ---------------------------------------------------------------------------
# Motion / Scene-change analysis
# ---------------------------------------------------------------------------
def _detect_scene_changes(video_path: str) -> List[dict]:
    """Detect scene changes via FFmpeg scdet filter.

    Returns list of ``{"time": float, "score": float}`` dicts.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", "scdet=s=1:t=10",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Scene detection timed out for foley analysis")
        return []

    detections: List[dict] = []
    score_re = re.compile(r"lavfi\.scd\.score=(\d+\.?\d*)")
    time_re = re.compile(r"lavfi\.scd\.time=(\d+\.?\d*)")

    for line in result.stderr.split("\n"):
        sm = score_re.search(line)
        tm = time_re.search(line)
        if sm and tm:
            detections.append({
                "time": float(tm.group(1)),
                "score": float(sm.group(1)) / 100.0,
            })
    return detections


def _detect_motion_peaks(video_path: str, sample_seconds: float = 0.0) -> List[dict]:
    """Estimate per-second motion magnitude via frame-diff analysis.

    Returns list of ``{"time": float, "magnitude": float}`` dicts
    where magnitude is normalized 0-1.
    """
    ffmpeg = get_ffmpeg_path()
    duration_args = ["-t", str(sample_seconds)] if sample_seconds > 0 else []

    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        *duration_args,
        "-vf", "select='gt(scene,0)',metadata=mode=print",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return []

    peaks: List[dict] = []
    pts_re = re.compile(r"pts_time:(\d+\.?\d*)")
    scene_re = re.compile(r"lavfi\.scene_score=(\d+\.?\d*)")

    lines = result.stderr.split("\n")
    for i, line in enumerate(lines):
        pts_m = pts_re.search(line)
        score = 0.0
        for offset in range(0, min(3, len(lines) - i)):
            sc_m = scene_re.search(lines[i + offset])
            if sc_m:
                score = float(sc_m.group(1))
                break
        if pts_m:
            peaks.append({
                "time": float(pts_m.group(1)),
                "magnitude": min(1.0, score),
            })

    return peaks


def _classify_cue(scene_score: float, motion_mag: float, timestamp: float) -> str:
    """Heuristically classify a detected cue into a foley category."""
    if scene_score > 0.7:
        return "door"
    if motion_mag > 0.6:
        return "impact"
    if motion_mag > 0.3:
        return "whoosh"
    if scene_score > 0.3 and motion_mag < 0.15:
        return "ambient"
    if motion_mag > 0.15:
        return "footstep"
    return "ambient"


# ---------------------------------------------------------------------------
# Public: cue detection
# ---------------------------------------------------------------------------
def detect_foley_cues(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> List[FoleySound]:
    """Analyze video for moments needing foley sound effects.

    Detects scene changes and motion peaks, then classifies each event
    into a foley category from ``FOLEY_CATEGORIES``.

    Args:
        video_path: Path to input video.
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        List of FoleySound cues (without generated audio yet).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Detecting scene changes...")

    scene_changes = _detect_scene_changes(video_path)

    if on_progress:
        on_progress(30, f"Found {len(scene_changes)} scene changes, analyzing motion...")

    motion_peaks = _detect_motion_peaks(video_path)

    if on_progress:
        on_progress(60, "Classifying foley cues...")

    # Merge scene changes and motion peaks into a unified timeline
    timeline: Dict[float, dict] = {}

    for sc in scene_changes:
        t = round(sc["time"], 2)
        if t not in timeline:
            timeline[t] = {"scene_score": 0.0, "motion_mag": 0.0}
        timeline[t]["scene_score"] = max(timeline[t]["scene_score"], sc["score"])

    for mp in motion_peaks:
        t = round(mp["time"], 2)
        if t not in timeline:
            timeline[t] = {"scene_score": 0.0, "motion_mag": 0.0}
        timeline[t]["motion_mag"] = max(timeline[t]["motion_mag"], mp["magnitude"])

    # Classify each cue
    cues: List[FoleySound] = []
    for t in sorted(timeline.keys()):
        info = timeline[t]
        sound_type = _classify_cue(info["scene_score"], info["motion_mag"], t)
        cat = FOLEY_CATEGORIES[sound_type]
        dur_lo, dur_hi = cat["duration_range"]
        confidence = max(info["scene_score"], info["motion_mag"])
        duration = dur_lo + (dur_hi - dur_lo) * min(1.0, confidence)

        cues.append(FoleySound(
            sound_type=sound_type,
            start_time=t,
            duration=round(duration, 3),
            confidence=round(confidence, 4),
        ))

    if on_progress:
        on_progress(90, f"Detected {len(cues)} foley cue points")

    return cues


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------
def _generate_pcm_samples(
    freq: float,
    duration: float,
    sample_rate: int = 44100,
    envelope: str = "percussive",
) -> bytes:
    """Synthesize PCM16 mono audio samples for a foley sound.

    Envelope types: percussive, thud, sweep, noise_wash, sustained.
    """
    import random

    num_samples = int(sample_rate * duration)
    if num_samples == 0:
        return b""

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        frac = i / max(1, num_samples - 1)

        # Base waveform
        if envelope == "noise_wash":
            val = random.uniform(-1.0, 1.0) * 0.3
            val += 0.15 * math.sin(2.0 * math.pi * freq * t)
        elif envelope == "sweep":
            inst_freq = freq * (1.0 + 2.0 * frac)
            val = 0.5 * math.sin(2.0 * math.pi * inst_freq * t)
        else:
            val = 0.5 * math.sin(2.0 * math.pi * freq * t)

        # Apply envelope
        if envelope == "percussive":
            env = math.exp(-6.0 * frac)
        elif envelope == "thud":
            env = math.exp(-3.0 * frac) * (1.0 + 0.5 * math.sin(2.0 * math.pi * 5.0 * t))
            env = max(0.0, min(1.0, env))
        elif envelope == "sweep":
            env = 1.0 - frac
        elif envelope == "noise_wash":
            if frac < 0.1:
                env = frac / 0.1
            elif frac > 0.9:
                env = (1.0 - frac) / 0.1
            else:
                env = 1.0
        elif envelope == "sustained":
            if frac < 0.05:
                env = frac / 0.05
            elif frac > 0.9:
                env = (1.0 - frac) / 0.1
            else:
                env = 1.0
        else:
            env = 1.0

        val *= env
        val = max(-1.0, min(1.0, val))
        samples.append(int(val * 32000))

    return struct.pack(f"<{len(samples)}h", *samples)


def _write_wav(pcm_data: bytes, wav_path: str, sample_rate: int = 44100) -> None:
    """Write raw PCM16 mono data to a WAV file."""
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def _synthesize_foley_sound(cue: FoleySound, wav_path: str) -> None:
    """Synthesize a single foley sound to a WAV file."""
    cat = FOLEY_CATEGORIES.get(cue.sound_type, FOLEY_CATEGORIES["ambient"])
    freq_lo, freq_hi = cat["freq_range"]
    freq = freq_lo + (freq_hi - freq_lo) * 0.5
    envelope = cat["envelope"]

    pcm = _generate_pcm_samples(freq, cue.duration, envelope=envelope)
    _write_wav(pcm, wav_path)


def _create_silence_wav(duration: float, wav_path: str, sample_rate: int = 44100) -> None:
    """Create a silent WAV file of given duration."""
    num_samples = int(sample_rate * duration)
    pcm = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    _write_wav(pcm, wav_path, sample_rate)


def _build_foley_track(
    cues: List[FoleySound],
    total_duration: float,
    track_path: str,
    on_progress: Optional[Callable] = None,
) -> None:
    """Build a single WAV track containing all foley sounds at their timestamps.

    Uses FFmpeg to place each synthesized clip at the correct offset via
    adelay + amix.
    """
    if not cues:
        _create_silence_wav(total_duration, track_path)
        return

    ffmpeg = get_ffmpeg_path()
    tmp_wavs: List[str] = []

    try:
        for i, cue in enumerate(cues):
            fd, wav = tempfile.mkstemp(suffix=".wav", prefix=f"opencut_foley_{i}_")
            os.close(fd)
            tmp_wavs.append(wav)
            _synthesize_foley_sound(cue, wav)

            if on_progress and i % max(1, len(cues) // 5) == 0:
                pct = 40 + int(20 * i / max(1, len(cues)))
                on_progress(pct, f"Synthesizing sound {i + 1}/{len(cues)}...")

        # Build FFmpeg command to mix all clips at correct offsets
        inputs = []
        filter_parts = []

        for i, (cue, wav) in enumerate(zip(cues, tmp_wavs)):
            inputs.extend(["-i", wav])
            delay_ms = int(cue.start_time * 1000)
            filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        if len(cues) == 1:
            fc = f"{filter_parts[0]};[a0]apad=whole_dur={total_duration}[out]"
        else:
            mix_inputs = "".join(f"[a{i}]" for i in range(len(cues)))
            fc = (
                ";".join(filter_parts)
                + f";{mix_inputs}amix=inputs={len(cues)}:duration=longest"
                + f",apad=whole_dur={total_duration}[out]"
            )

        cmd = [
            ffmpeg, "-hide_banner", "-y",
            *inputs,
            "-filter_complex", fc,
            "-map", "[out]",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            track_path,
        ]
        run_ffmpeg(cmd)

    finally:
        for w in tmp_wavs:
            try:
                os.unlink(w)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public: generate foley
# ---------------------------------------------------------------------------
def generate_foley(
    video_path: str,
    cues: Optional[List[FoleySound]] = None,
    output_path: Optional[str] = None,
    mix_level: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> FoleyResult:
    """Generate and mix AI foley sounds onto a video.

    Analyzes the video for foley cue points (or uses pre-detected cues),
    synthesizes matching sound effects, and mixes them with the original
    audio track.

    Args:
        video_path: Path to input video.
        cues: Pre-detected foley cues. Auto-detected if None.
        output_path: Output video path. Auto-generated if None.
        mix_level: Foley volume relative to original (0.0-1.0).
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        FoleyResult with output path, generated sounds list, and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    mix_level = max(0.0, min(1.0, float(mix_level)))

    if on_progress:
        on_progress(5, "Starting foley generation...")

    info = get_video_info(video_path)
    total_duration = info.get("duration", 0.0)

    # Detect cues if not provided
    if cues is None:
        if on_progress:
            on_progress(10, "Detecting foley cues...")
        cues = detect_foley_cues(video_path, on_progress=on_progress)

    if on_progress:
        on_progress(35, f"Generating {len(cues)} foley sounds...")

    if output_path is None:
        output_path = _output_path(video_path, "foley", "")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # If no cues, copy original
    if not cues:
        import shutil
        shutil.copy2(video_path, output_path)
        if on_progress:
            on_progress(100, "No foley cues detected")
        return FoleyResult(
            output_path=output_path,
            sounds_generated=[],
            duration=total_duration,
            mix_level=mix_level,
        )

    fd, foley_track = tempfile.mkstemp(suffix=".wav", prefix="opencut_foley_track_")
    os.close(fd)

    try:
        _build_foley_track(cues, total_duration, foley_track, on_progress=on_progress)

        if on_progress:
            on_progress(70, "Mixing foley with original audio...")

        ffmpeg = get_ffmpeg_path()
        orig_vol = 1.0 - mix_level * 0.5
        foley_vol = mix_level

        cmd = [
            ffmpeg, "-hide_banner", "-y",
            "-i", video_path,
            "-i", foley_track,
            "-filter_complex",
            f"[0:a]volume={orig_vol:.2f}[oa];"
            f"[1:a]volume={foley_vol:.2f}[fa];"
            f"[oa][fa]amix=inputs=2:duration=first[mixed]",
            "-map", "0:v",
            "-map", "[mixed]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(95, "Foley generation complete")

    finally:
        try:
            os.unlink(foley_track)
        except OSError:
            pass

    if on_progress:
        on_progress(100, f"Generated {len(cues)} foley sounds, mixed at {mix_level:.0%}")

    return FoleyResult(
        output_path=output_path,
        sounds_generated=cues,
        duration=total_duration,
        mix_level=mix_level,
    )
