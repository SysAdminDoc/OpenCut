"""
OpenCut QC/QA Checks

Automated quality-control checks for video files:
- Black frame detection (FFmpeg blackdetect)
- Frozen frame detection (FFmpeg freezedetect)
- Audio phase issues (FFmpeg aphasemeter)
- Silence gap detection (FFmpeg silencedetect)
- Color bars / slate / tone detection (leader elements)
- Dropout & glitch detection (SSIM-based)
- Comprehensive QC report generation

Uses FFmpeg only — no additional dependencies required.
"""

import html
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses — Black Frame Detection
# ---------------------------------------------------------------------------
@dataclass
class BlackFrame:
    """A detected black frame region."""
    start: float
    end: float
    duration: float


@dataclass
class BlackFrameResult:
    """Results from black frame detection."""
    frames: List[BlackFrame] = field(default_factory=list)
    total_black_duration: float = 0.0
    file_duration: float = 0.0
    black_percentage: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — Frozen Frame Detection
# ---------------------------------------------------------------------------
@dataclass
class FrozenFrame:
    """A detected frozen (still) frame region."""
    start: float
    end: float
    duration: float


@dataclass
class FrozenFrameResult:
    """Results from frozen frame detection."""
    frames: List[FrozenFrame] = field(default_factory=list)
    total_frozen_duration: float = 0.0
    file_duration: float = 0.0
    frozen_percentage: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — Audio Phase Check
# ---------------------------------------------------------------------------
@dataclass
class PhaseIssue:
    """A segment with out-of-phase audio."""
    start: float
    end: float
    avg_phase: float


@dataclass
class PhaseResult:
    """Results from audio phase analysis."""
    issues: List[PhaseIssue] = field(default_factory=list)
    overall_avg_phase: float = 0.0
    has_phase_problems: bool = False
    file_duration: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — Silence Gap Detection
# ---------------------------------------------------------------------------
@dataclass
class SilenceGap:
    """A detected silence gap."""
    start: float
    end: float
    duration: float


@dataclass
class SilenceGapResult:
    """Results from silence gap detection."""
    gaps: List[SilenceGap] = field(default_factory=list)
    total_silence_duration: float = 0.0
    file_duration: float = 0.0
    silence_percentage: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — Leader Element Detection
# ---------------------------------------------------------------------------
@dataclass
class LeaderResult:
    """Results from color bars / slate / tone detection."""
    bars_detected: bool = False
    bars_end_time: float = 0.0
    tone_detected: bool = False
    tone_end_time: float = 0.0
    slate_detected: bool = False
    slate_end_time: float = 0.0
    recommended_trim_point: float = 0.0


# ---------------------------------------------------------------------------
# Dataclasses — Full QC Report
# ---------------------------------------------------------------------------
@dataclass
class FullQCReport:
    """Combined results from all QC checks."""
    black_frames: Optional[BlackFrameResult] = None
    frozen_frames: Optional[FrozenFrameResult] = None
    audio_phase: Optional[PhaseResult] = None
    silence_gaps: Optional[SilenceGapResult] = None
    leader: Optional[LeaderResult] = None
    passed: bool = True
    issues_summary: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: probe file duration
# ---------------------------------------------------------------------------
def _probe_duration(input_path: str) -> float:
    """Return file duration in seconds via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, KeyError):
        return 0.0


# =========================================================================
# Feature 28.1: Black Frame Detection
# =========================================================================
def detect_black_frames(
    input_path: str,
    threshold: float = 0.98,
    min_duration: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> BlackFrameResult:
    """
    Detect black frames in a video using FFmpeg's blackdetect filter.

    Args:
        input_path: Source video file.
        threshold: Pixel darkness threshold (0.0-1.0). Higher = stricter.
        min_duration: Minimum black segment duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        BlackFrameResult with detected black frame regions.
    """
    if on_progress:
        on_progress(5, "Probing file duration...")

    file_duration = _probe_duration(input_path)

    if on_progress:
        on_progress(10, "Running black frame detection...")

    # FFmpeg blackdetect filter
    threshold = max(0.0, min(1.0, float(threshold)))
    min_duration = max(0.0, float(min_duration))

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-vf", f"blackdetect=d={min_duration}:pix_th={threshold}",
        "-an", "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Black frame detection timed out after 600 seconds.")

    if on_progress:
        on_progress(70, "Parsing black frame data...")

    # Parse blackdetect output from stderr
    # Format: [blackdetect @ 0x...] black_start:1.5 black_end:3.2 black_duration:1.7
    _black_re = re.compile(
        r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)\s+black_duration:\s*([\d.]+)"
    )

    frames: List[BlackFrame] = []
    for line in result.stderr.splitlines():
        if "blackdetect" not in line:
            continue
        m = _black_re.search(line)
        if m:
            start = float(m.group(1))
            end = float(m.group(2))
            duration = float(m.group(3))
            frames.append(BlackFrame(start=start, end=end, duration=duration))

    if on_progress:
        on_progress(90, "Calculating statistics...")

    total_black = sum(f.duration for f in frames)
    black_pct = (total_black / file_duration * 100.0) if file_duration > 0 else 0.0

    if on_progress:
        on_progress(100, f"Found {len(frames)} black frame regions")

    return BlackFrameResult(
        frames=frames,
        total_black_duration=round(total_black, 3),
        file_duration=round(file_duration, 3),
        black_percentage=round(black_pct, 2),
    )


# =========================================================================
# Feature 28.1: Frozen Frame Detection
# =========================================================================
def detect_frozen_frames(
    input_path: str,
    noise_threshold: float = 0.001,
    duration_threshold: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> FrozenFrameResult:
    """
    Detect frozen (still) frames using FFmpeg's freezedetect filter.

    Args:
        input_path: Source video file.
        noise_threshold: Noise tolerance (lower = stricter). Default 0.001.
        duration_threshold: Minimum frozen duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        FrozenFrameResult with detected frozen regions.
    """
    if on_progress:
        on_progress(5, "Probing file duration...")

    file_duration = _probe_duration(input_path)

    if on_progress:
        on_progress(10, "Running frozen frame detection...")

    noise_threshold = max(0.0, float(noise_threshold))
    duration_threshold = max(0.0, float(duration_threshold))

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-vf", f"freezedetect=n={noise_threshold}:d={duration_threshold}",
        "-an", "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Frozen frame detection timed out after 600 seconds.")

    if on_progress:
        on_progress(70, "Parsing frozen frame data...")

    # Parse freezedetect output from stderr
    # Format: [freezedetect @ 0x...] lavfi.freezedetect.freeze_start: 5.0
    #         [freezedetect @ 0x...] lavfi.freezedetect.freeze_duration: 3.0
    #         [freezedetect @ 0x...] lavfi.freezedetect.freeze_end: 8.0
    _start_re = re.compile(r"freeze_start:\s*([\d.]+)")
    _end_re = re.compile(r"freeze_end:\s*([\d.]+)")
    _dur_re = re.compile(r"freeze_duration:\s*([\d.]+)")

    # Collect start/end/duration triplets
    starts: List[float] = []
    ends: List[float] = []
    durations: List[float] = []

    for line in result.stderr.splitlines():
        if "freezedetect" not in line:
            continue
        sm = _start_re.search(line)
        if sm:
            starts.append(float(sm.group(1)))
            continue
        em = _end_re.search(line)
        if em:
            ends.append(float(em.group(1)))
            continue
        dm = _dur_re.search(line)
        if dm:
            durations.append(float(dm.group(1)))

    frames: List[FrozenFrame] = []
    # Match starts with ends; durations come between start and end
    for i, start_val in enumerate(starts):
        if i < len(ends):
            end_val = ends[i]
            dur_val = durations[i] if i < len(durations) else (end_val - start_val)
        else:
            # Freeze extends to end of file
            end_val = file_duration if file_duration > 0 else start_val
            dur_val = durations[i] if i < len(durations) else (end_val - start_val)
        frames.append(FrozenFrame(
            start=round(start_val, 3),
            end=round(end_val, 3),
            duration=round(dur_val, 3),
        ))

    if on_progress:
        on_progress(90, "Calculating statistics...")

    total_frozen = sum(f.duration for f in frames)
    frozen_pct = (total_frozen / file_duration * 100.0) if file_duration > 0 else 0.0

    if on_progress:
        on_progress(100, f"Found {len(frames)} frozen frame regions")

    return FrozenFrameResult(
        frames=frames,
        total_frozen_duration=round(total_frozen, 3),
        file_duration=round(file_duration, 3),
        frozen_percentage=round(frozen_pct, 2),
    )


# =========================================================================
# Feature 28.2: Audio Phase Check
# =========================================================================
def check_audio_phase(
    input_path: str,
    threshold: float = -0.5,
    on_progress: Optional[Callable] = None,
) -> PhaseResult:
    """
    Check audio phase correlation using FFmpeg's aphasemeter filter.

    Flags segments where phase drops below threshold (indicating
    out-of-phase / phase-cancelled audio).

    Args:
        input_path: Source video or audio file.
        threshold: Phase threshold. Segments below this are flagged.
                   Range -1.0 (fully out of phase) to 1.0 (mono).
        on_progress: Progress callback(pct, msg).

    Returns:
        PhaseResult with detected phase issues.
    """
    if on_progress:
        on_progress(5, "Probing file duration...")

    file_duration = _probe_duration(input_path)

    if on_progress:
        on_progress(10, "Running audio phase analysis...")

    threshold = max(-1.0, min(1.0, float(threshold)))

    # Use aphasemeter with video output disabled, log phase per frame
    # We use the metadata output to get per-frame phase values
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-af", "aphasemeter=video=0",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio phase analysis timed out after 600 seconds.")

    if on_progress:
        on_progress(60, "Parsing phase data...")

    # Parse aphasemeter output — looks for lines with phase value
    # [Parsed_aphasemeter_0 @ 0x...] mono: ... phase: ...
    # Or metadata: lavfi.aphasemeter.phase=0.xxxx
    _phase_re = re.compile(r"aphasemeter.*phase[=:]\s*([-\d.]+)")
    _ts_re = re.compile(r"pts_time:\s*([\d.]+)")

    phase_values: List[float] = []
    phase_timestamps: List[float] = []
    current_ts = 0.0

    for line in result.stderr.splitlines():
        # Try to extract timestamp
        ts_m = _ts_re.search(line)
        if ts_m:
            current_ts = float(ts_m.group(1))

        # Extract phase value
        pm = _phase_re.search(line)
        if pm:
            try:
                pv = float(pm.group(1))
                phase_values.append(pv)
                phase_timestamps.append(current_ts)
            except ValueError:
                continue

    if on_progress:
        on_progress(80, "Identifying phase issues...")

    # Group consecutive low-phase samples into issue segments
    issues: List[PhaseIssue] = []
    in_issue = False
    issue_start = 0.0
    issue_phases: List[float] = []

    # If we have timestamps, use them; otherwise estimate from count
    sample_interval = (file_duration / len(phase_values)) if phase_values else 0.0

    for i, pv in enumerate(phase_values):
        ts = phase_timestamps[i] if i < len(phase_timestamps) and phase_timestamps[i] > 0 else i * sample_interval

        if pv < threshold:
            if not in_issue:
                in_issue = True
                issue_start = ts
                issue_phases = []
            issue_phases.append(pv)
        else:
            if in_issue:
                issue_end = ts
                avg_phase = sum(issue_phases) / len(issue_phases) if issue_phases else 0.0
                issues.append(PhaseIssue(
                    start=round(issue_start, 3),
                    end=round(issue_end, 3),
                    avg_phase=round(avg_phase, 4),
                ))
                in_issue = False
                issue_phases = []

    # Close trailing issue
    if in_issue and issue_phases:
        issue_end = file_duration if file_duration > 0 else (len(phase_values) * sample_interval)
        avg_phase = sum(issue_phases) / len(issue_phases)
        issues.append(PhaseIssue(
            start=round(issue_start, 3),
            end=round(issue_end, 3),
            avg_phase=round(avg_phase, 4),
        ))

    overall_avg = (sum(phase_values) / len(phase_values)) if phase_values else 0.0
    has_problems = len(issues) > 0

    if on_progress:
        on_progress(100, f"Phase analysis complete — {len(issues)} issues found")

    return PhaseResult(
        issues=issues,
        overall_avg_phase=round(overall_avg, 4),
        has_phase_problems=has_problems,
        file_duration=round(file_duration, 3),
    )


# =========================================================================
# Feature 28.2: Silence Gap Detection
# =========================================================================
def detect_silence_gaps(
    input_path: str,
    noise_db: float = -50,
    min_duration: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> SilenceGapResult:
    """
    Detect silence gaps in audio using FFmpeg's silencedetect filter.

    Args:
        input_path: Source video or audio file.
        noise_db: Noise floor in dB. Audio below this is silence. Default -50.
        min_duration: Minimum silence duration to flag (seconds).
        on_progress: Progress callback(pct, msg).

    Returns:
        SilenceGapResult with detected silence gaps.
    """
    if on_progress:
        on_progress(5, "Probing file duration...")

    file_duration = _probe_duration(input_path)

    if on_progress:
        on_progress(10, "Running silence detection...")

    noise_db = max(-100.0, min(0.0, float(noise_db)))
    min_duration = max(0.0, float(min_duration))

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Silence detection timed out after 600 seconds.")

    if on_progress:
        on_progress(70, "Parsing silence data...")

    # Parse silencedetect output
    # [silencedetect @ 0x...] silence_start: 1.5
    # [silencedetect @ 0x...] silence_end: 3.2 | silence_duration: 1.7
    _start_re = re.compile(r"silence_start:\s*([\d.]+)")
    _end_re = re.compile(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)")

    starts: List[float] = []
    gaps: List[SilenceGap] = []

    for line in result.stderr.splitlines():
        if "silencedetect" not in line:
            continue
        sm = _start_re.search(line)
        if sm:
            starts.append(float(sm.group(1)))
            continue
        em = _end_re.search(line)
        if em:
            end_val = float(em.group(1))
            dur_val = float(em.group(2))
            start_val = starts.pop(0) if starts else (end_val - dur_val)
            gaps.append(SilenceGap(
                start=round(start_val, 3),
                end=round(end_val, 3),
                duration=round(dur_val, 3),
            ))

    # Handle unclosed silence (extends to end of file)
    if starts:
        for sv in starts:
            end_val = file_duration if file_duration > 0 else sv
            gaps.append(SilenceGap(
                start=round(sv, 3),
                end=round(end_val, 3),
                duration=round(end_val - sv, 3),
            ))

    if on_progress:
        on_progress(90, "Calculating statistics...")

    total_silence = sum(g.duration for g in gaps)
    silence_pct = (total_silence / file_duration * 100.0) if file_duration > 0 else 0.0

    if on_progress:
        on_progress(100, f"Found {len(gaps)} silence gaps")

    return SilenceGapResult(
        gaps=gaps,
        total_silence_duration=round(total_silence, 3),
        file_duration=round(file_duration, 3),
        silence_percentage=round(silence_pct, 2),
    )


# =========================================================================
# Feature 28.3: Color Bars & Slate Detection
# =========================================================================
def detect_leader_elements(
    input_path: str,
    scan_duration: float = 120.0,
    on_progress: Optional[Callable] = None,
) -> LeaderResult:
    """
    Detect SMPTE color bars, 1kHz tone, and slate in video leader.

    Scans the first `scan_duration` seconds for:
    - Color bars via histogram analysis (high saturation uniformity)
    - 1kHz reference tone via FFmpeg astats (high-energy narrow-band audio)
    - Slate frames via black-to-content transition after bars

    Args:
        input_path: Source video file.
        scan_duration: How many seconds to scan from start. Default 120.
        on_progress: Progress callback(pct, msg).

    Returns:
        LeaderResult with detection timestamps and recommended trim point.
    """
    if on_progress:
        on_progress(5, "Probing file info...")

    file_duration = _probe_duration(input_path)
    scan_duration = min(float(scan_duration), file_duration) if file_duration > 0 else float(scan_duration)

    bars_detected = False
    bars_end_time = 0.0
    tone_detected = False
    tone_end_time = 0.0
    slate_detected = False
    slate_end_time = 0.0

    # -----------------------------------------------------------------
    # Step 1: Detect color bars via signalstats (saturation analysis)
    # -----------------------------------------------------------------
    if on_progress:
        on_progress(10, "Scanning for color bars...")

    # Use signalstats to measure per-frame saturation statistics
    # Color bars have very high and uniform saturation values
    cmd_bars = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-t", str(scan_duration),
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=print:file=-",
        "-an", "-f", "null", "-",
    ]

    try:
        result_bars = subprocess.run(cmd_bars, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Color bars detection timed out.")

    if on_progress:
        on_progress(30, "Analyzing color bar patterns...")

    # Parse signalstats metadata for high BRNG (out-of-broadcast-range) values
    # Color bars intentionally produce very specific saturation patterns.
    # We look for TOUT (temporal outlier) and BRNG (broadcast range) spikes
    # that are characteristic of synthetic test patterns.
    _frame_ts_re = re.compile(r"pts_time:\s*([\d.]+)")
    _brng_re = re.compile(r"lavfi\.signalstats\.BRNG=\s*([\d.]+)")
    _satavg_re = re.compile(r"lavfi\.signalstats\.SATAVG=\s*([\d.]+)")

    high_sat_start = -1.0
    high_sat_end = 0.0
    current_ts = 0.0
    consecutive_bar_frames = 0
    BAR_SAT_THRESHOLD = 80.0  # Color bars have high average saturation
    BAR_MIN_FRAMES = 10  # Need at least 10 consecutive bar-like frames

    for line in result_bars.stderr.splitlines():
        ts_m = _frame_ts_re.search(line)
        if ts_m:
            current_ts = float(ts_m.group(1))

        sat_m = _satavg_re.search(line)
        if sat_m:
            sat_val = float(sat_m.group(1))
            if sat_val >= BAR_SAT_THRESHOLD:
                if high_sat_start < 0:
                    high_sat_start = current_ts
                high_sat_end = current_ts
                consecutive_bar_frames += 1
            else:
                if consecutive_bar_frames >= BAR_MIN_FRAMES:
                    bars_detected = True
                    bars_end_time = high_sat_end
                    break
                # Reset tracking
                high_sat_start = -1.0
                consecutive_bar_frames = 0

    # Handle case where bars run to the end of scan window
    if not bars_detected and consecutive_bar_frames >= BAR_MIN_FRAMES:
        bars_detected = True
        bars_end_time = high_sat_end

    # Fallback: check BRNG (broadcast range violations) which bars also produce
    if not bars_detected:
        brng_frames = 0
        brng_start = -1.0
        brng_end = 0.0
        current_ts = 0.0

        for line in result_bars.stderr.splitlines():
            ts_m = _frame_ts_re.search(line)
            if ts_m:
                current_ts = float(ts_m.group(1))

            brng_m = _brng_re.search(line)
            if brng_m:
                brng_val = float(brng_m.group(1))
                if brng_val > 5.0:  # Significant broadcast range violations
                    if brng_start < 0:
                        brng_start = current_ts
                    brng_end = current_ts
                    brng_frames += 1
                else:
                    if brng_frames >= BAR_MIN_FRAMES:
                        bars_detected = True
                        bars_end_time = brng_end
                        break
                    brng_start = -1.0
                    brng_frames = 0

        if not bars_detected and brng_frames >= BAR_MIN_FRAMES:
            bars_detected = True
            bars_end_time = brng_end

    # -----------------------------------------------------------------
    # Step 2: Detect 1kHz reference tone via astats
    # -----------------------------------------------------------------
    if on_progress:
        on_progress(50, "Scanning for reference tone...")

    # Use astats to detect high-RMS audio consistent with a test tone
    cmd_tone = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-t", str(scan_duration),
        "-af", "astats=metadata=1:reset=1,ametadata=print:file=-",
        "-vn", "-f", "null", "-",
    ]

    try:
        result_tone = subprocess.run(cmd_tone, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Tone detection timed out for %s", input_path)
        result_tone = type("R", (), {"stderr": "", "stdout": ""})()

    if on_progress:
        on_progress(70, "Analyzing tone patterns...")

    # Parse astats for sustained high RMS level (characteristic of 1kHz tone)
    _rms_re = re.compile(r"lavfi\.astats\.\d+\.RMS_level=\s*([-\d.]+)")
    _tone_ts_re = re.compile(r"pts_time:\s*([\d.]+)")

    tone_start = -1.0
    tone_end_val = 0.0
    consecutive_tone = 0
    TONE_RMS_THRESHOLD = -24.0  # dB — reference tone is typically -20 to -12 dBFS
    TONE_MIN_FRAMES = 5  # Need sustained tone
    current_ts = 0.0

    for line in result_tone.stderr.splitlines():
        ts_m = _tone_ts_re.search(line)
        if ts_m:
            current_ts = float(ts_m.group(1))

        rms_m = _rms_re.search(line)
        if rms_m:
            rms_val = float(rms_m.group(1))
            if rms_val > TONE_RMS_THRESHOLD and rms_val < 0:
                # High sustained RMS suggests a tone
                if tone_start < 0:
                    tone_start = current_ts
                tone_end_val = current_ts
                consecutive_tone += 1
            else:
                if consecutive_tone >= TONE_MIN_FRAMES and tone_start >= 0:
                    tone_detected = True
                    tone_end_time = tone_end_val
                    break
                tone_start = -1.0
                consecutive_tone = 0

    if not tone_detected and consecutive_tone >= TONE_MIN_FRAMES and tone_start >= 0:
        tone_detected = True
        tone_end_time = tone_end_val

    # -----------------------------------------------------------------
    # Step 3: Detect slate (dark frame after bars, before program)
    # -----------------------------------------------------------------
    if on_progress:
        on_progress(85, "Checking for slate...")

    # A slate typically appears as a dark/black region right after color bars
    # and before program content starts. Use blackdetect on the region after bars.
    if bars_detected and bars_end_time > 0:
        slate_scan_start = bars_end_time
        slate_scan_end = min(bars_end_time + 30.0, scan_duration)

        cmd_slate = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-ss", str(slate_scan_start),
            "-i", input_path,
            "-t", str(slate_scan_end - slate_scan_start),
            "-vf", "blackdetect=d=0.5:pix_th=0.10",
            "-an", "-f", "null", "-",
        ]

        try:
            result_slate = subprocess.run(
                cmd_slate, capture_output=True, text=True, timeout=120
            )
            _slate_end_re = re.compile(r"black_end:\s*([\d.]+)")
            for line in result_slate.stderr.splitlines():
                if "blackdetect" not in line:
                    continue
                sm = _slate_end_re.search(line)
                if sm:
                    # black_end is relative to the -ss offset
                    relative_end = float(sm.group(1))
                    slate_detected = True
                    slate_end_time = round(slate_scan_start + relative_end, 3)
                    break  # Take first black region as slate
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning("Slate detection failed: %s", e)

    # -----------------------------------------------------------------
    # Compute recommended trim point
    # -----------------------------------------------------------------
    if on_progress:
        on_progress(95, "Computing recommended trim point...")

    # Trim point = latest of all detected leader elements
    trim_candidates = []
    if bars_detected:
        trim_candidates.append(bars_end_time)
    if tone_detected:
        trim_candidates.append(tone_end_time)
    if slate_detected:
        trim_candidates.append(slate_end_time)

    recommended_trim = max(trim_candidates) if trim_candidates else 0.0

    if on_progress:
        on_progress(100, "Leader detection complete")

    return LeaderResult(
        bars_detected=bars_detected,
        bars_end_time=round(bars_end_time, 3),
        tone_detected=tone_detected,
        tone_end_time=round(tone_end_time, 3),
        slate_detected=slate_detected,
        slate_end_time=round(slate_end_time, 3),
        recommended_trim_point=round(recommended_trim, 3),
    )


# =========================================================================
# Full QC Check (runs all checks)
# =========================================================================
def run_full_qc(
    input_path: str,
    on_progress: Optional[Callable] = None,
    black_threshold: float = 0.98,
    black_min_duration: float = 0.5,
    freeze_noise: float = 0.001,
    freeze_duration: float = 2.0,
    phase_threshold: float = -0.5,
    silence_noise_db: float = -50,
    silence_min_duration: float = 2.0,
    leader_scan_duration: float = 120.0,
) -> FullQCReport:
    """
    Run all QC checks on a media file and return a combined report.

    Args:
        input_path: Source video or audio file.
        on_progress: Progress callback(pct, msg).
        All other args: Per-check configuration.

    Returns:
        FullQCReport with results from all checks.
    """
    report = FullQCReport()
    issues: List[str] = []

    def _sub_progress(stage_start: int, stage_end: int):
        """Create a sub-progress reporter for a stage."""
        def _cb(pct, msg=""):
            if on_progress:
                scaled = stage_start + (pct / 100.0) * (stage_end - stage_start)
                on_progress(int(scaled), msg)
        return _cb

    # 1. Black frames (0-18%)
    try:
        if on_progress:
            on_progress(0, "Checking for black frames...")
        report.black_frames = detect_black_frames(
            input_path,
            threshold=black_threshold,
            min_duration=black_min_duration,
            on_progress=_sub_progress(0, 18),
        )
        if report.black_frames.frames:
            issues.append(
                f"{len(report.black_frames.frames)} black frame regions "
                f"({report.black_frames.black_percentage}% of file)"
            )
    except Exception as e:
        logger.warning("Black frame check failed: %s", e)
        issues.append(f"Black frame check failed: {e}")

    # 2. Frozen frames (18-36%)
    try:
        if on_progress:
            on_progress(18, "Checking for frozen frames...")
        report.frozen_frames = detect_frozen_frames(
            input_path,
            noise_threshold=freeze_noise,
            duration_threshold=freeze_duration,
            on_progress=_sub_progress(18, 36),
        )
        if report.frozen_frames.frames:
            issues.append(
                f"{len(report.frozen_frames.frames)} frozen frame regions "
                f"({report.frozen_frames.frozen_percentage}% of file)"
            )
    except Exception as e:
        logger.warning("Frozen frame check failed: %s", e)
        issues.append(f"Frozen frame check failed: {e}")

    # 3. Audio phase (36-54%)
    try:
        if on_progress:
            on_progress(36, "Checking audio phase...")
        report.audio_phase = check_audio_phase(
            input_path,
            threshold=phase_threshold,
            on_progress=_sub_progress(36, 54),
        )
        if report.audio_phase.has_phase_problems:
            issues.append(
                f"{len(report.audio_phase.issues)} audio phase issues "
                f"(avg phase: {report.audio_phase.overall_avg_phase})"
            )
    except Exception as e:
        logger.warning("Audio phase check failed: %s", e)
        issues.append(f"Audio phase check failed: {e}")

    # 4. Silence gaps (54-72%)
    try:
        if on_progress:
            on_progress(54, "Checking for silence gaps...")
        report.silence_gaps = detect_silence_gaps(
            input_path,
            noise_db=silence_noise_db,
            min_duration=silence_min_duration,
            on_progress=_sub_progress(54, 72),
        )
        if report.silence_gaps.gaps:
            issues.append(
                f"{len(report.silence_gaps.gaps)} silence gaps "
                f"({report.silence_gaps.silence_percentage}% of file)"
            )
    except Exception as e:
        logger.warning("Silence gap check failed: %s", e)
        issues.append(f"Silence gap check failed: {e}")

    # 5. Leader elements (72-95%)
    try:
        if on_progress:
            on_progress(72, "Checking for leader elements...")
        report.leader = detect_leader_elements(
            input_path,
            scan_duration=leader_scan_duration,
            on_progress=_sub_progress(72, 95),
        )
        if report.leader.bars_detected:
            issues.append(f"Color bars detected (end at {report.leader.bars_end_time}s)")
        if report.leader.tone_detected:
            issues.append(f"Reference tone detected (end at {report.leader.tone_end_time}s)")
        if report.leader.slate_detected:
            issues.append(f"Slate detected (end at {report.leader.slate_end_time}s)")
        if report.leader.recommended_trim_point > 0:
            issues.append(f"Recommended trim at {report.leader.recommended_trim_point}s")
    except Exception as e:
        logger.warning("Leader detection failed: %s", e)
        issues.append(f"Leader detection failed: {e}")

    report.issues_summary = issues
    report.passed = len(issues) == 0

    if on_progress:
        status = "PASSED" if report.passed else f"FAILED — {len(issues)} issues"
        on_progress(100, f"QC complete: {status}")

    return report


# =========================================================================
# Feature 28.4: Dropout & Glitch Detection
# =========================================================================
@dataclass
class Dropout:
    """A detected dropout or glitch."""
    frame_num: int
    timestamp: float
    type: str           # "frame_glitch", "macroblocking", "timecode_break"
    severity: str       # "critical", "major", "minor"
    description: str


@dataclass
class DropoutResult:
    """Results from dropout/glitch detection."""
    dropouts: List[Dropout] = field(default_factory=list)
    total_dropouts: int = 0
    file_duration: float = 0.0
    frames_analyzed: int = 0


def detect_dropouts(
    input_path: str,
    ssim_threshold: float = 0.5,
    block_threshold: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> DropoutResult:
    """
    Detect frame dropouts and glitches using SSIM frame-to-frame analysis.

    Analyzes consecutive frame pairs via FFmpeg's ssim filter to find
    sudden quality drops that indicate glitches, macroblocking, or
    timecode breaks.

    Args:
        input_path: Source video file.
        ssim_threshold: SSIM threshold below which a frame is flagged as a
            glitch (0.0-1.0). Lower = more tolerant. Default 0.5.
        block_threshold: Blockiness threshold for macroblocking detection.
        on_progress: Progress callback(pct, msg).

    Returns:
        DropoutResult with detected dropout events.
    """
    if on_progress:
        on_progress(5, "Probing file info...")

    file_duration = _probe_duration(input_path)
    info = get_video_info(input_path)
    fps = info.get("fps", 30.0)

    if on_progress:
        on_progress(10, "Running SSIM frame analysis...")

    ssim_threshold = max(0.0, min(1.0, float(ssim_threshold)))

    # Use FFmpeg's ssim filter comparing each frame to the next
    # tblend creates a temporal difference; ssim measures frame similarity
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-vf", "split[a][b];[b]trim=start_frame=1[b1];[a][b1]ssim=stats_file=-",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Dropout detection timed out after 1200 seconds.")

    if on_progress:
        on_progress(60, "Parsing SSIM data...")

    # Parse SSIM output - format: n:FRAME Y:VALUE U:VALUE V:VALUE All:VALUE (dB)
    _ssim_re = re.compile(r"n:\s*(\d+)\s+.*?All:\s*([\d.]+)")
    dropouts: List[Dropout] = []
    frames_analyzed = 0

    for line in result.stderr.splitlines():
        m = _ssim_re.search(line)
        if not m:
            continue
        frame_num = int(m.group(1))
        ssim_val = float(m.group(2))
        frames_analyzed = max(frames_analyzed, frame_num + 1)

        if ssim_val < ssim_threshold:
            timestamp = frame_num / fps if fps > 0 else 0.0
            # Classify severity
            if ssim_val < 0.2:
                severity = "critical"
                d_type = "frame_glitch"
                desc = f"Severe frame glitch at frame {frame_num} (SSIM={ssim_val:.4f})"
            elif ssim_val < ssim_threshold * 0.6:
                severity = "major"
                d_type = "frame_glitch"
                desc = f"Frame glitch at frame {frame_num} (SSIM={ssim_val:.4f})"
            else:
                severity = "minor"
                d_type = "macroblocking"
                desc = f"Possible macroblocking at frame {frame_num} (SSIM={ssim_val:.4f})"

            dropouts.append(Dropout(
                frame_num=frame_num,
                timestamp=round(timestamp, 3),
                type=d_type,
                severity=severity,
                description=desc,
            ))

    if on_progress:
        on_progress(85, "Checking for timecode breaks...")

    # Check for timecode breaks via ffprobe packet timestamps
    tc_cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,dts_time",
        "-of", "csv=p=0",
        input_path,
    ]
    try:
        tc_result = subprocess.run(tc_cmd, capture_output=True, text=True, timeout=120)
        prev_pts = None
        frame_idx = 0
        expected_interval = 1.0 / fps if fps > 0 else 0.033
        for line in tc_result.stdout.splitlines():
            parts = line.strip().split(",")
            if not parts or not parts[0]:
                continue
            try:
                pts = float(parts[0])
            except ValueError:
                continue
            if prev_pts is not None:
                gap = pts - prev_pts
                # Flag gaps that are more than 3x the expected frame interval
                if gap > expected_interval * 3 and gap > 0.1:
                    dropouts.append(Dropout(
                        frame_num=frame_idx,
                        timestamp=round(pts, 3),
                        type="timecode_break",
                        severity="major",
                        description=f"Timecode break at {pts:.3f}s (gap={gap:.3f}s, expected ~{expected_interval:.4f}s)",
                    ))
            prev_pts = pts
            frame_idx += 1
    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.warning("Timecode break check failed: %s", exc)

    # Sort by timestamp
    dropouts.sort(key=lambda d: d.timestamp)

    if on_progress:
        on_progress(100, f"Found {len(dropouts)} dropouts/glitches")

    return DropoutResult(
        dropouts=dropouts,
        total_dropouts=len(dropouts),
        file_duration=round(file_duration, 3),
        frames_analyzed=frames_analyzed,
    )


# =========================================================================
# Feature 28.5: Comprehensive QC Report Generator
# =========================================================================
RULESETS = {
    "broadcast": {
        "label": "Broadcast (Strict)",
        "black_threshold": 0.98,
        "black_min_duration": 0.3,
        "freeze_noise": 0.001,
        "freeze_duration": 1.5,
        "phase_threshold": -0.3,
        "silence_noise_db": -50,
        "silence_min_duration": 1.0,
        "ssim_threshold": 0.5,
        "loudness_target_lufs": -24.0,
    },
    "netflix": {
        "label": "Netflix (Streaming)",
        "black_threshold": 0.95,
        "black_min_duration": 0.5,
        "freeze_noise": 0.003,
        "freeze_duration": 2.0,
        "phase_threshold": -0.5,
        "silence_noise_db": -50,
        "silence_min_duration": 2.0,
        "ssim_threshold": 0.4,
        "loudness_target_lufs": -14.0,
    },
    "youtube": {
        "label": "YouTube (Lenient)",
        "black_threshold": 0.90,
        "black_min_duration": 1.0,
        "freeze_noise": 0.01,
        "freeze_duration": 5.0,
        "phase_threshold": -0.7,
        "silence_noise_db": -60,
        "silence_min_duration": 5.0,
        "ssim_threshold": 0.3,
        "loudness_target_lufs": -14.0,
    },
}


@dataclass
class QCCheckResult:
    """Result of a single QC check within a comprehensive report."""
    check_name: str
    status: str         # "pass", "fail", "warning", "error"
    details: str
    issues: List[Dict] = field(default_factory=list)


@dataclass
class QCReport:
    """Comprehensive QC report."""
    overall_verdict: str = "pass"  # "pass", "fail", "warning"
    per_check: List[QCCheckResult] = field(default_factory=list)
    total_issues: int = 0
    critical_count: int = 0
    warning_count: int = 0
    ruleset: str = "broadcast"
    file_path: str = ""
    file_duration: float = 0.0
    html_report: str = ""


def generate_qc_report(
    input_path: str,
    ruleset: str = "broadcast",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> QCReport:
    """
    Generate a comprehensive QC report by running all available checks.

    Args:
        input_path: Source video file.
        ruleset: Ruleset name ("broadcast", "netflix", "youtube").
        output_path: Optional output path for JSON report file.
        on_progress: Progress callback(pct, msg).

    Returns:
        QCReport with aggregated results from all checks.
    """
    if ruleset not in RULESETS:
        ruleset = "broadcast"

    rules = RULESETS[ruleset]
    report = QCReport(ruleset=ruleset, file_path=input_path)
    checks: List[QCCheckResult] = []
    total_issues = 0
    critical = 0
    warnings = 0

    file_duration = _probe_duration(input_path)
    report.file_duration = round(file_duration, 3)

    def _sub_progress(stage_start: int, stage_end: int):
        def _cb(pct, msg=""):
            if on_progress:
                scaled = stage_start + (pct / 100.0) * (stage_end - stage_start)
                on_progress(int(scaled), msg)
        return _cb

    # --- 1. Black frames (0-12%) ---
    try:
        if on_progress:
            on_progress(0, "Checking black frames...")
        bf = detect_black_frames(
            input_path,
            threshold=rules["black_threshold"],
            min_duration=rules["black_min_duration"],
            on_progress=_sub_progress(0, 12),
        )
        issue_list = [
            {"start": f.start, "end": f.end, "duration": f.duration}
            for f in bf.frames
        ]
        if bf.frames:
            checks.append(QCCheckResult(
                check_name="black_frames",
                status="fail",
                details=f"{len(bf.frames)} black frame regions ({bf.black_percentage}%)",
                issues=issue_list,
            ))
            total_issues += len(bf.frames)
            critical += len(bf.frames)
        else:
            checks.append(QCCheckResult(
                check_name="black_frames", status="pass",
                details="No black frames detected",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="black_frames", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 2. Frozen frames (12-24%) ---
    try:
        if on_progress:
            on_progress(12, "Checking frozen frames...")
        ff = detect_frozen_frames(
            input_path,
            noise_threshold=rules["freeze_noise"],
            duration_threshold=rules["freeze_duration"],
            on_progress=_sub_progress(12, 24),
        )
        issue_list = [
            {"start": f.start, "end": f.end, "duration": f.duration}
            for f in ff.frames
        ]
        if ff.frames:
            checks.append(QCCheckResult(
                check_name="frozen_frames",
                status="fail",
                details=f"{len(ff.frames)} frozen regions ({ff.frozen_percentage}%)",
                issues=issue_list,
            ))
            total_issues += len(ff.frames)
            critical += len(ff.frames)
        else:
            checks.append(QCCheckResult(
                check_name="frozen_frames", status="pass",
                details="No frozen frames detected",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="frozen_frames", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 3. Audio phase (24-36%) ---
    try:
        if on_progress:
            on_progress(24, "Checking audio phase...")
        ap = check_audio_phase(
            input_path,
            threshold=rules["phase_threshold"],
            on_progress=_sub_progress(24, 36),
        )
        issue_list = [
            {"start": i.start, "end": i.end, "avg_phase": i.avg_phase}
            for i in ap.issues
        ]
        if ap.has_phase_problems:
            checks.append(QCCheckResult(
                check_name="audio_phase",
                status="fail",
                details=f"{len(ap.issues)} phase issues (avg={ap.overall_avg_phase})",
                issues=issue_list,
            ))
            total_issues += len(ap.issues)
            critical += len(ap.issues)
        else:
            checks.append(QCCheckResult(
                check_name="audio_phase", status="pass",
                details=f"Phase OK (avg={ap.overall_avg_phase})",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="audio_phase", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 4. Silence gaps (36-48%) ---
    try:
        if on_progress:
            on_progress(36, "Checking silence gaps...")
        sg = detect_silence_gaps(
            input_path,
            noise_db=rules["silence_noise_db"],
            min_duration=rules["silence_min_duration"],
            on_progress=_sub_progress(36, 48),
        )
        issue_list = [
            {"start": g.start, "end": g.end, "duration": g.duration}
            for g in sg.gaps
        ]
        if sg.gaps:
            checks.append(QCCheckResult(
                check_name="silence_gaps",
                status="warning",
                details=f"{len(sg.gaps)} silence gaps ({sg.silence_percentage}%)",
                issues=issue_list,
            ))
            total_issues += len(sg.gaps)
            warnings += len(sg.gaps)
        else:
            checks.append(QCCheckResult(
                check_name="silence_gaps", status="pass",
                details="No excessive silence gaps",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="silence_gaps", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 5. Leader elements (48-58%) ---
    try:
        if on_progress:
            on_progress(48, "Checking leader elements...")
        ld = detect_leader_elements(
            input_path, scan_duration=120.0,
            on_progress=_sub_progress(48, 58),
        )
        leader_issues = []
        if ld.bars_detected:
            leader_issues.append({"type": "bars", "end_time": ld.bars_end_time})
        if ld.tone_detected:
            leader_issues.append({"type": "tone", "end_time": ld.tone_end_time})
        if ld.slate_detected:
            leader_issues.append({"type": "slate", "end_time": ld.slate_end_time})

        if leader_issues:
            checks.append(QCCheckResult(
                check_name="leader_detect",
                status="warning",
                details=f"Leader elements found (trim at {ld.recommended_trim_point}s)",
                issues=leader_issues,
            ))
            warnings += len(leader_issues)
        else:
            checks.append(QCCheckResult(
                check_name="leader_detect", status="pass",
                details="No leader elements detected",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="leader_detect", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 6. Dropout/glitch detection (58-75%) ---
    try:
        if on_progress:
            on_progress(58, "Checking for dropouts...")
        dr = detect_dropouts(
            input_path,
            ssim_threshold=rules["ssim_threshold"],
            on_progress=_sub_progress(58, 75),
        )
        issue_list = [
            {"frame": d.frame_num, "timestamp": d.timestamp,
             "type": d.type, "severity": d.severity}
            for d in dr.dropouts
        ]
        if dr.dropouts:
            crit_drops = sum(1 for d in dr.dropouts if d.severity == "critical")
            checks.append(QCCheckResult(
                check_name="dropout_detect",
                status="fail" if crit_drops > 0 else "warning",
                details=f"{len(dr.dropouts)} dropouts ({crit_drops} critical)",
                issues=issue_list,
            ))
            total_issues += len(dr.dropouts)
            critical += crit_drops
            warnings += len(dr.dropouts) - crit_drops
        else:
            checks.append(QCCheckResult(
                check_name="dropout_detect", status="pass",
                details="No dropouts detected",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="dropout_detect", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    # --- 7. Loudness check (75-90%) ---
    try:
        if on_progress:
            on_progress(75, "Measuring loudness...")
        from opencut.core.audio_suite import measure_loudness
        loudness = measure_loudness(input_path)
        target = rules.get("loudness_target_lufs", -14.0)
        integrated = loudness.input_i
        true_peak = loudness.input_tp
        deviation = abs(integrated - target)
        if deviation > 3.0:
            checks.append(QCCheckResult(
                check_name="loudness_check",
                status="warning",
                details=f"Loudness {integrated:.1f} LUFS (target {target:.1f}, deviation {deviation:.1f})",
                issues=[{"integrated_lufs": integrated,
                         "true_peak": true_peak,
                         "target": target}],
            ))
            warnings += 1
        else:
            checks.append(QCCheckResult(
                check_name="loudness_check", status="pass",
                details=f"Loudness {integrated:.1f} LUFS (target {target:.1f})",
            ))
    except Exception as e:
        checks.append(QCCheckResult(
            check_name="loudness_check", status="error",
            details=f"Check failed: {e}",
        ))
        warnings += 1

    if on_progress:
        on_progress(90, "Generating report...")

    # Determine overall verdict
    report.per_check = checks
    report.total_issues = total_issues
    report.critical_count = critical
    report.warning_count = warnings

    if critical > 0:
        report.overall_verdict = "fail"
    elif warnings > 0:
        report.overall_verdict = "warning"
    else:
        report.overall_verdict = "pass"

    # Generate HTML report
    report.html_report = export_qc_report_html(report)

    # Write JSON report if output_path given
    if output_path:
        report_dict = {
            "overall_verdict": report.overall_verdict,
            "total_issues": report.total_issues,
            "critical_count": report.critical_count,
            "warning_count": report.warning_count,
            "ruleset": report.ruleset,
            "file_path": report.file_path,
            "file_duration": report.file_duration,
            "per_check": [
                {
                    "check_name": c.check_name,
                    "status": c.status,
                    "details": c.details,
                    "issues": c.issues,
                }
                for c in report.per_check
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

    if on_progress:
        on_progress(100, f"QC report: {report.overall_verdict.upper()}")

    return report


def export_qc_report_html(qc_report: QCReport, output_path: Optional[str] = None) -> str:
    """
    Export a QCReport as a styled HTML string.

    Args:
        qc_report: The QCReport to render.
        output_path: Optional file path to write the HTML to.

    Returns:
        HTML string of the report.
    """
    verdict_colors = {"pass": "#28a745", "fail": "#dc3545", "warning": "#ffc107"}
    status_colors = {"pass": "#28a745", "fail": "#dc3545", "warning": "#ffc107", "error": "#6c757d"}

    esc = html.escape
    verdict_color = verdict_colors.get(qc_report.overall_verdict, "#6c757d")

    rows = []
    for c in qc_report.per_check:
        sc = status_colors.get(c.status, "#6c757d")
        issue_count = len(c.issues) if c.issues else 0
        rows.append(
            f"<tr>"
            f"<td>{esc(c.check_name)}</td>"
            f"<td style='color:{sc};font-weight:bold'>{esc(c.status.upper())}</td>"
            f"<td>{esc(c.details)}</td>"
            f"<td>{issue_count}</td>"
            f"</tr>"
        )

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenCut QC Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
.verdict {{ font-size: 1.4em; font-weight: bold; color: {verdict_color};
            padding: 12px 20px; border-radius: 6px;
            background: {verdict_color}18; display: inline-block; margin: 10px 0 20px; }}
.summary {{ display: flex; gap: 20px; margin: 20px 0; }}
.stat {{ background: #f8f9fa; padding: 12px 20px; border-radius: 6px; text-align: center; }}
.stat .num {{ font-size: 1.6em; font-weight: bold; }}
.stat .label {{ font-size: 0.85em; color: #666; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; }}
tr:hover {{ background: #f8f9fa; }}
.meta {{ color: #666; font-size: 0.9em; margin-top: 30px; }}
</style>
</head>
<body>
<h1>OpenCut QC Report</h1>
<div class="verdict">{esc(qc_report.overall_verdict.upper())}</div>
<div class="summary">
  <div class="stat"><div class="num">{qc_report.total_issues}</div><div class="label">Total Issues</div></div>
  <div class="stat"><div class="num" style="color:#dc3545">{qc_report.critical_count}</div><div class="label">Critical</div></div>
  <div class="stat"><div class="num" style="color:#ffc107">{qc_report.warning_count}</div><div class="label">Warnings</div></div>
</div>
<table>
<thead><tr><th>Check</th><th>Status</th><th>Details</th><th>Issues</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<div class="meta">
  <p>File: {esc(qc_report.file_path)}</p>
  <p>Duration: {qc_report.file_duration:.1f}s</p>
  <p>Ruleset: {esc(qc_report.ruleset)}</p>
  <p>Generated by OpenCut QC Engine</p>
</div>
</body>
</html>"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    return html_str
