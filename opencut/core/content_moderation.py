"""
OpenCut AI Content Moderation Scanner

Scans video content for potential issues: profanity in transcripts,
photosensitive flash sequences, loudness violations, and unexpected silence.

Uses FFmpeg and existing OpenCut modules -- no additional dependencies required.
"""

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Default profanity word list (basic, configurable)
# ---------------------------------------------------------------------------
_DEFAULT_PROFANITY_WORDS = [
    "fuck", "shit", "damn", "bitch", "ass", "bastard", "crap",
    "dick", "piss", "hell", "cock", "cunt", "whore", "slut",
]

# Broadcast loudness limits (EBU R128 / ATSC A/85)
_LOUDNESS_PEAK_LIMIT_DBTP = -1.0    # True peak ceiling
_LOUDNESS_MAX_INTEGRATED = -10.0     # Above this is likely too loud
_SILENCE_THRESHOLD_DB = -50.0        # dB threshold for silence
_SILENCE_MIN_DURATION = 5.0          # Seconds of silence to flag


@dataclass
class ModerationIssue:
    """A single content moderation issue found during scanning."""
    timestamp: float        # seconds into the file
    type: str               # "profanity", "flash", "loudness", "silence"
    severity: str           # "low", "medium", "high"
    description: str


@dataclass
class ModerationResult:
    """Complete content moderation scan results."""
    issues: List[ModerationIssue] = field(default_factory=list)
    overall_risk: str = "low"   # "low", "medium", "high"
    checks_performed: List[str] = field(default_factory=list)
    duration_analyzed: float = 0.0


def _load_profanity_words() -> List[str]:
    """Load profanity word list from user config or use default."""
    user_list = os.path.join(os.path.expanduser("~"), ".opencut", "profanity_list.txt")
    if os.path.isfile(user_list):
        try:
            with open(user_list, "r", encoding="utf-8") as f:
                words = [
                    w.strip().lower()
                    for w in f.readlines()
                    if w.strip() and not w.strip().startswith("#")
                ]
                if words:
                    return words
        except OSError:
            pass
    return list(_DEFAULT_PROFANITY_WORDS)


PROFANITY_WORDS = _load_profanity_words()


def scan_content(
    input_path: str,
    checks: Optional[List[str]] = None,
    transcript_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Scan video/audio content for moderation issues.

    Args:
        input_path: Source video or audio file.
        checks: List of checks to run: ["profanity", "flash", "loudness", "silence"].
                Defaults to all checks.
        transcript_path: Optional path to transcript file (SRT/TXT) for profanity check.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with issues list, overall_risk, checks_performed, duration_analyzed.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_checks = ["profanity", "flash", "loudness", "silence"]
    if checks is None:
        checks = list(all_checks)
    else:
        checks = [c.lower().strip() for c in checks if c.lower().strip() in all_checks]
        if not checks:
            checks = list(all_checks)

    info = get_video_info(input_path)
    duration = info.get("duration", 0.0)

    if on_progress:
        on_progress(5, f"Starting content scan ({len(checks)} checks)...")

    issues: List[ModerationIssue] = []
    step_pct = 80 // max(len(checks), 1)
    current_pct = 5

    # --- Profanity check ---
    if "profanity" in checks:
        if on_progress:
            on_progress(current_pct, "Scanning for profanity...")
        profanity_issues = _check_profanity(input_path, transcript_path)
        issues.extend(profanity_issues)
        current_pct += step_pct

    # --- Flash detection ---
    if "flash" in checks:
        if on_progress:
            on_progress(current_pct, "Scanning for flash sequences...")
        flash_issues = _check_flash(input_path)
        issues.extend(flash_issues)
        current_pct += step_pct

    # --- Loudness check ---
    if "loudness" in checks:
        if on_progress:
            on_progress(current_pct, "Checking loudness levels...")
        loudness_issues = _check_loudness(input_path)
        issues.extend(loudness_issues)
        current_pct += step_pct

    # --- Silence check ---
    if "silence" in checks:
        if on_progress:
            on_progress(current_pct, "Detecting unexpected silence...")
        silence_issues = _check_silence(input_path, duration)
        issues.extend(silence_issues)
        current_pct += step_pct

    # Determine overall risk
    if any(i.severity == "high" for i in issues):
        overall_risk = "high"
    elif any(i.severity == "medium" for i in issues):
        overall_risk = "medium"
    else:
        overall_risk = "low"

    if on_progress:
        on_progress(100, f"Content scan complete: {overall_risk} risk, {len(issues)} issues")

    result = ModerationResult(
        issues=issues,
        overall_risk=overall_risk,
        checks_performed=checks,
        duration_analyzed=duration,
    )
    return _result_to_dict(result)


def _check_profanity(input_path: str, transcript_path: Optional[str]) -> List[ModerationIssue]:
    """Check transcript for profanity words."""
    issues = []
    transcript_text = ""

    # Try loading transcript from explicit path
    if transcript_path and os.path.isfile(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        except OSError:
            pass

    # Try loading auto-generated transcript sidecar
    if not transcript_text:
        for ext in (".srt", ".txt", ".vtt"):
            sidecar = os.path.splitext(input_path)[0] + ext
            if os.path.isfile(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as f:
                        transcript_text = f.read()
                    break
                except OSError:
                    pass

    if not transcript_text:
        return issues

    words_set = set(PROFANITY_WORDS)
    lines = transcript_text.splitlines()

    # Parse SRT-style timestamps if available
    timestamp_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    current_time = 0.0

    for line in lines:
        ts_match = timestamp_pattern.search(line)
        if ts_match:
            h, m, s, ms = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3)), int(ts_match.group(4))
            current_time = h * 3600 + m * 60 + s + ms / 1000.0
            continue

        line_lower = line.lower()
        for word in words_set:
            # Word boundary match
            if re.search(rf"\b{re.escape(word)}\b", line_lower):
                issues.append(ModerationIssue(
                    timestamp=current_time,
                    type="profanity",
                    severity="medium",
                    description=f"Profanity detected: '{word}' at {current_time:.1f}s",
                ))

    return issues


def _check_flash(input_path: str) -> List[ModerationIssue]:
    """Detect flash sequences using existing accessibility module."""
    issues = []
    try:
        from opencut.core.accessibility import detect_flashing
        result = detect_flashing(input_path, max_flashes_per_sec=3, min_luminance_change=0.2)
        events = result.get("events", [])
        for event in events:
            severity_map = {"low": "low", "medium": "medium", "high": "high"}
            issues.append(ModerationIssue(
                timestamp=event["start"],
                type="flash",
                severity=severity_map.get(event.get("severity", "medium"), "medium"),
                description=(
                    f"Flash sequence at {event['start']:.1f}s-{event['end']:.1f}s: "
                    f"{event['flash_count']} flashes, "
                    f"peak luminance change {event['peak_luminance_change']:.2f}"
                ),
            ))
    except Exception as e:
        logger.warning("Flash detection failed: %s", e)
    return issues


def _check_loudness(input_path: str) -> List[ModerationIssue]:
    """Check audio loudness peaks against broadcast limits."""
    issues = []
    try:
        from opencut.core.audio_analysis import measure_loudness
        result = measure_loudness(input_path)

        # Check integrated loudness
        integrated = result.integrated_lufs
        if integrated > _LOUDNESS_MAX_INTEGRATED:
            issues.append(ModerationIssue(
                timestamp=0.0,
                type="loudness",
                severity="high" if integrated > -5.0 else "medium",
                description=(
                    f"Integrated loudness too high: {integrated:.1f} LUFS "
                    f"(limit: {_LOUDNESS_MAX_INTEGRATED:.1f} LUFS)"
                ),
            ))

        # Check true peak
        true_peak = result.true_peak_dbtp
        if true_peak > _LOUDNESS_PEAK_LIMIT_DBTP:
            issues.append(ModerationIssue(
                timestamp=0.0,
                type="loudness",
                severity="medium",
                description=(
                    f"True peak exceeds limit: {true_peak:.1f} dBTP "
                    f"(limit: {_LOUDNESS_PEAK_LIMIT_DBTP:.1f} dBTP)"
                ),
            ))
    except Exception as e:
        logger.warning("Loudness check failed: %s", e)
    return issues


def _check_silence(input_path: str, total_duration: float) -> List[ModerationIssue]:
    """Detect unexpected long silence segments."""
    issues = []
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-i", input_path,
        "-af", f"silencedetect=noise={_SILENCE_THRESHOLD_DB}dB:d={_SILENCE_MIN_DURATION}",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=300)
    except _sp.TimeoutExpired:
        logger.warning("Silence detection timed out for %s", input_path)
        return issues

    # Parse silence start/end from stderr
    # Lines: [silencedetect @ 0x...] silence_start: 12.345
    # Lines: [silencedetect @ 0x...] silence_end: 18.678 | silence_duration: 6.333
    silence_starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    silence_ends = re.findall(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", result.stderr)

    for i, end_match in enumerate(silence_ends):
        end_time = float(end_match[0])
        duration = float(end_match[1])
        start_time = float(silence_starts[i]) if i < len(silence_starts) else end_time - duration

        # Ignore silence at the very end (common in exports)
        if total_duration > 0 and start_time > total_duration * 0.95:
            continue

        severity = "high" if duration > 15.0 else ("medium" if duration > 8.0 else "low")
        issues.append(ModerationIssue(
            timestamp=start_time,
            type="silence",
            severity=severity,
            description=(
                f"Unexpected silence: {duration:.1f}s at {start_time:.1f}s-{end_time:.1f}s"
            ),
        ))

    return issues


def _result_to_dict(result: ModerationResult) -> dict:
    """Convert ModerationResult to a JSON-serializable dict."""
    return {
        "issues": [
            {
                "timestamp": i.timestamp,
                "type": i.type,
                "severity": i.severity,
                "description": i.description,
            }
            for i in result.issues
        ],
        "overall_risk": result.overall_risk,
        "checks_performed": result.checks_performed,
        "duration_analyzed": result.duration_analyzed,
        "total_issues": len(result.issues),
    }
