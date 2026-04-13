"""
OpenCut Broadcast QC / Standards Checker v1.0.0

Checks broadcast compliance for video files:
  - Audio levels: EBU R128 (loudness -23 LUFS) and ATSC A/85 (-24 LKFS)
  - Video levels: signalstats for IRE range (0-100 legal range)
  - Resolution, codec, container compliance
  - Closed caption presence
  - Generates a pass/fail QC report
"""

import json
import logging
import os
import subprocess as _sp
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class QCCheckResult:
    """Result of a single QC check."""
    name: str
    passed: bool
    value: str = ""
    expected: str = ""
    severity: str = "error"  # "error", "warning", "info"
    details: str = ""


@dataclass
class AudioLevelResult:
    """EBU R128 / ATSC A/85 loudness measurement."""
    integrated_loudness: float = 0.0
    loudness_range: float = 0.0
    true_peak: float = 0.0
    standard: str = "ebu_r128"
    target_loudness: float = -23.0
    tolerance: float = 1.0
    passed: bool = False
    checks: List[QCCheckResult] = field(default_factory=list)


@dataclass
class VideoLevelResult:
    """Video signal level measurements."""
    ymin: float = 0.0
    ymax: float = 255.0
    yavg: float = 128.0
    above_100_ire: bool = False
    below_0_ire: bool = False
    passed: bool = False
    checks: List[QCCheckResult] = field(default_factory=list)


@dataclass
class BroadcastQCReport:
    """Full broadcast QC report."""
    video_path: str = ""
    standard: str = "ebu_r128"
    overall_pass: bool = False
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    audio_levels: Optional[AudioLevelResult] = None
    video_levels: Optional[VideoLevelResult] = None
    checks: List[QCCheckResult] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Broadcast Standards Definitions
# ---------------------------------------------------------------------------

BROADCAST_STANDARDS = {
    "ebu_r128": {
        "label": "EBU R128 (European)",
        "target_loudness": -23.0,
        "loudness_tolerance": 1.0,
        "max_true_peak": -1.0,
        "max_loudness_range": 20.0,
    },
    "atsc_a85": {
        "label": "ATSC A/85 (North America)",
        "target_loudness": -24.0,
        "loudness_tolerance": 2.0,
        "max_true_peak": -2.0,
        "max_loudness_range": 20.0,
    },
    "arib_tr_b32": {
        "label": "ARIB TR-B32 (Japan)",
        "target_loudness": -24.0,
        "loudness_tolerance": 2.0,
        "max_true_peak": -1.0,
        "max_loudness_range": 20.0,
    },
}

VALID_BROADCAST_CODECS = {
    "video": ["h264", "hevc", "mpeg2video", "prores", "dnxhd"],
    "audio": ["aac", "ac3", "eac3", "pcm_s16le", "pcm_s24le", "mp2"],
}

VALID_BROADCAST_CONTAINERS = ["mov", "mxf", "mp4", "ts", "mpegts"]

BROADCAST_RESOLUTIONS = {
    "sd_ntsc": (720, 480),
    "sd_pal": (720, 576),
    "hd_720": (1280, 720),
    "hd_1080": (1920, 1080),
    "uhd_4k": (3840, 2160),
}


# ---------------------------------------------------------------------------
# Audio Level Check
# ---------------------------------------------------------------------------

def check_audio_levels(
    video_path: str,
    standard: str = "ebu_r128",
    on_progress: Optional[Callable] = None,
) -> AudioLevelResult:
    """Measure audio loudness against EBU R128 or ATSC A/85 standards.

    Uses FFmpeg's loudnorm filter in measurement mode to get
    integrated loudness, loudness range, and true peak.

    Args:
        video_path: Path to the video file.
        standard: Broadcast standard — "ebu_r128", "atsc_a85", "arib_tr_b32".
        on_progress: Optional callback (percent, message).

    Returns:
        AudioLevelResult with measurements and pass/fail status.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If standard is unknown.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    standard = standard.lower().strip()
    if standard not in BROADCAST_STANDARDS:
        raise ValueError(
            f"Unknown broadcast standard: {standard}. "
            f"Valid: {', '.join(BROADCAST_STANDARDS.keys())}"
        )

    spec = BROADCAST_STANDARDS[standard]
    result = AudioLevelResult(
        standard=standard,
        target_loudness=spec["target_loudness"],
        tolerance=spec["loudness_tolerance"],
    )

    if on_progress:
        on_progress(10, "Measuring audio loudness...")

    # Run loudnorm in measure-only mode
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", video_path,
        "-af", "loudnorm=print_format=json:I=-23:TP=-1:LRA=20",
        "-f", "null", os.devnull,
    ]

    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=300)
        stderr = proc.stderr
    except _sp.TimeoutExpired:
        result.checks.append(QCCheckResult(
            name="audio_loudness",
            passed=False,
            value="timeout",
            expected="measurement within 5 minutes",
            severity="error",
            details="FFmpeg loudnorm measurement timed out.",
        ))
        return result

    # Parse loudnorm JSON output from stderr
    integrated = None
    lra = None
    true_peak = None

    # Find the JSON block in stderr
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            loudnorm_data = json.loads(stderr[json_start:json_end])
            integrated = float(loudnorm_data.get("input_i", 0))
            lra = float(loudnorm_data.get("input_lra", 0))
            true_peak = float(loudnorm_data.get("input_tp", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if integrated is None:
        # Fallback: parse line-by-line
        for line in stderr.splitlines():
            line_stripped = line.strip().lower()
            if "input integrated:" in line_stripped or '"input_i"' in line_stripped:
                try:
                    integrated = float(line_stripped.split(":")[-1].strip().rstrip(",").strip('"'))
                except (ValueError, IndexError):
                    pass
            elif "input lra:" in line_stripped or '"input_lra"' in line_stripped:
                try:
                    lra = float(line_stripped.split(":")[-1].strip().rstrip(",").strip('"'))
                except (ValueError, IndexError):
                    pass
            elif "input true peak:" in line_stripped or '"input_tp"' in line_stripped:
                try:
                    true_peak = float(line_stripped.split(":")[-1].strip().rstrip(",").strip('"'))
                except (ValueError, IndexError):
                    pass

    result.integrated_loudness = integrated if integrated is not None else 0.0
    result.loudness_range = lra if lra is not None else 0.0
    result.true_peak = true_peak if true_peak is not None else 0.0

    if on_progress:
        on_progress(50, "Evaluating audio levels...")

    # Check integrated loudness
    target = spec["target_loudness"]
    tolerance = spec["loudness_tolerance"]
    loudness_ok = abs(result.integrated_loudness - target) <= tolerance
    result.checks.append(QCCheckResult(
        name="integrated_loudness",
        passed=loudness_ok,
        value=f"{result.integrated_loudness:.1f} LUFS",
        expected=f"{target:.1f} LUFS (±{tolerance:.1f})",
        severity="error" if not loudness_ok else "info",
        details=f"Target: {target} LUFS, measured: {result.integrated_loudness:.1f} LUFS",
    ))

    # Check true peak
    max_tp = spec["max_true_peak"]
    tp_ok = result.true_peak <= max_tp
    result.checks.append(QCCheckResult(
        name="true_peak",
        passed=tp_ok,
        value=f"{result.true_peak:.1f} dBTP",
        expected=f"≤ {max_tp:.1f} dBTP",
        severity="error" if not tp_ok else "info",
        details=f"Max allowed: {max_tp} dBTP, measured: {result.true_peak:.1f} dBTP",
    ))

    # Check loudness range
    max_lra = spec["max_loudness_range"]
    lra_ok = result.loudness_range <= max_lra
    result.checks.append(QCCheckResult(
        name="loudness_range",
        passed=lra_ok,
        value=f"{result.loudness_range:.1f} LU",
        expected=f"≤ {max_lra:.1f} LU",
        severity="warning" if not lra_ok else "info",
        details=f"Max recommended: {max_lra} LU, measured: {result.loudness_range:.1f} LU",
    ))

    result.passed = loudness_ok and tp_ok
    if on_progress:
        on_progress(70, "Audio level check complete.")

    return result


# ---------------------------------------------------------------------------
# Video Level Check
# ---------------------------------------------------------------------------

def check_video_levels(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> VideoLevelResult:
    """Check video signal levels using FFmpeg signalstats filter.

    Detects if luma values exceed broadcast legal range (16-235 for
    8-bit or 0-100 IRE).

    Args:
        video_path: Path to the video file.
        on_progress: Optional callback (percent, message).

    Returns:
        VideoLevelResult with min/max luma and pass/fail.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    result = VideoLevelResult()

    if on_progress:
        on_progress(10, "Analyzing video signal levels...")

    # Use signalstats to measure luma min/max/avg
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", video_path,
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print:key=lavfi.signalstats.YMIN:key=lavfi.signalstats.YMAX:key=lavfi.signalstats.YAVG",
        "-f", "null", os.devnull,
    ]

    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=600)
        stderr = proc.stderr
    except _sp.TimeoutExpired:
        result.checks.append(QCCheckResult(
            name="video_levels",
            passed=False,
            value="timeout",
            severity="error",
            details="Video signal analysis timed out.",
        ))
        return result

    # Parse signalstats output — gather min/max across all frames
    ymin_values = []
    ymax_values = []
    yavg_values = []

    for line in stderr.splitlines():
        stripped = line.strip()
        if "YMIN" in stripped:
            try:
                val = float(stripped.split("YMIN=")[-1].split()[0])
                ymin_values.append(val)
            except (ValueError, IndexError):
                pass
        if "YMAX" in stripped:
            try:
                val = float(stripped.split("YMAX=")[-1].split()[0])
                ymax_values.append(val)
            except (ValueError, IndexError):
                pass
        if "YAVG" in stripped:
            try:
                val = float(stripped.split("YAVG=")[-1].split()[0])
                yavg_values.append(val)
            except (ValueError, IndexError):
                pass

    if ymin_values:
        result.ymin = min(ymin_values)
    if ymax_values:
        result.ymax = max(ymax_values)
    if yavg_values:
        result.yavg = sum(yavg_values) / len(yavg_values)

    if on_progress:
        on_progress(60, "Evaluating video levels...")

    # 8-bit broadcast legal range: 16-235 luma
    # Values below 16 (0 IRE) or above 235 (100 IRE) are illegal
    LEGAL_MIN = 16
    LEGAL_MAX = 235

    result.below_0_ire = result.ymin < LEGAL_MIN
    result.above_100_ire = result.ymax > LEGAL_MAX

    min_ok = not result.below_0_ire
    result.checks.append(QCCheckResult(
        name="video_luma_min",
        passed=min_ok,
        value=f"YMIN={result.ymin:.0f}",
        expected=f"≥ {LEGAL_MIN} (0 IRE)",
        severity="warning" if not min_ok else "info",
        details=f"Minimum luma: {result.ymin:.0f} (legal min: {LEGAL_MIN})",
    ))

    max_ok = not result.above_100_ire
    result.checks.append(QCCheckResult(
        name="video_luma_max",
        passed=max_ok,
        value=f"YMAX={result.ymax:.0f}",
        expected=f"≤ {LEGAL_MAX} (100 IRE)",
        severity="warning" if not max_ok else "info",
        details=f"Maximum luma: {result.ymax:.0f} (legal max: {LEGAL_MAX})",
    ))

    result.passed = min_ok and max_ok

    if on_progress:
        on_progress(80, "Video level check complete.")

    return result


# ---------------------------------------------------------------------------
# Full Broadcast Standards Check
# ---------------------------------------------------------------------------

def _check_codec_compliance(video_path: str) -> List[QCCheckResult]:
    """Check video/audio codec and container compliance."""
    checks = []

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "stream=codec_name,codec_type",
        "-show_entries", "format=format_name",
        "-of", "json", video_path,
    ]

    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, _sp.TimeoutExpired, OSError):
        checks.append(QCCheckResult(
            name="codec_probe",
            passed=False,
            severity="error",
            details="Failed to probe file codec information.",
        ))
        return checks

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Check container
    format_name = fmt.get("format_name", "").lower()
    format_parts = [f.strip() for f in format_name.split(",")]
    container_ok = any(f in VALID_BROADCAST_CONTAINERS for f in format_parts)
    checks.append(QCCheckResult(
        name="container_format",
        passed=container_ok,
        value=format_name,
        expected=", ".join(VALID_BROADCAST_CONTAINERS),
        severity="error" if not container_ok else "info",
        details=f"Container: {format_name}",
    ))

    # Check video codec
    video_codec = ""
    audio_codec = ""
    for stream in streams:
        if stream.get("codec_type") == "video" and not video_codec:
            video_codec = stream.get("codec_name", "").lower()
        elif stream.get("codec_type") == "audio" and not audio_codec:
            audio_codec = stream.get("codec_name", "").lower()

    if video_codec:
        vc_ok = video_codec in VALID_BROADCAST_CODECS["video"]
        checks.append(QCCheckResult(
            name="video_codec",
            passed=vc_ok,
            value=video_codec,
            expected=", ".join(VALID_BROADCAST_CODECS["video"]),
            severity="error" if not vc_ok else "info",
            details=f"Video codec: {video_codec}",
        ))
    else:
        checks.append(QCCheckResult(
            name="video_codec",
            passed=False,
            value="none",
            severity="error",
            details="No video stream found.",
        ))

    if audio_codec:
        ac_ok = audio_codec in VALID_BROADCAST_CODECS["audio"]
        checks.append(QCCheckResult(
            name="audio_codec",
            passed=ac_ok,
            value=audio_codec,
            expected=", ".join(VALID_BROADCAST_CODECS["audio"]),
            severity="warning" if not ac_ok else "info",
            details=f"Audio codec: {audio_codec}",
        ))
    else:
        checks.append(QCCheckResult(
            name="audio_codec",
            passed=False,
            value="none",
            severity="warning",
            details="No audio stream found.",
        ))

    return checks


def _check_resolution(video_path: str) -> QCCheckResult:
    """Check if resolution matches a standard broadcast resolution."""
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]

    matched = None
    for res_name, (rw, rh) in BROADCAST_RESOLUTIONS.items():
        if w == rw and h == rh:
            matched = res_name
            break

    is_ok = matched is not None
    return QCCheckResult(
        name="resolution",
        passed=is_ok,
        value=f"{w}x{h}",
        expected=", ".join(f"{rw}x{rh}" for rw, rh in BROADCAST_RESOLUTIONS.values()),
        severity="warning" if not is_ok else "info",
        details=f"Resolution: {w}x{h}" + (f" ({matched})" if matched else " (non-standard)"),
    )


def _check_captions(video_path: str) -> QCCheckResult:
    """Check for closed caption tracks in the file."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "json", video_path,
    ]

    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, _sp.TimeoutExpired, OSError):
        return QCCheckResult(
            name="closed_captions",
            passed=False,
            severity="warning",
            details="Could not probe for caption tracks.",
        )

    streams = data.get("streams", [])
    caption_streams = [
        s for s in streams
        if s.get("codec_type") == "subtitle"
        or s.get("codec_name", "").lower() in (
            "eia_608", "cea_608", "cea_708", "mov_text",
            "subrip", "ass", "webvtt", "dvb_subtitle",
        )
    ]

    has_captions = len(caption_streams) > 0
    return QCCheckResult(
        name="closed_captions",
        passed=has_captions,
        value=f"{len(caption_streams)} track(s)" if has_captions else "none",
        expected="≥ 1 caption track",
        severity="warning" if not has_captions else "info",
        details=f"Found {len(caption_streams)} caption/subtitle track(s).",
    )


def check_broadcast_standards(
    video_path: str,
    standard: str = "ebu_r128",
    check_audio: bool = True,
    check_video: bool = True,
    check_codecs: bool = True,
    check_resolution_flag: bool = True,
    check_captions_flag: bool = True,
    on_progress: Optional[Callable] = None,
) -> BroadcastQCReport:
    """Run a full broadcast QC check on a video file.

    Args:
        video_path: Path to the video file.
        standard: Broadcast standard — "ebu_r128", "atsc_a85", "arib_tr_b32".
        check_audio: Run audio level checks.
        check_video: Run video level checks.
        check_codecs: Run codec/container compliance checks.
        check_resolution_flag: Check resolution compliance.
        check_captions_flag: Check for closed caption presence.
        on_progress: Optional callback (percent, message).

    Returns:
        BroadcastQCReport with all check results.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    start_time = time.time()
    report = BroadcastQCReport(video_path=video_path, standard=standard)
    all_checks: List[QCCheckResult] = []

    step = 0
    total_steps = sum([check_audio, check_video, check_codecs,
                       check_resolution_flag, check_captions_flag])
    if total_steps == 0:
        total_steps = 1

    def _step_progress(msg):
        nonlocal step
        step += 1
        pct = min(int(step / total_steps * 90) + 5, 95)
        if on_progress:
            on_progress(pct, msg)

    if check_audio:
        _step_progress("Checking audio levels...")
        audio_result = check_audio_levels(video_path, standard)
        report.audio_levels = audio_result
        all_checks.extend(audio_result.checks)

    if check_video:
        _step_progress("Checking video levels...")
        video_result = check_video_levels(video_path)
        report.video_levels = video_result
        all_checks.extend(video_result.checks)

    if check_codecs:
        _step_progress("Checking codec compliance...")
        codec_checks = _check_codec_compliance(video_path)
        all_checks.extend(codec_checks)

    if check_resolution_flag:
        _step_progress("Checking resolution...")
        res_check = _check_resolution(video_path)
        all_checks.append(res_check)

    if check_captions_flag:
        _step_progress("Checking closed captions...")
        cc_check = _check_captions(video_path)
        all_checks.append(cc_check)

    report.checks = all_checks
    report.total_checks = len(all_checks)
    report.passed_checks = sum(1 for c in all_checks if c.passed)
    report.failed_checks = sum(
        1 for c in all_checks
        if not c.passed and c.severity == "error"
    )
    report.warning_checks = sum(
        1 for c in all_checks
        if not c.passed and c.severity == "warning"
    )
    report.overall_pass = report.failed_checks == 0
    report.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        status = "PASS" if report.overall_pass else "FAIL"
        on_progress(100, f"Broadcast QC complete — {status}")

    return report


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_qc_report(
    results: BroadcastQCReport,
    output_path: Optional[str] = None,
    format: str = "json",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a formatted QC report from check results.

    Args:
        results: BroadcastQCReport from check_broadcast_standards().
        output_path: Optional path to write report file.
        format: Report format — "json" or "text".
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with report content and metadata.
    """
    if on_progress:
        on_progress(10, "Generating QC report...")

    # Build report dict
    report_data = {
        "video_path": results.video_path,
        "standard": results.standard,
        "overall_pass": results.overall_pass,
        "summary": {
            "total_checks": results.total_checks,
            "passed": results.passed_checks,
            "failed": results.failed_checks,
            "warnings": results.warning_checks,
        },
        "checks": [asdict(c) for c in results.checks],
        "duration_seconds": results.duration_seconds,
    }

    if results.audio_levels:
        report_data["audio_levels"] = {
            "integrated_loudness": results.audio_levels.integrated_loudness,
            "loudness_range": results.audio_levels.loudness_range,
            "true_peak": results.audio_levels.true_peak,
            "standard": results.audio_levels.standard,
            "passed": results.audio_levels.passed,
        }

    if results.video_levels:
        report_data["video_levels"] = {
            "ymin": results.video_levels.ymin,
            "ymax": results.video_levels.ymax,
            "yavg": results.video_levels.yavg,
            "above_100_ire": results.video_levels.above_100_ire,
            "below_0_ire": results.video_levels.below_0_ire,
            "passed": results.video_levels.passed,
        }

    output_file = None

    if format == "text":
        lines = [
            "=== Broadcast QC Report ===",
            f"File: {results.video_path}",
            f"Standard: {results.standard}",
            f"Overall: {'PASS' if results.overall_pass else 'FAIL'}",
            f"Checks: {results.passed_checks}/{results.total_checks} passed, "
            f"{results.failed_checks} failed, {results.warning_checks} warnings",
            "",
        ]
        for check in results.checks:
            status = "PASS" if check.passed else ("WARN" if check.severity == "warning" else "FAIL")
            lines.append(f"[{status}] {check.name}: {check.value} (expected: {check.expected})")
            if check.details:
                lines.append(f"       {check.details}")
        report_text = "\n".join(lines)
        report_data["report_text"] = report_text

        if output_path:
            if not output_path.endswith(".txt"):
                output_path += ".txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            output_file = output_path
    else:
        if output_path:
            if not output_path.endswith(".json"):
                output_path += ".json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            output_file = output_path

    if on_progress:
        on_progress(100, "QC report generated.")

    return {
        "report": report_data,
        "output_path": output_file,
        "overall_pass": results.overall_pass,
        "format": format,
    }
