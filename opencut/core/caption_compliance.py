"""
OpenCut Caption Compliance Checker

Validates subtitle files against broadcast standards (Netflix, BBC, FCC, YouTube)
and provides auto-fix capabilities.

No additional dependencies required - pure Python SRT parsing.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Compliance Standards
# ---------------------------------------------------------------------------
STANDARDS = {
    "netflix": {
        "max_cpl": 42,
        "min_duration_ms": 200,
        "max_duration_ms": 7000,
        "max_cps": 20,
        "max_lines": 2,
        "min_gap_ms": 0,
        "max_wpm": None,
        "label": "Netflix Timed Text Style Guide",
    },
    "bbc": {
        "max_cpl": 37,
        "min_duration_ms": 300,
        "max_duration_ms": 7000,
        "max_cps": None,
        "max_lines": 2,
        "min_gap_ms": 300,
        "max_wpm": 160,
        "label": "BBC Subtitle Guidelines",
    },
    "fcc": {
        "max_cpl": 32,
        "min_duration_ms": 200,
        "max_duration_ms": 8000,
        "max_cps": None,
        "max_lines": 2,
        "min_gap_ms": 0,
        "max_wpm": None,
        "label": "FCC Closed Captioning Standards",
    },
    "youtube": {
        "max_cpl": 42,
        "min_duration_ms": 100,
        "max_duration_ms": 10000,
        "max_cps": 25,
        "max_lines": 2,
        "min_gap_ms": 0,
        "max_wpm": None,
        "label": "YouTube Subtitle Best Practices",
    },
}


@dataclass
class Violation:
    """A single compliance violation."""
    line_num: int
    start_time: float
    violation_type: str
    description: str
    severity: str = "error"       # "error", "warning", "info"
    fix_suggestion: str = ""


@dataclass
class ComplianceResult:
    """Complete compliance check results."""
    violations: List[Violation] = field(default_factory=list)
    pass_rate: float = 100.0
    overall_pass: bool = True
    standard: str = "netflix"
    total_subtitles: int = 0
    checked_rules: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SRT Parsing Utilities
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _parse_srt_time(s: str) -> float:
    """Parse SRT timestamp to seconds."""
    m = _TIME_RE.search(s)
    if not m:
        return 0.0
    h, mi, sec, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + sec + ms / 1000.0


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    remainder = seconds - h * 3600
    m = int(remainder // 60)
    s = remainder - m * 60
    sec = int(s)
    ms = int(round((s - sec) * 1000))
    if ms >= 1000:
        sec += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


@dataclass
class _Subtitle:
    """Internal subtitle representation."""
    index: int
    start: float
    end: float
    text: str
    raw_timing_line: str = ""


def _parse_srt(srt_path: str) -> List[_Subtitle]:
    """Parse an SRT file into subtitle objects."""
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: List[_Subtitle] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find timing line
        timing_line = None
        timing_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                timing_line = line.strip()
                timing_idx = i
                break

        if timing_line is None or timing_idx < 0:
            continue

        parts = timing_line.split("-->")
        if len(parts) != 2:
            continue

        start = _parse_srt_time(parts[0].strip())
        end = _parse_srt_time(parts[1].strip())

        text_lines = lines[timing_idx + 1:]
        text = "\n".join(line.strip() for line in text_lines if line.strip())

        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = len(subtitles) + 1

        subtitles.append(_Subtitle(
            index=idx, start=start, end=end,
            text=text, raw_timing_line=timing_line,
        ))

    return subtitles


def _write_srt(subtitles: List[_Subtitle], output: str) -> None:
    """Write subtitles to SRT file."""
    with open(output, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(sub.start)} --> {_format_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


def _strip_tags(text: str) -> str:
    """Remove HTML/SRT formatting tags."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# Compliance Checking
# ---------------------------------------------------------------------------
def check_caption_compliance(
    srt_path: str,
    standard: str = "netflix",
    on_progress: Optional[Callable] = None,
) -> ComplianceResult:
    """
    Check SRT subtitle file against a compliance standard.

    Args:
        srt_path: Path to the .srt file.
        standard: Standard name ("netflix", "bbc", "fcc", "youtube").
        on_progress: Progress callback(pct, msg).

    Returns:
        ComplianceResult with all violations found.
    """
    if standard not in STANDARDS:
        standard = "netflix"

    rules = STANDARDS[standard]
    subtitles = _parse_srt(srt_path)
    violations: List[Violation] = []
    checked_rules: List[str] = []

    total = len(subtitles)
    if total == 0:
        return ComplianceResult(
            standard=standard, total_subtitles=0,
            pass_rate=100.0, overall_pass=True,
        )

    if on_progress:
        on_progress(10, f"Checking {total} subtitles against {rules['label']}...")

    violated_indices = set()

    # --- Check: Characters Per Line (CPL) ---
    max_cpl = rules["max_cpl"]
    if max_cpl is not None:
        checked_rules.append("characters_per_line")
        for sub in subtitles:
            clean = _strip_tags(sub.text)
            for line in clean.split("\n"):
                if len(line) > max_cpl:
                    violations.append(Violation(
                        line_num=sub.index,
                        start_time=sub.start,
                        violation_type="characters_per_line",
                        description=f"Line has {len(line)} characters (max {max_cpl}): \"{line[:50]}...\"" if len(line) > 50 else f"Line has {len(line)} characters (max {max_cpl}): \"{line}\"",
                        severity="error",
                        fix_suggestion=f"Split line to stay under {max_cpl} characters",
                    ))
                    violated_indices.add(sub.index)

    if on_progress:
        on_progress(25, "Checking timing constraints...")

    # --- Check: Characters Per Second (CPS) ---
    max_cps = rules.get("max_cps")
    if max_cps is not None:
        checked_rules.append("chars_per_second")
        for sub in subtitles:
            dur = sub.end - sub.start
            if dur <= 0:
                continue
            char_count = len(_strip_tags(sub.text).replace("\n", ""))
            cps = char_count / dur
            if cps > max_cps:
                violations.append(Violation(
                    line_num=sub.index,
                    start_time=sub.start,
                    violation_type="chars_per_second",
                    description=f"CPS is {cps:.1f} (max {max_cps})",
                    severity="error",
                    fix_suggestion="Extend duration or reduce text",
                ))
                violated_indices.add(sub.index)

    # --- Check: Min Duration ---
    min_dur_ms = rules.get("min_duration_ms")
    if min_dur_ms is not None:
        checked_rules.append("min_duration")
        for sub in subtitles:
            dur_ms = (sub.end - sub.start) * 1000
            if dur_ms < min_dur_ms:
                violations.append(Violation(
                    line_num=sub.index,
                    start_time=sub.start,
                    violation_type="min_duration",
                    description=f"Duration {dur_ms:.0f}ms is below minimum {min_dur_ms}ms",
                    severity="error",
                    fix_suggestion=f"Extend to at least {min_dur_ms}ms",
                ))
                violated_indices.add(sub.index)

    if on_progress:
        on_progress(45, "Checking duration and line constraints...")

    # --- Check: Max Duration ---
    max_dur_ms = rules.get("max_duration_ms")
    if max_dur_ms is not None:
        checked_rules.append("max_duration")
        for sub in subtitles:
            dur_ms = (sub.end - sub.start) * 1000
            if dur_ms > max_dur_ms:
                violations.append(Violation(
                    line_num=sub.index,
                    start_time=sub.start,
                    violation_type="max_duration",
                    description=f"Duration {dur_ms:.0f}ms exceeds maximum {max_dur_ms}ms",
                    severity="warning",
                    fix_suggestion=f"Split into shorter subtitles (max {max_dur_ms}ms each)",
                ))
                violated_indices.add(sub.index)

    # --- Check: Max Lines ---
    max_lines = rules.get("max_lines")
    if max_lines is not None:
        checked_rules.append("max_lines")
        for sub in subtitles:
            clean = _strip_tags(sub.text)
            line_count = len(clean.strip().split("\n"))
            if line_count > max_lines:
                violations.append(Violation(
                    line_num=sub.index,
                    start_time=sub.start,
                    violation_type="max_lines",
                    description=f"Subtitle has {line_count} lines (max {max_lines})",
                    severity="error",
                    fix_suggestion=f"Reduce to {max_lines} lines or split",
                ))
                violated_indices.add(sub.index)

    if on_progress:
        on_progress(60, "Checking gaps and reading speed...")

    # --- Check: Subtitle Gap ---
    min_gap_ms = rules.get("min_gap_ms")
    if min_gap_ms and min_gap_ms > 0:
        checked_rules.append("subtitle_gap")
        for i in range(len(subtitles) - 1):
            gap_ms = (subtitles[i + 1].start - subtitles[i].end) * 1000
            if gap_ms < min_gap_ms and gap_ms >= 0:
                violations.append(Violation(
                    line_num=subtitles[i].index,
                    start_time=subtitles[i].end,
                    violation_type="subtitle_gap",
                    description=f"Gap to next subtitle is {gap_ms:.0f}ms (min {min_gap_ms}ms)",
                    severity="warning",
                    fix_suggestion=f"Increase gap to at least {min_gap_ms}ms",
                ))
                violated_indices.add(subtitles[i].index)

    # --- Check: Overlap ---
    checked_rules.append("overlap")
    for i in range(len(subtitles) - 1):
        if subtitles[i].end > subtitles[i + 1].start + 0.001:
            violations.append(Violation(
                line_num=subtitles[i].index,
                start_time=subtitles[i].end,
                violation_type="overlap",
                description=f"Subtitle overlaps with next (ends at {subtitles[i].end:.3f}, next starts at {subtitles[i + 1].start:.3f})",
                severity="error",
                fix_suggestion="Adjust timing to remove overlap",
            ))
            violated_indices.add(subtitles[i].index)

    # --- Check: Reading Speed (WPM) ---
    max_wpm = rules.get("max_wpm")
    if max_wpm is not None:
        checked_rules.append("reading_speed")
        for sub in subtitles:
            dur = sub.end - sub.start
            if dur <= 0:
                continue
            word_count = len(_strip_tags(sub.text).split())
            wpm = (word_count / dur) * 60
            if wpm > max_wpm:
                violations.append(Violation(
                    line_num=sub.index,
                    start_time=sub.start,
                    violation_type="reading_speed",
                    description=f"Reading speed is {wpm:.0f} WPM (max {max_wpm})",
                    severity="warning",
                    fix_suggestion="Extend duration or reduce word count",
                ))
                violated_indices.add(sub.index)

    if on_progress:
        on_progress(90, "Computing results...")

    # Calculate pass rate
    pass_rate = (1.0 - len(violated_indices) / total) * 100.0 if total > 0 else 100.0
    error_violations = [v for v in violations if v.severity == "error"]
    overall_pass = len(error_violations) == 0

    if on_progress:
        on_progress(100, f"Compliance check complete: {'PASS' if overall_pass else 'FAIL'}")

    return ComplianceResult(
        violations=violations,
        pass_rate=round(pass_rate, 1),
        overall_pass=overall_pass,
        standard=standard,
        total_subtitles=total,
        checked_rules=checked_rules,
    )


# ---------------------------------------------------------------------------
# Auto-Fix
# ---------------------------------------------------------------------------
def _find_break_point(text: str, max_len: int) -> int:
    """Find a natural break point in text, preferring spaces near the middle."""
    if len(text) <= max_len:
        return len(text)

    # Try to break at a space near the midpoint
    mid = len(text) // 2
    best = -1
    best_dist = len(text)

    for i, ch in enumerate(text):
        if ch == " " and i <= max_len:
            dist = abs(i - mid)
            if dist < best_dist:
                best = i
                best_dist = dist

    if best > 0:
        return best

    # Fallback: break at max_len
    return max_len


def auto_fix_compliance(
    srt_path: str,
    standard: str = "netflix",
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Auto-fix compliance violations in an SRT file.

    Applies fixes for:
    - Long lines (split at natural break points)
    - Duration violations (extend too-short, split too-long)
    - CPS violations (extend duration)

    Args:
        srt_path: Path to the .srt file.
        standard: Standard name.
        output_path_str: Output path (defaults to input_path with _fixed suffix).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, fixes_applied, and compliance_result.
    """
    if standard not in STANDARDS:
        standard = "netflix"

    rules = STANDARDS[standard]
    subtitles = _parse_srt(srt_path)
    fixes_applied = 0

    if not output_path_str:
        base, ext = os.path.splitext(srt_path)
        output_path_str = f"{base}_fixed{ext}"

    if on_progress:
        on_progress(10, f"Auto-fixing {len(subtitles)} subtitles...")

    # Fix 1: Split long lines
    max_cpl = rules["max_cpl"]
    if max_cpl:
        new_subs = []
        for sub in subtitles:
            clean = _strip_tags(sub.text)
            lines = clean.split("\n")
            fixed_lines = []
            for line in lines:
                if len(line) > max_cpl:
                    bp = _find_break_point(line, max_cpl)
                    fixed_lines.append(line[:bp].strip())
                    remainder = line[bp:].strip()
                    if remainder:
                        fixed_lines.append(remainder)
                    fixes_applied += 1
                else:
                    fixed_lines.append(line)

            # Enforce max_lines by splitting into multiple subtitles
            max_lines = rules.get("max_lines", 2)
            if len(fixed_lines) > max_lines:
                # Split into multiple subtitle blocks
                dur = sub.end - sub.start
                chunks = [fixed_lines[i:i + max_lines] for i in range(0, len(fixed_lines), max_lines)]
                chunk_dur = dur / len(chunks)
                for ci, chunk in enumerate(chunks):
                    new_sub = _Subtitle(
                        index=sub.index,
                        start=sub.start + ci * chunk_dur,
                        end=sub.start + (ci + 1) * chunk_dur,
                        text="\n".join(chunk),
                    )
                    new_subs.append(new_sub)
                fixes_applied += 1
            else:
                sub_copy = _Subtitle(
                    index=sub.index, start=sub.start, end=sub.end,
                    text="\n".join(fixed_lines),
                )
                new_subs.append(sub_copy)
        subtitles = new_subs

    if on_progress:
        on_progress(40, "Fixing timing violations...")

    # Fix 2: Min duration
    min_dur_ms = rules.get("min_duration_ms", 200)
    for sub in subtitles:
        dur_ms = (sub.end - sub.start) * 1000
        if dur_ms < min_dur_ms:
            sub.end = sub.start + min_dur_ms / 1000.0
            fixes_applied += 1

    # Fix 3: CPS violations - extend duration
    max_cps = rules.get("max_cps")
    if max_cps:
        for sub in subtitles:
            dur = sub.end - sub.start
            if dur <= 0:
                continue
            chars = len(_strip_tags(sub.text).replace("\n", ""))
            cps = chars / dur
            if cps > max_cps:
                needed_dur = chars / max_cps
                sub.end = sub.start + needed_dur
                fixes_applied += 1

    if on_progress:
        on_progress(60, "Fixing overlaps...")

    # Fix 4: Remove overlaps
    for i in range(len(subtitles) - 1):
        if subtitles[i].end > subtitles[i + 1].start:
            subtitles[i].end = subtitles[i + 1].start - 0.001
            fixes_applied += 1

    # Fix 5: Enforce min gap
    min_gap_ms = rules.get("min_gap_ms", 0)
    if min_gap_ms > 0:
        for i in range(len(subtitles) - 1):
            gap = (subtitles[i + 1].start - subtitles[i].end) * 1000
            if 0 <= gap < min_gap_ms:
                subtitles[i].end = subtitles[i + 1].start - min_gap_ms / 1000.0
                fixes_applied += 1

    if on_progress:
        on_progress(80, "Writing fixed subtitles...")

    # Write output
    _write_srt(subtitles, output_path_str)

    # Run compliance check on fixed output
    result = check_caption_compliance(output_path_str, standard)

    if on_progress:
        on_progress(100, f"Applied {fixes_applied} fixes, pass rate: {result.pass_rate}%")

    return {
        "output_path": output_path_str,
        "fixes_applied": fixes_applied,
        "remaining_violations": len(result.violations),
        "pass_rate": result.pass_rate,
        "overall_pass": result.overall_pass,
    }
