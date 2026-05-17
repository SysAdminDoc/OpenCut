"""Caption QC gate (F111).

A focused quality gate that wraps :mod:`opencut.core.caption_compliance`
with stricter defaults and adds two checks that aren't part of any
broadcast spec but should still fail a release:

* **Forbidden glyphs** — control characters, replacement chars (``U+FFFD``),
  zero-width spaces, and stray non-breaking BOMs that creep in through
  copy/paste from web sources.
* **Cue overlap** — two cues whose time ranges overlap. The compliance
  module catches this through ``min_gap_ms`` only when gap < 0; we make
  the rule explicit so the diagnostic message reads cleanly.

The gate is callable from both Python (``qc_captions``) and HTTP
(``POST /captions/qc``). It is also wired into the caption export
helpers so production exports fail closed on a hard QC failure unless
the caller explicitly opts in to ``force=True``.

Design choices:

* No new pip dependency. We re-use the existing SRT parser inside
  ``caption_compliance`` instead of adding ``pysrt`` / ``webvtt-py``.
* Diagnostic objects mirror the panel-facing shape: ``rule``,
  ``severity``, ``cue_index``, ``line_num``, ``start_time``, ``message``.
* Two pass modes: ``"strict"`` (release defaults) and ``"advisory"``
  (treat reading-speed and CPL above the strict cap as warnings rather
  than errors).
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from opencut.core.caption_compliance import (
    STANDARDS,
    Violation,
    _parse_srt,
    _strip_tags,
    check_caption_compliance,
)
from opencut.core.caption_reading_profiles import (
    get_reading_speed_profile,
    normalize_reading_profile,
    reading_profile_rule_overrides,
)

# ---------------------------------------------------------------------------
# Forbidden-glyph rule
# ---------------------------------------------------------------------------

# Captions almost never legitimately need these. The list is conservative —
# emoji and standard punctuation are not included.
_FORBIDDEN_GLYPHS = {
    "﻿": "BOM (U+FEFF) — strip before exporting",
    "​": "zero-width space (U+200B)",
    "‌": "zero-width non-joiner (U+200C)",
    "‍": "zero-width joiner (U+200D)",
    "�": "Unicode replacement char (U+FFFD) — usually a decoding error",
}

# Control chars except newline / tab.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass
class QcDiagnostic:
    """A single diagnostic emitted by the caption QC gate."""

    rule: str
    severity: str  # "error" | "warning" | "info"
    cue_index: int
    line_num: int
    start_time: float
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class QcResult:
    """Result of running the caption QC gate."""

    overall_pass: bool
    standard: str
    mode: str
    total_cues: int
    error_count: int
    warning_count: int
    reading_profile: Optional[str] = None
    reading_profile_label: Optional[str] = None
    reading_profile_source_confidence: Optional[str] = None
    diagnostics: List[QcDiagnostic] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = {
            "overall_pass": self.overall_pass,
            "standard": self.standard,
            "mode": self.mode,
            "total_cues": self.total_cues,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }
        if self.reading_profile:
            payload.update(
                {
                    "reading_profile": self.reading_profile,
                    "reading_profile_label": self.reading_profile_label,
                    "reading_profile_source_confidence": (
                        self.reading_profile_source_confidence
                    ),
                }
            )
        return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _violation_to_diagnostic(violation: Violation, mode: str) -> QcDiagnostic:
    severity = violation.severity
    if mode == "advisory" and violation.violation_type in {
        "characters_per_line",
        "chars_per_second",
        "reading_speed",
        "max_cpl_exceeded",
        "max_cps_exceeded",
        "max_wpm_exceeded",
    }:
        severity = "warning"
    return QcDiagnostic(
        rule=violation.violation_type,
        severity=severity,
        cue_index=violation.line_num,
        line_num=violation.line_num,
        start_time=violation.start_time,
        message=violation.description,
    )


def _scan_glyphs(subtitles) -> List[QcDiagnostic]:
    out: List[QcDiagnostic] = []
    for sub in subtitles:
        text = sub.text or ""
        stripped = _strip_tags(text)
        for glyph, desc in _FORBIDDEN_GLYPHS.items():
            if glyph in stripped:
                out.append(
                    QcDiagnostic(
                        rule="forbidden_glyph",
                        severity="error",
                        cue_index=sub.index,
                        line_num=sub.index,
                        start_time=sub.start,
                        message=f"forbidden glyph detected: {desc}",
                    )
                )
        match = _CONTROL_RE.search(stripped)
        if match:
            out.append(
                QcDiagnostic(
                    rule="control_character",
                    severity="error",
                    cue_index=sub.index,
                    line_num=sub.index,
                    start_time=sub.start,
                    message=(
                        f"control character U+{ord(match.group(0)):04X} at cue {sub.index}"
                    ),
                )
            )
    return out


def _scan_overlaps(subtitles) -> List[QcDiagnostic]:
    out: List[QcDiagnostic] = []
    prev = None
    for sub in subtitles:
        if prev is not None and sub.start < prev.end:
            out.append(
                QcDiagnostic(
                    rule="cue_overlap",
                    severity="error",
                    cue_index=sub.index,
                    line_num=sub.index,
                    start_time=sub.start,
                    message=(
                        f"cue {sub.index} starts at {sub.start:.3f}s before cue "
                        f"{prev.index} ends at {prev.end:.3f}s"
                    ),
                )
            )
        prev = sub
    return out


def _summarise(diagnostics: Iterable[QcDiagnostic]) -> tuple:
    errors = 0
    warnings = 0
    for d in diagnostics:
        if d.severity == "error":
            errors += 1
        elif d.severity == "warning":
            warnings += 1
    return errors, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def qc_captions(
    srt_path: Optional[str] = None,
    srt_text: Optional[str] = None,
    *,
    standard: str = "accessibility",
    mode: str = "strict",
    reading_profile: Optional[str] = None,
) -> QcResult:
    """Run the caption QC gate against an SRT file or inline string.

    Exactly one of ``srt_path`` / ``srt_text`` must be supplied. ``mode``
    is ``"strict"`` (default; release gate) or ``"advisory"`` (downgrades
    reading-speed/CPL violations to warnings).
    """
    if standard not in STANDARDS:
        raise ValueError(f"unknown caption standard: {standard!r}; choose from {sorted(STANDARDS)}")
    if mode not in {"strict", "advisory"}:
        raise ValueError(f"unknown QC mode: {mode!r}; expected 'strict' or 'advisory'")
    if (srt_path is None) == (srt_text is None):
        raise ValueError("supply exactly one of srt_path / srt_text")

    profile_key = normalize_reading_profile(reading_profile)
    profile_meta = get_reading_speed_profile(profile_key) if profile_key else None
    rule_overrides = reading_profile_rule_overrides(profile_key) if profile_key else None
    standard_label = None
    if profile_meta:
        standard_label = f"{STANDARDS[standard]['label']} + {profile_meta['label']}"

    cleanup_path: Optional[str] = None
    try:
        if srt_text is not None:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".srt",
                delete=False,
            )
            tmp.write(srt_text)
            tmp.flush()
            tmp.close()
            srt_path = tmp.name
            cleanup_path = tmp.name

        compliance = check_caption_compliance(
            srt_path,
            standard=standard,
            rule_overrides=rule_overrides,
            standard_label=standard_label,
        )
        subtitles = _parse_srt(srt_path)

        diagnostics: List[QcDiagnostic] = []
        for v in compliance.violations:
            diagnostics.append(_violation_to_diagnostic(v, mode))
        diagnostics.extend(_scan_glyphs(subtitles))
        diagnostics.extend(_scan_overlaps(subtitles))

        errors, warnings = _summarise(diagnostics)
        overall_pass = errors == 0
        return QcResult(
            overall_pass=overall_pass,
            standard=standard,
            mode=mode,
            total_cues=len(subtitles),
            error_count=errors,
            warning_count=warnings,
            reading_profile=profile_key,
            reading_profile_label=profile_meta["label"] if profile_meta else None,
            reading_profile_source_confidence=(
                profile_meta["source_confidence"] if profile_meta else None
            ),
            diagnostics=diagnostics,
        )
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass


def enforce_export_gate(
    srt_path: str,
    *,
    standard: str = "accessibility",
    reading_profile: Optional[str] = None,
    force: bool = False,
) -> QcResult:
    """Hook for export routes — raises :class:`ValueError` on failure.

    Caller chooses whether to surface the diagnostics; this wrapper just
    makes "fail closed unless ``force=True``" trivial to add at any
    export site.
    """
    result = qc_captions(
        srt_path=srt_path,
        standard=standard,
        mode="strict",
        reading_profile=reading_profile,
    )
    if not result.overall_pass and not force:
        raise CaptionQcFailure(result)
    return result


class CaptionQcFailure(Exception):
    """Raised by :func:`enforce_export_gate` when the QC gate fails."""

    def __init__(self, result: QcResult) -> None:
        super().__init__(
            f"Caption QC failed ({result.error_count} errors). "
            "Pass force=True to override."
        )
        self.result = result
