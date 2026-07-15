"""
Caption export preflight and fallback strategy selection.

Checks caption readiness before host export and recommends a fallback
strategy (native, srt_sidecar, or burnin) when native export is risky.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("opencut")

KNOWN_SAFE_HOST_VERSIONS = frozenset()
KNOWN_RISKY_HOST_VERSIONS = frozenset({"26.0", "26.0.1"})


@dataclass
class PreflightDiagnostic:
    check: str
    status: str  # "ok", "warning", "error"
    message: str


@dataclass
class CaptionPreflightResult:
    ready: bool = True
    fallback_strategy: str = "native"  # "native", "srt_sidecar", "burnin"
    diagnostics: List[PreflightDiagnostic] = field(default_factory=list)
    caption_count: int = 0
    qc_passed: bool = True
    qc_warnings: List[str] = field(default_factory=list)
    conformance_profile: str = ""
    conformance_valid: bool = True

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def keys(self):
        return self.__dataclass_fields__.keys()


def check_segments_valid(segments, video_duration=0.0):
    """Validate caption segments are non-empty and within video duration."""
    diags = []
    if not segments:
        diags.append(PreflightDiagnostic(
            check="segments_exist",
            status="error",
            message="No caption segments provided.",
        ))
        return diags, False

    diags.append(PreflightDiagnostic(
        check="segments_exist",
        status="ok",
        message=f"{len(segments)} caption segment(s) found.",
    ))

    if video_duration > 0:
        out_of_range = 0
        for seg in segments:
            end = float(seg.get("end", 0))
            if end > video_duration * 1.1:
                out_of_range += 1
        if out_of_range > 0:
            diags.append(PreflightDiagnostic(
                check="timecode_range",
                status="warning",
                message=f"{out_of_range} segment(s) extend beyond video duration ({video_duration:.1f}s).",
            ))

    return diags, True


def check_host_compatibility(host_version=None):
    """Check if the host Premiere version supports native caption tracks safely."""
    diags = []
    if not host_version:
        diags.append(PreflightDiagnostic(
            check="host_version",
            status="warning",
            message="Host version unknown; native caption export may fail. SRT sidecar fallback recommended.",
        ))
        return diags, "srt_sidecar"

    version_key = str(host_version).strip()

    if version_key in KNOWN_RISKY_HOST_VERSIONS:
        diags.append(PreflightDiagnostic(
            check="host_version",
            status="warning",
            message=f"Premiere {version_key} has known caption export issues. Burn-in or SRT sidecar recommended.",
        ))
        return diags, "srt_sidecar"

    diags.append(PreflightDiagnostic(
        check="host_version",
        status="ok",
        message=f"Premiere {version_key} — native caption export supported.",
    ))
    return diags, "native"


def run_caption_export_preflight(
    segments=None,
    srt_text=None,
    video_duration=0.0,
    host_version=None,
    force_strategy=None,
    target_profile=None,
    language="en",
) -> CaptionPreflightResult:
    """Run all caption export preflight checks.

    Args:
        segments: List of caption segment dicts ({start, end, text}).
        srt_text: Raw SRT text (alternative to segments).
        video_duration: Source video duration in seconds.
        host_version: Premiere Pro version string from the panel.
        force_strategy: Override fallback strategy ("native", "srt_sidecar", "burnin").
        target_profile: Optional XML target ("ttml", "imsc1", or "imsc1.3").
        language: BCP 47 language tag used for XML conformance preflight.

    Returns:
        CaptionPreflightResult with readiness, strategy, and diagnostics.
    """
    result = CaptionPreflightResult()
    all_diags = []

    if segments:
        seg_diags, seg_ok = check_segments_valid(segments, video_duration)
        all_diags.extend(seg_diags)
        result.caption_count = len(segments)
        if not seg_ok:
            result.ready = False
    elif srt_text:
        lines = [ln for ln in srt_text.strip().splitlines() if ln.strip()]
        result.caption_count = sum(1 for ln in lines if "-->" in ln)
        if result.caption_count == 0:
            all_diags.append(PreflightDiagnostic(
                check="srt_parse",
                status="error",
                message="SRT text contains no valid cues.",
            ))
            result.ready = False
        else:
            all_diags.append(PreflightDiagnostic(
                check="srt_parse",
                status="ok",
                message=f"{result.caption_count} SRT cue(s) found.",
            ))
    else:
        all_diags.append(PreflightDiagnostic(
            check="input",
            status="error",
            message="No segments or SRT text provided.",
        ))
        result.ready = False

    if srt_text and result.ready:
        try:
            from .caption_qc import qc_captions
            qc = qc_captions(srt_text=srt_text, mode="advisory")
            result.qc_passed = qc.passed
            result.qc_warnings = [v.message for v in (qc.violations or []) if v.severity == "warning"]
            status = "ok" if qc.passed else "warning"
            all_diags.append(PreflightDiagnostic(
                check="qc_gate",
                status=status,
                message=f"QC {status}: {len(result.qc_warnings)} warning(s).",
            ))
        except Exception as exc:
            all_diags.append(PreflightDiagnostic(
                check="qc_gate",
                status="warning",
                message=f"QC check skipped: {exc}",
            ))

    if target_profile:
        try:
            from .caption_interchange import (
                document_from_items,
                normalize_profile,
                serialize_caption_document,
                validate_ttml,
            )

            normalized_profile = normalize_profile(target_profile)
            if normalized_profile == "ebu_tt":
                raise ValueError("Use the dedicated EBU-TT export for the EBU profile.")
            result.conformance_profile = normalized_profile
            if not segments:
                all_diags.append(PreflightDiagnostic(
                    check="xml_conformance",
                    status="warning",
                    message="XML conformance preflight requires structured segments.",
                ))
            else:
                document = document_from_items(segments, language=str(language or "en"))
                payload = serialize_caption_document(document, normalized_profile)
                report = validate_ttml(payload, expected_profile=normalized_profile)
                result.conformance_valid = report.valid
                all_diags.append(PreflightDiagnostic(
                    check="xml_conformance",
                    status="ok" if report.valid else "error",
                    message=(
                        f"{normalized_profile} conformance passed for {report.cue_count} cue(s)."
                        if report.valid
                        else "; ".join(issue.message for issue in report.errors)
                    ),
                ))
                if not report.valid:
                    result.ready = False
        except (TypeError, ValueError) as exc:
            result.conformance_valid = False
            result.ready = False
            all_diags.append(PreflightDiagnostic(
                check="xml_conformance",
                status="error",
                message=f"XML conformance preflight failed: {exc}",
            ))

    host_diags, host_strategy = check_host_compatibility(host_version)
    all_diags.extend(host_diags)

    if force_strategy and force_strategy in ("native", "srt_sidecar", "burnin"):
        result.fallback_strategy = force_strategy
    elif not result.ready:
        result.fallback_strategy = "burnin"
    else:
        result.fallback_strategy = host_strategy

    result.diagnostics = all_diags
    return result
