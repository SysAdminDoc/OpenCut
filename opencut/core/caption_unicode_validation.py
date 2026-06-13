"""Unicode caption validation fixtures and export round-trip checks (F223).

This module keeps complex-script caption coverage deterministic and cheap:
it does not render pixels or download fonts. Instead it proves OpenCut's
caption text survives every committed text export path that feeds the shaped
renderer:

* UTF-8 SRT export.
* ASS export.
* The temporary ASS file generated before FFmpeg/libass burn-in.

FFmpeg/libass HarfBuzz/FriBidi linkage is enforced separately by
``opencut.tools.text_shaping_gate``. This gate catches regressions in OpenCut's
own text handling before text reaches that renderer.
"""

from __future__ import annotations

import io
import json
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from opencut.core.caption_burnin import _write_ass_file as _write_burnin_ass_file
from opencut.core.caption_line_breaks import all_lines_within, wrap_caption_text
from opencut.core.captions import CaptionSegment, TranscriptionResult
from opencut.export.srt import export_ass, export_srt, has_utf8_bom

RTL_BIDI_CLASSES = {"R", "AL", "AN"}
BIDI_CONTROL_CODEPOINTS = {
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}


@dataclass(frozen=True)
class CaptionScriptCase:
    """One complex-script caption fixture for export validation."""

    case_id: str
    language: str
    text: str
    expected_scripts: Sequence[str]
    requires_complex_shaping: bool
    note: str


@dataclass
class CaptionScriptReport:
    """Per-fixture validation result."""

    case_id: str
    language: str
    scripts: List[str]
    expected_scripts: List[str]
    requires_complex_shaping: bool
    srt_roundtrip: bool
    ass_roundtrip: bool
    burnin_ass_roundtrip: bool
    utf8_without_bom: bool
    cjk_line_break_required: bool
    cjk_line_break_supported: bool
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "fail" if self.failures else "ok"

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status
        return payload


DEFAULT_SCRIPT_CASES: Sequence[CaptionScriptCase] = (
    CaptionScriptCase(
        case_id="arabic_rtl",
        language="ar",
        text="مرحبا بكم في OpenCut",
        expected_scripts=("rtl",),
        requires_complex_shaping=True,
        note="Arabic requires FriBidi ordering plus contextual glyph shaping.",
    ),
    CaptionScriptCase(
        case_id="hebrew_latin_bidi",
        language="he",
        text="OpenCut גרסה 1.32 מוכנה",
        expected_scripts=("rtl", "latin", "mixed_bidi"),
        requires_complex_shaping=True,
        note="Mixed Hebrew, Latin, and numbers exercise bidi run ordering.",
    ),
    CaptionScriptCase(
        case_id="hindi_devanagari",
        language="hi",
        text="नमस्ते दुनिया",
        expected_scripts=("indic",),
        requires_complex_shaping=True,
        note="Devanagari conjuncts and matras require a shaping-capable renderer.",
    ),
    CaptionScriptCase(
        case_id="japanese_no_space",
        language="ja",
        text="これは改行検証用の長い字幕テキストです",
        expected_scripts=("cjk",),
        requires_complex_shaping=False,
        note="No-space Japanese should survive export; F242 handles line breaks.",
    ),
    CaptionScriptCase(
        case_id="chinese_no_space",
        language="zh",
        text="开放剪辑字幕渲染验证需要处理没有空格的长文本",
        expected_scripts=("cjk",),
        requires_complex_shaping=False,
        note="No-space Chinese should survive export; F242 handles line breaks.",
    ),
    CaptionScriptCase(
        case_id="bengali_indic",
        language="bn",
        text="ওপেনকাট ক্যাপশন পরীক্ষা",
        expected_scripts=("indic",),
        requires_complex_shaping=True,
        note="Bengali conjuncts and vowel signs require a shaping-capable renderer.",
    ),
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
    )


def _is_indic(char: str) -> bool:
    code = ord(char)
    return (
        0x0900 <= code <= 0x097F
        or 0x0980 <= code <= 0x09FF
        or 0x0A00 <= code <= 0x0A7F
        or 0x0A80 <= code <= 0x0AFF
        or 0x0B00 <= code <= 0x0B7F
        or 0x0B80 <= code <= 0x0BFF
        or 0x0C00 <= code <= 0x0C7F
        or 0x0C80 <= code <= 0x0CFF
        or 0x0D00 <= code <= 0x0D7F
    )


def classify_caption_text(text: str) -> List[str]:
    """Classify caption text into script/layout families relevant to captions."""
    scripts = set()
    has_ltr = False
    has_rtl = False

    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi in RTL_BIDI_CLASSES:
            has_rtl = True
            scripts.add("rtl")
        elif bidi == "L" and char.isalpha():
            has_ltr = True

        if "LATIN" in unicodedata.name(char, ""):
            scripts.add("latin")
        if _is_cjk(char):
            scripts.add("cjk")
        if _is_indic(char):
            scripts.add("indic")
        if ord(char) in BIDI_CONTROL_CODEPOINTS:
            scripts.add("bidi_control")

    if has_ltr and has_rtl:
        scripts.add("mixed_bidi")

    return sorted(scripts)


def needs_cjk_line_breaker(text: str, *, max_line_length: int = 14) -> bool:
    """Return True when a CJK cue is long and has no whitespace breakpoints."""
    if not any(_is_cjk(char) for char in text):
        return False
    if any(char.isspace() for char in text):
        return False
    return len(text) > max_line_length


def _result_for_case(case: CaptionScriptCase) -> TranscriptionResult:
    segment = CaptionSegment(
        text=case.text,
        start=0.0,
        end=3.0,
        language=case.language,
    )
    return TranscriptionResult(
        segments=[segment],
        language=case.language,
        duration=3.0,
    )


def _roundtrip_srt(case: CaptionScriptCase) -> tuple[bool, bool]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{case.case_id}.srt"
        export_srt(_result_for_case(case), str(path))
        text = path.read_text(encoding="utf-8")
        return case.text in text, not has_utf8_bom(str(path))


def _roundtrip_ass(case: CaptionScriptCase) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{case.case_id}.ass"
        export_ass(_result_for_case(case), str(path), karaoke=False)
        text = path.read_text(encoding="utf-8-sig")
        return case.text in text


def _roundtrip_burnin_ass(case: CaptionScriptCase) -> bool:
    stream = io.StringIO()
    _write_burnin_ass_file(
        stream,
        [{"start": 0.0, "end": 3.0, "text": case.text}],
        {
            "fontname": "Arial",
            "fontsize": 48,
            "primary_color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "outline": 3,
            "shadow": 1,
            "alignment": 2,
            "margin_v": 40,
        },
        {"width": 1920, "height": 1080},
    )
    return case.text in stream.getvalue()


def validate_caption_script_case(
    case: CaptionScriptCase,
    *,
    max_cjk_line_length: int = 14,
) -> CaptionScriptReport:
    """Run export-preservation checks for one caption script fixture."""
    scripts = classify_caption_text(case.text)
    expected = sorted(case.expected_scripts)
    failures: List[str] = []
    warnings: List[str] = []

    missing_scripts = sorted(set(expected) - set(scripts))
    if missing_scripts:
        failures.append(f"missing script classifications: {', '.join(missing_scripts)}")

    srt_ok, utf8_without_bom = _roundtrip_srt(case)
    ass_ok = _roundtrip_ass(case)
    burnin_ass_ok = _roundtrip_burnin_ass(case)

    if not srt_ok:
        failures.append("SRT export lost or rewrote fixture text")
    if not utf8_without_bom:
        failures.append("SRT export wrote a UTF-8 BOM despite the default no-BOM policy")
    if not ass_ok:
        failures.append("ASS export lost or rewrote fixture text")
    if not burnin_ass_ok:
        failures.append("burn-in ASS generation lost or rewrote fixture text")

    cjk_line_break_required = needs_cjk_line_breaker(
        case.text,
        max_line_length=max_cjk_line_length,
    )
    wrapped_cjk_lines = wrap_caption_text(
        case.text,
        max_cjk_line_length,
        3,
        ellipsis=False,
    ).splitlines()
    cjk_line_break_supported = (
        not cjk_line_break_required
        or (
            len(wrapped_cjk_lines) > 1
            and all_lines_within(wrapped_cjk_lines, max_cjk_line_length)
        )
    )
    if cjk_line_break_required:
        if not cjk_line_break_supported:
            failures.append("CJK cue has no whitespace breakpoints and F242 line breaking failed")

    return CaptionScriptReport(
        case_id=case.case_id,
        language=case.language,
        scripts=scripts,
        expected_scripts=expected,
        requires_complex_shaping=case.requires_complex_shaping,
        srt_roundtrip=srt_ok,
        ass_roundtrip=ass_ok,
        burnin_ass_roundtrip=burnin_ass_ok,
        utf8_without_bom=utf8_without_bom,
        cjk_line_break_required=cjk_line_break_required,
        cjk_line_break_supported=cjk_line_break_supported,
        warnings=warnings,
        failures=failures,
    )


def build_caption_unicode_report(
    cases: Iterable[CaptionScriptCase] = DEFAULT_SCRIPT_CASES,
) -> dict:
    """Build a machine-readable F223 complex-script caption validation report."""
    case_reports = [validate_caption_script_case(case) for case in cases]
    failures = sum(len(report.failures) for report in case_reports)
    warnings = sum(len(report.warnings) for report in case_reports)
    complex_cases = sum(1 for report in case_reports if report.requires_complex_shaping)
    cjk_cases = sum(1 for report in case_reports if "cjk" in report.scripts)

    return {
        "status": "fail" if failures else "ok",
        "summary": {
            "case_count": len(case_reports),
            "complex_shaping_cases": complex_cases,
            "cjk_cases": cjk_cases,
            "failures": failures,
            "warnings": warnings,
        },
        "cases": [report.as_dict() for report in case_reports],
        "known_followups": {
            "F241": "FFmpeg/libass HarfBuzz and FriBidi linkage gate",
            "F242": "CJK no-space line breaking handled before export/rendering",
        },
    }


def report_to_json(report: dict) -> str:
    """Serialize a caption Unicode validation report."""
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
