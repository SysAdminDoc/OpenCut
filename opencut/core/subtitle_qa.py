"""
OpenCut Subtitle QA v1.28.0

CPS check, min/max gap, overlap detection, max line length per broadcast standard.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("opencut")

PROFILES: Dict[str, Dict[str, Any]] = {
    "netflix": {"cps": 20, "min_gap_ms": 84,  "max_line_chars": 42, "max_lines": 2},
    "bbc":     {"cps": 17, "min_gap_ms": 160, "max_line_chars": 37, "max_lines": 2},
    "youtube": {"cps": 25, "min_gap_ms": 0,   "max_line_chars": 80, "max_lines": 3},
    "ebu_ttd": {"cps": 17, "min_gap_ms": 80,  "max_line_chars": 40, "max_lines": 2},
}


def check_subtitle_qa_available() -> bool:
    return True


@dataclass
class QAIssue:
    rule: str = ""
    severity: str = "warning"
    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    detail: str = ""

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("rule", "severity", "index", "start", "end", "text", "detail")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


@dataclass
class QAReport:
    issues: List[QAIssue] = field(default_factory=list)
    passed: bool = True
    total_cues: int = 0
    profile: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("issues", "passed", "total_cues", "profile", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def _tc_to_sec(tc: str) -> float:
    tc = tc.strip().replace(",", ".")
    parts = tc.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
    except (ValueError, TypeError):
        pass
    return 0.0


def _parse_srt(path: str) -> List[Dict]:
    cues = []
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = len(cues) + 1
        tc_match = re.match(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", lines[1])
        if not tc_match:
            continue
        start = _tc_to_sec(tc_match.group(1))
        end = _tc_to_sec(tc_match.group(2))
        text = "\n".join(lines[2:])
        cues.append({"index": idx, "start": start, "end": end, "text": text})
    return cues


def _parse_vtt(path: str) -> List[Dict]:
    cues = []
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    blocks = re.split(r"\n\s*\n", content.strip())
    idx = 0
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        start_line = 0
        if "-->" not in lines[0] and len(lines) > 1:
            start_line = 1
        if start_line >= len(lines):
            continue
        # Handle both HH:MM:SS.mmm and MM:SS.mmm WebVTT timestamp forms
        tc_match = re.match(
            r"(\d+:\d{2}:\d{2}[.,]\d+|\d{2}:\d{2}[.,]\d+)"
            r"\s*-->\s*"
            r"(\d+:\d{2}:\d{2}[.,]\d+|\d{2}:\d{2}[.,]\d+)",
            lines[start_line],
        )
        if not tc_match:
            continue
        idx += 1
        start = _tc_to_sec(tc_match.group(1))
        end = _tc_to_sec(tc_match.group(2))
        text = "\n".join(lines[start_line + 1:])
        cues.append({"index": idx, "start": start, "end": end, "text": text})
    return cues


def _parse_ass(path: str) -> List[Dict]:
    cues = []
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    def _ass_tc(tc: str) -> float:
        m = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", tc)
        if not m:
            return 0.0
        h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mn * 60 + s + cs / 100.0

    idx = 0
    in_events = False
    for line in lines:
        line = line.rstrip("\n")
        if line.strip().lower() == "[events]":
            in_events = True
            continue
        if in_events and line.startswith("Dialogue:"):
            parts = line[9:].split(",", 9)
            if len(parts) < 10:
                continue
            idx += 1
            start = _ass_tc(parts[1].strip())
            end = _ass_tc(parts[2].strip())
            text = re.sub(r"\{[^}]*\}", "", parts[9]).replace("\\N", "\n").replace("\\n", "\n")
            cues.append({"index": idx, "start": start, "end": end, "text": text})
    return cues


def validate(subtitle_path: str, profile: str = "netflix") -> QAReport:
    """Run QA checks on a subtitle file against a broadcast profile."""
    if not subtitle_path or not os.path.isfile(subtitle_path):
        raise ValueError(f"Subtitle file not found: {subtitle_path}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(PROFILES.keys())}")

    cfg = PROFILES[profile]
    max_cps = cfg["cps"]
    min_gap_ms = cfg["min_gap_ms"]
    max_line_chars = cfg["max_line_chars"]
    max_lines = cfg["max_lines"]

    ext = subtitle_path.lower().rsplit(".", 1)[-1]
    if ext in ("ass", "ssa"):
        cues = _parse_ass(subtitle_path)
    elif ext == "vtt":
        cues = _parse_vtt(subtitle_path)
    else:
        cues = _parse_srt(subtitle_path)

    issues: List[QAIssue] = []

    for i, cue in enumerate(cues):
        dur = cue["end"] - cue["start"]
        text = cue["text"]
        clean_text = re.sub(r"<[^>]+>|\{[^}]+\}", "", text)
        char_count = len(clean_text.replace("\n", ""))

        if dur > 0:
            cps = char_count / dur
            if cps > max_cps:
                issues.append(QAIssue(rule="cps", severity="error", index=cue["index"],
                    start=cue["start"], end=cue["end"], text=text,
                    detail=f"CPS={cps:.1f} exceeds limit {max_cps}"))

        for line in clean_text.splitlines():
            if len(line) > max_line_chars:
                issues.append(QAIssue(rule="line_length", severity="warning", index=cue["index"],
                    start=cue["start"], end=cue["end"], text=text,
                    detail=f"Line length {len(line)} > {max_line_chars}"))

        line_count = len([ln for ln in clean_text.splitlines() if ln.strip()])
        if line_count > max_lines:
            issues.append(QAIssue(rule="max_lines", severity="warning", index=cue["index"],
                start=cue["start"], end=cue["end"], text=text,
                detail=f"{line_count} lines > {max_lines} max"))

        if i < len(cues) - 1:
            gap_ms = (cues[i + 1]["start"] - cue["end"]) * 1000
            if min_gap_ms > 0 and gap_ms < min_gap_ms:
                issues.append(QAIssue(rule="min_gap", severity="warning", index=cue["index"],
                    start=cue["end"], end=cues[i + 1]["start"], text="",
                    detail=f"Gap {gap_ms:.0f}ms < {min_gap_ms}ms"))

        if i < len(cues) - 1 and cue["end"] > cues[i + 1]["start"]:
            issues.append(QAIssue(rule="overlap", severity="error", index=cue["index"],
                start=cue["start"], end=cue["end"], text=text,
                detail=f"Overlaps cue {cues[i+1]['index']} by {(cue['end'] - cues[i+1]['start']) * 1000:.0f}ms"))

    passed = not any(iss.severity == "error" for iss in issues)
    return QAReport(issues=issues, passed=passed, total_cues=len(cues), profile=profile, notes=[])


__all__ = ["PROFILES", "check_subtitle_qa_available", "QAIssue", "QAReport", "validate"]
