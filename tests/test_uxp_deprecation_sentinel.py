"""RA-18 - UXP deprecated API regression sentinel."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
SOURCE_SUFFIXES = {".html", ".js", ".mjs", ".ts", ".tsx", ".jsx"}

FORBIDDEN_PATTERNS = {
    "Clipboard.setContent": re.compile(r"\b(?:Clipboard|clipboard)\.setContent\b"),
    "Clipboard.getContent": re.compile(r"\b(?:Clipboard|clipboard)\.getContent\b"),
    "Clipboard.clearContent": re.compile(r"\b(?:Clipboard|clipboard)\.clearContent\b"),
    "object Clipboard.writeText": re.compile(
        r"\b(?:Clipboard|clipboard|navigator\.clipboard)\.writeText\s*\(\s*\{"
    ),
    "uxpvideoload": re.compile(r"\buxpvideoload\b", re.IGNORECASE),
    "uxpvideoplay": re.compile(r"\buxpvideoplay\b", re.IGNORECASE),
    "uxpvideocomplete": re.compile(r"\buxpvideocomplete\b", re.IGNORECASE),
    "uxpvideopause": re.compile(r"\buxpvideopause\b", re.IGNORECASE),
}


def _uxp_source_files() -> list[Path]:
    return sorted(
        path
        for path in UXP_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and "node_modules" not in path.parts
        and "dist" not in path.parts
    )


def test_uxp_sources_do_not_use_deprecated_clipboard_or_video_apis():
    violations: list[str] = []

    for path in _uxp_source_files():
        source = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(source):
                rel = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{rel}: {label}")

    assert not violations, "Deprecated UXP API usage found:\n" + "\n".join(violations)


def test_uxp_sources_keep_supported_string_clipboard_write_path():
    source = (UXP_ROOT / "main.js").read_text(encoding="utf-8")
    write_calls = re.findall(
        r"\bnavigator\.clipboard\.writeText\s*\((?P<argument>.*?)\)",
        source,
        re.DOTALL,
    )

    assert write_calls
    assert all(not argument.lstrip().startswith("{") for argument in write_calls)
