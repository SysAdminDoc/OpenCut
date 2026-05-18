"""Caption line breaking helpers (F242).

OpenCut needs deterministic line wrapping in the Python backend, including CJK
captions that do not contain spaces. ICU4X's line segmenter follows Unicode
line breaking (UAX #14), but there is no lightweight Python runtime binding
that fits the source-install surface today. This module keeps the same contract
shape for caption text: prefer whitespace breaks, allow CJK ideograph/kana/
hangul breaks, and suppress breaks around combining marks and CJK punctuation
that should stay attached to neighbouring text.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable, List

OPENING_PUNCTUATION = set("([{<“‘「『【（［｛〈《")
CLOSING_PUNCTUATION = set(")]}>.,!?;:、。，．！？；：”’」』】）］｝〉》")
NONSTARTER_PUNCTUATION = CLOSING_PUNCTUATION | set("ぁぃぅぇぉっゃゅょァィゥェォッャュョー")
ZERO_WIDTH_JOINERS = {"\u200c", "\u200d"}


def is_cjk(char: str) -> bool:
    """Return True for common CJK ideograph/kana/hangul codepoint ranges."""
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
    )


def _is_combining_or_joiner(char: str) -> bool:
    return bool(unicodedata.combining(char)) or char in ZERO_WIDTH_JOINERS


def _is_breakable_script_char(char: str) -> bool:
    return is_cjk(char)


def _allow_break_after(text: str, index: int) -> bool:
    """Return True when a break is allowed after ``text[index]``."""
    char = text[index]
    next_char = text[index + 1] if index + 1 < len(text) else ""

    if char in {"\r", "\n"}:
        return True
    if char.isspace():
        return True
    if not next_char:
        return True

    if char in OPENING_PUNCTUATION:
        return False
    if next_char in NONSTARTER_PUNCTUATION:
        return False
    if _is_combining_or_joiner(next_char):
        return False

    return _is_breakable_script_char(char) or _is_breakable_script_char(next_char)


def line_break_candidates(text: str) -> List[int]:
    """Return 1-based string offsets where caption text may be line-broken."""
    if not text:
        return []
    candidates = [
        index + 1
        for index in range(len(text))
        if _allow_break_after(text, index)
    ]
    if candidates[-1] != len(text):
        candidates.append(len(text))
    return candidates


def _strip_line(text: str) -> str:
    return text.strip()


def split_caption_text_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into display chunks using Unicode-aware break candidates."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    text = str(text or "").strip()
    if not text:
        return []

    candidates = line_break_candidates(text)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        limit = min(len(text), start + max_chars)
        usable = [candidate for candidate in candidates if start < candidate <= limit]
        end = usable[-1] if usable else limit
        chunk = _strip_line(text[start:end])
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def wrap_caption_text(
    text: str,
    max_line_length: int,
    max_lines: int,
    *,
    ellipsis: bool = True,
) -> str:
    """Wrap caption text without relying on whitespace-only tokenization."""
    if max_line_length <= 0:
        raise ValueError("max_line_length must be greater than zero")
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")

    normalized = str(text or "").strip()
    if len(normalized) <= max_line_length and normalized.count("\n") + 1 <= max_lines:
        return normalized

    if "\n" in normalized:
        lines: List[str] = []
        for source_line in normalized.splitlines():
            chunks = split_caption_text_chunks(source_line, max_line_length)
            lines.extend(chunks or [""])
    else:
        lines = split_caption_text_chunks(normalized, max_line_length)
    if len(lines) <= max_lines:
        return "\n".join(lines)

    kept = lines[:max_lines]
    if ellipsis and kept:
        last = kept[-1].rstrip()
        if not last.endswith("..."):
            if len(last) + 3 > max_line_length:
                last = last[: max(0, max_line_length - 3)].rstrip()
            kept[-1] = f"{last}..."
    return "\n".join(kept)


def caption_layout_tokens(text: str) -> List[str]:
    """Tokenize captions for overlay layout, splitting no-space CJK text."""
    tokens: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in str(text or ""):
        if char.isspace():
            flush()
            continue
        if is_cjk(char):
            flush()
            if tokens and char in NONSTARTER_PUNCTUATION:
                tokens[-1] += char
            else:
                tokens.append(char)
            continue
        if char in NONSTARTER_PUNCTUATION and current:
            current.append(char)
            flush()
            continue
        current.append(char)
    flush()
    return tokens


def all_lines_within(lines: Iterable[str], max_line_length: int) -> bool:
    """Return True when every non-empty wrapped line fits the character cap."""
    return all(len(line) <= max_line_length for line in lines if line)
