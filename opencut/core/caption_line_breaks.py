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
from typing import Iterable, List, Tuple

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


def caption_lines(text: str, max_line_length: int) -> List[str]:
    """Wrap caption text into display lines without dropping any of it.

    Blank source lines are left out, because a blank line ends an SRT cue.
    """
    if max_line_length <= 0:
        raise ValueError("max_line_length must be greater than zero")
    lines: List[str] = []
    for source_line in str(text or "").strip().splitlines():
        lines.extend(split_caption_text_chunks(source_line, max_line_length))
    return lines


def timed_caption_chunks(
    text: str,
    start: float,
    end: float,
    max_line_length: int,
    max_lines: int,
    min_duration: float = 0.0,
) -> List[Tuple[str, float, float]]:
    """Split caption text into cues that each wrap into at most ``max_lines`` lines.

    Returns ``(text, start, end)`` per cue. A cue's text runs from its first
    wrapped line to its last in the source, so no-space CJK text gains no
    spaces and nothing is dropped. Time is shared out by character count.
    With ``min_duration`` set, every cue gets at least that long when the
    span allows it, and the cues share the span equally when it doesn't.
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")
    text = str(text or "").strip()
    spans: List[Tuple[int, int]] = []
    position = 0
    for line in caption_lines(text, max_line_length):
        line_start = text.find(line, position)
        position = line_start + len(line)
        spans.append((line_start, position))
    chunks = [
        text[group[0][0]:group[-1][1]]
        for group in (spans[i:i + max_lines] for i in range(0, len(spans), max_lines))
    ]
    total_chars = sum(len(chunk) for chunk in chunks) or 1
    consumed = 0
    cue_start = start
    cues: List[Tuple[str, float, float]] = []
    for index, chunk in enumerate(chunks):
        consumed += len(chunk)
        cue_end = end if index == len(chunks) - 1 else start + (end - start) * consumed / total_chars
        cues.append((chunk, cue_start, cue_end))
        cue_start = cue_end
    if min_duration > 0 and any(cue_end - cue_start < min_duration for _, cue_start, cue_end in cues):
        cues = _cues_with_min_duration(cues, start, end, min_duration)
    return cues


def _cues_with_min_duration(
    cues: List[Tuple[str, float, float]],
    start: float,
    end: float,
    min_duration: float,
) -> List[Tuple[str, float, float]]:
    """Retime ``cues`` so each lasts at least ``min_duration`` inside ``start``-``end``.

    Cues that would fall short are pinned at the minimum, and the rest of the
    span is shared by character count among the others. When the span can't
    give every cue the minimum, the cues share it equally.
    """
    count = len(cues)
    span = end - start
    weights = [len(text) for text, _, _ in cues]
    if span < min_duration * count:
        durations = [span / count] * count
    else:
        pinned: set = set()
        while True:
            rest = span - min_duration * len(pinned)
            free_weight = sum(w for i, w in enumerate(weights) if i not in pinned) or 1
            durations = [
                min_duration if i in pinned else rest * w / free_weight
                for i, w in enumerate(weights)
            ]
            short = {i for i, d in enumerate(durations) if i not in pinned and d < min_duration}
            if not short:
                break
            pinned |= short
    retimed: List[Tuple[str, float, float]] = []
    cue_start = start
    elapsed = 0.0
    for index, ((text, _, _), duration) in enumerate(zip(cues, durations)):
        elapsed += duration
        cue_end = end if index == count - 1 else start + elapsed
        retimed.append((text, cue_start, cue_end))
        cue_start = cue_end
    return retimed


def wrap_caption_text(
    text: str,
    max_line_length: int,
    max_lines: int,
    *,
    ellipsis: bool = True,
) -> str:
    """Wrap caption text without relying on whitespace-only tokenization.

    Lines past ``max_lines`` are dropped, so subtitle exporters split cues with
    :func:`caption_lines` instead of calling this.
    """
    if max_line_length <= 0:
        raise ValueError("max_line_length must be greater than zero")
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")

    normalized = str(text or "").strip()
    if len(normalized) <= max_line_length and normalized.count("\n") + 1 <= max_lines:
        return normalized

    lines = caption_lines(normalized, max_line_length)
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
