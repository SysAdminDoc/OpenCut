"""Bulk transcript corrections and project glossary support.

The transcript editor keeps timing metadata separate from editorial wording.
This module therefore applies literal find/replace rules to segment text while
retaining existing word timing wherever the replacement leaves a word intact.
The same rules can be applied to a fresh ASR result, including a cache hit, so
project corrections do not disappear when a clip is transcribed again.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence


MAX_TRANSCRIPT_SEGMENTS = 10_000
MAX_CORRECTION_RULES = 500
MAX_TERM_LENGTH = 500


def _clean_text(value: Any, field: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{field} is required")
    if len(value) > MAX_TERM_LENGTH:
        raise ValueError(f"{field} is too long")
    return value


def normalize_correction_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one literal correction rule."""
    if not isinstance(rule, Mapping):
        raise ValueError("correction rule must be an object")
    find = _clean_text(rule.get("find", ""), "find", allow_empty=False)
    replace = _clean_text(rule.get("replace", ""), "replace")
    return {
        "find": find,
        "replace": replace,
        "case_sensitive": bool(rule.get("case_sensitive", False)),
        "whole_word": bool(rule.get("whole_word", False)),
    }


def normalize_correction_rules(
    rules: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    find: str | None = None,
    replace: str | None = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> list[dict[str, Any]]:
    """Normalize either a rule list or the single-rule route shorthand."""
    if rules is None:
        if find is None:
            return []
        return [
            normalize_correction_rule(
                {
                    "find": find,
                    "replace": replace or "",
                    "case_sensitive": case_sensitive,
                    "whole_word": whole_word,
                }
            )
        ]
    if isinstance(rules, Mapping):
        rules = [rules]
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Iterable):
        raise ValueError("rules must be a list of objects")
    normalized = [normalize_correction_rule(rule) for rule in rules]
    if len(normalized) > MAX_CORRECTION_RULES:
        raise ValueError(f"rules cannot exceed {MAX_CORRECTION_RULES} entries")
    return normalized


def project_identity(project_path: str | None) -> str:
    """Return a stable opaque identity for a project or source path."""
    value = str(project_path or "").strip()
    if not value:
        value = "default"
    normalized = value.replace("\\", "/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _copy_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("segments must be a list")
    if len(segments) > MAX_TRANSCRIPT_SEGMENTS:
        raise ValueError(f"segments cannot exceed {MAX_TRANSCRIPT_SEGMENTS} entries")
    copied: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"segment {index} must be an object")
        item = copy.deepcopy(dict(segment))
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"segment {index} text must be a string")
        if len(text) > 100_000:
            raise ValueError(f"segment {index} text is too long")
        item["text"] = text
        copied.append(item)
    return copied


def _pattern(rule: Mapping[str, Any]) -> re.Pattern[str]:
    source = re.escape(str(rule["find"]))
    if rule.get("whole_word"):
        source = rf"(?<!\w){source}(?!\w)"
    flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
    return re.compile(source, flags)


def apply_text_rule(text: str, rule: Mapping[str, Any]) -> tuple[str, int]:
    """Apply one literal rule without interpreting replacement backslashes."""
    pattern = _pattern(rule)
    replacement = str(rule["replace"])
    return pattern.subn(lambda _match: replacement, text)


def _word_value(word: Any) -> str:
    if isinstance(word, Mapping):
        return str(word.get("text", word.get("word", ""))).strip()
    return str(getattr(word, "text", getattr(word, "word", ""))).strip()


def _set_word_value(word: Any, value: str) -> None:
    if isinstance(word, dict):
        if "text" in word or "word" not in word:
            word["text"] = value
        else:
            word["word"] = value
        return
    if hasattr(word, "text"):
        word.text = value
    elif hasattr(word, "word"):
        word.word = value


def _time_value(word: Any, field: str, default: float) -> float:
    raw = word.get(field, default) if isinstance(word, Mapping) else getattr(word, field, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _copy_word(word: Any) -> Any:
    return copy.deepcopy(word)


def _new_word(template: Any, text: str, start: float, end: float) -> Any:
    word = _copy_word(template) if template is not None else {"text": text}
    _set_word_value(word, text)
    if isinstance(word, dict):
        word["start"] = round(start, 6)
        word["end"] = round(max(start, end), 6)
    else:
        if hasattr(word, "start"):
            word.start = round(start, 6)
        if hasattr(word, "end"):
            word.end = round(max(start, end), 6)
    return word


def _rewrite_words(
    words: Sequence[Any],
    before_text: str,
    after_text: str,
    segment_start: float,
    segment_end: float,
) -> list[Any]:
    """Rewrite word labels and preserve timing for unchanged token ranges."""
    if not words:
        return list(words)
    before_tokens = [_word_value(word) for word in words]
    after_tokens = re.findall(r"\S+", after_text)
    if not after_tokens:
        return []
    # Some backends include punctuation-only tokens in text but omit them from
    # the word array. In that case there is no safe alignment; evenly retime
    # only the replacement result across this segment.
    if len(before_tokens) != len(re.findall(r"\S+", before_text)):
        if len(after_tokens) == len(words):
            out = [_copy_word(word) for word in words]
            for word, token in zip(out, after_tokens):
                _set_word_value(word, token)
            return out
        template = words[0]
        span = max(0.01, segment_end - segment_start)
        return [
            _new_word(
                template,
                token,
                segment_start + span * index / len(after_tokens),
                segment_start + span * (index + 1) / len(after_tokens),
            )
            for index, token in enumerate(after_tokens)
        ]

    matcher = SequenceMatcher(
        None,
        [token.casefold() for token in before_tokens],
        [token.casefold() for token in after_tokens],
        autojunk=False,
    )
    output: list[Any] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            for old_index, new_index in zip(range(old_start, old_end), range(new_start, new_end)):
                word = _copy_word(words[old_index])
                _set_word_value(word, after_tokens[new_index])
                output.append(word)
            continue
        if tag == "delete":
            continue
        if old_start < old_end:
            start = _time_value(words[old_start], "start", segment_start)
            end = _time_value(words[old_end - 1], "end", segment_end)
            template = words[old_start]
        else:
            start = (
                _time_value(words[old_start - 1], "end", segment_start)
                if old_start > 0
                else segment_start
            )
            end = (
                _time_value(words[old_start], "start", segment_end)
                if old_start < len(words)
                else segment_end
            )
            template = words[old_start - 1] if old_start > 0 else words[0]
        if end < start:
            end = start
        count = max(1, new_end - new_start)
        span = max(0.01, end - start)
        for offset, new_index in enumerate(range(new_start, new_end)):
            output.append(
                _new_word(
                    template,
                    after_tokens[new_index],
                    start + span * offset / count,
                    start + span * (offset + 1) / count,
                )
            )
    return output


def _rewrite_mapping_words(segment: dict[str, Any], before: str, after: str) -> None:
    words = segment.get("words")
    if not isinstance(words, list) or not words:
        return
    try:
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
    except (TypeError, ValueError):
        start, end = 0.0, 0.0
    segment["words"] = _rewrite_words(words, before, after, start, end)


def _rewrite_object_words(segment: Any, before: str, after: str) -> None:
    words = getattr(segment, "words", None)
    if not isinstance(words, list) or not words:
        return
    start = _time_value(segment, "start", 0.0)
    end = _time_value(segment, "end", start)
    segment.words = _rewrite_words(words, before, after, start, end)


def _apply_rules_to_text(text: str, rules: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    total = 0
    current = text
    for rule in rules:
        current, count = apply_text_rule(current, rule)
        total += count
    return current, total


def preview_transcript_corrections(
    segments: Sequence[Mapping[str, Any]],
    rules: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    find: str | None = None,
    replace: str | None = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> dict[str, Any]:
    """Return a no-write correction preview with a complete undo baseline."""
    normalized_rules = normalize_correction_rules(
        rules,
        find=find,
        replace=replace,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
    )
    original = _copy_segments(segments)
    corrected = copy.deepcopy(original)
    changes: list[dict[str, Any]] = []
    total_replacements = 0
    for index, segment in enumerate(corrected):
        before = str(segment.get("text", ""))
        after, replacement_count = _apply_rules_to_text(before, normalized_rules)
        if replacement_count == 0 or after == before:
            continue
        segment["text"] = after
        _rewrite_mapping_words(segment, before, after)
        total_replacements += replacement_count
        changes.append(
            {
                "segment_index": index,
                "before": before,
                "after": after,
                "replacements": replacement_count,
            }
        )
    summary = {
        "total_segments": len(original),
        "changed_segments": len(changes),
        "total_replacements": total_replacements,
        "source_hash": _json_hash(original),
        "result_hash": _json_hash(corrected),
    }
    return {
        "segments": corrected,
        "original_segments": original,
        "changes": changes,
        "summary": summary,
        "rules": normalized_rules,
    }


def apply_correction_rules_to_result(result: Any, rules: Iterable[Mapping[str, Any]]) -> Any:
    """Apply normalized rules to a ``TranscriptionResult``-like object."""
    normalized_rules = normalize_correction_rules(rules)
    for segment in list(getattr(result, "segments", []) or []):
        before = str(getattr(segment, "text", "") or "")
        after, replacement_count = _apply_rules_to_text(before, normalized_rules)
        if replacement_count == 0 or after == before:
            continue
        segment.text = after
        _rewrite_object_words(segment, before, after)
    return result


def apply_glossary_to_result(result: Any, project_path: str | None = None) -> Any:
    """Apply the persisted glossary to a freshly generated or cached result."""
    from opencut.user_data import load_transcript_glossary

    rules = load_transcript_glossary(project_path)
    if rules:
        apply_correction_rules_to_result(result, rules)
        try:
            result.correction_rules_applied = rules
        except Exception:
            pass
    return result


def merge_glossary_rules(
    existing: Iterable[Mapping[str, Any]],
    additions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge rules by their matching semantics while preserving order."""
    merged = normalize_correction_rules(existing)
    for rule in normalize_correction_rules(additions):
        if rule not in merged:
            merged.append(rule)
    if len(merged) > MAX_CORRECTION_RULES:
        raise ValueError(f"glossary cannot exceed {MAX_CORRECTION_RULES} entries")
    return merged


__all__ = [
    "MAX_CORRECTION_RULES",
    "MAX_TRANSCRIPT_SEGMENTS",
    "apply_correction_rules_to_result",
    "apply_glossary_to_result",
    "apply_text_rule",
    "merge_glossary_rules",
    "normalize_correction_rule",
    "normalize_correction_rules",
    "preview_transcript_corrections",
    "project_identity",
]
