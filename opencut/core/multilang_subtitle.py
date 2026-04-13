"""
Multi-Language Subtitle Editing (24.2)

Shared timing track with per-language text. Sync timing changes
across all languages, export individual language subtitle files.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TimingEntry:
    """A single subtitle timing entry."""
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def validate(self):
        if self.end <= self.start:
            raise ValueError(
                f"Entry {self.index}: end ({self.end}) must be after start ({self.start})"
            )


@dataclass
class MultiLangProject:
    """Multi-language subtitle project with shared timing."""
    timing: List[TimingEntry] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    translations: Dict[str, List[str]] = field(default_factory=dict)
    # translations[language] = list of text strings, indexed by timing entry index
    project_name: str = ""
    source_language: str = ""

    def entry_count(self) -> int:
        return len(self.timing)

    def language_count(self) -> int:
        return len(self.languages)


# ---------------------------------------------------------------------------
# Create Project
# ---------------------------------------------------------------------------
def create_multilang_project(
    base_timing: List[dict],
    languages: List[str],
    source_text: Optional[List[str]] = None,
    source_language: str = "",
    project_name: str = "",
    on_progress: Optional[Callable] = None,
) -> MultiLangProject:
    """Create a multi-language subtitle project from base timing.

    Args:
        base_timing: List of timing entries [{'start', 'end'}] or
            [{'start', 'end', 'text'}].
        languages: List of language codes (e.g. ['en', 'es', 'fr']).
        source_text: Optional source language text list.
        source_language: Source language code.
        project_name: Optional project name.
        on_progress: Optional callback(pct, msg).

    Returns:
        MultiLangProject with timing and empty translation slots.
    """
    if not base_timing:
        raise ValueError("base_timing must be a non-empty list")
    if not languages:
        raise ValueError("At least one language is required")

    if on_progress:
        on_progress(10, "Creating multi-language project...")

    # Build timing entries
    timing_entries = []
    texts_from_timing = []
    for i, entry in enumerate(base_timing):
        start = float(entry.get("start", 0))
        end = float(entry.get("end", 0))
        if end <= start:
            raise ValueError(f"Entry {i}: end ({end}) must be after start ({start})")
        timing_entries.append(TimingEntry(index=i, start=start, end=end))
        texts_from_timing.append(entry.get("text", ""))

    # Initialize translations
    translations = {}
    for lang in languages:
        if lang == source_language and source_text:
            translations[lang] = list(source_text[:len(timing_entries)])
            # Pad if needed
            while len(translations[lang]) < len(timing_entries):
                translations[lang].append("")
        elif lang == source_language and texts_from_timing:
            translations[lang] = texts_from_timing[:len(timing_entries)]
            while len(translations[lang]) < len(timing_entries):
                translations[lang].append("")
        else:
            translations[lang] = [""] * len(timing_entries)

    project = MultiLangProject(
        timing=timing_entries,
        languages=list(languages),
        translations=translations,
        project_name=project_name or "Untitled",
        source_language=source_language or (languages[0] if languages else ""),
    )

    if on_progress:
        on_progress(100, f"Project created: {len(timing_entries)} entries, {len(languages)} languages")

    return project


# ---------------------------------------------------------------------------
# Update Language Text
# ---------------------------------------------------------------------------
def update_language_text(
    project: MultiLangProject,
    language: str,
    translations: List[str],
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Update text for a specific language.

    Args:
        project: MultiLangProject to update.
        language: Language code to update.
        translations: List of translated strings (same length as timing).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with status, language, entries_updated.
    """
    if language not in project.languages:
        raise ValueError(
            f"Language '{language}' not in project. "
            f"Available: {project.languages}"
        )

    if on_progress:
        on_progress(10, f"Updating {language} translations...")

    entry_count = len(project.timing)
    if len(translations) != entry_count:
        raise ValueError(
            f"Expected {entry_count} translations, got {len(translations)}"
        )

    project.translations[language] = list(translations)

    filled = sum(1 for t in translations if t.strip())

    if on_progress:
        on_progress(100, f"Updated {language}: {filled}/{entry_count} entries filled")

    return {
        "status": "ok",
        "language": language,
        "entries_updated": entry_count,
        "entries_filled": filled,
    }


# ---------------------------------------------------------------------------
# Export Language Subtitles
# ---------------------------------------------------------------------------
def export_language(
    project: MultiLangProject,
    language: str,
    format: str = "srt",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Export subtitles for a specific language.

    Args:
        project: MultiLangProject to export from.
        language: Language code to export.
        format: Export format — "srt", "vtt", "json".
        output_path: Output file path (auto-generated if empty).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, format, language, entry_count.
    """
    if language not in project.languages:
        raise ValueError(f"Language '{language}' not in project")

    format = format.lower()
    if format not in ("srt", "vtt", "json"):
        raise ValueError(f"Unsupported format: {format}. Use srt, vtt, or json")

    if on_progress:
        on_progress(10, f"Exporting {language} as {format}...")

    if not output_path:
        import tempfile
        ext = {"srt": ".srt", "vtt": ".vtt", "json": ".json"}[format]
        fd, output_path = tempfile.mkstemp(
            suffix=ext, prefix=f"subs_{language}_"
        )
        os.close(fd)

    texts = project.translations.get(language, [])

    if format == "srt":
        _export_srt(project.timing, texts, output_path)
    elif format == "vtt":
        _export_vtt(project.timing, texts, output_path)
    elif format == "json":
        _export_json(project.timing, texts, language, output_path)

    if on_progress:
        on_progress(100, f"Exported {language} subtitles")

    return {
        "output_path": output_path,
        "format": format,
        "language": language,
        "entry_count": len(project.timing),
    }


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _export_srt(timing: List[TimingEntry], texts: List[str], output_path: str):
    """Export as SRT format."""
    lines = []
    for i, entry in enumerate(timing):
        text = texts[i] if i < len(texts) else ""
        if not text.strip():
            text = f"[Subtitle {i + 1}]"
        lines.append(f"{i + 1}\n")
        lines.append(
            f"{_format_srt_time(entry.start)} --> {_format_srt_time(entry.end)}\n"
        )
        lines.append(f"{text}\n\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _export_vtt(timing: List[TimingEntry], texts: List[str], output_path: str):
    """Export as WebVTT format."""
    lines = ["WEBVTT\n\n"]
    for i, entry in enumerate(timing):
        text = texts[i] if i < len(texts) else ""
        if not text.strip():
            text = f"[Subtitle {i + 1}]"
        lines.append(
            f"{_format_vtt_time(entry.start)} --> {_format_vtt_time(entry.end)}\n"
        )
        lines.append(f"{text}\n\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _export_json(
    timing: List[TimingEntry], texts: List[str],
    language: str, output_path: str,
):
    """Export as JSON format."""
    entries = []
    for i, entry in enumerate(timing):
        text = texts[i] if i < len(texts) else ""
        entries.append({
            "index": i,
            "start": entry.start,
            "end": entry.end,
            "text": text,
        })

    data = {
        "language": language,
        "entry_count": len(entries),
        "entries": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sync Timing Change
# ---------------------------------------------------------------------------
def sync_timing_change(
    project: MultiLangProject,
    timing_update: Dict,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Apply a timing change to the shared timing track.

    Supported operations:
        - "shift": shift all entries by offset_seconds
        - "update": update a single entry's start/end
        - "insert": insert a new timing entry
        - "delete": delete a timing entry

    Args:
        project: MultiLangProject to update.
        timing_update: Dict with 'operation' and parameters.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with status and updated entry count.
    """
    operation = timing_update.get("operation", "").lower()
    if operation not in ("shift", "update", "insert", "delete"):
        raise ValueError(
            f"Unknown operation: {operation}. "
            "Use shift, update, insert, or delete"
        )

    if on_progress:
        on_progress(10, f"Applying {operation} to timing...")

    if operation == "shift":
        offset = float(timing_update.get("offset_seconds", 0))
        for entry in project.timing:
            entry.start = max(0.0, entry.start + offset)
            entry.end = max(entry.start + 0.01, entry.end + offset)
        result = {
            "status": "ok",
            "operation": "shift",
            "offset_seconds": offset,
            "entries_affected": len(project.timing),
        }

    elif operation == "update":
        index = int(timing_update.get("index", -1))
        if index < 0 or index >= len(project.timing):
            raise ValueError(f"Invalid index: {index}")
        entry = project.timing[index]
        if "start" in timing_update:
            entry.start = float(timing_update["start"])
        if "end" in timing_update:
            entry.end = float(timing_update["end"])
        entry.validate()
        result = {
            "status": "ok",
            "operation": "update",
            "index": index,
            "start": entry.start,
            "end": entry.end,
        }

    elif operation == "insert":
        start = float(timing_update.get("start", 0))
        end = float(timing_update.get("end", 0))
        if end <= start:
            raise ValueError("end must be after start")

        # Find insertion point
        insert_at = len(project.timing)
        for i, entry in enumerate(project.timing):
            if entry.start > start:
                insert_at = i
                break

        new_entry = TimingEntry(index=insert_at, start=start, end=end)
        project.timing.insert(insert_at, new_entry)

        # Reindex
        for i, entry in enumerate(project.timing):
            entry.index = i

        # Insert empty translation slot for all languages
        for lang in project.languages:
            text = timing_update.get("text", {}).get(lang, "")
            project.translations[lang].insert(insert_at, text)

        result = {
            "status": "ok",
            "operation": "insert",
            "index": insert_at,
            "entry_count": len(project.timing),
        }

    elif operation == "delete":
        index = int(timing_update.get("index", -1))
        if index < 0 or index >= len(project.timing):
            raise ValueError(f"Invalid index: {index}")

        project.timing.pop(index)

        # Reindex
        for i, entry in enumerate(project.timing):
            entry.index = i

        # Remove translation slot
        for lang in project.languages:
            if index < len(project.translations[lang]):
                project.translations[lang].pop(index)

        result = {
            "status": "ok",
            "operation": "delete",
            "deleted_index": index,
            "entry_count": len(project.timing),
        }

    if on_progress:
        on_progress(100, f"Timing {operation} applied")

    return result
