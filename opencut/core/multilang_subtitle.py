"""
OpenCut Multi-Language Simultaneous Subtitle Editing

Data model with a shared base timing track and per-language text arrays.
Operations: add/remove languages, update text, bulk import from SRT/VTT,
timing shift, and export to per-language SRT/VTT/ASS files or multi-track
MKV with embedded subtitles.

Storage as JSON in ~/.opencut/subtitles/.
"""

import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, get_ffmpeg_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBTITLE_DIR = os.path.join(OPENCUT_DIR, "subtitles")

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai", "pl": "Polish",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "hu": "Hungarian", "ro": "Romanian",
    "bg": "Bulgarian", "uk": "Ukrainian", "el": "Greek", "he": "Hebrew",
    "fa": "Persian", "id": "Indonesian", "ms": "Malay", "sw": "Swahili",
    "af": "Afrikaans", "ca": "Catalan", "hr": "Croatian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi",
    "ur": "Urdu", "pa": "Punjabi", "gu": "Gujarati", "fil": "Filipino",
    "cy": "Welsh", "ga": "Irish", "la": "Latin",
}


@dataclass
class TimingSegment:
    """A single time segment (shared across all languages)."""

    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class MultiLangProject:
    """Multi-language subtitle project metadata."""

    project_id: str = ""
    name: str = ""
    languages: List[str] = field(default_factory=list)
    segment_count: int = 0
    total_duration: float = 0.0
    video_path: str = ""
    fps: float = 24.0


@dataclass
class MultiLangData:
    """Internal representation of a multi-language subtitle project."""

    project_id: str = ""
    name: str = ""
    timing: List[TimingSegment] = field(default_factory=list)
    texts: Dict[str, List[str]] = field(default_factory=dict)
    video_path: str = ""
    fps: float = 24.0

    @property
    def segment_count(self) -> int:
        return len(self.timing)

    @property
    def total_duration(self) -> float:
        if not self.timing:
            return 0.0
        return max(seg.end for seg in self.timing)

    @property
    def languages(self) -> List[str]:
        return sorted(self.texts.keys())

    def info(self) -> MultiLangProject:
        return MultiLangProject(
            project_id=self.project_id,
            name=self.name,
            languages=self.languages,
            segment_count=self.segment_count,
            total_duration=self.total_duration,
            video_path=self.video_path,
            fps=self.fps,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _ensure_subtitle_dir():
    """Create the subtitles storage directory."""
    os.makedirs(SUBTITLE_DIR, exist_ok=True)


def _project_path(project_id: str) -> str:
    """Return the JSON file path for a project."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", project_id)
    return os.path.join(SUBTITLE_DIR, f"{safe_id}.json")


def _serialize(data: MultiLangData) -> dict:
    """Convert MultiLangData to JSON-serializable dict."""
    return {
        "project_id": data.project_id,
        "name": data.name,
        "timing": [{"start": s.start, "end": s.end} for s in data.timing],
        "texts": data.texts,
        "video_path": data.video_path,
        "fps": data.fps,
    }


def _deserialize(d: dict) -> MultiLangData:
    """Load MultiLangData from a dict."""
    timing = [
        TimingSegment(start=float(s["start"]), end=float(s["end"]))
        for s in d.get("timing", [])
    ]
    return MultiLangData(
        project_id=d.get("project_id", ""),
        name=d.get("name", ""),
        timing=timing,
        texts=d.get("texts", {}),
        video_path=d.get("video_path", ""),
        fps=float(d.get("fps", 24.0)),
    )


def save_project(data: MultiLangData) -> str:
    """Save project to JSON. Returns file path."""
    _ensure_subtitle_dir()
    path = _project_path(data.project_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(data), f, indent=2, ensure_ascii=False)
    logger.info("Saved multilang project %s to %s", data.project_id, path)
    return path


def load_project(project_id: str) -> MultiLangData:
    """Load project from JSON. Raises FileNotFoundError if missing."""
    path = _project_path(project_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Project not found: {project_id}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return _deserialize(d)


def delete_project(project_id: str) -> bool:
    """Delete a project file. Returns True if deleted."""
    path = _project_path(project_id)
    if os.path.isfile(path):
        os.unlink(path)
        logger.info("Deleted multilang project %s", project_id)
        return True
    return False


def list_projects() -> List[MultiLangProject]:
    """List all saved multilang projects."""
    _ensure_subtitle_dir()
    result: List[MultiLangProject] = []
    for fname in os.listdir(SUBTITLE_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            path = os.path.join(SUBTITLE_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            data = _deserialize(d)
            result.append(data.info())
        except Exception as e:
            logger.debug("Skipping invalid project file %s: %s", fname, e)
    return result


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------
def create_project(
    name: str,
    timing_segments: Optional[List[Dict]] = None,
    base_language: str = "en",
    base_texts: Optional[List[str]] = None,
    video_path: str = "",
    fps: float = 24.0,
) -> MultiLangData:
    """Create a new multi-language subtitle project.

    Args:
        name: Project name.
        timing_segments: List of dicts with 'start' and 'end' keys (seconds).
        base_language: ISO 639-1 code for the initial language.
        base_texts: Text for each timing segment in the base language.
        video_path: Optional associated video file path.
        fps: Video frame rate.

    Returns:
        MultiLangData instance (also saved to disk).
    """
    project_id = uuid.uuid4().hex[:12]
    timing: List[TimingSegment] = []
    if timing_segments:
        for seg in timing_segments:
            timing.append(TimingSegment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
            ))
    texts: Dict[str, List[str]] = {}
    if base_language:
        if base_texts and len(base_texts) == len(timing):
            texts[base_language] = list(base_texts)
        else:
            texts[base_language] = [""] * len(timing)
    data = MultiLangData(
        project_id=project_id,
        name=name,
        timing=timing,
        texts=texts,
        video_path=video_path,
        fps=fps,
    )
    save_project(data)
    return data


# ---------------------------------------------------------------------------
# Language operations
# ---------------------------------------------------------------------------
def add_language(project_id: str, language_code: str) -> MultiLangData:
    """Add a language track to the project. Initialises with empty strings."""
    data = load_project(project_id)
    lang = language_code.lower().strip()
    if lang in data.texts:
        raise ValueError(f"Language '{lang}' already exists in project")
    data.texts[lang] = [""] * data.segment_count
    save_project(data)
    logger.info("Added language %s to project %s", lang, project_id)
    return data


def remove_language(project_id: str, language_code: str) -> MultiLangData:
    """Remove a language track from the project."""
    data = load_project(project_id)
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not found in project")
    del data.texts[lang]
    save_project(data)
    logger.info("Removed language %s from project %s", lang, project_id)
    return data


def update_text(
    project_id: str,
    language_code: str,
    segment_index: int,
    text: str,
) -> MultiLangData:
    """Update text for a specific language and segment index."""
    data = load_project(project_id)
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not found in project")
    if segment_index < 0 or segment_index >= data.segment_count:
        raise IndexError(
            f"Segment index {segment_index} out of range "
            f"(0-{data.segment_count - 1})"
        )
    data.texts[lang][segment_index] = text
    save_project(data)
    return data


def bulk_update_texts(
    project_id: str,
    language_code: str,
    texts: List[str],
) -> MultiLangData:
    """Replace all texts for a language at once."""
    data = load_project(project_id)
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not found in project")
    if len(texts) != data.segment_count:
        raise ValueError(
            f"Text count ({len(texts)}) must match "
            f"segment count ({data.segment_count})"
        )
    data.texts[lang] = list(texts)
    save_project(data)
    return data


# ---------------------------------------------------------------------------
# SRT / VTT parsing (for import)
# ---------------------------------------------------------------------------
def _ts_to_seconds(ts: str) -> float:
    """Parse HH:MM:SS,mmm or HH:MM:SS.mmm to float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def _seconds_to_srt(s: float) -> str:
    """Seconds to SRT timestamp HH:MM:SS,mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def _seconds_to_vtt(s: float) -> str:
    """Seconds to VTT timestamp HH:MM:SS.mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _seconds_to_ass(s: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def _parse_srt_content(content: str) -> List[Dict]:
    """Parse SRT into list of {start, end, text} dicts."""
    segments: List[Dict] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                time_line = line
                text_start = i + 1
                break
        if not time_line:
            continue
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _ts_to_seconds(parts[0])
        end = _ts_to_seconds(parts[1].split()[0])
        text = "\n".join(lines[text_start:]).strip()
        segments.append({"start": start, "end": end, "text": text})
    return segments


def _parse_vtt_content(content: str) -> List[Dict]:
    """Parse WebVTT into list of {start, end, text} dicts."""
    lines_all = content.strip().split("\n")
    start_idx = 0
    for i, line in enumerate(lines_all):
        if line.strip().upper().startswith("WEBVTT"):
            start_idx = i + 1
            break
    body = "\n".join(lines_all[start_idx:])
    blocks = re.split(r"\n\s*\n", body.strip())
    segments: List[Dict] = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                time_line = line
                text_start = i + 1
                break
        if not time_line:
            continue
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _ts_to_seconds(parts[0])
        end = _ts_to_seconds(parts[1].split()[0])
        text = "\n".join(lines[text_start:]).strip()
        segments.append({"start": start, "end": end, "text": text})
    return segments


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def bulk_import(
    project_id: str,
    language_code: str,
    content: str,
    fmt: str = "srt",
    align_to_timing: bool = True,
) -> MultiLangData:
    """Import SRT/VTT content as a language track, aligned to base timing.

    Args:
        project_id: Project ID.
        language_code: Language to import as.
        content: SRT or VTT file content string.
        fmt: Format of content ('srt' or 'vtt').
        align_to_timing: If True, align imported text to existing timing
            segments by matching closest start times. If False, use imported
            timing to replace base timing (only when project has <=1 language).

    Returns:
        Updated MultiLangData.
    """
    data = load_project(project_id)
    lang = language_code.lower().strip()

    fmt = fmt.lower().strip()
    if fmt == "srt":
        imported = _parse_srt_content(content)
    elif fmt == "vtt":
        imported = _parse_vtt_content(content)
    else:
        raise ValueError(f"Unsupported import format: {fmt}. Use srt or vtt.")

    if not imported:
        raise ValueError("No subtitle segments found in imported content")

    if not data.timing or (not align_to_timing and len(data.texts) <= 1):
        # Use imported timing as base
        data.timing = [
            TimingSegment(start=s["start"], end=s["end"]) for s in imported
        ]
        data.texts[lang] = [s["text"] for s in imported]
        # Fill empty strings for other languages
        for existing_lang in data.texts:
            if existing_lang != lang:
                data.texts[existing_lang] = [""] * len(data.timing)
    elif align_to_timing:
        # Align to existing timing by overlap
        texts = []
        for seg in data.timing:
            best_text = ""
            best_overlap = 0.0
            for imp in imported:
                overlap_start = max(seg.start, imp["start"])
                overlap_end = min(seg.end, imp["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_text = imp["text"]
            texts.append(best_text)
        data.texts[lang] = texts
    else:
        raise ValueError(
            "Cannot replace timing when other languages exist. "
            "Use align_to_timing=True."
        )

    save_project(data)
    logger.info(
        "Imported %d segments for %s into project %s",
        len(imported), lang, project_id,
    )
    return data


# ---------------------------------------------------------------------------
# Timing operations
# ---------------------------------------------------------------------------
def timing_shift(
    project_id: str,
    offset_seconds: float,
) -> MultiLangData:
    """Shift all timing segments by offset (positive=later, negative=earlier)."""
    data = load_project(project_id)
    for seg in data.timing:
        seg.start = max(0.0, seg.start + offset_seconds)
        seg.end = max(seg.start + 0.01, seg.end + offset_seconds)
    save_project(data)
    logger.info("Shifted timing by %.3fs in project %s", offset_seconds, project_id)
    return data


def add_segment(
    project_id: str,
    start: float,
    end: float,
    texts: Optional[Dict[str, str]] = None,
) -> MultiLangData:
    """Add a new timing segment to the project."""
    data = load_project(project_id)
    data.timing.append(TimingSegment(start=start, end=end))
    for lang in data.texts:
        data.texts[lang].append(texts.get(lang, "") if texts else "")
    save_project(data)
    return data


def remove_segment(project_id: str, segment_index: int) -> MultiLangData:
    """Remove a timing segment and its texts across all languages."""
    data = load_project(project_id)
    if segment_index < 0 or segment_index >= data.segment_count:
        raise IndexError(f"Segment index {segment_index} out of range")
    data.timing.pop(segment_index)
    for lang in data.texts:
        data.texts[lang].pop(segment_index)
    save_project(data)
    return data


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
def export_srt(data: MultiLangData, language_code: str) -> str:
    """Export a single language track as SRT string."""
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not in project")
    texts = data.texts[lang]
    lines: List[str] = []
    for i, seg in enumerate(data.timing):
        lines.append(str(i + 1))
        lines.append(
            f"{_seconds_to_srt(seg.start)} --> {_seconds_to_srt(seg.end)}"
        )
        lines.append(texts[i] if i < len(texts) else "")
        lines.append("")
    return "\n".join(lines)


def export_vtt(data: MultiLangData, language_code: str) -> str:
    """Export a single language track as WebVTT string."""
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not in project")
    texts = data.texts[lang]
    lines: List[str] = ["WEBVTT", ""]
    for i, seg in enumerate(data.timing):
        lines.append(
            f"{_seconds_to_vtt(seg.start)} --> {_seconds_to_vtt(seg.end)}"
        )
        lines.append(texts[i] if i < len(texts) else "")
        lines.append("")
    return "\n".join(lines)


def export_ass(
    data: MultiLangData,
    language_code: str,
    title: str = "OpenCut Subtitles",
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """Export a single language track as ASS string."""
    lang = language_code.lower().strip()
    if lang not in data.texts:
        raise ValueError(f"Language '{lang}' not in project")
    texts = data.texts[lang]
    header = (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,40,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    events: List[str] = []
    for i, seg in enumerate(data.timing):
        start = _seconds_to_ass(seg.start)
        end = _seconds_to_ass(seg.end)
        text = (texts[i] if i < len(texts) else "").replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return header + "\n".join(events) + "\n"


def export_language_files(
    project_id: str,
    output_dir: str,
    fmt: str = "srt",
    languages: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, str]:
    """Export per-language subtitle files.

    Args:
        project_id: Project ID.
        output_dir: Directory to write files into.
        fmt: Format (srt, vtt, ass).
        languages: Specific languages to export (None = all).
        on_progress: Progress callback.

    Returns:
        Dict mapping language code to output file path.
    """
    data = load_project(project_id)
    os.makedirs(output_dir, exist_ok=True)

    langs = languages or data.languages
    result: Dict[str, str] = {}
    total = len(langs)

    for i, lang in enumerate(langs):
        if lang not in data.texts:
            logger.warning("Language %s not in project, skipping", lang)
            continue
        ext = fmt.lower()
        filename = f"{data.name}_{lang}.{ext}"
        filepath = os.path.join(output_dir, filename)

        if ext == "srt":
            content = export_srt(data, lang)
        elif ext == "vtt":
            content = export_vtt(data, lang)
        elif ext == "ass":
            content = export_ass(data, lang)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        result[lang] = filepath

        if on_progress and total > 0:
            on_progress(int(((i + 1) / total) * 90))

    if on_progress:
        on_progress(100)
    logger.info(
        "Exported %d language files for project %s", len(result), project_id,
    )
    return result


def export_mkv(
    project_id: str,
    video_path: str,
    output_path: str,
    languages: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export multi-track MKV with embedded subtitle tracks.

    Args:
        project_id: Project ID.
        video_path: Source video to mux subtitles into.
        output_path: Output MKV path.
        languages: Languages to include (None = all).
        on_progress: Progress callback.

    Returns:
        Output MKV path.
    """
    import tempfile

    data = load_project(project_id)
    ffmpeg = get_ffmpeg_path()
    langs = languages or data.languages

    if on_progress:
        on_progress(10)

    # Write temp SRT files for each language
    temp_files: List[str] = []
    valid_langs: List[str] = []
    try:
        for lang in langs:
            if lang not in data.texts:
                continue
            srt_content = export_srt(data, lang)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{lang}.srt",
                delete=False, encoding="utf-8",
            )
            tmp.write(srt_content)
            tmp.close()
            temp_files.append(tmp.name)
            valid_langs.append(lang)

        if on_progress:
            on_progress(30)

        # Build ffmpeg command
        cmd = [ffmpeg, "-y", "-i", video_path]
        for tf in temp_files:
            cmd.extend(["-i", tf])
        cmd.extend(["-map", "0:v", "-map", "0:a?"])
        for i in range(len(temp_files)):
            cmd.extend(["-map", str(i + 1)])
        cmd.extend(["-c", "copy", "-c:s", "srt"])
        for i, lang in enumerate(valid_langs):
            cmd.extend([f"-metadata:s:s:{i}", f"language={lang}"])
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            cmd.extend([f"-metadata:s:s:{i}", f"title={lang_name}"])
        cmd.append(output_path)

        if on_progress:
            on_progress(50)

        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"FFmpeg MKV mux failed: {stderr}")

        if on_progress:
            on_progress(100)
        logger.info("Exported multi-track MKV to %s", output_path)
        return output_path

    finally:
        for tf in temp_files:
            try:
                os.unlink(tf)
            except OSError:
                pass
