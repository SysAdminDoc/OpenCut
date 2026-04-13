"""
OpenCut Natural Language Batch Operations

Parses natural language commands into structured batch operations:
filter criteria + operation + parameters. Uses keyword matching as
the primary engine (no LLM required).

Supports filtering by media type, resolution, duration, filename pattern,
and applying operations like rename, move, transcode, export, tag.
"""

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FilterCriteria:
    """Parsed filter criteria from natural language."""
    media_type: Optional[str] = None       # "video", "audio", "image"
    min_duration: Optional[float] = None   # seconds
    max_duration: Optional[float] = None   # seconds
    min_resolution: Optional[Tuple[int, int]] = None  # (width, height)
    max_resolution: Optional[Tuple[int, int]] = None
    filename_pattern: Optional[str] = None  # regex or glob
    extension: Optional[str] = None
    has_audio: Optional[bool] = None
    aspect_ratio: Optional[str] = None     # "16:9", "9:16", "1:1"


@dataclass
class BatchOperation:
    """Parsed batch operation."""
    action: str = ""           # "rename", "move", "transcode", "export", "tag", "delete", "copy"
    parameters: Dict = field(default_factory=dict)
    description: str = ""


@dataclass
class BatchCommand:
    """Fully parsed batch command from natural language."""
    original_text: str = ""
    filters: FilterCriteria = field(default_factory=FilterCriteria)
    operation: BatchOperation = field(default_factory=BatchOperation)
    confidence: float = 0.0
    parsed: bool = False
    explanation: str = ""


@dataclass
class BatchResult:
    """Result of executing a batch command."""
    files_matched: int = 0
    files_processed: int = 0
    files_failed: int = 0
    results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    operation: str = ""


# ---------------------------------------------------------------------------
# Filter keyword patterns
# ---------------------------------------------------------------------------
_DURATION_PATTERNS = [
    (r'(?:longer|over|more)\s+(?:than\s+)?(\d+)\s*(?:sec|second|s\b)', "min_duration"),
    (r'(?:shorter|under|less)\s+(?:than\s+)?(\d+)\s*(?:sec|second|s\b)', "max_duration"),
    (r'(?:longer|over|more)\s+(?:than\s+)?(\d+)\s*(?:min|minute|m\b)', "min_duration_min"),
    (r'(?:shorter|under|less)\s+(?:than\s+)?(\d+)\s*(?:min|minute|m\b)', "max_duration_min"),
    (r'(?:at\s+least|minimum)\s+(\d+)\s*(?:sec|second|s\b)', "min_duration"),
    (r'(?:at\s+most|maximum)\s+(\d+)\s*(?:sec|second|s\b)', "max_duration"),
]

_RESOLUTION_PATTERNS = [
    (r'(?:4k|uhd|2160p)', (3840, 2160)),
    (r'(?:1080p|full\s*hd|fhd)', (1920, 1080)),
    (r'(?:720p|hd)', (1280, 720)),
    (r'(?:480p|sd)', (640, 480)),
    (r'(\d{3,4})\s*x\s*(\d{3,4})', None),  # Custom WxH
]

_MEDIA_TYPE_KEYWORDS = {
    "video": ["video", "videos", "clip", "clips", "footage", "mp4", "mov", "mkv"],
    "audio": ["audio", "sound", "music", "wav", "mp3", "aac"],
    "image": ["image", "images", "photo", "photos", "picture", "pictures", "jpg", "png"],
}

_OPERATION_KEYWORDS = {
    "rename": {
        "keywords": ["rename", "name", "relabel"],
        "params": {},
    },
    "move": {
        "keywords": ["move", "organize", "sort into", "put in", "place in"],
        "params": {},
    },
    "transcode": {
        "keywords": ["transcode", "convert", "re-encode", "encode", "compress"],
        "params": {},
    },
    "export": {
        "keywords": ["export", "save as", "output"],
        "params": {},
    },
    "tag": {
        "keywords": ["tag", "label", "mark", "categorize", "classify"],
        "params": {},
    },
    "delete": {
        "keywords": ["delete", "remove", "trash", "discard"],
        "params": {},
    },
    "copy": {
        "keywords": ["copy", "duplicate", "backup", "clone"],
        "params": {},
    },
    "proxy": {
        "keywords": ["proxy", "proxies", "low-res", "low res", "offline"],
        "params": {"codec": "libx264", "crf": 28, "scale": "1280:-2"},
    },
    "normalize": {
        "keywords": ["normalize", "loudness", "level audio", "match volume"],
        "params": {"target_lufs": -16},
    },
}

_ASPECT_RATIO_KEYWORDS = {
    "16:9": ["landscape", "widescreen", "horizontal", "16:9", "16x9"],
    "9:16": ["vertical", "portrait", "9:16", "9x16", "tall"],
    "1:1": ["square", "1:1", "1x1"],
}


# ---------------------------------------------------------------------------
# Keyword parser
# ---------------------------------------------------------------------------
def _parse_filters(text: str) -> Tuple[FilterCriteria, float]:
    """Parse filter criteria from natural language text."""
    filters = FilterCriteria()
    confidence = 0.0
    text_lower = text.lower()

    # Media type detection
    for mtype, keywords in _MEDIA_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                filters.media_type = mtype
                confidence += 0.15
                break

    # Duration filters
    for pattern, dtype in _DURATION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1))
            if dtype == "min_duration":
                filters.min_duration = value
            elif dtype == "max_duration":
                filters.max_duration = value
            elif dtype == "min_duration_min":
                filters.min_duration = value * 60
            elif dtype == "max_duration_min":
                filters.max_duration = value * 60
            confidence += 0.15

    # Resolution filters
    for pattern, resolution in _RESOLUTION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if resolution:
                # Check if "above" or "below" qualifier
                before_text = text_lower[:match.start()]
                if any(w in before_text for w in ("above", "over", "higher", "at least", "minimum")):
                    filters.min_resolution = resolution
                elif any(w in before_text for w in ("below", "under", "lower", "at most", "maximum")):
                    filters.max_resolution = resolution
                else:
                    # Exact match — treat as minimum
                    filters.min_resolution = resolution
            else:
                # Custom WxH pattern
                w, h = int(match.group(1)), int(match.group(2))
                filters.min_resolution = (w, h)
            confidence += 0.15

    # Aspect ratio
    for ar, keywords in _ASPECT_RATIO_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                filters.aspect_ratio = ar
                confidence += 0.1
                break

    # Extension filter
    ext_match = re.search(r'\.(\w{2,5})\b', text_lower)
    if ext_match:
        ext = ext_match.group(1)
        if ext in ("mp4", "mov", "mkv", "avi", "wav", "mp3", "jpg", "png", "gif", "webm"):
            filters.extension = f".{ext}"
            confidence += 0.1

    # Filename pattern
    name_match = re.search(r'(?:named?|called|matching|containing)\s+["\']?([^"\']+)["\']?', text_lower)
    if name_match:
        filters.filename_pattern = name_match.group(1).strip()
        confidence += 0.15

    # Audio presence
    if "without audio" in text_lower or "no audio" in text_lower or "silent" in text_lower:
        filters.has_audio = False
        confidence += 0.1
    elif "with audio" in text_lower or "has audio" in text_lower:
        filters.has_audio = True
        confidence += 0.1

    return filters, min(1.0, confidence)


def _parse_operation(text: str) -> Tuple[BatchOperation, float]:
    """Parse the operation from natural language text."""
    text_lower = text.lower()
    best_op = None
    best_confidence = 0.0
    best_params = {}

    for action, config in _OPERATION_KEYWORDS.items():
        for keyword in config["keywords"]:
            if keyword in text_lower:
                # Longer keyword matches get higher confidence
                conf = len(keyword) / 20.0 + 0.3
                if conf > best_confidence:
                    best_confidence = conf
                    best_op = action
                    best_params = dict(config.get("params", {}))

    if not best_op:
        return BatchOperation(), 0.0

    # Extract additional parameters from text
    params = dict(best_params)

    # Transcode codec
    if best_op == "transcode":
        if "h265" in text_lower or "hevc" in text_lower:
            params["codec"] = "libx265"
        elif "h264" in text_lower or "avc" in text_lower:
            params["codec"] = "libx264"
        elif "prores" in text_lower:
            params["codec"] = "prores_ks"
        elif "vp9" in text_lower:
            params["codec"] = "libvpx-vp9"

        # CRF/quality
        crf_match = re.search(r'(?:crf|quality)\s*[:=]?\s*(\d+)', text_lower)
        if crf_match:
            params["crf"] = int(crf_match.group(1))

    # Move/copy destination
    if best_op in ("move", "copy"):
        dest_match = re.search(r'(?:to|into)\s+["\']?([^"\']+)["\']?', text_lower)
        if dest_match:
            params["destination"] = dest_match.group(1).strip()

    # Rename pattern
    if best_op == "rename":
        prefix_match = re.search(r'(?:prefix|prepend)\s+["\']?([^"\']+)["\']?', text_lower)
        if prefix_match:
            params["prefix"] = prefix_match.group(1).strip()
        suffix_match = re.search(r'(?:suffix|append)\s+["\']?([^"\']+)["\']?', text_lower)
        if suffix_match:
            params["suffix"] = suffix_match.group(1).strip()
        replace_match = re.search(r'replace\s+["\']([^"\']+)["\']\s+(?:with)\s+["\']([^"\']+)["\']', text_lower)
        if replace_match:
            params["find"] = replace_match.group(1)
            params["replace"] = replace_match.group(2)

    # Tag value
    if best_op == "tag":
        tag_match = re.search(r'(?:tag|label)\s+(?:as|with)\s+["\']?([^"\']+)["\']?', text_lower)
        if tag_match:
            params["tag"] = tag_match.group(1).strip()

    operation = BatchOperation(
        action=best_op,
        parameters=params,
        description=f"{best_op} operation",
    )

    return operation, min(1.0, best_confidence)


# ---------------------------------------------------------------------------
# Public API: parse_batch_command
# ---------------------------------------------------------------------------
def parse_batch_command(
    text: str,
    on_progress: Optional[Callable] = None,
) -> BatchCommand:
    """
    Parse a natural language command into a structured batch command.

    Extracts filter criteria and operation from text using keyword
    matching. No LLM required.

    Args:
        text: Natural language command text.
        on_progress: Progress callback(pct, msg).

    Returns:
        BatchCommand with parsed filters, operation, and confidence.
    """
    if not text or not text.strip():
        return BatchCommand(
            original_text=text or "",
            parsed=False,
            confidence=0.0,
            explanation="Empty command text",
        )

    text = text.strip()

    if on_progress:
        on_progress(20, "Parsing filter criteria...")

    filters, filter_conf = _parse_filters(text)

    if on_progress:
        on_progress(50, "Parsing operation...")

    operation, op_conf = _parse_operation(text)

    # Combined confidence
    confidence = (filter_conf + op_conf) / 2.0
    parsed = confidence > 0.1 and operation.action != ""

    # Build explanation
    explanations = []
    if filters.media_type:
        explanations.append(f"Filter: {filters.media_type} files")
    if filters.min_duration is not None:
        explanations.append(f"Filter: duration >= {filters.min_duration}s")
    if filters.max_duration is not None:
        explanations.append(f"Filter: duration <= {filters.max_duration}s")
    if filters.min_resolution:
        explanations.append(f"Filter: resolution >= {filters.min_resolution[0]}x{filters.min_resolution[1]}")
    if filters.filename_pattern:
        explanations.append(f"Filter: filename matches '{filters.filename_pattern}'")
    if filters.aspect_ratio:
        explanations.append(f"Filter: aspect ratio {filters.aspect_ratio}")
    if filters.extension:
        explanations.append(f"Filter: extension {filters.extension}")
    if operation.action:
        param_str = ", ".join(f"{k}={v}" for k, v in operation.parameters.items())
        op_desc = f"Operation: {operation.action}"
        if param_str:
            op_desc += f" ({param_str})"
        explanations.append(op_desc)

    explanation = "; ".join(explanations) if explanations else "Could not parse command"

    if on_progress:
        on_progress(100, "Command parsed")

    return BatchCommand(
        original_text=text,
        filters=filters,
        operation=operation,
        confidence=round(confidence, 3),
        parsed=parsed,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------------
def _matches_filter(filepath: str, filters: FilterCriteria) -> bool:
    """Check if a file matches the given filter criteria."""
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Extension filter
    if filters.extension and ext != filters.extension:
        return False

    # Filename pattern
    if filters.filename_pattern:
        try:
            if not re.search(filters.filename_pattern, filename, re.IGNORECASE):
                # Also try simple substring match
                if filters.filename_pattern.lower() not in filename.lower():
                    return False
        except re.error:
            if filters.filename_pattern.lower() not in filename.lower():
                return False

    # Media type by extension
    if filters.media_type:
        from opencut.core.project_organizer import _classify_media_type
        actual_type = _classify_media_type(ext)
        if actual_type != filters.media_type:
            return False

    # Duration and resolution require probing
    needs_probe = (
        filters.min_duration is not None or
        filters.max_duration is not None or
        filters.min_resolution is not None or
        filters.max_resolution is not None or
        filters.has_audio is not None or
        filters.aspect_ratio is not None
    )

    if needs_probe:
        info = get_video_info(filepath)
        duration = info.get("duration", 0)
        width = info.get("width", 0)
        height = info.get("height", 0)

        if filters.min_duration is not None and duration < filters.min_duration:
            return False
        if filters.max_duration is not None and duration > filters.max_duration:
            return False
        if filters.min_resolution is not None:
            min_w, min_h = filters.min_resolution
            if width < min_w or height < min_h:
                return False
        if filters.max_resolution is not None:
            max_w, max_h = filters.max_resolution
            if width > max_w or height > max_h:
                return False
        if filters.aspect_ratio is not None and width > 0 and height > 0:
            actual_ratio = width / height
            expected_ratios = {
                "16:9": 16 / 9,
                "9:16": 9 / 16,
                "1:1": 1.0,
                "4:3": 4 / 3,
                "21:9": 21 / 9,
            }
            expected = expected_ratios.get(filters.aspect_ratio)
            if expected and abs(actual_ratio - expected) > 0.1:
                return False

    return True


# ---------------------------------------------------------------------------
# Operation executors
# ---------------------------------------------------------------------------
def _execute_rename(filepath: str, params: dict) -> dict:
    """Execute rename operation on a single file."""
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    new_name = name
    if "prefix" in params:
        new_name = params["prefix"] + new_name
    if "suffix" in params:
        new_name = new_name + params["suffix"]
    if "find" in params and "replace" in params:
        new_name = new_name.replace(params["find"], params["replace"])

    new_path = os.path.join(directory, new_name + ext)
    if new_path == filepath:
        return {"file": filepath, "status": "skipped", "reason": "name unchanged"}

    os.rename(filepath, new_path)
    return {"file": filepath, "new_path": new_path, "status": "renamed"}


def _execute_move(filepath: str, params: dict) -> dict:
    """Execute move operation on a single file."""
    destination = params.get("destination", "")
    if not destination:
        return {"file": filepath, "status": "error", "reason": "no destination specified"}

    os.makedirs(destination, exist_ok=True)
    new_path = os.path.join(destination, os.path.basename(filepath))
    shutil.move(filepath, new_path)
    return {"file": filepath, "new_path": new_path, "status": "moved"}


def _execute_copy(filepath: str, params: dict) -> dict:
    """Execute copy operation on a single file."""
    destination = params.get("destination", "")
    if not destination:
        return {"file": filepath, "status": "error", "reason": "no destination specified"}

    os.makedirs(destination, exist_ok=True)
    new_path = os.path.join(destination, os.path.basename(filepath))
    shutil.copy2(filepath, new_path)
    return {"file": filepath, "new_path": new_path, "status": "copied"}


def _execute_transcode(filepath: str, params: dict) -> dict:
    """Execute transcode operation on a single file."""
    codec = params.get("codec", "libx264")
    crf = params.get("crf", 18)
    scale = params.get("scale", "")
    out = output_path(filepath, "transcoded")

    cmd = FFmpegCmd().input(filepath)

    if scale:
        cmd = cmd.video_filter(f"scale={scale}")

    cmd = (
        cmd.video_codec(codec, crf=crf, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
    )

    run_ffmpeg(cmd.build(), timeout=600)
    return {"file": filepath, "output_path": out, "status": "transcoded", "codec": codec}


def _execute_proxy(filepath: str, params: dict) -> dict:
    """Generate proxy file for a video."""
    codec = params.get("codec", "libx264")
    crf = params.get("crf", 28)
    scale = params.get("scale", "1280:-2")
    out = output_path(filepath, "proxy")

    cmd = (
        FFmpegCmd()
        .input(filepath)
        .video_filter(f"scale={scale}")
        .video_codec(codec, crf=crf, preset="fast")
        .audio_codec("aac", bitrate="128k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=600)
    return {"file": filepath, "output_path": out, "status": "proxy_created"}


def _execute_tag(filepath: str, params: dict) -> dict:
    """Tag a file (metadata operation — returns tag info without modifying file)."""
    tag_value = params.get("tag", "untagged")
    return {
        "file": filepath,
        "tag": tag_value,
        "status": "tagged",
        "filename": os.path.basename(filepath),
    }


# ---------------------------------------------------------------------------
# Public API: execute_batch
# ---------------------------------------------------------------------------
def execute_batch(
    command: BatchCommand,
    file_list: List[str],
    dry_run: bool = False,
    on_progress: Optional[Callable] = None,
) -> BatchResult:
    """
    Execute a parsed batch command against a file list.

    Filters files matching the criteria, then applies the operation.

    Args:
        command: Parsed BatchCommand from parse_batch_command().
        file_list: List of file paths to operate on.
        dry_run: If True, only report what would happen without executing.
        on_progress: Progress callback(pct, msg).

    Returns:
        BatchResult with processing results.
    """
    if not command.parsed:
        return BatchResult(
            operation=command.operation.action or "none",
            errors=["Command could not be parsed: " + command.explanation],
        )

    if not file_list:
        return BatchResult(
            operation=command.operation.action,
            errors=["No files provided"],
        )

    if on_progress:
        on_progress(5, "Filtering files...")

    # Filter files
    matched_files: List[str] = []
    for fp in file_list:
        if os.path.isfile(fp) and _matches_filter(fp, command.filters):
            matched_files.append(fp)

    if on_progress:
        on_progress(20, f"Matched {len(matched_files)}/{len(file_list)} files")

    if not matched_files:
        return BatchResult(
            files_matched=0,
            operation=command.operation.action,
            errors=["No files matched the filter criteria"],
        )

    # Execute or simulate
    results: List[Dict] = []
    errors: List[str] = []
    processed = 0
    failed = 0
    action = command.operation.action
    params = command.operation.parameters

    executors = {
        "rename": _execute_rename,
        "move": _execute_move,
        "copy": _execute_copy,
        "transcode": _execute_transcode,
        "proxy": _execute_proxy,
        "tag": _execute_tag,
    }

    total = len(matched_files)
    for i, fp in enumerate(matched_files):
        if on_progress and (i % max(1, total // 20) == 0):
            pct = 20 + int(75 * i / total)
            on_progress(pct, f"Processing {i + 1}/{total}: {os.path.basename(fp)}")

        if dry_run:
            results.append({
                "file": fp,
                "action": action,
                "params": params,
                "status": "dry_run",
            })
            processed += 1
            continue

        executor = executors.get(action)
        if not executor:
            if action == "delete":
                try:
                    os.unlink(fp)
                    results.append({"file": fp, "status": "deleted"})
                    processed += 1
                except OSError as exc:
                    errors.append(f"Delete failed for {fp}: {exc}")
                    failed += 1
            elif action == "export":
                # Export is essentially a copy with potential format conversion
                try:
                    result = _execute_copy(fp, params)
                    results.append(result)
                    processed += 1
                except Exception as exc:
                    errors.append(f"Export failed for {fp}: {exc}")
                    failed += 1
            else:
                errors.append(f"Unknown operation: {action}")
                failed += 1
            continue

        try:
            result = executor(fp, params)
            results.append(result)
            processed += 1
        except Exception as exc:
            errors.append(f"{action} failed for {fp}: {exc}")
            failed += 1

    if on_progress:
        on_progress(100, f"Batch complete: {processed} processed, {failed} failed")

    return BatchResult(
        files_matched=len(matched_files),
        files_processed=processed,
        files_failed=failed,
        results=results,
        errors=errors,
        operation=action,
    )
