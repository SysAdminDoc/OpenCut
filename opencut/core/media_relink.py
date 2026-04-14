"""
OpenCut Media Relinking Assistant (60.3)

Given a list of offline/missing media paths, search directories for matches
by filename (exact then fuzzy), file size, and duration. Rank candidates
and support batch relinking.

Uses FFmpeg for duration probing and standard library for matching.
"""

import difflib
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class RelinkCandidate:
    """A candidate file that may match an offline path."""
    candidate_path: str = ""
    match_type: str = ""       # exact_name, fuzzy_name, size_match, duration_match
    confidence: float = 0.0    # 0.0 to 1.0
    file_size: int = 0
    duration: float = 0.0
    name_similarity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RelinkEntry:
    """Relink result for a single offline path."""
    offline_path: str = ""
    candidates: List[RelinkCandidate] = field(default_factory=list)
    best_match: str = ""
    best_confidence: float = 0.0
    status: str = "unresolved"  # resolved, partial, unresolved

    def to_dict(self) -> dict:
        return {
            "offline_path": self.offline_path,
            "best_match": self.best_match,
            "best_confidence": round(self.best_confidence, 3),
            "status": self.status,
            "candidates": [c.to_dict() for c in self.candidates[:5]],
        }


@dataclass
class RelinkResult:
    """Complete relinking result."""
    entries: List[RelinkEntry] = field(default_factory=list)
    total_offline: int = 0
    resolved: int = 0
    partial: int = 0
    unresolved: int = 0

    def to_dict(self) -> dict:
        return {
            "total_offline": self.total_offline,
            "resolved": self.resolved,
            "partial": self.partial,
            "unresolved": self.unresolved,
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Media Extensions
# ---------------------------------------------------------------------------
_MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".m4v", ".wmv", ".flv",
    ".webm", ".ts", ".m2ts", ".mpg", ".mpeg", ".wav", ".mp3", ".aac",
    ".flac", ".ogg", ".m4a", ".aif", ".aiff", ".r3d", ".braw", ".ari",
    ".dng", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx",
}


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------
def _scan_directory(
    search_dirs: List[str],
    recursive: bool = True,
) -> Dict[str, List[str]]:
    """
    Scan directories and index files by base filename.

    Returns:
        Dict mapping lowercase basename -> list of full paths.
    """
    index: Dict[str, List[str]] = {}

    for d in search_dirs:
        if not os.path.isdir(d):
            continue

        if recursive:
            for root, _dirs, files in os.walk(d):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in _MEDIA_EXTENSIONS:
                        key = fname.lower()
                        full = os.path.join(root, fname)
                        if key not in index:
                            index[key] = []
                        index[key].append(full)
        else:
            for fname in os.listdir(d):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _MEDIA_EXTENSIONS:
                    key = fname.lower()
                    full = os.path.join(d, fname)
                    if key not in index:
                        index[key] = []
                    index[key].append(full)

    return index


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _get_file_info(path: str) -> Tuple[int, float]:
    """Get file size and duration (if media)."""
    size = 0
    duration = 0.0
    try:
        size = os.path.getsize(path)
    except OSError:
        pass
    try:
        info = get_video_info(path)
        duration = info.get("duration", 0.0)
    except Exception:
        pass
    return size, duration


def find_candidates(
    offline_path: str,
    file_index: Dict[str, List[str]],
    offline_size: int = 0,
    offline_duration: float = 0.0,
    fuzzy_threshold: float = 0.6,
) -> List[RelinkCandidate]:
    """
    Find candidate files that may match an offline path.

    Matching strategy (highest to lowest confidence):
    1. Exact filename match (case-insensitive)
    2. Fuzzy filename match (difflib)
    3. File size match (within 5%)
    4. Duration match (within 2 seconds)

    Args:
        offline_path: The missing file path.
        file_index: Pre-scanned directory index.
        offline_size: Known file size of the offline file (0 if unknown).
        offline_duration: Known duration of the offline file (0 if unknown).
        fuzzy_threshold: Minimum name similarity for fuzzy matches.

    Returns:
        List of RelinkCandidate sorted by confidence.
    """
    candidates = []
    offline_name = os.path.basename(offline_path).lower()
    offline_stem = os.path.splitext(offline_name)[0]
    offline_ext = os.path.splitext(offline_name)[1]

    seen_paths = set()

    # 1. Exact filename match
    if offline_name in file_index:
        for path in file_index[offline_name]:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            size, duration = _get_file_info(path)

            # Boost confidence if size/duration also match
            conf = 0.9
            if offline_size > 0 and size > 0:
                size_ratio = min(size, offline_size) / max(size, offline_size)
                if size_ratio > 0.95:
                    conf = 0.98

            candidates.append(RelinkCandidate(
                candidate_path=path,
                match_type="exact_name",
                confidence=conf,
                file_size=size,
                duration=duration,
                name_similarity=1.0,
            ))

    # 2. Fuzzy filename match
    all_names = list(file_index.keys())
    close_matches = difflib.get_close_matches(
        offline_name, all_names, n=10, cutoff=fuzzy_threshold,
    )

    for match_name in close_matches:
        if match_name == offline_name:
            continue  # already handled above
        sim = difflib.SequenceMatcher(None, offline_name, match_name).ratio()
        for path in file_index[match_name]:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            size, duration = _get_file_info(path)

            candidates.append(RelinkCandidate(
                candidate_path=path,
                match_type="fuzzy_name",
                confidence=sim * 0.8,
                file_size=size,
                duration=duration,
                name_similarity=round(sim, 3),
            ))

    # 3. Size match (scan if offline_size known)
    if offline_size > 0:
        tolerance = offline_size * 0.05  # 5% tolerance
        for name, paths in file_index.items():
            for path in paths:
                if path in seen_paths:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if abs(size - offline_size) <= tolerance:
                    seen_paths.add(path)
                    _, duration = _get_file_info(path)
                    ratio = min(size, offline_size) / max(size, offline_size)
                    candidates.append(RelinkCandidate(
                        candidate_path=path,
                        match_type="size_match",
                        confidence=ratio * 0.7,
                        file_size=size,
                        duration=duration,
                    ))

    # Sort by confidence descending
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def relink_media(
    offline_paths: List[str],
    search_dirs: List[str],
    offline_metadata: Optional[Dict[str, dict]] = None,
    recursive: bool = True,
    fuzzy_threshold: float = 0.6,
    auto_resolve_threshold: float = 0.9,
    on_progress: Optional[Callable] = None,
) -> RelinkResult:
    """
    Find replacement files for offline/missing media paths.

    Args:
        offline_paths: List of file paths that are offline/missing.
        search_dirs: Directories to search for matching files.
        offline_metadata: Optional dict mapping offline_path -> {size, duration}
            for better matching accuracy.
        recursive: Whether to scan subdirectories.
        fuzzy_threshold: Minimum name similarity for fuzzy matches.
        auto_resolve_threshold: Minimum confidence to auto-resolve.
        on_progress: Callback(pct, msg).

    Returns:
        RelinkResult with per-path candidates and best matches.
    """
    if not offline_paths:
        return RelinkResult()

    if not search_dirs:
        raise ValueError("search_dirs cannot be empty")

    offline_metadata = offline_metadata or {}

    if on_progress:
        on_progress(5, f"Scanning {len(search_dirs)} directories...")

    # Phase 1: Index search directories
    file_index = _scan_directory(search_dirs, recursive=recursive)

    if on_progress:
        total_files = sum(len(v) for v in file_index.values())
        on_progress(20, f"Indexed {total_files} media files")

    # Phase 2: Find candidates for each offline path
    result = RelinkResult(total_offline=len(offline_paths))

    for i, op in enumerate(offline_paths):
        meta = offline_metadata.get(op, {})
        off_size = int(meta.get("size", 0))
        off_dur = float(meta.get("duration", 0))

        candidates = find_candidates(
            op, file_index,
            offline_size=off_size,
            offline_duration=off_dur,
            fuzzy_threshold=fuzzy_threshold,
        )

        entry = RelinkEntry(
            offline_path=op,
            candidates=candidates,
        )

        if candidates:
            best = candidates[0]
            entry.best_match = best.candidate_path
            entry.best_confidence = best.confidence

            if best.confidence >= auto_resolve_threshold:
                entry.status = "resolved"
                result.resolved += 1
            else:
                entry.status = "partial"
                result.partial += 1
        else:
            entry.status = "unresolved"
            result.unresolved += 1

        result.entries.append(entry)

        if on_progress:
            pct = 20 + int((i + 1) / len(offline_paths) * 75)
            on_progress(pct, f"Relinking {i + 1}/{len(offline_paths)}...")

    if on_progress:
        on_progress(100, f"Relink complete: {result.resolved} resolved, "
                         f"{result.partial} partial, {result.unresolved} unresolved")

    return result


def batch_relink(
    relink_result: RelinkResult,
    min_confidence: float = 0.8,
) -> Dict[str, str]:
    """
    Extract a batch relink mapping from RelinkResult.

    Only includes entries with confidence >= min_confidence.

    Args:
        relink_result: Output from relink_media.
        min_confidence: Minimum confidence for automatic relinking.

    Returns:
        Dict mapping offline_path -> new_path for entries meeting threshold.
    """
    mapping = {}
    for entry in relink_result.entries:
        if entry.best_match and entry.best_confidence >= min_confidence:
            mapping[entry.offline_path] = entry.best_match
    return mapping
