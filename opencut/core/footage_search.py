"""
Footage Search

Indexes media clips by their transcription content.
Stores a simple JSON index on disk. Searches by keyword or phrase.
"""

import json
import logging
import os
import string
from typing import List

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Index location
# ---------------------------------------------------------------------------

_INDEX_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_INDEX_PATH = os.path.join(_INDEX_DIR, "footage_index.json")

# ---------------------------------------------------------------------------
# File locking (cross-platform best-effort)
# ---------------------------------------------------------------------------

try:
    import fcntl  # Unix only
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import msvcrt  # Windows only
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


class _FileLock:
    """Simple cross-platform advisory file lock."""

    def __init__(self, path: str):
        self._path = path + ".lock"
        self._fh = None

    def __enter__(self):
        self._fh = open(self._path, "w", encoding="utf-8")
        if _HAS_FCNTL:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX)
            except Exception:
                pass
        elif _HAS_MSVCRT:
            try:
                self._fh.write(" ")
                self._fh.flush()
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            except Exception:
                pass
        return self

    def __exit__(self, *_):
        if self._fh:
            try:
                if _HAS_MSVCRT:
                    try:
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
                self._fh.close()
            except Exception:
                pass
            try:
                os.unlink(self._path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, return non-empty tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# Index I/O
# ---------------------------------------------------------------------------

def load_index() -> dict:
    """
    Load the footage index from disk.

    Returns:
        Dict mapping filepath → {"mtime": float, "segments": [...], "full_text": str}.
        Returns an empty dict if the index file does not exist.
    """
    if not os.path.exists(_INDEX_PATH):
        return {}
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load footage index: %s", exc)
        return {}


def save_index(index: dict) -> None:
    """
    Save the footage index to disk with a file lock.

    Args:
        index: The index dict to persist.
    """
    os.makedirs(_INDEX_DIR, exist_ok=True)
    with _FileLock(_INDEX_PATH):
        tmp_path = _INDEX_PATH + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(index, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _INDEX_PATH)
        except OSError as exc:
            logger.error("Failed to save footage index: %s", exc)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise


def index_file(filepath: str, segments: List[dict]) -> None:
    """
    Add or update one file in the footage index.

    Args:
        filepath: Absolute path to the media file.
        segments: List of {"start": float, "end": float, "text": str} dicts.
    """
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        mtime = 0.0

    full_text = " ".join(seg.get("text", "") for seg in segments).strip()

    entry = {
        "mtime": mtime,
        "segments": [
            {
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
            if seg.get("text", "").strip()
        ],
        "full_text": full_text,
    }

    index = load_index()
    index[filepath] = entry
    save_index(index)
    logger.info("Indexed %d segments for %s", len(entry["segments"]), filepath)


def remove_missing_files(index: dict) -> dict:
    """
    Remove entries for files that no longer exist on disk.

    Args:
        index: The index dict (modified in-place).

    Returns:
        The cleaned index dict.
    """
    missing = [path for path in list(index.keys()) if not os.path.exists(path)]
    for path in missing:
        del index[path]
        logger.info("Removed missing file from index: %s", path)
    return index


# ---------------------------------------------------------------------------
# BM25-lite scoring
# ---------------------------------------------------------------------------

def _score_segment(query_tokens: List[str], seg_tokens: List[str], avg_dl: float) -> float:
    """
    BM25-lite scoring: term frequency with length normalisation.

    k1=1.5, b=0.75 (standard BM25 defaults).
    """
    if not query_tokens or not seg_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    dl = len(seg_tokens)
    score = 0.0

    from collections import Counter
    tf_map = Counter(seg_tokens)

    for term in query_tokens:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        # IDF approximation (without corpus; weight all terms equally)
        idf = 1.0
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0))
        score += idf * (numerator / denominator)

    return score


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_footage(query: str, top_k: int = 10) -> List[dict]:
    """
    Search the footage index for clips matching the query.

    Scores each clip segment using BM25-lite term overlap and returns the
    top_k highest-scoring segments.

    Args:
        query: Search string (keywords or phrase).
        top_k: Maximum number of results to return.

    Returns:
        List of {"path": str, "start": float, "end": float,
                 "text": str, "score": float}, sorted by score descending.
    """
    index = load_index()
    if not index:
        return []

    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    # Compute average segment length for BM25 normalisation
    all_seg_lengths = []
    for entry in index.values():
        for seg in entry.get("segments", []):
            all_seg_lengths.append(len(_tokenise(seg.get("text", ""))))
    avg_dl = sum(all_seg_lengths) / len(all_seg_lengths) if all_seg_lengths else 1.0

    results = []
    for filepath, entry in index.items():
        for seg in entry.get("segments", []):
            seg_tokens = _tokenise(seg.get("text", ""))
            score = _score_segment(query_tokens, seg_tokens, avg_dl)
            if score > 0.0:
                results.append({
                    "path": filepath,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", ""),
                    "score": round(score, 4),
                })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_index_stats() -> dict:
    """
    Return summary statistics about the footage index.

    Returns:
        Dict with "total_files", "total_segments", "index_size_bytes".
    """
    index = load_index()
    total_files = len(index)
    total_segments = sum(len(entry.get("segments", [])) for entry in index.values())
    try:
        index_size_bytes = os.path.getsize(_INDEX_PATH)
    except OSError:
        index_size_bytes = 0

    return {
        "total_files": total_files,
        "total_segments": total_segments,
        "index_size_bytes": index_size_bytes,
    }


def clear_index() -> dict:
    """
    Clear all entries from the footage search index.

    Returns:
        dict with keys: removed_files (int), success (bool)
    """
    index_path = _INDEX_PATH
    removed = 0
    try:
        if os.path.exists(index_path):
            current = load_index()
            removed = len(current)
            save_index({})
            logger.info(f"Cleared footage index: {removed} entries removed")
        return {"removed_files": removed, "success": True}
    except Exception as e:
        logger.error(f"clear_index error: {e}")
        return {"removed_files": 0, "success": False, "error": str(e)}
