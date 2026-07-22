"""Interview Polish resume state (v1.11.0, feature Q).

Cache the expensive step of ``/interview-polish`` (Whisper transcription)
so re-running the pipeline on the same file skips re-transcribing when
the file hasn't changed. On a 60-minute interview that's the difference
between ~2 minutes and ~15 seconds for the whole pipeline.

State key = sha256 of filepath truncated. State file =
``~/.opencut/polish_state/<key>.json``. We store the raw Whisper result
(segments + metadata) plus the source file mtime/size so we can verify
the cache is still valid before using it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from typing import Any, Optional

logger = logging.getLogger("opencut")

STATE_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "polish_state")

# Per-key write lock so concurrent saves on the same source file don't race
# each other writing the same JSON file. Bounded — in practice the number of
# distinct active polish jobs is tiny (<10).
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def _get_write_lock(key: str) -> threading.Lock:
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[key] = lock
        return lock


def _state_key(filepath: str) -> str:
    abs_path = os.path.realpath(filepath) if filepath else ""
    return hashlib.sha256(abs_path.encode("utf-8", errors="ignore")).hexdigest()[:32]


def _state_path(filepath: str) -> str:
    return os.path.join(STATE_DIR, _state_key(filepath) + ".json")


def _file_signature(filepath: str) -> "tuple[float, int]":
    try:
        st = os.stat(filepath)
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def load_state(filepath: str) -> Optional[dict]:
    """Return cached state dict if it matches the current file, else None."""
    if not filepath or not os.path.isfile(filepath):
        return None
    path = _state_path(filepath)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    sig = _file_signature(filepath)
    cached_sig = (float(cache.get("mtime", 0)), int(cache.get("size", 0)))
    if sig != cached_sig or not sig[1]:
        logger.info("Polish cache invalid for %s (sig %s vs %s)",
                    filepath, sig, cached_sig)
        return None
    return cache


def save_state(filepath: str, transcription_dict: dict) -> None:
    """Persist a transcription result next to *filepath*.

    Accepts a plain dict (either the TranscriptionResult marshalled via
    ``__dict__`` or a simple ``{segments, language, ...}`` shape). No
    validation beyond "must be JSON-serializable" — the reader tolerates
    missing fields by treating the state as invalid.
    """
    if not filepath:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
    except OSError as e:
        logger.warning("Could not create polish state dir: %s", e)
        return
    mtime, size = _file_signature(filepath)
    cache = {
        "filepath": os.path.realpath(filepath),
        "mtime": mtime,
        "size": size,
        "transcription": transcription_dict,
    }
    target_path = _state_path(filepath)
    # Atomic write: serialize to a temp file in the same dir, then rename.
    # Direct ``open(target, "w")`` truncates immediately, so a crash mid-
    # write or two threads racing on the same key would leave a corrupt
    # JSON that ``load_state`` then silently treats as "no cache".
    with _get_write_lock(_state_key(filepath)):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=STATE_DIR, suffix=".tmp", prefix="polish."
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, target_path)
            tmp_path = None
        except (OSError, TypeError) as e:
            logger.warning("Could not write polish state: %s", e)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def clear_state(filepath: str) -> bool:
    """Remove any cached state for *filepath*. Returns True if anything was removed."""
    if not filepath:
        return False
    path = _state_path(filepath)
    if os.path.isfile(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False
    return False


def _transcription_to_dict(result: Any) -> dict:
    """Best-effort marshal of a TranscriptionResult into a plain dict."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    out: dict = {}
    for attr in ("language", "duration", "word_count", "language_confidence"):
        if hasattr(result, attr):
            out[attr] = getattr(result, attr)
    from opencut.core.asr_provenance import provenance_to_dict

    out["provenance"] = provenance_to_dict(getattr(result, "provenance", None))
    segs = getattr(result, "segments", None) or []
    out["segments"] = []
    for seg in segs:
        if isinstance(seg, dict):
            out["segments"].append(seg)
            continue
        seg_dict = {
            "start": getattr(seg, "start", 0),
            "end":   getattr(seg, "end", 0),
            "text":  getattr(seg, "text", ""),
            "speaker": getattr(seg, "speaker", None),
            "language": getattr(seg, "language", None),
            "language_confidence": getattr(seg, "language_confidence", 1.0),
            "confidence": getattr(seg, "confidence", 1.0),
            "boundary_confidence": getattr(seg, "boundary_confidence", None),
            "human_review_recommended": getattr(seg, "human_review_recommended", False),
            "review_reasons": list(getattr(seg, "review_reasons", []) or []),
        }
        words = getattr(seg, "words", None)
        if words:
            seg_dict["words"] = [
                {
                    "start": getattr(w, "start", 0),
                    "end":   getattr(w, "end", 0),
                    "word":  getattr(w, "word", "") or getattr(w, "text", ""),
                    "text":  getattr(w, "text", "") or getattr(w, "word", ""),
                    "confidence": getattr(w, "confidence", 1.0),
                    "boundary_confidence": getattr(w, "boundary_confidence", None),
                }
                for w in words
            ]
        out["segments"].append(seg_dict)
    return out


def _transcription_from_dict(d: dict):
    """Rebuild a TranscriptionResult-like object from cached dict form.

    Uses SimpleNamespace so downstream code that reads ``.segments``,
    ``.language``, and ``.word_count`` keeps working without importing
    the actual dataclass — which would re-import faster-whisper at cache-
    read time on systems that only want the cached path.
    """
    from types import SimpleNamespace

    try:
        from opencut.core.captions import caption_review_reasons
    except Exception:
        caption_review_reasons = None

    segs = []
    for seg in d.get("segments", []) or []:
        review_reasons = list(seg.get("review_reasons") or [])
        if caption_review_reasons is not None:
            review_reasons.extend(
                caption_review_reasons(
                    language=seg.get("language"),
                    language_confidence=seg.get("language_confidence", 1.0),
                    confidence=seg.get("confidence", 1.0),
                    boundary_confidence=seg.get("boundary_confidence"),
                )
            )
        review_reasons = list(dict.fromkeys(str(r) for r in review_reasons if str(r).strip()))
        words = [
            SimpleNamespace(
                start=w.get("start", 0),
                end=w.get("end", 0),
                word=w.get("word", ""),
                text=w.get("text", w.get("word", "")),
                confidence=w.get("confidence", 1.0),
                boundary_confidence=w.get("boundary_confidence"),
            )
            for w in (seg.get("words") or [])
        ]
        segs.append(SimpleNamespace(
            start=seg.get("start", 0),
            end=seg.get("end", 0),
            text=seg.get("text", ""),
            words=words,
            speaker=seg.get("speaker"),
            language=seg.get("language"),
            language_confidence=seg.get("language_confidence", 1.0),
            confidence=seg.get("confidence", 1.0),
            boundary_confidence=seg.get("boundary_confidence"),
            human_review_recommended=bool(seg.get("human_review_recommended", False) or review_reasons),
            review_reasons=review_reasons,
        ))
    from opencut.core.asr_provenance import provenance_from_dict

    return SimpleNamespace(
        language=d.get("language", ""),
        duration=d.get("duration", 0),
        language_confidence=d.get("language_confidence", 1.0),
        word_count=d.get("word_count", sum(len(s.text.split()) for s in segs)),
        segments=segs,
        human_review_recommended=any(
            bool(getattr(s, "human_review_recommended", False)) for s in segs
        ),
        review_segment_count=sum(
            1 for s in segs if bool(getattr(s, "human_review_recommended", False))
        ),
        provenance=provenance_from_dict(d.get("provenance")),
    )
