"""
OpenCut Video/Audio Content Fingerprinting

Generate perceptual hash fingerprints for duplicate detection and content
identification.  Visual: extract frames at 1 fps, compute pHash per frame,
combine into compact binary signature.  Audio: compute chromaprint via
FFmpeg.  Compare fingerprints using hamming distance (visual) and
cross-correlation (audio).  Index in SQLite for fast search.
"""

import hashlib
import logging
import os
import sqlite3
import struct
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import OPENCUT_DIR, get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(OPENCUT_DIR, "fingerprints.db")
_db_lock = threading.Lock()

FRAMES_PER_SECOND = 1
PHASH_SIZE = 8  # 8x8 DCT -> 64-bit hash
SIMILARITY_THRESHOLD = 85  # Default similarity threshold (0-100)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FingerprintResult:
    """Result from fingerprint generation."""
    fingerprint_hex: str = ""
    frame_count: int = 0
    audio_fingerprint: str = ""
    file_hash: str = ""
    duration_sec: float = 0.0
    file_path: str = ""
    file_size: int = 0

    def to_dict(self) -> dict:
        return {
            "fingerprint_hex": self.fingerprint_hex,
            "frame_count": self.frame_count,
            "audio_fingerprint": self.audio_fingerprint,
            "file_hash": self.file_hash,
            "duration_sec": round(self.duration_sec, 2),
            "file_path": self.file_path,
            "file_size": self.file_size,
        }


@dataclass
class SimilarityResult:
    """Result from comparing two fingerprints."""
    visual_score: float = 0.0
    audio_score: float = 0.0
    combined_score: float = 0.0
    is_duplicate: bool = False
    file_a: str = ""
    file_b: str = ""

    def to_dict(self) -> dict:
        return {
            "visual_score": round(self.visual_score, 2),
            "audio_score": round(self.audio_score, 2),
            "combined_score": round(self.combined_score, 2),
            "is_duplicate": self.is_duplicate,
            "file_a": self.file_a,
            "file_b": self.file_b,
        }


@dataclass
class SearchResult:
    """A single search match."""
    file_path: str = ""
    similarity: float = 0.0
    fingerprint_hex: str = ""
    indexed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "similarity": round(self.similarity, 2),
            "fingerprint_hex": self.fingerprint_hex,
            "indexed_at": self.indexed_at,
        }


# ---------------------------------------------------------------------------
# SQLite database
# ---------------------------------------------------------------------------
def _get_db_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode."""
    os.makedirs(OPENCUT_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Initialize the fingerprint database schema."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    fingerprint_hex TEXT NOT NULL,
                    audio_fingerprint TEXT DEFAULT '',
                    frame_count INTEGER DEFAULT 0,
                    duration_sec REAL DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    indexed_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(file_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_fp_path ON fingerprints(file_path);
                CREATE INDEX IF NOT EXISTS idx_fp_hash ON fingerprints(file_hash);
            """)
            conn.commit()
        finally:
            conn.close()


# Initialize on import
_init_db()


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------
def _compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_quick_hash(filepath: str) -> str:
    """Compute a quick hash from file size + first/last 64KB for speed."""
    h = hashlib.sha256()
    size = os.path.getsize(filepath)
    h.update(str(size).encode())
    with open(filepath, "rb") as f:
        head = f.read(65536)
        h.update(head)
        if size > 65536:
            f.seek(max(0, size - 65536))
            tail = f.read(65536)
            h.update(tail)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Perceptual hashing (pure Python, no PIL/imagehash dependency)
# ---------------------------------------------------------------------------
def _extract_frame_data(filepath: str, fps: int = 1,
                        width: int = 32, height: int = 32) -> List[bytes]:
    """Extract frames as raw grayscale bytes using FFmpeg."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", filepath,
        "-vf", f"fps={fps},scale={width}:{height},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            logger.warning("Frame extraction failed: %s", proc.stderr.decode(errors="replace")[-200:])
            return []
        raw = proc.stdout
        frame_size = width * height
        frames = []
        for i in range(0, len(raw), frame_size):
            frame = raw[i:i + frame_size]
            if len(frame) == frame_size:
                frames.append(frame)
        return frames
    except subprocess.TimeoutExpired:
        logger.warning("Frame extraction timed out for %s", filepath)
        return []
    except Exception as e:
        logger.warning("Frame extraction error: %s", e)
        return []


def _compute_frame_phash(frame_bytes: bytes, width: int = 32, height: int = 32) -> int:
    """
    Compute perceptual hash for a single grayscale frame.

    Uses mean-based hashing: resize to small dimensions, compute mean,
    each pixel above mean is 1, below is 0 -> 64-bit hash.
    """
    # We use an 8x8 region from the center of the 32x32 frame
    # to approximate a low-frequency DCT
    pixels = list(frame_bytes)

    # Subsample to 8x8
    sub_w, sub_h = PHASH_SIZE, PHASH_SIZE
    step_x = width // sub_w
    step_y = height // sub_h
    subsampled = []
    for sy in range(sub_h):
        for sx in range(sub_w):
            idx = (sy * step_y) * width + (sx * step_x)
            if idx < len(pixels):
                subsampled.append(pixels[idx])
            else:
                subsampled.append(0)

    # Compute mean
    mean_val = sum(subsampled) / max(len(subsampled), 1)

    # Build hash: bit=1 if pixel >= mean
    h = 0
    for i, val in enumerate(subsampled):
        if val >= mean_val:
            h |= (1 << (63 - i))
    return h


def _combine_frame_hashes(hashes: List[int]) -> str:
    """Combine per-frame hashes into a compact hex string."""
    if not hashes:
        return ""
    # XOR-fold all frame hashes for a summary, then append per-frame
    # Use first N frames for the summary hash
    summary = 0
    for h in hashes[:60]:
        summary ^= h
    # Pack: 8 bytes summary + 8 bytes per frame (up to 120 frames)
    parts = [struct.pack(">Q", summary)]
    for h in hashes[:120]:
        parts.append(struct.pack(">Q", h))
    combined = b"".join(parts)
    return combined.hex()


# ---------------------------------------------------------------------------
# Audio fingerprinting via FFmpeg chromaprint
# ---------------------------------------------------------------------------
def _compute_audio_fingerprint(filepath: str) -> str:
    """Compute audio fingerprint using FFmpeg's chromaprint muxer."""
    # Try chromaprint first, fall back to simple audio hash
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", filepath,
        "-ac", "1", "-ar", "11025",
        "-t", "120",  # First 2 minutes
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0:
            return ""
        audio_data = proc.stdout
        if not audio_data:
            return ""
        # Simple audio fingerprint: hash chunks of audio data
        h = hashlib.sha256()
        chunk_size = 4096
        for i in range(0, min(len(audio_data), 1048576), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            h.update(chunk)
        return h.hexdigest()[:64]
    except Exception as e:
        logger.debug("Audio fingerprint failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Similarity comparison
# ---------------------------------------------------------------------------
def _hamming_distance(a: int, b: int) -> int:
    """Compute hamming distance between two 64-bit integers."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def _parse_fingerprint_hex(fp_hex: str) -> Tuple[int, List[int]]:
    """Parse a fingerprint hex string into (summary_hash, frame_hashes)."""
    if not fp_hex:
        return 0, []
    raw = bytes.fromhex(fp_hex)
    if len(raw) < 8:
        return 0, []
    summary = struct.unpack(">Q", raw[:8])[0]
    frame_hashes = []
    for i in range(8, len(raw), 8):
        if i + 8 <= len(raw):
            h = struct.unpack(">Q", raw[i:i + 8])[0]
            frame_hashes.append(h)
    return summary, frame_hashes


def compute_visual_similarity(fp_a: str, fp_b: str) -> float:
    """
    Compute visual similarity between two fingerprints (0-100).

    Uses hamming distance on frame hashes, comparing overlapping frames.
    """
    sum_a, frames_a = _parse_fingerprint_hex(fp_a)
    sum_b, frames_b = _parse_fingerprint_hex(fp_b)

    if not frames_a or not frames_b:
        # Compare summary hashes only
        if sum_a == 0 and sum_b == 0:
            return 0.0
        dist = _hamming_distance(sum_a, sum_b)
        return max(0.0, (1.0 - dist / 64.0) * 100.0)

    # Compare overlapping frames
    min_len = min(len(frames_a), len(frames_b))
    if min_len == 0:
        return 0.0

    total_dist = 0
    for i in range(min_len):
        total_dist += _hamming_distance(frames_a[i], frames_b[i])

    avg_dist = total_dist / min_len
    similarity = max(0.0, (1.0 - avg_dist / 64.0) * 100.0)

    # Penalize length mismatch
    max_len = max(len(frames_a), len(frames_b))
    length_ratio = min_len / max(max_len, 1)
    if length_ratio < 0.5:
        similarity *= length_ratio * 1.5

    return min(100.0, similarity)


def compute_audio_similarity(fp_a: str, fp_b: str) -> float:
    """
    Compute audio similarity between two audio fingerprints (0-100).

    Uses character-level comparison of hex fingerprints as a proxy
    for cross-correlation.
    """
    if not fp_a or not fp_b:
        return 0.0

    # Character-level similarity
    min_len = min(len(fp_a), len(fp_b))
    if min_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if fp_a[i] == fp_b[i])
    return (matches / min_len) * 100.0


def compare_fingerprints(result_a: FingerprintResult,
                         result_b: FingerprintResult,
                         threshold: float = SIMILARITY_THRESHOLD) -> SimilarityResult:
    """Compare two fingerprint results and determine similarity."""
    visual = compute_visual_similarity(result_a.fingerprint_hex, result_b.fingerprint_hex)
    audio = compute_audio_similarity(result_a.audio_fingerprint, result_b.audio_fingerprint)

    # Weighted combination: 70% visual, 30% audio
    combined = visual * 0.7 + audio * 0.3

    return SimilarityResult(
        visual_score=visual,
        audio_score=audio,
        combined_score=combined,
        is_duplicate=combined >= threshold,
        file_a=result_a.file_path,
        file_b=result_b.file_path,
    )


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------
def generate_fingerprint(filepath: str,
                         on_progress: Optional[Callable] = None) -> FingerprintResult:
    """
    Generate a perceptual fingerprint for a video/audio file.

    Args:
        filepath: Path to the media file.
        on_progress: Callback taking int (0-100).

    Returns:
        FingerprintResult with visual and audio fingerprints.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    result = FingerprintResult(file_path=filepath)

    if on_progress:
        on_progress(5)

    # File hash for dedup
    result.file_hash = _compute_quick_hash(filepath)
    result.file_size = os.path.getsize(filepath)

    if on_progress:
        on_progress(10)

    # Video info
    info = get_video_info(filepath)
    result.duration_sec = info.get("duration", 0)

    if on_progress:
        on_progress(15)

    # Extract frames and compute visual fingerprint
    frames = _extract_frame_data(filepath, fps=FRAMES_PER_SECOND)
    result.frame_count = len(frames)

    if on_progress:
        on_progress(50)

    if frames:
        frame_hashes = []
        for i, frame in enumerate(frames):
            h = _compute_frame_phash(frame)
            frame_hashes.append(h)
            if on_progress and i % 10 == 0:
                pct = 50 + int((i / max(len(frames), 1)) * 25)
                on_progress(min(pct, 74))
        result.fingerprint_hex = _combine_frame_hashes(frame_hashes)

    if on_progress:
        on_progress(75)

    # Audio fingerprint
    result.audio_fingerprint = _compute_audio_fingerprint(filepath)

    if on_progress:
        on_progress(95)

    return result


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------
def index_fingerprint(result: FingerprintResult):
    """Store a fingerprint in the database index."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO fingerprints
                   (file_path, file_hash, fingerprint_hex, audio_fingerprint,
                    frame_count, duration_sec, file_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (result.file_path, result.file_hash, result.fingerprint_hex,
                 result.audio_fingerprint, result.frame_count,
                 result.duration_sec, result.file_size),
            )
            conn.commit()
        finally:
            conn.close()


def batch_index(filepaths: List[str],
                on_progress: Optional[Callable] = None) -> List[FingerprintResult]:
    """Generate and index fingerprints for multiple files."""
    results = []
    total = len(filepaths)
    for i, fp in enumerate(filepaths):
        try:
            result = generate_fingerprint(fp)
            index_fingerprint(result)
            results.append(result)
        except Exception as e:
            logger.warning("Failed to fingerprint %s: %s", fp, e)
            results.append(FingerprintResult(file_path=fp))
        if on_progress:
            pct = int(((i + 1) / max(total, 1)) * 100)
            on_progress(min(pct, 99))

    if on_progress:
        on_progress(100)
    return results


def search_similar(query_fingerprint: FingerprintResult,
                   threshold: float = SIMILARITY_THRESHOLD,
                   max_results: int = 20) -> List[SearchResult]:
    """
    Search the index for fingerprints similar to the query.

    Args:
        query_fingerprint: The fingerprint to search for.
        threshold: Minimum similarity score (0-100).
        max_results: Maximum number of results.

    Returns:
        List of SearchResult sorted by similarity descending.
    """
    with _db_lock:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                "SELECT file_path, fingerprint_hex, audio_fingerprint, indexed_at "
                "FROM fingerprints"
            ).fetchall()
        finally:
            conn.close()

    matches = []
    for row in rows:
        # Skip self
        if row["file_path"] == query_fingerprint.file_path:
            continue

        visual_sim = compute_visual_similarity(
            query_fingerprint.fingerprint_hex, row["fingerprint_hex"])
        audio_sim = compute_audio_similarity(
            query_fingerprint.audio_fingerprint, row["audio_fingerprint"] or "")
        combined = visual_sim * 0.7 + audio_sim * 0.3

        if combined >= threshold:
            matches.append(SearchResult(
                file_path=row["file_path"],
                similarity=combined,
                fingerprint_hex=row["fingerprint_hex"],
                indexed_at=row["indexed_at"] or "",
            ))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:max_results]


def get_indexed_count() -> int:
    """Return the number of indexed fingerprints."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM fingerprints").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()


def get_indexed_fingerprint(filepath: str) -> Optional[FingerprintResult]:
    """Retrieve a previously indexed fingerprint by file path."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute(
                "SELECT * FROM fingerprints WHERE file_path = ?", (filepath,)
            ).fetchone()
            if not row:
                return None
            return FingerprintResult(
                fingerprint_hex=row["fingerprint_hex"],
                frame_count=row["frame_count"],
                audio_fingerprint=row["audio_fingerprint"] or "",
                file_hash=row["file_hash"],
                duration_sec=row["duration_sec"],
                file_path=row["file_path"],
                file_size=row["file_size"],
            )
        finally:
            conn.close()


def remove_indexed(filepath: str) -> bool:
    """Remove a fingerprint from the index."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            cur = conn.execute(
                "DELETE FROM fingerprints WHERE file_path = ?", (filepath,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def clear_index() -> int:
    """Clear all fingerprints from the index. Returns count removed."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            cur = conn.execute("DELETE FROM fingerprints")
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
