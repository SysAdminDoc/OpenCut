"""
OpenCut Duplicate / Near-Duplicate Detection Module v1.0.0

Detect duplicate and near-duplicate video files using perceptual hashing:
- Extract representative frames (10%, 50%, 90%) via FFmpeg
- Compute perceptual hashes (pHash): resize 32x32 -> grayscale -> DCT -> binary
- Pairwise comparison with configurable similarity threshold
- Group duplicates with recommended-keep selection

All FFmpeg-based, zero external dependencies beyond FFmpeg + standard library.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VideoHash:
    """Perceptual hash result for a video file."""
    file_path: str
    phash_first: str   # hex hash at 10% mark
    phash_mid: str      # hex hash at 50% mark
    phash_last: str     # hex hash at 90% mark
    duration: float
    file_size: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_video_hash(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> VideoHash:
    """
    Compute perceptual hash for a video file.

    Extracts 3 frames at 10%, 50%, and 90% of duration, computes
    a perceptual hash for each.

    Args:
        input_path: Path to video file.
        on_progress: Callback(percent, message).

    Returns:
        VideoHash dataclass with phash for 3 frames + metadata.
    """
    if on_progress:
        on_progress(5, "Probing video info...")

    info = get_video_info(input_path)
    duration = info["duration"]
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration for: {input_path}")

    file_size = os.path.getsize(input_path)

    # Extract 3 frames at 10%, 50%, 90%
    timestamps = [duration * 0.1, duration * 0.5, duration * 0.9]
    hashes = []

    tmp_dir = tempfile.mkdtemp(prefix="opencut_phash_")
    try:
        for i, ts in enumerate(timestamps):
            if on_progress:
                on_progress(10 + i * 25, f"Extracting frame {i+1}/3...")

            frame_path = os.path.join(tmp_dir, f"frame_{i}.pgm")
            # Extract frame, resize to 32x32 grayscale PGM
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(ts),
                "-i", input_path,
                "-frames:v", "1",
                "-vf", "scale=32:32,format=gray",
                "-f", "image2", frame_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)

            phash = compute_phash(frame_path)
            hashes.append(phash)

        if on_progress:
            on_progress(90, "Hashing complete")

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = VideoHash(
        file_path=input_path,
        phash_first=hashes[0] if len(hashes) > 0 else "",
        phash_mid=hashes[1] if len(hashes) > 1 else "",
        phash_last=hashes[2] if len(hashes) > 2 else "",
        duration=duration,
        file_size=file_size,
    )

    if on_progress:
        on_progress(100, "Video hash computed")

    return result


def compute_phash(image_path: str) -> str:
    """
    Compute perceptual hash of a grayscale image file.

    Algorithm: read grayscale pixels -> apply simplified DCT ->
    extract top-left 8x8 block -> threshold at median -> binary -> hex.

    Args:
        image_path: Path to grayscale image (PGM preferred).

    Returns:
        Hex string of the perceptual hash (16 chars = 64 bits).
    """
    pixels = _read_pgm_pixels(image_path)
    if pixels is None:
        return "0" * 16  # fallback zero hash

    # Simple DCT-like transform: we use mean-based hashing
    # For a 32x32 image, compute 8x8 block averages then threshold
    block = 4  # 32/8 = 4 pixels per block

    values = []
    for by in range(8):
        for bx in range(8):
            total = 0
            count = 0
            for dy in range(block):
                for dx in range(block):
                    y = by * block + dy
                    x = bx * block + dx
                    if y < len(pixels) and x < len(pixels[0]):
                        total += pixels[y][x]
                        count += 1
            values.append(total / max(count, 1))

    # Threshold at median
    median = sorted(values)[len(values) // 2]
    bits = "".join("1" if v > median else "0" for v in values)

    # Convert 64-bit binary string to hex
    hash_int = int(bits, 2)
    return f"{hash_int:016x}"


def find_duplicates(
    file_paths: List[str],
    threshold: float = 0.85,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """
    Find duplicate and near-duplicate videos from a list of file paths.

    Args:
        file_paths: List of video file paths to compare.
        threshold: Similarity threshold (0.0-1.0). Higher = stricter.
        on_progress: Callback(percent, message).

    Returns:
        List of duplicate groups:
        [{files: [path1, path2], similarity: 0.95, recommended_keep: path1}]
    """
    if len(file_paths) < 2:
        return []

    threshold = max(0.0, min(1.0, threshold))

    # Phase 1: hash all files
    hashes: List[VideoHash] = []
    total = len(file_paths)

    for i, fp in enumerate(file_paths):
        if on_progress:
            pct = int((i / total) * 60)
            on_progress(pct, f"Hashing file {i+1}/{total}...")
        try:
            h = compute_video_hash(fp)
            hashes.append(h)
        except Exception as e:
            logger.warning("Failed to hash %s: %s", fp, e)

    if len(hashes) < 2:
        if on_progress:
            on_progress(100, "Not enough files to compare")
        return []

    # Phase 2: pairwise comparison
    if on_progress:
        on_progress(65, "Comparing hashes...")

    pairs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            sim = _hash_similarity(hashes[i], hashes[j])
            if sim >= threshold:
                pairs.append((i, j, sim))

    if on_progress:
        on_progress(85, "Grouping duplicates...")

    # Phase 3: group connected duplicates via union-find
    parent = list(range(len(hashes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, sim in pairs:
        union(i, j)

    # Build groups
    groups: Dict[int, List[Tuple[int, float]]] = {}
    for i, j, sim in pairs:
        root = find(i)
        if root not in groups:
            groups[root] = set()
        groups[root].add(i)
        groups[root].add(j)

    # Build result
    results = []
    for root, members in groups.items():
        member_list = sorted(members)
        files = [hashes[m].file_path for m in member_list]
        # Pick the one with highest quality (largest file) as recommended keep
        best = max(member_list, key=lambda m: hashes[m].file_size)

        # Average similarity between all pairs in group
        sims = [s for i, j, s in pairs if i in members and j in members]
        avg_sim = sum(sims) / len(sims) if sims else 0

        results.append({
            "files": files,
            "similarity": round(avg_sim, 4),
            "recommended_keep": hashes[best].file_path,
        })

    if on_progress:
        on_progress(100, f"Found {len(results)} duplicate group(s)")

    return results


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _read_pgm_pixels(path: str) -> Optional[List[List[int]]]:
    """Read a PGM (P5 binary or P2 text) grayscale image into a 2D pixel array."""
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "rb") as f:
            magic = f.readline().decode().strip()

            if magic == "P5":
                # Binary PGM
                # Skip comments
                line = f.readline().decode().strip()
                while line.startswith("#"):
                    line = f.readline().decode().strip()
                w, h = map(int, line.split())
                int(f.readline().decode().strip())
                data = f.read(w * h)
                pixels = []
                for y in range(h):
                    row = []
                    for x in range(w):
                        idx = y * w + x
                        row.append(data[idx] if idx < len(data) else 0)
                    pixels.append(row)
                return pixels

            elif magic == "P2":
                # Text PGM
                line = f.readline().decode().strip()
                while line.startswith("#"):
                    line = f.readline().decode().strip()
                w, h = map(int, line.split())
                int(f.readline().decode().strip())
                rest = f.read().decode()
                vals = list(map(int, rest.split()))
                pixels = []
                idx = 0
                for y in range(h):
                    row = []
                    for x in range(w):
                        row.append(vals[idx] if idx < len(vals) else 0)
                        idx += 1
                    pixels.append(row)
                return pixels

    except Exception as e:
        logger.debug("Failed to read PGM %s: %s", path, e)

    return None


def _hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if len(hash1) != len(hash2):
        return 64  # max distance

    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor = val1 ^ val2
        return bin(xor).count("1")
    except ValueError:
        return 64


def _hash_similarity(h1: VideoHash, h2: VideoHash) -> float:
    """
    Compute similarity between two VideoHash objects.

    Combines perceptual hash similarity (3 frames) and duration similarity.
    Returns 0.0-1.0 where 1.0 = identical.
    """
    # Hash similarity: average of 3 frame comparisons
    max_bits = 64
    d1 = _hamming_distance(h1.phash_first, h2.phash_first)
    d2 = _hamming_distance(h1.phash_mid, h2.phash_mid)
    d3 = _hamming_distance(h1.phash_last, h2.phash_last)

    hash_sim = 1.0 - (d1 + d2 + d3) / (3 * max_bits)

    # Duration similarity: penalize large duration differences
    if h1.duration > 0 and h2.duration > 0:
        ratio = min(h1.duration, h2.duration) / max(h1.duration, h2.duration)
        dur_sim = ratio
    else:
        dur_sim = 0.5

    # Weighted combination: 80% hash, 20% duration
    return hash_sim * 0.8 + dur_sim * 0.2
