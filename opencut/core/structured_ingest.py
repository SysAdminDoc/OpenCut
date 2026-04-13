"""
OpenCut Structured Ingest Module v1.0.0

Rules-based ingest: auto-rename by pattern, checksum verify,
sort into folder structure, generate report.
"""

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IngestFileResult:
    """Result for a single ingested file."""
    source_path: str
    dest_path: str = ""
    new_name: str = ""
    status: str = "pending"  # pending, copied, renamed, failed, skipped
    checksum: str = ""
    checksum_ok: bool = True
    error: str = ""


@dataclass
class IngestResult:
    """Result for the entire ingest operation."""
    total: int = 0
    copied: int = 0
    failed: int = 0
    skipped: int = 0
    files: List[IngestFileResult] = field(default_factory=list)
    report_path: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

_SUPPORTED_ALGORITHMS = {"md5", "sha1", "sha256", "sha512", "xxhash"}


def verify_checksum(
    file_path: str,
    algorithm: str = "sha256",
    expected: str = "",
) -> dict:
    """Compute file checksum and optionally verify against expected value.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm (md5, sha1, sha256, sha512, xxhash).
        expected: If provided, verify the computed hash matches.

    Returns:
        Dict with ``hash``, ``algorithm``, ``verified`` (bool or None), ``match``.
    """
    algorithm = algorithm.lower().strip()
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use one of {_SUPPORTED_ALGORITHMS}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if algorithm == "xxhash":
        try:
            import xxhash
            h = xxhash.xxh64()
        except ImportError:
            raise RuntimeError("xxhash package not installed. Use pip install xxhash")
    else:
        h = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    computed = h.hexdigest()
    result = {
        "hash": computed,
        "algorithm": algorithm,
        "file_path": file_path,
        "verified": None,
        "match": None,
    }

    if expected:
        match = computed.lower() == expected.lower().strip()
        result["verified"] = True
        result["match"] = match

    return result


# ---------------------------------------------------------------------------
# Rename by pattern
# ---------------------------------------------------------------------------

def rename_by_pattern(
    filename: str,
    pattern: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Generate a new filename based on a naming pattern.

    Pattern placeholders:
        {name}      - original filename without extension
        {ext}       - original extension (with dot)
        {date}      - current date YYYY-MM-DD
        {counter}   - auto-increment counter (from metadata)
        {camera}    - camera ID (from metadata)
        {scene}     - scene number (from metadata)
        {take}      - take number (from metadata)
        {resolution} - e.g. 1920x1080 (from metadata)

    Args:
        filename: Original filename.
        pattern: Naming pattern with {placeholders}.
        metadata: Optional dict with values for placeholders.

    Returns:
        New filename string.
    """
    if not pattern:
        return filename

    meta = metadata or {}
    base, ext = os.path.splitext(filename)

    replacements = {
        "name": base,
        "ext": ext,
        "date": time.strftime("%Y-%m-%d"),
        "counter": str(meta.get("counter", 1)).zfill(4),
        "camera": str(meta.get("camera", "A")),
        "scene": str(meta.get("scene", "001")),
        "take": str(meta.get("take", "01")),
        "resolution": str(meta.get("resolution", "")),
    }

    result = pattern
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)

    # Sanitize: remove characters invalid in filenames
    result = re.sub(r'[<>:"/\\|?*]', '_', result)

    # Ensure extension is present
    if ext and not result.endswith(ext):
        result += ext

    return result


# ---------------------------------------------------------------------------
# Ingest report generation
# ---------------------------------------------------------------------------

def generate_ingest_report(
    results: List[IngestFileResult],
    output_path: str,
    format: str = "json",
) -> str:
    """Generate an ingest report from file results.

    Args:
        results: List of IngestFileResult objects.
        output_path: Path to write the report.
        format: Report format - 'json' or 'csv'.

    Returns:
        Path to the written report file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if format == "csv":
        if not output_path.endswith(".csv"):
            output_path = os.path.splitext(output_path)[0] + ".csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source_path", "dest_path", "new_name",
                "status", "checksum", "checksum_ok", "error",
            ])
            for r in results:
                writer.writerow([
                    r.source_path, r.dest_path, r.new_name,
                    r.status, r.checksum, r.checksum_ok, r.error,
                ])
    else:
        if not output_path.endswith(".json"):
            output_path = os.path.splitext(output_path)[0] + ".json"
        report_data = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": len(results),
            "copied": sum(1 for r in results if r.status in ("copied", "renamed")),
            "failed": sum(1 for r in results if r.status == "failed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "files": [asdict(r) for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

# Default config for ingest rules
_DEFAULT_CONFIG = {
    "rename_pattern": "",
    "folder_structure": "",       # e.g. "{date}/{camera}"
    "checksum_algorithm": "",     # empty = skip checksum
    "expected_checksums": {},     # filename -> expected hash
    "extensions_filter": [],      # empty = all files
    "skip_existing": True,
    "copy_mode": "copy",          # "copy" or "move"
}


def run_ingest(
    source_dir: str,
    config: Optional[Dict] = None,
    dest_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> IngestResult:
    """Run structured ingest from source directory.

    Args:
        source_dir: Directory containing source media files.
        config: Ingest configuration dict (see _DEFAULT_CONFIG).
        dest_dir: Destination directory. Defaults to source_dir + '_ingested'.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`IngestResult` with per-file status.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    if not dest_dir:
        dest_dir = source_dir.rstrip(os.sep) + "_ingested"
    os.makedirs(dest_dir, exist_ok=True)

    # Collect files
    ext_filter = set()
    if cfg.get("extensions_filter"):
        ext_filter = {e.lower().lstrip(".") for e in cfg["extensions_filter"]}

    all_files = []
    for entry in sorted(os.listdir(source_dir)):
        full_path = os.path.join(source_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if ext_filter:
            ext = os.path.splitext(entry)[1].lower().lstrip(".")
            if ext not in ext_filter:
                continue
        all_files.append(full_path)

    if not all_files:
        raise ValueError(f"No matching files found in {source_dir}")

    result = IngestResult(total=len(all_files))
    start_time = time.time()

    if on_progress:
        on_progress(5, f"Ingesting {len(all_files)} files...")

    expected_checksums = cfg.get("expected_checksums", {})
    counter = 1

    for i, src_path in enumerate(all_files):
        filename = os.path.basename(src_path)
        file_result = IngestFileResult(source_path=src_path)

        try:
            # Build metadata for renaming
            meta = {"counter": counter}
            try:
                info = get_video_info(src_path)
                meta["resolution"] = f"{info['width']}x{info['height']}"
            except Exception:
                pass

            # Rename if pattern provided
            new_name = filename
            if cfg.get("rename_pattern"):
                new_name = rename_by_pattern(filename, cfg["rename_pattern"], meta)
                file_result.new_name = new_name

            # Build folder structure
            sub_dir = ""
            if cfg.get("folder_structure"):
                sub_dir = cfg["folder_structure"]
                sub_dir = sub_dir.replace("{date}", time.strftime("%Y-%m-%d"))
                sub_dir = sub_dir.replace("{camera}", str(meta.get("camera", "default")))
                sub_dir = re.sub(r'[<>:"|?*]', '_', sub_dir)

            target_dir = os.path.join(dest_dir, sub_dir) if sub_dir else dest_dir
            os.makedirs(target_dir, exist_ok=True)
            dest_path = os.path.join(target_dir, new_name)

            # Skip existing
            if cfg.get("skip_existing") and os.path.exists(dest_path):
                file_result.status = "skipped"
                file_result.dest_path = dest_path
                result.skipped += 1
                result.files.append(file_result)
                counter += 1
                continue

            # Copy or move
            if cfg.get("copy_mode") == "move":
                shutil.move(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

            file_result.dest_path = dest_path
            file_result.status = "renamed" if new_name != filename else "copied"

            # Checksum verification
            if cfg.get("checksum_algorithm"):
                algo = cfg["checksum_algorithm"]
                expected = expected_checksums.get(filename, "")
                cksum = verify_checksum(dest_path, algo, expected)
                file_result.checksum = cksum["hash"]
                if expected and not cksum.get("match", True):
                    file_result.checksum_ok = False
                    file_result.error = f"Checksum mismatch: expected {expected}, got {cksum['hash']}"
                    file_result.status = "failed"
                    result.failed += 1
                    result.files.append(file_result)
                    counter += 1
                    continue

            result.copied += 1

        except Exception as e:
            file_result.status = "failed"
            file_result.error = str(e)
            result.failed += 1
            logger.error("Ingest failed for %s: %s", src_path, e)

        result.files.append(file_result)
        counter += 1

        if on_progress:
            pct = min(int(((i + 1) / len(all_files)) * 90) + 5, 95)
            on_progress(pct, f"Ingested {i + 1}/{len(all_files)}: {filename}")

    result.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(100, f"Ingest complete: {result.copied} copied, {result.failed} failed")

    return result
