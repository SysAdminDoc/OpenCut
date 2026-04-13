"""
OpenCut Batch Transcode Module v1.0.0

Generates a job matrix of files x presets and executes each combination
using ``export_with_preset`` from the export presets module.  Supports
parallel execution and per-file progress tracking.

Output organization:
    output_dir/{preset_name}/{original_filename}
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TranscodeItemResult:
    """Result for a single file+preset combination."""
    input_path: str
    preset: str
    output_path: str = ""
    status: str = "pending"  # pending, running, complete, failed
    error: str = ""
    duration_seconds: float = 0.0


@dataclass
class BatchTranscodeResult:
    """Result for the entire batch transcode operation."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[TranscodeItemResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Size / time estimation
# ---------------------------------------------------------------------------

# Rough bitrate estimates per preset category (Mbps) for estimation
_BITRATE_ESTIMATES = {
    "youtube": 12.0,
    "social": 6.0,
    "web": 8.0,
    "audio": 0.2,
    "archive": 100.0,
    "hw_accel": 12.0,
    "prores": 150.0,
    "dnxhr": 120.0,
    "av1": 5.0,
}

# Rough encode speed multipliers (how many times realtime)
_SPEED_ESTIMATES = {
    "youtube": 2.0,
    "social": 2.5,
    "web": 1.5,
    "audio": 10.0,
    "archive": 3.0,
    "hw_accel": 5.0,
    "prores": 4.0,
    "dnxhr": 4.0,
    "av1": 0.3,
}


def estimate_batch_size(
    file_paths: List[str],
    preset_names: List[str],
) -> dict:
    """Estimate total output size and time for a batch transcode job.

    Args:
        file_paths: List of input file paths.
        preset_names: List of preset names from EXPORT_PRESETS.

    Returns:
        Dict with ``total_files``, ``total_combinations``,
        ``estimated_size_mb``, ``estimated_time_seconds``,
        ``per_preset`` breakdown.
    """
    from opencut.core.export_presets import EXPORT_PRESETS

    total_duration = 0.0
    for fp in file_paths:
        if os.path.isfile(fp):
            info = get_video_info(fp)
            total_duration += info.get("duration", 0)

    per_preset = []
    total_size_mb = 0.0
    total_time_sec = 0.0

    for preset_name in preset_names:
        preset = EXPORT_PRESETS.get(preset_name)
        if not preset:
            per_preset.append({
                "preset": preset_name,
                "error": "Unknown preset",
                "estimated_size_mb": 0,
                "estimated_time_seconds": 0,
            })
            continue

        category = preset.get("category", "web")
        bitrate_mbps = _BITRATE_ESTIMATES.get(category, 8.0)
        speed_mult = _SPEED_ESTIMATES.get(category, 1.0)

        # Size = bitrate * duration (per file, summed)
        size_mb = (bitrate_mbps * total_duration) / 8.0  # Mbps to MB
        time_sec = total_duration / max(speed_mult, 0.01)

        per_preset.append({
            "preset": preset_name,
            "label": preset.get("label", preset_name),
            "estimated_size_mb": round(size_mb, 1),
            "estimated_time_seconds": round(time_sec, 1),
        })

        total_size_mb += size_mb
        total_time_sec += time_sec

    return {
        "total_files": len(file_paths),
        "total_combinations": len(file_paths) * len(preset_names),
        "total_source_duration_seconds": round(total_duration, 1),
        "estimated_size_mb": round(total_size_mb, 1),
        "estimated_time_seconds": round(total_time_sec, 1),
        "per_preset": per_preset,
    }


# ---------------------------------------------------------------------------
# Batch transcode
# ---------------------------------------------------------------------------

def batch_transcode(
    file_paths: List[str],
    preset_names: List[str],
    output_base_dir: Optional[str] = None,
    parallel: int = 1,
    on_progress: Optional[Callable] = None,
) -> BatchTranscodeResult:
    """Transcode multiple files across multiple presets.

    Generates a matrix of ``len(file_paths) * len(preset_names)`` jobs
    and runs them sequentially or in parallel.

    Args:
        file_paths: List of input video/audio file paths.
        preset_names: List of preset keys from ``EXPORT_PRESETS``.
        output_base_dir: Root output directory.  Files are organized as
            ``output_base_dir/{preset_name}/{filename}``.
            If *None*, uses the directory of each input file.
        parallel: Maximum number of concurrent encodes (1 = sequential).
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`BatchTranscodeResult` with per-item status.
    """
    from opencut.core.export_presets import EXPORT_PRESETS, export_with_preset

    if not file_paths:
        raise ValueError("No files provided for batch transcode.")
    if not preset_names:
        raise ValueError("No presets provided for batch transcode.")

    # Validate presets
    valid_presets = []
    for p in preset_names:
        if p in EXPORT_PRESETS:
            valid_presets.append(p)
        else:
            logger.warning("Skipping unknown preset: %s", p)

    if not valid_presets:
        raise ValueError("No valid presets provided.")

    parallel = max(1, min(parallel, 8))  # Clamp 1-8

    # Build the job matrix
    matrix: List[TranscodeItemResult] = []
    for fp in file_paths:
        for preset_name in valid_presets:
            matrix.append(TranscodeItemResult(
                input_path=fp,
                preset=preset_name,
            ))

    result = BatchTranscodeResult(total=len(matrix))

    if on_progress:
        on_progress(5, f"Starting batch transcode: {len(matrix)} combinations...")

    start_time = time.time()
    completed_count = 0

    def _process_item(item: TranscodeItemResult) -> TranscodeItemResult:
        """Process a single file+preset combination."""
        nonlocal completed_count

        if not os.path.isfile(item.input_path):
            item.status = "failed"
            item.error = f"File not found: {item.input_path}"
            return item

        # Determine output directory
        preset = EXPORT_PRESETS.get(item.preset, {})
        ext = preset.get("ext", ".mp4")
        base_name = os.path.splitext(os.path.basename(item.input_path))[0]

        if output_base_dir:
            preset_dir = os.path.join(output_base_dir, item.preset)
        else:
            preset_dir = os.path.join(os.path.dirname(item.input_path), item.preset)

        os.makedirs(preset_dir, exist_ok=True)
        out_path = os.path.join(preset_dir, f"{base_name}{ext}")

        item.status = "running"
        item_start = time.time()

        try:
            export_with_preset(
                input_path=item.input_path,
                preset_name=item.preset,
                output_path=out_path,
            )
            item.output_path = out_path
            item.status = "complete"
        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            logger.error(
                "Batch transcode failed for %s + %s: %s",
                item.input_path, item.preset, e,
            )

        item.duration_seconds = round(time.time() - item_start, 2)

        completed_count += 1
        if on_progress:
            pct = min(int((completed_count / len(matrix)) * 90) + 5, 95)
            on_progress(
                pct,
                f"Completed {completed_count}/{len(matrix)}: "
                f"{os.path.basename(item.input_path)} -> {item.preset}",
            )

        return item

    # Execute the matrix
    if parallel <= 1:
        # Sequential
        for item in matrix:
            _process_item(item)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_process_item, item): item for item in matrix}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    item.status = "failed"
                    item.error = str(e)

    result.total_duration_seconds = round(time.time() - start_time, 2)
    result.results = matrix
    result.completed = sum(1 for i in matrix if i.status == "complete")
    result.failed = sum(1 for i in matrix if i.status == "failed")
    result.skipped = sum(1 for i in matrix if i.status == "pending")

    if on_progress:
        on_progress(
            100,
            f"Batch transcode complete: {result.completed}/{result.total} succeeded",
        )

    return result
