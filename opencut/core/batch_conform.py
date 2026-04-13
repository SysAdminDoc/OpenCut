"""
OpenCut Batch Conforming Module v1.0.0

Conform mixed files to a single target spec (fps, resolution, codec,
audio sample rate). Analyzes deviations and re-encodes as needed.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

@dataclass
class ConformSpec:
    """Target specification for batch conforming."""
    width: int = 1920
    height: int = 1080
    fps: float = 24.0
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConformSpec":
        """Create a ConformSpec from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformFileResult:
    """Result for a single conformed file."""
    input_path: str
    output_path: str = ""
    status: str = "pending"  # pending, conformed, copied, failed
    deviations: List[str] = field(default_factory=list)
    needs_conform: bool = False
    error: str = ""
    duration_seconds: float = 0.0


@dataclass
class ConformBatchResult:
    """Result for the entire batch conform operation."""
    total: int = 0
    conformed: int = 0
    copied: int = 0
    failed: int = 0
    results: List[ConformFileResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Conformance analysis
# ---------------------------------------------------------------------------

def _analyze_single(file_path: str, target: ConformSpec) -> ConformFileResult:
    """Analyze a single file against the target spec."""
    result = ConformFileResult(input_path=file_path)

    if not os.path.isfile(file_path):
        result.status = "failed"
        result.error = "File not found"
        return result

    try:
        info = get_video_info(file_path)
    except Exception as e:
        result.status = "failed"
        result.error = f"Failed to probe: {e}"
        return result

    # Check deviations
    if info["width"] != target.width or info["height"] != target.height:
        result.deviations.append(
            f"resolution: {info['width']}x{info['height']} -> {target.width}x{target.height}"
        )

    if abs(info["fps"] - target.fps) > 0.01:
        result.deviations.append(
            f"fps: {info['fps']:.2f} -> {target.fps:.2f}"
        )

    result.needs_conform = len(result.deviations) > 0
    return result


def analyze_conformance_batch(
    file_paths: List[str],
    target_spec: Dict,
) -> List[dict]:
    """Analyze multiple files against a target spec.

    Args:
        file_paths: List of media file paths.
        target_spec: Dict with target specification fields.

    Returns:
        List of analysis dicts per file with deviations.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    target = ConformSpec.from_dict(target_spec) if isinstance(target_spec, dict) else target_spec

    results = []
    for fpath in file_paths:
        analysis = _analyze_single(fpath, target)
        results.append({
            "file_path": analysis.input_path,
            "needs_conform": analysis.needs_conform,
            "deviations": analysis.deviations,
            "status": analysis.status,
            "error": analysis.error,
        })

    return results


# ---------------------------------------------------------------------------
# Batch conforming
# ---------------------------------------------------------------------------

def conform_batch(
    file_paths: List[str],
    target_spec: Dict,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ConformBatchResult:
    """Conform multiple files to a target specification.

    Files that already match the target are stream-copied.
    Files with deviations are re-encoded.

    Args:
        file_paths: List of input file paths.
        target_spec: Dict with target specification fields.
        output_dir: Output directory. Defaults to input file's directory.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`ConformBatchResult` with per-file status.
    """
    if not file_paths:
        raise ValueError("No files provided for conforming")

    target = ConformSpec.from_dict(target_spec) if isinstance(target_spec, dict) else target_spec
    batch_result = ConformBatchResult(total=len(file_paths))
    start_time = time.time()

    if on_progress:
        on_progress(5, f"Conforming {len(file_paths)} files...")

    for i, fpath in enumerate(file_paths):
        file_result = _analyze_single(fpath, target)

        if file_result.status == "failed":
            batch_result.failed += 1
            batch_result.results.append(file_result)
            continue

        # Determine output path
        base = os.path.splitext(os.path.basename(fpath))[0]
        ext = ".mp4"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{base}_conformed{ext}")
        else:
            out_path = os.path.join(
                os.path.dirname(fpath), f"{base}_conformed{ext}"
            )

        item_start = time.time()

        try:
            if file_result.needs_conform:
                # Build FFmpeg command for re-encoding
                vf_parts = []
                vf_parts.append(
                    f"scale={target.width}:{target.height}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"pad={target.width}:{target.height}:(ow-iw)/2:(oh-ih)/2"
                )

                cmd = (
                    FFmpegCmd()
                    .input(fpath)
                    .video_codec(target.video_codec, crf=target.crf, preset=target.preset,
                                 pix_fmt=target.pix_fmt)
                    .audio_codec(target.audio_codec, bitrate="192k")
                    .video_filter(",".join(vf_parts))
                    .option("r", str(target.fps))
                    .option("ar", str(target.audio_sample_rate))
                    .option("ac", str(target.audio_channels))
                    .faststart()
                    .output(out_path)
                    .build()
                )

                run_ffmpeg(cmd)
                file_result.status = "conformed"
                file_result.output_path = out_path
                batch_result.conformed += 1

            else:
                # Stream copy - already conforms
                cmd = (
                    FFmpegCmd()
                    .input(fpath)
                    .copy_streams()
                    .faststart()
                    .output(out_path)
                    .build()
                )

                run_ffmpeg(cmd)
                file_result.status = "copied"
                file_result.output_path = out_path
                batch_result.copied += 1

        except Exception as e:
            file_result.status = "failed"
            file_result.error = str(e)
            batch_result.failed += 1
            logger.error("Conform failed for %s: %s", fpath, e)

        file_result.duration_seconds = round(time.time() - item_start, 2)
        batch_result.results.append(file_result)

        if on_progress:
            pct = min(int(((i + 1) / len(file_paths)) * 90) + 5, 95)
            on_progress(
                pct,
                f"Conformed {i + 1}/{len(file_paths)}: {os.path.basename(fpath)}",
            )

    batch_result.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(
            100,
            f"Conform complete: {batch_result.conformed} re-encoded, "
            f"{batch_result.copied} copied, {batch_result.failed} failed",
        )

    return batch_result
