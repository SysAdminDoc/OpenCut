"""
OpenCut Batch & Workflow Engine v1.0.0

Multi-file batch processing with chained workflow operations.

Features:
  - Batch queue: Process multiple files through any single operation
  - Workflow chains: Chain multiple operations sequentially per file
  - Workflow presets: Save/load named operation chains
  - Media inspector: Detailed codec/format/stream analysis
  - Watch folder: Monitor directory and auto-process new files

All processing uses existing OpenCut core modules.
"""

import json
import logging
import os
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Media Inspector
# ---------------------------------------------------------------------------
def inspect_media(filepath: str) -> Dict:
    """
    Detailed media analysis including all streams, codec info,
    container metadata, bitrates, and technical properties.
    """
    import subprocess

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams", "-show_chapters",
        "-count_frames", "-count_packets",
        filepath,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {r.stderr[:300]}")
        raw = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse ffprobe output")

    # Build structured result
    result = {
        "file": {
            "path": filepath,
            "name": os.path.basename(filepath),
            "size_bytes": os.path.getsize(filepath),
            "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
            "extension": os.path.splitext(filepath)[1].lower(),
        },
        "container": {},
        "video": None,
        "audio": None,
        "subtitle_tracks": [],
        "chapters": [],
        "all_streams": [],
    }

    # Container info
    fmt = raw.get("format", {})
    result["container"] = {
        "format": fmt.get("format_name", ""),
        "format_long": fmt.get("format_long_name", ""),
        "duration": round(float(fmt.get("duration", 0)), 3),
        "bitrate": int(fmt.get("bit_rate", 0)),
        "bitrate_str": _format_bitrate(int(fmt.get("bit_rate", 0))),
        "streams": int(fmt.get("nb_streams", 0)),
        "tags": fmt.get("tags", {}),
    }

    # Streams
    for stream in raw.get("streams", []):
        codec_type = stream.get("codec_type", "")
        stream_info = {
            "index": stream.get("index", 0),
            "codec_type": codec_type,
            "codec_name": stream.get("codec_name", ""),
            "codec_long_name": stream.get("codec_long_name", ""),
            "profile": stream.get("profile", ""),
            "tags": stream.get("tags", {}),
        }

        if codec_type == "video":
            vid = {
                **stream_info,
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "resolution": f"{stream.get('width', 0)}x{stream.get('height', 0)}",
                "display_aspect_ratio": stream.get("display_aspect_ratio", ""),
                "pixel_format": stream.get("pix_fmt", ""),
                "color_space": stream.get("color_space", ""),
                "color_range": stream.get("color_range", ""),
                "color_transfer": stream.get("color_transfer", ""),
                "color_primaries": stream.get("color_primaries", ""),
                "fps": _parse_fps(stream.get("r_frame_rate", "0/1")),
                "avg_fps": _parse_fps(stream.get("avg_frame_rate", "0/1")),
                "bitrate": int(stream.get("bit_rate", 0)),
                "bitrate_str": _format_bitrate(int(stream.get("bit_rate", 0))),
                "duration": round(float(stream.get("duration", 0)), 3) if stream.get("duration") else None,
                "frames": stream.get("nb_frames", ""),
                "bit_depth": stream.get("bits_per_raw_sample", ""),
                "level": stream.get("level", ""),
                "field_order": stream.get("field_order", ""),
            }
            if result["video"] is None:
                result["video"] = vid
            stream_info.update(vid)

        elif codec_type == "audio":
            aud = {
                **stream_info,
                "sample_rate": int(stream.get("sample_rate", 0)),
                "channels": int(stream.get("channels", 0)),
                "channel_layout": stream.get("channel_layout", ""),
                "sample_format": stream.get("sample_fmt", ""),
                "bitrate": int(stream.get("bit_rate", 0)),
                "bitrate_str": _format_bitrate(int(stream.get("bit_rate", 0))),
                "duration": round(float(stream.get("duration", 0)), 3) if stream.get("duration") else None,
                "bit_depth": stream.get("bits_per_raw_sample", ""),
            }
            if result["audio"] is None:
                result["audio"] = aud
            stream_info.update(aud)

        elif codec_type == "subtitle":
            sub = {
                **stream_info,
                "language": stream.get("tags", {}).get("language", ""),
                "title": stream.get("tags", {}).get("title", ""),
            }
            result["subtitle_tracks"].append(sub)

        result["all_streams"].append(stream_info)

    # Chapters
    for ch in raw.get("chapters", []):
        result["chapters"].append({
            "id": ch.get("id", 0),
            "start": round(float(ch.get("start_time", 0)), 3),
            "end": round(float(ch.get("end_time", 0)), 3),
            "title": ch.get("tags", {}).get("title", ""),
        })

    return result


def _parse_fps(rate_str: str) -> float:
    """Parse fractional FPS string like '30000/1001'."""
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            return round(float(num) / float(den), 3) if float(den) else 0
        return round(float(rate_str), 3)
    except Exception:
        return 0.0


def _format_bitrate(bps: int) -> str:
    """Format bitrate to human-readable string."""
    if bps <= 0:
        return ""
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} Mbps"
    if bps >= 1_000:
        return f"{bps / 1_000:.0f} kbps"
    return f"{bps} bps"


# ---------------------------------------------------------------------------
# Workflow Preset Definitions
# ---------------------------------------------------------------------------
WORKFLOW_PRESETS = {
    "youtube_ready": {
        "label": "YouTube Ready",
        "description": "Denoise audio, normalize loudness, export for YouTube 1080p",
        "steps": [
            {"operation": "denoise", "params": {"strength": "moderate"}},
            {"operation": "normalize", "params": {"target_lufs": -14}},
            {"operation": "platform_export", "params": {"preset": "youtube_1080"}},
        ],
    },
    "podcast_clean": {
        "label": "Podcast Clean",
        "description": "Denoise, remove silence, normalize to podcast standards, export MP3",
        "steps": [
            {"operation": "denoise", "params": {"strength": "moderate"}},
            {"operation": "silence_remove", "params": {"threshold": -35, "min_silence": 0.7}},
            {"operation": "normalize", "params": {"target_lufs": -16}},
            {"operation": "audio_extract", "params": {"codec": "libmp3lame", "bitrate": "192k"}},
        ],
    },
    "social_vertical": {
        "label": "Social Vertical (9:16)",
        "description": "Reframe to 9:16, add color grade, export for TikTok",
        "steps": [
            {"operation": "reframe", "params": {"aspect": "9:16", "mode": "center"}},
            {"operation": "color_grade", "params": {"preset": "cinematic", "intensity": 0.6}},
            {"operation": "platform_export", "params": {"preset": "tiktok"}},
        ],
    },
    "archive_master": {
        "label": "Archive Master",
        "description": "Upscale, denoise, export as ProRes for archival",
        "steps": [
            {"operation": "denoise", "params": {"strength": "light"}},
            {"operation": "custom_render", "params": {"video_codec": "prores_ks", "audio_codec": "flac"}},
        ],
    },
    "quick_gif": {
        "label": "Quick GIF",
        "description": "Extract best thumbnail + animated GIF preview",
        "steps": [
            {"operation": "thumbnail", "params": {"mode": "single", "timestamp": -1}},
            {"operation": "gif_export", "params": {"start": 0, "duration": 5, "width": 480, "fps": 15}},
        ],
    },
    "full_clean": {
        "label": "Full Clean & Grade",
        "description": "Denoise, EQ voice, normalize, apply cinematic color, export 1080p",
        "steps": [
            {"operation": "denoise", "params": {"strength": "moderate"}},
            {"operation": "eq", "params": {"preset": "voice_clarity"}},
            {"operation": "normalize", "params": {"target_lufs": -14}},
            {"operation": "color_grade", "params": {"preset": "cinematic", "intensity": 0.5}},
            {"operation": "platform_export", "params": {"preset": "youtube_1080"}},
        ],
    },
}


# ---------------------------------------------------------------------------
# Batch Queue
# ---------------------------------------------------------------------------
@dataclass
class BatchItem:
    """Single item in a batch queue."""
    id: str = ""
    input_path: str = ""
    status: str = "pending"  # pending, running, done, error
    progress: int = 0
    message: str = ""
    result: Dict = field(default_factory=dict)
    error: str = ""


@dataclass
class BatchJob:
    """A batch processing job with multiple files and one or more operations."""
    id: str = ""
    items: List[BatchItem] = field(default_factory=list)
    workflow: List[Dict] = field(default_factory=list)  # steps
    status: str = "pending"  # pending, running, done, cancelled
    created_at: float = 0.0
    completed_items: int = 0
    total_items: int = 0
    output_dir: str = ""


# Active batch jobs storage
_batch_jobs: Dict[str, BatchJob] = {}
_batch_lock = threading.Lock()


def create_batch(
    file_paths: List[str],
    workflow_steps: List[Dict],
    output_dir: str = "",
) -> str:
    """
    Create a new batch job.

    Args:
        file_paths: List of input file paths to process.
        workflow_steps: List of operation dicts, each with 'operation' and 'params'.
        output_dir: Output directory for results.

    Returns:
        Batch job ID.
    """
    job_id = str(uuid.uuid4())[:8]

    items = []
    for fp in file_paths:
        if os.path.isfile(fp):
            items.append(BatchItem(
                id=str(uuid.uuid4())[:6],
                input_path=fp,
            ))
        else:
            logger.warning(f"Batch: skipping missing file: {fp}")

    if not items:
        raise ValueError("No valid files provided")

    job = BatchJob(
        id=job_id,
        items=items,
        workflow=workflow_steps,
        status="pending",
        created_at=time.time(),
        total_items=len(items),
        output_dir=output_dir or os.path.dirname(file_paths[0]),
    )

    with _batch_lock:
        _batch_jobs[job_id] = job

    return job_id


def start_batch(job_id: str, executor_fn: Callable) -> None:
    """
    Start processing a batch job in a background thread.

    Args:
        job_id: Batch job ID.
        executor_fn: Function(input_path, steps, output_dir, progress_cb) -> Dict result.
    """
    with _batch_lock:
        job = _batch_jobs.get(job_id)
        if not job:
            raise ValueError(f"Batch job not found: {job_id}")
        job.status = "running"

    def _process():
        try:
            for i, item in enumerate(job.items):
                if job.status == "cancelled":
                    item.status = "cancelled"
                    continue

                item.status = "running"
                item.message = f"Processing {os.path.basename(item.input_path)}..."

                try:
                    def item_progress(pct, msg):
                        item.progress = pct
                        item.message = msg

                    result = executor_fn(
                        item.input_path,
                        job.workflow,
                        job.output_dir,
                        item_progress,
                    )
                    item.status = "done"
                    item.progress = 100
                    item.result = result or {}
                    item.message = "Complete"
                except Exception as e:
                    item.status = "error"
                    item.error = str(e)
                    item.message = f"Error: {str(e)[:100]}"
                    logger.exception(f"Batch item failed: {item.input_path}")

                with _batch_lock:
                    job.completed_items = sum(1 for it in job.items if it.status in ("done", "error"))

            with _batch_lock:
                job.status = "done"
        except Exception as e:
            logger.exception("Batch processing failed")
            with _batch_lock:
                job.status = "error"

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()


def cancel_batch(job_id: str) -> bool:
    """Cancel a running batch job."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)
        if job and job.status == "running":
            job.status = "cancelled"
            return True
    return False


def get_batch_status(job_id: str) -> Optional[Dict]:
    """Get current status of a batch job."""
    with _batch_lock:
        job = _batch_jobs.get(job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "status": job.status,
            "total": job.total_items,
            "completed": job.completed_items,
            "progress_pct": int((job.completed_items / max(job.total_items, 1)) * 100),
            "items": [
                {
                    "id": it.id,
                    "file": os.path.basename(it.input_path),
                    "status": it.status,
                    "progress": it.progress,
                    "message": it.message,
                    "result": it.result,
                    "error": it.error,
                }
                for it in job.items
            ],
        }


def list_batch_jobs() -> List[Dict]:
    """List all batch jobs."""
    with _batch_lock:
        return [
            {
                "id": j.id,
                "status": j.status,
                "total": j.total_items,
                "completed": j.completed_items,
                "created_at": j.created_at,
            }
            for j in sorted(_batch_jobs.values(), key=lambda x: x.created_at, reverse=True)
        ]


# ---------------------------------------------------------------------------
# Watch Folder
# ---------------------------------------------------------------------------
_watchers: Dict[str, Dict] = {}
_watcher_lock = threading.Lock()


def start_watch(
    folder: str,
    workflow_steps: List[Dict],
    output_dir: str = "",
    extensions: List[str] = None,
    executor_fn: Callable = None,
) -> str:
    """
    Start watching a folder for new files.

    Args:
        folder: Directory to monitor.
        workflow_steps: Operations to apply to new files.
        output_dir: Output directory.
        extensions: File extensions to watch (default: video/audio).
        executor_fn: Processing function.

    Returns:
        Watcher ID.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Folder not found: {folder}")

    if extensions is None:
        extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".flac"]

    watcher_id = str(uuid.uuid4())[:8]
    output = output_dir or os.path.join(folder, "_processed")
    os.makedirs(output, exist_ok=True)

    watcher = {
        "id": watcher_id,
        "folder": folder,
        "output_dir": output,
        "extensions": extensions,
        "workflow": workflow_steps,
        "active": True,
        "processed": [],
        "seen_files": set(),
        "errors": [],
    }

    # Pre-populate seen files
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp):
            watcher["seen_files"].add(fp)

    with _watcher_lock:
        _watchers[watcher_id] = watcher

    def _watch_loop():
        while watcher["active"]:
            try:
                for f in os.listdir(folder):
                    fp = os.path.join(folder, f)
                    if not os.path.isfile(fp):
                        continue
                    if fp in watcher["seen_files"]:
                        continue
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in extensions:
                        watcher["seen_files"].add(fp)
                        continue

                    # Wait for file to finish writing
                    _wait_for_stable(fp)

                    watcher["seen_files"].add(fp)
                    logger.info(f"Watch folder: processing new file {f}")

                    try:
                        if executor_fn:
                            result = executor_fn(fp, workflow_steps, output, lambda p, m: None)
                            watcher["processed"].append({
                                "file": f,
                                "result": result,
                                "time": time.time(),
                            })
                    except Exception as e:
                        watcher["errors"].append({
                            "file": f,
                            "error": str(e),
                            "time": time.time(),
                        })
                        logger.exception(f"Watch folder error: {f}")

            except Exception as e:
                logger.warning(f"Watch loop error: {e}")

            time.sleep(5)  # Poll every 5 seconds

    thread = threading.Thread(target=_watch_loop, daemon=True)
    thread.start()
    return watcher_id


def stop_watch(watcher_id: str) -> bool:
    """Stop a folder watcher."""
    with _watcher_lock:
        w = _watchers.get(watcher_id)
        if w:
            w["active"] = False
            return True
    return False


def get_watch_status(watcher_id: str) -> Optional[Dict]:
    """Get watcher status."""
    with _watcher_lock:
        w = _watchers.get(watcher_id)
        if not w:
            return None
        return {
            "id": w["id"],
            "folder": w["folder"],
            "active": w["active"],
            "processed_count": len(w["processed"]),
            "error_count": len(w["errors"]),
            "processed": w["processed"][-20:],
            "errors": w["errors"][-10:],
        }


def list_watchers() -> List[Dict]:
    """List all watchers."""
    with _watcher_lock:
        return [
            {
                "id": w["id"],
                "folder": w["folder"],
                "active": w["active"],
                "processed_count": len(w["processed"]),
                "error_count": len(w["errors"]),
            }
            for w in _watchers.values()
        ]


def _wait_for_stable(filepath: str, checks: int = 3, interval: float = 1.0):
    """Wait until file size stabilizes (done writing)."""
    prev_size = -1
    stable = 0
    for _ in range(checks * 5):
        try:
            size = os.path.getsize(filepath)
        except OSError:
            time.sleep(interval)
            continue
        if size == prev_size and size > 0:
            stable += 1
            if stable >= checks:
                return
        else:
            stable = 0
        prev_size = size
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Scan Folder for media files
# ---------------------------------------------------------------------------
def scan_folder(
    folder: str,
    extensions: List[str] = None,
    recursive: bool = False,
) -> List[Dict]:
    """
    Scan a folder for media files with basic info.

    Returns list of dicts with path, name, size, extension.
    """
    if not os.path.isdir(folder):
        raise ValueError(f"Folder not found: {folder}")

    if extensions is None:
        extensions = [
            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv",
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma",
        ]

    files = []
    if recursive:
        for root, dirs, filenames in os.walk(folder):
            for f in sorted(filenames):
                fp = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in extensions:
                    files.append(_file_info(fp))
    else:
        for f in sorted(os.listdir(folder)):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                ext = os.path.splitext(f)[1].lower()
                if ext in extensions:
                    files.append(_file_info(fp))

    return files


def _file_info(filepath: str) -> Dict:
    """Basic file information."""
    stat = os.stat(filepath)
    return {
        "path": filepath,
        "name": os.path.basename(filepath),
        "extension": os.path.splitext(filepath)[1].lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": stat.st_mtime,
    }


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def get_batch_capabilities() -> Dict:
    """Return batch processing features and workflow presets."""
    return {
        "workflow_presets": {
            k: {
                "label": v["label"],
                "description": v["description"],
                "steps": len(v["steps"]),
                "operations": [s["operation"] for s in v["steps"]],
            }
            for k, v in WORKFLOW_PRESETS.items()
        },
        "available_operations": [
            "denoise", "normalize", "eq", "silence_remove", "ducking",
            "isolation", "beats", "effects",
            "color_grade", "chroma_key", "bg_remove", "slow_motion",
            "upscale", "reframe", "speed_ramp",
            "platform_export", "custom_render", "thumbnail",
            "burn_subtitles", "watermark", "gif_export", "audio_extract",
        ],
        "features": ["batch_queue", "workflow_chains", "watch_folder", "media_inspector", "folder_scan"],
    }
