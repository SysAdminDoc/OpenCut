"""
OpenCut One-Click Shorts Pipeline

Automated workflow: transcribe -> highlight -> reframe -> captions -> export.
Chains existing modules to produce social-media-ready vertical clips.

Depends on: captions, highlights, face_reframe, caption_burnin
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

from opencut.core.export_presets import EXPORT_PRESETS
from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")

MAGIC_CLIPS_RUN_SCHEMA = "opencut.magic_clips.run.v1"
MAGIC_CLIPS_MANIFEST_NAME = "magic_clips_run_manifest.json"


@dataclass
class ShortsPipelineConfig:
    """Configuration for the shorts pipeline."""
    # Whisper / transcription
    whisper_model: str = "base"
    language: str = ""
    # Highlights
    max_shorts: int = 3
    min_duration: float = 15.0
    max_duration: float = 60.0
    # Reframe
    target_w: int = 1080
    target_h: int = 1920
    face_track: bool = True
    smoothing: float = 0.85
    # Captions
    burn_captions: bool = True
    caption_style: str = "default"
    font_size: int = 48
    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:11434"
    # Optional reviewed Magic Clips handoff. When supplied, the pipeline
    # renders only these windows instead of selecting highlights again.
    approved_plan_id: str = ""
    approved_candidates: Optional[List[dict[str, Any]]] = None
    platform_presets: Optional[List[str]] = None
    transcript_segments: Optional[List[dict[str, Any]]] = None
    checkpoint_manifest_path: str = ""
    resume: bool = True


@dataclass
class ShortClip:
    """A generated short clip."""
    index: int
    output_path: str
    start: float
    end: float
    duration: float
    title: str = ""
    score: float = 0.0
    engagement: object = None  # EngagementScore from highlights module
    platform_preset: str = ""
    width: int = 0
    height: int = 0
    manifest_path: str = ""


def _probe_duration(filepath: str) -> float:
    """Get media duration via ffprobe."""
    try:
        import json
        result = subprocess.run(
            [get_ffprobe_path(), "-v", "quiet", "-print_format", "json",
             "-show_format", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _trim_clip(input_path: str, start: float, end: float, output_path: str):
    """Trim a clip using FFmpeg stream copy (fast)."""
    import math
    start = float(start)
    end = float(end)
    if math.isnan(start) or math.isinf(start) or start < 0:
        start = 0.0
    if math.isnan(end) or math.isinf(end) or end <= start:
        raise ValueError(f"Invalid trim range: start={start}, end={end}")
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-to", str(end - start),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        # Fallback to re-encode if stream copy fails
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-to", str(end - start),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg trim failed: {result.stderr.decode(errors='replace')[-300:]}"
            )


def _conform_clip_dimensions(input_path: str, width: int, height: int, output_path: str) -> None:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output dimensions: {width}x{height}")
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg dimension conform failed: {result.stderr.decode(errors='replace')[-300:]}"
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _safe_slug(value: Any, fallback: str = "run") -> str:
    text = str(value or "").strip()
    slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in text).strip("_")
    return (slug[:80] or fallback)


def _stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: str) -> dict[str, Any]:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "sha256": _file_sha256(path),
        "size": int(stat.st_size),
    }


def _checkpoint_config_payload(config: ShortsPipelineConfig) -> dict[str, Any]:
    return {
        "whisper_model": config.whisper_model,
        "language": config.language,
        "max_shorts": config.max_shorts,
        "min_duration": config.min_duration,
        "max_duration": config.max_duration,
        "target_w": config.target_w,
        "target_h": config.target_h,
        "face_track": config.face_track,
        "smoothing": config.smoothing,
        "burn_captions": config.burn_captions,
        "caption_style": config.caption_style,
        "font_size": config.font_size,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "llm_base_url": config.llm_base_url,
        "approved_plan_id": config.approved_plan_id,
        "approved_candidates": config.approved_candidates or [],
        "platform_presets": config.platform_presets or [],
        "transcript_segments": config.transcript_segments or [],
    }


def _checkpoint_config_hash(config: ShortsPipelineConfig) -> str:
    return _stable_json_hash(_checkpoint_config_payload(config))


def _checkpoint_requested(config: ShortsPipelineConfig) -> bool:
    return bool(
        config.checkpoint_manifest_path
        or config.approved_plan_id
        or config.approved_candidates
    )


def magic_clips_run_manifest_path(
    input_path: str,
    config: Optional[ShortsPipelineConfig] = None,
    output_dir: str = "",
) -> str:
    """Return the deterministic manifest path for a Magic Clips/shorts run."""
    config = config or ShortsPipelineConfig()
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_path), "shorts")
    source_hash = _file_sha256(input_path)[:12] if os.path.isfile(input_path) else "missing"
    config_hash = _checkpoint_config_hash(config)[:12]
    base = _safe_slug(config.approved_plan_id or os.path.splitext(os.path.basename(input_path))[0], "run")
    run_id = f"{base}_{source_hash}_{config_hash}"
    return os.path.join(output_dir, "magic_clips_runs", run_id, MAGIC_CLIPS_MANIFEST_NAME)


def _default_manifest(
    input_path: str,
    config: ShortsPipelineConfig,
    output_dir: str,
    manifest_path: str,
    source: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    run_dir = os.path.dirname(manifest_path)
    return {
        "schema_version": MAGIC_CLIPS_RUN_SCHEMA,
        "status": "running",
        "next_step": "transcribe",
        "source": source,
        "config_hash": config_hash,
        "config": _checkpoint_config_payload(config),
        "approved_plan_id": config.approved_plan_id,
        "output_dir": os.path.abspath(output_dir),
        "run_dir": run_dir,
        "intermediates_dir": os.path.join(run_dir, "intermediates"),
        "transcript_cache_id": "",
        "highlight_cache_id": "",
        "transcript_segments": [],
        "highlights": [],
        "trimmed_clips": {},
        "reframe_outputs": {},
        "caption_files": {},
        "thumbnail_candidates": {},
        "final_exports": [],
        "resume": {
            "accepted": False,
            "source_job_id": "",
            "attempt": 0,
            "mismatch": "",
        },
    }


def _load_json_file(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Could not load Magic Clips manifest %s: %s", path, exc)
        return {}


def _write_manifest(manifest: dict[str, Any]) -> None:
    path = str(manifest.get("manifest_path") or "")
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _prepare_checkpoint_manifest(
    input_path: str,
    config: ShortsPipelineConfig,
    output_dir: str,
) -> tuple[dict[str, Any], bool]:
    manifest_path = os.path.abspath(
        config.checkpoint_manifest_path or magic_clips_run_manifest_path(
            input_path,
            config=config,
            output_dir=output_dir,
        )
    )
    source = _source_fingerprint(input_path)
    config_hash = _checkpoint_config_hash(config)
    existing = _load_json_file(manifest_path) if config.resume else {}
    resume_ok = (
        bool(existing)
        and existing.get("schema_version") == MAGIC_CLIPS_RUN_SCHEMA
        and existing.get("source", {}).get("sha256") == source["sha256"]
        and existing.get("config_hash") == config_hash
    )
    if resume_ok:
        manifest = existing
        manifest["status"] = "running"
        manifest.setdefault("resume", {})
        manifest["resume"].update({
            "accepted": True,
            "source_job_id": "",
            "attempt": 0,
            "mismatch": "",
        })
    else:
        manifest = _default_manifest(input_path, config, output_dir, manifest_path, source, config_hash)
        if existing:
            mismatch = "source" if existing.get("source", {}).get("sha256") != source["sha256"] else "config"
            manifest["resume"]["mismatch"] = mismatch
            manifest["resume"]["previous_manifest_path"] = manifest_path
    manifest["manifest_path"] = manifest_path
    os.makedirs(manifest["intermediates_dir"], exist_ok=True)
    _write_manifest(manifest)
    return manifest, resume_ok


def _transient_manifest(temp_dir: str, output_dir: str) -> dict[str, Any]:
    return {
        "status": "running",
        "next_step": "transcribe",
        "manifest_path": "",
        "output_dir": os.path.abspath(output_dir),
        "intermediates_dir": temp_dir,
        "transcript_segments": [],
        "highlights": [],
        "trimmed_clips": {},
        "reframe_outputs": {},
        "caption_files": {},
        "thumbnail_candidates": {},
        "final_exports": [],
        "resume": {"accepted": False, "mismatch": ""},
    }


def _mark_manifest_interrupted(manifest: dict[str, Any], exc: Exception) -> None:
    if not manifest.get("manifest_path"):
        return
    manifest["status"] = "interrupted"
    manifest["last_error"] = str(exc)
    manifest.setdefault("errors", []).append({
        "next_step": manifest.get("next_step", ""),
        "message": str(exc),
    })
    _write_manifest(manifest)


def _manifest_highlights(raw: Any) -> List[SimpleNamespace]:
    if not isinstance(raw, list):
        return []
    highlights: List[SimpleNamespace] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start"), -1.0)
        end = _safe_float(item.get("end"), start)
        if start < 0 or end <= start:
            continue
        highlights.append(SimpleNamespace(
            start=start,
            end=end,
            duration=end - start,
            title=str(item.get("title") or "short").strip() or "short",
            score=max(0.0, min(100.0, _safe_float(item.get("score"), 0.0))),
            engagement=None,
            candidate_id=str(item.get("candidate_id") or ""),
        ))
    return highlights


def _highlight_to_manifest(highlight: Any) -> dict[str, Any]:
    return {
        "candidate_id": str(getattr(highlight, "candidate_id", "") or ""),
        "start": _safe_float(getattr(highlight, "start", 0.0), 0.0),
        "end": _safe_float(getattr(highlight, "end", 0.0), 0.0),
        "duration": _safe_float(getattr(highlight, "duration", 0.0), 0.0),
        "title": str(getattr(highlight, "title", "") or ""),
        "score": _safe_float(getattr(highlight, "score", 0.0), 0.0),
    }


def _checkpoint_key(highlight: Any, contract: Any, index: int) -> str:
    candidate_id = str(getattr(highlight, "candidate_id", "") or "")
    preset_id = str(getattr(contract, "preset_id", "") or "default")
    return _safe_slug(f"{candidate_id or index}_{preset_id}", f"clip_{index}_{preset_id}")


def _valid_file(path: Any) -> bool:
    return bool(path) and isinstance(path, str) and os.path.isfile(path)


def _completed_export(
    manifest: dict[str, Any],
    key: str,
    output_index: int,
    highlight: Any,
    contract: Any,
) -> Optional[ShortClip]:
    for item in manifest.get("final_exports") or []:
        if not isinstance(item, dict) or item.get("key") != key:
            continue
        output_path = str(item.get("output_path") or "")
        if not _valid_file(output_path):
            continue
        return ShortClip(
            index=output_index,
            output_path=output_path,
            start=_safe_float(item.get("start"), getattr(highlight, "start", 0.0)),
            end=_safe_float(item.get("end"), getattr(highlight, "end", 0.0)),
            duration=_safe_float(item.get("duration"), getattr(highlight, "duration", 0.0)),
            title=str(item.get("title") or getattr(highlight, "title", "")),
            score=_safe_float(item.get("score"), getattr(highlight, "score", 0.0)),
            engagement=getattr(highlight, "engagement", None),
            platform_preset=str(item.get("platform_preset") or getattr(contract, "preset_id", "")),
            width=int(item.get("width") or getattr(contract, "width", 0)),
            height=int(item.get("height") or getattr(contract, "height", 0)),
            manifest_path=str(manifest.get("manifest_path") or ""),
        )
    return None


def _upsert_final_export(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    exports = [item for item in manifest.get("final_exports") or [] if item.get("key") != record["key"]]
    exports.append(record)
    manifest["final_exports"] = exports


def _normalise_transcript_segments(raw_segments: Any) -> List[dict[str, Any]]:
    if raw_segments in (None, ""):
        return []
    if not isinstance(raw_segments, list):
        raise ValueError("transcript_segments must be a list")

    transcript_segments: List[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        start = _safe_float(raw.get("start", raw.get("start_s")), 0.0)
        end = _safe_float(raw.get("end", raw.get("end_s")), start)
        if start < 0 or end <= start:
            continue
        transcript_segments.append({
            "start": start,
            "end": end,
            "text": str(raw.get("text") or raw.get("transcript") or ""),
        })
    return sorted(transcript_segments, key=lambda item: (item["start"], item["end"]))


def _approved_candidate_highlights(config: ShortsPipelineConfig) -> List[SimpleNamespace]:
    raw_candidates = config.approved_candidates or []
    if not raw_candidates:
        return []
    if not isinstance(raw_candidates, list):
        raise ValueError("approved_candidates must be a list")

    highlights: List[SimpleNamespace] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError("approved_candidates must contain objects")
        start = _safe_float(raw.get("start", raw.get("start_s")), -1.0)
        end = _safe_float(raw.get("end", raw.get("end_s")), start)
        if start < 0 or end <= start:
            raise ValueError("approved_candidates require valid start/end windows")
        title = str(raw.get("title") or raw.get("name") or f"short_{index + 1}").strip()
        score = max(0.0, min(100.0, _safe_float(raw.get("score"), 0.0)))
        highlights.append(SimpleNamespace(
            start=start,
            end=end,
            duration=end - start,
            title=title or f"short_{index + 1}",
            score=score,
            engagement=None,
            candidate_id=str(raw.get("candidate_id") or raw.get("id") or ""),
        ))
        if len(highlights) >= config.max_shorts:
            break
    return highlights


def _platform_render_contracts(config: ShortsPipelineConfig) -> List[SimpleNamespace]:
    raw_presets = config.platform_presets or []
    if isinstance(raw_presets, str):
        raw_presets = [raw_presets]
    if not isinstance(raw_presets, list):
        raise ValueError("platform_presets must be a list")

    preset_ids: list[str] = []
    for raw in raw_presets:
        preset_id = str(raw or "").strip()
        if not preset_id:
            continue
        if preset_id not in EXPORT_PRESETS:
            raise ValueError(f"Unknown platform preset: {preset_id}")
        if preset_id not in preset_ids:
            preset_ids.append(preset_id)

    if not preset_ids:
        return [
            SimpleNamespace(
                preset_id="",
                filename_slug="",
                width=config.target_w,
                height=config.target_h,
                max_duration=config.max_duration,
                ext=".mp4",
            )
        ]

    contracts: List[SimpleNamespace] = []
    for preset_id in preset_ids:
        preset = EXPORT_PRESETS[preset_id]
        if preset.get("audio_only"):
            raise ValueError(f"Platform preset is audio-only: {preset_id}")
        width = int(preset.get("width") or config.target_w)
        height = int(preset.get("height") or config.target_h)
        max_duration = _safe_float(preset.get("max_duration"), config.max_duration)
        contracts.append(SimpleNamespace(
            preset_id=preset_id,
            filename_slug=preset_id,
            width=width,
            height=height,
            max_duration=max_duration,
            ext=str(preset.get("ext") or ".mp4"),
        ))
    return contracts


def generate_shorts(
    input_path: str,
    config: Optional[ShortsPipelineConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> List[ShortClip]:
    """
    Run the full shorts pipeline on a video.

    Steps:
      1. Transcribe (faster-whisper)
      2. Extract highlights via LLM
      3. For each highlight: trim -> face reframe -> burn captions -> output

    Args:
        input_path: Source video path.
        config: Pipeline configuration. Uses defaults if None.
        output_dir: Output directory for generated shorts.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ShortClip objects with output paths.
    """
    if config is None:
        config = ShortsPipelineConfig()

    if config.target_h <= 0:
        raise ValueError("target_h must be positive")
    if config.target_w <= 0:
        raise ValueError("target_w must be positive")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set up output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_path), "shorts")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    checkpoint_requested = _checkpoint_requested(config)
    if checkpoint_requested:
        manifest, resume_ok = _prepare_checkpoint_manifest(input_path, config, output_dir)
        temp_dir = str(manifest.get("intermediates_dir") or "")
        preserve_temp_dir = True
        config.checkpoint_manifest_path = str(manifest.get("manifest_path") or config.checkpoint_manifest_path)
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="opencut_shorts_")
        manifest = _transient_manifest(temp_dir, output_dir)
        resume_ok = False
        preserve_temp_dir = False

    results = []

    try:
        render_contracts = _platform_render_contracts(config)
        approved_highlights = _approved_candidate_highlights(config)
        if approved_highlights:
            transcript_segments = _normalise_transcript_segments(config.transcript_segments)
            highlights = approved_highlights
            manifest["transcript_segments"] = transcript_segments
            manifest["highlights"] = [_highlight_to_manifest(highlight) for highlight in highlights]
            manifest["highlight_cache_id"] = config.approved_plan_id or _stable_json_hash(manifest["highlights"])
            manifest["next_step"] = "render"
            _write_manifest(manifest)
            if on_progress:
                on_progress(40, f"Using {len(highlights)} approved candidate clips")
        else:
            # Step 1: Transcribe
            transcript_segments = (
                _normalise_transcript_segments(manifest.get("transcript_segments"))
                if resume_ok and manifest.get("transcript_segments")
                else []
            )
            if transcript_segments:
                if on_progress:
                    on_progress(20, f"Resumed {len(transcript_segments)} transcript segments")
            else:
                if on_progress:
                    on_progress(5, "Transcribing video...")

                try:
                    from opencut.core.captions import transcribe
                except ImportError:
                    raise RuntimeError(
                        "Transcription module not available. "
                        "Install faster-whisper: pip install faster-whisper"
                    )

                from opencut.utils.config import CaptionConfig as _CapCfg
                _cap_cfg = _CapCfg(
                    model=config.whisper_model,
                    language=config.language or None,
                )
                transcript_result = transcribe(input_path, config=_cap_cfg)

                # Extract segments list from transcription result
                if hasattr(transcript_result, "segments"):
                    segments = transcript_result.segments
                elif isinstance(transcript_result, dict):
                    segments = transcript_result.get("segments", [])
                else:
                    segments = []

                # Normalize segments to list of dicts
                transcript_segments = []
                for seg in segments:
                    if isinstance(seg, dict):
                        transcript_segments.append(seg)
                    elif hasattr(seg, "start"):
                        transcript_segments.append({
                            "start": seg.start,
                            "end": seg.end,
                            "text": getattr(seg, "text", ""),
                        })
                transcript_segments = _normalise_transcript_segments(transcript_segments)
                manifest["transcript_segments"] = transcript_segments
                manifest["transcript_cache_id"] = _stable_json_hash({
                    "source": manifest.get("source", {}).get("sha256", ""),
                    "model": config.whisper_model,
                    "language": config.language,
                    "segments": transcript_segments,
                })
                manifest["next_step"] = "highlight"
                _write_manifest(manifest)

            if not transcript_segments:
                logger.warning("No transcript segments found")
                if on_progress:
                    on_progress(100, "No speech detected in video")
                manifest["status"] = "complete"
                manifest["next_step"] = "complete"
                _write_manifest(manifest)
                return []

            if on_progress:
                on_progress(20, f"Transcribed {len(transcript_segments)} segments")

            # Step 2: Extract highlights via LLM
            highlights = _manifest_highlights(manifest.get("highlights")) if resume_ok else []
            if highlights:
                if on_progress:
                    on_progress(25, f"Resumed {len(highlights)} highlight windows")
            else:
                if on_progress:
                    on_progress(25, "Analyzing for highlights...")

                from opencut.core.highlights import extract_highlights
                from opencut.core.llm import LLMConfig

                llm_config = LLMConfig(
                    provider=config.llm_provider,
                    model=config.llm_model,
                    api_key=config.llm_api_key,
                    base_url=config.llm_base_url,
                )

                manifest["next_step"] = "highlight"
                _write_manifest(manifest)
                try:
                    highlight_result = extract_highlights(
                        transcript_segments=transcript_segments,
                        max_highlights=config.max_shorts,
                        min_duration=config.min_duration,
                        max_duration=config.max_duration,
                        llm_config=llm_config,
                    )
                except Exception as exc:
                    _mark_manifest_interrupted(manifest, exc)
                    raise

                highlights = highlight_result.highlights
                manifest["highlights"] = [_highlight_to_manifest(highlight) for highlight in highlights]
                manifest["highlight_cache_id"] = _stable_json_hash({
                    "transcript_cache_id": manifest.get("transcript_cache_id", ""),
                    "llm_provider": config.llm_provider,
                    "llm_model": config.llm_model,
                    "highlights": manifest["highlights"],
                })
                manifest["next_step"] = "render"
                _write_manifest(manifest)
            if not highlights:
                logger.warning("No highlights found by LLM")
                if on_progress:
                    on_progress(100, "No interesting segments found")
                manifest["status"] = "complete"
                manifest["next_step"] = "complete"
                _write_manifest(manifest)
                return []

        # Clamp highlight bounds to valid duration range
        total_dur = _probe_duration(input_path)
        if total_dur > 0:
            for hl in highlights:
                hl.start = max(0.0, min(total_dur - 0.1, hl.start))
                hl.end = max(hl.start + 0.1, min(total_dur, hl.end))
                try:
                    hl.duration = hl.end - hl.start
                except Exception:
                    pass

        if on_progress:
            on_progress(40, f"Found {len(highlights)} highlights")

        # Step 3: Process each highlight/platform target.
        total_clips = len(highlights) * len(render_contracts)
        output_index = 0
        for i, highlight in enumerate(highlights):
            for contract in render_contracts:
                output_index += 1
                clip_pct_base = 40 + int(((output_index - 1) / total_clips) * 55)
                preset_label = f" {contract.preset_id}" if contract.preset_id else ""
                checkpoint_key = _checkpoint_key(highlight, contract, i + 1)

                if on_progress:
                    on_progress(
                        clip_pct_base,
                        f"Processing clip {i + 1}/{len(highlights)}{preset_label}...",
                    )

                # 3a. Trim the clip
                clip_start = float(highlight.start)
                clip_end = float(highlight.end)
                if contract.max_duration and clip_end - clip_start > contract.max_duration:
                    clip_end = clip_start + float(contract.max_duration)
                clip_duration = max(0.0, clip_end - clip_start)
                if clip_duration <= 0:
                    continue

                completed = _completed_export(manifest, checkpoint_key, output_index, highlight, contract)
                if completed:
                    results.append(completed)
                    manifest["next_step"] = "render"
                    _write_manifest(manifest)
                    continue

                safe_title = (highlight.title or f"short_{i + 1}").replace(" ", "_")[:40]
                safe_title = "".join(c for c in safe_title if c.isalnum() or c in "_-")
                name_part = f"{i + 1}_{contract.filename_slug}" if contract.filename_slug else f"{i + 1}"
                trimmed_path = os.path.join(temp_dir, f"trim_{name_part}.mp4")

                trim_record = (manifest.get("trimmed_clips") or {}).get(checkpoint_key, {})
                if resume_ok and _valid_file(trim_record.get("path")):
                    trimmed_path = trim_record["path"]
                else:
                    manifest["next_step"] = f"trim:{checkpoint_key}"
                    _write_manifest(manifest)
                    try:
                        _trim_clip(input_path, clip_start, clip_end, trimmed_path)
                    except Exception as exc:
                        logger.warning("Trim failed for clip %d%s: %s", i + 1, preset_label, exc)
                        if checkpoint_requested:
                            _mark_manifest_interrupted(manifest, exc)
                            raise
                        continue
                    manifest.setdefault("trimmed_clips", {})[checkpoint_key] = {
                        "path": trimmed_path,
                        "start": clip_start,
                        "end": clip_end,
                        "duration": clip_duration,
                    }
                    manifest["next_step"] = f"render:{checkpoint_key}"
                    _write_manifest(manifest)

                current_path = trimmed_path
                dimensions_conformed = False

                # 3b. Face reframe (if enabled and target is vertical)
                if config.face_track and contract.width < contract.height:
                    reframe_record = (manifest.get("reframe_outputs") or {}).get(checkpoint_key, {})
                    if resume_ok and _valid_file(reframe_record.get("path")):
                        current_path = reframe_record["path"]
                        dimensions_conformed = True
                    else:
                        try:
                            from opencut.core.face_reframe import face_reframe

                            reframed_path = os.path.join(temp_dir, f"reframed_{name_part}.mp4")

                            if on_progress:
                                on_progress(clip_pct_base + 5, f"Reframing clip {i + 1}{preset_label}...")

                            manifest["next_step"] = f"reframe:{checkpoint_key}"
                            _write_manifest(manifest)
                            face_reframe(
                                input_path=current_path,
                                target_w=contract.width,
                                target_h=contract.height,
                                smoothing=config.smoothing,
                                output_path=reframed_path,
                            )
                            current_path = reframed_path
                            dimensions_conformed = True
                            manifest.setdefault("reframe_outputs", {})[checkpoint_key] = {
                                "path": current_path,
                                "kind": "face_reframe",
                                "width": contract.width,
                                "height": contract.height,
                            }
                            _write_manifest(manifest)

                        except Exception as exc:
                            logger.warning("Face reframe failed for clip %d%s: %s", i + 1, preset_label, exc)
                            # Fallback: simple center crop via FFmpeg
                            cropped_path = os.path.join(temp_dir, f"cropped_{name_part}.mp4")
                            crop_cmd = [
                                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                                "-i", current_path,
                                "-vf", f"crop=ih*{int(contract.width)}/{int(contract.height)}:ih,"
                                       f"scale={int(contract.width)}:{int(contract.height)}",
                                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                "-pix_fmt", "yuv420p", "-c:a", "copy",
                                cropped_path,
                            ]
                            result = subprocess.run(crop_cmd, capture_output=True, timeout=300)
                            if result.returncode == 0:
                                current_path = cropped_path
                                dimensions_conformed = True
                                manifest.setdefault("reframe_outputs", {})[checkpoint_key] = {
                                    "path": current_path,
                                    "kind": "crop_fallback",
                                    "width": contract.width,
                                    "height": contract.height,
                                }
                                _write_manifest(manifest)

                # 3c. Burn captions (if enabled)
                if config.burn_captions:
                    caption_record = (manifest.get("caption_files") or {}).get(checkpoint_key, {})
                    if resume_ok and _valid_file(caption_record.get("path")):
                        current_path = caption_record["path"]
                    else:
                        try:
                            from opencut.core.caption_burnin import burnin_segments

                            captioned_path = os.path.join(temp_dir, f"captioned_{name_part}.mp4")

                            if on_progress:
                                on_progress(clip_pct_base + 10, f"Adding captions to clip {i + 1}{preset_label}...")

                            # Get clip-specific transcript segments
                            clip_segments = [
                                s for s in transcript_segments
                                if s.get("start", 0) >= clip_start - 0.5
                                and s.get("end", 0) <= clip_end + 0.5
                            ]

                            if clip_segments:
                                adjusted = []
                                for s in clip_segments:
                                    adj_start = max(0, s["start"] - clip_start)
                                    adj_end = max(adj_start, s["end"] - clip_start)
                                    adjusted.append({
                                        "start": adj_start,
                                        "end": adj_end,
                                        "text": s.get("text", ""),
                                    })

                                manifest["next_step"] = f"captions:{checkpoint_key}"
                                _write_manifest(manifest)
                                burnin_segments(
                                    video_path=current_path,
                                    segments=adjusted,
                                    output_path=captioned_path,
                                    style=config.caption_style,
                                )
                                current_path = captioned_path
                                manifest.setdefault("caption_files", {})[checkpoint_key] = {
                                    "path": captioned_path,
                                    "style": config.caption_style,
                                    "segments": adjusted,
                                }
                                _write_manifest(manifest)

                        except Exception as exc:
                            logger.warning("Caption burn-in failed for clip %d%s: %s", i + 1, preset_label, exc)

                # 3d. Enforce preset dimensions for square/feed and non-face-track renders.
                if contract.preset_id and not dimensions_conformed:
                    conform_key = f"{checkpoint_key}:conform"
                    conform_record = (manifest.get("reframe_outputs") or {}).get(conform_key, {})
                    if resume_ok and _valid_file(conform_record.get("path")):
                        current_path = conform_record["path"]
                    else:
                        try:
                            conformed_path = os.path.join(temp_dir, f"conformed_{name_part}{contract.ext}")
                            if on_progress:
                                on_progress(clip_pct_base + 12, f"Conforming clip {i + 1}{preset_label}...")
                            manifest["next_step"] = f"conform:{checkpoint_key}"
                            _write_manifest(manifest)
                            _conform_clip_dimensions(current_path, contract.width, contract.height, conformed_path)
                            current_path = conformed_path
                            manifest.setdefault("reframe_outputs", {})[conform_key] = {
                                "path": current_path,
                                "kind": "dimension_conform",
                                "width": contract.width,
                                "height": contract.height,
                            }
                            _write_manifest(manifest)
                        except Exception as exc:
                            logger.warning("Dimension conform failed for clip %d%s: %s", i + 1, preset_label, exc)
                            if checkpoint_requested:
                                _mark_manifest_interrupted(manifest, exc)
                                raise
                            continue

                # 3e. Copy to final output
                preset_name = f"_{contract.filename_slug}" if contract.filename_slug else ""
                final_name = f"{base_name}_short_{i + 1}{preset_name}_{safe_title}{contract.ext}"
                final_path = os.path.join(output_dir, final_name)

                counter = 2
                while os.path.exists(final_path):
                    final_name = f"{base_name}_short_{i + 1}{preset_name}_{safe_title}_{counter}{contract.ext}"
                    final_path = os.path.join(output_dir, final_name)
                    counter += 1

                manifest["next_step"] = f"final_export:{checkpoint_key}"
                manifest.setdefault("thumbnail_candidates", {})[checkpoint_key] = {
                    "kind": "first_frame",
                    "timestamp": clip_start,
                }
                _write_manifest(manifest)
                shutil.copy2(current_path, final_path)

                clip = ShortClip(
                    index=output_index,
                    output_path=final_path,
                    start=clip_start,
                    end=clip_end,
                    duration=clip_duration,
                    title=highlight.title,
                    score=highlight.score,
                    engagement=highlight.engagement,
                    platform_preset=contract.preset_id,
                    width=contract.width,
                    height=contract.height,
                    manifest_path=str(manifest.get("manifest_path") or ""),
                )
                results.append(clip)
                _upsert_final_export(manifest, {
                    "key": checkpoint_key,
                    "candidate_id": str(getattr(highlight, "candidate_id", "") or ""),
                    "output_path": final_path,
                    "platform_preset": contract.preset_id,
                    "start": clip_start,
                    "end": clip_end,
                    "duration": clip_duration,
                    "title": highlight.title,
                    "score": highlight.score,
                    "width": contract.width,
                    "height": contract.height,
                    "intermediate_paths": {
                        "trimmed": trimmed_path,
                        "current": current_path,
                    },
                    "thumbnail_candidate": manifest["thumbnail_candidates"][checkpoint_key],
                })
                manifest["next_step"] = "render"
                _write_manifest(manifest)

                logger.info(
                    "Generated short %d%s: %s (%.1fs)",
                    i + 1,
                    preset_label,
                    final_path,
                    clip_duration,
                )

    except Exception as exc:
        if manifest.get("status") != "interrupted":
            _mark_manifest_interrupted(manifest, exc)
        raise
    finally:
        if not preserve_temp_dir:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    if on_progress:
        on_progress(100, f"Generated {len(results)} shorts")

    manifest["status"] = "complete"
    manifest["next_step"] = "complete"
    _write_manifest(manifest)

    return results
