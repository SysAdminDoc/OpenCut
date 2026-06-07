"""
OpenCut One-Click Shorts Pipeline

Automated workflow: transcribe -> highlight -> reframe -> captions -> export.
Chains existing modules to produce social-media-ready vertical clips.

Depends on: captions, highlights, face_reframe, caption_burnin
"""

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
    temp_dir = tempfile.mkdtemp(prefix="opencut_shorts_")

    results = []

    try:
        render_contracts = _platform_render_contracts(config)
        approved_highlights = _approved_candidate_highlights(config)
        if approved_highlights:
            transcript_segments = _normalise_transcript_segments(config.transcript_segments)
            highlights = approved_highlights
            if on_progress:
                on_progress(40, f"Using {len(highlights)} approved candidate clips")
        else:
            # Step 1: Transcribe
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

            if not transcript_segments:
                logger.warning("No transcript segments found")
                if on_progress:
                    on_progress(100, "No speech detected in video")
                return []

            if on_progress:
                on_progress(20, f"Transcribed {len(transcript_segments)} segments")

            # Step 2: Extract highlights via LLM
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

            highlight_result = extract_highlights(
                transcript_segments=transcript_segments,
                max_highlights=config.max_shorts,
                min_duration=config.min_duration,
                max_duration=config.max_duration,
                llm_config=llm_config,
            )

            highlights = highlight_result.highlights
            if not highlights:
                logger.warning("No highlights found by LLM")
                if on_progress:
                    on_progress(100, "No interesting segments found")
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

                safe_title = (highlight.title or f"short_{i + 1}").replace(" ", "_")[:40]
                safe_title = "".join(c for c in safe_title if c.isalnum() or c in "_-")
                name_part = f"{i + 1}_{contract.filename_slug}" if contract.filename_slug else f"{i + 1}"
                trimmed_path = os.path.join(temp_dir, f"trim_{name_part}.mp4")

                try:
                    _trim_clip(input_path, clip_start, clip_end, trimmed_path)
                except Exception as exc:
                    logger.warning("Trim failed for clip %d%s: %s", i + 1, preset_label, exc)
                    continue

                current_path = trimmed_path
                dimensions_conformed = False

                # 3b. Face reframe (if enabled and target is vertical)
                if config.face_track and contract.width < contract.height:
                    try:
                        from opencut.core.face_reframe import face_reframe

                        reframed_path = os.path.join(temp_dir, f"reframed_{name_part}.mp4")

                        if on_progress:
                            on_progress(clip_pct_base + 5, f"Reframing clip {i + 1}{preset_label}...")

                        face_reframe(
                            input_path=current_path,
                            target_w=contract.width,
                            target_h=contract.height,
                            smoothing=config.smoothing,
                            output_path=reframed_path,
                        )
                        current_path = reframed_path
                        dimensions_conformed = True

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

                # 3c. Burn captions (if enabled)
                if config.burn_captions:
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

                            burnin_segments(
                                video_path=current_path,
                                segments=adjusted,
                                output_path=captioned_path,
                                style=config.caption_style,
                            )
                            current_path = captioned_path

                    except Exception as exc:
                        logger.warning("Caption burn-in failed for clip %d%s: %s", i + 1, preset_label, exc)

                # 3d. Enforce preset dimensions for square/feed and non-face-track renders.
                if contract.preset_id and not dimensions_conformed:
                    try:
                        conformed_path = os.path.join(temp_dir, f"conformed_{name_part}{contract.ext}")
                        if on_progress:
                            on_progress(clip_pct_base + 12, f"Conforming clip {i + 1}{preset_label}...")
                        _conform_clip_dimensions(current_path, contract.width, contract.height, conformed_path)
                        current_path = conformed_path
                    except Exception as exc:
                        logger.warning("Dimension conform failed for clip %d%s: %s", i + 1, preset_label, exc)
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

                shutil.copy2(current_path, final_path)

                results.append(ShortClip(
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
                ))

                logger.info(
                    "Generated short %d%s: %s (%.1fs)",
                    i + 1,
                    preset_label,
                    final_path,
                    clip_duration,
                )

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, f"Generated {len(results)} shorts")

    return results
