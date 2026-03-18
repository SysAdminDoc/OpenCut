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
from typing import Callable, List, Optional

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


def _probe_duration(filepath: str) -> float:
    """Get media duration via ffprobe."""
    try:
        import json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
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
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
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
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
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

        if on_progress:
            on_progress(40, f"Found {len(highlights)} highlights")

        # Step 3: Process each highlight
        total_clips = len(highlights)
        for i, highlight in enumerate(highlights):
            clip_pct_base = 40 + int((i / total_clips) * 55)

            if on_progress:
                on_progress(clip_pct_base, f"Processing clip {i + 1}/{total_clips}...")

            # 3a. Trim the clip
            safe_title = (highlight.title or f"short_{i + 1}").replace(" ", "_")[:40]
            safe_title = "".join(c for c in safe_title if c.isalnum() or c in "_-")
            trimmed_path = os.path.join(temp_dir, f"trim_{i + 1}.mp4")

            try:
                _trim_clip(input_path, highlight.start, highlight.end, trimmed_path)
            except Exception as exc:
                logger.warning("Trim failed for clip %d: %s", i + 1, exc)
                continue

            current_path = trimmed_path

            # 3b. Face reframe (if enabled and target is vertical)
            if config.face_track and config.target_w < config.target_h:
                try:
                    from opencut.core.face_reframe import face_reframe

                    reframed_path = os.path.join(temp_dir, f"reframed_{i + 1}.mp4")

                    if on_progress:
                        on_progress(clip_pct_base + 5, f"Reframing clip {i + 1}...")

                    face_reframe(
                        input_path=current_path,
                        target_w=config.target_w,
                        target_h=config.target_h,
                        smoothing=config.smoothing,
                        output_path=reframed_path,
                    )
                    current_path = reframed_path

                except Exception as exc:
                    logger.warning("Face reframe failed for clip %d: %s", i + 1, exc)
                    # Fallback: simple center crop via FFmpeg
                    cropped_path = os.path.join(temp_dir, f"cropped_{i + 1}.mp4")
                    crop_cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", current_path,
                        "-vf", f"crop=ih*{config.target_w}/{config.target_h}:ih,"
                               f"scale={config.target_w}:{config.target_h}",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-c:a", "copy",
                        cropped_path,
                    ]
                    result = subprocess.run(crop_cmd, capture_output=True, timeout=300)
                    if result.returncode == 0:
                        current_path = cropped_path

            # 3c. Burn captions (if enabled)
            if config.burn_captions:
                try:
                    from opencut.core.caption_burnin import burnin_segments

                    captioned_path = os.path.join(temp_dir, f"captioned_{i + 1}.mp4")

                    if on_progress:
                        on_progress(clip_pct_base + 10, f"Adding captions to clip {i + 1}...")

                    # Get clip-specific transcript segments
                    clip_segments = [
                        s for s in transcript_segments
                        if s.get("start", 0) >= highlight.start - 0.5
                        and s.get("end", 0) <= highlight.end + 0.5
                    ]

                    if clip_segments:
                        adjusted = []
                        for s in clip_segments:
                            adj_start = max(0, s["start"] - highlight.start)
                            adj_end = max(adj_start, s["end"] - highlight.start)
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
                    logger.warning("Caption burn-in failed for clip %d: %s", i + 1, exc)

            # 3d. Copy to final output
            final_name = f"{base_name}_short_{i + 1}_{safe_title}.mp4"
            final_path = os.path.join(output_dir, final_name)

            counter = 2
            while os.path.exists(final_path):
                final_name = f"{base_name}_short_{i + 1}_{safe_title}_{counter}.mp4"
                final_path = os.path.join(output_dir, final_name)
                counter += 1

            shutil.copy2(current_path, final_path)

            results.append(ShortClip(
                index=i + 1,
                output_path=final_path,
                start=highlight.start,
                end=highlight.end,
                duration=highlight.duration,
                title=highlight.title,
                score=highlight.score,
            ))

            logger.info("Generated short %d: %s (%.1fs)", i + 1, final_path, highlight.duration)

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, f"Generated {len(results)} shorts")

    return results
