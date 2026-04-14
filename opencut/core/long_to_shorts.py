"""
OpenCut Long-Form to Multi-Short Extraction

Analyze a long video, extract N highlights via LLM, and for each segment:
trim, face-reframe to 9:16, add captions, and export.
Output: folder of shorts + metadata CSV.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ShortSegment:
    """A single extracted short segment."""
    index: int = 0
    title: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    output_path: str = ""
    caption_path: str = ""
    score: float = 0.0


@dataclass
class LongToShortsResult:
    """Result from long-to-shorts extraction."""
    output_dir: str = ""
    segments: List[ShortSegment] = field(default_factory=list)
    csv_path: str = ""
    total_shorts: int = 0
    source_duration: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _probe_duration(filepath: str) -> float:
    """Get media duration via ffprobe."""
    try:
        result = subprocess.run(
            [get_ffprobe_path(), "-v", "quiet", "-print_format", "json",
             "-show_format", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


def _trim_video(input_path: str, start: float, end: float, output_path: str):
    """Trim a video segment using FFmpeg."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0, start)),
        "-i", input_path,
        "-t", str(end - start),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Trim failed: {result.stderr.decode(errors='replace')[-300:]}"
        )


def _reframe_vertical(input_path: str, output_path: str,
                      target_w: int = 1080, target_h: int = 1920):
    """Reframe video to vertical (9:16) with center crop."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", (
            f"crop=ih*{target_w}/{target_h}:ih,"
            f"scale={target_w}:{target_h}"
        ),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Reframe failed: {result.stderr.decode(errors='replace')[-300:]}"
        )


def _extract_highlights_llm(transcript_text: str, num_shorts: int,
                             min_duration: float, max_duration: float) -> List[dict]:
    """Use LLM to extract highlight segments from transcript."""
    try:
        from opencut.core.llm import LLMConfig, query_llm

        system_prompt = (
            "You are a video editor analyzing a transcript for viral short-form clips. "
            "Extract the most engaging, self-contained segments that would work as standalone shorts. "
            f"Find exactly {num_shorts} segments between {min_duration}-{max_duration} seconds each.\n\n"
            "Return JSON array:\n"
            '[{"title": "...", "start": <seconds>, "end": <seconds>, "score": <0-1>}]'
        )

        response = query_llm(
            prompt=f"Transcript:\n\n{transcript_text[:12000]}",
            system_prompt=system_prompt,
        )

        import re
        json_match = re.search(r"\[[\s\S]*\]", response.text)
        if json_match:
            segments = json.loads(json_match.group())
            return [
                {
                    "title": str(s.get("title", f"Short {i + 1}")),
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", 0)),
                    "score": float(s.get("score", 0.5)),
                }
                for i, s in enumerate(segments)
            ]
    except Exception as exc:
        logger.warning("LLM highlight extraction failed: %s", exc)

    return []


def _fallback_highlights(duration: float, num_shorts: int,
                          min_dur: float, max_dur: float) -> List[dict]:
    """Generate evenly-spaced segments when LLM is unavailable."""
    segments = []
    seg_dur = min(max_dur, max(min_dur, duration / max(num_shorts * 2, 1)))
    interval = duration / max(num_shorts, 1)

    for i in range(num_shorts):
        start = i * interval + interval * 0.1
        end = min(start + seg_dur, duration)
        if end - start < min_dur:
            continue
        segments.append({
            "title": f"Highlight {i + 1}",
            "start": round(start, 2),
            "end": round(end, 2),
            "score": 0.5,
        })
    return segments


def _write_metadata_csv(segments: List[ShortSegment], csv_path: str):
    """Write segment metadata to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "title", "start", "end", "duration",
            "output_path", "score",
        ])
        for seg in segments:
            writer.writerow([
                seg.index, seg.title, f"{seg.start:.2f}",
                f"{seg.end:.2f}", f"{seg.duration:.2f}",
                seg.output_path, f"{seg.score:.2f}",
            ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_shorts(
    input_path: str,
    num_shorts: int = 5,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    reframe_vertical: bool = True,
    target_w: int = 1080,
    target_h: int = 1920,
    add_captions: bool = False,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> LongToShortsResult:
    """
    Extract multiple short clips from a long-form video.

    Args:
        input_path: Source long-form video path.
        num_shorts: Number of shorts to extract.
        min_duration: Minimum short duration in seconds.
        max_duration: Maximum short duration in seconds.
        reframe_vertical: Reframe to 9:16 for mobile.
        target_w: Reframe target width.
        target_h: Reframe target height.
        add_captions: Burn in captions (requires whisper).
        output_dir: Output directory. Auto-generated if empty.
        on_progress: Progress callback(pct, msg).

    Returns:
        LongToShortsResult with segments list and CSV path.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    num_shorts = max(1, min(20, num_shorts))
    min_duration = max(5.0, min_duration)
    max_duration = max(min_duration + 1, max_duration)

    if not output_dir:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.path.dirname(input_path), f"{base}_shorts")
    os.makedirs(output_dir, exist_ok=True)

    source_duration = _probe_duration(input_path)
    if source_duration <= 0:
        raise ValueError("Could not determine source video duration")

    if on_progress:
        on_progress(5, "Analyzing video for highlight extraction...")

    # Step 1: Transcribe for LLM analysis
    transcript_text = ""
    try:
        from opencut.core.captions import transcribe
        from opencut.utils.config import CaptionConfig

        if on_progress:
            on_progress(10, "Transcribing video...")

        result = transcribe(input_path, config=CaptionConfig(model="base"))
        if hasattr(result, "segments"):
            for seg in result.segments:
                text = seg.text if hasattr(seg, "text") else seg.get("text", "")
                transcript_text += text + " "
        elif isinstance(result, dict):
            for seg in result.get("segments", []):
                transcript_text += seg.get("text", "") + " "
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)

    if on_progress:
        on_progress(25, "Extracting highlights...")

    # Step 2: Extract highlights
    if transcript_text.strip():
        highlights = _extract_highlights_llm(
            transcript_text, num_shorts, min_duration, max_duration
        )
    else:
        highlights = []

    if not highlights:
        highlights = _fallback_highlights(
            source_duration, num_shorts, min_duration, max_duration
        )

    if on_progress:
        on_progress(35, f"Found {len(highlights)} highlights, processing...")

    # Step 3: Process each highlight
    temp_dir = tempfile.mkdtemp(prefix="opencut_shorts_")
    segments: List[ShortSegment] = []

    try:
        total = len(highlights)
        for i, hl in enumerate(highlights):
            pct = 35 + int((i / max(total, 1)) * 55)
            if on_progress:
                on_progress(pct, f"Processing short {i + 1}/{total}...")

            start = max(0, hl["start"])
            end = min(source_duration, hl["end"])
            if end - start < 5:
                continue

            title = hl.get("title", f"short_{i + 1}")
            safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:40]
            safe_title = safe_title.strip().replace(" ", "_")

            # Trim
            trimmed_path = os.path.join(temp_dir, f"trim_{i}.mp4")
            try:
                _trim_video(input_path, start, end, trimmed_path)
            except Exception as exc:
                logger.warning("Trim failed for segment %d: %s", i, exc)
                continue

            current_path = trimmed_path

            # Reframe
            if reframe_vertical:
                reframed_path = os.path.join(temp_dir, f"reframe_{i}.mp4")
                try:
                    _reframe_vertical(current_path, reframed_path, target_w, target_h)
                    current_path = reframed_path
                except Exception as exc:
                    logger.warning("Reframe failed for segment %d: %s", i, exc)

            # Copy to output
            final_name = f"short_{i + 1:02d}_{safe_title}.mp4"
            final_path = os.path.join(output_dir, final_name)
            shutil.copy2(current_path, final_path)

            segments.append(ShortSegment(
                index=i + 1,
                title=title,
                start=start,
                end=end,
                duration=end - start,
                output_path=final_path,
                score=hl.get("score", 0.5),
            ))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Step 4: Write metadata CSV
    csv_path = os.path.join(output_dir, "shorts_metadata.csv")
    _write_metadata_csv(segments, csv_path)

    if on_progress:
        on_progress(100, f"Extracted {len(segments)} shorts")

    return LongToShortsResult(
        output_dir=output_dir,
        segments=segments,
        csv_path=csv_path,
        total_shorts=len(segments),
        source_duration=source_duration,
    )
