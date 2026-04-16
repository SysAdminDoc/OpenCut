"""
OpenCut Podcast Episode to Multi-Platform Bundle

Full pipeline: denoise + normalize -> export clean audio -> transcribe ->
chapters -> highlight clips -> audiogram -> show notes -> bundle all outputs.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PodcastBundleResult:
    """Result from the full podcast bundle pipeline."""
    output_dir: str = ""
    clean_audio_path: str = ""
    transcript_text: str = ""
    chapters: List[Dict] = field(default_factory=list)
    highlight_clips: List[str] = field(default_factory=list)
    audiogram_path: str = ""
    show_notes_markdown: str = ""
    show_notes_html: str = ""
    manifest: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def _get_audio_duration(filepath: str) -> float:
    """Get audio file duration via ffprobe."""
    try:
        result = subprocess.run(
            [get_ffprobe_path(), "-v", "quiet", "-print_format", "json",
             "-show_format", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


def _denoise_and_normalize(input_path: str, output_path: str,
                            on_progress: Optional[Callable] = None) -> str:
    """Clean audio: high-pass filter, normalization, and light noise gate."""
    af_chain = (
        "highpass=f=80,"
        "agate=threshold=0.01:ratio=3:attack=5:release=50,"
        "loudnorm=I=-16:LRA=11:TP=-1.5"
    )
    cmd = (
        FFmpegCmd()
        .input(input_path)
        .audio_filter(af_chain)
        .audio_codec("aac", bitrate="192k")
        .option("vn")
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=1800)
    return output_path


def _export_clean_audio(input_path: str, output_dir: str,
                         formats: List[str] = None) -> Dict[str, str]:
    """Export clean audio in multiple formats."""
    if formats is None:
        formats = ["mp3", "wav"]

    results = {}
    base = os.path.splitext(os.path.basename(input_path))[0]

    for fmt in formats:
        out_path = os.path.join(output_dir, f"{base}_clean.{fmt}")
        cmd = FFmpegCmd().input(input_path)

        if fmt == "mp3":
            cmd.audio_codec("libmp3lame", bitrate="192k")
        elif fmt == "wav":
            cmd.option("c:a", "pcm_s16le")
        elif fmt == "flac":
            cmd.option("c:a", "flac")
        else:
            cmd.audio_codec("aac", bitrate="192k")

        cmd.option("vn")
        cmd.output(out_path)
        run_ffmpeg(cmd.build(), timeout=600)
        results[fmt] = out_path

    return results


def _transcribe_audio(audio_path: str, on_progress=None) -> tuple:
    """Transcribe audio, returning (full_text, segments_list)."""
    try:
        from opencut.core.captions import transcribe
        from opencut.utils.config import CaptionConfig

        result = transcribe(audio_path, config=CaptionConfig(model="base"))
        full_text = ""
        segments = []

        if hasattr(result, "segments"):
            for seg in result.segments:
                text = seg.text if hasattr(seg, "text") else seg.get("text", "")
                start = seg.start if hasattr(seg, "start") else seg.get("start", 0)
                end = seg.end if hasattr(seg, "end") else seg.get("end", 0)
                full_text += text + " "
                segments.append({"start": start, "end": end, "text": text})
        elif isinstance(result, dict):
            for seg in result.get("segments", []):
                full_text += seg.get("text", "") + " "
                segments.append(seg)

        return full_text.strip(), segments
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        return "", []


def _extract_chapters(transcript_text: str, segments: List[dict]) -> List[dict]:
    """Extract chapter markers from transcript via LLM or heuristics."""
    try:
        import re

        from opencut.core.llm import query_llm

        response = query_llm(
            prompt=f"Transcript:\n\n{transcript_text[:10000]}",
            system_prompt=(
                "Extract chapter markers from this podcast transcript. "
                "Return JSON array: "
                '[{"timestamp": <seconds>, "title": "Chapter title"}]'
            ),
        )

        json_match = re.search(r"\[[\s\S]*\]", response.text)
        if json_match:
            chapters = json.loads(json_match.group())
            return [
                {"timestamp": float(c.get("timestamp", 0)),
                 "title": str(c.get("title", ""))}
                for c in chapters if c.get("title")
            ]
    except Exception as exc:
        logger.warning("LLM chapter extraction failed: %s", exc)

    # Fallback: split evenly
    if segments:
        total_dur = max(s.get("end", 0) for s in segments) if segments else 0
        n_chapters = min(8, max(2, int(total_dur / 300)))
        interval = total_dur / n_chapters
        chapters = []
        for i in range(n_chapters):
            t = i * interval
            chapters.append({"timestamp": round(t, 1), "title": f"Part {i + 1}"})
        return chapters

    return []


def _extract_highlight_clips(audio_path: str, transcript_text: str,
                              output_dir: str, max_clips: int = 3) -> List[str]:
    """Extract highlight clips from the podcast."""
    clips = []
    try:
        import re

        from opencut.core.llm import query_llm

        response = query_llm(
            prompt=f"Transcript:\n\n{transcript_text[:10000]}",
            system_prompt=(
                f"Find {max_clips} most engaging/quotable moments in this podcast. "
                "Return JSON: "
                '[{"start": <seconds>, "end": <seconds>, "title": "..."}]'
            ),
        )

        json_match = re.search(r"\[[\s\S]*\]", response.text)
        if json_match:
            highlights = json.loads(json_match.group())
            for i, hl in enumerate(highlights[:max_clips]):
                start = float(hl.get("start", 0))
                end = float(hl.get("end", start + 30))
                out_path = os.path.join(output_dir, f"highlight_{i + 1}.mp3")

                cmd = [
                    get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(start), "-i", audio_path,
                    "-t", str(end - start),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode == 0:
                    clips.append(out_path)
    except Exception as exc:
        logger.warning("Highlight extraction failed: %s", exc)

    return clips


def _generate_audiogram(audio_path: str, output_dir: str,
                         title: str = "") -> str:
    """Generate an audiogram video from the podcast audio."""
    try:
        from opencut.core.audiogram import generate_audiogram

        out_path = os.path.join(output_dir, "audiogram.mp4")
        result = generate_audiogram(
            audio_path=audio_path,
            output_path_str=out_path,
            style="bars",
            width=1080,
            height=1080,
            title_text=title[:200] if title else None,
            duration=60.0,  # 60s preview
        )
        return result.get("output_path", "")
    except Exception as exc:
        logger.warning("Audiogram generation failed: %s", exc)
        return ""


def _generate_show_notes(transcript_text: str) -> tuple:
    """Generate show notes in markdown and HTML."""
    try:
        from opencut.core.show_notes import export_show_notes, generate_show_notes

        notes = generate_show_notes(transcript_text)
        md = export_show_notes(notes, format="markdown")
        html = export_show_notes(notes, format="html")
        return md, html
    except Exception as exc:
        logger.warning("Show notes generation failed: %s", exc)
        return "# Show Notes\n\nCould not generate show notes.", ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_podcast_bundle(
    audio_path: str,
    title: str = "",
    output_dir: str = "",
    export_formats: Optional[List[str]] = None,
    max_highlight_clips: int = 3,
    generate_audiogram_flag: bool = True,
    on_progress: Optional[Callable] = None,
) -> PodcastBundleResult:
    """
    Create a full multi-platform podcast bundle from an audio file.

    Pipeline:
      1. Denoise + normalize audio
      2. Export clean audio in multiple formats
      3. Transcribe
      4. Extract chapters
      5. Extract highlight clips
      6. Generate audiogram video
      7. Generate show notes
      8. Bundle all outputs with manifest

    Args:
        audio_path: Path to the podcast audio file.
        title: Podcast episode title.
        output_dir: Output bundle directory.
        export_formats: Audio export formats (default: mp3, wav).
        max_highlight_clips: Number of highlight clips to extract.
        generate_audiogram_flag: Whether to generate an audiogram video.
        on_progress: Progress callback(pct, msg).

    Returns:
        PodcastBundleResult with all output paths and content.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not output_dir:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(os.path.dirname(audio_path), f"{base}_bundle")
    os.makedirs(output_dir, exist_ok=True)

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    result = PodcastBundleResult(output_dir=output_dir)
    duration = _get_audio_duration(audio_path)

    # Step 1: Denoise + normalize
    if on_progress:
        on_progress(5, "Cleaning and normalizing audio...")

    clean_path = os.path.join(output_dir, "clean_audio.m4a")
    try:
        _denoise_and_normalize(audio_path, clean_path)
        result.clean_audio_path = clean_path
    except Exception as exc:
        logger.warning("Denoise failed, using original: %s", exc)
        result.clean_audio_path = audio_path
        clean_path = audio_path

    # Step 2: Export clean audio
    if on_progress:
        on_progress(15, "Exporting clean audio formats...")

    try:
        audio_exports = _export_clean_audio(
            clean_path, output_dir,
            formats=export_formats or ["mp3", "wav"],
        )
    except Exception as exc:
        logger.warning("Audio export failed: %s", exc)
        audio_exports = {}

    # Step 3: Transcribe
    if on_progress:
        on_progress(25, "Transcribing podcast...")

    transcript_text, segments = _transcribe_audio(clean_path)
    result.transcript_text = transcript_text

    # Save transcript
    if transcript_text:
        tx_path = os.path.join(output_dir, "transcript.txt")
        with open(tx_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

    # Step 4: Extract chapters
    if on_progress:
        on_progress(40, "Extracting chapters...")

    chapters = _extract_chapters(transcript_text, segments)
    result.chapters = chapters

    # Save chapters
    if chapters:
        ch_path = os.path.join(output_dir, "chapters.json")
        with open(ch_path, "w", encoding="utf-8") as f:
            json.dump(chapters, f, indent=2)

    # Step 5: Highlight clips
    if on_progress:
        on_progress(55, "Extracting highlight clips...")

    highlight_clips = _extract_highlight_clips(
        clean_path, transcript_text, clips_dir,
        max_clips=max_highlight_clips,
    )
    result.highlight_clips = highlight_clips

    # Step 6: Audiogram
    if generate_audiogram_flag:
        if on_progress:
            on_progress(70, "Generating audiogram...")

        audiogram_path = _generate_audiogram(clean_path, output_dir, title=title)
        result.audiogram_path = audiogram_path

    # Step 7: Show notes
    if on_progress:
        on_progress(85, "Generating show notes...")

    md, html = _generate_show_notes(transcript_text)
    result.show_notes_markdown = md
    result.show_notes_html = html

    # Save show notes
    if md:
        md_path = os.path.join(output_dir, "show_notes.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
    if html:
        html_path = os.path.join(output_dir, "show_notes.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    # Step 8: Bundle manifest
    if on_progress:
        on_progress(95, "Creating bundle manifest...")

    manifest = {
        "title": title or os.path.splitext(os.path.basename(audio_path))[0],
        "source": os.path.abspath(audio_path),
        "duration_seconds": round(duration, 2),
        "output_dir": os.path.abspath(output_dir),
        "clean_audio": result.clean_audio_path,
        "audio_exports": audio_exports,
        "transcript_available": bool(transcript_text),
        "chapter_count": len(chapters),
        "highlight_count": len(highlight_clips),
        "audiogram": result.audiogram_path,
        "show_notes_formats": ["markdown", "html"] if md else [],
    }
    result.manifest = manifest

    manifest_path = os.path.join(output_dir, "bundle_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if on_progress:
        on_progress(100, "Podcast bundle complete")

    return result
