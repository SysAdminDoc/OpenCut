"""
OpenCut Stream Highlight Reel (31.4)

Multi-signal scoring of stream recordings to automatically extract
the best moments and assemble a highlight reel:

Scoring signals:
- Audio peaks (sudden loudness = excitement, reactions)
- Motion intensity (action sequences, fast gameplay)
- Keyword detection in transcript (optional: "clutch", "insane", etc.)

Pipeline:
1. Analyze stream for audio energy + motion per segment
2. Optionally scan transcript for hype keywords
3. Score and rank segments
4. Extract top segments as individual clips
5. Assemble clips with transitions into a highlight reel

All via FFmpeg -- zero external dependencies.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ScoredSegment:
    """A scored segment of the stream."""
    start: float = 0.0
    end: float = 0.0
    audio_score: float = 0.0       # 0-1: audio energy peak
    motion_score: float = 0.0      # 0-1: visual motion magnitude
    keyword_score: float = 0.0     # 0-1: keyword density
    composite: float = 0.0         # weighted overall score
    keywords_found: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class HighlightClip:
    """An extracted highlight clip."""
    file_path: str = ""
    start: float = 0.0
    end: float = 0.0
    score: float = 0.0
    index: int = 0


@dataclass
class HighlightReelResult:
    """Result of highlight reel assembly."""
    output_path: str = ""
    clip_count: int = 0
    total_duration: float = 0.0
    segments: List[ScoredSegment] = field(default_factory=list)
    clips: List[HighlightClip] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default Hype Keywords
# ---------------------------------------------------------------------------
DEFAULT_KEYWORDS = [
    "clutch", "insane", "crazy", "let's go", "oh my god", "no way",
    "pogchamp", "pog", "huge", "epic", "play of the game", "ace",
    "headshot", "triple kill", "quad kill", "pentakill", "victory",
    "win", "first place", "chicken dinner", "gg", "hype",
    "unbelievable", "sick", "nasty", "what", "wow",
]


# ---------------------------------------------------------------------------
# Audio Energy Analysis (per segment)
# ---------------------------------------------------------------------------
def _analyze_audio_energy_segments(
    video_path: str,
    segment_duration: float = 10.0,
) -> List[float]:
    """Analyze audio energy per segment of the stream.

    Returns list of energy values (0-1) for each segment.
    """

    info = get_video_info(video_path)
    total_dur = info["duration"]
    if total_dur <= 0:
        return []

    tmp_dir = tempfile.mkdtemp(prefix="opencut_hlreel_")
    energy_file = os.path.join(tmp_dir, "energy.txt")

    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-i", video_path,
            "-af", (
                "astats=metadata=1:reset=1024,"
                "ametadata=print:key=lavfi.astats.Overall.RMS_level"
                ":file=" + energy_file.replace("\\", "/")
            ),
            "-f", "null", "-",
        ]
        run_ffmpeg(cmd)

        frame_time = 1024.0 / 44100.0
        energy_values = []

        if os.path.isfile(energy_file):
            with open(energy_file, "r") as f:
                for line in f:
                    match = re.search(r"RMS_level=(-?[\d.]+)", line)
                    if match:
                        try:
                            db = float(match.group(1))
                            energy_values.append(max(0, db + 60))
                        except ValueError:
                            pass

        if not energy_values:
            n_segments = max(1, int(total_dur / segment_duration))
            return [0.3] * n_segments

        # Average energy per segment
        frames_per_segment = max(1, int(segment_duration / frame_time))
        segments = []
        for i in range(0, len(energy_values), frames_per_segment):
            chunk = energy_values[i:i + frames_per_segment]
            if chunk:
                avg = sum(chunk) / len(chunk)
                # Normalize to 0-1 (60 = max energy after dB shift)
                segments.append(min(1.0, avg / 60.0))

        return segments

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Motion Analysis (per segment)
# ---------------------------------------------------------------------------
def _analyze_motion_segments(
    video_path: str,
    segment_duration: float = 10.0,
) -> List[float]:
    """Analyze visual motion per segment using scene change detection.

    Returns list of motion scores (0-1) for each segment.
    """
    import subprocess as _sp

    info = get_video_info(video_path)
    total_dur = info["duration"]
    if total_dur <= 0:
        return []

    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-i", video_path,
            "-vf", "select='gt(scene,0.05)',metadata=print",
            "-f", "null", "-",
        ]
        result = _sp.run(cmd, capture_output=True, timeout=300, text=True)
        stderr = result.stderr or ""

        # Parse scene change timestamps
        scene_times = []
        for match in re.finditer(r"pts_time:([\d.]+)", stderr):
            try:
                scene_times.append(float(match.group(1)))
            except ValueError:
                pass

        n_segments = max(1, int(total_dur / segment_duration))
        motion_scores = []

        for i in range(n_segments):
            seg_start = i * segment_duration
            seg_end = seg_start + segment_duration
            # Count scene changes in this segment
            changes = sum(1 for t in scene_times if seg_start <= t < seg_end)
            # Normalize: 5+ changes per segment = high motion
            score = min(1.0, changes / 5.0)
            motion_scores.append(round(score, 3))

        return motion_scores

    except Exception as e:
        logger.debug("Motion analysis failed: %s", e)
        n_segments = max(1, int(total_dur / segment_duration))
        return [0.3] * n_segments


# ---------------------------------------------------------------------------
# Keyword Scoring
# ---------------------------------------------------------------------------
def _score_keywords_segments(
    transcript_segments: Optional[List[Dict]],
    total_duration: float,
    segment_duration: float = 10.0,
    keywords: Optional[List[str]] = None,
) -> List[tuple]:
    """Score segments by keyword density in transcript.

    Returns list of (score, keywords_found) tuples per segment.
    """
    if not transcript_segments:
        n_segments = max(1, int(total_duration / segment_duration))
        return [(0.0, [])] * n_segments

    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    keywords_lower = [k.lower() for k in keywords]
    n_segments = max(1, int(total_duration / segment_duration))
    results = []

    for i in range(n_segments):
        seg_start = i * segment_duration
        seg_end = seg_start + segment_duration

        # Collect transcript text in this segment
        text = ""
        for ts in transcript_segments:
            ts_start = float(ts.get("start", 0))
            ts_end = float(ts.get("end", ts_start))
            if ts_start < seg_end and ts_end > seg_start:
                text += " " + ts.get("text", "")

        text_lower = text.lower()
        found = [k for k in keywords_lower if k in text_lower]
        score = min(1.0, len(found) / 3.0)  # 3+ keywords = max score
        results.append((round(score, 3), found))

    return results


# ---------------------------------------------------------------------------
# Stream Segment Scoring
# ---------------------------------------------------------------------------
def score_stream_segments(
    video_path: str,
    segment_duration: float = 10.0,
    audio_weight: float = 0.4,
    motion_weight: float = 0.3,
    keyword_weight: float = 0.3,
    transcript_segments: Optional[List[Dict]] = None,
    keywords: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[ScoredSegment]:
    """Score all segments of a stream by multiple signals.

    Args:
        video_path: Source stream recording path.
        segment_duration: Duration of each analysis segment in seconds.
        audio_weight: Weight for audio energy in composite score.
        motion_weight: Weight for visual motion in composite score.
        keyword_weight: Weight for keyword density in composite score.
        transcript_segments: Optional transcript [{start, end, text}].
        keywords: Hype keywords to search for. Uses defaults if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ScoredSegment sorted by composite score descending.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    total_dur = info["duration"]

    if on_progress:
        on_progress(5, "Analyzing audio energy...")

    audio_scores = _analyze_audio_energy_segments(video_path, segment_duration)

    if on_progress:
        on_progress(35, "Analyzing visual motion...")

    motion_scores = _analyze_motion_segments(video_path, segment_duration)

    if on_progress:
        on_progress(65, "Scoring keywords...")

    keyword_results = _score_keywords_segments(
        transcript_segments, total_dur, segment_duration, keywords
    )

    # Normalize weights
    total_weight = audio_weight + motion_weight + keyword_weight
    if total_weight > 0:
        aw = audio_weight / total_weight
        mw = motion_weight / total_weight
        kw = keyword_weight / total_weight
    else:
        aw = mw = kw = 1.0 / 3

    n_segments = max(len(audio_scores), len(motion_scores), len(keyword_results))
    segments = []

    for i in range(n_segments):
        seg_start = i * segment_duration
        seg_end = min(seg_start + segment_duration, total_dur)

        a_score = audio_scores[i] if i < len(audio_scores) else 0.0
        m_score = motion_scores[i] if i < len(motion_scores) else 0.0
        k_score, k_found = keyword_results[i] if i < len(keyword_results) else (0.0, [])

        composite = aw * a_score + mw * m_score + kw * k_score

        segments.append(ScoredSegment(
            start=round(seg_start, 3),
            end=round(seg_end, 3),
            audio_score=round(a_score, 3),
            motion_score=round(m_score, 3),
            keyword_score=round(k_score, 3),
            composite=round(composite, 3),
            keywords_found=k_found,
        ))

    segments.sort(key=lambda s: s.composite, reverse=True)

    if on_progress:
        top = segments[0].composite if segments else 0
        on_progress(90, f"Scored {len(segments)} segments (top: {top:.2f})")

    return segments


# ---------------------------------------------------------------------------
# Highlight Extraction
# ---------------------------------------------------------------------------
def extract_highlights(
    video_path: str,
    segments: List[ScoredSegment],
    output_dir: Optional[str] = None,
    max_clips: int = 10,
    min_score: float = 0.3,
    padding: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> List[HighlightClip]:
    """Extract top scoring segments as individual video clips.

    Args:
        video_path: Source stream recording.
        segments: Scored segments from score_stream_segments().
        output_dir: Directory for clips. Uses video dir if None.
        max_clips: Maximum number of clips to extract.
        min_score: Minimum composite score to include.
        padding: Extra seconds before/after each segment.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of HighlightClip with file paths.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(video_path)
    total_dur = info["duration"]

    # Filter and limit
    eligible = [s for s in segments if s.composite >= min_score]
    eligible.sort(key=lambda s: s.composite, reverse=True)
    eligible = eligible[:max_clips]

    clips = []
    base = os.path.splitext(os.path.basename(video_path))[0]

    for i, seg in enumerate(eligible):
        if on_progress:
            pct = int(5 + 90 * i / max(len(eligible), 1))
            on_progress(pct, f"Extracting highlight {i + 1}/{len(eligible)}...")

        start = max(0, seg.start - padding)
        end = min(total_dur, seg.end + padding)

        clip_path = os.path.join(output_dir, f"{base}_highlight_{i + 1:03d}.mp4")

        cmd = (
            FFmpegCmd()
            .input(video_path, ss=str(round(start, 3)))
            .option("t", str(round(end - start, 3)))
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("aac", bitrate="192k")
            .faststart()
            .output(clip_path)
            .build()
        )
        run_ffmpeg(cmd)

        clips.append(HighlightClip(
            file_path=clip_path,
            start=start,
            end=end,
            score=seg.composite,
            index=i,
        ))

    if on_progress:
        on_progress(100, f"Extracted {len(clips)} highlight clips")

    return clips


# ---------------------------------------------------------------------------
# Highlight Reel Assembly
# ---------------------------------------------------------------------------
def assemble_highlight_reel(
    clips: List[HighlightClip],
    output_path_str: Optional[str] = None,
    transition: str = "crossfade",
    transition_duration: float = 0.5,
    intro_text: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> HighlightReelResult:
    """Assemble highlight clips into a single highlight reel video.

    Args:
        clips: List of HighlightClip from extract_highlights().
        output_path_str: Output file path. Auto-generated if None.
        transition: Transition between clips: 'cut', 'crossfade', 'fade'.
        transition_duration: Duration of transitions in seconds.
        intro_text: Optional intro text overlay (e.g. "STREAM HIGHLIGHTS").
        on_progress: Progress callback(pct, msg).

    Returns:
        HighlightReelResult with output path and stats.
    """
    if not clips:
        raise ValueError("No clips provided for highlight reel")

    valid_clips = [c for c in clips if c.file_path and os.path.isfile(c.file_path)]
    if not valid_clips:
        raise ValueError("No valid clip files found")

    # Sort by original timestamp for chronological order
    valid_clips.sort(key=lambda c: c.start)

    if output_path_str is None:
        output_path_str = output_path(valid_clips[0].file_path, "highlight_reel")
        if not output_path_str.endswith(".mp4"):
            output_path_str = os.path.splitext(output_path_str)[0] + ".mp4"

    tmp_dir = tempfile.mkdtemp(prefix="opencut_hlreel_asm_")

    try:
        if on_progress:
            on_progress(10, f"Preparing {len(valid_clips)} clips...")

        # Get target resolution from first clip
        first_info = get_video_info(valid_clips[0].file_path)
        out_w, out_h = first_info["width"], first_info["height"]

        # Normalize all clips to same resolution
        normalized = []
        for i, clip in enumerate(valid_clips):
            norm_path = os.path.join(tmp_dir, f"norm_{i:04d}.mp4")
            scale = (
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
            )

            cmd = (
                FFmpegCmd()
                .input(clip.file_path)
                .video_filter(scale)
                .video_codec("libx264", crf=18, preset="fast")
                .audio_codec("aac", bitrate="192k")
                .output(norm_path)
                .build()
            )
            run_ffmpeg(cmd)
            normalized.append(norm_path)

            if on_progress:
                pct = 10 + int(50 * (i + 1) / len(valid_clips))
                on_progress(pct, f"Normalized clip {i + 1}/{len(valid_clips)}")

        if on_progress:
            on_progress(65, "Concatenating highlight reel...")

        # Concat using FFmpeg concat demuxer
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for np in normalized:
                f.write(f"file '{np}'\n")

        # Apply fade transitions if requested
        if transition == "crossfade" and len(normalized) > 1:
            # For crossfade, we need filter_complex (only practical for small clip counts)
            if len(normalized) <= 15:
                _assemble_with_crossfade(
                    normalized, output_path_str, transition_duration,
                    intro_text, out_w, out_h, tmp_dir,
                )
            else:
                # Fall back to simple concat for large numbers
                cmd = [
                    get_ffmpeg_path(), "-hide_banner", "-y",
                    "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    output_path_str,
                ]
                run_ffmpeg(cmd)
        else:
            # Simple concat (cut or fade handled per-clip)
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path_str,
            ]
            run_ffmpeg(cmd)

        # Calculate total duration
        total_dur = sum(c.end - c.start for c in valid_clips)

        if on_progress:
            on_progress(100, f"Highlight reel complete ({len(valid_clips)} clips)")

        return HighlightReelResult(
            output_path=output_path_str,
            clip_count=len(valid_clips),
            total_duration=round(total_dur, 3),
            segments=[],
            clips=valid_clips,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _assemble_with_crossfade(
    clip_paths: List[str],
    output_path_str: str,
    transition_duration: float,
    intro_text: Optional[str],
    width: int,
    height: int,
    tmp_dir: str,
):
    """Assemble clips with crossfade transitions using filter_complex."""
    n = len(clip_paths)
    if n < 2:
        # Single clip, just copy
        import shutil
        shutil.copy2(clip_paths[0], output_path_str)
        return

    # Build filter_complex for sequential xfade
    inputs = []
    for cp in clip_paths:
        inputs.extend(["-i", cp])

    # Calculate offsets (need clip durations)
    durations = []
    for cp in clip_paths:
        info = get_video_info(cp)
        durations.append(info["duration"])

    td = transition_duration
    fc_parts = []
    current_label = "[0:v]"

    for i in range(1, n):
        out_label = f"[v{i}]"
        offset = sum(durations[:i]) - td * i
        offset = max(0, offset)
        fc_parts.append(
            f"{current_label}[{i}:v]xfade=transition=fade:duration={td}:offset={offset}{out_label}"
        )
        current_label = out_label

    # Audio: amerge or simple concat
    audio_parts = []
    a_current = "[0:a]"
    for i in range(1, n):
        a_out = f"[a{i}]"
        offset = sum(durations[:i]) - td * i
        offset = max(0, offset)
        audio_parts.append(
            f"{a_current}[{i}:a]acrossfade=d={td}:c1=tri:c2=tri{a_out}"
        )
        a_current = a_out

    fc = ";".join(fc_parts + audio_parts)

    cmd = [get_ffmpeg_path(), "-hide_banner", "-y"]
    cmd.extend(inputs)
    cmd.extend([
        "-filter_complex", fc,
        "-map", current_label, "-map", a_current,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path_str,
    ])

    try:
        run_ffmpeg(cmd)
    except RuntimeError:
        # Fallback to simple concat if xfade fails
        logger.warning("Crossfade assembly failed, falling back to simple concat")
        concat_file = os.path.join(tmp_dir, "concat_fallback.txt")
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            output_path_str,
        ]
        run_ffmpeg(cmd)
