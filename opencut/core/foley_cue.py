"""
OpenCut Foley Cueing & Sound Effect Placement

Analyze video for action events that need foley, generate cue sheets,
and auto-place sound effects from a library at detected cue timecodes.

Foley categories (8):
- footstep:   Periodic low-frequency impacts during walking motion
- door:       Large frame changes with audio transient
- impact:     Sudden motion + audio spike
- cloth:      Continuous motion (cloth rustle)
- glass:      High-frequency transient
- water:      Sustained low-mid motion
- mechanical: Repetitive motion patterns
- ambient:    Background atmosphere

Detection uses FFmpeg scene-change, motion, and audio transient analysis.
SFX placement mixes categorized library files at cue timecodes via FFmpeg.
"""

import csv
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOLEY_CATEGORIES: Dict[str, Dict] = {
    "footstep": {
        "label": "Footstep",
        "description": "Walking/running detected via periodic low-frequency impacts",
        "freq_range": (80, 250),
        "duration_range": (0.05, 0.15),
        "min_interval": 0.3,
        "keywords": ["step", "walk", "run", "foot"],
    },
    "door": {
        "label": "Door",
        "description": "Scene boundary with large frame change suggesting door open/close",
        "freq_range": (100, 400),
        "duration_range": (0.2, 0.6),
        "min_interval": 1.0,
        "keywords": ["door", "open", "close", "slam", "creak"],
    },
    "impact": {
        "label": "Impact",
        "description": "Sudden motion spike with audio transient -- collision or hit",
        "freq_range": (60, 300),
        "duration_range": (0.05, 0.3),
        "min_interval": 0.5,
        "keywords": ["hit", "bang", "crash", "punch", "thud", "impact"],
    },
    "cloth": {
        "label": "Cloth Rustle",
        "description": "Continuous motion suggesting fabric movement",
        "freq_range": (800, 4000),
        "duration_range": (0.2, 1.0),
        "min_interval": 0.5,
        "keywords": ["cloth", "fabric", "rustle", "swoosh"],
    },
    "glass": {
        "label": "Glass/Ceramic",
        "description": "High-frequency transient suggesting glass or ceramic interaction",
        "freq_range": (2000, 8000),
        "duration_range": (0.1, 0.5),
        "min_interval": 0.5,
        "keywords": ["glass", "ceramic", "clink", "shatter", "break"],
    },
    "water": {
        "label": "Water",
        "description": "Sustained low-mid frequency patterns suggesting water",
        "freq_range": (200, 1500),
        "duration_range": (0.5, 3.0),
        "min_interval": 1.0,
        "keywords": ["water", "splash", "drip", "pour", "rain"],
    },
    "mechanical": {
        "label": "Mechanical",
        "description": "Repetitive motion patterns suggesting machinery or mechanisms",
        "freq_range": (100, 2000),
        "duration_range": (0.1, 1.0),
        "min_interval": 0.3,
        "keywords": ["machine", "mechanical", "click", "gear", "lever", "switch"],
    },
    "ambient": {
        "label": "Ambient",
        "description": "Background atmosphere for scene context",
        "freq_range": (80, 1000),
        "duration_range": (2.0, 10.0),
        "min_interval": 5.0,
        "keywords": ["ambient", "room", "tone", "background", "atmo"],
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class FoleyCue:
    """A single foley cue point detected in video."""

    cue_id: str = ""
    event_type: str = ""
    timecode: float = 0.0
    duration: float = 0.1
    intensity: float = 0.5
    suggested_sound: str = ""
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoleySession:
    """A foley session containing detected cues and placement info."""

    session_id: str = ""
    source_path: str = ""
    cues: List[FoleyCue] = field(default_factory=list)
    total_duration: float = 0.0
    categories_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source_path": self.source_path,
            "cue_count": len(self.cues),
            "cues": [c.to_dict() for c in self.cues],
            "total_duration": self.total_duration,
            "categories_detected": self.categories_detected,
        }


# ---------------------------------------------------------------------------
# Public API - Detection
# ---------------------------------------------------------------------------
def detect_foley_cues(
    video_path: str,
    categories: Optional[List[str]] = None,
    sensitivity: float = 0.5,
    max_cues: int = 500,
    on_progress: Optional[Callable] = None,
) -> FoleySession:
    """
    Analyze video for action events that need foley sound effects.

    Detection pipeline:
    1. Scene-change detection via FFmpeg scdet
    2. Motion magnitude analysis via frame-diff
    3. Audio transient detection via volumedetect
    4. Classify events into foley categories
    5. Generate cues with suggested sounds

    Args:
        video_path: Source video file.
        categories: Filter to specific categories (None = all).
        sensitivity: Detection sensitivity 0.0-1.0 (higher = more cues).
        max_cues: Maximum number of cues to return.
        on_progress: Progress callback taking one int (percentage).

    Returns:
        FoleySession with detected cues.
    """
    sensitivity = max(0.0, min(1.0, float(sensitivity)))
    max_cues = max(1, min(2000, int(max_cues)))

    if categories:
        categories = [c for c in categories if c in FOLEY_CATEGORIES]
    if not categories:
        categories = list(FOLEY_CATEGORIES.keys())

    if on_progress:
        on_progress(5)

    # Get video info
    info = get_video_info(video_path)
    total_duration = info.get("duration", 0)

    # Phase 1: Scene change detection
    if on_progress:
        on_progress(10)
    scene_changes = _detect_scene_changes(video_path)

    # Phase 2: Motion analysis
    if on_progress:
        on_progress(30)
    motion_peaks = _detect_motion_peaks(video_path)

    # Phase 3: Audio transient detection
    if on_progress:
        on_progress(50)
    audio_transients = _detect_audio_transients(video_path)

    if on_progress:
        on_progress(65)

    # Phase 4: Classify events into foley categories
    raw_cues = _classify_events(
        scene_changes, motion_peaks, audio_transients,
        categories, sensitivity,
    )

    if on_progress:
        on_progress(80)

    # Phase 5: Deduplicate and sort
    cues = _deduplicate_cues(raw_cues, categories)

    # Trim to max_cues
    cues = cues[:max_cues]

    # Assign IDs and suggested sounds
    detected_cats = set()
    for i, cue in enumerate(cues):
        cue.cue_id = f"FOL-{i + 1:04d}"
        cue.suggested_sound = _suggest_sound(cue.event_type)
        detected_cats.add(cue.event_type)

    if on_progress:
        on_progress(95)

    import uuid
    session = FoleySession(
        session_id=uuid.uuid4().hex[:12],
        source_path=video_path,
        cues=cues,
        total_duration=total_duration,
        categories_detected=sorted(detected_cats),
    )

    if on_progress:
        on_progress(100)

    logger.info("Detected %d foley cues in %s (%d categories)",
                len(cues), video_path, len(detected_cats))
    return session


# ---------------------------------------------------------------------------
# Public API - Cue Sheet Export
# ---------------------------------------------------------------------------
def export_cue_sheet(
    session: FoleySession,
    output_path: str,
    export_format: str = "csv",
    on_progress: Optional[Callable] = None,
) -> str:
    """Export foley cues as CSV or JSON cue sheet.

    Args:
        session: FoleySession with cues.
        output_path: Output file path.
        export_format: "csv" or "json".
        on_progress: Progress callback.

    Returns:
        Path to the exported file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if on_progress:
        on_progress(10)

    if export_format == "json":
        data = {
            "session_id": session.session_id,
            "source": session.source_path,
            "total_duration": session.total_duration,
            "cue_count": len(session.cues),
            "categories": session.categories_detected,
            "cues": [c.to_dict() for c in session.cues],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Cue#", "Type", "Timecode", "Duration",
                "Intensity", "Suggested Sound", "Confidence",
            ])
            for cue in session.cues:
                writer.writerow([
                    cue.cue_id, cue.event_type,
                    f"{cue.timecode:.3f}", f"{cue.duration:.3f}",
                    f"{cue.intensity:.2f}", cue.suggested_sound,
                    f"{cue.confidence:.2f}",
                ])

    if on_progress:
        on_progress(100)

    logger.info("Exported %d foley cues to %s (%s)", len(session.cues), output_path, export_format)
    return output_path


# ---------------------------------------------------------------------------
# Public API - SFX Placement
# ---------------------------------------------------------------------------
def place_sfx(
    session: FoleySession,
    sfx_library_dir: str,
    source_audio_path: str,
    output_path: Optional[str] = None,
    mix_level: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Auto-place sound effects from a library at foley cue timecodes.

    Matches cues to SFX files by category keyword, then mixes them onto
    the source audio at the cue positions. Volume is scaled by cue intensity.

    Args:
        session: FoleySession with detected cues.
        sfx_library_dir: Directory of categorized SFX files.
        source_audio_path: Original audio to mix SFX onto.
        output_path: Output file path (auto-generated if None).
        mix_level: Overall SFX mix level 0.0-1.0.
        on_progress: Progress callback.

    Returns:
        dict with output_path, sfx_placed, total_cues, duration.
    """
    if not os.path.isdir(sfx_library_dir):
        raise ValueError(f"SFX library directory not found: {sfx_library_dir}")

    if output_path is None:
        output_path = _output_path(source_audio_path, "foley_mix")

    mix_level = max(0.0, min(1.0, float(mix_level)))

    if on_progress:
        on_progress(5)

    # Catalog available SFX files by category
    sfx_catalog = _catalog_sfx_library(sfx_library_dir)

    if on_progress:
        on_progress(15)

    # Match cues to SFX files
    placements = _match_cues_to_sfx(session.cues, sfx_catalog)

    if on_progress:
        on_progress(30)

    if not placements:
        # No matches found, just copy original
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_audio_path,
            "-c", "copy",
            output_path,
        ]
        run_ffmpeg(cmd, timeout=120)
        return {
            "output_path": output_path,
            "sfx_placed": 0,
            "total_cues": len(session.cues),
            "duration": session.total_duration,
            "notes": "No matching SFX files found in library",
        }

    # Generate SFX mix track
    sfx_track = _generate_sfx_track(
        placements, session.total_duration, mix_level, on_progress,
    )

    if on_progress:
        on_progress(75)

    # Mix SFX track with original audio
    _mix_audio_tracks(source_audio_path, sfx_track, output_path, mix_level)

    if on_progress:
        on_progress(95)

    # Clean up temp SFX track
    try:
        os.unlink(sfx_track)
    except OSError:
        pass

    if on_progress:
        on_progress(100)

    logger.info("Placed %d SFX from library onto %s -> %s",
                len(placements), source_audio_path, output_path)
    return {
        "output_path": output_path,
        "sfx_placed": len(placements),
        "total_cues": len(session.cues),
        "categories_used": list({p["category"] for p in placements}),
        "duration": session.total_duration,
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def list_categories() -> List[dict]:
    """List available foley categories with descriptions."""
    return [
        {"id": k, "label": v["label"], "description": v["description"],
         "keywords": v["keywords"]}
        for k, v in FOLEY_CATEGORIES.items()
    ]


# ---------------------------------------------------------------------------
# Detection Internals
# ---------------------------------------------------------------------------
def _detect_scene_changes(video_path: str) -> List[dict]:
    """Detect scene changes via FFmpeg scdet filter."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", "scdet=s=1:t=10",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Scene detection timed out")
        return []

    detections = []
    score_re = re.compile(r"lavfi\.scd\.score=(\d+\.?\d*)")
    time_re = re.compile(r"lavfi\.scd\.time=(\d+\.?\d*)")

    for line in result.stderr.split("\n"):
        sm = score_re.search(line)
        tm = time_re.search(line)
        if sm and tm:
            detections.append({
                "time": float(tm.group(1)),
                "score": float(sm.group(1)) / 100.0,
            })
    return detections


def _detect_motion_peaks(video_path: str) -> List[dict]:
    """Detect motion magnitude peaks via frame-difference analysis."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", "select='gt(scene,0)',metadata=mode=print",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return []

    peaks = []
    pts_re = re.compile(r"pts_time:(\d+\.?\d*)")
    scene_re = re.compile(r"lavfi\.scene_score=(\d+\.?\d*)")

    lines = result.stderr.split("\n")
    for i, line in enumerate(lines):
        pts_m = pts_re.search(line)
        score = 0.0
        for offset in range(min(3, len(lines) - i)):
            sc_m = scene_re.search(lines[i + offset])
            if sc_m:
                score = float(sc_m.group(1))
                break
        if pts_m:
            peaks.append({
                "time": float(pts_m.group(1)),
                "magnitude": min(1.0, score),
            })
    return peaks


def _detect_audio_transients(video_path: str) -> List[dict]:
    """Detect audio transients via FFmpeg silencedetect inverse approach.

    Identifies sudden loud events by analyzing volume peaks.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-af", "silencedetect=n=-30dB:d=0.1",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return []

    # Parse silence end times (these are where loud audio resumes = transient)
    transients = []
    end_re = re.compile(r"silence_end:\s*(\d+\.?\d*)")
    dur_re = re.compile(r"silence_duration:\s*(\d+\.?\d*)")

    for line in result.stderr.split("\n"):
        end_m = end_re.search(line)
        dur_m = dur_re.search(line)
        if end_m:
            t = float(end_m.group(1))
            sil_dur = float(dur_m.group(1)) if dur_m else 0.0
            # Shorter preceding silence with abrupt end = sharper transient
            intensity = min(1.0, 1.0 / max(0.1, sil_dur))
            transients.append({
                "time": t,
                "intensity": intensity,
            })
    return transients


def _classify_events(
    scene_changes: List[dict],
    motion_peaks: List[dict],
    audio_transients: List[dict],
    categories: List[str],
    sensitivity: float,
) -> List[FoleyCue]:
    """Classify detected events into foley categories."""
    cues = []
    threshold = 1.0 - sensitivity  # Lower threshold = more cues

    # Scene changes -> door, impact
    for sc in scene_changes:
        score = sc.get("score", 0)
        if score < threshold * 0.3:
            continue

        if score > 0.7 and "door" in categories:
            cues.append(FoleyCue(
                event_type="door",
                timecode=sc["time"],
                duration=0.4,
                intensity=min(1.0, score),
                confidence=score * 0.8,
            ))
        elif score > 0.5 and "impact" in categories:
            cues.append(FoleyCue(
                event_type="impact",
                timecode=sc["time"],
                duration=0.15,
                intensity=min(1.0, score),
                confidence=score * 0.6,
            ))

    # Motion peaks -> footstep, cloth, mechanical
    prev_motion_time = -10.0
    consecutive_motions = 0

    for mp in sorted(motion_peaks, key=lambda m: m["time"]):
        mag = mp.get("magnitude", 0)
        if mag < threshold * 0.2:
            continue

        t = mp["time"]
        gap = t - prev_motion_time

        if 0.3 < gap < 0.8:
            consecutive_motions += 1
        else:
            consecutive_motions = 0

        # Periodic low motion = footsteps
        if consecutive_motions >= 2 and mag < 0.5 and "footstep" in categories:
            cues.append(FoleyCue(
                event_type="footstep",
                timecode=t,
                duration=0.1,
                intensity=min(1.0, mag * 1.5),
                confidence=0.5 + min(0.3, consecutive_motions * 0.1),
            ))
        # Continuous motion = cloth
        elif 0.1 < mag < 0.4 and "cloth" in categories:
            cues.append(FoleyCue(
                event_type="cloth",
                timecode=t,
                duration=0.5,
                intensity=mag,
                confidence=0.4,
            ))
        # Repetitive patterns = mechanical
        elif consecutive_motions >= 3 and "mechanical" in categories:
            cues.append(FoleyCue(
                event_type="mechanical",
                timecode=t,
                duration=0.3,
                intensity=mag,
                confidence=0.5,
            ))

        prev_motion_time = t

    # Audio transients -> impact, glass, water
    for at in audio_transients:
        intensity = at.get("intensity", 0)
        if intensity < threshold * 0.3:
            continue

        if intensity > 0.8 and "impact" in categories:
            cues.append(FoleyCue(
                event_type="impact",
                timecode=at["time"],
                duration=0.15,
                intensity=intensity,
                confidence=intensity * 0.7,
            ))
        elif intensity > 0.6 and "glass" in categories:
            cues.append(FoleyCue(
                event_type="glass",
                timecode=at["time"],
                duration=0.25,
                intensity=intensity * 0.8,
                confidence=0.4,
            ))

    return cues


def _deduplicate_cues(
    cues: List[FoleyCue],
    categories: List[str],
) -> List[FoleyCue]:
    """Remove duplicate/overlapping cues, preferring higher confidence."""
    if not cues:
        return []

    # Sort by timecode, then confidence descending
    sorted_cues = sorted(cues, key=lambda c: (c.timecode, -c.confidence))

    result = []
    for cue in sorted_cues:
        if cue.event_type not in categories:
            continue

        cat = FOLEY_CATEGORIES.get(cue.event_type)
        min_interval = cat["min_interval"] if cat else 0.3

        # Check if too close to an existing cue of same type
        too_close = False
        for existing in result:
            if existing.event_type == cue.event_type:
                if abs(existing.timecode - cue.timecode) < min_interval:
                    too_close = True
                    break
        if not too_close:
            result.append(cue)

    return sorted(result, key=lambda c: c.timecode)


def _suggest_sound(event_type: str) -> str:
    """Suggest a descriptive sound name for a foley category."""
    suggestions = {
        "footstep": "hard_surface_step",
        "door": "wooden_door_close",
        "impact": "body_impact_medium",
        "cloth": "fabric_rustle",
        "glass": "glass_clink",
        "water": "water_splash_small",
        "mechanical": "switch_click",
        "ambient": "room_tone",
    }
    return suggestions.get(event_type, event_type)


# ---------------------------------------------------------------------------
# SFX Library & Placement
# ---------------------------------------------------------------------------
def _catalog_sfx_library(sfx_dir: str) -> Dict[str, List[str]]:
    """Catalog SFX files in a directory, organized by category.

    Matches files to categories by filename keywords.
    """
    catalog: Dict[str, List[str]] = {cat: [] for cat in FOLEY_CATEGORIES}
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff")

    for root, dirs, files in os.walk(sfx_dir):
        for fname in files:
            if not any(fname.lower().endswith(ext) for ext in audio_exts):
                continue

            fpath = os.path.join(root, fname)
            lower_name = fname.lower()
            # Also check parent directory name
            parent_name = os.path.basename(root).lower()
            search_text = f"{lower_name} {parent_name}"

            for cat_id, cat_info in FOLEY_CATEGORIES.items():
                for keyword in cat_info["keywords"]:
                    if keyword in search_text:
                        catalog[cat_id].append(fpath)
                        break

    return catalog


def _match_cues_to_sfx(
    cues: List[FoleyCue],
    sfx_catalog: Dict[str, List[str]],
) -> List[dict]:
    """Match foley cues to SFX files from the catalog."""
    placements = []

    for cue in cues:
        available = sfx_catalog.get(cue.event_type, [])
        if not available:
            continue

        # Pick a random file from the category (for variety)
        sfx_file = available[hash(cue.cue_id + cue.event_type) % len(available)]

        placements.append({
            "cue_id": cue.cue_id,
            "category": cue.event_type,
            "timecode": cue.timecode,
            "duration": cue.duration,
            "intensity": cue.intensity,
            "sfx_file": sfx_file,
        })

    return placements


def _generate_sfx_track(
    placements: List[dict],
    total_duration: float,
    mix_level: float,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate a single audio track with all SFX placed at cue timecodes.

    Creates a silence base track, then overlays each SFX at its position
    with volume scaled by intensity.

    Returns path to temp WAV file.
    """
    ffmpeg = get_ffmpeg_path()

    if total_duration <= 0:
        total_duration = max(p["timecode"] + p["duration"] for p in placements) + 1.0

    # Generate silent base track
    base_track = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, prefix="foley_base_",
    )
    base_track.close()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={total_duration:.3f}",
        "-t", str(total_duration),
        "-c:a", "pcm_s16le",
        base_track.name,
    ]
    run_ffmpeg(cmd, timeout=120)

    if not placements:
        return base_track.name

    # Overlay SFX one at a time (chaining would be too complex for many cues)
    current_track = base_track.name
    temp_files = [base_track.name]

    batch_size = 10  # Process in batches to avoid too many temp files
    batches = [placements[i:i + batch_size] for i in range(0, len(placements), batch_size)]

    for batch_idx, batch in enumerate(batches):
        next_track = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix=f"foley_mix_{batch_idx}_",
        )
        next_track.close()

        # Build complex filter for this batch
        inputs = ["-i", current_track]
        for p in batch:
            inputs.extend(["-i", p["sfx_file"]])

        filter_parts = []
        for i, p in enumerate(batch):
            stream_idx = i + 1  # 0 is the base track
            vol = p["intensity"] * mix_level
            delay_ms = int(p["timecode"] * 1000)
            filter_parts.append(
                f"[{stream_idx}]volume={vol:.3f},"
                f"adelay={delay_ms}|{delay_ms},"
                f"atrim=0:{total_duration:.3f},"
                f"apad=whole_dur={total_duration:.3f}[s{i}]"
            )

        # Mix all with base
        mix_inputs = "[0]"
        for i in range(len(batch)):
            mix_inputs += f"[s{i}]"
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(batch) + 1}:duration=first:dropout_transition=0"
        )

        filter_str = ";".join(filter_parts)

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-c:a", "pcm_s16le",
            next_track.name,
        ]

        try:
            run_ffmpeg(cmd, timeout=600)
            current_track = next_track.name
            temp_files.append(next_track.name)
        except Exception as exc:
            logger.warning("SFX batch %d overlay failed: %s", batch_idx, exc)
            try:
                os.unlink(next_track.name)
            except OSError:
                pass

        if on_progress:
            pct = 30 + int(40 * (batch_idx + 1) / max(1, len(batches)))
            on_progress(pct)

    # Clean up intermediate files (keep the final one)
    for tf in temp_files:
        if tf != current_track:
            try:
                os.unlink(tf)
            except OSError:
                pass

    return current_track


def _mix_audio_tracks(
    original_path: str,
    sfx_path: str,
    output_path: str,
    mix_level: float,
) -> None:
    """Mix SFX track with original audio."""
    ffmpeg = get_ffmpeg_path()

    # Weight: original at full, SFX at mix_level
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", original_path,
        "-i", sfx_path,
        "-filter_complex",
        f"[0:a]volume=1.0[a];[1:a]volume={mix_level:.3f}[b];"
        "[a][b]amix=inputs=2:duration=first:dropout_transition=2",
        "-c:a", "pcm_s16le",
        "-vn",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=1800)
