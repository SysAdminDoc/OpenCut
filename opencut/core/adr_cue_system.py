"""
OpenCut ADR Cue System

Full ADR (Automated Dialogue Replacement) workflow for audio post-production:
- Create and manage ADR sessions per project
- Add/update/remove cues with character, timecode, reason, priority
- Generate cue sheets in CSV (industry standard) and PDF-ready JSON
- Extract guide audio segments for actors from original media
- Accept replacement recordings and time-align to original
- Optional TTS placeholder generation for temp ADR

Persistent storage in ~/.opencut/adr/ as JSON per project.
All audio processing via FFmpeg.
"""

import csv
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADR_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "adr")

ADR_REASONS = ("noise", "script_change", "performance", "technical", "censorship", "other")
ADR_PRIORITIES = (1, 2, 3, 4, 5)
ADR_STATUSES = ("pending", "recorded", "approved", "rejected")

CSV_COLUMNS = ("Cue#", "Character", "Reel", "TC In", "TC Out", "Line", "Reason")


# ---------------------------------------------------------------------------
# Timecode Helpers
# ---------------------------------------------------------------------------
def _seconds_to_timecode(seconds: float, fps: float = 24.0) -> str:
    """Convert seconds to SMPTE timecode HH:MM:SS:FF."""
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    ff = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _timecode_to_seconds(tc: str, fps: float = 24.0) -> float:
    """Convert SMPTE timecode HH:MM:SS:FF to seconds."""
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: {tc}")
    hh, mm, ss, ff = (int(p) for p in parts)
    total_frames = hh * 3600 * fps + mm * 60 * fps + ss * fps + ff
    return total_frames / fps


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ADRCue:
    """A single ADR cue for dialogue replacement."""

    cue_id: str = ""
    character_name: str = ""
    original_line: str = ""
    timecode_in: float = 0.0
    timecode_out: float = 0.0
    scene_context: str = ""
    reason: str = "performance"
    priority: int = 3
    status: str = "pending"
    notes: str = ""
    takes_recorded: int = 0
    replacement_path: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def duration(self) -> float:
        return max(0.0, self.timecode_out - self.timecode_in)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration()
        return d


@dataclass
class ADRSession:
    """An ADR session for a project, containing all cues."""

    session_id: str = ""
    project_name: str = ""
    source_path: str = ""
    reel: str = "R1"
    fps: float = 24.0
    cues: List[ADRCue] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "project_name": self.project_name,
            "source_path": self.source_path,
            "reel": self.reel,
            "fps": self.fps,
            "cues": [c.to_dict() for c in self.cues],
            "cue_count": len(self.cues),
            "total_duration": sum(c.duration() for c in self.cues),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _ensure_adr_dir():
    """Create ADR storage directory if needed."""
    os.makedirs(ADR_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    """Return filesystem path for a session JSON file."""
    return os.path.join(ADR_DIR, f"{session_id}.json")


def _save_session(session: ADRSession) -> None:
    """Persist an ADR session to disk as JSON."""
    _ensure_adr_dir()
    session.updated_at = time.time()
    path = _session_path(session.session_id)
    data = {
        "session_id": session.session_id,
        "project_name": session.project_name,
        "source_path": session.source_path,
        "reel": session.reel,
        "fps": session.fps,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "cues": [asdict(c) for c in session.cues],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    logger.debug("Saved ADR session %s (%d cues)", session.session_id, len(session.cues))


def _load_session(session_id: str) -> Optional[ADRSession]:
    """Load an ADR session from disk. Returns None if not found."""
    path = _session_path(session_id)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    session = ADRSession(
        session_id=data.get("session_id", session_id),
        project_name=data.get("project_name", ""),
        source_path=data.get("source_path", ""),
        reel=data.get("reel", "R1"),
        fps=data.get("fps", 24.0),
        created_at=data.get("created_at", 0.0),
        updated_at=data.get("updated_at", 0.0),
    )
    for cd in data.get("cues", []):
        session.cues.append(ADRCue(**{
            k: v for k, v in cd.items()
            if k in ADRCue.__dataclass_fields__
        }))
    return session


def list_sessions() -> List[dict]:
    """List all stored ADR sessions (summary only)."""
    _ensure_adr_dir()
    results = []
    for fname in os.listdir(ADR_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ADR_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "session_id": data.get("session_id", ""),
                "project_name": data.get("project_name", ""),
                "cue_count": len(data.get("cues", [])),
                "created_at": data.get("created_at", 0),
                "updated_at": data.get("updated_at", 0),
            })
        except (json.JSONDecodeError, OSError):
            logger.debug("Skipping invalid ADR session file: %s", fname)
    return sorted(results, key=lambda x: x.get("updated_at", 0), reverse=True)


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------
def create_session(
    project_name: str,
    source_path: str = "",
    reel: str = "R1",
    fps: float = 24.0,
) -> ADRSession:
    """Create a new ADR session for a project."""
    session = ADRSession(
        session_id=uuid.uuid4().hex[:12],
        project_name=project_name.strip() or "Untitled",
        source_path=source_path,
        reel=reel.strip() or "R1",
        fps=max(1.0, min(120.0, fps)),
        created_at=time.time(),
        updated_at=time.time(),
    )
    _save_session(session)
    logger.info("Created ADR session %s for project '%s'", session.session_id, session.project_name)
    return session


def get_session(session_id: str) -> Optional[ADRSession]:
    """Retrieve an ADR session by ID."""
    return _load_session(session_id)


def delete_session(session_id: str) -> bool:
    """Delete an ADR session and its JSON file."""
    path = _session_path(session_id)
    if os.path.isfile(path):
        os.unlink(path)
        logger.info("Deleted ADR session %s", session_id)
        return True
    return False


# ---------------------------------------------------------------------------
# Cue Management
# ---------------------------------------------------------------------------
def add_cue(
    session_id: str,
    character_name: str,
    original_line: str,
    timecode_in: float,
    timecode_out: float,
    scene_context: str = "",
    reason: str = "performance",
    priority: int = 3,
    notes: str = "",
) -> ADRCue:
    """Add a new ADR cue to a session."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    reason = reason.strip().lower()
    if reason not in ADR_REASONS:
        reason = "other"
    priority = max(1, min(5, int(priority)))

    cue = ADRCue(
        cue_id=uuid.uuid4().hex[:8],
        character_name=character_name.strip(),
        original_line=original_line.strip(),
        timecode_in=max(0.0, float(timecode_in)),
        timecode_out=max(0.0, float(timecode_out)),
        scene_context=scene_context.strip(),
        reason=reason,
        priority=priority,
        status="pending",
        notes=notes.strip(),
        created_at=time.time(),
        updated_at=time.time(),
    )

    if cue.timecode_out <= cue.timecode_in:
        cue.timecode_out = cue.timecode_in + 1.0

    session.cues.append(cue)
    _save_session(session)
    logger.info("Added cue %s to session %s (%s: '%s')",
                cue.cue_id, session_id, cue.character_name, cue.original_line[:40])
    return cue


def update_cue(
    session_id: str,
    cue_id: str,
    **kwargs,
) -> ADRCue:
    """Update fields on an existing ADR cue."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    cue = None
    for c in session.cues:
        if c.cue_id == cue_id:
            cue = c
            break
    if cue is None:
        raise ValueError(f"ADR cue not found: {cue_id}")

    allowed = {
        "character_name", "original_line", "timecode_in", "timecode_out",
        "scene_context", "reason", "priority", "status", "notes",
    }
    for key, value in kwargs.items():
        if key not in allowed:
            continue
        if key == "reason":
            value = str(value).strip().lower()
            if value not in ADR_REASONS:
                value = "other"
        elif key == "priority":
            value = max(1, min(5, int(value)))
        elif key == "status":
            value = str(value).strip().lower()
            if value not in ADR_STATUSES:
                continue
        elif key in ("timecode_in", "timecode_out"):
            value = max(0.0, float(value))
        else:
            value = str(value).strip()
        setattr(cue, key, value)

    cue.updated_at = time.time()
    _save_session(session)
    return cue


def remove_cue(session_id: str, cue_id: str) -> bool:
    """Remove a cue from a session."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    original_len = len(session.cues)
    session.cues = [c for c in session.cues if c.cue_id != cue_id]
    if len(session.cues) == original_len:
        return False
    _save_session(session)
    return True


def list_cues(
    session_id: str,
    status_filter: Optional[str] = None,
    character_filter: Optional[str] = None,
    sort_by: str = "timecode",
) -> List[dict]:
    """List cues for a session with optional filters."""
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    cues = session.cues

    if status_filter and status_filter in ADR_STATUSES:
        cues = [c for c in cues if c.status == status_filter]

    if character_filter:
        cf = character_filter.strip().lower()
        cues = [c for c in cues if cf in c.character_name.lower()]

    if sort_by == "priority":
        cues = sorted(cues, key=lambda c: c.priority)
    elif sort_by == "character":
        cues = sorted(cues, key=lambda c: c.character_name.lower())
    elif sort_by == "status":
        status_order = {s: i for i, s in enumerate(ADR_STATUSES)}
        cues = sorted(cues, key=lambda c: status_order.get(c.status, 99))
    else:  # timecode (default)
        cues = sorted(cues, key=lambda c: c.timecode_in)

    return [c.to_dict() for c in cues]


# ---------------------------------------------------------------------------
# Cue Sheet Export
# ---------------------------------------------------------------------------
def export_cue_sheet_csv(
    session_id: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export ADR cue sheet as CSV with industry standard columns.

    Columns: Cue#, Character, Reel, TC In, TC Out, Line, Reason

    Returns:
        Path to the CSV file.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    if on_progress:
        on_progress(10)

    if output_path is None:
        _ensure_adr_dir()
        output_path = os.path.join(ADR_DIR, f"{session.session_id}_cuesheet.csv")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cues_sorted = sorted(session.cues, key=lambda c: c.timecode_in)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(CSV_COLUMNS))
        for i, cue in enumerate(cues_sorted):
            writer.writerow([
                f"ADR-{i + 1:04d}",
                cue.character_name,
                session.reel,
                _seconds_to_timecode(cue.timecode_in, session.fps),
                _seconds_to_timecode(cue.timecode_out, session.fps),
                cue.original_line,
                cue.reason,
            ])
            if on_progress:
                pct = 10 + int(80 * (i + 1) / max(1, len(cues_sorted)))
                on_progress(pct)

    if on_progress:
        on_progress(100)

    logger.info("Exported CSV cue sheet: %s (%d cues)", output_path, len(cues_sorted))
    return output_path


def export_cue_sheet_json(
    session_id: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export ADR cue sheet as PDF-ready JSON.

    The JSON structure is designed for direct consumption by PDF
    renderers or report generators.

    Returns:
        Path to the JSON file.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    if on_progress:
        on_progress(10)

    if output_path is None:
        _ensure_adr_dir()
        output_path = os.path.join(ADR_DIR, f"{session.session_id}_cuesheet.json")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cues_sorted = sorted(session.cues, key=lambda c: c.timecode_in)

    pdf_data = {
        "title": f"ADR Cue Sheet - {session.project_name}",
        "project_name": session.project_name,
        "reel": session.reel,
        "fps": session.fps,
        "generated_at": time.time(),
        "total_cues": len(cues_sorted),
        "total_duration": sum(c.duration() for c in cues_sorted),
        "characters": list({c.character_name for c in cues_sorted}),
        "status_summary": {
            s: sum(1 for c in cues_sorted if c.status == s)
            for s in ADR_STATUSES
        },
        "cues": [],
    }

    for i, cue in enumerate(cues_sorted):
        pdf_data["cues"].append({
            "cue_number": f"ADR-{i + 1:04d}",
            "cue_id": cue.cue_id,
            "character": cue.character_name,
            "reel": session.reel,
            "tc_in": _seconds_to_timecode(cue.timecode_in, session.fps),
            "tc_out": _seconds_to_timecode(cue.timecode_out, session.fps),
            "tc_in_seconds": cue.timecode_in,
            "tc_out_seconds": cue.timecode_out,
            "duration": cue.duration(),
            "line": cue.original_line,
            "scene_context": cue.scene_context,
            "reason": cue.reason,
            "priority": cue.priority,
            "status": cue.status,
            "notes": cue.notes,
        })
        if on_progress:
            pct = 10 + int(80 * (i + 1) / max(1, len(cues_sorted)))
            on_progress(pct)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=2)

    if on_progress:
        on_progress(100)

    logger.info("Exported JSON cue sheet: %s (%d cues)", output_path, len(cues_sorted))
    return output_path


# ---------------------------------------------------------------------------
# Guide Audio Extraction
# ---------------------------------------------------------------------------
def extract_guide_audio(
    session_id: str,
    cue_id: str,
    preroll: float = 3.0,
    postroll: float = 2.0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract a guide audio segment from the original media for an ADR cue.

    The guide includes preroll/postroll around the cue timecodes so the
    actor can hear context before and after the line.

    Args:
        session_id: ADR session ID.
        cue_id: ID of the specific cue.
        preroll: Seconds of audio before the cue start.
        postroll: Seconds of audio after the cue end.
        output_dir: Directory for the output file.
        on_progress: Progress callback taking one int (percentage).

    Returns:
        dict with output_path, cue_id, start, end, duration.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    if not session.source_path or not os.path.isfile(session.source_path):
        raise ValueError(f"Source media not found: {session.source_path}")

    cue = None
    for c in session.cues:
        if c.cue_id == cue_id:
            cue = c
            break
    if cue is None:
        raise ValueError(f"ADR cue not found: {cue_id}")

    if on_progress:
        on_progress(10)

    preroll = max(0.0, min(30.0, preroll))
    postroll = max(0.0, min(30.0, postroll))

    start = max(0.0, cue.timecode_in - preroll)
    end = cue.timecode_out + postroll
    duration = end - start

    out_dir = output_dir or ADR_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"guide_{session.session_id}_{cue.cue_id}.wav")

    if on_progress:
        on_progress(30)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", session.source_path,
        "-vn", "-c:a", "pcm_s16le", "-ar", "48000",
        out_path,
    ]
    run_ffmpeg(cmd, timeout=300)

    if on_progress:
        on_progress(100)

    logger.info("Extracted guide audio for cue %s: %.1fs at %s", cue_id, duration, out_path)
    return {
        "output_path": out_path,
        "cue_id": cue_id,
        "character": cue.character_name,
        "start": start,
        "end": end,
        "duration": duration,
        "preroll": preroll,
        "postroll": postroll,
    }


# ---------------------------------------------------------------------------
# Replacement Recording Processing
# ---------------------------------------------------------------------------
def process_replacement(
    session_id: str,
    cue_id: str,
    recording_path: str,
    sync_method: str = "align",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Process a replacement ADR recording and sync to original timecode.

    Args:
        session_id: ADR session ID.
        cue_id: ID of the target cue.
        recording_path: Path to the replacement recording WAV/MP3.
        sync_method: "align" (time-stretch to match duration) or
                     "crosscorr" (cross-correlation offset detection).
        output_dir: Directory for the output file.
        on_progress: Progress callback taking one int (percentage).

    Returns:
        dict with output_path, cue_id, sync_method, time_offset, quality.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    if not os.path.isfile(recording_path):
        raise ValueError(f"Recording file not found: {recording_path}")

    cue = None
    for c in session.cues:
        if c.cue_id == cue_id:
            cue = c
            break
    if cue is None:
        raise ValueError(f"ADR cue not found: {cue_id}")

    if on_progress:
        on_progress(10)

    target_duration = cue.duration()
    if target_duration <= 0:
        raise ValueError("Cue has zero or negative duration")

    out_dir = output_dir or ADR_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"adr_{session.session_id}_{cue.cue_id}_take{cue.takes_recorded + 1}.wav")

    ffmpeg = get_ffmpeg_path()

    # Measure recording duration
    rec_duration = _get_audio_duration(recording_path)
    if rec_duration <= 0:
        raise ValueError("Could not determine recording duration")

    if on_progress:
        on_progress(30)

    time_offset = 0.0
    quality = "good"

    if sync_method == "crosscorr" and session.source_path and os.path.isfile(session.source_path):
        # Cross-correlation: extract original segment, compute offset
        time_offset = _cross_correlate_offset(
            session.source_path, recording_path,
            cue.timecode_in, cue.timecode_out,
        )
        if on_progress:
            on_progress(60)

        # Apply offset and trim to cue duration
        pad_start = max(0.0, -time_offset)
        trim_start = max(0.0, time_offset)
        af_parts = []
        if trim_start > 0:
            af_parts.append(f"atrim=start={trim_start:.4f}")
        af_parts.append(f"apad=pad_dur={pad_start:.4f}")
        af_parts.append(f"atrim=duration={target_duration:.4f}")
        af_parts.append("aresample=48000")
        af_chain = ",".join(af_parts)

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", recording_path,
            "-af", af_chain,
            "-c:a", "pcm_s16le",
            out_path,
        ]
        run_ffmpeg(cmd, timeout=300)
    else:
        # Simple time-alignment: stretch/compress recording to match cue duration
        if on_progress:
            on_progress(50)

        tempo_ratio = rec_duration / target_duration if target_duration > 0 else 1.0
        tempo_ratio = max(0.5, min(2.0, tempo_ratio))

        af_parts = []
        if abs(tempo_ratio - 1.0) > 0.01:
            af_parts.append(f"atempo={tempo_ratio:.6f}")
        af_parts.append(f"atrim=duration={target_duration:.4f}")
        af_parts.append("aresample=48000")
        af_chain = ",".join(af_parts) if af_parts else "anull"

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", recording_path,
            "-af", af_chain,
            "-c:a", "pcm_s16le",
            out_path,
        ]
        run_ffmpeg(cmd, timeout=300)

        if abs(tempo_ratio - 1.0) > 0.15:
            quality = "fair"
        if abs(tempo_ratio - 1.0) > 0.30:
            quality = "poor"

    if on_progress:
        on_progress(90)

    # Update cue status
    cue.takes_recorded += 1
    cue.replacement_path = out_path
    cue.status = "recorded"
    cue.updated_at = time.time()
    _save_session(session)

    if on_progress:
        on_progress(100)

    logger.info("Processed ADR replacement for cue %s (take %d, quality: %s)",
                cue_id, cue.takes_recorded, quality)
    return {
        "output_path": out_path,
        "cue_id": cue_id,
        "character": cue.character_name,
        "take_number": cue.takes_recorded,
        "sync_method": sync_method,
        "time_offset": time_offset,
        "quality": quality,
        "original_duration": target_duration,
        "recording_duration": rec_duration,
    }


# ---------------------------------------------------------------------------
# TTS Placeholder Generation
# ---------------------------------------------------------------------------
def generate_tts_placeholder(
    session_id: str,
    cue_id: str,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a TTS placeholder for an ADR cue using FFmpeg sine synthesis.

    This creates a simple speech-cadence tone pattern that acts as a timing
    reference placeholder until live recording is available. If a real TTS
    engine is not available, we generate a tone sequence whose duration
    matches the cue.

    Returns:
        dict with output_path, cue_id, method, duration.
    """
    session = _load_session(session_id)
    if session is None:
        raise ValueError(f"ADR session not found: {session_id}")

    cue = None
    for c in session.cues:
        if c.cue_id == cue_id:
            cue = c
            break
    if cue is None:
        raise ValueError(f"ADR cue not found: {cue_id}")

    if on_progress:
        on_progress(10)

    target_dur = cue.duration()
    if target_dur <= 0:
        target_dur = 1.0

    out_dir = output_dir or ADR_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tts_{session.session_id}_{cue.cue_id}.wav")

    method = "tone_sequence"

    # Generate a tone sequence approximating speech cadence
    # ~4 syllables per second, alternating frequencies
    syllable_dur = 0.15
    gap_dur = 0.10
    word_count = max(1, len(cue.original_line.split()))
    syllable_count = max(1, int(word_count * 1.5))

    # Constrain total duration to match cue
    total_syllable_time = syllable_count * (syllable_dur + gap_dur)
    time_scale = target_dur / total_syllable_time if total_syllable_time > 0 else 1.0

    ffmpeg = get_ffmpeg_path()

    if on_progress:
        on_progress(40)

    # Use sine source with envelope for speech-like timing
    af = (
        f"sine=frequency=200:duration={target_dur:.3f},"
        f"tremolo=f={1.0 / max(0.1, syllable_dur * time_scale):.2f}:d=0.7,"
        "volume=0.3"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", af,
        "-t", str(target_dur),
        "-c:a", "pcm_s16le", "-ar", "48000",
        out_path,
    ]
    run_ffmpeg(cmd, timeout=120)

    if on_progress:
        on_progress(100)

    logger.info("Generated TTS placeholder for cue %s (%.1fs)", cue_id, target_dur)
    return {
        "output_path": out_path,
        "cue_id": cue_id,
        "character": cue.character_name,
        "method": method,
        "duration": target_dur,
        "line": cue.original_line,
    }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------
def _get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout.decode())
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _cross_correlate_offset(
    source_path: str,
    recording_path: str,
    tc_in: float,
    tc_out: float,
) -> float:
    """Estimate time offset between original and recording using energy correlation.

    Extracts the original segment and compares RMS energy envelopes with
    the recording to find the best alignment offset. This is a simplified
    approach using FFmpeg volumedetect on sliding windows.

    Returns offset in seconds (positive = recording is late).
    """
    ffmpeg = get_ffmpeg_path()
    duration = tc_out - tc_in
    if duration <= 0:
        return 0.0

    # Extract original segment to temp file
    tmp_orig = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="adr_orig_")
    tmp_orig.close()

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(tc_in), "-t", str(duration),
            "-i", source_path,
            "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_orig.name,
        ]
        run_ffmpeg(cmd, timeout=120)

        # Get RMS levels at a few sample points for both files
        orig_levels = _sample_rms_levels(tmp_orig.name, duration, samples=10)
        rec_levels = _sample_rms_levels(recording_path, duration, samples=10)

        if not orig_levels or not rec_levels:
            return 0.0

        # Simple cross-correlation of energy envelopes
        best_offset = 0
        best_corr = -1.0
        step = duration / len(orig_levels) if len(orig_levels) > 0 else 0.1

        for shift in range(-3, 4):
            corr = 0.0
            count = 0
            for j in range(len(orig_levels)):
                k = j + shift
                if 0 <= k < len(rec_levels):
                    corr += orig_levels[j] * rec_levels[k]
                    count += 1
            if count > 0:
                corr /= count
            if corr > best_corr:
                best_corr = corr
                best_offset = shift

        return best_offset * step
    except Exception as exc:
        logger.debug("Cross-correlation failed: %s", exc)
        return 0.0
    finally:
        try:
            os.unlink(tmp_orig.name)
        except OSError:
            pass


def _sample_rms_levels(filepath: str, duration: float, samples: int = 10) -> List[float]:
    """Sample RMS levels at evenly spaced points in an audio file."""
    ffmpeg = get_ffmpeg_path()
    step = duration / max(1, samples)
    levels = []

    for i in range(samples):
        t = i * step
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "info",
            "-ss", f"{t:.3f}", "-t", f"{step:.3f}",
            "-i", filepath,
            "-af", "volumedetect",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            stderr = result.stderr
            # Parse mean_volume from volumedetect output
            for line in stderr.split("\n"):
                if "mean_volume" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[-1].strip().replace("dB", "").strip()
                        try:
                            # Convert dB to linear
                            db = float(val)
                            levels.append(10 ** (db / 20.0))
                        except ValueError:
                            levels.append(0.0)
                    break
            else:
                levels.append(0.0)
        except Exception:
            levels.append(0.0)

    return levels
