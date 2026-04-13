"""
OpenCut Gaming Routes

Routes for gaming/streaming-focused features:
- Chat Replay Overlay (12.2)
- Auto Montage Builder (12.3)
- Multi-POV Sync (12.4)
- Multi-Track ISO Ingest (31.1)
- Instant Replay Builder (31.2)
- Stream Highlight Reel (31.4)
"""

import logging

from flask import Blueprint

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

gaming_bp = Blueprint("gaming", __name__)


# ---------------------------------------------------------------------------
# 12.2 — Chat Replay Overlay
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/chat-replay", methods=["POST"])
@require_csrf
@async_job("chat_replay")
def chat_replay(job_id, filepath, data):
    """Parse chat log and render scrolling chat overlay on video."""
    from opencut.core.chat_replay import (
        parse_twitch_chat,
        parse_youtube_chat,
        render_chat_overlay,
    )

    chat_path = data.get("chat_path", "").strip()
    if not chat_path:
        raise ValueError("No chat log path provided (chat_path)")
    chat_path = validate_filepath(chat_path)

    chat_format = data.get("chat_format", "twitch").strip().lower()
    position = data.get("position", "right").strip()
    font_size = safe_int(data.get("font_size", 24), 24, min_val=10, max_val=120)
    opacity = safe_float(data.get("opacity", 0.85), 0.85, min_val=0.0, max_val=1.0)
    max_visible = safe_int(data.get("max_visible", 15), 15, min_val=1, max_val=50)
    style = data.get("style", "scrolling").strip()
    output_path = data.get("output_path")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, f"Parsing {chat_format} chat log...")

    if chat_format == "youtube":
        messages = parse_youtube_chat(chat_path)
    else:
        messages = parse_twitch_chat(chat_path)

    if not messages:
        raise ValueError("No chat messages found in log file")

    _progress(20, f"Parsed {len(messages)} messages, rendering overlay...")

    result = render_chat_overlay(
        video_path=filepath,
        messages=messages,
        output_path_str=output_path,
        position=position,
        font_size=font_size,
        opacity=opacity,
        max_visible=max_visible,
        style=style,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "message_count": result.message_count,
        "duration": result.duration,
        "style": result.style,
    }


# ---------------------------------------------------------------------------
# 12.3 — Auto Montage Builder
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/auto-montage/score", methods=["POST"])
@require_csrf
@async_job("montage_score", filepath_required=False)
def auto_montage_score(job_id, filepath, data):
    """Score clips by excitement (audio energy + motion)."""
    from opencut.core.auto_montage import score_clips

    clip_paths = data.get("clip_paths", [])
    if not clip_paths:
        raise ValueError("No clip paths provided (clip_paths)")
    clip_paths = [validate_filepath(p.strip()) for p in clip_paths if p.strip()]

    audio_weight = safe_float(data.get("audio_weight", 0.5), 0.5, min_val=0.0, max_val=1.0)
    motion_weight = safe_float(data.get("motion_weight", 0.5), 0.5, min_val=0.0, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    scores = score_clips(
        clip_paths=clip_paths,
        audio_weight=audio_weight,
        motion_weight=motion_weight,
        on_progress=_progress,
    )

    return {
        "scores": [
            {
                "clip_path": s.clip_path,
                "audio_energy": s.audio_energy,
                "motion_score": s.motion_score,
                "composite": s.composite,
                "duration": s.duration,
            }
            for s in scores
        ],
        "count": len(scores),
    }


@gaming_bp.route("/gaming/auto-montage/assemble", methods=["POST"])
@require_csrf
@async_job("montage_assemble", filepath_required=False)
def auto_montage_assemble(job_id, filepath, data):
    """Assemble top clips into a beat-synced montage with music."""
    from opencut.core.auto_montage import assemble_montage, score_clips, select_top_clips

    clip_paths = data.get("clip_paths", [])
    if not clip_paths:
        raise ValueError("No clip paths provided (clip_paths)")
    clip_paths = [validate_filepath(p.strip()) for p in clip_paths if p.strip()]

    music_path = data.get("music_path", "").strip()
    if not music_path:
        raise ValueError("No music path provided (music_path)")
    music_path = validate_filepath(music_path)

    count = safe_int(data.get("count", 10), 10, min_val=1, max_val=100)
    transition = data.get("transition", "cut").strip()
    output_path = data.get("output_path")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Scoring clips...")
    scores = score_clips(clip_paths=clip_paths, on_progress=_progress)
    top = select_top_clips(scores, count=count)

    _progress(50, f"Assembling montage from {len(top)} clips...")
    result = assemble_montage(
        clips=top,
        music_path=music_path,
        output_path_str=output_path,
        transition=transition,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "clip_count": result.clip_count,
        "total_duration": result.total_duration,
    }


# ---------------------------------------------------------------------------
# 12.4 — Multi-POV Sync
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/multi-pov/sync", methods=["POST"])
@require_csrf
@async_job("multi_pov_sync", filepath_required=False)
def multi_pov_sync(job_id, filepath, data):
    """Sync multiple player recordings via audio cross-correlation."""
    from opencut.core.multi_pov import sync_pov_recordings

    file_paths = data.get("file_paths", [])
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 file paths (file_paths)")
    file_paths = [validate_filepath(p.strip()) for p in file_paths if p.strip()]

    reference_index = safe_int(data.get("reference_index", 0), 0, min_val=0)
    max_offset = safe_float(data.get("max_offset", 30.0), 30.0, min_val=1.0, max_val=300.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    synced = sync_pov_recordings(
        file_paths=file_paths,
        reference_index=reference_index,
        max_offset=max_offset,
        on_progress=_progress,
    )

    return {
        "synced_clips": [
            {
                "file_path": c.file_path,
                "offset": c.offset,
                "duration": c.duration,
                "label": c.label,
                "confidence": c.confidence,
            }
            for c in synced
        ],
        "count": len(synced),
    }


@gaming_bp.route("/gaming/multi-pov/export-xml", methods=["POST"])
@require_csrf
@async_job("multi_pov_xml", filepath_required=False)
def multi_pov_export_xml(job_id, filepath, data):
    """Generate multicam FCP XML from synced POV data."""
    from opencut.core.multi_pov import SyncedClip, generate_multicam_xml

    synced_data = data.get("synced_clips", [])
    if not synced_data:
        raise ValueError("No synced clip data provided (synced_clips)")

    clips = []
    for item in synced_data:
        fp = validate_filepath(item.get("file_path", "").strip())
        clips.append(SyncedClip(
            file_path=fp,
            offset=float(item.get("offset", 0)),
            duration=float(item.get("duration", 0)),
            label=item.get("label", ""),
            confidence=float(item.get("confidence", 0)),
        ))

    output_path = data.get("output_path")
    sequence_name = data.get("sequence_name", "OpenCut Multi-POV")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(50, "Generating multicam XML...")
    result = generate_multicam_xml(
        synced_clips=clips,
        output_path=output_path,
        sequence_name=sequence_name,
    )

    return {
        "xml_length": len(result.get("xml", "")),
        "output": result.get("output"),
        "clip_count": result.get("clip_count", 0),
        "duration": result.get("duration", 0),
    }


# ---------------------------------------------------------------------------
# 12.4 — Shared Audio Detection
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/multi-pov/detect-audio", methods=["POST"])
@require_csrf
@async_job("detect_shared_audio", filepath_required=False)
def multi_pov_detect_audio(job_id, filepath, data):
    """Detect shared audio content between recordings."""
    from opencut.core.multi_pov import detect_shared_audio

    file_paths = data.get("file_paths", [])
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 file paths (file_paths)")
    file_paths = [validate_filepath(p.strip()) for p in file_paths if p.strip()]

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Detecting shared audio...")
    result = detect_shared_audio(file_paths)
    _progress(100, "Detection complete")

    return result


# ---------------------------------------------------------------------------
# 31.1 — Multi-Track ISO Ingest
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/iso-ingest/detect", methods=["POST"])
@require_csrf
@async_job("iso_detect")
def iso_ingest_detect(job_id, filepath, data):
    """Detect ISO recording properties (timecode, codec, etc.)."""
    from opencut.core.iso_ingest import detect_iso_tracks

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(20, "Detecting ISO track properties...")
    track = detect_iso_tracks(filepath)
    _progress(100, "Detection complete")

    return {
        "file_path": track.file_path,
        "timecode": track.timecode,
        "timecode_seconds": track.timecode_seconds,
        "duration": track.duration,
        "width": track.width,
        "height": track.height,
        "fps": track.fps,
        "has_audio": track.has_audio,
        "codec": track.codec,
        "label": track.label,
    }


@gaming_bp.route("/gaming/iso-ingest/sync", methods=["POST"])
@require_csrf
@async_job("iso_sync", filepath_required=False)
def iso_ingest_sync(job_id, filepath, data):
    """Sync multiple ISO recordings via timecode or audio."""
    from opencut.core.iso_ingest import sync_iso_recordings

    file_paths = data.get("file_paths", [])
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 ISO recordings (file_paths)")
    file_paths = [validate_filepath(p.strip()) for p in file_paths if p.strip()]

    method = data.get("method", "auto").strip()
    reference_index = safe_int(data.get("reference_index", 0), 0, min_val=0)
    max_offset = safe_float(data.get("max_offset", 30.0), 30.0, min_val=1.0, max_val=300.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    synced = sync_iso_recordings(
        file_paths=file_paths,
        method=method,
        reference_index=reference_index,
        max_offset=max_offset,
        on_progress=_progress,
    )

    return {
        "synced_tracks": [
            {
                "file_path": st.track.file_path,
                "label": st.track.label,
                "offset": st.offset,
                "sync_method": st.sync_method,
                "confidence": st.confidence,
                "timecode": st.track.timecode,
                "duration": st.track.duration,
            }
            for st in synced
        ],
        "count": len(synced),
        "method": synced[0].sync_method if synced else "",
    }


@gaming_bp.route("/gaming/iso-ingest/timeline", methods=["POST"])
@require_csrf
@async_job("iso_timeline", filepath_required=False)
def iso_ingest_timeline(job_id, filepath, data):
    """Generate multicam timeline from synced ISO recordings."""
    from opencut.core.iso_ingest import ISOTrack, SyncedISOTrack, generate_multicam_timeline

    synced_data = data.get("synced_tracks", [])
    if not synced_data:
        raise ValueError("No synced track data provided (synced_tracks)")

    tracks = []
    for item in synced_data:
        fp = validate_filepath(item.get("file_path", "").strip())
        track = ISOTrack(
            file_path=fp,
            duration=float(item.get("duration", 0)),
            label=item.get("label", ""),
            timecode=item.get("timecode", ""),
        )
        tracks.append(SyncedISOTrack(
            track=track,
            offset=float(item.get("offset", 0)),
            sync_method=item.get("sync_method", ""),
            confidence=float(item.get("confidence", 0)),
        ))

    output_path = data.get("output_path")
    sequence_name = data.get("sequence_name", "OpenCut ISO Multicam")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_multicam_timeline(
        synced_tracks=tracks,
        output_path=output_path,
        sequence_name=sequence_name,
        on_progress=_progress,
    )

    return {
        "timeline_path": result.timeline_path,
        "total_duration": result.total_duration,
        "sync_method": result.sync_method,
        "track_count": len(result.tracks),
    }


# ---------------------------------------------------------------------------
# 31.2 — Instant Replay Builder
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/instant-replay", methods=["POST"])
@require_csrf
@async_job("instant_replay")
def instant_replay(job_id, filepath, data):
    """Generate an instant replay for a specific moment."""
    from opencut.core.instant_replay import ReplayConfig, create_replay

    timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
    if timestamp <= 0:
        raise ValueError("A valid timestamp is required")

    config = ReplayConfig(
        pre_roll=safe_float(data.get("pre_roll", 3.0), 3.0, min_val=0.5, max_val=30.0),
        post_roll=safe_float(data.get("post_roll", 2.0), 2.0, min_val=0.5, max_val=30.0),
        slow_factor=safe_float(data.get("slow_factor", 0.5), 0.5, min_val=0.1, max_val=1.0),
        overlay_text=data.get("overlay_text", "REPLAY"),
        font_size=safe_int(data.get("font_size", 72), 72, min_val=20, max_val=200),
        transition=data.get("transition", "flash").strip(),
        include_original=data.get("include_original", True),
    )

    output_path = data.get("output_path")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_replay(
        video_path=filepath,
        timestamp=timestamp,
        output_path_str=output_path,
        config=config,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "timestamp": result.timestamp,
        "duration": result.duration,
        "slow_duration": result.slow_duration,
    }


@gaming_bp.route("/gaming/instant-replay/batch", methods=["POST"])
@require_csrf
@async_job("instant_replay_batch")
def instant_replay_batch(job_id, filepath, data):
    """Generate instant replays for multiple timestamps."""
    from opencut.core.instant_replay import ReplayConfig, batch_replays

    timestamps = data.get("timestamps", [])
    if not timestamps:
        raise ValueError("No timestamps provided")
    timestamps = [safe_float(t, 0, min_val=0.0) for t in timestamps]
    timestamps = [t for t in timestamps if t > 0]

    if not timestamps:
        raise ValueError("No valid timestamps provided")

    config = ReplayConfig(
        pre_roll=safe_float(data.get("pre_roll", 3.0), 3.0, min_val=0.5, max_val=30.0),
        post_roll=safe_float(data.get("post_roll", 2.0), 2.0, min_val=0.5, max_val=30.0),
        slow_factor=safe_float(data.get("slow_factor", 0.5), 0.5, min_val=0.1, max_val=1.0),
        overlay_text=data.get("overlay_text", "REPLAY"),
        transition=data.get("transition", "flash").strip(),
        include_original=data.get("include_original", True),
    )

    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_replays(
        video_path=filepath,
        timestamps=timestamps,
        output_dir=output_dir or None,
        config=config,
        on_progress=_progress,
    )

    return {
        "output_dir": result.output_dir,
        "total_count": result.total_count,
        "success_count": result.success_count,
        "replays": [
            {
                "output_path": r.output_path,
                "timestamp": r.timestamp,
                "duration": r.duration,
            }
            for r in result.replays
        ],
    }


# ---------------------------------------------------------------------------
# 31.4 — Stream Highlight Reel
# ---------------------------------------------------------------------------
@gaming_bp.route("/gaming/stream-highlights/score", methods=["POST"])
@require_csrf
@async_job("highlight_score")
def stream_highlights_score(job_id, filepath, data):
    """Score stream segments by audio, motion, and keywords."""
    from opencut.core.stream_highlights import score_stream_segments

    segment_duration = safe_float(data.get("segment_duration", 10.0), 10.0, min_val=3.0, max_val=60.0)
    audio_weight = safe_float(data.get("audio_weight", 0.4), 0.4, min_val=0.0, max_val=1.0)
    motion_weight = safe_float(data.get("motion_weight", 0.3), 0.3, min_val=0.0, max_val=1.0)
    keyword_weight = safe_float(data.get("keyword_weight", 0.3), 0.3, min_val=0.0, max_val=1.0)
    keywords = data.get("keywords")
    transcript = data.get("transcript_segments")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    segments = score_stream_segments(
        video_path=filepath,
        segment_duration=segment_duration,
        audio_weight=audio_weight,
        motion_weight=motion_weight,
        keyword_weight=keyword_weight,
        transcript_segments=transcript,
        keywords=keywords,
        on_progress=_progress,
    )

    return {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "audio_score": s.audio_score,
                "motion_score": s.motion_score,
                "keyword_score": s.keyword_score,
                "composite": s.composite,
                "keywords_found": s.keywords_found,
            }
            for s in segments[:50]  # limit response size
        ],
        "total_segments": len(segments),
    }


@gaming_bp.route("/gaming/stream-highlights/extract", methods=["POST"])
@require_csrf
@async_job("highlight_extract")
def stream_highlights_extract(job_id, filepath, data):
    """Extract top highlight segments as individual clips."""
    from opencut.core.stream_highlights import ScoredSegment, extract_highlights

    segments_data = data.get("segments", [])
    if not segments_data:
        raise ValueError("No segment data provided (segments)")

    segments = [
        ScoredSegment(
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            composite=float(s.get("composite", s.get("score", 0))),
        )
        for s in segments_data
    ]

    max_clips = safe_int(data.get("max_clips", 10), 10, min_val=1, max_val=50)
    min_score = safe_float(data.get("min_score", 0.3), 0.3, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    clips = extract_highlights(
        video_path=filepath,
        segments=segments,
        output_dir=output_dir or None,
        max_clips=max_clips,
        min_score=min_score,
        on_progress=_progress,
    )

    return {
        "clips": [
            {
                "file_path": c.file_path,
                "start": c.start,
                "end": c.end,
                "score": c.score,
            }
            for c in clips
        ],
        "count": len(clips),
    }


@gaming_bp.route("/gaming/stream-highlights/assemble", methods=["POST"])
@require_csrf
@async_job("highlight_assemble", filepath_required=False)
def stream_highlights_assemble(job_id, filepath, data):
    """Assemble highlight clips into a highlight reel."""
    from opencut.core.stream_highlights import HighlightClip, assemble_highlight_reel

    clips_data = data.get("clips", [])
    if not clips_data:
        raise ValueError("No clip data provided (clips)")

    clips = []
    for c in clips_data:
        fp = validate_filepath(c.get("file_path", "").strip())
        clips.append(HighlightClip(
            file_path=fp,
            start=float(c.get("start", 0)),
            end=float(c.get("end", 0)),
            score=float(c.get("score", 0)),
            index=int(c.get("index", 0)),
        ))

    transition = data.get("transition", "crossfade").strip()
    transition_duration = safe_float(data.get("transition_duration", 0.5), 0.5, min_val=0.1, max_val=3.0)
    intro_text = data.get("intro_text")
    output_path = data.get("output_path")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = assemble_highlight_reel(
        clips=clips,
        output_path_str=output_path,
        transition=transition,
        transition_duration=transition_duration,
        intro_text=intro_text,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "clip_count": result.clip_count,
        "total_duration": result.total_duration,
    }
