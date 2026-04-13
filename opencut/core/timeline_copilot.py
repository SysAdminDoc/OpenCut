"""
Multimodal Timeline Copilot (21.1)

Chat with timeline using natural language, backed by
video + audio + transcript understanding. Execute edit actions
from NL commands.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CopilotAction:
    """An action the copilot suggests or executes."""
    action_type: str  # "cut", "trim", "delete", "insert", "speed", "volume",
                      # "caption", "transition", "reorder", "info", "find"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    executed: bool = False
    result: Optional[Dict] = None


@dataclass
class TimelineContext:
    """Context about the current timeline state for copilot queries."""
    video_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    transcript: List[dict] = field(default_factory=list)
    scenes: List[dict] = field(default_factory=list)
    markers: List[dict] = field(default_factory=list)
    current_position: float = 0.0


# ---------------------------------------------------------------------------
# NL Pattern Matching
# ---------------------------------------------------------------------------
_PATTERNS = [
    # Trim/Cut
    (r"(?:trim|cut)\s+(?:the\s+)?(?:first|beginning)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)",
     "trim", lambda m: {"action": "trim_start", "seconds": float(m.group(1))}),
    (r"(?:trim|cut)\s+(?:the\s+)?(?:last|end(?:ing)?)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)",
     "trim", lambda m: {"action": "trim_end", "seconds": float(m.group(1))}),
    (r"(?:cut|remove|delete)\s+(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec)?\s*(?:to|through|-)\s*(\d+(?:\.\d+)?)\s*(?:s|sec)?",
     "cut", lambda m: {"start": float(m.group(1)), "end": float(m.group(2))}),

    # Speed
    (r"(?:speed\s+up|faster)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*x?",
     "speed", lambda m: {"factor": float(m.group(1))}),
    (r"(?:slow\s+down|slower)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*x?",
     "speed", lambda m: {"factor": 1.0 / max(0.1, float(m.group(1)))}),
    (r"(?:set\s+)?speed\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*x",
     "speed", lambda m: {"factor": float(m.group(1))}),

    # Volume
    (r"(?:set\s+)?volume\s+(?:to\s+)?(\d+)\s*%",
     "volume", lambda m: {"level": int(m.group(1)) / 100.0}),
    (r"(?:mute|silence)\s+(?:the\s+)?(?:audio|sound|video)",
     "volume", lambda m: {"level": 0.0}),

    # Info queries
    (r"(?:how\s+long|what(?:'s| is)\s+(?:the\s+)?duration|length)",
     "info", lambda m: {"query": "duration"}),
    (r"(?:what(?:'s| is)\s+(?:the\s+)?resolution|dimensions|size)",
     "info", lambda m: {"query": "resolution"}),
    (r"(?:how\s+many\s+)?(?:scenes?|segments?|cuts?)",
     "info", lambda m: {"query": "scenes"}),

    # Find/Search
    (r"(?:find|search|locate|where)\s+(?:is|does|do|the\s+)?\s*[\"'](.+?)[\"']",
     "find", lambda m: {"search_text": m.group(1)}),
    (r"(?:find|search|locate)\s+(?:where\s+)?(?:someone\s+)?(?:says?|mentions?)\s+[\"'](.+?)[\"']",
     "find", lambda m: {"search_text": m.group(1)}),

    # Caption
    (r"(?:add|insert)\s+(?:a\s+)?(?:caption|subtitle|text)\s+[\"'](.+?)[\"']",
     "caption", lambda m: {"text": m.group(1)}),
]


def _match_intent(query: str) -> Optional[CopilotAction]:
    """Match a natural language query to an action using patterns."""
    query_lower = query.lower().strip()

    for pattern, action_type, param_fn in _PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            params = param_fn(match)
            return CopilotAction(
                action_type=action_type,
                description=query,
                parameters=params,
                confidence=0.85,
            )

    return None


# ---------------------------------------------------------------------------
# Build Context
# ---------------------------------------------------------------------------
def build_timeline_context(
    video_path: str,
    transcript: Optional[List[dict]] = None,
    scenes: Optional[List[dict]] = None,
    on_progress: Optional[Callable] = None,
) -> TimelineContext:
    """Build context about the current timeline for copilot queries.

    Args:
        video_path: Path to the video file.
        transcript: Optional transcript segments.
        scenes: Optional scene detection results.
        on_progress: Optional callback(pct, msg).

    Returns:
        TimelineContext object.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Analyzing video...")

    info = get_video_info(video_path)

    ctx = TimelineContext(
        video_path=video_path,
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        transcript=transcript or [],
        scenes=scenes or [],
    )

    if on_progress:
        on_progress(100, "Context built")

    return ctx


# ---------------------------------------------------------------------------
# Process Query
# ---------------------------------------------------------------------------
def process_copilot_query(
    query: str,
    context: Optional[TimelineContext] = None,
    on_progress: Optional[Callable] = None,
) -> CopilotAction:
    """Process a natural language query about the timeline.

    Args:
        query: Natural language query/command.
        context: Optional TimelineContext for richer responses.
        on_progress: Optional callback(pct, msg).

    Returns:
        CopilotAction with the interpreted action.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if on_progress:
        on_progress(10, "Analyzing query...")

    # Try pattern matching
    action = _match_intent(query)

    if action:
        # Enrich with context if available
        if context and action.action_type == "info":
            action = _resolve_info_query(action, context)
        elif context and action.action_type == "find":
            action = _resolve_find_query(action, context)

        if on_progress:
            on_progress(100, f"Matched action: {action.action_type}")
        return action

    # Fallback: try to provide a helpful response
    action = CopilotAction(
        action_type="info",
        description=query,
        parameters={"query": "general", "original_query": query},
        confidence=0.3,
    )

    if context:
        action.result = {
            "message": f"I understood your query but couldn't match it to a specific action. "
                       f"Video is {context.duration:.1f}s, {context.width}x{context.height}. "
                       f"Try commands like 'trim first 5 seconds' or 'find where someone says hello'.",
            "duration": context.duration,
            "resolution": f"{context.width}x{context.height}",
        }
    else:
        action.result = {
            "message": "Could not match query to an action. "
                       "Try: 'trim first 5s', 'cut 10s to 20s', 'speed up 2x', 'mute audio'."
        }

    if on_progress:
        on_progress(100, "Query processed (no specific action matched)")

    return action


def _resolve_info_query(action: CopilotAction, ctx: TimelineContext) -> CopilotAction:
    """Resolve an info query using timeline context."""
    query_type = action.parameters.get("query", "")

    if query_type == "duration":
        mins = int(ctx.duration // 60)
        secs = ctx.duration % 60
        action.result = {
            "duration": ctx.duration,
            "message": f"Video duration: {mins}m {secs:.1f}s ({ctx.duration:.1f} seconds)",
        }
    elif query_type == "resolution":
        action.result = {
            "width": ctx.width,
            "height": ctx.height,
            "fps": ctx.fps,
            "message": f"Resolution: {ctx.width}x{ctx.height} @ {ctx.fps:.1f} fps",
        }
    elif query_type == "scenes":
        count = len(ctx.scenes)
        action.result = {
            "scene_count": count,
            "message": f"Found {count} scene(s) in the timeline",
            "scenes": ctx.scenes[:20],
        }

    action.executed = True
    action.confidence = 1.0
    return action


def _resolve_find_query(action: CopilotAction, ctx: TimelineContext) -> CopilotAction:
    """Resolve a find/search query against the transcript."""
    search_text = action.parameters.get("search_text", "").lower()
    if not search_text:
        action.result = {"matches": [], "message": "No search text provided"}
        return action

    matches = []
    for seg in ctx.transcript:
        seg_text = seg.get("text", "").lower()
        if search_text in seg_text:
            matches.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", ""),
            })

    action.result = {
        "matches": matches[:20],
        "total_matches": len(matches),
        "message": (
            f"Found {len(matches)} match(es) for '{search_text}'"
            if matches else
            f"No matches found for '{search_text}' in transcript"
        ),
    }
    action.executed = True
    return action


# ---------------------------------------------------------------------------
# Execute Action
# ---------------------------------------------------------------------------
def execute_copilot_action(
    action: CopilotAction,
    video_path: str,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Execute a copilot action on a video.

    Args:
        action: CopilotAction to execute.
        video_path: Source video file.
        output_dir: Output directory.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with result of the action.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, f"Executing {action.action_type}...")

    out_dir = output_dir or os.path.dirname(os.path.abspath(video_path))
    info = get_video_info(video_path)

    if action.action_type == "trim":
        result = _exec_trim(action, video_path, info, out_dir, on_progress)
    elif action.action_type == "cut":
        result = _exec_cut(action, video_path, info, out_dir, on_progress)
    elif action.action_type == "speed":
        result = _exec_speed(action, video_path, info, out_dir, on_progress)
    elif action.action_type == "volume":
        result = _exec_volume(action, video_path, info, out_dir, on_progress)
    elif action.action_type == "info":
        result = action.result or {"message": "No additional info available"}
    elif action.action_type == "find":
        result = action.result or {"message": "Search requires transcript context"}
    else:
        result = {"message": f"Action type '{action.action_type}' not yet supported for execution"}

    action.executed = True
    action.result = result

    if on_progress:
        on_progress(100, f"Action {action.action_type} complete")

    return result


def _exec_trim(action, video_path, info, out_dir, on_progress):
    """Execute a trim action."""
    params = action.parameters
    trim_action = params.get("action", "")
    seconds = float(params.get("seconds", 0))
    out = output_path(video_path, "trimmed", out_dir)

    if trim_action == "trim_start":
        cmd = (FFmpegCmd()
               .input(video_path, ss=seconds)
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out)
               .build())
    elif trim_action == "trim_end":
        end_time = max(0, info["duration"] - seconds)
        cmd = (FFmpegCmd()
               .input(video_path)
               .seek(end=end_time)
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out)
               .build())
    else:
        return {"error": f"Unknown trim action: {trim_action}"}

    if on_progress:
        on_progress(50, "Trimming...")

    run_ffmpeg(cmd)
    return {"output_path": out, "action": trim_action, "seconds_removed": seconds}


def _exec_cut(action, video_path, info, out_dir, on_progress):
    """Execute a cut (remove segment) action."""
    params = action.parameters
    cut_start = float(params.get("start", 0))
    cut_end = float(params.get("end", 0))
    out = output_path(video_path, "cut", out_dir)

    if cut_start >= cut_end:
        return {"error": "Cut start must be before end"}

    # Build filter to remove the segment
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="opencut_cut_")
    segments = []

    try:
        # Part before the cut
        if cut_start > 0.1:
            seg1 = os.path.join(tmp_dir, "before.mp4")
            cmd = (FFmpegCmd()
                   .input(video_path)
                   .seek(end=cut_start)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .output(seg1)
                   .build())
            run_ffmpeg(cmd)
            segments.append(seg1)

        # Part after the cut
        if cut_end < info["duration"] - 0.1:
            seg2 = os.path.join(tmp_dir, "after.mp4")
            cmd = (FFmpegCmd()
                   .input(video_path, ss=cut_end)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .output(seg2)
                   .build())
            run_ffmpeg(cmd)
            segments.append(seg2)

        if not segments:
            return {"error": "Cut would remove the entire video"}

        if on_progress:
            on_progress(60, "Joining segments...")

        # Concat
        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg in segments:
                safe = seg.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_path)
               .copy_streams()
               .faststart()
               .output(out)
               .build())
        run_ffmpeg(cmd)

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "output_path": out,
        "removed_start": cut_start,
        "removed_end": cut_end,
        "removed_duration": cut_end - cut_start,
    }


def _exec_speed(action, video_path, info, out_dir, on_progress):
    """Execute a speed change action."""
    factor = float(action.parameters.get("factor", 1.0))
    if factor <= 0 or factor > 100:
        return {"error": f"Invalid speed factor: {factor}"}

    out = output_path(video_path, f"speed_{factor:.1f}x", out_dir)
    atempo = factor
    # atempo filter range is 0.5-100.0; chain for extreme values
    atempo_filters = []
    remaining = atempo
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining *= 2.0
    atempo_filters.append(f"atempo={remaining:.4f}")
    atempo_chain = ",".join(atempo_filters)

    setpts = f"setpts={1.0 / factor:.4f}*PTS"

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(setpts)
           .audio_filter(atempo_chain)
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .output(out)
           .build())

    if on_progress:
        on_progress(50, f"Applying {factor:.1f}x speed...")

    run_ffmpeg(cmd)
    return {
        "output_path": out,
        "speed_factor": factor,
        "new_duration": info["duration"] / factor,
    }


def _exec_volume(action, video_path, info, out_dir, on_progress):
    """Execute a volume change action."""
    level = float(action.parameters.get("level", 1.0))
    out = output_path(video_path, f"vol_{int(level * 100)}pct", out_dir)

    if level == 0:
        cmd = (FFmpegCmd()
               .input(video_path)
               .video_codec("copy")
               .audio_filter("volume=0")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out)
               .build())
    else:
        cmd = (FFmpegCmd()
               .input(video_path)
               .video_codec("copy")
               .audio_filter(f"volume={level:.2f}")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out)
               .build())

    if on_progress:
        on_progress(50, f"Adjusting volume to {int(level * 100)}%...")

    run_ffmpeg(cmd)
    return {"output_path": out, "volume_level": level}
