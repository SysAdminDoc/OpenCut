"""
OpenCut Clip Narrative Builder

AI-driven clip chaining into cohesive narrative arcs.
Given multiple video clips, analyzes content and orders them for storytelling:
- Analyze each clip's content (transcript, visual type, mood)
- Score narrative arc potential
- Order clips for story flow (hook -> conflict -> resolution)
- Suggest transitions between clips
- Generate assembly cut list

Supports multiple narrative styles: documentary, vlog, commercial, montage, etc.
"""

import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Narrative styles
# ---------------------------------------------------------------------------
NARRATIVE_STYLES: Dict[str, Dict] = {
    "documentary": {
        "description": "Classic documentary arc: establish context, explore, conclude",
        "structure": ["establishing", "context", "rising_action", "climax", "resolution", "conclusion"],
        "pace": "moderate",
        "preferred_transitions": ["crossfade", "dip_to_black", "cut"],
        "max_clip_duration": 60.0,
    },
    "vlog": {
        "description": "Casual vlog style: hook, content blocks, call to action",
        "structure": ["hook", "intro", "content", "content", "content", "cta"],
        "pace": "fast",
        "preferred_transitions": ["jump_cut", "swipe", "zoom"],
        "max_clip_duration": 30.0,
    },
    "commercial": {
        "description": "Short-form ad: attention grab, problem, solution, CTA",
        "structure": ["attention", "problem", "solution", "proof", "cta"],
        "pace": "fast",
        "preferred_transitions": ["cut", "whip_pan", "zoom"],
        "max_clip_duration": 15.0,
    },
    "montage": {
        "description": "Music-driven montage: rhythm-matched visual sequence",
        "structure": ["opener", "build", "build", "peak", "cool_down", "closer"],
        "pace": "variable",
        "preferred_transitions": ["beat_cut", "crossfade", "match_cut"],
        "max_clip_duration": 10.0,
    },
    "tutorial": {
        "description": "Educational flow: overview, step-by-step, recap",
        "structure": ["overview", "step", "step", "step", "recap", "outro"],
        "pace": "slow",
        "preferred_transitions": ["crossfade", "cut", "slide"],
        "max_clip_duration": 120.0,
    },
    "story": {
        "description": "Classic three-act story arc",
        "structure": ["setup", "rising_action", "rising_action", "climax", "falling_action", "resolution"],
        "pace": "moderate",
        "preferred_transitions": ["crossfade", "dip_to_black", "dissolve"],
        "max_clip_duration": 90.0,
    },
    "highlight_reel": {
        "description": "Best-of compilation: strongest moments first",
        "structure": ["best", "great", "great", "good", "good", "closer"],
        "pace": "fast",
        "preferred_transitions": ["cut", "flash", "zoom"],
        "max_clip_duration": 20.0,
    },
    "interview": {
        "description": "Interview format: question-answer with B-roll",
        "structure": ["intro", "question", "answer", "broll", "question", "answer", "outro"],
        "pace": "slow",
        "preferred_transitions": ["cut", "crossfade", "j_cut"],
        "max_clip_duration": 180.0,
    },
}

# Mood keywords and their narrative role affinities
_MOOD_ROLE_MAP = {
    "exciting": ["hook", "attention", "climax", "peak", "best"],
    "calm": ["establishing", "context", "resolution", "cool_down", "outro"],
    "dramatic": ["rising_action", "climax", "conflict", "problem"],
    "happy": ["intro", "content", "solution", "proof", "great"],
    "sad": ["falling_action", "problem", "context"],
    "neutral": ["context", "step", "content", "answer", "good"],
    "energetic": ["hook", "build", "peak", "best", "attention"],
    "tense": ["rising_action", "climax", "conflict"],
    "peaceful": ["establishing", "resolution", "conclusion", "closer"],
    "informative": ["overview", "step", "context", "answer"],
}

# Transition definitions with FFmpeg filter parameters
TRANSITIONS = {
    "cut": {"filter": None, "duration": 0.0, "description": "Hard cut"},
    "crossfade": {"filter": "xfade=transition=fade:duration={d}:offset={o}", "duration": 1.0,
                  "description": "Cross-dissolve between clips"},
    "dip_to_black": {"filter": "xfade=transition=fadeblack:duration={d}:offset={o}", "duration": 1.5,
                     "description": "Fade to black, then in"},
    "dissolve": {"filter": "xfade=transition=dissolve:duration={d}:offset={o}", "duration": 1.0,
                 "description": "Soft dissolve"},
    "wipe": {"filter": "xfade=transition=wipeleft:duration={d}:offset={o}", "duration": 0.8,
             "description": "Horizontal wipe"},
    "slide": {"filter": "xfade=transition=slideleft:duration={d}:offset={o}", "duration": 0.6,
              "description": "Slide transition"},
    "zoom": {"filter": "xfade=transition=zoomin:duration={d}:offset={o}", "duration": 0.5,
             "description": "Zoom transition"},
    "jump_cut": {"filter": None, "duration": 0.0, "description": "Jump cut (no transition)"},
    "flash": {"filter": "xfade=transition=fadewhite:duration={d}:offset={o}", "duration": 0.3,
              "description": "Flash white transition"},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClipInfo:
    """Analysis results for a single clip."""
    index: int = 0
    path: str = ""
    filename: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True
    has_speech: bool = False
    transcript: str = ""
    mood: str = "neutral"
    visual_type: str = "general"  # "talking_head", "broll", "action", "static", "general"
    energy_level: float = 0.5     # 0..1
    brightness: float = 0.5       # 0..1
    narrative_role: str = ""      # assigned role in the narrative
    role_score: float = 0.0       # how well this clip fits its role
    suggested_start: float = 0.0  # trim start
    suggested_end: float = 0.0    # trim end (0 = full clip)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransitionSuggestion:
    """Suggested transition between two clips."""
    from_clip: int = 0
    to_clip: int = 0
    transition_type: str = "cut"
    duration: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NarrativeResult:
    """Complete narrative assembly plan."""
    style: str = "documentary"
    clip_count: int = 0
    total_duration: float = 0.0
    assembly_order: List[int] = field(default_factory=list)  # clip indices in narrative order
    clips: List[Dict] = field(default_factory=list)
    transitions: List[Dict] = field(default_factory=list)
    structure_labels: List[str] = field(default_factory=list)
    narrative_score: float = 0.0   # how well the clips fit the narrative arc (0..100)
    assembly_edl: str = ""         # text-based edit decision list
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Clip analysis helpers
# ---------------------------------------------------------------------------
def _analyze_clip_audio(clip_path: str) -> Tuple[float, bool]:
    """Measure audio energy and detect speech in a clip.

    Returns (energy_0_1, has_speech).
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", clip_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        stderr = result.stderr.decode(errors="replace")

        # Parse integrated loudness
        match = re.search(r"I:\s*(-?[0-9.]+)\s*LUFS", stderr)
        lufs = float(match.group(1)) if match else -70.0

        # Normalize LUFS to 0..1 (-40 = 0, -10 = 1)
        energy = max(0, min(1.0, (lufs + 40) / 30.0))

        # Speech detection: LRA > 5 suggests dynamic speech content
        lra_match = re.search(r"LRA:\s*([0-9.]+)", stderr)
        lra = float(lra_match.group(1)) if lra_match else 0
        has_speech = lra > 4.0 and energy > 0.2

        return energy, has_speech

    except Exception as exc:
        logger.debug("Audio analysis failed for %s: %s", clip_path, exc)
        return 0.5, False


def _analyze_clip_visual(clip_path: str, duration: float) -> Tuple[str, float, float]:
    """Analyze visual characteristics of a clip.

    Returns (visual_type, motion_level_0_1, brightness_0_1).
    """
    # Sample brightness and motion from middle of clip
    midpoint = min(duration / 2, 5.0)

    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(midpoint),
        "-i", clip_path,
        "-vframes", "5",
        "-vf", "signalstats",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        stderr = result.stderr.decode(errors="replace")

        # Brightness from YAVG
        yavg_values = []
        for line in stderr.split("\n"):
            match = re.search(r"YAVG=\s*([0-9.]+)", line)
            if match:
                yavg_values.append(float(match.group(1)) / 255.0)
        brightness = sum(yavg_values) / len(yavg_values) if yavg_values else 0.5

    except Exception:
        brightness = 0.5

    # Motion detection via scene score
    cmd2 = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", clip_path,
        "-vf", "select='gte(scene,0.01)',metadata=print",
        "-vsync", "vfr", "-f", "null", "-"
    ]
    try:
        result2 = subprocess.run(cmd2, capture_output=True, timeout=60)
        stderr2 = result2.stderr.decode(errors="replace")
        scores = []
        for line in stderr2.split("\n"):
            match = re.search(r"scene_score=([0-9.]+)", line)
            if match:
                scores.append(float(match.group(1)))
        motion = sum(scores) / len(scores) if scores else 0.0
        motion = min(1.0, motion * 5)  # amplify for usable range
    except Exception:
        motion = 0.3

    # Classify visual type
    if motion > 0.6:
        visual_type = "action"
    elif motion < 0.05:
        visual_type = "static"
    elif brightness > 0.3 and motion < 0.3:
        visual_type = "talking_head"  # low motion, normal brightness
    else:
        visual_type = "broll" if motion > 0.15 else "general"

    return visual_type, motion, brightness


def _classify_mood(energy: float, brightness: float, has_speech: bool,
                   motion: float) -> str:
    """Classify the overall mood of a clip from metrics."""
    if energy > 0.7 and motion > 0.5:
        return "exciting"
    if energy > 0.6 and motion > 0.3:
        return "energetic"
    if energy < 0.2 and brightness < 0.3:
        return "sad"
    if brightness > 0.6 and energy > 0.4:
        return "happy"
    if motion > 0.4 and energy > 0.5:
        return "dramatic"
    if motion < 0.1 and energy < 0.3:
        return "peaceful"
    if has_speech and motion < 0.2:
        return "informative"
    if energy > 0.5 and brightness < 0.4:
        return "tense"
    if brightness > 0.5 and motion < 0.2:
        return "calm"
    return "neutral"


def _get_transcript_snippet(clip_path: str) -> str:
    """Try to get a brief transcript of the clip via Whisper."""
    try:
        from opencut.core.captions import transcribe
        result = transcribe(clip_path, model="tiny")
        if hasattr(result, "text"):
            return result.text[:500]
        if isinstance(result, dict):
            return result.get("text", "")[:500]
    except Exception:
        pass
    return ""


def _analyze_single_clip(
    clip_path: str,
    index: int,
    on_progress: Optional[Callable] = None,
) -> ClipInfo:
    """Fully analyze a single clip."""
    info = get_video_info(clip_path)
    duration = info.get("duration", 0)
    filename = os.path.basename(clip_path)

    # Audio analysis
    energy, has_speech = _analyze_clip_audio(clip_path)

    # Visual analysis
    visual_type, motion, brightness = _analyze_clip_visual(clip_path, duration)

    # Mood classification
    mood = _classify_mood(energy, brightness, has_speech, motion)

    # Optional transcript
    transcript = _get_transcript_snippet(clip_path) if has_speech else ""

    return ClipInfo(
        index=index,
        path=clip_path,
        filename=filename,
        duration=round(duration, 3),
        width=info.get("width", 0),
        height=info.get("height", 0),
        fps=info.get("fps", 0),
        has_audio=True,
        has_speech=has_speech,
        transcript=transcript,
        mood=mood,
        visual_type=visual_type,
        energy_level=round(energy, 3),
        brightness=round(brightness, 3),
        suggested_end=round(duration, 3),
    )


# ---------------------------------------------------------------------------
# Narrative ordering
# ---------------------------------------------------------------------------
def _compute_role_affinity(clip: ClipInfo, role: str) -> float:
    """Score how well a clip fits a narrative role (0..1)."""
    mood_roles = _MOOD_ROLE_MAP.get(clip.mood, [])
    base = 0.6 if role in mood_roles else 0.2

    # Bonus for specific visual types
    if role in ("hook", "attention", "best") and clip.energy_level > 0.6:
        base += 0.2
    if role in ("establishing", "context", "overview") and clip.visual_type in ("broll", "static"):
        base += 0.15
    if role in ("content", "step", "answer") and clip.has_speech:
        base += 0.2
    if role in ("climax", "peak") and clip.energy_level > 0.5:
        base += 0.15
    if role in ("resolution", "conclusion", "closer", "outro") and clip.mood in ("calm", "peaceful"):
        base += 0.15
    if role in ("cta",) and clip.has_speech:
        base += 0.1

    return min(1.0, base)


def _assign_narrative_order(
    clips: List[ClipInfo],
    style: str,
) -> Tuple[List[int], List[str], float]:
    """Assign clips to narrative roles and determine ordering.

    Returns (ordered_indices, structure_labels, narrative_fit_score).
    """
    style_def = NARRATIVE_STYLES.get(style, NARRATIVE_STYLES["documentary"])
    structure = style_def["structure"]

    # If more clips than roles, repeat middle roles
    while len(structure) < len(clips):
        mid = len(structure) // 2
        structure = structure[:mid] + [structure[mid]] + structure[mid:]

    # If fewer clips than roles, trim middle roles
    while len(structure) > len(clips) and len(structure) > 2:
        mid = len(structure) // 2
        structure = structure[:mid] + structure[mid + 1:]

    # Build affinity matrix: clips x roles
    n = min(len(clips), len(structure))
    affinities = []
    for ci, clip in enumerate(clips[:n]):
        row = []
        for ri, role in enumerate(structure[:n]):
            row.append(_compute_role_affinity(clip, role))
        affinities.append(row)

    # Greedy assignment: for each role, pick best available clip
    assigned = [False] * n
    order = []
    used_roles = []
    total_affinity = 0.0

    for ri, role in enumerate(structure[:n]):
        best_ci = -1
        best_score = -1
        for ci in range(n):
            if not assigned[ci] and affinities[ci][ri] > best_score:
                best_score = affinities[ci][ri]
                best_ci = ci
        if best_ci >= 0:
            assigned[best_ci] = True
            order.append(best_ci)
            used_roles.append(role)
            total_affinity += best_score
            clips[best_ci].narrative_role = role
            clips[best_ci].role_score = round(best_score, 3)

    # Add any unassigned clips at the end
    for ci in range(n):
        if not assigned[ci]:
            order.append(ci)
            used_roles.append("extra")
            clips[ci].narrative_role = "extra"

    fit_score = (total_affinity / n * 100) if n > 0 else 0
    return order, used_roles, round(fit_score, 1)


# ---------------------------------------------------------------------------
# Transition suggestions
# ---------------------------------------------------------------------------
def _suggest_transitions(
    clips: List[ClipInfo],
    order: List[int],
    style: str,
) -> List[TransitionSuggestion]:
    """Suggest transitions between ordered clips."""
    style_def = NARRATIVE_STYLES.get(style, NARRATIVE_STYLES["documentary"])
    preferred = style_def.get("preferred_transitions", ["cut"])
    transitions = []

    for i in range(len(order) - 1):
        from_clip = clips[order[i]]
        to_clip = clips[order[i + 1]]

        # Choose transition based on mood change
        mood_change = abs(from_clip.energy_level - to_clip.energy_level)
        brightness_change = abs(from_clip.brightness - to_clip.brightness)

        if mood_change > 0.5 or brightness_change > 0.4:
            # Big mood shift -> dip to black or dissolve
            tr_type = "dip_to_black" if "dip_to_black" in preferred else "crossfade"
            reason = "Significant mood/energy shift between clips"
        elif from_clip.visual_type == to_clip.visual_type and mood_change < 0.1:
            # Similar content -> hard cut or jump cut
            tr_type = preferred[0] if preferred else "cut"
            reason = "Similar content — clean cut maintains flow"
        else:
            # Default: use first preferred transition
            tr_type = preferred[0] if preferred else "crossfade"
            reason = "Standard transition for style"

        tr_def = TRANSITIONS.get(tr_type, TRANSITIONS["cut"])
        transitions.append(TransitionSuggestion(
            from_clip=order[i],
            to_clip=order[i + 1],
            transition_type=tr_type,
            duration=tr_def["duration"],
            reason=reason,
        ))

    return transitions


# ---------------------------------------------------------------------------
# EDL generation
# ---------------------------------------------------------------------------
def _generate_edl(
    clips: List[ClipInfo],
    order: List[int],
    transitions: List[TransitionSuggestion],
) -> str:
    """Generate a text-based edit decision list."""
    lines = ["TITLE: OpenCut Narrative Assembly", ""]
    running_time = 0.0

    for seq, ci in enumerate(order):
        clip = clips[ci]
        tc_in = _format_tc(running_time)
        tc_out = _format_tc(running_time + clip.duration)

        lines.append(f"{seq + 1:03d}  {clip.filename:<40s}  V  C  {tc_in} {tc_out}")
        lines.append(f"     * Role: {clip.narrative_role} | Mood: {clip.mood} | "
                     f"Energy: {clip.energy_level:.2f}")

        # Transition line
        if seq < len(transitions):
            tr = transitions[seq]
            lines.append(f"     >>> {tr.transition_type} ({tr.duration:.1f}s) — {tr.reason}")

        lines.append("")
        running_time += clip.duration

    lines.append(f"TOTAL DURATION: {_format_tc(running_time)}")
    return "\n".join(lines)


def _format_tc(seconds: float) -> str:
    """Format seconds as HH:MM:SS:FF (at 30fps)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1) * 30)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def _generate_narrative_summary(
    clips: List[ClipInfo],
    order: List[int],
    style: str,
    score: float,
) -> str:
    """Generate human-readable narrative summary."""
    parts = [f"Narrative style: {style}"]
    parts.append(f"Arc fitness score: {score:.0f}/100")
    parts.append(f"Assembled {len(order)} clips into narrative order")

    moods = [clips[i].mood for i in order]
    unique_moods = list(dict.fromkeys(moods))
    parts.append(f"Mood progression: {' -> '.join(unique_moods)}")

    total_dur = sum(clips[i].duration for i in order)
    parts.append(f"Total duration: {total_dur:.1f}s")

    speech_count = sum(1 for i in order if clips[i].has_speech)
    if speech_count > 0:
        parts.append(f"{speech_count} clip(s) contain speech")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def build_narrative(
    clip_paths: List[str],
    style: str = "documentary",
    on_progress: Optional[Callable] = None,
) -> NarrativeResult:
    """
    Analyze clips and build a cohesive narrative assembly plan.

    Args:
        clip_paths: List of paths to video clips.
        style: Narrative style key from NARRATIVE_STYLES.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        NarrativeResult with ordered clips, transitions, and EDL.

    Raises:
        FileNotFoundError: If any clip path does not exist.
        ValueError: If no clips provided or style is unknown.
    """
    if not clip_paths:
        raise ValueError("No clip paths provided")

    if style not in NARRATIVE_STYLES:
        raise ValueError(f"Unknown narrative style '{style}'. "
                         f"Available: {', '.join(NARRATIVE_STYLES.keys())}")

    # Validate all paths
    for p in clip_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Clip not found: {p}")

    if on_progress:
        on_progress(1, "Starting narrative analysis...")

    # 1. Analyze all clips
    clips: List[ClipInfo] = []
    for i, path in enumerate(clip_paths):
        if on_progress:
            pct = 5 + int(50 * i / len(clip_paths))
            on_progress(pct, f"Analyzing clip {i + 1}/{len(clip_paths)}...")
        clip = _analyze_single_clip(path, i, on_progress)
        clips.append(clip)

    if on_progress:
        on_progress(60, "Building narrative order...")

    # 2. Assign narrative roles and ordering
    order, structure_labels, fit_score = _assign_narrative_order(clips, style)

    if on_progress:
        on_progress(75, "Suggesting transitions...")

    # 3. Suggest transitions
    transitions = _suggest_transitions(clips, order, style)

    if on_progress:
        on_progress(85, "Generating EDL...")

    # 4. Generate EDL
    edl = _generate_edl(clips, order, transitions)

    # 5. Summary
    summary = _generate_narrative_summary(clips, order, style, fit_score)

    total_duration = sum(clips[i].duration for i in order)

    result = NarrativeResult(
        style=style,
        clip_count=len(clips),
        total_duration=round(total_duration, 3),
        assembly_order=order,
        clips=[clips[i].to_dict() for i in order],
        transitions=[t.to_dict() for t in transitions],
        structure_labels=structure_labels,
        narrative_score=fit_score,
        assembly_edl=edl,
        summary=summary,
    )

    if on_progress:
        on_progress(100, "Narrative assembly complete")

    return result
