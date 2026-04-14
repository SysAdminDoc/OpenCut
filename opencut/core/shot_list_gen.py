"""
OpenCut Shot List Generator from Screenplay (59.2)

Parse screenplay files (.fountain, plain text) into structured shot lists:
- Detect INT./EXT. scene headings, ACTION, CHARACTER, DIALOGUE blocks
- Suggest camera angles per scene based on content analysis
- Export as CSV or JSON

No external dependencies beyond the standard library.
"""

import csv
import io
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ScreenplayElement:
    """A parsed element from the screenplay."""
    index: int
    element_type: str  # scene_heading, action, character, dialogue, parenthetical, transition
    text: str
    character: str = ""
    scene_number: int = 0
    scene_heading: str = ""


@dataclass
class ShotEntry:
    """A single shot in the generated shot list."""
    shot_number: int
    scene_number: int = 0
    scene_heading: str = ""
    shot_type: str = "MS"            # WS, MS, CU, ECU, OTS, etc.
    camera_angle: str = "EYE_LEVEL"  # HIGH, LOW, EYE_LEVEL, BIRD, DUTCH
    camera_movement: str = "STATIC"  # STATIC, PAN, TILT, DOLLY, TRACKING, CRANE
    description: str = ""
    characters: List[str] = field(default_factory=list)
    dialogue_summary: str = ""
    notes: str = ""
    duration_estimate: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShotListResult:
    """Complete shot list output."""
    shots: List[ShotEntry] = field(default_factory=list)
    total_shots: int = 0
    total_scenes: int = 0
    csv_path: str = ""
    json_path: str = ""

    def to_dict(self) -> dict:
        return {
            "total_shots": self.total_shots,
            "total_scenes": self.total_scenes,
            "csv_path": self.csv_path,
            "json_path": self.json_path,
            "shots": [s.to_dict() for s in self.shots],
        }


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
_SCENE_HEADING = re.compile(
    r"^\s*(INT\.?|EXT\.?|INT\.?[/\\]EXT\.?|EXT\.?[/\\]INT\.?)\s+(.+?)(?:\s*-\s*(.+))?\s*$",
    re.I | re.M,
)

_CHARACTER_CUE = re.compile(r"^([A-Z][A-Z0-9 _\-\.]+)(\s*\(.*\))?\s*$")
_TRANSITION = re.compile(
    r"^(CUT TO:|FADE IN:|FADE OUT\.|DISSOLVE TO:|SMASH CUT TO:|MATCH CUT TO:|WIPE TO:)",
    re.I,
)
_PARENTHETICAL = re.compile(r"^\(.*\)\s*$")

# Fountain-specific
_FOUNTAIN_SCENE = re.compile(r"^\.(.*)", re.M)
_FOUNTAIN_FORCED_ACTION = re.compile(r"^!(.*)")
_FOUNTAIN_NOTE = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
_FOUNTAIN_CENTERED = re.compile(r"^>(.*)<$")

# Camera/shot keywords for intelligent suggestion
_ACTION_KEYWORDS = {
    "fight": ("MS", "HANDHELD", "LOW"),
    "chase": ("WS", "TRACKING", "EYE_LEVEL"),
    "kiss": ("CU", "STATIC", "EYE_LEVEL"),
    "whisper": ("ECU", "STATIC", "EYE_LEVEL"),
    "enter": ("MS", "STATIC", "EYE_LEVEL"),
    "exit": ("MS", "PAN", "EYE_LEVEL"),
    "run": ("WS", "TRACKING", "EYE_LEVEL"),
    "sit": ("MS", "STATIC", "EYE_LEVEL"),
    "stand": ("MS", "TILT", "LOW"),
    "look": ("CU", "STATIC", "EYE_LEVEL"),
    "stare": ("ECU", "STATIC", "EYE_LEVEL"),
    "cry": ("CU", "STATIC", "EYE_LEVEL"),
    "scream": ("CU", "STATIC", "LOW"),
    "drive": ("MS", "TRACKING", "EYE_LEVEL"),
    "walk": ("MS", "TRACKING", "EYE_LEVEL"),
    "open": ("CU", "STATIC", "EYE_LEVEL"),
    "close": ("CU", "STATIC", "EYE_LEVEL"),
    "reveal": ("WS", "CRANE", "HIGH"),
    "discover": ("MS", "DOLLY", "EYE_LEVEL"),
    "phone": ("CU", "STATIC", "EYE_LEVEL"),
    "type": ("INSERT", "STATIC", "HIGH"),
    "write": ("INSERT", "STATIC", "HIGH"),
    "read": ("CU", "STATIC", "EYE_LEVEL"),
    "crowd": ("EWS", "CRANE", "HIGH"),
    "explosion": ("WS", "STATIC", "LOW"),
    "fall": ("MS", "TILT", "HIGH"),
    "climb": ("MS", "TILT", "LOW"),
    "dance": ("WS", "TRACKING", "EYE_LEVEL"),
    "hug": ("MS", "STATIC", "EYE_LEVEL"),
}


# ---------------------------------------------------------------------------
# Screenplay Parsing
# ---------------------------------------------------------------------------
def parse_screenplay(text: str, is_fountain: bool = False) -> List[ScreenplayElement]:
    """
    Parse a screenplay text into structured elements.

    Supports:
    - Standard screenplay format (INT./EXT. headings, CHARACTER, DIALOGUE)
    - Fountain markup (.fountain files)
    - Plain text with scene headings

    Args:
        text: Raw screenplay text.
        is_fountain: If True, apply Fountain-specific parsing rules.

    Returns:
        List of ScreenplayElement objects.
    """
    if not text or not text.strip():
        return []

    # Strip Fountain notes
    if is_fountain:
        text = _FOUNTAIN_NOTE.sub("", text)

    lines = text.split("\n")
    elements = []
    current_scene = ""
    current_character = ""
    scene_num = 0
    idx = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        i += 1

        if not stripped:
            current_character = ""
            continue

        # Fountain forced scene heading
        if is_fountain and _FOUNTAIN_SCENE.match(stripped):
            scene_num += 1
            heading = _FOUNTAIN_SCENE.match(stripped).group(1).strip()
            current_scene = heading
            elements.append(ScreenplayElement(
                index=idx, element_type="scene_heading",
                text=heading, scene_number=scene_num,
                scene_heading=heading,
            ))
            idx += 1
            continue

        # Standard scene heading
        m = _SCENE_HEADING.match(stripped)
        if m:
            scene_num += 1
            current_scene = stripped
            elements.append(ScreenplayElement(
                index=idx, element_type="scene_heading",
                text=stripped, scene_number=scene_num,
                scene_heading=stripped,
            ))
            idx += 1
            current_character = ""
            continue

        # Transition
        if _TRANSITION.match(stripped):
            elements.append(ScreenplayElement(
                index=idx, element_type="transition",
                text=stripped, scene_number=scene_num,
                scene_heading=current_scene,
            ))
            idx += 1
            current_character = ""
            continue

        # Parenthetical
        if _PARENTHETICAL.match(stripped):
            elements.append(ScreenplayElement(
                index=idx, element_type="parenthetical",
                text=stripped, character=current_character,
                scene_number=scene_num, scene_heading=current_scene,
            ))
            idx += 1
            continue

        # Character cue
        if _CHARACTER_CUE.match(stripped) and len(stripped) < 60:
            current_character = stripped.split("(")[0].strip()
            elements.append(ScreenplayElement(
                index=idx, element_type="character",
                text=stripped, character=current_character,
                scene_number=scene_num, scene_heading=current_scene,
            ))
            idx += 1
            continue

        # Dialogue (follows character cue, indented)
        if current_character and (line.startswith("    ") or line.startswith("\t")):
            elements.append(ScreenplayElement(
                index=idx, element_type="dialogue",
                text=stripped, character=current_character,
                scene_number=scene_num, scene_heading=current_scene,
            ))
            idx += 1
            continue

        # Default: action
        elements.append(ScreenplayElement(
            index=idx, element_type="action",
            text=stripped, scene_number=scene_num,
            scene_heading=current_scene,
        ))
        current_character = ""
        idx += 1

    return elements


# ---------------------------------------------------------------------------
# Camera Suggestion
# ---------------------------------------------------------------------------
def _suggest_camera(elements: List[ScreenplayElement], scene_num: int) -> tuple:
    """
    Suggest shot type, camera movement, and angle for a scene based on content.

    Returns:
        (shot_type, camera_movement, camera_angle)
    """
    scene_elements = [e for e in elements if e.scene_number == scene_num]
    actions = [e.text.lower() for e in scene_elements if e.element_type == "action"]
    dialogues = [e for e in scene_elements if e.element_type == "dialogue"]
    characters = list({e.character for e in scene_elements if e.character})

    combined_action = " ".join(actions)

    # Check for keyword matches in action text
    for keyword, (shot, move, angle) in _ACTION_KEYWORDS.items():
        if keyword in combined_action:
            return shot, move, angle

    # Dialogue-heavy scenes: OTS or MS
    if len(dialogues) > 3:
        if len(characters) >= 2:
            return "OTS", "STATIC", "EYE_LEVEL"
        return "MS", "STATIC", "EYE_LEVEL"

    # Scene heading analysis
    heading = ""
    for e in scene_elements:
        if e.element_type == "scene_heading":
            heading = e.text.lower()
            break

    if "establishing" in heading or heading.startswith("ext."):
        return "EWS", "STATIC", "EYE_LEVEL"

    return "MS", "STATIC", "EYE_LEVEL"


def _estimate_duration(elements: List[ScreenplayElement], scene_num: int) -> str:
    """Estimate scene duration from element count (1 page ~ 1 minute)."""
    scene_els = [e for e in elements if e.scene_number == scene_num]
    # Rough: dialogue line ~ 2sec, action line ~ 3sec
    secs = 0
    for e in scene_els:
        if e.element_type == "dialogue":
            secs += 2
        elif e.element_type == "action":
            secs += 3
        elif e.element_type == "scene_heading":
            secs += 1
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


# ---------------------------------------------------------------------------
# Shot List Generation
# ---------------------------------------------------------------------------
def generate_shot_list(
    screenplay_text: str,
    output_dir: str = "",
    is_fountain: bool = False,
    export_csv: bool = True,
    export_json: bool = True,
    on_progress: Optional[Callable] = None,
) -> ShotListResult:
    """
    Generate a shot list from screenplay text.

    Pipeline:
    1. Parse screenplay into elements
    2. Group by scene
    3. Suggest camera settings per scene
    4. Export as CSV and/or JSON

    Args:
        screenplay_text: Raw screenplay or .fountain text.
        output_dir: Directory for output files.
        is_fountain: Whether input uses Fountain markup.
        export_csv: Whether to export CSV file.
        export_json: Whether to export JSON file.
        on_progress: Callback(pct, msg).

    Returns:
        ShotListResult with shots and export paths.
    """
    if not screenplay_text or not screenplay_text.strip():
        raise ValueError("screenplay_text cannot be empty")

    if on_progress:
        on_progress(5, "Parsing screenplay...")

    elements = parse_screenplay(screenplay_text, is_fountain=is_fountain)

    if not elements:
        raise ValueError("No elements parsed from screenplay")

    if on_progress:
        on_progress(20, f"Parsed {len(elements)} elements, generating shots...")

    # Group by scene
    scenes: Dict[int, List[ScreenplayElement]] = {}
    for e in elements:
        sn = e.scene_number or 0
        if sn not in scenes:
            scenes[sn] = []
        scenes[sn].append(e)

    result = ShotListResult()
    result.total_scenes = len(scenes)
    shot_num = 0

    for scene_num in sorted(scenes.keys()):
        scene_els = scenes[scene_num]
        shot_type, cam_move, cam_angle = _suggest_camera(elements, scene_num)

        # Get scene heading
        heading = ""
        for e in scene_els:
            if e.element_type == "scene_heading":
                heading = e.text
                break

        # Collect characters in scene
        chars = list({e.character for e in scene_els if e.character})

        # Summarize dialogue
        dial_lines = [e.text for e in scene_els if e.element_type == "dialogue"]
        dial_summary = "; ".join(dial_lines[:3])
        if len(dial_lines) > 3:
            dial_summary += f" (+{len(dial_lines) - 3} more)"

        # Action description
        actions = [e.text for e in scene_els if e.element_type == "action"]
        description = " ".join(actions)[:200]

        shot_num += 1
        entry = ShotEntry(
            shot_number=shot_num,
            scene_number=scene_num,
            scene_heading=heading,
            shot_type=shot_type,
            camera_angle=cam_angle,
            camera_movement=cam_move,
            description=description,
            characters=chars,
            dialogue_summary=dial_summary[:200],
            duration_estimate=_estimate_duration(elements, scene_num),
        )
        result.shots.append(entry)

        if on_progress:
            pct = 20 + int((shot_num / max(len(scenes), 1)) * 50)
            on_progress(pct, f"Shot {shot_num}: {heading[:40]}...")

    result.total_shots = len(result.shots)

    # Export
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        if export_csv and result.shots:
            if on_progress:
                on_progress(75, "Exporting CSV...")
            csv_path = os.path.join(output_dir, "shot_list.csv")
            _export_csv(result.shots, csv_path)
            result.csv_path = csv_path

        if export_json and result.shots:
            if on_progress:
                on_progress(85, "Exporting JSON...")
            json_path = os.path.join(output_dir, "shot_list.json")
            _export_json(result.shots, json_path)
            result.json_path = json_path

    if on_progress:
        on_progress(100, f"Shot list complete: {result.total_shots} shots from {result.total_scenes} scenes")

    return result


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------
def _export_csv(shots: List[ShotEntry], path: str):
    """Export shots to CSV file."""
    fieldnames = [
        "shot_number", "scene_number", "scene_heading", "shot_type",
        "camera_angle", "camera_movement", "description", "characters",
        "dialogue_summary", "notes", "duration_estimate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in shots:
            row = s.to_dict()
            row["characters"] = ", ".join(row["characters"])
            writer.writerow(row)


def _export_json(shots: List[ShotEntry], path: str):
    """Export shots to JSON file."""
    data = {
        "shot_list": [s.to_dict() for s in shots],
        "total_shots": len(shots),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def shots_to_csv_string(shots: List[ShotEntry]) -> str:
    """Serialize shots to a CSV string (for in-memory use)."""
    output = io.StringIO()
    fieldnames = [
        "shot_number", "scene_number", "scene_heading", "shot_type",
        "camera_angle", "camera_movement", "description", "characters",
        "dialogue_summary", "notes", "duration_estimate",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for s in shots:
        row = s.to_dict()
        row["characters"] = ", ".join(row["characters"])
        writer.writerow(row)
    return output.getvalue()
