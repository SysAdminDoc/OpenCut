"""
Script/Storyboard Integration (4.7)

Parse script files (text/PDF), align script lines to transcript segments,
flag missing coverage, and suggest B-roll from script descriptions.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ScriptLine:
    """A single line/element parsed from the script."""
    index: int
    line_type: str  # "dialogue", "action", "scene_heading", "parenthetical", "transition"
    text: str
    character: str = ""
    scene: str = ""


@dataclass
class ScriptAlignment:
    """Result of aligning a script line to transcript segments."""
    script_line: ScriptLine
    transcript_start: float = -1.0
    transcript_end: float = -1.0
    confidence: float = 0.0
    covered: bool = False


@dataclass
class MissingCoverage:
    """A script line or range with no matching transcript coverage."""
    script_line: ScriptLine
    reason: str = "no_match"


@dataclass
class BrollSuggestion:
    """Suggested B-roll for a script line."""
    script_line: ScriptLine
    suggestion: str = ""
    keywords: List[str] = field(default_factory=list)
    timestamp_hint: float = 0.0


# ---------------------------------------------------------------------------
# Script Parsing
# ---------------------------------------------------------------------------
_SCENE_HEADING_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.)\s+", re.IGNORECASE
)
_CHARACTER_RE = re.compile(r"^([A-Z][A-Z0-9 _\-\.]+)(\s*\(.*\))?\s*$")
_TRANSITION_RE = re.compile(
    r"^(CUT TO:|FADE IN:|FADE OUT\.|DISSOLVE TO:|SMASH CUT TO:|MATCH CUT TO:)",
    re.IGNORECASE,
)
_PARENTHETICAL_RE = re.compile(r"^\(.*\)\s*$")


def parse_script(
    script_path: str,
    on_progress: Optional[Callable] = None,
) -> List[ScriptLine]:
    """Parse a script file (plain text or PDF) into structured lines.

    Args:
        script_path: Path to .txt or .pdf script file.
        on_progress: Optional callback(pct, msg).

    Returns:
        List of ScriptLine objects.
    """
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    ext = os.path.splitext(script_path)[1].lower()
    if on_progress:
        on_progress(5, "Reading script file...")

    if ext == ".pdf":
        raw_text = _read_pdf(script_path)
    else:
        with open(script_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

    if on_progress:
        on_progress(20, "Parsing script structure...")

    lines = _parse_text_to_lines(raw_text)

    if on_progress:
        on_progress(100, f"Parsed {len(lines)} script elements")

    return lines


def _read_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
    except ImportError:
        try:
            import pypdf as PyPDF2
        except ImportError:
            raise RuntimeError(
                "PDF reading requires PyPDF2 or pypdf. "
                "Install with: pip install pypdf"
            )
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _parse_text_to_lines(raw_text: str) -> List[ScriptLine]:
    """Parse raw script text into ScriptLine objects."""
    lines = raw_text.split("\n")
    result = []
    current_scene = ""
    current_character = ""
    idx = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        i += 1

        if not stripped:
            continue

        # Scene heading
        if _SCENE_HEADING_RE.match(stripped):
            current_scene = stripped
            result.append(ScriptLine(
                index=idx, line_type="scene_heading",
                text=stripped, scene=current_scene,
            ))
            idx += 1
            continue

        # Transition
        if _TRANSITION_RE.match(stripped):
            result.append(ScriptLine(
                index=idx, line_type="transition",
                text=stripped, scene=current_scene,
            ))
            idx += 1
            continue

        # Character cue (all caps, centered)
        if _CHARACTER_RE.match(stripped) and len(stripped) < 60:
            current_character = stripped.split("(")[0].strip()
            # Next non-empty line(s) are dialogue or parenthetical
            continue

        # Parenthetical
        if _PARENTHETICAL_RE.match(stripped):
            result.append(ScriptLine(
                index=idx, line_type="parenthetical",
                text=stripped, character=current_character,
                scene=current_scene,
            ))
            idx += 1
            continue

        # Dialogue (if we have a current character and line is indented or follows a cue)
        if current_character and (line.startswith("    ") or line.startswith("\t")):
            result.append(ScriptLine(
                index=idx, line_type="dialogue",
                text=stripped, character=current_character,
                scene=current_scene,
            ))
            idx += 1
            continue

        # Default: action line
        result.append(ScriptLine(
            index=idx, line_type="action",
            text=stripped, scene=current_scene,
        ))
        current_character = ""
        idx += 1

    return result


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _word_overlap(a: str, b: str) -> float:
    """Compute Jaccard word overlap between two strings."""
    words_a = set(_normalize(a).split())
    words_b = set(_normalize(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def align_script_to_transcript(
    script: List[ScriptLine],
    transcript: List[dict],
    threshold: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> List[ScriptAlignment]:
    """Align script lines to transcript segments by text similarity.

    Args:
        script: Parsed script lines from parse_script().
        transcript: List of transcript segments, each with keys
            'text', 'start', 'end'.
        threshold: Minimum word overlap to consider a match.
        on_progress: Optional callback(pct, msg).

    Returns:
        List of ScriptAlignment objects.
    """
    if on_progress:
        on_progress(5, "Aligning script to transcript...")

    alignments = []
    total = len(script)

    for i, line in enumerate(script):
        if on_progress and total > 0 and i % max(1, total // 10) == 0:
            pct = 5 + int((i / total) * 85)
            on_progress(pct, f"Aligning line {i + 1}/{total}...")

        # Only align dialogue and action lines
        if line.line_type not in ("dialogue", "action"):
            alignments.append(ScriptAlignment(
                script_line=line, covered=False,
            ))
            continue

        best_score = 0.0
        best_seg = None

        for seg in transcript:
            seg_text = seg.get("text", "")
            score = _word_overlap(line.text, seg_text)
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_score >= threshold and best_seg:
            alignments.append(ScriptAlignment(
                script_line=line,
                transcript_start=float(best_seg.get("start", 0)),
                transcript_end=float(best_seg.get("end", 0)),
                confidence=round(best_score, 3),
                covered=True,
            ))
        else:
            alignments.append(ScriptAlignment(
                script_line=line,
                confidence=round(best_score, 3),
                covered=False,
            ))

    if on_progress:
        covered = sum(1 for a in alignments if a.covered)
        on_progress(100, f"Aligned {covered}/{total} lines")

    return alignments


# ---------------------------------------------------------------------------
# Missing Coverage
# ---------------------------------------------------------------------------
def find_missing_coverage(
    alignment: List[ScriptAlignment],
    on_progress: Optional[Callable] = None,
) -> List[MissingCoverage]:
    """Find script lines that have no transcript coverage.

    Args:
        alignment: Output from align_script_to_transcript().
        on_progress: Optional callback(pct, msg).

    Returns:
        List of MissingCoverage for uncovered dialogue/action lines.
    """
    if on_progress:
        on_progress(10, "Scanning for missing coverage...")

    missing = []
    for a in alignment:
        if a.script_line.line_type in ("dialogue", "action") and not a.covered:
            reason = "no_match"
            if a.confidence > 0:
                reason = "low_confidence"
            missing.append(MissingCoverage(
                script_line=a.script_line, reason=reason,
            ))

    if on_progress:
        on_progress(100, f"Found {len(missing)} uncovered lines")

    return missing


# ---------------------------------------------------------------------------
# B-Roll Suggestions
# ---------------------------------------------------------------------------
_BROLL_KEYWORDS = {
    "car", "vehicle", "road", "street", "city", "skyline", "sunset",
    "ocean", "beach", "mountain", "forest", "rain", "storm", "crowd",
    "building", "house", "office", "restaurant", "park", "river",
    "phone", "computer", "screen", "door", "window", "table", "food",
    "night", "day", "morning", "evening", "fire", "water", "sky",
}


def suggest_broll_from_script(
    script_lines: List[ScriptLine],
    media_library: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[BrollSuggestion]:
    """Suggest B-roll clips based on script action/scene descriptions.

    Args:
        script_lines: Parsed script lines.
        media_library: Optional list of available media file paths.
        on_progress: Optional callback(pct, msg).

    Returns:
        List of BrollSuggestion objects.
    """
    if on_progress:
        on_progress(5, "Analyzing script for B-roll opportunities...")

    suggestions = []
    action_lines = [ln for ln in script_lines
                    if ln.line_type in ("action", "scene_heading")]
    total = len(action_lines)

    for i, line in enumerate(action_lines):
        if on_progress and total > 0 and i % max(1, total // 5) == 0:
            pct = 5 + int((i / total) * 85)
            on_progress(pct, f"Processing line {i + 1}/{total}...")

        words = set(_normalize(line.text).split())
        keywords = sorted(words & _BROLL_KEYWORDS)

        if not keywords:
            continue

        suggestion_text = f"B-roll for: {line.text[:80]}"
        if media_library:
            # Simple keyword matching against file names
            matches = []
            for path in media_library:
                basename = os.path.splitext(os.path.basename(path))[0].lower()
                for kw in keywords:
                    if kw in basename:
                        matches.append(path)
                        break
            if matches:
                suggestion_text += f" (found {len(matches)} matching clip(s))"

        suggestions.append(BrollSuggestion(
            script_line=line,
            suggestion=suggestion_text,
            keywords=keywords,
            timestamp_hint=float(line.index),
        ))

    if on_progress:
        on_progress(100, f"Generated {len(suggestions)} B-roll suggestions")

    return suggestions
