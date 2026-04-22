"""
OpenCut Screenplay Parser v1.28.0 — Tier 3 (partial)

IntelliScript .fdx / Fountain import. Parse scene headings + fuzzy-match transcript.
Fountain parsing and FDX parsing work in v1.28.0.
assemble_from_screenplay() ships in v1.29.0.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

INSTALL_HINT = "No install required for Fountain. Final Draft .fdx: pip install fdx (optional)"


def check_screenplay_parser_available() -> bool:
    """Always True — uses stdlib XML for FDX, plain text for Fountain."""
    return True


@dataclass
class Scene:
    heading: str = ""
    action: str = ""
    characters: List[str] = field(default_factory=list)
    dialogue: List[str] = field(default_factory=list)


def parse_fountain(path: str) -> List[Scene]:
    """Parse Fountain (.fountain) screenplay file into Scene list."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    scenes: List[Scene] = []
    current = None
    current_char = ""
    in_dialogue = False

    int_ext = re.compile(r"^(INT\.|EXT\.|INT/EXT\.|I/E\.)", re.IGNORECASE)
    heading_prefix = re.compile(r"^(#{1,3}\s+|\.(?=[A-Z]))", re.IGNORECASE)

    for line in lines:
        stripped = line.rstrip("\n")
        text = stripped.strip()
        is_heading = bool(int_ext.match(text)) or bool(heading_prefix.match(text))

        if is_heading:
            if current is not None:
                scenes.append(current)
            current = Scene(heading=text)
            current_char = ""
            in_dialogue = False
        elif current is None:
            current = Scene()
        elif text.isupper() and len(text) > 1 and not text.startswith("("):
            current_char = text
            if current_char not in current.characters:
                current.characters.append(current_char)
            in_dialogue = True
        elif in_dialogue and text and current_char:
            current.dialogue.append(f"{current_char}: {text}")
            in_dialogue = False
        elif text:
            if current.action:
                current.action += " " + text
            else:
                current.action = text

    if current is not None:
        scenes.append(current)

    return scenes


def parse_fdx(path: str) -> List[Scene]:
    """Parse Final Draft (.fdx) XML screenplay file into Scene list."""
    tree = ET.parse(path)
    root = tree.getroot()

    def tag(el):
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    scenes: List[Scene] = []
    current = None
    current_char = ""

    for elem in root.iter():
        t = tag(elem)
        if t == "Paragraph":
            ptype = elem.get("Type", "")
            text = "".join((sub.text or "") for sub in elem) or (elem.text or "")
            text = text.strip()
            if not text:
                continue
            if ptype == "Scene Heading":
                if current is not None:
                    scenes.append(current)
                current = Scene(heading=text)
                current_char = ""
            elif ptype == "Action" and current is not None:
                if current.action:
                    current.action += " " + text
                else:
                    current.action = text
            elif ptype == "Character" and current is not None:
                current_char = text
                if current_char not in current.characters:
                    current.characters.append(current_char)
            elif ptype == "Dialogue" and current is not None and current_char:
                current.dialogue.append(f"{current_char}: {text}")
                current_char = ""

    if current is not None:
        scenes.append(current)

    return scenes


def assemble_from_screenplay(screenplay_path, video_path, transcript_segments):
    raise NotImplementedError(
        "assemble_from_screenplay ships in v1.29.0. Track ROADMAP-NEXT.md Wave K3.3."
    )


__all__ = ["check_screenplay_parser_available", "INSTALL_HINT", "Scene",
           "parse_fountain", "parse_fdx", "assemble_from_screenplay"]
