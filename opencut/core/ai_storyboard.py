"""
OpenCut AI Storyboard Generator Module v0.1.0

Parse a script, generate storyboard images, and assemble into grid/PDF:
- Parse script text into shot descriptions with camera directions
- Generate placeholder storyboard images (silhouette/sketch style)
- Assemble into grid layout with annotations
- Export as PDF with descriptions and shot numbers

Uses Pillow for image generation (no AI model required for base mode).
"""

import logging
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import ensure_package

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ShotDescription:
    """A single shot parsed from the script."""
    shot_number: int
    shot_type: str = ""         # WIDE, MEDIUM, CLOSE-UP, etc.
    description: str = ""       # Visual description
    action: str = ""            # Character action/dialogue
    camera_direction: str = ""  # PAN, TILT, DOLLY, STATIC, etc.
    duration_hint: str = ""     # Duration suggestion
    dialogue: str = ""          # Dialogue in the shot
    notes: str = ""             # Director notes


@dataclass
class StoryboardImage:
    """A generated storyboard panel."""
    shot_number: int
    image_path: str = ""
    description: str = ""
    shot_type: str = ""
    camera_direction: str = ""
    width: int = 640
    height: int = 360


@dataclass
class Storyboard:
    """Complete storyboard with all panels."""
    panels: List[StoryboardImage] = field(default_factory=list)
    shots: List[ShotDescription] = field(default_factory=list)
    grid_path: str = ""
    pdf_path: str = ""
    total_shots: int = 0


# ---------------------------------------------------------------------------
# Script Parsing
# ---------------------------------------------------------------------------

# Shot type indicators
_SHOT_TYPE_PATTERNS = {
    "ECU": re.compile(r"\b(extreme\s+close[\s-]?up|ECU)\b", re.I),
    "CU": re.compile(r"\b(close[\s-]?up|CU)\b", re.I),
    "MCU": re.compile(r"\b(medium\s+close[\s-]?up|MCU)\b", re.I),
    "MS": re.compile(r"\b(medium\s+shot|MS|mid[\s-]?shot)\b", re.I),
    "MWS": re.compile(r"\b(medium\s+wide|MWS)\b", re.I),
    "WS": re.compile(r"\b(wide\s+shot|WS|full\s+shot)\b", re.I),
    "EWS": re.compile(r"\b(extreme\s+wide|EWS|establishing)\b", re.I),
    "OTS": re.compile(r"\b(over[\s-]?the[\s-]?shoulder|OTS)\b", re.I),
    "POV": re.compile(r"\b(point[\s-]?of[\s-]?view|POV)\b", re.I),
    "AERIAL": re.compile(r"\b(aerial|drone|bird'?s?[\s-]?eye)\b", re.I),
    "INSERT": re.compile(r"\b(insert|detail)\b", re.I),
    "TWO-SHOT": re.compile(r"\b(two[\s-]?shot|2[\s-]?shot)\b", re.I),
}

_CAMERA_DIRECTION_PATTERNS = {
    "PAN LEFT": re.compile(r"\bpan\s+(left|L)\b", re.I),
    "PAN RIGHT": re.compile(r"\bpan\s+(right|R)\b", re.I),
    "TILT UP": re.compile(r"\btilt\s+(up)\b", re.I),
    "TILT DOWN": re.compile(r"\btilt\s+(down)\b", re.I),
    "DOLLY IN": re.compile(r"\b(dolly|push)\s+in\b", re.I),
    "DOLLY OUT": re.compile(r"\b(dolly|pull)\s+out\b", re.I),
    "ZOOM IN": re.compile(r"\bzoom\s+in\b", re.I),
    "ZOOM OUT": re.compile(r"\bzoom\s+out\b", re.I),
    "TRACKING": re.compile(r"\b(tracking|follow)\b", re.I),
    "CRANE UP": re.compile(r"\bcrane\s+up\b", re.I),
    "CRANE DOWN": re.compile(r"\bcrane\s+down\b", re.I),
    "HANDHELD": re.compile(r"\b(handheld|shaky)\b", re.I),
    "STEADICAM": re.compile(r"\b(steadicam|gimbal)\b", re.I),
    "STATIC": re.compile(r"\b(static|locked[\s-]?off|tripod)\b", re.I),
}

# Scene heading pattern (INT./EXT.)
_SCENE_HEADING = re.compile(
    r"^\s*(INT\.?|EXT\.?|INT\.?/EXT\.?)\s+(.+?)(?:\s*-\s*(.+))?\s*$",
    re.I | re.M,
)

# Shot delimiter patterns
_SHOT_DELIMITERS = [
    re.compile(r"^\s*(?:SHOT|Shot)\s*#?\s*(\d+)\s*[:\-]?\s*(.*)", re.M),
    re.compile(r"^\s*(\d+)\s*[.)]\s*(.*)", re.M),
    re.compile(r"^\s*\[([A-Z\s\-]+)\]\s*(.*)", re.M),
]


def parse_shot_descriptions(script_text: str) -> List[ShotDescription]:
    """
    Parse a script or shot list into individual shot descriptions.

    Handles formats:
    - Numbered shots: "1. Wide shot of office..."
    - Shot headings: "SHOT #1: ..."
    - Scene headings: "INT. OFFICE - DAY"
    - Bracketed directions: "[WIDE SHOT] Character enters..."
    - Freeform paragraphs (each paragraph = one shot)

    Args:
        script_text: Full script text.

    Returns:
        List of ShotDescription objects.
    """
    shots = []
    text = script_text.strip()

    if not text:
        return shots

    # Try structured shot parsing first
    segments = []

    # Check for numbered shots
    for pattern in _SHOT_DELIMITERS:
        found = list(pattern.finditer(text))
        if len(found) >= 2:
            for i, match in enumerate(found):
                start = match.start()
                end = found[i + 1].start() if i + 1 < len(found) else len(text)
                segments.append(text[start:end].strip())
            break

    # Check for scene headings
    if not segments:
        headings = list(_SCENE_HEADING.finditer(text))
        if headings:
            for i, match in enumerate(headings):
                start = match.start()
                end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
                segments.append(text[start:end].strip())

    # Fall back to paragraph splitting
    if not segments:
        paragraphs = re.split(r"\n\s*\n", text)
        segments = [p.strip() for p in paragraphs if p.strip()]

    # Parse each segment
    for i, segment in enumerate(segments):
        shot = ShotDescription(shot_number=i + 1)

        # Detect shot type
        for shot_type, pattern in _SHOT_TYPE_PATTERNS.items():
            if pattern.search(segment):
                shot.shot_type = shot_type
                break

        if not shot.shot_type:
            shot.shot_type = "MS"  # Default to medium shot

        # Detect camera direction
        for direction, pattern in _CAMERA_DIRECTION_PATTERNS.items():
            if pattern.search(segment):
                shot.camera_direction = direction
                break

        if not shot.camera_direction:
            shot.camera_direction = "STATIC"

        # Extract dialogue (text in quotes or after character name:)
        dialogue_match = re.search(r'"([^"]+)"', segment)
        if dialogue_match:
            shot.dialogue = dialogue_match.group(1)
        else:
            char_dialogue = re.search(r"^([A-Z][A-Z\s]+):\s*(.+)$", segment, re.M)
            if char_dialogue:
                shot.dialogue = char_dialogue.group(2).strip()

        # Clean description
        shot.description = segment[:200].strip()

        # Extract action (first sentence without shot type prefix)
        first_line = segment.split("\n")[0].strip()
        shot.action = first_line[:150]

        shots.append(shot)

    return shots


# ---------------------------------------------------------------------------
# Storyboard Image Generation
# ---------------------------------------------------------------------------
def _generate_panel_image(
    shot: ShotDescription,
    width: int = 640,
    height: int = 360,
    output_path: str = "",
) -> str:
    """Generate a single storyboard panel image."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFont

    # Create base image with film-like gray background
    img = Image.new("RGB", (width, height), (45, 45, 50))
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font_large = ImageFont.truetype("arial.ttf", 20)
        font_medium = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Draw frame border (like a camera viewfinder)
    margin = 15
    draw.rectangle(
        [margin, margin, width - margin, height - margin],
        outline=(200, 200, 200), width=2,
    )

    # Draw crosshairs (center marks)
    cx, cy = width // 2, height // 2
    cross_len = 20
    draw.line([(cx - cross_len, cy), (cx + cross_len, cy)], fill=(100, 100, 100), width=1)
    draw.line([(cx, cy - cross_len), (cx, cy + cross_len)], fill=(100, 100, 100), width=1)

    # Draw rule-of-thirds grid
    for i in range(1, 3):
        x = margin + (width - 2 * margin) * i // 3
        draw.line([(x, margin), (x, height - margin)], fill=(60, 60, 65), width=1)
        y = margin + (height - 2 * margin) * i // 3
        draw.line([(margin, y), (width - margin, y)], fill=(60, 60, 65), width=1)

    # Draw shot type silhouettes based on framing
    _draw_shot_silhouette(draw, shot.shot_type, width, height, margin)

    # Shot type label (top left)
    draw.text(
        (margin + 8, margin + 5),
        f"#{shot.shot_number}  {shot.shot_type}",
        fill=(255, 220, 100),
        font=font_large,
    )

    # Camera direction (top right)
    if shot.camera_direction:
        cam_text = shot.camera_direction
        bbox = draw.textbbox((0, 0), cam_text, font=font_medium)
        tw = bbox[2] - bbox[0]
        draw.text(
            (width - margin - tw - 8, margin + 8),
            cam_text,
            fill=(100, 200, 255),
            font=font_medium,
        )

    # Description text (bottom area)
    desc = shot.description[:120]
    wrapped = textwrap.wrap(desc, width=55)
    y_text = height - margin - 12 * len(wrapped) - 8
    for line in wrapped[:3]:
        draw.text(
            (margin + 8, y_text),
            line,
            fill=(180, 180, 180),
            font=font_small,
        )
        y_text += 14

    # Dialogue indicator
    if shot.dialogue:
        draw.text(
            (margin + 8, height - margin - 18),
            f'"{shot.dialogue[:50]}..."' if len(shot.dialogue) > 50 else f'"{shot.dialogue}"',
            fill=(255, 200, 150),
            font=font_small,
        )

    img.save(output_path, "PNG")
    return output_path


def _draw_shot_silhouette(draw, shot_type: str, width: int, height: int, margin: int):
    """Draw a simple silhouette figure based on shot framing."""
    cx = width // 2
    cy = height // 2

    # Colors for silhouette
    body_color = (80, 80, 90)
    head_color = (90, 90, 100)

    if shot_type in ("ECU", "CU"):
        # Close-up: large head/face
        head_r = min(width, height) // 4
        draw.ellipse(
            [cx - head_r, cy - head_r, cx + head_r, cy + head_r],
            fill=head_color, outline=(100, 100, 110),
        )
        # Eyes
        eye_y = cy - head_r // 4
        draw.ellipse([cx - head_r // 3 - 5, eye_y - 5, cx - head_r // 3 + 5, eye_y + 5],
                     fill=(60, 60, 70))
        draw.ellipse([cx + head_r // 3 - 5, eye_y - 5, cx + head_r // 3 + 5, eye_y + 5],
                     fill=(60, 60, 70))

    elif shot_type in ("MS", "MCU", "TWO-SHOT"):
        # Medium shot: head + shoulders
        head_r = min(width, height) // 8
        head_y = cy - height // 6
        draw.ellipse(
            [cx - head_r, head_y - head_r, cx + head_r, head_y + head_r],
            fill=head_color,
        )
        # Shoulders/torso
        shoulder_w = head_r * 3
        draw.rounded_rectangle(
            [cx - shoulder_w, head_y + head_r, cx + shoulder_w, height - margin - 10],
            radius=15, fill=body_color,
        )

        if shot_type == "TWO-SHOT":
            # Second person
            cx2 = cx + width // 4
            draw.ellipse(
                [cx2 - head_r, head_y - head_r, cx2 + head_r, head_y + head_r],
                fill=head_color,
            )
            draw.rounded_rectangle(
                [cx2 - shoulder_w, head_y + head_r, cx2 + shoulder_w, height - margin - 10],
                radius=15, fill=body_color,
            )

    elif shot_type in ("WS", "MWS", "EWS"):
        # Wide shot: full body, smaller
        scale = 0.4 if shot_type == "WS" else 0.25
        head_r = int(min(width, height) * 0.04 * (1 / scale))
        head_r = max(8, min(head_r, 30))
        body_h = int(height * scale)
        person_y = height - margin - body_h

        # Person
        draw.ellipse(
            [cx - head_r, person_y, cx + head_r, person_y + head_r * 2],
            fill=head_color,
        )
        draw.rounded_rectangle(
            [cx - head_r * 2, person_y + head_r * 2,
             cx + head_r * 2, height - margin - 10],
            radius=8, fill=body_color,
        )

        # Horizon line for wide shots
        horizon_y = cy - height // 8
        draw.line(
            [(margin, horizon_y), (width - margin, horizon_y)],
            fill=(70, 70, 75), width=1,
        )

    elif shot_type == "AERIAL":
        # Aerial: ground pattern
        for i in range(4):
            y = margin + (height - 2 * margin) * (i + 1) // 5
            offset = (i % 2) * 40
            draw.line([(margin + offset, y), (width - margin - offset, y)],
                      fill=(70, 75, 70), width=1)
        # Small figure
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=head_color)

    elif shot_type == "INSERT":
        # Insert: object/detail focus
        obj_w = width // 4
        obj_h = height // 4
        draw.rounded_rectangle(
            [cx - obj_w, cy - obj_h, cx + obj_w, cy + obj_h],
            radius=10, fill=body_color, outline=(100, 100, 110),
        )

    else:
        # Default: medium shot
        head_r = min(width, height) // 8
        draw.ellipse(
            [cx - head_r, cy - head_r - 20, cx + head_r, cy + head_r - 20],
            fill=head_color,
        )
        draw.rounded_rectangle(
            [cx - head_r * 2, cy + head_r - 20, cx + head_r * 2, height - margin - 10],
            radius=12, fill=body_color,
        )


# ---------------------------------------------------------------------------
# Grid Assembly
# ---------------------------------------------------------------------------
def render_storyboard_grid(
    images: List[StoryboardImage],
    descriptions: List[ShotDescription],
    output_path: str,
    columns: int = 3,
    panel_width: int = 640,
    panel_height: int = 360,
) -> str:
    """
    Assemble storyboard panels into a grid image.

    Args:
        images: List of StoryboardImage objects.
        descriptions: List of ShotDescription objects.
        output_path: Where to save the grid image.
        columns: Number of columns in the grid.
        panel_width: Width of each panel.
        panel_height: Height of each panel.

    Returns:
        Path to the grid image.
    """
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFont

    if not images:
        raise ValueError("No images to render into grid")

    rows = math.ceil(len(images) / columns)
    label_height = 40  # Space for shot label below each panel
    grid_w = columns * panel_width
    grid_h = rows * (panel_height + label_height)

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 35))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, panel in enumerate(images):
        col = i % columns
        row = i // columns
        x = col * panel_width
        y = row * (panel_height + label_height)

        try:
            img = Image.open(panel.image_path)
            img = img.resize((panel_width, panel_height))
            grid.paste(img, (x, y))
        except Exception as e:
            logger.warning("Cannot load panel image %s: %s", panel.image_path, e)
            draw.rectangle([x, y, x + panel_width, y + panel_height],
                           fill=(50, 50, 55))

        # Label
        label = f"#{panel.shot_number}  {panel.shot_type}"
        if panel.camera_direction:
            label += f"  |  {panel.camera_direction}"
        draw.text(
            (x + 5, y + panel_height + 5),
            label,
            fill=(200, 200, 200),
            font=font,
        )

        # Brief description
        desc = panel.description[:80]
        draw.text(
            (x + 5, y + panel_height + 22),
            desc,
            fill=(150, 150, 150),
            font=font,
        )

    grid.save(output_path, "PNG")
    return output_path


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------
def export_storyboard_pdf(
    storyboard: Storyboard,
    output_path: str,
) -> str:
    """
    Export storyboard to a PDF document.

    Each page contains panels with descriptions, shot types,
    camera directions, and dialogue.

    Args:
        storyboard: Complete Storyboard object.
        output_path: Where to save the PDF.

    Returns:
        Path to the PDF file.
    """
    ensure_package("PIL", "Pillow")
    from PIL import Image

    if not storyboard.panels:
        raise ValueError("No panels in storyboard to export")

    # Use Pillow-based multi-page image PDF (no reportlab needed)
    page_w, page_h = 2480, 3508  # A4 at 300dpi
    margin = 100
    cols = 2
    panel_w = (page_w - 3 * margin) // cols
    panel_h = int(panel_w * 9 / 16)
    label_h = 120
    row_h = panel_h + label_h

    rows_per_page = max(1, (page_h - 2 * margin) // row_h)
    panels_per_page = rows_per_page * cols

    pages = []
    panels = storyboard.panels

    try:
        font_large = None  # Use default
        from PIL import ImageDraw, ImageFont
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large
    except Exception:
        pass

    for page_idx in range(0, len(panels), panels_per_page):
        page_panels = panels[page_idx:page_idx + panels_per_page]
        page = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        draw = ImageDraw.Draw(page)

        # Title on first page
        if page_idx == 0:
            draw.text((margin, margin // 2), "STORYBOARD", fill=(0, 0, 0),
                      font=font_large)

        for i, panel in enumerate(page_panels):
            col = i % cols
            row = i // cols
            x = margin + col * (panel_w + margin)
            y = margin + 60 + row * row_h

            # Draw panel frame
            draw.rectangle([x, y, x + panel_w, y + panel_h],
                           outline=(0, 0, 0), width=2)

            try:
                img = Image.open(panel.image_path)
                img = img.resize((panel_w, panel_h))
                page.paste(img, (x, y))
            except Exception:
                draw.rectangle([x + 2, y + 2, x + panel_w - 2, y + panel_h - 2],
                               fill=(240, 240, 240))

            # Labels
            label_y = y + panel_h + 5
            draw.text((x, label_y),
                      f"Shot #{panel.shot_number} - {panel.shot_type}",
                      fill=(0, 0, 0), font=font_medium)
            label_y += 28
            if panel.camera_direction:
                draw.text((x, label_y),
                          f"Camera: {panel.camera_direction}",
                          fill=(80, 80, 80), font=font_small)
                label_y += 22
            desc = panel.description[:100]
            draw.text((x, label_y), desc, fill=(60, 60, 60), font=font_small)

        pages.append(page)

    if pages:
        pages[0].save(output_path, "PDF", resolution=300.0,
                      save_all=True, append_images=pages[1:])

    return output_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def generate_storyboard(
    script_text: str,
    output_dir: str,
    columns: int = 3,
    panel_width: int = 640,
    panel_height: int = 360,
    export_pdf: bool = True,
    on_progress: Optional[Callable] = None,
) -> Storyboard:
    """
    Generate a complete storyboard from a script.

    Pipeline:
    1. Parse script into shot descriptions
    2. Generate storyboard panel images
    3. Assemble into grid image
    4. Optionally export as PDF

    Args:
        script_text: Full script or shot list text.
        output_dir: Directory for output files.
        columns: Grid columns for layout.
        panel_width: Width of each panel image.
        panel_height: Height of each panel image.
        export_pdf: Whether to also generate a PDF.
        on_progress: Callback(pct, msg) for progress.

    Returns:
        Storyboard with all panels, grid path, and optional PDF path.
    """
    if not ensure_package("PIL", "Pillow", on_progress):
        raise RuntimeError("Pillow is required for storyboard generation")

    if not script_text.strip():
        raise ValueError("script_text cannot be empty")

    os.makedirs(output_dir, exist_ok=True)

    storyboard = Storyboard()

    if on_progress:
        on_progress(5, "Parsing script...")

    shots = parse_shot_descriptions(script_text)
    storyboard.shots = shots
    storyboard.total_shots = len(shots)

    if not shots:
        raise ValueError("No shots found in script")

    if on_progress:
        on_progress(15, f"Parsed {len(shots)} shots, generating panels...")

    # Generate panel images
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    panels = []
    for i, shot in enumerate(shots):
        panel_path = os.path.join(panels_dir, f"shot_{shot.shot_number:03d}.png")

        _generate_panel_image(
            shot,
            width=panel_width,
            height=panel_height,
            output_path=panel_path,
        )

        panel = StoryboardImage(
            shot_number=shot.shot_number,
            image_path=panel_path,
            description=shot.description[:120],
            shot_type=shot.shot_type,
            camera_direction=shot.camera_direction,
            width=panel_width,
            height=panel_height,
        )
        panels.append(panel)

        if on_progress and (i + 1) % 3 == 0:
            pct = 15 + int((i / len(shots)) * 50)
            on_progress(pct, f"Generating panel {i + 1}/{len(shots)}...")

    storyboard.panels = panels

    if on_progress:
        on_progress(70, "Assembling grid...")

    # Generate grid
    grid_path = os.path.join(output_dir, "storyboard_grid.png")
    render_storyboard_grid(
        panels, shots, grid_path,
        columns=columns,
        panel_width=panel_width,
        panel_height=panel_height,
    )
    storyboard.grid_path = grid_path

    # Generate PDF
    if export_pdf:
        if on_progress:
            on_progress(85, "Exporting PDF...")

        pdf_path = os.path.join(output_dir, "storyboard.pdf")
        export_storyboard_pdf(storyboard, pdf_path)
        storyboard.pdf_path = pdf_path

    if on_progress:
        on_progress(100, f"Storyboard complete: {len(panels)} panels")

    return storyboard
