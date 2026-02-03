"""
Styled caption overlay renderer.

Generates transparent video overlays with styled captions including:
- Multiple visual presets (YouTube Bold, Clean Modern, etc.)
- Word-by-word highlight animation
- Action word emphasis (auto-detected or user-specified)
- Stroke, shadow, background box styling

Uses Pillow for text rendering and FFmpeg for video assembly.
"""

import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .captions import CaptionSegment, TranscriptionResult, Word

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    """Visual style for rendered captions."""
    name: str
    label: str                          # Human-readable label
    font_name: str = "Arial Bold"       # Font lookup name
    font_file: str = "arialbd.ttf"      # Fallback filename
    font_size: int = 56                 # Pixels at 1080p
    text_color: Tuple = (255, 255, 255, 255)
    highlight_color: Tuple = (255, 255, 0, 255)
    action_color: Tuple = (255, 80, 80, 255)
    stroke_color: Tuple = (0, 0, 0, 255)
    stroke_width: int = 3
    shadow_color: Tuple = (0, 0, 0, 0)  # alpha 0 = no shadow
    shadow_offset: Tuple = (0, 0)
    bg_color: Tuple = (0, 0, 0, 0)      # alpha 0 = no background
    bg_padding: Tuple = (16, 8)          # h, v padding
    bg_radius: int = 8
    margin_bottom: int = 80             # px from bottom
    max_width_pct: float = 0.82         # Max text width as fraction of video width
    line_spacing: float = 1.3           # Line height multiplier
    # CSS-like properties for panel preview
    preview_css: str = ""


STYLES: Dict[str, CaptionStyle] = {
    "youtube_bold": CaptionStyle(
        name="youtube_bold",
        label="YouTube Bold",
        font_name="Impact",
        font_file="impact.ttf",
        font_size=62,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 230, 0, 255),
        action_color=(255, 50, 50, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=4,
        margin_bottom=80,
        preview_css="font-family:Impact,sans-serif;color:#fff;-webkit-text-stroke:2px #000;text-shadow:3px 3px 0 #000",
    ),
    "clean_modern": CaptionStyle(
        name="clean_modern",
        label="Clean Modern",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=52,
        text_color=(255, 255, 255, 255),
        highlight_color=(0, 200, 255, 255),
        action_color=(255, 100, 100, 255),
        stroke_color=(0, 0, 0, 200),
        stroke_width=2,
        bg_color=(0, 0, 0, 160),
        bg_padding=(20, 10),
        bg_radius=12,
        margin_bottom=70,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;background:rgba(0,0,0,0.6);padding:4px 10px;border-radius:6px;text-shadow:1px 1px 2px rgba(0,0,0,0.8)",
    ),
    "neon_pop": CaptionStyle(
        name="neon_pop",
        label="Neon Pop",
        font_name="Arial Black",
        font_file="ariblk.ttf",
        font_size=58,
        text_color=(255, 255, 80, 255),
        highlight_color=(80, 255, 80, 255),
        action_color=(255, 50, 200, 255),
        stroke_color=(100, 0, 150, 255),
        stroke_width=3,
        shadow_color=(150, 0, 255, 100),
        shadow_offset=(0, 0),
        margin_bottom=80,
        preview_css="font-family:'Arial Black',Arial,sans-serif;color:#ffff50;-webkit-text-stroke:2px #6400a0;text-shadow:0 0 12px rgba(150,0,255,0.7),0 0 24px rgba(150,0,255,0.3)",
    ),
    "minimal": CaptionStyle(
        name="minimal",
        label="Minimal",
        font_name="Arial",
        font_file="arial.ttf",
        font_size=44,
        text_color=(230, 230, 230, 255),
        highlight_color=(255, 255, 255, 255),
        action_color=(255, 180, 100, 255),
        stroke_color=(0, 0, 0, 180),
        stroke_width=1,
        shadow_color=(0, 0, 0, 120),
        shadow_offset=(2, 2),
        margin_bottom=60,
        preview_css="font-family:Arial,Helvetica,sans-serif;color:#e6e6e6;text-shadow:2px 2px 3px rgba(0,0,0,0.5);font-size:0.85em",
    ),
    "boxed": CaptionStyle(
        name="boxed",
        label="Boxed",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=50,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 200, 50, 255),
        action_color=(255, 80, 80, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(15, 15, 15, 220),
        bg_padding=(24, 12),
        bg_radius=4,
        margin_bottom=70,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;background:rgba(15,15,15,0.85);padding:6px 14px;border-radius:3px",
    ),
    "cinematic": CaptionStyle(
        name="cinematic",
        label="Cinematic",
        font_name="Georgia",
        font_file="georgia.ttf",
        font_size=46,
        text_color=(220, 220, 210, 255),
        highlight_color=(255, 240, 200, 255),
        action_color=(255, 160, 80, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        shadow_color=(0, 0, 0, 180),
        shadow_offset=(3, 3),
        margin_bottom=90,
        line_spacing=1.5,
        preview_css="font-family:Georgia,'Times New Roman',serif;color:#dcdcd2;text-shadow:3px 3px 5px rgba(0,0,0,0.7);letter-spacing:0.5px",
    ),
}

DEFAULT_STYLE = "youtube_bold"

# ---------------------------------------------------------------------------
# Action word detection
# ---------------------------------------------------------------------------

# Common emphasis / action words
_ACTION_KEYWORDS: Set[str] = {
    # Exclamations & emphasis
    "amazing", "awesome", "incredible", "insane", "crazy", "unbelievable",
    "massive", "huge", "enormous", "epic", "legendary", "ultimate",
    "absolutely", "completely", "totally", "exactly", "definitely",
    "perfect", "fantastic", "brilliant", "stunning", "gorgeous",
    # Strong verbs
    "crush", "smash", "destroy", "explode", "launch", "blast",
    "dominate", "transform", "revolutionize", "skyrocket",
    # Negative emphasis
    "never", "worst", "terrible", "horrible", "disaster",
    "impossible", "ridiculous", "outrageous",
    # Quantifiers
    "everything", "nothing", "always", "million", "billion",
    # Common YouTube emphasis
    "free", "secret", "hack", "trick", "game-changer", "breakthrough",
    "important", "critical", "essential", "key", "crucial",
}


def detect_action_words_by_energy(
    filepath: str,
    words: List[Word],
    threshold: float = 1.6,
) -> Set[int]:
    """
    Detect action words by comparing per-word audio energy to the average.

    Args:
        filepath: Path to the media file.
        words: List of Word objects with timestamps.
        threshold: RMS multiplier above mean to flag as action.

    Returns:
        Set of word indices that are action words.
    """
    if not words:
        return set()

    try:
        from .audio import extract_audio_pcm
        pcm_bytes, sr = extract_audio_pcm(filepath, sample_rate=16000, mono=True)
    except Exception:
        return set()

    # Convert to samples
    n_samples = len(pcm_bytes) // 2
    if n_samples == 0:
        return set()

    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)

    # Compute RMS for each word
    word_rms = []
    for w in words:
        s_start = max(0, int(w.start * sr))
        s_end = min(n_samples, int(w.end * sr))
        if s_end <= s_start:
            word_rms.append(0.0)
            continue
        chunk = samples[s_start:s_end]
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0
        word_rms.append(rms)

    if not word_rms:
        return set()

    mean_rms = sum(word_rms) / len(word_rms)
    if mean_rms < 0.001:
        return set()

    action_indices = set()
    for i, rms in enumerate(word_rms):
        if rms > mean_rms * threshold:
            action_indices.add(i)

    return action_indices


def get_action_word_indices(
    all_words: List[Word],
    custom_words: Optional[List[str]] = None,
    use_keywords: bool = True,
    energy_indices: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Combine multiple methods to identify action word indices.

    Args:
        all_words: Flat list of all Word objects.
        custom_words: User-specified action words.
        use_keywords: Whether to use the built-in keyword list.
        energy_indices: Word indices flagged by energy analysis.

    Returns:
        Set of word indices that should be styled as action words.
    """
    result = set()

    # Build lookup set
    keywords = set()
    if use_keywords:
        keywords.update(_ACTION_KEYWORDS)
    if custom_words:
        for cw in custom_words:
            keywords.add(cw.strip().lower())

    # Check each word
    for i, w in enumerate(all_words):
        norm = w.text.strip().lower().strip(".,!?;:\"'()-")
        if norm in keywords:
            result.add(i)

    # Merge energy-detected indices
    if energy_indices:
        result.update(energy_indices)

    return result


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

_FONT_DIRS = []
_font_cache: Dict[str, str] = {}


def _get_font_dirs() -> List[str]:
    """Get platform-specific font directories."""
    global _FONT_DIRS
    if _FONT_DIRS:
        return _FONT_DIRS

    dirs = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            dirs.append(os.path.join(localappdata, "Microsoft", "Windows", "Fonts"))
    else:
        dirs.extend([
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
            "/Library/Fonts",
            "/System/Library/Fonts",
        ])

    _FONT_DIRS = [d for d in dirs if os.path.isdir(d)]
    return _FONT_DIRS


def find_font(filename: str) -> Optional[str]:
    """Find a font file by filename across system font directories."""
    if filename in _font_cache:
        return _font_cache[filename]

    # Check direct path
    if os.path.isfile(filename):
        _font_cache[filename] = filename
        return filename

    fn_lower = filename.lower()
    for d in _get_font_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() == fn_lower:
                    path = os.path.join(root, f)
                    _font_cache[filename] = path
                    return path

    return None


def _load_font(style: "CaptionStyle", size_override: Optional[int] = None):
    """Load a PIL ImageFont for the given style."""
    from PIL import ImageFont

    size = size_override or style.font_size

    # Try the style's font file
    font_path = find_font(style.font_file)
    if font_path:
        return ImageFont.truetype(font_path, size)

    # Try common fallbacks
    for fallback in ["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        fp = find_font(fallback)
        if fp:
            return ImageFont.truetype(fp, size)

    # Last resort: Pillow default
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------

@dataclass
class WordLayout:
    """Layout information for a single word."""
    text: str
    x: float
    y: float
    width: float
    line_idx: int
    word_idx: int       # Global word index (for action word lookup)
    is_highlight: bool
    is_action: bool


def layout_caption_text(
    words: List[str],
    font,
    max_width: int,
    video_width: int,
    y_base: int,
    line_spacing: float,
    highlight_idx: int = -1,
    action_indices: Optional[Set[int]] = None,
    global_offset: int = 0,
) -> List[WordLayout]:
    """
    Calculate word positions for a caption, with line wrapping and centering.

    Returns:
        List of WordLayout for each word.
    """
    if not words:
        return []

    act = action_indices or set()

    # Measure each word
    space_width = font.getlength(" ")
    word_widths = [font.getlength(w) for w in words]

    # Wrap into lines
    lines: List[List[int]] = [[]]  # Each line is a list of word indices
    line_widths: List[float] = [0.0]

    for i, (w, ww) in enumerate(zip(words, word_widths)):
        needed = ww + (space_width if lines[-1] else 0)
        if lines[-1] and line_widths[-1] + needed > max_width:
            lines.append([])
            line_widths.append(0.0)
        if lines[-1]:
            line_widths[-1] += space_width
        lines[-1].append(i)
        line_widths[-1] += ww

    # Get font height
    ascent, descent = font.getmetrics()
    line_height = int((ascent + descent) * line_spacing)

    # Calculate total text block height
    total_height = line_height * len(lines)

    # Position lines from y_base upward
    results = []
    for line_num, (line_indices, lw) in enumerate(zip(lines, line_widths)):
        y = y_base - total_height + (line_num * line_height)
        x_start = (video_width - lw) / 2  # Center the line
        x = x_start

        for local_idx, word_idx in enumerate(line_indices):
            global_idx = global_offset + word_idx
            results.append(WordLayout(
                text=words[word_idx],
                x=x,
                y=y,
                width=word_widths[word_idx],
                line_idx=line_num,
                word_idx=global_idx,
                is_highlight=(word_idx == highlight_idx),
                is_action=(global_idx in act),
            ))
            x += word_widths[word_idx] + space_width

    return results


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    layouts: List[WordLayout],
    style: CaptionStyle,
    video_size: Tuple[int, int],
    font,
) -> "Image":
    """Render a single caption frame as a transparent RGBA image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", video_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not layouts:
        return img

    # Draw background box if style has one
    if style.bg_color[3] > 0 and layouts:
        # Calculate bounding box of all words
        min_x = min(wl.x for wl in layouts) - style.bg_padding[0]
        max_x = max(wl.x + wl.width for wl in layouts) + style.bg_padding[0]
        min_y = min(wl.y for wl in layouts) - style.bg_padding[1]
        ascent, descent = font.getmetrics()
        max_y = max(wl.y for wl in layouts) + ascent + descent + style.bg_padding[1]

        try:
            draw.rounded_rectangle(
                [(int(min_x), int(min_y)), (int(max_x), int(max_y))],
                radius=style.bg_radius,
                fill=style.bg_color,
            )
        except AttributeError:
            # Pillow < 8.2 fallback
            draw.rectangle(
                [(int(min_x), int(min_y)), (int(max_x), int(max_y))],
                fill=style.bg_color,
            )

    # Draw each word
    for wl in layouts:
        # Determine color
        if wl.is_action:
            color = style.action_color
        elif wl.is_highlight:
            color = style.highlight_color
        else:
            color = style.text_color

        x, y = int(wl.x), int(wl.y)

        # Shadow pass
        if style.shadow_color[3] > 0:
            sx = x + style.shadow_offset[0]
            sy = y + style.shadow_offset[1]
            # For glow effect (offset 0,0), draw multiple blurred passes
            if style.shadow_offset == (0, 0):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        draw.text(
                            (sx + dx, sy + dy), wl.text, font=font,
                            fill=style.shadow_color,
                        )
            else:
                draw.text(
                    (sx, sy), wl.text, font=font,
                    fill=style.shadow_color,
                )

        # Main text with stroke
        if style.stroke_width > 0:
            draw.text(
                (x, y), wl.text, font=font, fill=color,
                stroke_width=style.stroke_width,
                stroke_fill=style.stroke_color,
            )
        else:
            draw.text((x, y), wl.text, font=font, fill=color)

    return img


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def render_styled_caption_video(
    transcription: TranscriptionResult,
    output_path: str,
    style_name: str = DEFAULT_STYLE,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: float = 30.0,
    action_indices: Optional[Set[int]] = None,
    total_duration: Optional[float] = None,
) -> dict:
    """
    Render a transparent video overlay with styled, word-highlighted captions.

    Args:
        transcription: Whisper transcription with word-level timestamps.
        output_path: Output video path (.mov).
        style_name: Name of the style preset.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        fps: Frame rate.
        action_indices: Set of global word indices to style as action words.
        total_duration: Total overlay duration. If None, uses last caption end + 1s.

    Returns:
        Dict with output_path, frames_rendered, duration.
    """
    from PIL import Image

    style = STYLES.get(style_name, STYLES[DEFAULT_STYLE])
    font = _load_font(style)
    video_size = (video_width, video_height)
    max_text_width = int(video_width * style.max_width_pct)
    y_base = video_height - style.margin_bottom
    act = action_indices or set()

    # Flatten all words with global indices
    all_words_flat = []
    seg_word_map = []  # (seg_idx, word_indices_in_flat_list)
    for seg in transcription.segments:
        start_idx = len(all_words_flat)
        for w in seg.words:
            all_words_flat.append(w)
        seg_word_map.append((start_idx, len(all_words_flat)))

    # Determine total duration
    if total_duration is None:
        if transcription.segments:
            total_duration = transcription.segments[-1].end + 1.0
        else:
            total_duration = 1.0

    # Create temp directory for frames
    tmp_dir = tempfile.mkdtemp(prefix="opencut_captions_")
    concat_lines = ["ffconcat version 1.0"]
    frames_rendered = 0

    def _save_frame(img, duration_sec):
        nonlocal frames_rendered, concat_lines
        if duration_sec < 0.001:
            return
        fname = f"frame_{frames_rendered:06d}.png"
        fpath = os.path.join(tmp_dir, fname)
        img.save(fpath, "PNG")
        concat_lines.append(f"file '{fname}'")
        concat_lines.append(f"duration {duration_sec:.6f}")
        frames_rendered += 1

    # Generate frames for each segment
    prev_end = 0.0

    for seg_idx, seg in enumerate(transcription.segments):
        seg_start_global, seg_end_global = seg_word_map[seg_idx]

        # Gap before this segment (transparent)
        gap = seg.start - prev_end
        if gap > 0.01:
            blank = Image.new("RGBA", video_size, (0, 0, 0, 0))
            _save_frame(blank, gap)

        # If no word-level timestamps, render entire segment as one frame
        if not seg.words:
            words_text = seg.text.strip().split()
            layouts = layout_caption_text(
                words_text, font, max_text_width, video_width, y_base,
                style.line_spacing, highlight_idx=-1,
                action_indices=act, global_offset=seg_start_global,
            )
            frame = render_frame(layouts, style, video_size, font)
            _save_frame(frame, seg.end - seg.start)
            prev_end = seg.end
            continue

        # Word-by-word highlighting
        words_text = [w.text for w in seg.words]
        seg_time = seg.start

        for w_local, word_obj in enumerate(seg.words):
            # Gap between segment start / previous word end and this word
            word_gap = word_obj.start - seg_time
            if word_gap > 0.01:
                # Show caption with no highlight during the gap
                layouts = layout_caption_text(
                    words_text, font, max_text_width, video_width, y_base,
                    style.line_spacing, highlight_idx=-1,
                    action_indices=act, global_offset=seg_start_global,
                )
                frame = render_frame(layouts, style, video_size, font)
                _save_frame(frame, word_gap)

            # Render with this word highlighted
            word_dur = word_obj.end - word_obj.start
            if word_dur < 0.01:
                word_dur = 0.05

            layouts = layout_caption_text(
                words_text, font, max_text_width, video_width, y_base,
                style.line_spacing, highlight_idx=w_local,
                action_indices=act, global_offset=seg_start_global,
            )
            frame = render_frame(layouts, style, video_size, font)
            _save_frame(frame, word_dur)

            seg_time = word_obj.end

        # Gap between last word end and segment end
        tail = seg.end - seg_time
        if tail > 0.01:
            layouts = layout_caption_text(
                words_text, font, max_text_width, video_width, y_base,
                style.line_spacing, highlight_idx=-1,
                action_indices=act, global_offset=seg_start_global,
            )
            frame = render_frame(layouts, style, video_size, font)
            _save_frame(frame, tail)

        prev_end = seg.end

    # Final gap to fill total duration
    final_gap = total_duration - prev_end
    if final_gap > 0.01:
        blank = Image.new("RGBA", video_size, (0, 0, 0, 0))
        _save_frame(blank, final_gap)

    # Repeat last frame (FFmpeg concat quirk)
    if frames_rendered > 0:
        last_fname = f"frame_{frames_rendered - 1:06d}.png"
        concat_lines.append(f"file '{last_fname}'")

    # Write concat file
    concat_path = os.path.join(tmp_dir, "concat.txt")
    with open(concat_path, "w") as f:
        f.write("\n".join(concat_lines))

    # Assemble video with FFmpeg
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_path,
        "-c:v", "qtrle",
        "-pix_fmt", "argb",
        "-r", str(fps),
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=600)

    # Clean up temp frames (keep the MOV)
    for fname in os.listdir(tmp_dir):
        fpath = os.path.join(tmp_dir, fname)
        try:
            os.remove(fpath)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    return {
        "output_path": output_path,
        "frames_rendered": frames_rendered,
        "duration": total_duration,
        "style": style_name,
    }


def get_style_info() -> List[dict]:
    """Get serializable info for all styles (for the panel preview system)."""
    result = []
    for key, s in STYLES.items():
        result.append({
            "name": s.name,
            "label": s.label,
            "preview_css": s.preview_css,
            "highlight_color": f"rgb({s.highlight_color[0]},{s.highlight_color[1]},{s.highlight_color[2]})",
            "action_color": f"rgb({s.action_color[0]},{s.action_color[1]},{s.action_color[2]})",
        })
    return result
