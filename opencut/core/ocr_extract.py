"""
OpenCut OCR Text Extraction

Samples video frames at configurable intervals and runs OCR to extract
on-screen text. Deduplicates repeated text across consecutive frames.

Uses FFmpeg for frame extraction and pytesseract (or stub) for OCR.
"""

import difflib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OCRFrame:
    """OCR result for a single frame."""
    timestamp: float
    text: str = ""
    confidence: float = 0.0
    words: List[Dict] = field(default_factory=list)  # [{word, x, y, w, h, conf}]


@dataclass
class OCRResult:
    """Complete OCR extraction results."""
    frames: List[OCRFrame] = field(default_factory=list)
    unique_texts: List[Dict] = field(default_factory=list)  # [{text, first_seen, last_seen, count}]
    total_frames_sampled: int = 0
    frames_with_text: int = 0
    duration: float = 0.0


@dataclass
class TextSearchHit:
    """A search result when querying for text in video."""
    timestamp: float
    text: str = ""
    context: str = ""  # surrounding text
    confidence: float = 0.0


@dataclass
class TextSearchResult:
    """Results of searching for text in a video."""
    query: str = ""
    hits: List[TextSearchHit] = field(default_factory=list)
    total_hits: int = 0


# ---------------------------------------------------------------------------
# OCR engine abstraction
# ---------------------------------------------------------------------------
def _run_tesseract_ocr(image_path: str) -> Tuple[str, float, List[Dict]]:
    """Run OCR on an image using pytesseract. Returns (text, confidence, words)."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(image_path)

        # Get detailed data with bounding boxes
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words = []
        confidences = []
        text_parts = []

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
            if word and conf > 20:
                words.append({
                    "word": word,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": conf,
                })
                text_parts.append(word)
                confidences.append(conf)

        full_text = " ".join(text_parts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, avg_conf / 100.0, words

    except ImportError:
        logger.debug("pytesseract not available, using stub OCR")
        return _run_stub_ocr(image_path)
    except Exception as exc:
        logger.debug("Tesseract OCR failed for %s: %s", image_path, exc)
        return "", 0.0, []


def _run_stub_ocr(image_path: str) -> Tuple[str, float, List[Dict]]:
    """Stub OCR fallback using basic pixel analysis to detect text-like regions."""
    try:
        # Use FFmpeg to detect high-contrast regions that might contain text
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-i", image_path,
            "-vf", "edgedetect=low=0.1:high=0.4,scale=32:18,format=gray",
            "-frames:v", "1",
            "-f", "rawvideo", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or not result.stdout:
            return "", 0.0, []

        pixels = result.stdout
        # Count bright pixels (edges) - text creates many horizontal edges
        bright_count = sum(1 for p in pixels if p > 128)
        total = len(pixels) if pixels else 1
        edge_ratio = bright_count / total

        # Heuristic: text-heavy frames have moderate edge ratios (0.1 - 0.5)
        # Very high = detailed image; very low = plain background
        has_text_like_content = 0.08 < edge_ratio < 0.55

        if has_text_like_content:
            return "[text-like content detected - install pytesseract for actual OCR]", 0.1, []
        return "", 0.0, []

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Stub OCR analysis failed: %s", exc)
        return "", 0.0, []


# ---------------------------------------------------------------------------
# Text deduplication
# ---------------------------------------------------------------------------
def _text_similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings (0.0-1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _deduplicate_frames(frames: List[OCRFrame], similarity_threshold: float = 0.8) -> List[Dict]:
    """
    Deduplicate OCR text across consecutive frames.

    Merges frames with similar text into unique text entries with
    first_seen, last_seen, and occurrence count.
    """
    unique_texts: List[Dict] = []

    for frame in frames:
        text = frame.text.strip()
        if not text:
            continue

        # Check if this text is similar to the most recent unique entry
        merged = False
        if unique_texts:
            last = unique_texts[-1]
            similarity = _text_similarity(last["text"], text)
            if similarity >= similarity_threshold:
                # Merge: update last_seen and increment count
                last["last_seen"] = frame.timestamp
                last["count"] += 1
                # Keep the longer/more complete version
                if len(text) > len(last["text"]):
                    last["text"] = text
                merged = True

        if not merged:
            # Also check against all previous entries (not just last)
            for entry in unique_texts:
                if _text_similarity(entry["text"], text) >= similarity_threshold:
                    entry["last_seen"] = frame.timestamp
                    entry["count"] += 1
                    merged = True
                    break

        if not merged:
            unique_texts.append({
                "text": text,
                "first_seen": frame.timestamp,
                "last_seen": frame.timestamp,
                "count": 1,
            })

    return unique_texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_text_frames(
    video_path: str,
    interval: float = 1.0,
    use_tesseract: bool = True,
    similarity_threshold: float = 0.8,
    on_progress: Optional[Callable] = None,
) -> OCRResult:
    """
    Extract on-screen text from video frames at regular intervals.

    Samples frames at the specified interval, runs OCR on each frame,
    and deduplicates repeated text across consecutive frames.

    Args:
        video_path: Path to video file.
        interval: Seconds between sampled frames (default 1.0).
        use_tesseract: Try pytesseract if available (True) or always use stub (False).
        similarity_threshold: Text similarity threshold for deduplication (0.0-1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        OCRResult with extracted text and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    interval = max(0.1, float(interval))

    if on_progress:
        on_progress(5, "Getting video info...")

    video_info = get_video_info(video_path)
    duration = video_info.get("duration", 0)

    if duration <= 0:
        raise ValueError("Could not determine video duration")

    # Calculate sample count
    num_samples = max(1, int(duration / interval))
    # Cap at 1000 frames to prevent excessive processing
    if num_samples > 1000:
        interval = duration / 1000
        num_samples = 1000
        logger.info("Capped OCR sampling to 1000 frames (interval=%.2fs)", interval)

    if on_progress:
        on_progress(10, f"Extracting {num_samples} frames at {interval:.1f}s intervals...")

    # Attempt to install pytesseract if requested
    tesseract_available = False
    if use_tesseract:
        tesseract_available = ensure_package("pytesseract", pip_name="pytesseract")
        if tesseract_available:
            ensure_package("PIL", pip_name="Pillow")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_ocr_")
    frames: List[OCRFrame] = []

    try:
        # Extract frames using FFmpeg fps filter for efficiency
        frame_pattern = os.path.join(tmp_dir, "frame_%06d.jpg")
        fps_filter = f"fps=1/{interval}" if interval >= 1.0 else f"fps={1.0/interval}"

        cmd = (
            FFmpegCmd()
            .input(video_path)
            .video_filter(fps_filter)
            .option("q:v", "2")
            .output(frame_pattern)
            .build()
        )

        try:
            run_ffmpeg(cmd, timeout=max(300, int(duration * 2)))
        except RuntimeError as exc:
            raise RuntimeError(f"Frame extraction failed: {exc}")

        if on_progress:
            on_progress(30, "Running OCR on extracted frames...")

        # Process each frame
        frame_files = sorted(
            f for f in os.listdir(tmp_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )

        for i, fname in enumerate(frame_files):
            frame_path = os.path.join(tmp_dir, fname)
            timestamp = i * interval

            if use_tesseract and tesseract_available:
                text, confidence, words = _run_tesseract_ocr(frame_path)
            else:
                text, confidence, words = _run_stub_ocr(frame_path)

            frames.append(OCRFrame(
                timestamp=round(timestamp, 3),
                text=text,
                confidence=round(confidence, 3),
                words=words,
            ))

            if on_progress and (i % max(1, len(frame_files) // 20) == 0):
                pct = 30 + int(60 * (i + 1) / len(frame_files))
                on_progress(pct, f"OCR frame {i + 1}/{len(frame_files)}")

        if on_progress:
            on_progress(92, "Deduplicating text...")

        # Deduplicate
        unique_texts = _deduplicate_frames(frames, similarity_threshold)
        frames_with_text = sum(1 for f in frames if f.text.strip())

        if on_progress:
            on_progress(100, f"Extracted text from {frames_with_text}/{len(frames)} frames")

        return OCRResult(
            frames=frames,
            unique_texts=unique_texts,
            total_frames_sampled=len(frames),
            frames_with_text=frames_with_text,
            duration=duration,
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def search_text_in_video(
    video_path: str,
    query: str,
    interval: float = 1.0,
    case_sensitive: bool = False,
    on_progress: Optional[Callable] = None,
) -> TextSearchResult:
    """
    Search for specific text within a video's on-screen content.

    First extracts text from frames, then searches for the query string.

    Args:
        video_path: Path to video file.
        query: Text string to search for.
        interval: Frame sampling interval in seconds.
        case_sensitive: Whether search is case-sensitive.
        on_progress: Progress callback(pct, msg).

    Returns:
        TextSearchResult with matching timestamps and context.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    query = query.strip()

    def _ocr_progress(pct, msg=""):
        if on_progress:
            # Scale OCR progress to 0-80%
            scaled = int(pct * 0.8)
            on_progress(scaled, msg)

    # Extract text from all frames
    ocr_result = extract_text_frames(
        video_path,
        interval=interval,
        on_progress=_ocr_progress,
    )

    if on_progress:
        on_progress(85, f"Searching for '{query}'...")

    hits: List[TextSearchHit] = []
    search_query = query if case_sensitive else query.lower()

    for frame in ocr_result.frames:
        text = frame.text.strip()
        if not text:
            continue

        compare_text = text if case_sensitive else text.lower()
        if search_query in compare_text:
            # Find context around the match
            idx = compare_text.find(search_query)
            start = max(0, idx - 30)
            end = min(len(text), idx + len(query) + 30)
            context = text[start:end]
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."

            hits.append(TextSearchHit(
                timestamp=frame.timestamp,
                text=query,
                context=context,
                confidence=frame.confidence,
            ))

    if on_progress:
        on_progress(100, f"Found {len(hits)} matches for '{query}'")

    return TextSearchResult(
        query=query,
        hits=hits,
        total_hits=len(hits),
    )
