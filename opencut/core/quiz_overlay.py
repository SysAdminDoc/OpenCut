"""
OpenCut Auto-Quiz Overlay (Feature 33.3)

Generate quiz questions from transcript via keyword extraction,
render as overlay cards, and insert at chapter boundaries.

Uses FFmpeg drawtext for rendering. Keyword extraction uses TF-IDF
heuristics (no external NLP libraries required).
"""

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QuizQuestion:
    """A generated quiz question."""
    question: str
    options: List[str]       # 4 multiple-choice options
    correct_index: int       # 0-based index of correct answer
    source_sentence: str     # the sentence this was derived from
    keyword: str             # the key term being tested
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "options": self.options,
            "correct_index": self.correct_index,
            "source_sentence": self.source_sentence,
            "keyword": self.keyword,
            "difficulty": self.difficulty,
        }


@dataclass
class QuizOverlayResult:
    """Result of rendering quiz overlay."""
    output_path: str
    question_count: int
    resolution: Tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "question_count": self.question_count,
            "resolution": list(self.resolution),
        }


@dataclass
class QuizInsertResult:
    """Result of inserting quizzes at chapters."""
    output_path: str
    questions_inserted: int
    chapter_count: int

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "questions_inserted": self.questions_inserted,
            "chapter_count": self.chapter_count,
        }


# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "now", "also", "like", "get", "got", "go", "going",
    "really", "well", "right", "thing", "things", "one", "two",
    "know", "think", "want", "make", "see", "say", "said", "use",
    "used", "way", "even", "much", "still", "many", "new", "first",
    "also", "into", "back", "come", "take", "long", "just", "people",
    "time", "good", "look", "give", "day", "part",
})


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, removing punctuation."""
    text = text.lower()
    # Remove punctuation except hyphens within words
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_keywords(text: str, top_n: int = 20) -> List[Tuple[str, float]]:
    """Extract keywords using TF-IDF-like scoring.

    Computes term frequency, penalizes stop words and very short tokens,
    and boosts multi-word terms that appear as capitalized phrases.

    Returns list of (keyword, score) tuples sorted by score.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Token frequency across all sentences
    all_tokens = _tokenize(text)
    if not all_tokens:
        return []

    tf = Counter(all_tokens)
    total = len(all_tokens)
    num_sentences = len(sentences)

    # Document frequency: how many sentences contain each token
    df = Counter()
    for sent in sentences:
        sent_tokens = set(_tokenize(sent))
        for t in sent_tokens:
            df[t] += 1

    # Score each token: TF * IDF-like, excluding stop words
    scored = {}
    for token, freq in tf.items():
        if token in _STOP_WORDS:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue

        tf_score = freq / total
        # IDF: log(N / df) -- boost rare terms
        idf = math.log((num_sentences + 1) / (df.get(token, 1) + 1)) + 1
        score = tf_score * idf

        # Boost tokens that appear capitalized in original text
        if re.search(r"\b" + re.escape(token[:1].upper() + token[1:]) + r"\b", text):
            score *= 1.5

        scored[token] = score

    # Sort and return top N
    sorted_kw = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return sorted_kw[:top_n]


def _find_sentence_for_keyword(keyword: str, sentences: List[str]) -> str:
    """Find the best sentence containing a keyword."""
    keyword_lower = keyword.lower()
    for sent in sentences:
        if keyword_lower in sent.lower():
            return sent
    return sentences[0] if sentences else ""


def _generate_distractors(keyword: str, all_keywords: List[str], count: int = 3) -> List[str]:
    """Generate plausible distractor options for a quiz question.

    Uses other keywords from the text as distractors.
    """
    distractors = []
    for kw in all_keywords:
        if kw.lower() != keyword.lower() and len(distractors) < count:
            distractors.append(kw)

    # Pad with generic distractors if needed
    generic = ["all of the above", "none of the above", "not applicable"]
    while len(distractors) < count:
        for g in generic:
            if g not in distractors and len(distractors) < count:
                distractors.append(g)

    return distractors[:count]


# ---------------------------------------------------------------------------
# Quiz question generation
# ---------------------------------------------------------------------------

def generate_quiz_questions(
    transcript: str,
    count: int = 5,
    difficulty: str = "medium",
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Generate quiz questions from a transcript.

    Extracts keywords via TF-IDF scoring, finds sentences containing
    those keywords, and generates fill-in-the-blank style questions
    with multiple choice options.

    Args:
        transcript: Full text transcript of the video.
        count: Number of questions to generate.
        difficulty: Difficulty level ("easy", "medium", "hard").
            Easy: shorter questions, more context.
            Hard: less context, more distractors.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of QuizQuestion dicts.

    Raises:
        ValueError: If transcript is too short or count < 1.
    """
    if not transcript or len(transcript.strip()) < 50:
        raise ValueError("Transcript is too short to generate questions (minimum 50 characters)")
    if count < 1:
        raise ValueError(f"Question count must be >= 1, got {count}")
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Invalid difficulty: {difficulty!r}. Valid: easy, medium, hard")

    if on_progress:
        on_progress(10, "Extracting keywords...")

    sentences = _split_sentences(transcript)
    keywords_scored = _extract_keywords(transcript, top_n=count * 3)

    if not keywords_scored:
        raise ValueError("Could not extract keywords from transcript")

    all_keyword_strings = [kw for kw, _ in keywords_scored]

    if on_progress:
        on_progress(40, f"Generating {count} questions...")

    questions = []
    used_keywords = set()

    for kw, score in keywords_scored:
        if len(questions) >= count:
            break
        if kw in used_keywords:
            continue

        source = _find_sentence_for_keyword(kw, sentences)
        if not source:
            continue

        # Create fill-in-the-blank question
        # Replace keyword in sentence with "______"
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        blanked = pattern.sub("______", source, count=1)

        if difficulty == "easy":
            question_text = f"Fill in the blank: {blanked}"
        elif difficulty == "hard":
            # Less context: truncate long questions
            if len(blanked) > 100:
                # Find the blank and keep surrounding context
                blank_pos = blanked.find("______")
                start = max(0, blank_pos - 40)
                end = min(len(blanked), blank_pos + 50)
                blanked = "..." + blanked[start:end] + "..."
            question_text = f"What term completes: {blanked}"
        else:
            question_text = f"Complete the following: {blanked}"

        # Generate options
        distractors = _generate_distractors(kw, all_keyword_strings)
        correct_answer = kw

        # Shuffle options: place correct answer at a deterministic position
        import hashlib
        hash_val = int(hashlib.md5(kw.encode()).hexdigest()[:8], 16)
        correct_idx = hash_val % 4

        options = list(distractors[:3])
        options.insert(correct_idx, correct_answer)
        # Ensure exactly 4 options
        options = options[:4]
        while len(options) < 4:
            options.append("not applicable")

        questions.append(QuizQuestion(
            question=question_text,
            options=options,
            correct_index=correct_idx,
            source_sentence=source,
            keyword=kw,
            difficulty=difficulty,
        ))
        used_keywords.add(kw)

    if on_progress:
        on_progress(100, "Done")

    return [q.to_dict() for q in questions]


# ---------------------------------------------------------------------------
# Quiz card rendering
# ---------------------------------------------------------------------------

def _escape_drawtext(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    # Truncate extremely long text
    if len(text) > 120:
        text = text[:117] + "..."
    return text


def render_quiz_overlay(
    question_data: List[dict],
    output_path_str: str,
    resolution: Tuple[int, int] = (1920, 1080),
    display_duration: float = 10.0,
    bg_color: str = "black",
    text_color: str = "white",
    highlight_color: str = "00FF00",
    font_size: int = 32,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Render quiz questions as an overlay video (transparent or solid).

    Creates a video with quiz cards shown sequentially, suitable for
    overlaying onto or concatenating with the source video.

    Args:
        question_data: List of QuizQuestion dicts.
        output_path_str: Output video file path.
        resolution: Output resolution (width, height).
        display_duration: Seconds each question is displayed.
        bg_color: Background color.
        text_color: Text color.
        highlight_color: Color for correct answer highlight.
        font_size: Base font size.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, question_count, resolution.

    Raises:
        ValueError: If question_data is empty.
    """
    if not question_data:
        raise ValueError("No question data provided")

    out_w, out_h = resolution
    n = len(question_data)
    total_duration = display_duration * n

    if on_progress:
        on_progress(10, f"Building overlay for {n} questions...")

    # Build filter_complex: generate a black background video,
    # then overlay drawtext for each question card at its time slot
    filters = []

    # Create color source for background
    bg_input = (
        f"color=c={bg_color}:s={out_w}x{out_h}:d={total_duration}:r=30[bg]"
    )
    filters.append(bg_input)

    # Add drawtext for each question
    current_label = "bg"
    for qi, qd in enumerate(question_data):
        if on_progress:
            pct = 10 + int((qi / n) * 60)
            on_progress(pct, f"Rendering question {qi + 1}/{n}...")

        t_start = qi * display_duration
        t_end = t_start + display_duration
        enable = f"between(t,{t_start:.3f},{t_end:.3f})"

        q_text = _escape_drawtext(qd.get("question", f"Question {qi + 1}"))
        options = qd.get("options", [])
        correct_idx = qd.get("correct_index", 0)

        out_label = f"q{qi}"

        # Question text (centered, upper area)
        q_filter = (
            f"[{current_label}]drawtext=text='{q_text}'"
            f":fontsize={font_size}:fontcolor={text_color}"
            f":x=(w-tw)/2:y=h*0.15"
            f":enable='{enable}'"
        )

        # Option texts (stacked vertically)
        for oi, opt in enumerate(options[:4]):
            opt_text = _escape_drawtext(f"{'ABCD'[oi]}) {opt}")
            y_pos = f"h*{0.35 + oi * 0.12:.2f}"

            # Highlight correct answer
            if oi == correct_idx:
                opt_color = highlight_color
            else:
                opt_color = text_color

            q_filter += (
                f",drawtext=text='{opt_text}'"
                f":fontsize={int(font_size * 0.85)}:fontcolor={opt_color}"
                f":x=w*0.15:y={y_pos}"
                f":enable='{enable}'"
            )

        q_filter += f"[{out_label}]"
        filters.append(q_filter)
        current_label = out_label

    filter_complex = ";".join(filters)

    if on_progress:
        on_progress(75, "Encoding quiz overlay video...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path_str)), exist_ok=True)

    cmd = [
        get_ffmpeg_path(),
        "-hide_banner", "-y",
        "-filter_complex", filter_complex,
        "-map", f"[{current_label}]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path_str,
    ]

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Done")

    result = QuizOverlayResult(
        output_path=output_path_str,
        question_count=n,
        resolution=resolution,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Insert quizzes at chapter boundaries
# ---------------------------------------------------------------------------

def insert_quiz_at_chapters(
    video_path: str,
    questions: List[dict],
    chapters: List[dict],
    output_path_str: str,
    quiz_duration: float = 10.0,
    position: str = "after",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Insert quiz overlay cards at chapter boundaries in a video.

    For each chapter, renders the corresponding quiz question as a
    drawtext overlay that appears at the chapter's end (or start).

    Args:
        video_path: Source video file path.
        questions: List of QuizQuestion dicts.
        chapters: List of chapter dicts with start_time, end_time.
        output_path_str: Output file path.
        quiz_duration: How long each quiz card is shown (seconds).
        position: Where to show quiz - "after" chapter end,
                  "before" chapter start, or "overlay" on top.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, questions_inserted, chapter_count.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If questions or chapters are empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not questions:
        raise ValueError("No questions provided")
    if not chapters:
        raise ValueError("No chapters provided")

    valid_positions = ("after", "before", "overlay")
    if position not in valid_positions:
        raise ValueError(f"Invalid position: {position!r}. Valid: {', '.join(valid_positions)}")

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    _vid_w, _vid_h = info["width"], info["height"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, "Building quiz overlays...")

    # Sort chapters by start time
    sorted_chapters = sorted(chapters, key=lambda c: c.get("start_time", 0))

    # Pair questions with chapters (cycle if fewer questions than chapters)
    filters = []
    inserted = 0

    for ci, chapter in enumerate(sorted_chapters):
        if ci >= len(questions):
            break

        qd = questions[ci]
        q_text = _escape_drawtext(qd.get("question", f"Quiz {ci + 1}"))
        options = qd.get("options", [])
        correct_idx = qd.get("correct_index", 0)

        # Determine display time range
        if position == "after":
            t_start = chapter.get("end_time", 0)
            # Clamp to video duration
            if duration > 0 and t_start > duration:
                t_start = max(0, duration - quiz_duration)
        elif position == "before":
            t_start = max(0, chapter.get("start_time", 0) - quiz_duration)
        else:  # overlay
            t_start = chapter.get("end_time", 0) - quiz_duration
            t_start = max(0, t_start)

        t_end = t_start + quiz_duration
        if duration > 0:
            t_end = min(t_end, duration)

        enable = f"between(t,{t_start:.3f},{t_end:.3f})"

        # Semi-transparent background box
        filters.append(
            f"drawbox=x=w*0.05:y=h*0.1:w=w*0.9:h=h*0.8"
            f":color=black@0.75:t=fill:enable='{enable}'"
        )

        # Question text
        filters.append(
            f"drawtext=text='{q_text}'"
            f":fontsize=28:fontcolor=white"
            f":x=(w-tw)/2:y=h*0.2"
            f":enable='{enable}'"
        )

        # Options
        for oi, opt in enumerate(options[:4]):
            opt_text = _escape_drawtext(f"{'ABCD'[oi]}) {opt}")
            y_frac = 0.38 + oi * 0.12
            color = "00FF00" if oi == correct_idx else "white"
            filters.append(
                f"drawtext=text='{opt_text}'"
                f":fontsize=24:fontcolor={color}"
                f":x=w*0.15:y=h*{y_frac:.2f}"
                f":enable='{enable}'"
            )

        inserted += 1

    if not filters:
        raise ValueError("No quizzes could be placed at the given chapters")

    vf_chain = ",".join(filters)

    if on_progress:
        on_progress(40, "Encoding video with quiz overlays...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path_str)), exist_ok=True)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf_chain)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output_path_str)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Done")

    result = QuizInsertResult(
        output_path=output_path_str,
        questions_inserted=inserted,
        chapter_count=len(sorted_chapters),
    )
    return result.to_dict()
