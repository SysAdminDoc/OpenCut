"""
OpenCut Video-to-Blog-Post Generator

Transcribe video -> LLM structured article with headings + screenshot markers
-> extract frames -> assemble as markdown/HTML + images. Include SEO metadata.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BlogSection:
    """A section of the blog post."""
    heading: str = ""
    content: str = ""
    timestamp: float = 0.0
    screenshot_path: str = ""


@dataclass
class SEOMetadata:
    """SEO metadata for the blog post."""
    title: str = ""
    meta_description: str = ""
    keywords: List[str] = field(default_factory=list)
    slug: str = ""
    reading_time_min: int = 0


@dataclass
class BlogPostResult:
    """Result from video-to-blog generation."""
    title: str = ""
    sections: List[BlogSection] = field(default_factory=list)
    markdown: str = ""
    html: str = ""
    seo: Optional[SEOMetadata] = None
    output_dir: str = ""
    image_paths: List[str] = field(default_factory=list)
    word_count: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract a single frame at the given timestamp."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0, timestamp)),
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed at {timestamp}s")
    return output_path


def _generate_blog_via_llm(transcript_text: str, tone: str = "professional") -> dict:
    """Use LLM to generate a structured blog post from transcript."""
    try:
        from opencut.core.llm import query_llm

        system_prompt = (
            f"You are a {tone} content writer. Convert this video transcript into "
            "a well-structured blog post article.\n\n"
            "Return JSON with this exact structure:\n"
            "{\n"
            '  "title": "Blog post title",\n'
            '  "meta_description": "SEO meta description (150-160 chars)",\n'
            '  "keywords": ["keyword1", "keyword2"],\n'
            '  "sections": [\n'
            '    {"heading": "Section heading", "content": "Section body text...", '
            '"timestamp": <seconds where this topic starts>}\n'
            "  ]\n"
            "}"
        )

        response = query_llm(
            prompt=f"Transcript:\n\n{transcript_text[:12000]}",
            system_prompt=system_prompt,
        )

        json_match = re.search(r"\{[\s\S]*\}", response.text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as exc:
        logger.warning("LLM blog generation failed: %s", exc)

    return {}


def _fallback_blog(transcript_text: str) -> dict:
    """Generate a basic blog structure without LLM."""
    sentences = re.split(r"[.!?]+", transcript_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return {
            "title": "Video Summary",
            "meta_description": "A summary of the video content.",
            "keywords": [],
            "sections": [{"heading": "Content", "content": transcript_text[:2000], "timestamp": 0}],
        }

    # Split into ~3-5 sections
    chunk_size = max(1, len(sentences) // 4)
    sections = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        heading = chunk[0][:80] if chunk else f"Section {len(sections) + 1}"
        content = ". ".join(chunk) + "."
        sections.append({
            "heading": heading,
            "content": content,
            "timestamp": 0,
        })

    title = sentences[0][:100] if sentences else "Video Summary"

    # Extract keywords
    words = re.findall(r"\b[a-zA-Z]{4,}\b", transcript_text.lower())
    from collections import Counter
    word_freq = Counter(words)
    keywords = [w for w, _ in word_freq.most_common(10)]

    return {
        "title": title,
        "meta_description": (sentences[0][:155] + "...") if sentences else "",
        "keywords": keywords,
        "sections": sections,
    }


def _render_markdown(title: str, sections: List[BlogSection], seo: SEOMetadata) -> str:
    """Render the blog post as Markdown."""
    parts = [f"# {title}\n"]

    if seo and seo.meta_description:
        parts.append(f"*{seo.meta_description}*\n")

    if seo and seo.keywords:
        parts.append(f"**Keywords:** {', '.join(seo.keywords)}\n")

    if seo and seo.reading_time_min:
        parts.append(f"**Reading time:** {seo.reading_time_min} min\n")

    parts.append("---\n")

    for section in sections:
        parts.append(f"\n## {section.heading}\n")
        if section.screenshot_path:
            img_name = os.path.basename(section.screenshot_path)
            parts.append(f"\n![{section.heading}](images/{img_name})\n")
        parts.append(f"\n{section.content}\n")

    return "\n".join(parts)


def _render_html(title: str, sections: List[BlogSection], seo: SEOMetadata) -> str:
    """Render the blog post as HTML."""
    def _esc(text):
        return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    parts = ["<!DOCTYPE html>", "<html>", "<head>"]
    parts.append(f"<title>{_esc(title)}</title>")
    if seo and seo.meta_description:
        parts.append(f'<meta name="description" content="{_esc(seo.meta_description)}">')
    if seo and seo.keywords:
        parts.append(f'<meta name="keywords" content="{_esc(", ".join(seo.keywords))}">')
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<article>")
    parts.append(f"<h1>{_esc(title)}</h1>")

    for section in sections:
        parts.append(f"<h2>{_esc(section.heading)}</h2>")
        if section.screenshot_path:
            img_name = os.path.basename(section.screenshot_path)
            parts.append(f'<img src="images/{img_name}" alt="{_esc(section.heading)}">')
        for para in section.content.split("\n\n"):
            para = para.strip()
            if para:
                parts.append(f"<p>{_esc(para)}</p>")

    parts.append("</article>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_blog_post(
    video_path: str,
    tone: str = "professional",
    extract_screenshots: bool = True,
    output_format: str = "both",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> BlogPostResult:
    """
    Generate a blog post from a video.

    Args:
        video_path: Path to the source video.
        tone: Writing tone (professional, casual, technical, educational).
        extract_screenshots: Extract key frame screenshots.
        output_format: Output format: "markdown", "html", or "both".
        output_dir: Output directory. Auto-generated if empty.
        on_progress: Progress callback(pct, msg).

    Returns:
        BlogPostResult with generated content and file paths.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base}_blog")
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if on_progress:
        on_progress(5, "Transcribing video...")

    # Step 1: Transcribe
    transcript_text = ""
    try:
        from opencut.core.captions import transcribe
        from opencut.utils.config import CaptionConfig

        result = transcribe(video_path, config=CaptionConfig(model="base"))
        if hasattr(result, "segments"):
            for seg in result.segments:
                text = seg.text if hasattr(seg, "text") else seg.get("text", "")
                transcript_text += text + " "
        elif isinstance(result, dict):
            for seg in result.get("segments", []):
                transcript_text += seg.get("text", "") + " "
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        transcript_text = "Video content could not be transcribed."

    if on_progress:
        on_progress(30, "Generating blog structure...")

    # Step 2: Generate blog structure
    blog_data = _generate_blog_via_llm(transcript_text, tone)
    if not blog_data or not blog_data.get("sections"):
        blog_data = _fallback_blog(transcript_text)

    if on_progress:
        on_progress(50, "Extracting screenshots...")

    # Step 3: Build sections with screenshots
    sections: List[BlogSection] = []
    image_paths: List[str] = []
    raw_sections = blog_data.get("sections", [])

    for i, sec_data in enumerate(raw_sections):
        heading = str(sec_data.get("heading", f"Section {i + 1}"))
        content = str(sec_data.get("content", ""))
        timestamp = float(sec_data.get("timestamp", 0))

        screenshot_path = ""
        if extract_screenshots and timestamp > 0:
            img_name = f"screenshot_{i + 1:02d}.jpg"
            img_path = os.path.join(images_dir, img_name)
            try:
                _extract_frame(video_path, timestamp, img_path)
                screenshot_path = img_path
                image_paths.append(img_path)
            except Exception as exc:
                logger.warning("Screenshot extraction failed at %.1fs: %s", timestamp, exc)

        sections.append(BlogSection(
            heading=heading,
            content=content,
            timestamp=timestamp,
            screenshot_path=screenshot_path,
        ))

    if on_progress:
        on_progress(70, "Rendering blog post...")

    # Step 4: SEO metadata
    title = str(blog_data.get("title", "Video Summary"))
    word_count = sum(len(s.content.split()) for s in sections)
    reading_time = max(1, word_count // 200)

    seo = SEOMetadata(
        title=title,
        meta_description=str(blog_data.get("meta_description", ""))[:160],
        keywords=blog_data.get("keywords", [])[:15],
        slug=re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80],
        reading_time_min=reading_time,
    )

    # Step 5: Render
    markdown = _render_markdown(title, sections, seo)
    html = _render_html(title, sections, seo)

    # Write files
    if output_format in ("markdown", "both"):
        md_path = os.path.join(output_dir, "blog_post.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

    if output_format in ("html", "both"):
        html_path = os.path.join(output_dir, "blog_post.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    if on_progress:
        on_progress(100, "Blog post generated")

    return BlogPostResult(
        title=title,
        sections=sections,
        markdown=markdown,
        html=html,
        seo=seo,
        output_dir=output_dir,
        image_paths=image_paths,
        word_count=word_count,
    )
