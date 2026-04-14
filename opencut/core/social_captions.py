"""
OpenCut Auto-Hashtag & Caption Generator

Generates platform-optimized titles, descriptions, hashtags, and tags
from a transcript using the configured LLM provider.  Falls back to
simple TF-IDF-style keyword extraction when no LLM is available.
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SocialCaptionResult:
    """Generated social media captions for a single platform."""
    title: str = ""
    description: str = ""
    hashtags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    platform: str = ""


# ---------------------------------------------------------------------------
# Platform prompt templates
# ---------------------------------------------------------------------------
_PLATFORM_PROMPTS = {
    "youtube": (
        "You are a YouTube SEO expert. Given the following video transcript, generate:\n"
        "1. An SEO-optimized title (max 100 chars, keyword-rich)\n"
        "2. A keyword-rich description with timestamps if the transcript mentions sections "
        "(max 5000 chars). Include relevant keywords naturally.\n"
        "3. A list of 10-15 relevant tags for YouTube search\n"
        "4. 3-5 hashtags to include in the description\n\n"
        "Format your response EXACTLY as JSON:\n"
        '{{"title": "...", "description": "...", "tags": ["..."], "hashtags": ["..."]}}'
    ),
    "tiktok": (
        "You are a TikTok content strategist. Given the following video transcript, generate:\n"
        "1. A short, catchy caption (max 150 chars) that hooks viewers\n"
        "2. 5-8 trending and relevant hashtags (include a mix of popular and niche)\n"
        "3. 3-5 keyword tags\n\n"
        "Format your response EXACTLY as JSON:\n"
        '{{"title": "...", "description": "...", "tags": ["..."], "hashtags": ["..."]}}'
    ),
    "instagram": (
        "You are an Instagram content creator. Given the following video transcript, generate:\n"
        "1. An engaging title/hook (first line of caption)\n"
        "2. An engaging caption with emojis, line breaks, and a call to action "
        "(max 2200 chars)\n"
        "3. Up to 30 relevant hashtags (mix of popular, medium, and niche)\n"
        "4. 5-10 keyword tags\n\n"
        "Format your response EXACTLY as JSON:\n"
        '{{"title": "...", "description": "...", "tags": ["..."], "hashtags": ["..."]}}'
    ),
    "twitter": (
        "You are a Twitter/X content strategist. Given the following video transcript, generate:\n"
        "1. A concise, attention-grabbing hook (the tweet text, max 250 chars to leave room for hashtags)\n"
        "2. 2-3 relevant hashtags (total tweet must be under 280 chars)\n"
        "3. 2-3 keyword tags\n\n"
        "Format your response EXACTLY as JSON:\n"
        '{{"title": "...", "description": "...", "tags": ["..."], "hashtags": ["..."]}}'
    ),
}

# Stop words for keyword extraction fallback
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it that this was were be been "
    "being have has had do does did will would shall should may might can could "
    "i me my we our you your he him his she her they them their its not no nor "
    "so if then than too very just about also back been before between both but "
    "by came come could each from get got had has have her here him his how into "
    "its like make many me might more most much must my no now of only or other "
    "our out over said same see she should show side since so some still such "
    "take tell than that the their them then there these they thing this those "
    "through time to together too under up us use very want was way we well were "
    "what when where which while who why will with within without would".split()
)


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction fallback
# ---------------------------------------------------------------------------
def _extract_keywords_tfidf(text: str, top_n: int = 20) -> List[str]:
    """
    Extract keywords from *text* using a simplified TF-IDF approach.

    Computes term frequency and inverse-document frequency treating
    each sentence as a mini-document.  Returns up to *top_n* keywords
    sorted by TF-IDF score.
    """
    # Tokenise
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    words = [w for w in words if w not in _STOP_WORDS]

    if not words:
        return []

    # Term frequency across entire text
    tf = Counter(words)
    total_words = len(words)

    # Split into sentences for IDF
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_docs = max(len(sentences), 1)

    # Document frequency
    df: Counter = Counter()
    for sent in sentences:
        sent_words = set(re.findall(r"[a-zA-Z]{3,}", sent.lower()))
        for w in sent_words:
            if w not in _STOP_WORDS:
                df[w] += 1

    # TF-IDF scoring
    scores = {}
    for word, count in tf.items():
        word_tf = count / total_words
        word_idf = math.log(1 + num_docs / (1 + df.get(word, 0)))
        scores[word] = word_tf * word_idf

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_n]


def _fallback_captions(
    transcript_text: str,
    platform: str,
) -> SocialCaptionResult:
    """Generate captions without an LLM using keyword extraction."""
    keywords = _extract_keywords_tfidf(transcript_text, top_n=30)

    # Build title from top 5 keywords
    title_words = keywords[:5]
    title = " ".join(w.capitalize() for w in title_words)

    # Platform-specific sizing
    if platform == "twitter":
        hashtags = [f"#{w}" for w in keywords[:3]]
        tags = keywords[:3]
        description = transcript_text[:200].strip()
        # Ensure total < 280
        tweet = f"{title} {' '.join(hashtags)}"
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        description = tweet
    elif platform == "tiktok":
        hashtags = [f"#{w}" for w in keywords[:8]]
        tags = keywords[:5]
        description = title
    elif platform == "instagram":
        hashtags = [f"#{w}" for w in keywords[:30]]
        tags = keywords[:10]
        description = transcript_text[:500].strip()
    else:
        # youtube default
        hashtags = [f"#{w}" for w in keywords[:5]]
        tags = keywords[:15]
        description = transcript_text[:2000].strip()

    return SocialCaptionResult(
        title=title[:100],
        description=description,
        hashtags=hashtags,
        tags=tags,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_social_captions(
    transcript_text: str,
    platform: str = "youtube",
    custom_instructions: str = "",
    on_progress: Optional[Callable] = None,
) -> SocialCaptionResult:
    """
    Generate social-media-ready captions from a transcript.

    Uses the configured LLM when available; otherwise falls back to
    keyword extraction.

    Args:
        transcript_text: The video transcript text.
        platform: Target platform (youtube, tiktok, instagram, twitter).
        custom_instructions: Extra instructions appended to the LLM prompt.
        on_progress: Progress callback(pct, msg).

    Returns:
        SocialCaptionResult with title, description, hashtags, tags.
    """
    platform = platform.lower().strip()
    if platform not in _PLATFORM_PROMPTS:
        platform = "youtube"

    if not transcript_text or not transcript_text.strip():
        return SocialCaptionResult(platform=platform)

    if on_progress:
        on_progress(10, f"Generating {platform} captions...")

    # Try LLM first
    try:
        from opencut.core.llm import LLMConfig, query_llm

        system_prompt = _PLATFORM_PROMPTS[platform]
        if custom_instructions:
            system_prompt += f"\n\nAdditional instructions: {custom_instructions}"

        user_prompt = f"Transcript:\n\n{transcript_text[:8000]}"

        if on_progress:
            on_progress(30, "Querying LLM...")

        config = LLMConfig()
        response = query_llm(
            prompt=user_prompt,
            config=config,
            system_prompt=system_prompt,
        )

        if on_progress:
            on_progress(70, "Parsing LLM response...")

        # Check if response contains an error indicator
        if response.text.startswith("LLM error:") or response.text.startswith("Unknown LLM provider"):
            raise RuntimeError(response.text)

        # Parse JSON from the LLM response
        result = _parse_llm_response(response.text, platform)

        if on_progress:
            on_progress(100, "Captions generated")

        return result

    except Exception as exc:
        logger.warning("LLM unavailable for caption generation, using fallback: %s", exc)
        if on_progress:
            on_progress(50, "LLM unavailable, using keyword extraction...")

        result = _fallback_captions(transcript_text, platform)

        if on_progress:
            on_progress(100, "Captions generated (fallback)")

        return result


def _parse_llm_response(text: str, platform: str) -> SocialCaptionResult:
    """Parse the LLM JSON response into a SocialCaptionResult."""
    # Try to find JSON in the response
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        raise ValueError("No JSON found in LLM response")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in LLM response")

    title = str(data.get("title", ""))
    description = str(data.get("description", ""))
    tags = data.get("tags", [])
    hashtags = data.get("hashtags", [])

    # Ensure hashtags have # prefix
    hashtags = [
        h if h.startswith("#") else f"#{h}"
        for h in hashtags
        if isinstance(h, str) and h.strip()
    ]

    # Ensure tags are plain strings (no #)
    tags = [
        t.lstrip("#").strip()
        for t in tags
        if isinstance(t, str) and t.strip()
    ]

    # Platform-specific constraints
    if platform == "twitter":
        combined = f"{title} {' '.join(hashtags)}"
        if len(combined) > 280:
            # Trim hashtags
            while hashtags and len(f"{title} {' '.join(hashtags)}") > 280:
                hashtags.pop()

    if platform == "youtube":
        title = title[:100]
        description = description[:5000]

    return SocialCaptionResult(
        title=title,
        description=description,
        hashtags=hashtags,
        tags=tags,
        platform=platform,
    )


# ---------------------------------------------------------------------------
# Platform Caption Generator (58.3)
# ---------------------------------------------------------------------------
@dataclass
class PlatformCaption:
    """A caption tailored for a specific platform."""
    platform: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    char_count: int = 0
    tone: str = ""


_PLATFORM_CHAR_LIMITS = {
    "twitter": 280,
    "instagram": 2200,
    "linkedin": 3000,
    "tiktok": 300,
    "youtube": 5000,
    "facebook": 63206,
}

_TONE_TEMPLATES = {
    "professional": {
        "twitter": "Key insight: {summary}\n\n{hashtags}",
        "instagram": "{hook}\n\n{body}\n\n{cta}\n\n{hashtags}",
        "linkedin": "{hook}\n\n{body}\n\n{cta}\n\n{hashtags}",
        "tiktok": "{hook} {hashtags}",
    },
    "casual": {
        "twitter": "{hook} {hashtags}",
        "instagram": "{hook}\n\n{body}\n\nThoughts? Drop a comment!\n\n{hashtags}",
        "linkedin": "{hook}\n\n{body}\n\n{hashtags}",
        "tiktok": "{hook} {hashtags}",
    },
    "educational": {
        "twitter": "Did you know? {summary}\n\n{hashtags}",
        "instagram": "Here's what you need to know:\n\n{body}\n\nSave this for later!\n\n{hashtags}",
        "linkedin": "A lesson worth sharing:\n\n{body}\n\n{cta}\n\n{hashtags}",
        "tiktok": "Learn this: {hook} {hashtags}",
    },
    "trendy": {
        "twitter": "{hook} {hashtags}",
        "instagram": "{hook}\n\n{body}\n\nLink in bio!\n\n{hashtags}",
        "linkedin": "{hook}\n\n{body}\n\n{hashtags}",
        "tiktok": "{hook} {hashtags}",
    },
}


def _generate_caption_via_llm(transcript: str, platform: str, tone: str) -> str:
    """Generate a platform caption using LLM."""
    try:
        from opencut.core.llm import LLMConfig, query_llm

        char_limit = _PLATFORM_CHAR_LIMITS.get(platform, 2000)
        system_prompt = (
            f"You are a {tone} social media copywriter for {platform}. "
            f"Write a caption (max {char_limit} chars) for this content. "
        )

        if platform == "twitter":
            system_prompt += "Be concise and punchy. Under 280 chars total including hashtags."
        elif platform == "instagram":
            system_prompt += "Use a hook first line, body with line breaks, and call to action."
        elif platform == "linkedin":
            system_prompt += "Professional and insightful. Use line breaks for readability."
        elif platform == "tiktok":
            system_prompt += "Trendy, use emojis, keep it short and catchy."

        system_prompt += "\nReturn just the caption text, nothing else."

        response = query_llm(
            prompt=f"Content:\n\n{transcript[:5000]}",
            system_prompt=system_prompt,
        )

        if response.text and not response.text.startswith("LLM error:"):
            return response.text.strip()
    except Exception as exc:
        logger.warning("LLM caption generation failed: %s", exc)

    return ""


def _build_caption_from_template(transcript: str, platform: str, tone: str) -> str:
    """Build a caption using templates when LLM is unavailable."""
    # Extract a hook (first meaningful sentence)
    sentences = re.split(r"[.!?]+", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    hook = sentences[0][:100] if sentences else "Check this out"
    summary = sentences[0][:200] if sentences else transcript[:200]
    body = ". ".join(sentences[:3])[:500] if sentences else transcript[:500]
    cta = "What do you think?"

    # Extract keywords for hashtags
    keywords = _extract_keywords_tfidf(transcript, top_n=8)
    hashtag_str = " ".join(f"#{w}" for w in keywords[:5])

    templates = _TONE_TEMPLATES.get(tone, _TONE_TEMPLATES["professional"])
    template = templates.get(platform, "{hook}\n\n{body}\n\n{hashtags}")

    caption = template.format(
        hook=hook,
        summary=summary,
        body=body,
        cta=cta,
        hashtags=hashtag_str,
    )

    # Trim to platform limit
    limit = _PLATFORM_CHAR_LIMITS.get(platform, 2000)
    if len(caption) > limit:
        caption = caption[:limit - 3] + "..."

    return caption


def generate_platform_caption(
    transcript: str,
    platform: str = "twitter",
    tone: str = "professional",
    custom_hashtags: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> PlatformCaption:
    """
    Generate a platform-specific caption from transcript text.

    Per-platform templates:
      - Twitter: <=280 chars, concise
      - Instagram: hook + body + hashtags
      - LinkedIn: professional tone
      - TikTok: trendy with emojis

    Args:
        transcript: Source transcript text.
        platform: Target platform.
        tone: Caption tone (professional, casual, educational, trendy).
        custom_hashtags: Additional hashtags to include.
        on_progress: Progress callback(pct, msg).

    Returns:
        PlatformCaption with caption text and metadata.
    """
    platform = platform.lower().strip()
    if platform not in _PLATFORM_CHAR_LIMITS:
        platform = "twitter"

    tone = tone.lower().strip()
    if tone not in _TONE_TEMPLATES:
        tone = "professional"

    if not transcript or not transcript.strip():
        return PlatformCaption(platform=platform, tone=tone)

    if on_progress:
        on_progress(10, f"Generating {platform} caption ({tone} tone)...")

    # Try LLM first
    caption = _generate_caption_via_llm(transcript, platform, tone)

    if not caption:
        if on_progress:
            on_progress(40, "Using template fallback...")
        caption = _build_caption_from_template(transcript, platform, tone)

    # Extract/add hashtags
    keywords = _extract_keywords_tfidf(transcript, top_n=8)
    hashtags = [f"#{w}" for w in keywords[:5]]

    if custom_hashtags:
        for h in custom_hashtags:
            tag = h if h.startswith("#") else f"#{h}"
            if tag not in hashtags:
                hashtags.append(tag)

    # Enforce char limit
    limit = _PLATFORM_CHAR_LIMITS.get(platform, 2000)
    if len(caption) > limit:
        caption = caption[:limit - 3] + "..."

    if on_progress:
        on_progress(100, "Caption generated")

    return PlatformCaption(
        platform=platform,
        caption=caption,
        hashtags=hashtags,
        char_count=len(caption),
        tone=tone,
    )
