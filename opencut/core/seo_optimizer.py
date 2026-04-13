"""
OpenCut AI SEO Optimizer

Generates optimized titles, descriptions, tags, and hashtags
for video content using LLM or heuristic fallback.

Supports YouTube, TikTok, Instagram, and Twitter/X platforms.
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Platform constraints & templates
# ---------------------------------------------------------------------------
PLATFORM_LIMITS: Dict[str, Dict] = {
    "youtube": {
        "title_max": 100,
        "description_max": 5000,
        "tags_max": 500,   # total characters in tags
        "tag_count_max": 30,
        "hashtag_count": 3,
        "title_ideal_min": 40,
        "title_ideal_max": 70,
    },
    "tiktok": {
        "title_max": 150,
        "description_max": 2200,
        "tags_max": 300,
        "tag_count_max": 15,
        "hashtag_count": 5,
        "title_ideal_min": 20,
        "title_ideal_max": 80,
    },
    "instagram": {
        "title_max": 0,     # Instagram uses description/caption
        "description_max": 2200,
        "tags_max": 300,
        "tag_count_max": 30,
        "hashtag_count": 10,
        "title_ideal_min": 0,
        "title_ideal_max": 0,
    },
    "twitter": {
        "title_max": 280,
        "description_max": 280,
        "tags_max": 200,
        "tag_count_max": 5,
        "hashtag_count": 3,
        "title_ideal_min": 20,
        "title_ideal_max": 100,
    },
}

# Common English stop words for keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "that", "this", "these",
    "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "what", "which",
    "who", "whom", "how", "when", "where", "why", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "because", "as", "until",
    "while", "each", "all", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "also", "well", "really",
    "like", "going", "gonna", "got", "get", "make", "thing", "things",
    "know", "think", "want", "say", "said", "one", "two", "much", "many",
})

# Optimal posting time suggestions by platform (UTC)
_OPTIMAL_POSTING_TIMES: Dict[str, str] = {
    "youtube": "Tuesday-Thursday 2:00-4:00 PM EST (peak engagement)",
    "tiktok": "Tuesday-Thursday 10:00 AM - 12:00 PM EST",
    "instagram": "Monday-Friday 11:00 AM - 1:00 PM EST",
    "twitter": "Monday-Friday 8:00 - 10:00 AM EST",
}


@dataclass
class SEOResult:
    """SEO optimization results."""
    title_options: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    optimal_posting_time: str = ""
    platform: str = "youtube"
    method: str = "heuristic"  # "llm" or "heuristic"


def optimize_seo(
    transcript_text: str,
    platform: str = "youtube",
    custom_context: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate SEO-optimized titles, descriptions, tags, and hashtags.

    Attempts LLM-based optimization first, falls back to heuristic
    keyword extraction and template-based generation.

    Args:
        transcript_text: Full transcript or description of video content.
        platform: Target platform ("youtube", "tiktok", "instagram", "twitter").
        custom_context: Additional context or instructions for optimization.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with title_options, description, tags, hashtags, optimal_posting_time.
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("transcript_text is required for SEO optimization")

    platform = platform.lower().strip()
    if platform not in PLATFORM_LIMITS:
        platform = "youtube"

    if on_progress:
        on_progress(10, f"Optimizing SEO for {platform}...")

    # Try LLM first
    result = _try_llm_optimization(transcript_text, platform, custom_context, on_progress)

    if result is None:
        # Fallback to heuristic
        if on_progress:
            on_progress(40, "LLM unavailable, using keyword extraction...")
        result = _heuristic_optimization(transcript_text, platform, custom_context, on_progress)

    result.optimal_posting_time = _OPTIMAL_POSTING_TIMES.get(platform, "Varies by audience timezone")
    result.platform = platform

    if on_progress:
        on_progress(100, f"SEO optimization complete ({result.method})")

    return _result_to_dict(result)


def score_title(title: str, platform: str = "youtube") -> dict:
    """
    Score a title for SEO effectiveness.

    Args:
        title: Title string to evaluate.
        platform: Target platform.

    Returns:
        dict with length_ok, keyword_count, estimated_ctr_tier, suggestions.
    """
    platform = platform.lower().strip()
    if platform not in PLATFORM_LIMITS:
        platform = "youtube"

    limits = PLATFORM_LIMITS[platform]
    length = len(title)

    length_ok = length <= limits["title_max"]
    ideal_length = limits["title_ideal_min"] <= length <= limits["title_ideal_max"]

    # Count meaningful words (potential keywords)
    words = _tokenize(title)
    keyword_count = len([w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2])

    # Power words that tend to boost CTR
    power_words = {
        "how", "why", "best", "top", "ultimate", "guide", "secret", "tips",
        "tricks", "easy", "fast", "simple", "proven", "amazing", "incredible",
        "complete", "free", "new", "review", "tutorial", "vs",
    }
    has_power_word = any(w.lower() in power_words for w in words)

    # Number in title (tends to boost CTR)
    has_number = bool(re.search(r"\d+", title))

    # Estimate CTR tier
    score = 0
    if ideal_length:
        score += 2
    elif length_ok:
        score += 1
    if keyword_count >= 3:
        score += 2
    elif keyword_count >= 2:
        score += 1
    if has_power_word:
        score += 2
    if has_number:
        score += 1
    if title[0].isupper():
        score += 1

    if score >= 6:
        ctr_tier = "high"
    elif score >= 4:
        ctr_tier = "medium"
    else:
        ctr_tier = "low"

    suggestions = []
    if not length_ok:
        suggestions.append(f"Title exceeds {limits['title_max']} chars for {platform}")
    if not ideal_length and length_ok:
        suggestions.append(
            f"Ideal length for {platform} is {limits['title_ideal_min']}-{limits['title_ideal_max']} chars"
        )
    if not has_number:
        suggestions.append("Consider adding a number (e.g., '5 Tips', '3 Ways')")
    if not has_power_word:
        suggestions.append("Consider a power word (e.g., 'Ultimate', 'Complete', 'Best')")
    if keyword_count < 2:
        suggestions.append("Add more descriptive keywords to the title")

    return {
        "length_ok": length_ok,
        "ideal_length": ideal_length,
        "character_count": length,
        "keyword_count": keyword_count,
        "has_power_word": has_power_word,
        "has_number": has_number,
        "estimated_ctr_tier": ctr_tier,
        "score": score,
        "suggestions": suggestions,
    }


def _try_llm_optimization(
    transcript_text: str,
    platform: str,
    custom_context: str,
    on_progress: Optional[Callable],
) -> Optional[SEOResult]:
    """Try LLM-based SEO optimization. Returns None on failure."""
    try:
        from opencut.core.llm import LLMConfig, check_llm_reachable, query_llm

        config = LLMConfig()
        reachable = check_llm_reachable(config)
        if not reachable.get("available"):
            return None

        if on_progress:
            on_progress(20, "Querying LLM for SEO suggestions...")

        limits = PLATFORM_LIMITS[platform]

        # Truncate transcript to avoid token limits
        max_transcript = 3000
        truncated = transcript_text[:max_transcript]
        if len(transcript_text) > max_transcript:
            truncated += "... [truncated]"

        context_part = f"\nAdditional context: {custom_context}" if custom_context else ""

        prompt = f"""Analyze this video transcript and generate SEO-optimized content for {platform}.

Transcript:
{truncated}
{context_part}

Respond in STRICT JSON format with these exact keys:
{{
  "title_options": ["title1", "title2", "title3", "title4", "title5"],
  "description": "SEO-optimized description (2-3 paragraphs)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"],
  "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"]
}}

Rules:
- Titles should be {limits['title_ideal_min']}-{limits['title_ideal_max']} characters
- Include numbers and power words in at least 2 titles
- Tags should be specific and relevant keywords
- Description should include key topics naturally
- Hashtags should be trending-style for {platform}"""

        system_prompt = (
            "You are an expert social media SEO specialist. "
            "Always respond with valid JSON only, no markdown or extra text."
        )

        response = query_llm(prompt, config=config, system_prompt=system_prompt)

        if on_progress:
            on_progress(70, "Parsing LLM response...")

        # Parse JSON from response
        text = response.text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)

        result = SEOResult(
            title_options=data.get("title_options", [])[:5],
            description=data.get("description", ""),
            tags=data.get("tags", [])[:limits["tag_count_max"]],
            hashtags=data.get("hashtags", [])[:limits["hashtag_count"]],
            method="llm",
        )

        # Validate we got reasonable results
        if not result.title_options or not result.tags:
            return None

        return result

    except Exception as e:
        logger.debug("LLM SEO optimization failed: %s", e)
        return None


def _heuristic_optimization(
    transcript_text: str,
    platform: str,
    custom_context: str,
    on_progress: Optional[Callable],
) -> SEOResult:
    """Fallback heuristic SEO optimization using TF-IDF keyword extraction."""
    limits = PLATFORM_LIMITS[platform]

    if on_progress:
        on_progress(50, "Extracting keywords...")

    # Extract keywords using TF-IDF-like scoring
    keywords = _extract_keywords(transcript_text, max_keywords=20)
    top_keywords = keywords[:10]

    if on_progress:
        on_progress(65, "Generating titles...")

    # Generate title options from top keywords
    title_options = _generate_titles(top_keywords, limits)

    if on_progress:
        on_progress(75, "Building description and tags...")

    # Build description
    description = _generate_description(transcript_text, top_keywords, platform)

    # Tags from keywords
    tags = [kw for kw, _ in keywords[:limits["tag_count_max"]]]

    # Hashtags from top keywords
    hashtags = [
        f"#{kw.replace(' ', '')}" for kw, _ in keywords[:limits["hashtag_count"]]
    ]

    # Merge custom context keywords
    if custom_context:
        context_kw = _extract_keywords(custom_context, max_keywords=5)
        for kw, score in context_kw:
            if kw not in tags:
                tags.append(kw)

    return SEOResult(
        title_options=title_options,
        description=description,
        tags=tags[:limits["tag_count_max"]],
        hashtags=hashtags,
        method="heuristic",
    )


def _tokenize(text: str) -> List[str]:
    """Split text into words, stripping punctuation."""
    return re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b", text.lower())


def _extract_keywords(text: str, max_keywords: int = 20) -> List[tuple]:
    """Extract keywords using term frequency with stop-word filtering."""
    words = _tokenize(text)
    # Filter stop words and short words
    filtered = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    if not filtered:
        return []

    # Term frequency
    tf = Counter(filtered)
    total = len(filtered)

    # Score by frequency, boosted for longer words (proxy for specificity)
    scored = []
    for word, count in tf.items():
        freq = count / total
        length_boost = min(1.5, len(word) / 5.0)
        score = freq * length_boost
        scored.append((word, score))

    # Also extract bigrams
    bigrams = []
    for i in range(len(filtered) - 1):
        bigram = f"{filtered[i]} {filtered[i + 1]}"
        bigrams.append(bigram)

    bigram_counts = Counter(bigrams)
    for bigram, count in bigram_counts.most_common(10):
        if count >= 2:
            scored.append((bigram, count / total * 2.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_keywords]


def _generate_titles(keywords: List[tuple], limits: Dict) -> List[str]:
    """Generate title options from keywords."""
    if not keywords:
        return ["Video Content"]

    top_words = [kw for kw, _ in keywords[:5]]

    templates = [
        "{kw1}: Complete Guide to {kw2}",
        "How to {kw1} - {kw2} Tips & Tricks",
        "{kw1} Tutorial: Everything You Need to Know",
        "Top 10 {kw1} Tips for Better {kw2}",
        "The Ultimate {kw1} Guide ({kw2})",
    ]

    titles = []
    for i, template in enumerate(templates):
        kw1 = top_words[0].title() if top_words else "Topic"
        kw2 = top_words[min(i + 1, len(top_words) - 1)].title() if len(top_words) > 1 else "Results"
        title = template.format(kw1=kw1, kw2=kw2)
        # Trim to platform limit
        max_len = limits.get("title_max", 100)
        if max_len > 0 and len(title) > max_len:
            title = title[:max_len - 3] + "..."
        titles.append(title)

    return titles


def _generate_description(
    transcript_text: str,
    keywords: List[tuple],
    platform: str,
) -> str:
    """Generate a description from transcript and keywords."""
    # Take first ~200 chars of transcript as summary base
    sentences = re.split(r"[.!?]+", transcript_text)
    summary_sentences = []
    char_count = 0
    for s in sentences:
        s = s.strip()
        if s and char_count + len(s) < 300:
            summary_sentences.append(s)
            char_count += len(s)
        if char_count >= 200:
            break

    summary = ". ".join(summary_sentences)
    if summary and not summary.endswith("."):
        summary += "."

    # Add keyword section
    kw_list = ", ".join(kw for kw, _ in keywords[:8])

    limits = PLATFORM_LIMITS[platform]
    max_desc = limits.get("description_max", 2000)

    description = f"{summary}\n\nTopics covered: {kw_list}"

    if len(description) > max_desc:
        description = description[:max_desc - 3] + "..."

    return description


def _result_to_dict(result: SEOResult) -> dict:
    """Convert SEOResult to a JSON-serializable dict."""
    return {
        "title_options": result.title_options,
        "description": result.description,
        "tags": result.tags,
        "hashtags": result.hashtags,
        "optimal_posting_time": result.optimal_posting_time,
        "platform": result.platform,
        "method": result.method,
    }
