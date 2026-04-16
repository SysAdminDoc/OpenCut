"""
NLP Command Parser

Maps natural language instructions to OpenCut API endpoints.
Uses LLM when available, falls back to keyword matching.
"""

import json
import logging
import re
from typing import List, Optional

from .llm import LLMConfig, query_llm

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Command map
# ---------------------------------------------------------------------------

COMMAND_MAP = [
    {"keywords": ["silence", "remove silence", "cut silence"], "route": "/captions/silence", "params": {}},
    {"keywords": ["filler", "um", "uh"], "route": "/captions/fillers", "params": {}},
    {"keywords": ["caption", "subtitle", "transcribe"], "route": "/captions/transcribe", "params": {}},
    {"keywords": ["chapter", "chapters", "youtube"], "route": "/captions/chapters", "params": {}},
    {"keywords": ["noise", "denoise", "clean audio"], "route": "/audio/denoise", "params": {}},
    {"keywords": ["zoom", "push in", "auto zoom"], "route": "/video/auto-zoom", "params": {}},
    {"keywords": ["color match", "match color"], "route": "/video/color-match", "params": {}},
    {"keywords": ["normalize", "loudness", "lufs"], "route": "/audio/loudness-match", "params": {}},
    {"keywords": ["multicam", "podcast", "speaker cut"], "route": "/video/multicam-cuts", "params": {}},
    {"keywords": ["repeat", "duplicate take", "fumble"], "route": "/captions/repeat-detect", "params": {}},
    {"keywords": ["beat", "marker", "sync beat"], "route": "/audio/beat-markers", "params": {}},
    {"keywords": ["search", "find footage", "footage search"], "route": "/search/footage", "params": {}},
    {"keywords": ["vfx sheet", "vfx list", "effects list"], "route": "/deliverables/vfx-sheet", "params": {}},
    {"keywords": ["adr", "dialogue list"], "route": "/deliverables/adr-list", "params": {}},
    {"keywords": ["music cue", "cue sheet"], "route": "/deliverables/music-cue-sheet", "params": {}},
    {"keywords": ["asset list", "media list"], "route": "/deliverables/asset-list", "params": {}},
    {"keywords": ["batch rename", "rename clips"], "route": "/timeline/batch-rename", "params": {}},
    {"keywords": ["smart bin", "organize", "bins"], "route": "/timeline/smart-bins", "params": {}},
    {"keywords": ["export markers", "marker export", "batch export"], "route": "/timeline/export-from-markers", "params": {}},
]

# ---------------------------------------------------------------------------
# Keyword parsing
# ---------------------------------------------------------------------------

# Map longer (more specific) keywords first for higher confidence scoring
_SORTED_COMMAND_MAP = sorted(
    COMMAND_MAP,
    key=lambda e: max(len(k) for k in e["keywords"]),
    reverse=True,
)


def parse_command_keyword(text: str) -> Optional[dict]:
    """
    Match natural language text to a route using keyword substring matching.

    Longer keyword matches are preferred and score higher confidence.
    Multiple keywords from the same entry found in the text raise confidence.

    Args:
        text: Raw user input string.

    Returns:
        Dict with "route", "params", "confidence", "matched_keyword", or None
        if no match is found.
    """
    text_lower = text.lower()

    best_match = None
    best_confidence = 0.0
    best_keyword = ""

    for entry in _SORTED_COMMAND_MAP:
        keywords = entry["keywords"]
        matched_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        if not matched_keywords:
            continue

        # Confidence: proportion of entry keywords matched + length bonus
        keyword_coverage = len(matched_keywords) / len(keywords)
        # Give a bonus for longer, more specific keywords
        longest_match = max(len(kw) for kw in matched_keywords)
        length_bonus = min(1.0, longest_match / 20.0)  # cap at 1.0 normalised

        confidence = round(0.6 * keyword_coverage + 0.4 * length_bonus, 4)

        if confidence > best_confidence:
            best_confidence = confidence
            best_match = entry
            best_keyword = max(matched_keywords, key=len)

    if best_match is None:
        return None

    return {
        "route": best_match["route"],
        "params": dict(best_match["params"]),
        "confidence": best_confidence,
        "matched_keyword": best_keyword,
    }


# ---------------------------------------------------------------------------
# LLM-based parsing
# ---------------------------------------------------------------------------

_ROUTES_DESCRIPTION = "\n".join(
    f"  {entry['route']}: triggered by keywords like {', '.join(repr(k) for k in entry['keywords'][:2])}"
    for entry in COMMAND_MAP
)

_LLM_SYSTEM_PROMPT = (
    "You are an AI assistant for OpenCut, a video editing tool. "
    "Map user instructions to the correct API route and extract any relevant parameters."
)

_LLM_PROMPT_TEMPLATE = """\
Available OpenCut API routes:
{routes}

User instruction: "{text}"

Identify the best matching route and extract any parameters from the user's instruction.
Return ONLY a JSON object with this exact structure:
{{"route": "/route/path", "params": {{}}, "confidence": 0.95, "matched_keyword": "keyword"}}

If no route matches, return: {{"route": null, "params": {{}}, "confidence": 0.0, "matched_keyword": ""}}
"""


def parse_command_llm(
    text: str,
    llm_config: LLMConfig,
    available_routes: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Use an LLM to map a user instruction to an OpenCut API route.

    Args:
        text: Raw user instruction.
        llm_config: LLMConfig for the LLM provider to use.
        available_routes: Optional list of route strings to restrict to.
                          If None, uses all routes in COMMAND_MAP.

    Returns:
        Dict with "route", "params", "confidence", "matched_keyword", or None
        if the LLM returns null route or parsing fails.
    """
    routes_section = _ROUTES_DESCRIPTION
    if available_routes:
        routes_section = "\n".join(
            f"  {r}" for r in available_routes
        )

    prompt = _LLM_PROMPT_TEMPLATE.format(routes=routes_section, text=text)

    try:
        response = query_llm(prompt, config=llm_config, system_prompt=_LLM_SYSTEM_PROMPT)
    except Exception as exc:
        logger.warning("LLM command parse failed: %s", exc)
        return None

    raw = response.text.strip()

    # Extract JSON object from response. Try ``json.loads(raw)`` first so a
    # clean response is parsed verbatim. Fall back to a balanced-brace scan
    # because a greedy ``\{.*\}`` regex eats stray prose ``{`` earlier in the
    # response and produces invalid JSON (see chat_editor for the same class
    # of bug).
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass
    if parsed is None:
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        start = -1
                        continue
    if not isinstance(parsed, dict):
        logger.warning("No JSON object found in LLM command response: %s", raw[:200])
        return None

    route = parsed.get("route")
    if not route:
        return None

    # Validate route against known COMMAND_MAP routes
    valid_routes = {e["route"] for e in COMMAND_MAP}
    route_str = str(route)
    if route_str not in valid_routes:
        logger.warning("LLM returned unknown route %r, discarding", route_str)
        return None

    params = parsed.get("params", {})
    if not isinstance(params, dict):
        params = {}

    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "route": route_str,
        "params": params,
        "confidence": confidence,
        "matched_keyword": str(parsed.get("matched_keyword", "")),
    }


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

# Language names → ISO 639-1 codes
_LANGUAGE_MAP = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
}

# Intensity words → numeric modifiers
_INTENSITY_MAP = {
    "aggressive": 0.9,
    "strong": 0.8,
    "high": 0.8,
    "medium": 0.5,
    "moderate": 0.5,
    "low": 0.2,
    "light": 0.2,
    "gentle": 0.2,
    "slow": 0.2,
    "fast": 0.9,
    "quick": 0.9,
}


def extract_params_from_text(text: str) -> dict:
    """
    Extract common OpenCut parameters from natural language text.

    Extracted params:
        - Numbers → "threshold" or "duration" (first two numbers found)
        - Language names → "language" (ISO 639-1 code)
        - Intensity words → "intensity" (0.0–1.0 float)

    Args:
        text: Natural language instruction.

    Returns:
        Dict of extracted parameter names and values. Only keys for which
        values were found will be present.
    """
    params = {}
    text_lower = text.lower()

    # Extract numbers (integers and decimals, including negatives so dB
    # thresholds like "-30 dB" keep their sign). The leading lookbehind
    # allows start-of-string or whitespace/``,`` / ``(`` before the number
    # so we don't pick up digits inside words like ``abc1``. A trailing
    # ``\b`` on the integer form would reject ``1.5`` because ``1`` is
    # followed by ``.`` and the decimal portion wouldn't re-anchor. Allow
    # an optional decimal body without ``\b`` at the end so ``1.5s``,
    # ``1.5 seconds``, and ``30%`` all extract the full number.
    numbers = re.findall(
        r"(?:(?<=^)|(?<=[\s,(]))-?\d+(?:\.\d+)?(?![\d.])",
        text,
    )
    numeric_values = []
    for n in numbers:
        try:
            numeric_values.append(float(n))
        except ValueError:
            continue
    if len(numeric_values) >= 1:
        params["threshold"] = numeric_values[0]
    if len(numeric_values) >= 2:
        params["duration"] = numeric_values[1]

    # Extract language
    for lang_name, lang_code in _LANGUAGE_MAP.items():
        if lang_name in text_lower:
            params["language"] = lang_code
            break

    # Extract intensity
    for intensity_word, intensity_val in _INTENSITY_MAP.items():
        if intensity_word in text_lower:
            params["intensity"] = intensity_val
            break

    # LUFS target (common patterns: "-14 LUFS", "minus 14 LUFS")
    lufs_match = re.search(r"(-?\d+(?:\.\d+)?)\s*lufs", text_lower)
    if lufs_match:
        params["target_lufs"] = float(lufs_match.group(1))

    # dB threshold (e.g. "-30 dB threshold", "at -40dB") — override the
    # positional heuristic so the semantic extraction wins over order.
    db_match = re.search(r"(-?\d+(?:\.\d+)?)\s*d[bB]\b", text_lower)
    if db_match:
        try:
            params["threshold"] = float(db_match.group(1))
        except (TypeError, ValueError):
            pass

    # "X seconds" / "X sec" → duration (again, semantic > positional)
    sec_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)", text_lower)
    if sec_match:
        try:
            params["duration"] = float(sec_match.group(1))
        except (TypeError, ValueError):
            pass

    # Percentage (e.g. "15% zoom" → zoom_amount=1.15)
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*zoom", text_lower)
    if pct_match:
        params["zoom_amount"] = 1.0 + float(pct_match.group(1)) / 100.0

    return params


# ---------------------------------------------------------------------------
# Unified parse entry point
# ---------------------------------------------------------------------------

def parse_command(text: str, llm_config: Optional[LLMConfig] = None) -> Optional[dict]:
    """
    Parse a natural language command, returning the matching route and params.

    Tries LLM first if llm_config is provided, then falls back to keyword matching.
    Always enriches the result with extract_params_from_text.

    Args:
        text: Raw user instruction (e.g. "remove silences from my video").
        llm_config: Optional LLMConfig. If None, skips LLM and uses keywords only.

    Returns:
        Dict with "route", "params", "confidence", "matched_keyword", and
        "param_source" ("llm" or "keyword"), or None if no match found.
    """
    result = None

    # Attempt LLM first
    if llm_config is not None:
        try:
            result = parse_command_llm(text, llm_config)
            if result:
                result["param_source"] = "llm"
                logger.info("NLP command matched via LLM: %s (conf=%.2f)", result["route"], result["confidence"])
        except Exception as exc:
            logger.warning("LLM command parse raised exception: %s", exc)
            result = None

    # Fall back to keyword matching
    if result is None:
        result = parse_command_keyword(text)
        if result:
            result["param_source"] = "keyword"
            logger.info("NLP command matched via keyword: %s (conf=%.2f)", result["route"], result["confidence"])

    if result is None:
        logger.debug("NLP command not matched: %r", text)
        return None

    # Enrich params with values extracted from the text
    extracted = extract_params_from_text(text)
    # Don't overwrite params already set by LLM
    for k, v in extracted.items():
        result["params"].setdefault(k, v)

    return result
