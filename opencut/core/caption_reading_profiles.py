"""Source-backed caption reading-speed profiles.

The older caption compliance table mixed delivery rules with reading-speed
heuristics.  This registry keeps the speed assumptions explicit so QC callers
can choose a target profile without pretending every platform publishes a hard
numeric limit.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

SOURCE_URLS = {
    "netflix_english_us": (
        "https://partnerhelp.netflixstudios.com/hc/en-us/articles/"
        "217350977-English-USA-Timed-Text-Style-Guide"
    ),
    "bbc_subtitle_guidelines_archive": (
        "https://archive.ph/2026.01.08-135056/"
        "https%3A/www.bbc.co.uk/accessibility/forproducts/guides/subtitles/%23Spelling-out"
    ),
    "dcmp_captioning_key": "https://dcmp.org/captioningkey/print",
    "fcc_47_cfr_79_1": "https://www.law.cornell.edu/cfr/text/47/79.1",
    "youtube_help_captions": "https://support.google.com/youtube/answer/2734796?hl=en",
}


READING_SPEED_PROFILES = {
    "netflix-adult": {
        "id": "netflix-adult",
        "label": "Netflix adult English timed text",
        "target": "netflix",
        "audience": "adult",
        "measurement": "cps",
        "max_cps": 20,
        "max_wpm": None,
        "recommended_wpm_range": None,
        "enforcement": "hard",
        "source_confidence": "official",
        "source_url": SOURCE_URLS["netflix_english_us"],
        "source_note": (
            "Netflix English (USA) timed-text guide lists adult programs as up to "
            "20 characters per second."
        ),
    },
    "netflix-children": {
        "id": "netflix-children",
        "label": "Netflix children English timed text",
        "target": "netflix",
        "audience": "children",
        "measurement": "cps",
        "max_cps": 17,
        "max_wpm": None,
        "recommended_wpm_range": None,
        "enforcement": "hard",
        "source_confidence": "official",
        "source_url": SOURCE_URLS["netflix_english_us"],
        "source_note": (
            "Netflix English (USA) timed-text guide lists children's programs as "
            "up to 17 characters per second."
        ),
    },
    "bbc-editorial": {
        "id": "bbc-editorial",
        "label": "BBC editorial subtitle speed",
        "target": "bbc",
        "audience": "general",
        "measurement": "wpm",
        "max_cps": None,
        "max_wpm": 180,
        "recommended_wpm_range": [160, 180],
        "enforcement": "advisory",
        "source_confidence": "archived_official",
        "source_url": SOURCE_URLS["bbc_subtitle_guidelines_archive"],
        "source_note": (
            "BBC Subtitle Guidelines version 1.2.4a recommends 160-180 words "
            "per minute, with editorial adjustment by programme."
        ),
    },
    "dcmp-upper": {
        "id": "dcmp-upper",
        "label": "DCMP upper-level educational media",
        "target": "accessibility",
        "audience": "upper-level educational",
        "measurement": "wpm",
        "max_cps": None,
        "max_wpm": 160,
        "recommended_wpm_range": [0, 160],
        "enforcement": "advisory",
        "source_confidence": "official",
        "source_url": SOURCE_URLS["dcmp_captioning_key"],
        "source_note": (
            "DCMP Captioning Key says upper-level educational media should not "
            "exceed 160 words per minute."
        ),
    },
    "fcc-quality": {
        "id": "fcc-quality",
        "label": "FCC caption-quality qualitative timing",
        "target": "fcc",
        "audience": "televised video programming",
        "measurement": "qualitative",
        "max_cps": None,
        "max_wpm": None,
        "recommended_wpm_range": None,
        "enforcement": "qualitative",
        "source_confidence": "official",
        "source_url": SOURCE_URLS["fcc_47_cfr_79_1"],
        "source_note": (
            "47 CFR 79.1 requires offline captions to be displayed with enough "
            "time to be read completely; it does not publish a fixed WPM cap."
        ),
    },
    "youtube-advisory": {
        "id": "youtube-advisory",
        "label": "YouTube creator-paced advisory speed",
        "target": "youtube",
        "audience": "online creator captions",
        "measurement": "wpm",
        "max_cps": None,
        "max_wpm": 220,
        "recommended_wpm_range": [160, 220],
        "enforcement": "advisory",
        "source_confidence": "heuristic",
        "source_url": SOURCE_URLS["youtube_help_captions"],
        "source_note": (
            "YouTube Help documents caption text and timestamps but does not "
            "publish a hard reading-speed rule; 220 WPM is an OpenCut advisory "
            "ceiling for fast creator captions, not a YouTube requirement."
        ),
    },
}


PROFILE_ALIASES = {
    "netflix": "netflix-adult",
    "netflix_adult": "netflix-adult",
    "netflix-adults": "netflix-adult",
    "netflix_child": "netflix-children",
    "netflix-child": "netflix-children",
    "netflix_children": "netflix-children",
    "bbc": "bbc-editorial",
    "bbc_editorial": "bbc-editorial",
    "dcmp": "dcmp-upper",
    "dcmp_upper": "dcmp-upper",
    "fcc": "fcc-quality",
    "fcc_quality": "fcc-quality",
    "youtube": "youtube-advisory",
    "youtube_advisory": "youtube-advisory",
}


CORRECTION_NOTE = (
    "F240 source verification corrected the backlog premise: current Netflix "
    "English guidance uses 20 CPS for adult programs and 17 CPS for children's "
    "programs. FCC and YouTube do not publish hard numeric reading-speed caps "
    "in the official sources used here, so their numeric automation is either "
    "qualitative or explicitly advisory."
)


def normalize_reading_profile(profile_id: Optional[str]) -> Optional[str]:
    """Return the canonical profile id or ``None`` for empty input."""

    if profile_id is None:
        return None
    key = str(profile_id).strip().lower().replace(" ", "-")
    if not key:
        return None
    key = PROFILE_ALIASES.get(key, key)
    if key not in READING_SPEED_PROFILES:
        choices = ", ".join(sorted(READING_SPEED_PROFILES))
        raise ValueError(f"unknown reading profile: {profile_id!r}; choose from {choices}")
    return key


def get_reading_speed_profile(profile_id: str) -> dict:
    """Return one canonical reading-speed profile."""

    key = normalize_reading_profile(profile_id)
    if key is None:
        raise ValueError("reading profile is required")
    return deepcopy(READING_SPEED_PROFILES[key])


def get_reading_speed_profiles() -> dict:
    """Return every reading-speed profile for API responses and tests."""

    return deepcopy(READING_SPEED_PROFILES)


def reading_profile_rule_overrides(profile_id: str) -> dict:
    """Return compliance-rule overrides for a reading-speed profile."""

    profile = get_reading_speed_profile(profile_id)
    return {
        "max_cps": profile.get("max_cps"),
        "max_wpm": profile.get("max_wpm"),
        "label": profile["label"],
    }
