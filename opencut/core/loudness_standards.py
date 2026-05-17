"""Loudness standards and preset metadata.

The numeric targets here are intentionally separate from the FFmpeg runners so
UI, API, QC, and release tests can all cite the same source-backed facts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

SOURCE_URLS = {
    "ebu_r128": "https://tech.ebu.ch/fr/publications/r128",
    "itu_bs1770": "https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en",
    "itu_bs1770_pdf": "https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf",
    "ffmpeg_loudnorm": "https://ffmpeg.org/ffmpeg-filters.html#loudnorm",
    "spotify_loudness": "https://support.spotify.com/ee-en/artists/article/loudness-normalization/",
}


LOUDNESS_STANDARDS = {
    "itu_r_bs1770": {
        "id": "itu_r_bs1770",
        "label": "ITU-R BS.1770",
        "current_version": "BS.1770-5",
        "approved": "2023-11-22",
        "status": "in_force",
        "previous_version": "BS.1770-4",
        "previous_status": "superseded",
        "measurement_terms": ["integrated_loudness", "loudness_range", "true_peak"],
        "source_url": SOURCE_URLS["itu_bs1770"],
        "pdf_url": SOURCE_URLS["itu_bs1770_pdf"],
        "notes": (
            "BS.1770-5 is the current ITU-R recommendation for programme "
            "loudness and true-peak measurement; BS.1770-4 is superseded."
        ),
    },
    "ebu_r128": {
        "id": "ebu_r128",
        "label": "EBU R 128",
        "current_version": "5.0",
        "published": "2023-11",
        "target_lufs": -23.0,
        "max_true_peak_dbtp": -1.0,
        "descriptors": ["Programme Loudness", "Loudness Range", "Maximum True Peak Level"],
        "measurement_basis": "itu_r_bs1770",
        "source_url": SOURCE_URLS["ebu_r128"],
        "notes": "EBU R 128 v5.0 keeps the -23 LUFS programme target and -1 dBTP true-peak ceiling.",
    },
    "ffmpeg_loudnorm": {
        "id": "ffmpeg_loudnorm",
        "label": "FFmpeg loudnorm",
        "measurement_basis": "ebu_r128",
        "supports_two_pass": True,
        "targets": ["integrated_loudness", "loudness_range", "maximum_true_peak"],
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "OpenCut's normalization paths use FFmpeg loudnorm/ebur128 rather than bespoke metering.",
    },
}


LOUDNESS_PRESETS: Dict[str, dict] = {
    "youtube": {
        "i": -14.0,
        "tp": -1.0,
        "lra": 11.0,
        "label": "YouTube / online video",
        "category": "platform",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Common creator-platform profile; use a platform-specific preset when available.",
    },
    "podcast": {
        "i": -16.0,
        "tp": -1.0,
        "lra": 8.0,
        "label": "Podcast speech",
        "category": "spoken_word",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Speech-safe podcast profile; the previous roadmap note suggesting -14 LUFS for podcast was not source-backed.",
    },
    "broadcast": {
        "i": -23.0,
        "tp": -1.0,
        "lra": 7.0,
        "label": "EBU R 128 broadcast",
        "category": "broadcast",
        "measurement_standard": "ebu_r128",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ebu_r128"],
        "notes": "EBU R 128 v5.0 target programme loudness and true-peak ceiling.",
    },
    "streaming": {
        "i": -16.0,
        "tp": -1.0,
        "lra": 11.0,
        "label": "General online speech/video",
        "category": "online_fallback",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Generic online fallback. Platform presets such as spotify/youtube stay at their explicit platform targets.",
    },
    "tiktok": {
        "i": -14.0,
        "tp": -1.0,
        "lra": 11.0,
        "label": "TikTok / short-form",
        "category": "platform",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Short-form platform approximation; user override remains available.",
    },
    "spotify": {
        "i": -14.0,
        "tp": -1.0,
        "lra": 9.0,
        "label": "Spotify",
        "category": "platform",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["spotify_loudness"],
        "notes": "Spotify documents Normal playback at -14 LUFS and mastering guidance below -1 dBTP.",
    },
    "apple_music": {
        "i": -16.0,
        "tp": -1.0,
        "lra": 10.0,
        "label": "Apple Music / Sound Check",
        "category": "platform",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Conservative music-platform profile; user override remains available.",
    },
    "cinema": {
        "i": -24.0,
        "tp": -2.0,
        "lra": 20.0,
        "label": "Cinema / wide dynamic range",
        "category": "delivery",
        "measurement_standard": "itu_r_bs1770",
        "implementation": "ffmpeg_loudnorm",
        "source_url": SOURCE_URLS["ffmpeg_loudnorm"],
        "notes": "Wide-LRA delivery profile; not an EBU broadcast preset.",
    },
}


PLATFORM_TARGETS: Dict[str, float] = {
    "youtube": -14.0,
    "spotify": -14.0,
    "apple_podcasts": -16.0,
    "broadcast": -24.0,
    "ebu_broadcast": -23.0,
    "online_video": -16.0,
    "tiktok": -14.0,
}


def get_loudness_presets() -> List[dict]:
    """Return API-safe loudness preset records with alias fields."""
    presets = []
    for name, vals in LOUDNESS_PRESETS.items():
        record = deepcopy(vals)
        record.update({
            "name": name,
            "target_lufs": vals["i"],
            "target_tp": vals["tp"],
            "target_lra": vals["lra"],
        })
        presets.append(record)
    return presets


def get_loudness_standards() -> Dict[str, dict]:
    """Return source-backed standard metadata for API responses and tests."""
    return deepcopy(LOUDNESS_STANDARDS)


def get_loudness_preset(name: str) -> dict:
    """Return a preset by name, falling back to the YouTube profile."""
    return deepcopy(LOUDNESS_PRESETS.get(name, LOUDNESS_PRESETS["youtube"]))
