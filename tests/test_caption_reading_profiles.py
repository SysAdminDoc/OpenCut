"""Regression tests for F240 caption reading-speed profiles."""

from __future__ import annotations

from opencut.core.caption_qc import qc_captions
from opencut.core.caption_reading_profiles import (
    CORRECTION_NOTE,
    get_reading_speed_profile,
    get_reading_speed_profiles,
    normalize_reading_profile,
)

NETFLIX_18_CPS_SRT = """1
00:00:00,000 --> 00:00:02,000
ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJ
"""


BBC_200_WPM_SRT = """1
00:00:00,000 --> 00:00:03,000
one two three four five
six seven eight nine ten
"""


def test_profiles_capture_corrected_source_facts():
    profiles = get_reading_speed_profiles()

    assert profiles["netflix-adult"]["max_cps"] == 20
    assert profiles["netflix-children"]["max_cps"] == 17
    assert profiles["bbc-editorial"]["recommended_wpm_range"] == [160, 180]
    assert profiles["bbc-editorial"]["max_wpm"] == 180
    assert profiles["fcc-quality"]["max_wpm"] is None
    assert profiles["youtube-advisory"]["max_wpm"] == 220
    assert profiles["youtube-advisory"]["source_confidence"] == "heuristic"
    assert "20 CPS for adult programs" in CORRECTION_NOTE


def test_profile_aliases_normalize_to_canonical_ids():
    assert normalize_reading_profile("netflix_children") == "netflix-children"
    assert normalize_reading_profile("BBC") == "bbc-editorial"
    assert get_reading_speed_profile("fcc")["id"] == "fcc-quality"


def test_netflix_children_profile_is_stricter_than_adult_profile():
    adult = qc_captions(
        srt_text=NETFLIX_18_CPS_SRT,
        standard="netflix",
        reading_profile="netflix-adult",
    )
    children = qc_captions(
        srt_text=NETFLIX_18_CPS_SRT,
        standard="netflix",
        reading_profile="netflix-children",
    )

    assert not [d for d in adult.diagnostics if d.rule == "chars_per_second"]
    assert any(d.rule == "chars_per_second" for d in children.diagnostics)
    assert children.reading_profile == "netflix-children"
    assert children.error_count == 1


def test_bbc_editorial_profile_warns_above_180_wpm():
    result = qc_captions(
        srt_text=BBC_200_WPM_SRT,
        standard="bbc",
        reading_profile="bbc-editorial",
    )

    reading_speed = [d for d in result.diagnostics if d.rule == "reading_speed"]
    assert reading_speed
    assert "max 180" in reading_speed[0].message
    assert result.warning_count == 1
    assert result.error_count == 0


def test_reading_profiles_route_returns_source_metadata(client):
    resp = client.get("/captions/qc/reading-profiles")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["profiles"]["netflix-children"]["max_cps"] == 17
    assert payload["profiles"]["fcc-quality"]["enforcement"] == "qualitative"
    assert "youtube_help_captions" in payload["source_urls"]


def test_qc_route_accepts_reading_profile_alias(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/captions/qc",
        json={
            "srt_text": NETFLIX_18_CPS_SRT,
            "standard": "netflix",
            "reading_profile": "netflix_children",
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["reading_profile"] == "netflix-children"
    assert payload["reading_profile_source_confidence"] == "official"
    assert any(d["rule"] == "chars_per_second" for d in payload["diagnostics"])
