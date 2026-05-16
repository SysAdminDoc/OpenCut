"""Tests for the CSV/EDL/marker importer (F102)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from opencut.core import marker_import as mi


# ---------------------------------------------------------------------------
# Timecode parsing
# ---------------------------------------------------------------------------


def test_parse_timecode_accepts_hms_frames():
    assert mi.parse_timecode("01:00:00:00", fps=30) == 3600.0
    assert mi.parse_timecode("00:00:01:15", fps=30) == pytest.approx(1.5)
    # Drop-frame separator.
    assert mi.parse_timecode("00:00:01;15", fps=30) == pytest.approx(1.5)


def test_parse_timecode_accepts_dot_seconds():
    assert mi.parse_timecode("00:00:01.250") == pytest.approx(1.25)
    assert mi.parse_timecode("00:00:01,250") == pytest.approx(1.25)


def test_parse_timecode_accepts_float_seconds():
    assert mi.parse_timecode("1.5") == 1.5
    assert mi.parse_timecode("90") == 90.0


def test_parse_timecode_returns_none_for_garbage():
    for bad in (None, "", "  ", "not a time", "99:99:99:99x"):
        assert mi.parse_timecode(bad) is None


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_parse_csv_handles_canonical_columns():
    csv_text = textwrap.dedent(
        """
        timecode,name,duration,color,comment
        00:00:01:00,Intro start,00:00:02:00,red,Wide shot
        00:00:05:00,B-roll,,blue,
        """
    ).strip()

    result = mi.parse_csv(csv_text, fps=30)

    assert result.format == "csv"
    assert len(result.markers) == 2
    assert result.markers[0].name == "Intro start"
    assert result.markers[0].start_seconds == pytest.approx(1.0)
    assert result.markers[0].duration_seconds == pytest.approx(2.0)
    assert result.markers[0].color == "red"
    assert result.markers[1].color == "cyan"  # blue → cyan synonym


def test_parse_csv_supports_premiere_header_aliases():
    csv_text = textwrap.dedent(
        """
        Marker Name,In,Out,Marker Color
        Hook,00:00:00.500,00:00:02.000,Magenta
        """
    ).strip()

    result = mi.parse_premiere_csv(csv_text)

    assert len(result.markers) == 1
    marker = result.markers[0]
    assert marker.name == "Hook"
    assert marker.start_seconds == pytest.approx(0.5)
    assert marker.duration_seconds == pytest.approx(1.5)
    assert marker.color == "purple"  # magenta → purple synonym


def test_parse_csv_rejects_rows_with_bad_timecodes():
    csv_text = textwrap.dedent(
        """
        timecode,name
        00:00:01:00,Good
        not a time,Bad
        """
    ).strip()
    result = mi.parse_csv(csv_text, fps=30)
    assert [m.name for m in result.markers] == ["Good"]
    assert any("Bad" in line or "not a time" in line for line in result.rejected)


def test_parse_csv_warns_when_no_timecode_column():
    result = mi.parse_csv("name,note\nfoo,bar\n", fps=30)
    assert not result.markers
    assert result.warnings
    assert "timecode" in result.warnings[0].lower()


# ---------------------------------------------------------------------------
# EDL
# ---------------------------------------------------------------------------


def test_parse_edl_extracts_manual_markers():
    edl_text = textwrap.dedent(
        """
        TITLE: TEST
        FCM: NON-DROP FRAME

        001  AX       V     C        00:00:00:00 00:00:05:00 01:00:00:00 01:00:05:00
        * FROM CLIP NAME: clip.mov
        M: AX     RED      00:00:02:00 Highlight
        M: AX     GREEN    00:00:04:00 Tag
        """
    ).strip()

    result = mi.parse_edl(edl_text, fps=30)

    assert result.format == "edl"
    assert len(result.markers) == 2
    assert result.markers[0].name == "Highlight"
    assert result.markers[0].start_seconds == pytest.approx(2.0)
    assert result.markers[0].color == "red"
    assert result.markers[1].name == "Tag"


def test_parse_edl_warns_when_no_markers():
    edl_text = "TITLE: TEST\nFCM: NON-DROP FRAME\n001  AX  V  C  00:00:00:00 00:00:01:00 01:00:00:00 01:00:01:00\n"
    result = mi.parse_edl(edl_text, fps=30)
    assert not result.markers
    assert result.warnings


# ---------------------------------------------------------------------------
# Dispatch + filesystem
# ---------------------------------------------------------------------------


def test_detect_format_uses_extension():
    assert mi.detect_format("markers.edl") == "edl"
    assert mi.detect_format("markers.csv") == "csv"
    assert mi.detect_format("notes.txt", body="timecode,name\n00:00:01:00,Hi\n") == "csv"


def test_import_markers_dispatches_by_format(tmp_path):
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("timecode,name\n00:00:01:00,Hi\n", encoding="utf-8")
    result = mi.import_markers(path=str(csv_path), fps=30)
    assert result.format == "csv"
    assert result.markers[0].name == "Hi"


def test_import_markers_rejects_invalid_format_token():
    with pytest.raises(ValueError):
        mi.import_markers(text="anything", format="not_a_format")


def test_import_markers_requires_exactly_one_input():
    with pytest.raises(ValueError):
        mi.import_markers()
    with pytest.raises(ValueError):
        mi.import_markers(text="x", path="/tmp/y.csv")


# ---------------------------------------------------------------------------
# Route smoke
# ---------------------------------------------------------------------------


def test_markers_import_route_parses_inline_csv(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/markers/import",
        json={
            "format": "csv",
            "text": "timecode,name\n00:00:01:00,Hello\n",
            "fps": 30.0,
        },
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["count"] == 1
    assert payload["markers"][0]["name"] == "Hello"


def test_markers_import_route_validates_inputs(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/markers/import",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400


def test_markers_import_route_rejects_both_inputs(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/markers/import",
        json={"text": "x", "path": "/tmp/y.csv"},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
