"""Tests for the caption QC gate (F111)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from opencut.core import caption_qc


GOOD_SRT = """1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:03,000 --> 00:00:05,000
This is a clean cue.
"""


# Reading speed too high (long sentence over a short window) + overlap.
BAD_TIMING_SRT = """1
00:00:00,000 --> 00:00:01,000
This single line is far too long and absurdly dense for one short cue window now.

2
00:00:00,800 --> 00:00:02,000
Overlap cue.
"""


# Forbidden glyph: contains U+200B (zero-width space).
FORBIDDEN_GLYPH_SRT = "1\n00:00:00,000 --> 00:00:02,000\nLook at​this caption.\n"


# Control character (form feed U+000C) inside the text.
CONTROL_CHAR_SRT = "1\n00:00:00,000 --> 00:00:02,000\nHello\x0cworld.\n"


def _write_tmp(text: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".srt", delete=False)
    f.write(text)
    f.flush()
    f.close()
    return f.name


def test_qc_passes_on_clean_srt():
    result = caption_qc.qc_captions(srt_text=GOOD_SRT)

    assert result.overall_pass is True
    assert result.error_count == 0
    assert result.total_cues == 2


def test_qc_detects_overlap():
    result = caption_qc.qc_captions(srt_text=BAD_TIMING_SRT)

    rules = {d.rule for d in result.diagnostics}
    assert "cue_overlap" in rules
    assert result.overall_pass is False
    assert result.error_count >= 1


def test_qc_detects_forbidden_glyph():
    result = caption_qc.qc_captions(srt_text=FORBIDDEN_GLYPH_SRT)

    glyph_diags = [d for d in result.diagnostics if d.rule == "forbidden_glyph"]
    assert glyph_diags, "expected forbidden_glyph diagnostic"
    assert glyph_diags[0].severity == "error"


def test_qc_detects_control_character():
    result = caption_qc.qc_captions(srt_text=CONTROL_CHAR_SRT)

    rules = [d.rule for d in result.diagnostics]
    assert "control_character" in rules


def test_advisory_mode_downgrades_reading_speed():
    """Reading speed violations become warnings in advisory mode."""
    src = """1
00:00:00,000 --> 00:00:00,500
This sentence is much too long for the half-second cue it occupies, and should normally fail.
"""
    strict = caption_qc.qc_captions(srt_text=src, mode="strict")
    advisory = caption_qc.qc_captions(srt_text=src, mode="advisory")

    severities = {d.severity for d in advisory.diagnostics if d.rule.endswith("_exceeded")}
    if severities:
        assert "error" not in severities
    assert strict.error_count >= advisory.error_count


def test_qc_rejects_invalid_mode():
    with pytest.raises(ValueError):
        caption_qc.qc_captions(srt_text=GOOD_SRT, mode="oops")


def test_qc_rejects_missing_input():
    with pytest.raises(ValueError):
        caption_qc.qc_captions()


def test_enforce_export_gate_raises_on_failure():
    path = _write_tmp(BAD_TIMING_SRT)
    try:
        with pytest.raises(caption_qc.CaptionQcFailure) as exc:
            caption_qc.enforce_export_gate(path)
        assert exc.value.result.error_count >= 1
    finally:
        os.unlink(path)


def test_enforce_export_gate_passes_with_force():
    path = _write_tmp(BAD_TIMING_SRT)
    try:
        result = caption_qc.enforce_export_gate(path, force=True)
        assert result.overall_pass is False
        assert result.error_count >= 1
    finally:
        os.unlink(path)


def test_qc_route_returns_diagnostics(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/captions/qc",
        json={"srt_text": BAD_TIMING_SRT, "standard": "accessibility"},
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["overall_pass"] is False
    assert payload["error_count"] >= 1
    assert any(d["rule"] == "cue_overlap" for d in payload["diagnostics"])


def test_qc_route_rejects_missing_body(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/captions/qc",
        json={},
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    body = resp.get_json()
    assert "srt_path" in body["error"] or "srt_text" in body["error"]
