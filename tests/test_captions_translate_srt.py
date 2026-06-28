"""F139 — caption translation endpoint SRT-in / SRT-out support.

The endpoint already accepted a `segments` list. F139 spec adds the
SRT-in / SRT-out paths so the workflow works for users who already
have an `.srt` file on disk and want a translated `.srt` back.

These tests cover the new shape WITHOUT requiring NLLB / SeamlessM4T
to be installed — we monkeypatch the underlying translator to a
pass-through that just tags the text. That keeps the gate fast
(<5s) and runnable in the dev VM and Linux CI legs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List

import pytest

from opencut.routes.captions import (
    _segments_to_srt_text,
    _srt_to_translate_segments,
    _validate_translate_input,
)

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_accepts_segments_list():
    err = _validate_translate_input({
        "segments": [{"start": 0, "end": 1, "text": "hi"}],
        "backend": "nllb",
        "accept_restricted_license": True,
    })
    assert err is None


def test_validator_accepts_srt_path_only():
    err = _validate_translate_input({
        "srt_path": "/tmp/x.srt",
        "backend": "nllb",
        "accept_restricted_license": True,
    })
    assert err is None


def test_validator_accepts_srt_content_only():
    err = _validate_translate_input({
        "srt_content": "1\n00:00:00,000 --> 00:00:01,000\nhi\n",
        "backend": "nllb",
        "accept_restricted_license": True,
    })
    assert err is None


def test_validator_rejects_empty_input():
    err = _validate_translate_input({})
    assert err == "Provide `segments`, `srt_path`, or `srt_content`."


def test_validator_rejects_non_list_segments():
    err = _validate_translate_input({"segments": "oops"})
    assert err == "segments must be a list"


def test_validator_rejects_too_many_segments():
    err = _validate_translate_input(
        {"segments": [{"start": 0, "end": 1, "text": "x"}] * 5},
        max_segments=4,
    )
    assert "Too many segments" in err


# ---------------------------------------------------------------------------
# SRT round-trip helpers
# ---------------------------------------------------------------------------


SAMPLE_SRT = (
    "1\n00:00:00,500 --> 00:00:02,000\nHello world.\n\n"
    "2\n00:00:02,500 --> 00:00:04,000\nHow are you?\n"
)


def test_srt_to_segments_parses_two_cues():
    segs = _srt_to_translate_segments(SAMPLE_SRT)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == pytest.approx(0.5, abs=1e-3)
    assert segs[0]["end"] == pytest.approx(2.0, abs=1e-3)
    assert segs[1]["text"] == "How are you?"


def test_srt_to_segments_rejects_empty_blob():
    with pytest.raises(ValueError):
        _srt_to_translate_segments("")


def test_srt_to_segments_rejects_too_many_cues():
    long_srt = "".join(
        f"{i+1}\n00:00:00,000 --> 00:00:01,000\nline {i}\n\n" for i in range(5)
    )
    with pytest.raises(ValueError):
        _srt_to_translate_segments(long_srt, max_segments=4)


def test_segments_to_srt_preserves_count_and_timing():
    segs = [
        {"start": 0.5, "end": 2.0, "text": "Bonjour le monde."},
        {"start": 2.5, "end": 4.0, "text": "Comment ça va ?"},
    ]
    out = _segments_to_srt_text(segs)
    # Cue index, timeline, text, blank-line separator
    assert "1\n00:00:00,500 --> 00:00:02,000\nBonjour le monde." in out
    assert "2\n00:00:02,500 --> 00:00:04,000\nComment ça va ?" in out


def test_round_trip_idempotent_on_text():
    """Parse -> render -> parse must give identical text and timing."""
    segs = _srt_to_translate_segments(SAMPLE_SRT)
    rendered = _segments_to_srt_text(segs)
    re_parsed = _srt_to_translate_segments(rendered)
    assert len(re_parsed) == len(segs)
    for original, recovered in zip(segs, re_parsed):
        assert recovered["text"] == original["text"]
        assert recovered["start"] == pytest.approx(original["start"], abs=1e-3)
        assert recovered["end"] == pytest.approx(original["end"], abs=1e-3)


# ---------------------------------------------------------------------------
# Route-level SRT-in / SRT-out integration
# ---------------------------------------------------------------------------


def _fake_translate(segments: Iterable[dict], **kwargs) -> List[dict]:
    """Identity-ish translator: prefixes 'ES: ' so we can assert it ran."""
    out = []
    for seg in segments:
        new = dict(seg)
        new["text"] = "ES: " + (seg.get("text") or "")
        out.append(new)
    return out


@pytest.fixture(name="client")
def _client_fixture(monkeypatch):
    """Spin up the Flask test client with translation mocked."""
    import opencut.core.captions_enhanced as ce
    monkeypatch.setattr(ce, "translate_segments_auto", _fake_translate, raising=False)
    monkeypatch.setattr(ce, "translate_segments", _fake_translate, raising=False)

    from opencut.server import _get_app
    app = _get_app()
    app.config["TESTING"] = True
    return app.test_client()


def _csrf_headers(client):
    resp = client.get("/health")
    token = resp.get_json().get("csrf_token", "")
    return {"Content-Type": "application/json", "X-OpenCut-Token": token}


def _poll_job(client, job_id, *, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}")
        body = resp.get_json() or {}
        if body.get("status") in ("complete", "error", "cancelled"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout}s")


def test_route_translates_segments_input(client):
    headers = _csrf_headers(client)
    body = {
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
        "target_lang": "es",
        "backend": "nllb",
        "accept_restricted_license": True,
    }
    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)
    assert resp.status_code == 200, resp.get_json()
    job = _poll_job(client, resp.get_json()["job_id"])
    assert job["status"] == "complete", job
    result = job["result"]
    assert result["count"] == 1
    assert result["input_format"] == "segments"
    assert result["segments"][0]["text"] == "ES: hello"
    assert "srt" not in result  # output_srt defaults to False


def test_route_translates_srt_content_input_and_emits_srt(client):
    headers = _csrf_headers(client)
    body = {
        "srt_content": SAMPLE_SRT,
        "target_lang": "es",
        "backend": "nllb",
        "accept_restricted_license": True,
    }
    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)
    assert resp.status_code == 200, resp.get_json()
    job = _poll_job(client, resp.get_json()["job_id"])
    assert job["status"] == "complete", job
    result = job["result"]
    assert result["count"] == 2
    assert result["input_format"] == "srt"
    # Default-on SRT output for SRT-in callers.
    assert "srt" in result
    assert "ES: Hello world." in result["srt"]
    assert "ES: How are you?" in result["srt"]
    assert result["srt_encoding"] == "utf-8"


def test_route_writes_srt_to_disk_when_path_provided(tmp_path, client):
    headers = _csrf_headers(client)
    target = tmp_path / "translated.srt"
    body = {
        "srt_content": SAMPLE_SRT,
        "target_lang": "es",
        "backend": "nllb",
        "accept_restricted_license": True,
        "srt_output_path": str(target),
    }
    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)
    assert resp.status_code == 200, resp.get_json()
    job = _poll_job(client, resp.get_json()["job_id"])
    assert job["status"] == "complete", job
    result = job["result"]
    assert result["srt_output_path"] == str(target.resolve())
    assert Path(result["srt_output_path"]).is_file()
    on_disk = Path(result["srt_output_path"]).read_bytes()
    # F243 — default is UTF-8 without BOM.
    assert not on_disk.startswith(b"\xef\xbb\xbf")
    assert b"ES: Hello world." in on_disk


def test_route_legacy_bom_round_trip_with_srt_path(tmp_path, client, monkeypatch):
    """Reads an SRT off disk, translates, writes SRT with legacy Windows BOM."""
    headers = _csrf_headers(client)
    src = tmp_path / "source.srt"
    src.write_bytes(b"\xef\xbb\xbf" + SAMPLE_SRT.encode("utf-8"))
    dst = tmp_path / "translated.srt"

    body = {
        "srt_path": str(src),
        "target_lang": "es",
        "backend": "nllb",
        "accept_restricted_license": True,
        "srt_output_path": str(dst),
        "srt_legacy_bom": True,
    }
    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)
    assert resp.status_code == 200, resp.get_json()
    job = _poll_job(client, resp.get_json()["job_id"])
    assert job["status"] == "complete", job
    on_disk = dst.read_bytes()
    assert on_disk.startswith(b"\xef\xbb\xbf"), "legacy BOM toggle must write BOM"
    assert "ES:" in on_disk.decode("utf-8-sig")


def test_route_rejects_empty_input_with_400(client):
    headers = _csrf_headers(client)
    resp = client.post("/captions/translate", data=json.dumps({}), headers=headers)
    assert resp.status_code == 400
    body = resp.get_json()
    assert "Provide" in body.get("error", "")


def test_route_rejects_auto_backend_without_commercial_safe_default(client):
    headers = _csrf_headers(client)
    body = {
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
        "target_lang": "es",
    }

    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)

    assert resp.status_code == 400
    payload = resp.get_json()
    assert "No commercial-safe local caption translation backend" in payload.get("error", "")
    assert "accept_restricted_license=true" in payload.get("error", "")


def test_route_rejects_nllb_without_restricted_license_opt_in(client):
    headers = _csrf_headers(client)
    body = {
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
        "target_lang": "es",
        "backend": "nllb",
    }

    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)

    assert resp.status_code == 400
    payload = resp.get_json()
    assert "NLLB-200 distilled 600M uses CC-BY-NC-4.0" in payload.get("error", "")
    assert "accept_restricted_license=true" in payload.get("error", "")


def test_route_reports_restricted_backend_license_after_opt_in(client):
    headers = _csrf_headers(client)
    body = {
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
        "target_lang": "es",
        "backend": "nllb",
        "accept_restricted_license": True,
    }

    resp = client.post("/captions/translate", data=json.dumps(body), headers=headers)
    assert resp.status_code == 200, resp.get_json()
    job = _poll_job(client, resp.get_json()["job_id"])
    assert job["status"] == "complete", job
    result = job["result"]
    assert result["backend"] == "nllb"
    assert result["backend_license"] == "CC-BY-NC-4.0"
    assert result["backend_commercial_safe"] is False
    assert result["restricted_license_accepted"] is True
    assert "not selected automatically" in result["license_notice"]


def test_install_nllb_requires_restricted_license_opt_in(client):
    headers = _csrf_headers(client)
    resp = client.post(
        "/captions/enhanced/install",
        data=json.dumps({"component": "nllb", "no_input": True}),
        headers=headers,
    )

    assert resp.status_code == 400
    assert "accept_restricted_license=true" in resp.get_json().get("error", "")
