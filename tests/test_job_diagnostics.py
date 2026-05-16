"""Tests for the job diagnostic surface (F010)."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencut.core import job_diagnostics as jd


def test_build_diagnostic_not_found(monkeypatch):
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: {})

    diag = jd.build_diagnostic("does-not-exist")

    assert diag.found is False
    assert diag.job_id == "does-not-exist"
    assert diag.status == ""


def test_build_diagnostic_merges_persisted_and_live(monkeypatch, tmp_path):
    sample = {
        "status": "running",
        "type": "captions",
        "progress": 42,
        "request_id": "req-123",
        "filepath": "/home/me/clip.mp4",
        "error": "",
        "created_at": 1700000000.0,
        "finished_at": None,
    }
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: sample)
    monkeypatch.setattr(jd, "_tail_log", lambda max_bytes=4 * 1024 * 1024: [])

    diag = jd.build_diagnostic("job-x")

    assert diag.found is True
    assert diag.status == "running"
    assert diag.progress == 42
    assert diag.request_id == "req-123"
    # Filepath in metadata should be scrubbed of the user's home dir
    # (best-effort — only triggers when the running user's home matches).
    assert diag.metadata["filepath"] in {sample["filepath"], "~/clip.mp4"}


def test_build_diagnostic_includes_log_slice(monkeypatch, tmp_path):
    sample = {"status": "done", "type": "render", "request_id": "req-abc"}
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: sample)
    monkeypatch.setattr(
        jd,
        "_tail_log",
        lambda max_bytes=4 * 1024 * 1024: [
            "2026-05-16 12:00:00 INFO unrelated line",
            "2026-05-16 12:00:01 INFO [req-abc] job-y started",
            "2026-05-16 12:00:02 INFO another unrelated line",
            "2026-05-16 12:00:03 INFO [req-abc] job-y finished",
        ],
    )

    diag = jd.build_diagnostic("job-y")

    assert diag.log_slice is not None
    body_lines = diag.log_slice.body.splitlines()
    assert all("req-abc" in line for line in body_lines)
    assert diag.log_slice.line_count == 2


def test_build_diagnostic_respects_log_tail_cap(monkeypatch):
    sample = {"status": "done", "request_id": "req-z"}
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: sample)
    monkeypatch.setattr(
        jd,
        "_tail_log",
        lambda max_bytes=4 * 1024 * 1024: [
            f"line {i} [req-z] job-y" for i in range(50)
        ],
    )

    diag = jd.build_diagnostic("job-y", log_tail_lines=10)

    body_lines = diag.log_slice.body.splitlines()
    # Cap floors to 10 to keep the response cheap.
    assert len(body_lines) == 10


def test_export_diagnostic_writes_json(monkeypatch, tmp_path):
    sample = {"status": "done", "request_id": "req-w", "type": "captions"}
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: sample)
    monkeypatch.setattr(jd, "_tail_log", lambda max_bytes=4 * 1024 * 1024: [])

    out = tmp_path / "diag.json"
    result = jd.export_diagnostic("job-w", out)

    assert out.exists()
    assert result["job_id"] == "job-w"
    assert result["found"] is True
    body = out.read_text(encoding="utf-8")
    assert "req-w" in body


def test_diagnostic_route_returns_404_for_unknown(client, monkeypatch):
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: {})
    monkeypatch.setattr(jd, "_tail_log", lambda max_bytes=4 * 1024 * 1024: [])

    resp = client.get("/jobs/does-not-exist/diagnostics")

    assert resp.status_code == 404
    payload = resp.get_json()
    assert payload["found"] is False


def test_diagnostic_route_returns_payload(client, monkeypatch):
    sample = {"status": "queued", "request_id": "req-known", "type": "render"}
    monkeypatch.setattr(jd, "_resolve_job", lambda job_id: sample)
    monkeypatch.setattr(jd, "_tail_log", lambda max_bytes=4 * 1024 * 1024: [])

    resp = client.get("/jobs/known/diagnostics")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["found"] is True
    assert payload["job_id"] == "known"
    assert payload["request_id"] == "req-known"
