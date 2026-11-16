"""Tests for the crash + recovery diagnostic packet (F066)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from opencut.core import crash_packet as cp


def test_build_packet_produces_zip_with_environment(tmp_path):
    out = tmp_path / "packet.zip"
    result = cp.build_packet(output_path=out, include_jobs=False)

    assert Path(result.output_path).exists()
    with zipfile.ZipFile(result.output_path) as zf:
        names = sorted(zf.namelist())
        env_text = zf.read("environment.txt").decode("utf-8")

    assert "environment.txt" in names
    assert "manifest.json" in names
    assert "python_version" in env_text
    assert "platform" in env_text


def test_packet_records_per_entry_hash(tmp_path):
    out = tmp_path / "packet.zip"
    result = cp.build_packet(output_path=out, include_jobs=False)

    arcnames = {entry.arcname for entry in result.entries}
    assert "manifest.json" in arcnames
    assert all(len(entry.sha256) == 64 for entry in result.entries)


def test_packet_includes_crash_log_when_present(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_data = fake_home / ".opencut"
    fake_data.mkdir(parents=True)
    (fake_data / "crash.log").write_text("traceback line\nanother line\n", encoding="utf-8")
    (fake_data / "server.log").write_text("info line\n", encoding="utf-8")

    monkeypatch.setattr(cp, "_redact_sensitive_text", lambda s: s)
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home) if p == "~" else p)

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        names = zf.namelist()
        assert "crash.log" in names
        assert "server.log" in names
        crash = zf.read("crash.log").decode("utf-8")
        assert "traceback line" in crash


def test_packet_scrubs_home_paths(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_data = fake_home / ".opencut"
    fake_data.mkdir(parents=True)
    (fake_data / "server.log").write_text(
        f"reading from {fake_home}/projects/clip.mp4\n", encoding="utf-8"
    )

    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home) if p == "~" else p)
    monkeypatch.setattr(cp, "_HOME" if hasattr(cp, "_HOME") else "DOES_NOT_EXIST", str(fake_home), raising=False)
    # Patch the issue_report symbols the module already imported.
    import re

    import opencut.core.issue_report as ir

    monkeypatch.setattr(ir, "_HOME", str(fake_home))
    monkeypatch.setattr(
        ir,
        "_HOME_PATTERNS",
        [re.compile(re.escape(str(fake_home)), re.IGNORECASE)],
    )

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        log_text = zf.read("server.log").decode("utf-8")
        assert str(fake_home) not in log_text
        assert "~/projects/clip.mp4" in log_text


def test_packet_manifest_lists_only_pre_entries(tmp_path):
    """manifest.json should describe every file *other than itself*."""
    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    arcnames = {entry["arcname"] for entry in manifest["entries"]}
    assert "manifest.json" not in arcnames
    assert manifest["version"] == cp.PACKET_VERSION


@pytest.fixture
def job_sources(tmp_path, monkeypatch):
    """A private jobs.db and an empty in-memory registry."""
    import opencut.job_store as store
    import opencut.jobs as jobs_mod

    monkeypatch.setattr(store, "_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(store, "_INITIALIZED", False)
    monkeypatch.setattr(store, "_INITIALIZED_PATH", None)
    monkeypatch.setattr(store, "_LOCAL", type(store._LOCAL)())
    monkeypatch.setattr(store, "_ALL_CONNECTIONS", {})
    monkeypatch.setattr(jobs_mod, "jobs", {})
    return store, jobs_mod


def _run_job(jobs_mod, store, job_type, filepath, payload, **final):
    """Create a job the way ``async_job`` does, finish it, and wait for
    its terminal row to land in jobs.db."""
    import time

    job_id = jobs_mod._new_job(job_type, filepath)
    with jobs_mod.job_lock:
        jobs_mod.jobs[job_id]["_endpoint"] = f"/{job_type}"
        jobs_mod.jobs[job_id]["_payload"] = jobs_mod._sanitize_payload_for_storage(payload)
    jobs_mod._update_job(job_id, **final)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = store.get_job(job_id)
        if row and row["status"] == final["status"]:
            return job_id
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached jobs.db")


def _two_finished_jobs(job_sources):
    import os

    import opencut.core.issue_report as ir

    store, jobs_mod = job_sources
    home = ir._HOME
    clip = os.path.join(home, "Videos", "interview.mp4")
    broll = os.path.join(home, "Videos", "b-roll.mov")
    first = _run_job(
        jobs_mod, store, "silence", clip,
        # camelCase passes the storage sanitizer, so the raw value is
        # what jobs.db holds and the packet has to catch it.
        {"filepath": clip, "llmApiKey": "LeakCamelJobKey71", "threshold": -30},
        status="complete", progress=100,
        result={"output_path": os.path.join(home, "Videos", "interview_cut.mp4")},
    )
    second = _run_job(
        jobs_mod, store, "captions", broll,
        {"filepath": broll, "openaiApiKey": "LeakCamelJobKey72", "model": "base"},
        status="error",
        error=f"Could not open {broll}",
    )
    return home, first, second


def test_packet_lists_recent_jobs_from_the_store_redacted(tmp_path, job_sources):
    home, first, second = _two_finished_jobs(job_sources)
    _store, jobs_mod = job_sources
    # The registry drops finished jobs after a while; the packet still
    # has to find them in jobs.db.
    jobs_mod.jobs.clear()

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=True)
    with zipfile.ZipFile(result.output_path) as zf:
        assert "recent_jobs.json" in zf.namelist()
        text = zf.read("recent_jobs.json").decode("utf-8")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert "recent_jobs.json" in {e["arcname"] for e in manifest["entries"]}
    jobs = {job["id"]: job for job in json.loads(text)}
    assert set(jobs) == {first, second}
    assert jobs[first]["status"] == "complete"
    assert jobs[second]["status"] == "error"

    from opencut.core.issue_report import REDACTION_MARKER

    assert jobs[first]["payload"]["llmApiKey"] == REDACTION_MARKER
    assert jobs[second]["payload"]["openaiApiKey"] == REDACTION_MARKER
    assert jobs[first]["payload"]["threshold"] == -30
    assert jobs[second]["payload"]["model"] == "base"
    assert "LeakCamelJobKey" not in text
    assert home not in text
    assert home.replace("\\", "\\\\") not in text
    assert jobs[first]["filepath"].endswith("interview.mp4")
    assert jobs[second]["error"].endswith("b-roll.mov")


def test_packet_shows_a_running_job_from_memory(tmp_path, job_sources):
    _store, jobs_mod = job_sources
    running = jobs_mod._new_job("export", "clip.mp4")
    jobs_mod._update_job(running, progress=40, message="Encoding")

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=True)
    with zipfile.ZipFile(result.output_path) as zf:
        jobs = json.loads(zf.read("recent_jobs.json").decode("utf-8"))

    assert [(j["id"], j["status"], j["progress"]) for j in jobs] == [(running, "running", 40)]


def test_packet_without_jobs_has_no_jobs_file(tmp_path, job_sources):
    _two_finished_jobs(job_sources)

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        assert "recent_jobs.json" not in zf.namelist()
    assert "recent_jobs.json" not in {e.arcname for e in result.entries}


def test_route_smoke(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    out = tmp_path / "packet.zip"
    resp = client.post(
        "/system/crash-packet",
        json={
            "output_path": str(out),
            "log_tail_lines": 100,
            "crash_tail_bytes": 4000,
            "include_jobs": False,
        },
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["bundle_sha256"]
    assert Path(payload["output_path"]).exists()


def test_route_requires_output_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/system/crash-packet",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
