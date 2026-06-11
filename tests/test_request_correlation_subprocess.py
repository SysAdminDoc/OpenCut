"""Tests for request-ID propagation into subprocess execution (N10)."""

from __future__ import annotations

import logging
import subprocess

from flask import Blueprint


def test_subprocess_env_and_prefix_helpers_sanitise_request_id():
    from opencut.core.request_correlation import (
        SUBPROCESS_REQUEST_ID_ENV,
        prefix_subprocess_output,
        subprocess_env,
    )

    env = subprocess_env("r-good\nbad", base_env={"PATH": "demo"})

    assert env["PATH"] == "demo"
    assert env[SUBPROCESS_REQUEST_ID_ENV] == "r-goodbad"
    assert prefix_subprocess_output("one\ntwo", "r-123") == "[r-123] one\n[r-123] two"


def test_run_ffmpeg_tags_env_and_logs_prefixed_failure(monkeypatch, caplog):
    from opencut import helpers
    from opencut.core.request_correlation import clear_request_id, set_request_id

    captured = {}

    def fake_run(cmd, capture_output, timeout, **kwargs):
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout=b"",
            stderr=b"first failure line\nsecond failure line\n",
        )

    monkeypatch.setattr(helpers._sp, "run", fake_run)
    monkeypatch.setattr(helpers, "_ffmpeg_path", "ffmpeg")
    caplog.set_level(logging.WARNING, logger="opencut")
    set_request_id("r-runffmpeg")
    try:
        try:
            helpers.run_ffmpeg(["ffmpeg", "-version"])
        except RuntimeError as exc:
            assert "first failure line" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("run_ffmpeg should raise on non-zero return code")
    finally:
        clear_request_id()

    assert captured["kwargs"]["env"]["OPENCUT_REQUEST_ID"] == "r-runffmpeg"
    assert "[r-runffmpeg] first failure line" in caplog.text
    assert "[r-runffmpeg] second failure line" in caplog.text


def test_progress_runner_tags_env_but_returns_unprefixed_stderr(monkeypatch, caplog):
    from opencut import helpers
    from opencut.core.request_correlation import clear_request_id, set_request_id

    captured = {}

    class FakeStdout:
        def __iter__(self):
            return iter(["out_time_us=1000000\n", "progress=end\n"])

    class FakeStderr:
        def read(self):
            return "progress failure line\n"

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            self.stdout = FakeStdout()
            self.stderr = FakeStderr()
            self.returncode = 1

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(helpers._sp, "Popen", FakeProc)
    caplog.set_level(logging.WARNING, logger="opencut")
    set_request_id("r-progress")
    try:
        rc, stderr = helpers._run_ffmpeg_with_progress("job-progress", ["ffmpeg"], 10.0)
    finally:
        clear_request_id()

    assert rc == 1
    assert stderr == "progress failure line\n"
    assert captured["kwargs"]["env"]["OPENCUT_REQUEST_ID"] == "r-progress"
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert "[r-progress] progress failure line" in caplog.text


def test_async_job_worker_receives_original_request_id(app, monkeypatch):
    import opencut.jobs as jobs_mod
    from opencut.core import job_diagnostics as jd

    class NoopSampler:
        def start(self):
            return self

        def stop(self):
            return {}

    monkeypatch.setattr(jobs_mod, "_persist_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(jd, "JobResourceSampler", NoopSampler)

    bp = Blueprint("n10_request_id", __name__)

    @bp.route("/n10/request-id", methods=["POST"])
    @jobs_mod.async_job("n10_request_id", filepath_required=False)
    def n10_request_id(job_id, filepath, data):
        from opencut.core.request_correlation import get_request_id

        return {"worker_request_id": get_request_id()}

    app.register_blueprint(bp)
    client = app.test_client()
    csrf_token = client.get("/health").get_json()["csrf_token"]
    response = client.post(
        "/n10/request-id",
        json={"ok": True},
        headers={
            "X-OpenCut-Token": csrf_token,
            "X-Request-ID": "client-supplied",
        },
    )
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    job_id = response.get_json()["job_id"]

    with jobs_mod.job_lock:
        future = jobs_mod.jobs[job_id]["_future"]
    future.result(timeout=5)
    live = jobs_mod._get_job_copy(job_id)

    assert request_id.startswith("r-")
    assert request_id != "client-supplied"
    assert live["request_id"] == request_id
    assert live["client_request_id"] == "client-supplied"
    assert live["result"]["worker_request_id"] == request_id
