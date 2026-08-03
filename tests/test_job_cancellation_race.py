"""F216 regression tests for job cancellation / child-process cleanup."""

from __future__ import annotations

import sys
import threading
import time

from flask import Blueprint


def _isolate_job_store(monkeypatch, tmp_path):
    import opencut.job_store as store

    store.close_all_connections()
    monkeypatch.setattr(store, "_DB_PATH", str(tmp_path / "jobs.db"))
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    store._LOCAL = type(store._LOCAL)()
    store._ALL_CONNECTIONS = {}
    return store

def _wait_for_registered_process(job_id: str, *, timeout: float = 5.0):
    from opencut.jobs import _job_processes, job_lock

    deadline = time.time() + timeout
    while time.time() < deadline:
        with job_lock:
            proc = _job_processes.get(job_id)
        if proc is not None:
            return proc
        time.sleep(0.01)
    raise AssertionError(f"job process for {job_id} was not registered")


def test_cancel_job_terminates_registered_progress_process(monkeypatch):
    """Cancelling a running progress job must kill and unregister its child.

    This uses a sleeping Python process as a deterministic stand-in for a long
    FFmpeg render. `_run_ffmpeg_with_progress()` appends the same
    `-progress pipe:1` arguments it would append for FFmpeg; Python treats them
    as script argv while the process sleeps.
    """
    from opencut.helpers import _run_ffmpeg_with_progress
    from opencut.jobs import _cancel_job, _job_processes, _new_job, job_lock, jobs

    monkeypatch.setattr("opencut.jobs._persist_job", lambda *args, **kwargs: None)

    job_id = _new_job("race-test", "sleeping-process")
    results: list[tuple[int, str]] = []
    errors: list[BaseException] = []

    def _runner():
        try:
            results.append(
                _run_ffmpeg_with_progress(
                    job_id,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    duration_sec=30.0,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    try:
        thread.start()
        proc = _wait_for_registered_process(job_id)

        cancelled_job, state = _cancel_job(job_id, message="race cancel", persist_sync=True)

        thread.join(timeout=8)

        assert state == "cancelled"
        assert cancelled_job is not None
        assert cancelled_job["status"] == "cancelled"
        assert not thread.is_alive(), "progress runner stayed blocked after cancellation"
        assert not errors
        assert results
        assert results[0][0] != 0
        assert proc.poll() is not None
        with job_lock:
            assert _job_processes.get(job_id) is None
            assert jobs[job_id]["status"] == "cancelled"
    finally:
        if thread.is_alive():
            _cancel_job(job_id, message="test cleanup", persist_sync=True)
            thread.join(timeout=3)
        with job_lock:
            _job_processes.pop(job_id, None)
            jobs.pop(job_id, None)


def test_cleanup_old_jobs_kills_process_for_timed_out_job(monkeypatch):
    import opencut.jobs as jobs_mod

    class FakeProcess:
        stdin = None
        stdout = None
        stderr = None

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.wait_timeouts = []

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            return 0

        def kill(self):
            self.killed = True

    job_id = "stuck-process"
    proc = FakeProcess()
    persisted = []

    monkeypatch.setattr(jobs_mod, "_JOB_STUCK_TIMEOUT", 1)
    monkeypatch.setattr(
        jobs_mod,
        "_persist_job",
        lambda job_dict, **_kwargs: persisted.append(job_dict.copy()),
    )

    with jobs_mod.job_lock:
        jobs_mod.jobs[job_id] = {
            "id": job_id,
            "type": "render",
            "filepath": "clip.mp4",
            "status": "running",
            "progress": 10,
            "created": time.time() - 10,
        }
        jobs_mod._job_processes[job_id] = proc

    try:
        jobs_mod._cleanup_old_jobs()

        assert proc.terminated is True
        assert proc.killed is False
        with jobs_mod.job_lock:
            assert job_id not in jobs_mod._job_processes
            assert jobs_mod.jobs[job_id]["status"] == "error"
            assert jobs_mod.jobs[job_id]["message"] == "Timed out"
        assert persisted
        assert persisted[0]["id"] == job_id
        assert persisted[0]["status"] == "error"
    finally:
        with jobs_mod.job_lock:
            jobs_mod._job_processes.pop(job_id, None)
            jobs_mod.jobs.pop(job_id, None)


def test_cleanup_old_terminal_job_uses_completed_at_for_status_retention(app, monkeypatch):
    import opencut.jobs as jobs_mod

    job_id = "recently-finished-long-job"
    now = time.time()
    monkeypatch.setattr(jobs_mod, "JOB_MAX_AGE", 3600)
    with jobs_mod.job_lock:
        jobs_mod.jobs[job_id] = {
            "id": job_id,
            "type": "render",
            "status": "complete",
            "created": now - (90 * 60),
            "completed_at": now - 60,
        }

    try:
        jobs_mod._cleanup_old_jobs()

        response = app.test_client().get(f"/status/{job_id}")
        assert response.status_code == 200
        assert response.get_json()["id"] == job_id
    finally:
        with jobs_mod.job_lock:
            jobs_mod.jobs.pop(job_id, None)


def test_worker_exception_keeps_cancelled_state_and_emits_one_webhook(
    app, monkeypatch, tmp_path
):
    """A cancellation-induced worker error cannot replace the terminal state."""
    store = _isolate_job_store(monkeypatch, tmp_path)
    import opencut.jobs as jobs_mod
    from opencut.core import job_diagnostics as jd

    original_persist = jobs_mod._persist_job

    def sync_persist(job_dict, *, sync=False):
        original_persist(job_dict, sync=True)

    webhook_jobs = []
    monkeypatch.setattr(jobs_mod, "_persist_job", sync_persist)
    monkeypatch.setattr(
        jobs_mod,
        "_emit_job_webhook",
        lambda job_dict: webhook_jobs.append(job_dict.copy()),
    )

    class FakeSampler:
        def start(self):
            return self

        def stop(self):
            return {"peak_rss_mb": 128}

    monkeypatch.setattr(jd, "JobResourceSampler", FakeSampler)

    entered = threading.Event()
    release = threading.Event()
    bp = Blueprint("f216_cancel_lifecycle", __name__)

    @bp.route("/f216/cancel-lifecycle", methods=["POST"])
    @jobs_mod.async_job("f216_cancel_lifecycle", filepath_required=False)
    def cancel_lifecycle(job_id, filepath, data):
        entered.set()
        assert release.wait(5), "test worker was not released"
        raise RuntimeError("worker failed after cancellation")

    app.register_blueprint(bp)
    job_id = ""
    try:
        client = app.test_client()
        csrf = client.get("/health").get_json()["csrf_token"]
        response = client.post(
            "/f216/cancel-lifecycle",
            json={},
            headers={"X-OpenCut-Token": csrf},
        )
        assert response.status_code == 200, response.get_json()
        job_id = response.get_json()["job_id"]
        assert entered.wait(5), "worker did not start"

        with jobs_mod.job_lock:
            future = jobs_mod.jobs[job_id]["_future"]

        cancelled_job, state = jobs_mod._cancel_job(job_id, persist_sync=True)
        assert state == "cancelled"
        assert cancelled_job is not None
        assert cancelled_job["status"] == "cancelled"

        release.set()
        future.result(timeout=5)

        live = jobs_mod._get_job_copy(job_id)
        persisted = store.get_job(job_id)
        for record in (live, persisted):
            assert record["status"] == "cancelled"
            assert record["exit_reason"] == "cancelled"
            assert record["peak_rss_mb"] == 128
        assert [job["status"] for job in webhook_jobs] == ["cancelled"]
    finally:
        release.set()
        with jobs_mod.job_lock:
            jobs_mod._job_processes.pop(job_id, None)
            jobs_mod.jobs.pop(job_id, None)
