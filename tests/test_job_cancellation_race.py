"""F216 regression tests for job cancellation / child-process cleanup."""

from __future__ import annotations

import sys
import threading
import time


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
