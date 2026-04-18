import threading
from unittest.mock import patch

import pytest


def test_worker_pool_shutdown_cancels_queued_jobs():
    from opencut.workers import WorkerPool

    started = threading.Event()
    release = threading.Event()

    def slow_job():
        started.set()
        release.wait(timeout=2)
        return "done"

    pool = WorkerPool(max_workers=1)
    running = pool.submit("running-job", slow_job)
    assert started.wait(timeout=1)

    queued = pool.submit("queued-job", lambda: "should-not-run")

    with patch("opencut.workers._update_job") as update_job:
        timer = threading.Timer(0.1, release.set)
        timer.start()
        try:
            pool.shutdown(wait=True)
        finally:
            timer.cancel()
            release.set()

    assert running.result(timeout=1) == "done"
    assert queued.cancelled()
    update_job.assert_any_call(
        "queued-job",
        status="cancelled",
        message="Cancelled during server shutdown",
    )


def test_worker_pool_rejects_submit_after_shutdown():
    from opencut.workers import WorkerPool

    pool = WorkerPool(max_workers=1)
    pool.shutdown(wait=True)

    with pytest.raises(RuntimeError, match="shut down"):
        pool.submit("after-shutdown", lambda: None)
