"""
Shared pytest fixtures for OpenCut integration tests.
"""

import pytest


@pytest.fixture
def app():
    """Create a Flask app instance configured for testing."""
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    test_config = OpenCutConfig()
    flask_app = create_app(config=test_config)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client -- no real network, no subprocess needed."""
    return app.test_client()


@pytest.fixture
def csrf_token(client):
    """Fetch a valid CSRF token from the /health endpoint."""
    resp = client.get("/health")
    data = resp.get_json()
    return data.get("csrf_token", "")


def csrf_headers(token):
    """Build headers dict with CSRF token and JSON content type."""
    return {
        "X-OpenCut-Token": token,
        "Content-Type": "application/json",
    }


@pytest.fixture(autouse=True, scope="session")
def _shutdown_worker_pool():
    """Shut down the WorkerPool after all tests to prevent pytest hang on exit."""
    yield
    from opencut.workers import shutdown_pool
    shutdown_pool(wait=False)


@pytest.fixture(autouse=True)
def _isolate_global_state():
    """Reset module-level mutable state between tests.

    The hot-path job dict, queue, and process registry in ``opencut.jobs``
    survive across tests because they're module globals. Without this
    fixture, a test that posts to ``/silence`` leaves a "running" entry
    that the next test sees when it queries ``/jobs`` or ``/jobs/stats``.
    The same applies to the in-memory queue in ``routes.jobs_routes`` and
    the per-app caches in ``routes.system``. Cleanup runs *after* each
    test so individual test bodies still see their own writes.
    """
    yield
    try:
        from opencut import jobs as _jobs_mod
        with _jobs_mod.job_lock:
            _jobs_mod.jobs.clear()
            _jobs_mod._job_processes.clear()
    except Exception:
        pass
    try:
        from opencut.routes import jobs_routes as _jr
        with _jr.job_queue_lock:
            _jr.job_queue.clear()
            _jr._queue_state["running"] = False
    except Exception:
        pass
    try:
        from opencut.routes.system import invalidate_caps_cache
        invalidate_caps_cache()
    except Exception:
        pass
