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
