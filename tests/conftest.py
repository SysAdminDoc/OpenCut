"""
Shared pytest fixtures for OpenCut integration tests.
"""

import pytest


@pytest.fixture
def app():
    """Create a Flask app instance configured for testing."""
    from opencut.server import app as flask_app
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
