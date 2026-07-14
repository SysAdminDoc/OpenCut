"""Low-cost reliability/security hardening:

- /health exposes the CSRF bootstrap token on an allowlist basis.
- check_disk_space fails open only for network paths, closed for local ones.
- job admission is clamped to the worker-pool size.
"""

import shutil

from opencut import helpers


# --- /health CSRF token allowlist ------------------------------------------

def _expose(app, origin):
    from opencut.routes import system

    headers = {} if origin is None else {"Origin": origin}
    with app.test_request_context("/health", headers=headers):
        return system._health_should_expose_csrf_token()


def test_health_exposes_token_without_origin(app):
    assert _expose(app, None) is True


def test_health_exposes_token_same_origin(app):
    # test_request_context host defaults to localhost.
    assert _expose(app, "http://localhost") is True


def test_health_denies_arbitrary_cross_origin(app):
    assert _expose(app, "https://evil.example") is False


def test_health_denies_blocklisted_origins(app):
    assert _expose(app, "null") is False
    assert _expose(app, "file://") is False


def test_health_allows_configured_cors_origin(app):
    app.config["OPENCUT"].cors_origins = ["https://studio.example"]
    assert _expose(app, "https://studio.example") is True
    assert _expose(app, "https://other.example") is False


# --- check_disk_space fail open/closed -------------------------------------

def test_disk_space_ok_when_space_available(tmp_path):
    result = helpers.check_disk_space(str(tmp_path), min_bytes=1)
    assert result["ok"] is True
    assert result["free_bytes"] > 0


def test_disk_space_probes_nearest_existing_ancestor(tmp_path):
    # A not-yet-created output dir must probe its real mount, not fail.
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    result = helpers.check_disk_space(str(nested), min_bytes=1)
    assert result["ok"] is True


def test_disk_space_fails_closed_on_local_probe_error(monkeypatch, tmp_path):
    def boom(_):
        raise OSError("cannot probe")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    result = helpers.check_disk_space(str(tmp_path), min_bytes=1)
    assert result["ok"] is False  # local path => fail closed
    assert result["probe_failed"] is True


def test_disk_space_fails_open_on_network_probe_error(monkeypatch):
    def boom(_):
        raise OSError("network unreachable")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    result = helpers.check_disk_space(r"\\server\share\out", min_bytes=1)
    assert result["ok"] is True  # UNC/network => fail open
    assert result["probe_failed"] is True


def test_is_network_path():
    assert helpers._is_network_path(r"\\nas\media")
    assert helpers._is_network_path("//nas/media")
    assert not helpers._is_network_path(r"C:\Users\x")
    assert not helpers._is_network_path("/home/x")


# --- job admission clamp ---------------------------------------------------

def test_effective_limit_clamps_to_pool_size(monkeypatch):
    from opencut import jobs

    monkeypatch.setattr(jobs, "MAX_CONCURRENT_JOBS", 100)
    monkeypatch.setattr("opencut.workers.configured_max_workers", lambda default=10: 10)
    assert jobs._effective_concurrent_limit() == 10


def test_effective_limit_respects_lower_configured_limit(monkeypatch):
    from opencut import jobs

    monkeypatch.setattr(jobs, "MAX_CONCURRENT_JOBS", 4)
    monkeypatch.setattr("opencut.workers.configured_max_workers", lambda default=10: 10)
    assert jobs._effective_concurrent_limit() == 4


def test_configured_max_workers_default_without_pool(monkeypatch):
    import opencut.workers as workers

    monkeypatch.setattr(workers, "_pool", None)
    assert workers.configured_max_workers() == 10
