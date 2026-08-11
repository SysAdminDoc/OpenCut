"""F303 — CSRF bootstrap for host-embedded panels loaded from ``file://``.

The CEP panel document is a ``file://`` URL, so its XHR carries ``Origin: null``
and ``/health`` must refuse it a CSRF token — a hostile local page presents the
same origin. These tests pin both halves of the contract: the real panel
recovers by presenting the 0600 local secret, and a page that cannot read that
file stays refused.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut import panel_bootstrap  # noqa: E402
from opencut.panel_bootstrap import BOOTSTRAP_HEADER, BOOTSTRAP_PATH_ENV  # noqa: E402

BLOCKED_ORIGINS = ("null", "file://", "FILE://", "NULL")


@pytest.fixture
def bootstrap_file(tmp_path, monkeypatch):
    """Point the bootstrap secret at an isolated temp file."""
    target = tmp_path / "panel_bootstrap.token"
    monkeypatch.setenv(BOOTSTRAP_PATH_ENV, str(target))
    panel_bootstrap.clear_cached_secret()
    yield target
    panel_bootstrap.clear_cached_secret()


class TestSecretLifecycle:
    def test_ensure_creates_secret_and_is_idempotent(self, bootstrap_file):
        first = panel_bootstrap.ensure_bootstrap_secret()
        assert first
        assert bootstrap_file.exists()

        panel_bootstrap.clear_cached_secret()
        second = panel_bootstrap.ensure_bootstrap_secret()
        assert second == first, "a restart must not invalidate a live panel's secret"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not available")
    def test_secret_file_is_owner_only(self, bootstrap_file):
        panel_bootstrap.ensure_bootstrap_secret()
        mode = bootstrap_file.lstat().st_mode
        assert not mode & (stat.S_IRWXG | stat.S_IRWXO)

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not available")
    def test_world_readable_secret_is_rejected(self, bootstrap_file):
        panel_bootstrap.ensure_bootstrap_secret()
        bootstrap_file.chmod(0o644)
        panel_bootstrap.clear_cached_secret()
        assert panel_bootstrap.current_secret() is None

    def test_validation_rejects_junk(self, bootstrap_file):
        secret = panel_bootstrap.ensure_bootstrap_secret()
        assert panel_bootstrap.is_bootstrap_secret_valid(secret)
        for candidate in ("", "   ", None, 12345, "x" * 500, secret + "a", secret[:-1]):
            assert not panel_bootstrap.is_bootstrap_secret_valid(candidate)

    def test_disabled_when_path_is_empty(self, monkeypatch):
        monkeypatch.setenv(BOOTSTRAP_PATH_ENV, "")
        panel_bootstrap.clear_cached_secret()
        assert panel_bootstrap.ensure_bootstrap_secret() == ""
        assert not panel_bootstrap.is_bootstrap_secret_valid("anything")

    def test_unwritable_target_degrades_instead_of_raising(self, tmp_path, monkeypatch):
        # A directory where the file should be: writing must fail, not crash startup.
        target = tmp_path / "as_a_directory"
        target.mkdir()
        monkeypatch.setenv(BOOTSTRAP_PATH_ENV, str(target))
        panel_bootstrap.clear_cached_secret()
        assert panel_bootstrap.ensure_bootstrap_secret() == ""


class TestHealthBootstrap:
    """The behaviour issue #5 actually reported."""

    @pytest.mark.parametrize("origin", BLOCKED_ORIGINS)
    def test_opaque_origin_without_secret_is_refused(self, client, origin):
        resp = client.get("/health", headers={"Origin": origin})
        assert resp.status_code == 200
        assert "csrf_token" not in resp.get_json()

    @pytest.mark.parametrize("origin", BLOCKED_ORIGINS)
    def test_opaque_origin_with_secret_bootstraps(self, client, bootstrap_file, origin):
        secret = panel_bootstrap.ensure_bootstrap_secret()
        resp = client.get("/health", headers={"Origin": origin, BOOTSTRAP_HEADER: secret})
        assert resp.status_code == 200
        token = resp.get_json().get("csrf_token")
        assert token, "the host-embedded panel must be able to bootstrap"

    def test_wrong_secret_stays_refused(self, client, bootstrap_file):
        panel_bootstrap.ensure_bootstrap_secret()
        resp = client.get(
            "/health",
            headers={"Origin": "null", BOOTSTRAP_HEADER: "0" * 64},
        )
        assert "csrf_token" not in resp.get_json()

    def test_no_origin_still_bootstraps(self, client):
        resp = client.get("/health")
        assert resp.get_json().get("csrf_token")

    def test_unrelated_cross_origin_is_refused(self, client, bootstrap_file):
        panel_bootstrap.ensure_bootstrap_secret()
        resp = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "csrf_token" not in resp.get_json()

    def test_bootstrapped_token_actually_authorises_a_mutation(self, client, bootstrap_file):
        """End-to-end: the whole point is that mutations stop 403-ing."""
        secret = panel_bootstrap.ensure_bootstrap_secret()
        health = client.get("/health", headers={"Origin": "null", BOOTSTRAP_HEADER: secret})
        token = health.get_json()["csrf_token"]

        refused = client.post("/settings/loudness-target", json={"target_lufs": -14.0})
        assert refused.status_code == 403

        allowed = client.post(
            "/settings/loudness-target",
            json={"target_lufs": -14.0},
            headers={"X-OpenCut-Token": token},
        )
        assert allowed.status_code != 403


class TestBootstrapAudit:
    def test_withheld_bootstrap_is_recorded(self, client, monkeypatch):
        recorded = []

        import opencut.security_audit as security_audit

        def _capture(event, reason, **kwargs):
            recorded.append((event, kwargs.get("metadata") or {}))
            return {"written": False}

        monkeypatch.setattr(security_audit, "record_security_event", _capture)

        client.get("/health", headers={"Origin": "null"})

        assert recorded, "a refused bootstrap must leave a diagnosable trace"
        event, metadata = recorded[0]
        assert event == "csrf_bootstrap_withheld"
        assert metadata.get("origin") == "null"
        assert metadata.get("panel_bootstrap_presented") is False

    def test_successful_bootstrap_records_nothing(self, client, bootstrap_file, monkeypatch):
        recorded = []

        import opencut.security_audit as security_audit

        monkeypatch.setattr(
            security_audit,
            "record_security_event",
            lambda event, reason, **kw: recorded.append(event) or {"written": False},
        )

        secret = panel_bootstrap.ensure_bootstrap_secret()
        client.get("/health", headers={"Origin": "null", BOOTSTRAP_HEADER: secret})
        assert not recorded
