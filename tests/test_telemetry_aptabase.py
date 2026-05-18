"""F250 Aptabase opt-in telemetry tests."""

from __future__ import annotations

import json

import pytest

from tests.conftest import csrf_headers


@pytest.fixture(autouse=True)
def isolated_telemetry_state(monkeypatch, tmp_path):
    from opencut import user_data
    from opencut.core import telemetry_aptabase

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    for name in (
        "OPENCUT_TELEMETRY_ENABLED",
        "OPENCUT_APTABASE_APP_KEY",
        "APTABASE_APP_KEY",
        "OPENCUT_APTABASE_BASE_URL",
        "APTABASE_BASE_URL",
        "OPENCUT_TELEMETRY_DEBUG",
        "OPENCUT_TELEMETRY_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    telemetry_aptabase._reset_for_tests()
    yield
    telemetry_aptabase._reset_for_tests()


def test_f250_aptabase_defaults_to_disabled():
    from opencut.core import telemetry_aptabase

    cfg = telemetry_aptabase.get_config()

    assert cfg.enabled is False
    assert cfg.configured is False
    assert telemetry_aptabase.track("app_started", sync=True) is False
    assert telemetry_aptabase.queue_depth() == 0
    info = telemetry_aptabase.telemetry_info()
    assert info["provider"] == "aptabase"
    assert info["privacy"]["fresh_install_emits"] is False
    assert info["privacy"]["opt_in_required"] is True


def test_f250_settings_mask_app_key_and_resolve_region_host():
    from opencut.core import telemetry_aptabase

    settings = telemetry_aptabase.update_settings({
        "enabled": True,
        "app_key": "A-US-1234567890",
        "include_diagnostics": True,
    })

    public = telemetry_aptabase.public_settings(settings)
    cfg = telemetry_aptabase.get_config()
    assert cfg.enabled is True
    assert cfg.base_url == "https://us.aptabase.com"
    assert public["app_key_set"] is True
    assert public["app_key_masked"] == "A-US-...7890"
    assert "1234567890" not in json.dumps(public)
    assert public["include_diagnostics"] is True


def test_f250_self_hosted_key_requires_public_base_url():
    from opencut.core import telemetry_aptabase

    with pytest.raises(ValueError, match="base_url"):
        telemetry_aptabase.update_settings({
            "enabled": True,
            "app_key": "A-SH-1234567890",
        })

    with pytest.raises(ValueError, match="localhost"):
        telemetry_aptabase.update_settings({
            "enabled": True,
            "app_key": "A-SH-1234567890",
            "base_url": "http://localhost:8000",
        })

    settings = telemetry_aptabase.update_settings({
        "enabled": True,
        "app_key": "A-SH-1234567890",
        "base_url": "https://telemetry.example.com",
    })
    assert settings["base_url"] == "https://telemetry.example.com"
    assert telemetry_aptabase.get_config().endpoint == (
        "https://telemetry.example.com/api/v0/events"
    )


def test_f250_event_payload_scrubs_sensitive_props():
    from opencut.core import telemetry_aptabase

    cfg = telemetry_aptabase.AptabaseConfig(
        enabled=True,
        configured=True,
        app_key="A-EU-1234567890",
        base_url="https://eu.aptabase.com",
        endpoint="https://eu.aptabase.com/api/v0/events",
        app_version="test",
        is_debug=False,
        timeout=1.0,
        include_diagnostics=False,
        source="test",
    )

    event = telemetry_aptabase.build_event(
        "feature used!?",
        props={
            "feature_id": "captions.qc",
            "duration_ms": 1.23456789,
            "file_path": r"C:\Users\me\video.mp4",
            "mediaPath": "/Users/me/video.mp4",
            "profile": "caption_qc",
            "prompt": "summarize this transcript",
            "link": "https://example.com/private",
            "_internal": "drop",
        },
        cfg=cfg,
    )

    assert event["eventName"] == "feature_used"
    assert event["timestamp"].endswith("Z")
    assert event["sessionId"].isdigit()
    assert event["props"]["feature_id"] == "captions.qc"
    assert event["props"]["duration_ms"] == 1.234568
    assert event["props"]["profile"] == "caption_qc"
    assert "file_path" not in event["props"]
    assert "mediaPath" not in event["props"]
    assert "prompt" not in event["props"]
    assert "link" not in event["props"]
    assert "_internal" not in event["props"]
    assert event["systemProps"]["sdkVersion"] == "opencut-aptabase@1"


def test_f250_sync_track_posts_aptabase_batch(monkeypatch):
    from opencut.core import telemetry_aptabase

    telemetry_aptabase.update_settings({
        "enabled": True,
        "app_key": "A-EU-1234567890",
    })
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _n):
            return b""

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert telemetry_aptabase.track(
        "app_started", props={"feature_id": "system.health"}, sync=True
    )

    assert len(calls) == 1
    req, timeout = calls[0]
    payload = json.loads(req.data.decode("utf-8"))
    assert req.full_url == "https://eu.aptabase.com/api/v0/events"
    assert req.get_header("App-key") == "A-EU-1234567890"
    assert timeout == 4.0
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["eventName"] == "app_started"
    assert payload[0]["props"]["feature_id"] == "system.health"


def test_f250_routes_expose_info_and_require_csrf_for_changes(client, csrf_token):
    info = client.get("/telemetry/aptabase/info")
    assert info.status_code == 200
    assert info.get_json()["enabled"] is False

    no_csrf = client.post(
        "/telemetry/aptabase/settings",
        json={"enabled": True, "app_key": "A-US-1234567890"},
    )
    assert no_csrf.status_code == 403

    bad = client.post(
        "/telemetry/aptabase/settings",
        json={"enabled": True},
        headers=csrf_headers(csrf_token),
    )
    assert bad.status_code == 400

    saved = client.post(
        "/telemetry/aptabase/settings",
        json={"enabled": True, "app_key": "A-US-1234567890"},
        headers=csrf_headers(csrf_token),
    )
    assert saved.status_code == 200
    saved_json = saved.get_json()
    assert saved_json["success"] is True
    assert saved_json["settings"]["app_key_masked"] == "A-US-...7890"
    assert "1234567890" not in json.dumps(saved_json["settings"])

    current = client.get("/telemetry/aptabase/settings")
    assert current.status_code == 200
    assert current.get_json()["app_key_set"] is True


def test_f250_track_route_queues_nothing_when_disabled(client, csrf_token):
    response = client.post(
        "/telemetry/aptabase/track",
        json={"event_name": "route.test", "props": {"feature_id": "demo"}},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["provider"] == "aptabase"
    assert payload["enabled"] is False
    assert payload["queued"] is False
    assert payload["queue_depth"] == 0
