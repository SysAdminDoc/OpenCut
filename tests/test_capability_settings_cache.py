"""Settings that /health's capabilities read must show on the very next poll.

The cheap capability facts are cached for 30 s. Saving Whisper's CPU mode
or connecting a social account left /health on the old value until the
cache ran out, so the panel redrew the Whisper card from stale state.
"""

import json
import os

import pytest

from tests.conftest import csrf_headers


@pytest.fixture(autouse=True)
def _restore_whisper_settings():
    """Both CPU mode tests end with it saved on, in the home every later test
    on this worker shares. Left there, it sent the next faster-whisper test
    to the CPU, so put the file back the way the test found it."""
    from opencut import user_data

    path = user_data._safe_user_filepath("whisper_settings.json")
    before = None
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            before = handle.read()
    yield
    if before is None:
        if os.path.exists(path):
            os.unlink(path)
    else:
        with open(path, "wb") as handle:
            handle.write(before)


def _caps(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    return resp.get_json()["capabilities"]


def test_whisper_cpu_mode_reaches_health_on_the_next_poll(client, csrf_token):
    for cpu_mode in (True, False, True):
        assert isinstance(_caps(client)["whisper_cpu_mode"], bool)
        resp = client.post(
            "/whisper/settings",
            data=json.dumps({"cpu_mode": cpu_mode}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        assert _caps(client)["whisper_cpu_mode"] is cpu_mode


def test_a_health_build_that_overlaps_a_save_is_not_kept(client, csrf_token, monkeypatch):
    """The server is threaded. A /health that read the settings before a save
    finished building after it, and stored the old value for 30 s over it."""
    from opencut.routes import system

    resp = client.post("/whisper/settings", data=json.dumps({"cpu_mode": False}),
                       headers=csrf_headers(csrf_token))
    assert resp.status_code == 200
    real_build = system._build_capabilities

    def build_across_a_save():
        caps = real_build()
        monkeypatch.setattr(system, "_build_capabilities", real_build)
        # What POST /whisper/settings does, landing mid-build.
        settings = system.load_whisper_settings()
        settings["cpu_mode"] = True
        system.save_whisper_settings(settings)
        system.forget_capability_facts()
        return caps

    monkeypatch.setattr(system, "_build_capabilities", build_across_a_save)
    assert _caps(client)["whisper_cpu_mode"] is False  # what that request read
    assert _caps(client)["whisper_cpu_mode"] is True


def test_social_connect_and_disconnect_reach_health_on_the_next_poll(
    client, csrf_token, monkeypatch,
):
    # The real credentials live in the OS vault, so stand in for it here.
    from opencut import checks
    from opencut.core import social_post
    from opencut.routes.system import invalidate_caps_cache

    connected = set()
    monkeypatch.setattr(social_post, "store_auth", lambda platform, **_kw: connected.add(platform))
    monkeypatch.setattr(social_post, "disconnect_platform", connected.discard)
    monkeypatch.setattr(checks, "check_social_post_available", lambda: bool(connected))
    invalidate_caps_cache()

    assert _caps(client)["social_post"] is False
    resp = client.post(
        "/social/connect",
        data=json.dumps({"platform": "youtube", "access_token": "token"}),
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200
    assert _caps(client)["social_post"] is True

    resp = client.post(
        "/social/disconnect",
        data=json.dumps({"platform": "youtube"}),
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200
    assert _caps(client)["social_post"] is False
