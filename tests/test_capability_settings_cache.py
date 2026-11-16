"""Settings that /health's capabilities read must show on the very next poll.

The cheap capability facts are cached for 30 s. Saving Whisper's CPU mode
or connecting a social account left /health on the old value until the
cache ran out, so the panel redrew the Whisper card from stale state.
"""

import json

from tests.conftest import csrf_headers


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
