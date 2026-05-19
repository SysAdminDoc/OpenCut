"""Tests for F231 local-LAN review portal share links."""

from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlparse

import pytest

from opencut.core import review_portal as rp


def _seed_reviews(tmp_path, monkeypatch):
    reviews_file = tmp_path / "reviews.json"
    reviews_file.write_text(
        json.dumps(
            {
                "r1": {
                    "review_id": "r1",
                    "video_path": str(tmp_path / "rough-cut.mp4"),
                    "token": "secret-token",
                    "status": "pending",
                    "title": "Client Cut",
                    "created_at": 1,
                    "expires_at": None,
                    "comments": [
                        {
                            "comment_id": "c1",
                            "review_id": "r1",
                            "timestamp": 3.5,
                            "text": "Trim this beat",
                            "author": "Alice",
                            "created_at": 2,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("opencut.core.review_links._reviews_path", lambda: str(reviews_file))
    return reviews_file


def test_build_portal_share_returns_hmac_url_and_descriptors(tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)
    now = int(time.time())

    share = rp.build_portal_share(
        review_id="r1",
        host="opencut-review.local",
        port=8080,
        ttl_seconds=3600,
        service_name="OpenCut Client Review",
        now=now,
    )

    parsed = urlparse(share.url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "http"
    assert parsed.netloc == "opencut-review.local:8080"
    assert parsed.path == "/review/portal/r1"
    assert query["expires"] == [str(now + 3600)]
    assert query["sig"][0].startswith("sha256=")
    assert "reverse_proxy 127.0.0.1:5679" in share.caddyfile
    assert share.mdns["service_type"] == "_http._tcp.local."
    assert share.mdns["txt"]["auth"] == "hmac-url"

    payload = rp.resolve_portal_review("r1", int(query["expires"][0]), query["sig"][0])
    assert payload["title"] == "Client Cut"
    assert payload["video_basename"] == "rough-cut.mp4"
    assert payload["comment_count"] == 1
    assert payload["comments"][0]["text"] == "Trim this beat"


def test_build_portal_share_can_include_headscale_plan(tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)

    share = rp.build_portal_share(
        review_id="r1",
        host="opencut-review.local",
        port=8080,
        ttl_seconds=3600,
        headscale={
            "url": "https://headscale.example.net/",
            "user": "post-team",
            "machine_name": "Client Review Node",
            "tags": ["opencut-review", "tag:client-review"],
            "ttl_hours": 12,
        },
        now=100,
    )

    plan = share.headscale
    assert plan["enabled"] is True
    assert plan["control_plane_url"] == "https://headscale.example.net"
    assert plan["user"] == "post-team"
    assert plan["machine_name"] == "Client-Review-Node"
    assert plan["tags"] == ["tag:opencut-review", "tag:client-review"]
    assert plan["portal"]["url"] == share.url
    assert plan["commands"]["create_preauth_key"] == [
        "headscale",
        "--url",
        "https://headscale.example.net",
        "preauthkeys",
        "create",
        "--user",
        "post-team",
        "--expiration",
        "12h",
        "--tags",
        "tag:opencut-review,tag:client-review",
    ]
    assert "--login-server" in plan["commands"]["join_tailnet"]
    assert "secret-token" not in json.dumps(plan)


def test_headscale_plan_rejects_credentials_in_url(tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="credentials"):
        rp.build_portal_share(
            review_id="r1",
            host="opencut-review.local",
            port=8080,
            headscale={"url": "https://user:pass@headscale.example.net"},
        )


def test_portal_signature_rejects_expired_urls():
    sig = rp.sign_portal_url("r1", 10, "secret-token")
    assert not rp.verify_portal_signature("r1", 10, sig, "secret-token", now=11)


def test_portal_share_rejects_hosts_with_schemes(tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        rp.build_portal_share(review_id="r1", host="http://bad", port=5679)


def test_review_portal_share_and_signed_view_routes(client, csrf_token, tmp_path, monkeypatch):
    from tests.conftest import csrf_headers

    _seed_reviews(tmp_path, monkeypatch)
    resp = client.post(
        "/review/portal/share",
        json={"review_id": "r1", "host": "opencut-review.local", "port": 8080, "ttl_seconds": 3600},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200, resp.get_json()
    share = resp.get_json()
    assert share["url"].startswith("http://opencut-review.local:8080/review/portal/r1?")
    assert share["hmac_algorithm"] == "HMAC-SHA256"

    parsed = urlparse(share["url"])
    query = parse_qs(parsed.query)
    view = client.get(
        f"/review/portal/r1?expires={query['expires'][0]}&sig={query['sig'][0]}&format=json"
    )
    assert view.status_code == 200, view.get_json()
    payload = view.get_json()
    assert payload["review_id"] == "r1"
    assert payload["comment_count"] == 1


def test_review_portal_share_route_reports_headscale_plan(client, csrf_token, tmp_path, monkeypatch):
    from tests.conftest import csrf_headers

    _seed_reviews(tmp_path, monkeypatch)
    resp = client.post(
        "/review/portal/share",
        json={
            "review_id": "r1",
            "host": "opencut-review.local",
            "port": 8080,
            "headscale": {
                "url": "https://headscale.example.net",
                "user": "post-team",
                "tags": ["opencut-review"],
            },
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    share = resp.get_json()
    assert share["headscale"]["control_plane_url"] == "https://headscale.example.net"
    assert share["headscale"]["commands"]["join_tailnet"][:4] == [
        "tailscale",
        "up",
        "--login-server",
        "https://headscale.example.net",
    ]


def test_review_portal_view_rejects_bad_signature(client, tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)
    resp = client.get(f"/review/portal/r1?expires={int(time.time()) + 100}&sig=sha256=bad&format=json")
    assert resp.status_code == 403
