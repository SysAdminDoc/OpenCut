"""Tests for F233 review Atom feeds and signed webhook notifications."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

from tests.conftest import csrf_headers

ATOM = "{http://www.w3.org/2005/Atom}"


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
                    "project_id": "project-a",
                    "created_at": 100,
                    "comments": [
                        {
                            "comment_id": "c1",
                            "review_id": "r1",
                            "timestamp": 3.5,
                            "text": "Trim this beat",
                            "author": "Alice",
                            "created_at": 120,
                        }
                    ],
                },
                "r2": {
                    "review_id": "r2",
                    "video_path": str(tmp_path / "other-cut.mp4"),
                    "token": "other-secret",
                    "status": "approved",
                    "title": "Other Cut",
                    "project_id": "project-b",
                    "created_at": 90,
                    "comments": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("opencut.core.review_links._reviews_path", lambda: str(reviews_file))
    return reviews_file


def test_build_review_atom_feed_filters_by_project(tmp_path, monkeypatch):
    from opencut.core.review_notifications import build_review_atom_feed

    _seed_reviews(tmp_path, monkeypatch)
    xml = build_review_atom_feed(project_id="project-a", base_url="http://localhost:5679")
    root = ET.fromstring(xml)
    titles = [node.text for node in root.findall(f"{ATOM}entry/{ATOM}title")]

    assert root.tag == f"{ATOM}feed"
    assert any("Client Cut" in title for title in titles)
    assert not any("Other Cut" in title for title in titles)
    assert "project_id=project-a" in xml


def test_build_review_webhook_details_includes_comment_payload(tmp_path, monkeypatch):
    from opencut.core.review_notifications import build_review_webhook_details

    _seed_reviews(tmp_path, monkeypatch)
    details = build_review_webhook_details(
        review_id="r1",
        event_type="review.comment_added",
        comment={
            "comment_id": "c2",
            "timestamp": 7.0,
            "text": "New note",
            "author": "Bob",
            "created_at": 130,
        },
    )

    assert details["event_type"] == "review.comment_added"
    assert details["project_id"] == "project-a"
    assert details["title"] == "Client Cut"
    assert details["comment"]["text"] == "New note"


def test_review_feed_atom_route(client, tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)

    resp = client.get("/review/feed.atom?project_id=project-a")

    assert resp.status_code == 200
    assert "application/atom+xml" in resp.content_type
    body = resp.get_data(as_text=True)
    assert "Client Cut" in body
    assert "Other Cut" not in body


def test_review_comment_route_fires_review_webhook_event(client, csrf_token, tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)
    calls = []

    def fake_fire_event(event_type, details, job_id=""):
        calls.append((event_type, details, job_id))
        return []

    monkeypatch.setattr("opencut.core.webhook_system.fire_event", fake_fire_event)

    resp = client.post(
        "/review/comment",
        headers=csrf_headers(csrf_token),
        json={
            "review_id": "r1",
            "timestamp": 5.0,
            "text": "Needs a title beat",
            "author": "Casey",
        },
    )

    assert resp.status_code == 200, resp.get_json()
    assert calls
    event_type, details, job_id = calls[-1]
    assert event_type == "review.comment_added"
    assert job_id == "r1"
    assert details["project_id"] == "project-a"
    assert details["comment"]["text"] == "Needs a title beat"


def test_review_status_route_fires_review_webhook_event(client, csrf_token, tmp_path, monkeypatch):
    _seed_reviews(tmp_path, monkeypatch)
    calls = []

    def fake_fire_event(event_type, details, job_id=""):
        calls.append((event_type, details, job_id))
        return []

    monkeypatch.setattr("opencut.core.webhook_system.fire_event", fake_fire_event)

    resp = client.post(
        "/review/status",
        headers=csrf_headers(csrf_token),
        json={"review_id": "r1", "status": "approved"},
    )

    assert resp.status_code == 200, resp.get_json()
    assert calls
    event_type, details, job_id = calls[-1]
    assert event_type == "review.status_changed"
    assert job_id == "r1"
    assert details["status"] == "approved"
