"""Acceptance coverage for immutable review artifact versions."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests.conftest import csrf_headers


def _use_review_store(tmp_path, monkeypatch):
    reviews_file = tmp_path / "reviews.json"
    monkeypatch.setattr("opencut.core.review_links._reviews_path", lambda: str(reviews_file))
    return reviews_file


def test_review_versions_keep_artifacts_and_feedback_immutable(tmp_path, monkeypatch):
    from opencut.core.review_links import (
        add_review_comment,
        add_review_version,
        create_review_link,
        export_review_data,
        get_review,
        get_review_comments,
        get_review_versions,
        update_review_status,
    )

    _use_review_store(tmp_path, monkeypatch)
    first_render = tmp_path / "rough-cut.mp4"
    second_render = tmp_path / "fine-cut.mp4"
    first_render.write_bytes(b"first immutable render")
    second_render.write_bytes(b"second immutable render")

    review = create_review_link(str(first_render), title="Client cut")
    first_version = review.versions[0]
    first_render.write_bytes(b"source file was overwritten")
    assert open(first_version.video_path, "rb").read() == b"first immutable render"
    assert first_version.artifact_sha256 == hashlib.sha256(b"first immutable render").hexdigest()

    review = add_review_version(review.review_id, str(second_render), label="Color pass")
    assert review.current_version_id == "v2"
    assert [version.version_id for version in get_review_versions(review.review_id)] == ["v1", "v2"]

    add_review_comment(review.review_id, 1.5, "Keep the first grade", "Alice", version_id="v1")
    add_review_comment(review.review_id, 2.5, "Color is final", "Bob", version_id="v2")
    update_review_status(review.review_id, "approved", version_id="v1")
    update_review_status(review.review_id, "revision_requested", version_id="v2")

    assert [comment.text for comment in get_review_comments(review.review_id, version_id="v1")] == [
        "Keep the first grade"
    ]
    assert get_review(review.review_id, review.token, version_id="v1").status == "approved"
    assert get_review(review.review_id, review.token, version_id="v2").status == "revision_requested"

    export_path = tmp_path / "review-export.json"
    export_review_data(str(export_path))
    exported = json.loads(export_path.read_text(encoding="utf-8"))[review.review_id]
    assert len(exported["versions"]) == 2
    assert {comment["version_id"] for comment in exported["comments"]} == {"v1", "v2"}


def test_legacy_migration_backup_and_rollback_are_lossless(tmp_path, monkeypatch):
    from opencut.core.review_links import _load_reviews, rollback_review_migration

    reviews_file = _use_review_store(tmp_path, monkeypatch)
    legacy = {
        "legacy-review": {
            "review_id": "legacy-review",
            "video_path": str(tmp_path / "legacy.mp4"),
            "token": "legacy-token",
            "status": "approved",
            "title": "Legacy cut",
            "project_id": "client-a",
            "created_at": 0,
            "status_updated_at": 12,
            "expires_at": None,
            "comments": [
                {
                    "comment_id": "c1",
                    "review_id": "legacy-review",
                    "timestamp": 4.25,
                    "text": "Preserve this note",
                    "author": "Reviewer",
                    "created_at": 5,
                }
            ],
        }
    }
    reviews_file.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = _load_reviews()["legacy-review"]
    backup_path = tmp_path / "reviews.pre-versioning.json"
    assert json.loads(backup_path.read_text(encoding="utf-8")) == legacy
    assert migrated["schema_version"] == 2
    assert migrated["versions"][0]["version_id"] == "v1"
    assert migrated["comments"][0]["version_id"] == "v1"
    assert migrated["comments"][0]["text"] == "Preserve this note"

    migrated["comments"].append({"comment_id": "new-feedback", "version_id": "v1"})
    reviews_file.write_text(json.dumps({"legacy-review": migrated}), encoding="utf-8")
    assert rollback_review_migration() == str(backup_path)
    assert json.loads(reviews_file.read_text(encoding="utf-8")) == legacy


def test_guest_link_scopes_versions_and_renders_comparison(tmp_path, monkeypatch):
    from opencut.core.review_links import add_review_comment, add_review_version, create_review_link
    from opencut.core.review_portal import (
        build_portal_share,
        render_portal_html,
        resolve_portal_media,
        resolve_portal_review,
    )

    _use_review_store(tmp_path, monkeypatch)
    first_render = tmp_path / "v1.mp4"
    second_render = tmp_path / "v2.mp4"
    first_render.write_bytes(b"version one")
    second_render.write_bytes(b"version two")
    review = create_review_link(str(first_render), title="Scoped review")
    review = add_review_version(review.review_id, str(second_render))
    add_review_comment(review.review_id, 3, "Only on v1", "Alice", version_id="v1")
    add_review_comment(review.review_id, 4, "Only on v2", "Bob", version_id="v2")

    share = build_portal_share(
        review_id=review.review_id,
        host="review.local",
        port=5679,
        version_ids=["v1"],
    )
    query = parse_qs(urlparse(share.url).query)
    payload = resolve_portal_review(review.review_id, int(query["expires"][0]), query["sig"][0], ["v1"])
    assert payload["version_ids"] == ["v1"]
    assert [comment["text"] for comment in payload["comments"]] == ["Only on v1"]
    with pytest.raises(PermissionError):
        resolve_portal_review(review.review_id, int(query["expires"][0]), query["sig"][0], ["v2"])
    with pytest.raises(PermissionError):
        resolve_portal_media(review.review_id, int(query["expires"][0]), query["sig"][0], "v2", ["v1"])

    compare_share = build_portal_share(
        review_id=review.review_id,
        host="review.local",
        port=5679,
        version_ids=["v1", "v2"],
    )
    compare_query = parse_qs(urlparse(compare_share.url).query)
    compare_payload = resolve_portal_review(
        review.review_id,
        int(compare_query["expires"][0]),
        compare_query["sig"][0],
        ["v1", "v2"],
    )
    for version in compare_payload["versions"]:
        version["media_url"] = f"/media/{version['version_id']}"
    page = render_portal_html(compare_payload)
    assert 'aria-label="Version comparison"' in page
    assert "/media/v1" in page and "/media/v2" in page
    assert "Version A" in page and "Version B" in page


def test_portal_route_serves_only_signed_version_media(client, csrf_token, tmp_path, monkeypatch):
    from opencut.core.review_links import add_review_version, create_review_link

    _use_review_store(tmp_path, monkeypatch)
    first_render = tmp_path / "first.mp4"
    second_render = tmp_path / "second.mp4"
    first_render.write_bytes(b"first portal artifact")
    second_render.write_bytes(b"second portal artifact")
    review = create_review_link(str(first_render), title="Portal versions")
    add_review_version(review.review_id, str(second_render))

    response = client.post(
        "/review/portal/share",
        json={
            "review_id": review.review_id,
            "host": "review.local",
            "port": 5679,
            "version_ids": ["v1", "v2"],
        },
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 200, response.get_json()
    share = response.get_json()
    assert share["version_ids"] == ["v1", "v2"]
    parsed = urlparse(share["url"])

    json_response = client.get(f"{parsed.path}?{parsed.query}&format=json")
    assert json_response.status_code == 200, json_response.get_json()
    payload = json_response.get_json()
    assert [version["version_id"] for version in payload["versions"]] == ["v1", "v2"]

    first_media = client.get(payload["versions"][0]["media_url"])
    second_media = client.get(payload["versions"][1]["media_url"])
    assert first_media.status_code == 200
    assert first_media.data == b"first portal artifact"
    assert second_media.status_code == 200
    assert second_media.data == b"second portal artifact"

    tampered = client.get(
        f"{parsed.path}?expires={parse_qs(parsed.query)['expires'][0]}"
        f"&sig={parse_qs(parsed.query)['sig'][0]}&versions=v2&format=json"
    )
    assert tampered.status_code == 403
