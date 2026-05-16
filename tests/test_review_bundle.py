"""Tests for the portable review bundle export (F105)."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from opencut.core import review_bundle as rb


@pytest.fixture()
def sample_assets(tmp_path: Path):
    media = tmp_path / "render.mp4"
    media.write_bytes(b"\x00\x01\x02\x03demo media bytes")
    captions = tmp_path / "captions.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    extra = tmp_path / "lut.cube"
    extra.write_text("# LUT placeholder\n", encoding="utf-8")
    return {
        "media": media,
        "captions": captions,
        "extra": extra,
        "out": tmp_path / "review.zip",
    }


def test_bundle_contains_expected_files(sample_assets):
    bundle = rb.build_review_bundle(
        output_path=sample_assets["out"],
        job_label="Rough Cut",
        media_path=str(sample_assets["media"]),
        captions_path=str(sample_assets["captions"]),
        markers_payload={"markers": [{"name": "Hook", "start_seconds": 1.5}]},
        notes="Reviewer notes here",
        extra_files=[str(sample_assets["extra"])],
    )

    assert Path(bundle.output_path).exists()
    with zipfile.ZipFile(bundle.output_path) as zf:
        names = sorted(zf.namelist())
    assert "manifest.json" in names
    assert "summary.html" in names
    assert "markers.json" in names
    assert "captions/captions.srt" in names
    assert "media/render.mp4" in names
    assert "extras/lut.cube" in names


def test_bundle_manifest_records_per_file_hashes(sample_assets):
    bundle = rb.build_review_bundle(
        output_path=sample_assets["out"],
        media_path=str(sample_assets["media"]),
        captions_path=str(sample_assets["captions"]),
        markers_payload={"markers": []},
        notes="",
    )

    with zipfile.ZipFile(bundle.output_path) as zf:
        payload = json.loads(zf.read("manifest.json").decode("utf-8"))

    arcnames = {entry["arcname"] for entry in payload["entries"]}
    # The manifest doesn't list itself (would create a circular hash);
    # everything else must be present with a sha-256.
    assert "summary.html" in arcnames
    assert "markers.json" in arcnames
    assert "media/render.mp4" in arcnames
    assert "captions/captions.srt" in arcnames
    assert all(len(entry["sha256"]) == 64 for entry in payload["entries"])


def test_bundle_omits_media_when_include_media_false(sample_assets):
    bundle = rb.build_review_bundle(
        output_path=sample_assets["out"],
        media_path=str(sample_assets["media"]),
        captions_path=str(sample_assets["captions"]),
        include_media=False,
    )

    with zipfile.ZipFile(bundle.output_path) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert "media/render.mp4" not in names
    media_entries = [e for e in manifest["entries"] if e["arcname"].startswith("media/")]
    assert media_entries and "omitted" in media_entries[0]["note"]


def test_bundle_is_deterministic_for_same_inputs(sample_assets, tmp_path):
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"

    rb.build_review_bundle(
        output_path=out_a,
        media_path=str(sample_assets["media"]),
        captions_path=str(sample_assets["captions"]),
        markers_payload={"markers": [{"name": "x", "start_seconds": 0.5}]},
        notes="hi",
        include_media=True,
    )
    rb.build_review_bundle(
        output_path=out_b,
        media_path=str(sample_assets["media"]),
        captions_path=str(sample_assets["captions"]),
        markers_payload={"markers": [{"name": "x", "start_seconds": 0.5}]},
        notes="hi",
        include_media=True,
    )

    # Bundles include a generated_at timestamp inside manifest.json, so
    # we can't compare the full zip bytes for equality. We assert the
    # per-file ordering and structure are stable instead.
    with zipfile.ZipFile(out_a) as za, zipfile.ZipFile(out_b) as zb:
        assert za.namelist() == zb.namelist()
        # Non-manifest files must be byte-identical because they come
        # straight from the source files.
        for arcname in za.namelist():
            if arcname in ("manifest.json", "summary.html"):
                continue
            assert za.read(arcname) == zb.read(arcname), arcname


def test_missing_media_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        rb.build_review_bundle(
            output_path=tmp_path / "bundle.zip",
            media_path=str(tmp_path / "does_not_exist.mp4"),
        )


def test_sanitise_arcname_strips_traversal():
    assert rb._sanitise_arcname("../etc/passwd") == "etc/passwd"
    assert rb._sanitise_arcname("//abs/path/file") == "abs/path/file"
    assert rb._sanitise_arcname("\\windows\\path\\file") == "windows/path/file"
    assert rb._sanitise_arcname("./relative/./path") == "relative/path"


# ---- Route smoke ----------------------------------------------------------


def test_review_bundle_route(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    media = tmp_path / "demo.mp4"
    media.write_bytes(b"media-bytes")
    out_path = tmp_path / "out.zip"

    resp = client.post(
        "/review/bundle",
        json={
            "output_path": str(out_path),
            "job_label": "Test",
            "media_path": str(media),
            "notes": "demo",
            "markers_payload": {"markers": []},
            "include_media": True,
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["bundle_sha256"]
    assert Path(payload["output_path"]).exists()


def test_review_bundle_route_requires_output_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/review/bundle",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
