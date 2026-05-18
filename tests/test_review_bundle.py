"""Tests for the portable review bundle export (F105)."""

from __future__ import annotations

import json
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
    assert "markers.otio" in names
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
    assert "markers.otio" in arcnames
    assert "media/render.mp4" in arcnames
    assert "captions/captions.srt" in arcnames
    assert all(len(entry["sha256"]) == 64 for entry in payload["entries"])
    assert payload["otio_markers_basename"] == "markers.otio"


def test_bundle_writes_otio_marker_timeline(sample_assets):
    bundle = rb.build_review_bundle(
        output_path=sample_assets["out"],
        job_label="Client Review",
        media_path=str(sample_assets["media"]),
        markers_payload={
            "comments": [
                {
                    "id": "c1",
                    "text": "Brighten this shot",
                    "author": "Client",
                    "status": "open",
                    "timestamp_sec": 2.5,
                    "duration_seconds": 0.5,
                    "color": "Rose",
                    "annotation_type": "text",
                }
            ]
        },
        framerate=24.0,
        duration_seconds=12.0,
    )

    assert bundle.otio_markers_path == "markers.otio"
    with zipfile.ZipFile(bundle.output_path) as zf:
        otio = json.loads(zf.read("markers.otio").decode("utf-8"))

    assert otio["OTIO_SCHEMA"] == "Timeline.1"
    assert otio["metadata"]["opencut"]["schema"] == "review-bundle-markers.v1"
    assert otio["metadata"]["opencut"]["marker_count"] == 1
    gap = otio["tracks"]["children"][0]["children"][0]
    marker = gap["markers"][0]
    assert marker["OTIO_SCHEMA"] == "Marker.2"
    assert marker["name"] == "Brighten this shot"
    assert marker["color"] == "PINK"
    assert marker["marked_range"]["start_time"]["rate"] == 24.0
    assert marker["marked_range"]["start_time"]["value"] == 60
    assert marker["marked_range"]["duration"]["value"] == 12
    assert marker["metadata"]["opencut"]["status"] == "open"
    assert marker["metadata"]["opencut"]["comment"] == "Brighten this shot"


def test_bundle_writes_svg_drawing_annotations(sample_assets):
    bundle = rb.build_review_bundle(
        output_path=sample_assets["out"],
        job_label="Annotated Review",
        markers_payload={
            "comments": [
                {
                    "id": "rect-1",
                    "text": "Box the logo",
                    "timestamp_sec": 1.0,
                    "annotation_type": "drawing_rect",
                    "annotation_data": {
                        "x": 0.1,
                        "y": 0.2,
                        "w": 0.3,
                        "h": 0.4,
                        "normalized": True,
                        "stroke": "#ff0000",
                        "stroke_width": 6,
                    },
                },
                {
                    "id": "circle-1",
                    "text": "Circle the face",
                    "timestamp_sec": 2.0,
                    "annotation_type": "drawing_circle",
                    "annotation_data": {"cx": 320, "cy": 180, "r": 40, "color": "blue"},
                },
                {
                    "id": "arrow-1",
                    "text": "Point at lower third",
                    "timestamp_sec": 3.0,
                    "annotation_type": "drawing_arrow",
                    "annotation_data": {"x1": 10, "y1": 20, "x2": 300, "y2": 120, "color": "yellow"},
                },
            ]
        },
        framerate=30.0,
        annotation_width=1000,
        annotation_height=500,
    )

    assert bundle.annotations_path == "annotations/index.json"
    assert bundle.annotation_count == 3
    with zipfile.ZipFile(bundle.output_path) as zf:
        names = sorted(zf.namelist())
        index = json.loads(zf.read("annotations/index.json").decode("utf-8"))
        rect_svg = zf.read("annotations/00000030_rect-1.svg").decode("utf-8")
        circle_svg = zf.read("annotations/00000060_circle-1.svg").decode("utf-8")
        arrow_svg = zf.read("annotations/00000090_arrow-1.svg").decode("utf-8")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert "annotations/index.json" in names
    assert index["schema"] == "opencut.review-annotations.v1"
    assert index["width"] == 1000
    assert index["height"] == 500
    assert [ann["frame_number"] for ann in index["annotations"]] == [30, 60, 90]
    assert '<rect x="100.00" y="100.00" width="300.00" height="200.00"' in rect_svg
    assert 'stroke="#ff0000"' in rect_svg
    assert "<circle" in circle_svg
    assert 'stroke="#2563eb"' in circle_svg
    assert "<line" in arrow_svg
    assert "marker-end=" in arrow_svg
    assert manifest["annotations_basename"] == "annotations/index.json"
    assert manifest["annotation_count"] == 3


def test_normalise_review_markers_accepts_single_marker_payload():
    markers = rb.normalise_review_markers(
        {"name": "Hook", "frame_number": 48, "comment": "Check title", "color": "Yellow"},
        framerate=24.0,
    )

    assert markers == [
        {
            "name": "Hook",
            "start_seconds": 2.0,
            "duration_seconds": 0.0,
            "color": "yellow",
            "comment": "Check title",
            "metadata": {"frame_number": 48},
        }
    ]


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
            "framerate": 24.0,
            "duration_seconds": 10.0,
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["bundle_sha256"]
    assert payload["otio_markers_path"] == "markers.otio"
    assert Path(payload["output_path"]).exists()


def test_review_bundle_route_reports_svg_annotations(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    out_path = tmp_path / "annotated.zip"
    resp = client.post(
        "/review/bundle",
        json={
            "output_path": str(out_path),
            "markers_payload": {
                "comments": [
                    {
                        "id": "a1",
                        "text": "Mark region",
                        "timestamp_sec": 1.0,
                        "annotation_type": "drawing_rect",
                        "annotation_data": {"x": 1, "y": 2, "w": 3, "h": 4},
                    }
                ]
            },
            "annotation_width": 640,
            "annotation_height": 360,
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["annotations_path"] == "annotations/index.json"
    assert payload["annotation_count"] == 1


def test_review_bundle_route_requires_output_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/review/bundle",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
