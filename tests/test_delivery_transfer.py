"""Tests for F234 croc/rclone delivery transfer bundles."""

from __future__ import annotations

import json
import time
import zipfile

import pytest

from tests.conftest import csrf_headers


def _sample_file(tmp_path, name="final.mp4", content=b"delivery"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_list_transfer_methods_reports_croc_and_rclone():
    from opencut.core.delivery_transfer import list_transfer_methods

    methods = {item["method"]: item for item in list_transfer_methods()}

    assert {"croc", "rclone"}.issubset(methods)
    assert methods["croc"]["binary"] == "croc"
    assert methods["rclone"]["binary"] == "rclone"
    assert "available" in methods["croc"]


def test_prepare_transfer_bundle_creates_zip_manifest_and_croc_command(tmp_path):
    from opencut.core.delivery_transfer import prepare_transfer_bundle

    source_a = _sample_file(tmp_path, "final.mp4", b"video")
    source_b = _sample_file(tmp_path, "captions.srt", b"1\n")
    bundle = tmp_path / "client_delivery.zip"

    result = prepare_transfer_bundle(
        paths=[str(source_a), str(source_b)],
        output_path=str(bundle),
        method="croc",
        croc_code="client-cut",
    )

    assert result.bundle_path == str(bundle)
    assert result.source_count == 2
    assert result.total_source_bytes == len(b"video") + len(b"1\n")
    assert result.commands[0].method == "croc"
    assert result.commands[0].argv[:2] == ["croc", "send"]
    assert "--code" in result.commands[0].argv

    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        assert "delivery_transfer_manifest.json" in names
        manifest = json.loads(zf.read("delivery_transfer_manifest.json").decode("utf-8"))
    assert manifest["schema"] == "opencut.delivery-transfer.v1"
    assert manifest["methods"] == ["croc"]
    assert len(manifest["files"]) == 2


def test_prepare_transfer_bundle_builds_rclone_command(tmp_path):
    from opencut.core.delivery_transfer import prepare_transfer_bundle

    source = _sample_file(tmp_path)
    bundle = tmp_path / "cloud_delivery.zip"

    result = prepare_transfer_bundle(
        paths=[str(source)],
        output_path=str(bundle),
        method=["rclone"],
        rclone_remote="s3:client-bucket/deliveries",
        rclone_path="rough-cut",
    )

    command = result.commands[0]
    assert command.method == "rclone"
    assert command.argv == [
        "rclone",
        "copy",
        str(bundle),
        "s3:client-bucket/deliveries/rough-cut",
    ]


def test_prepare_transfer_bundle_requires_rclone_remote(tmp_path):
    from opencut.core.delivery_transfer import prepare_transfer_bundle

    source = _sample_file(tmp_path)

    with pytest.raises(ValueError, match="rclone_remote"):
        prepare_transfer_bundle(paths=[str(source)], output_path=str(tmp_path / "x.zip"), method="rclone")


def test_transfer_options_route(client):
    resp = client.get("/delivery/transfer/options")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["count"] == 2
    assert {item["method"] for item in payload["methods"]} == {"croc", "rclone"}


def test_transfer_bundle_route_prevalidates_missing_paths(client, csrf_token):
    resp = client.post(
        "/delivery/transfer-bundle",
        headers=csrf_headers(csrf_token),
        json={"method": "croc"},
    )

    assert resp.status_code == 400
    assert "paths or filepath" in resp.get_json()["error"]


def test_transfer_bundle_route_creates_job(client, csrf_token, tmp_path):
    source = _sample_file(tmp_path)
    output_path = tmp_path / "route_bundle.zip"

    resp = client.post(
        "/delivery/transfer-bundle",
        headers=csrf_headers(csrf_token),
        json={
            "paths": [str(source)],
            "output_path": str(output_path),
            "method": "croc",
        },
    )

    assert resp.status_code == 200, resp.get_json()
    job_id = resp.get_json()["job_id"]
    final = {}
    for _ in range(20):
        status = client.get(f"/status/{job_id}")
        final = status.get_json() or {}
        if final.get("status") in {"complete", "error", "cancelled"}:
            break
        time.sleep(0.05)
    assert final["status"] == "complete"
    assert output_path.exists()
