"""Contract tests for durable incremental library indexing."""

import time
from types import SimpleNamespace

from tests.conftest import csrf_headers


def _wait_for_job(client, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/status/{job_id}")
        payload = response.get_json()
        if payload and payload.get("status") in {
            "complete",
            "error",
            "cancelled",
            "interrupted",
        }:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {payload}")


def _patch_index_dependencies(monkeypatch):
    import opencut.core.captions as captions
    import opencut.helpers as helpers

    monkeypatch.setattr(captions, "check_whisper_available", lambda: (True, "test"))
    monkeypatch.setattr(
        captions,
        "transcribe",
        lambda _path, config: SimpleNamespace(
            segments=[SimpleNamespace(text="hello searchable world")]
        ),
    )
    monkeypatch.setattr(helpers, "get_video_info", lambda _path: {"duration": 12.5})


def test_auto_index_persists_job_and_updates_search_index(
    client, csrf_token, monkeypatch, tmp_path
):
    import opencut.core.footage_index_db as footage_index_db

    db_path = str(tmp_path / "footage.db")
    footage_index_db.close_all_connections()
    monkeypatch.setattr(footage_index_db, "_DB_PATH", db_path)
    _patch_index_dependencies(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"test media")
    response = client.post(
        "/search/auto-index",
        json={"files": [{"path": str(media), "duration": 12.5}]},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]
    job = _wait_for_job(client, job_id)

    assert job["status"] == "complete"
    assert job["result"]["indexed"] == 1
    assert job["result"]["queued"] == 1
    assert footage_index_db.search("searchable world")[0]["file_path"] == str(media)


def test_auto_index_reports_up_to_date_noop_in_job_result(
    client, csrf_token, monkeypatch, tmp_path
):
    import opencut.core.footage_index_db as footage_index_db

    db_path = str(tmp_path / "footage.db")
    footage_index_db.close_all_connections()
    monkeypatch.setattr(footage_index_db, "_DB_PATH", db_path)
    _patch_index_dependencies(monkeypatch)

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"test media")
    footage_index_db.index_file(str(media), "already indexed")

    response = client.post(
        "/search/auto-index",
        json={"files": [{"path": str(media)}]},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    job = _wait_for_job(client, response.get_json()["job_id"])
    assert job["status"] == "complete"
    assert job["result"]["message"] == "All files are up to date"
    assert job["result"]["queued"] == 0
    assert job["result"]["skipped"] == 1


def _capture_semantic_search(monkeypatch):
    import opencut.core.semantic_video_search as svs

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return {"query": kwargs.get("query", ""), "results": []}

    monkeypatch.setattr(svs, "semantic_search", fake_search)
    return captured


def test_visual_search_route_passes_project_dir_for_the_sidecar(
    client, csrf_token, monkeypatch, tmp_path
):
    import os

    captured = _capture_semantic_search(monkeypatch)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"test media")
    project = tmp_path / "project"
    project.mkdir()

    response = client.post(
        "/search/semantic",
        json={"clip_paths": [str(media)], "query": "a dog", "project_dir": str(project)},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    job = _wait_for_job(client, response.get_json()["job_id"])
    assert job["status"] == "complete", job
    assert os.path.normcase(captured["project_dir"]) == os.path.normcase(
        os.path.realpath(str(project))
    )


def test_visual_search_route_without_project_dir_skips_the_sidecar(
    client, csrf_token, monkeypatch, tmp_path
):
    captured = _capture_semantic_search(monkeypatch)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"test media")

    response = client.post(
        "/search/semantic",
        json={"clip_paths": [str(media)], "query": "a dog"},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    job = _wait_for_job(client, response.get_json()["job_id"])
    assert job["status"] == "complete", job
    assert captured["project_dir"] is None
