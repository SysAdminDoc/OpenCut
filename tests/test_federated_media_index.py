"""Coverage for the offline configured-root media federation contract."""

import json
import os
import sqlite3
import time

import pytest
from click.testing import CliRunner

from opencut.core import federated_media_index as federation
from opencut.local_db_migrations import LocalDatabaseVersionError


@pytest.fixture
def isolated_federation(tmp_path, monkeypatch):
    db_path = tmp_path / "federated.db"
    monkeypatch.setattr(federation, "_DB_PATH", str(db_path))
    monkeypatch.setattr(federation, "_video_duration", lambda _path, _kind: 5.0)
    monkeypatch.setattr(federation, "_visual_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "opencut.core.footage_search.load_index",
        lambda: {},
    )
    monkeypatch.setattr(
        "opencut.core.footage_index_db.get_indexed_file",
        lambda _path: None,
    )
    federation.init_db()
    return tmp_path


def _write_media(root, name="clip.mp4", payload=b"frame-0001"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_schema_rejects_future_database(isolated_federation):
    with sqlite3.connect(federation._DB_PATH) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(LocalDatabaseVersionError):
        federation.init_db()


def test_root_registry_redacts_paths_and_reenables(isolated_federation):
    root = isolated_federation / "project"
    root.mkdir()
    record = federation.add_root(str(root), label="Demo")
    assert record["path"] == os.path.realpath(root)
    redacted = federation.list_roots()
    assert redacted[0]["label"] == "Demo"
    assert "path" not in redacted[0]

    federation.remove_root(record["root_id"])
    assert federation.list_roots() == []
    assert federation.list_roots(include_disabled=True)[0]["enabled"] is False
    reenabled = federation.add_root(str(root), label="Demo 2")
    assert reenabled["root_id"] == record["root_id"]
    assert federation.list_roots()[0]["label"] == "Demo 2"


def test_incremental_scan_limits_and_content_changes_mark_modalities_stale(
    isolated_federation, monkeypatch
):
    root = isolated_federation / "project"
    root.mkdir()
    clip = _write_media(root)
    root_record = federation.add_root(str(root))

    first = federation.scan_roots([root_record["root_id"]])
    assert first["complete"] is True
    assert first["roots"][0]["scanned"] == 1
    second = federation.scan_roots([root_record["root_id"]])
    assert second["roots"][0]["unchanged"] == 1

    _write_media(root, "second.mp4", b"frame-0002")
    limited = federation.scan_roots([root_record["root_id"]], max_files=1)
    assert limited["complete"] is False
    assert federation.status()["media_counts"]["active"] == 1

    # Seed one modality so the content-change transition is observable.
    with sqlite3.connect(federation._DB_PATH) as conn:
        media_id = conn.execute("SELECT id FROM media WHERE relative_path = 'clip.mp4'").fetchone()[0]
        conn.execute(
            """
            UPDATE modalities SET state = 'available', text = 'old transcript', updated_at = ?
            WHERE media_id = ? AND modality = 'text'
            """,
            (time.time(), media_id),
        )
    clip.write_bytes(b"frame-CHANGED")
    changed = federation.scan_roots([root_record["root_id"]], max_files=1)
    assert changed["roots"][0]["indexed"] == 1
    with sqlite3.connect(federation._DB_PATH) as conn:
        state = conn.execute(
            "SELECT state FROM modalities WHERE media_id = ? AND modality = 'text'",
            (media_id,),
        ).fetchone()[0]
    assert state == "stale"


def test_legacy_text_ocr_audio_import_has_clamped_timestamps(isolated_federation, monkeypatch):
    root = isolated_federation / "project"
    root.mkdir()
    clip = _write_media(root)
    mtime = clip.stat().st_mtime
    monkeypatch.setattr(
        "opencut.core.footage_search.load_index",
        lambda: {
            str(clip): {
                "mtime": mtime,
                "full_text": "speaker at podium",
                "segments": [
                    {"start": -4, "end": 99, "text": "speaker at podium"},
                ],
            }
        },
    )
    monkeypatch.setattr(
        "opencut.core.footage_index_db.get_indexed_file",
        lambda _path: {
            "file_mtime": mtime,
            "transcript": "speaker at podium",
            "ocr_text": "OpenCut title",
            "audio_tags": "speech applause",
        },
    )
    root_record = federation.add_root(str(root))
    federation.scan_roots([root_record["root_id"]])

    result = federation.search("podium", modalities=["text"])
    assert result["results"][0]["relative_path"] == "clip.mp4"
    assert result["results"][0]["timestamps"] == [
        {"start": 0.0, "end": 5.0, "text": "speaker at podium"}
    ]
    ocr = federation.search("title", modalities=["ocr"])
    assert ocr["results"][0]["matched_modalities"] == ["ocr"]
    audio = federation.search("applause", modalities=["audio"])
    assert audio["results"][0]["matched_modalities"] == ["audio"]
    assert "path" not in result["results"][0]


def test_move_delete_and_retention_prune(isolated_federation):
    root = isolated_federation / "project"
    root.mkdir()
    old = _write_media(root, "old.mp4", b"move-stable")
    root_record = federation.add_root(str(root), retention_days=1)
    federation.scan_roots([root_record["root_id"]])
    new = root / "nested" / "renamed.mp4"
    new.parent.mkdir()
    old.rename(new)
    moved = federation.scan_roots([root_record["root_id"]])
    assert moved["roots"][0]["relinked"] == 1
    assert federation.search("", modalities=["text"])["results"] == []

    new.unlink()
    federation.scan_roots([root_record["root_id"]])
    with sqlite3.connect(federation._DB_PATH) as conn:
        conn.execute(
            "UPDATE media SET missing_since = ? WHERE root_id = ?",
            (time.time() - 3 * 86400, root_record["root_id"]),
        )
    preview = federation.prune_missing(root_ids=[root_record["root_id"]], dry_run=True)
    assert preview["pruned"] == 1
    applied = federation.prune_missing(root_ids=[root_record["root_id"]])
    assert applied["pruned"] == 1
    assert federation.status()["media_counts"]["deleted"] == 1


def test_visual_sidecar_capability_and_schema_incompatibility(isolated_federation, monkeypatch):
    root = isolated_federation / "project"
    root.mkdir()
    _write_media(root)
    root_record = federation.add_root(str(root))
    monkeypatch.setattr(
        federation,
        "_visual_metadata",
        lambda *_args, **_kwargs: {
            "engine": "clip-vit-b32",
            "schema_version": 1,
            "timestamps": [{"start": 1.0, "end": 1.0}],
            "thumbnail_path": "",
            "capability": {
                "network_required": False,
                "sidecar_available": True,
                "timestamps_available": True,
                "thumbnail_available": False,
            },
        },
    )
    federation.scan_roots([root_record["root_id"]])
    visual = federation.search("", modalities=["visual"])
    assert visual["results"][0]["capabilities"]["visual"]["state"] == "available"
    assert visual["results"][0]["timestamps"] == [{"start": 1.0, "end": 1.0}]

    with sqlite3.connect(federation._DB_PATH) as conn:
        media_id = conn.execute("SELECT id FROM media").fetchone()[0]
        conn.execute(
            "UPDATE modalities SET engine = 'old-engine', schema_version = 99 WHERE media_id = ? AND modality = 'visual'",
            (media_id,),
        )
    monkeypatch.setattr(federation, "_visual_metadata", lambda *_args, **_kwargs: None)
    federation.scan_roots([root_record["root_id"]])
    with sqlite3.connect(federation._DB_PATH) as conn:
        state = conn.execute(
            "SELECT state FROM modalities WHERE media_id = ? AND modality = 'visual'",
            (media_id,),
        ).fetchone()[0]
    assert state == "schema_incompatible"


def test_search_include_paths_is_explicit(isolated_federation, monkeypatch):
    root = isolated_federation / "project"
    root.mkdir()
    clip = _write_media(root)
    root_record = federation.add_root(str(root))
    mtime = clip.stat().st_mtime
    # Use the normal importer path with a current legacy row.
    import opencut.core.footage_search as footage_search
    monkeypatch.setattr(footage_search, "load_index", lambda: {
            str(clip): {
                "mtime": mtime,
                "full_text": "private test",
                "segments": [{"start": 0, "end": 1, "text": "private test"}],
            }
        })
    federation.scan_roots([root_record["root_id"]])
    redacted = federation.search("private")
    exposed = federation.search("private", include_paths=True)
    assert "path" not in redacted["results"][0]
    assert exposed["results"][0]["path"] == os.path.realpath(clip)


def test_federated_routes_redact_roots_and_query(client, csrf_token, tmp_path, monkeypatch):
    root = tmp_path / "route-project"
    root.mkdir()
    clip = _write_media(root)
    monkeypatch.setattr(federation, "_DB_PATH", str(tmp_path / "route.db"))
    monkeypatch.setattr(federation, "_video_duration", lambda _path, _kind: 5.0)
    monkeypatch.setattr(federation, "_visual_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "opencut.core.footage_search.load_index",
        lambda: {
            str(clip): {
                "mtime": clip.stat().st_mtime,
                "full_text": "route marker",
                "segments": [{"start": 0, "end": 1, "text": "route marker"}],
            }
        },
    )
    monkeypatch.setattr("opencut.core.footage_index_db.get_indexed_file", lambda _path: None)

    headers = {"X-OpenCut-Token": csrf_token, "Content-Type": "application/json"}
    response = client.post(
        "/search/federated/roots",
        json={"path": str(root), "label": "Route root"},
        headers=headers,
    )
    assert response.status_code == 201
    assert "path" not in response.get_json()["root"]
    root_id = response.get_json()["root"]["root_id"]
    federation.scan_roots([root_id])
    response = client.post(
        "/search/federated/query",
        json={"query": "", "include_paths": False},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.get_json()["include_paths"] is False
    assert "path" not in response.get_json()["results"][0]


def test_federated_mcp_actions_are_allowlisted(monkeypatch):
    from opencut import mcp_server

    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True, "method": method, "path": path}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    assert mcp_server.handle_tool_call(
        "opencut_federated_search", {"action": "status", "root_ids": [2]}
    )["path"] == "/search/federated/status?root_id=2"
    assert mcp_server.handle_tool_call(
        "opencut_federated_search", {"action": "add_root", "root_path": "C:/media"}
    )["path"] == "/search/federated/roots"
    assert mcp_server.handle_tool_call(
        "opencut_federated_search", {"action": "remove_root", "root_id": 2}
    )["path"] == "/search/federated/roots/2"
    before = len(calls)
    invalid = mcp_server.handle_tool_call(
        "opencut_federated_search", {"action": "add_root", "root_path": "../secret"}
    )
    assert "Invalid root_path" in invalid["error"]
    assert len(calls) == before


def test_federated_cli_root_and_query_are_local_and_redacted(isolated_federation, monkeypatch):
    root = isolated_federation / "cli-project"
    root.mkdir()
    clip = _write_media(root)
    mtime = clip.stat().st_mtime
    monkeypatch.setattr(
        "opencut.core.footage_search.load_index",
        lambda: {
            str(clip): {
                "mtime": mtime,
                "full_text": "cli marker",
                "segments": [{"start": 0, "end": 1, "text": "cli marker"}],
            }
        },
    )
    from opencut import cli as cli_module

    runner = CliRunner()
    added = runner.invoke(cli_module.cli, ["search", "federated", "root-add", str(root), "--json"])
    assert added.exit_code == 0, added.output
    assert "\"path\"" not in added.output
    indexed = runner.invoke(cli_module.cli, ["search", "federated", "index", "--json"])
    assert indexed.exit_code == 0, indexed.output
    queried = runner.invoke(
        cli_module.cli,
        ["search", "federated", "query", "cli", "--json"],
    )
    assert queried.exit_code == 0, queried.output
    assert json.loads(queried.output)["count"] == 1
