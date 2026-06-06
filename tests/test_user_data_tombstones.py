"""Regression tests for user-data tombstone snapshots."""

from __future__ import annotations


def _headers(token: str) -> dict[str, str]:
    return {"X-OpenCut-Token": token}


def _isolate_user_data(monkeypatch, tmp_path):
    import opencut.user_data as user_data

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    with user_data._file_locks_guard:
        user_data._file_locks.clear()
    return user_data


def test_preset_delete_creates_restorable_tombstone(client, csrf_token, monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)
    user_data.save_presets({"Clean": {"settings": {"gain": 2}, "saved": 1}})

    rejected = client.post(
        "/presets/delete",
        json={"name": "Clean"},
        headers=_headers(csrf_token),
    )
    assert rejected.status_code == 409
    assert "Clean" in user_data.load_presets()

    preview = client.post(
        "/presets/delete",
        json={"name": "Clean", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()
    assert preview.status_code == 200
    assert preview_data["confirm_token"]

    response = client.post(
        "/presets/delete",
        json={"name": "Clean", "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    data = response.get_json()
    tombstone_id = data["tombstone"]["id"]
    assert response.status_code == 200
    assert "Clean" not in user_data.load_presets()

    listed = client.get("/settings/tombstones?kind=preset").get_json()
    assert listed["total"] == 1
    assert listed["tombstones"][0]["id"] == tombstone_id

    restored = client.post(
        "/settings/tombstones/restore",
        json={"id": tombstone_id},
        headers=_headers(csrf_token),
    )

    assert restored.status_code == 200
    assert user_data.load_presets()["Clean"]["settings"]["gain"] == 2
    assert "restored_at_iso" in restored.get_json()["restored"]


def test_favorites_replace_restores_previous_list(client, csrf_token, monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)
    user_data.save_favorites(["/silence", "/captions"])

    response = client.post(
        "/favorites/save",
        json={"favorites": ["/video/scenes"]},
        headers=_headers(csrf_token),
    )

    tombstone_id = response.get_json()["tombstone"]["id"]
    assert user_data.load_favorites() == ["/video/scenes"]

    restored = client.post(
        "/settings/tombstones/restore",
        json={"id": tombstone_id},
        headers=_headers(csrf_token),
    )

    assert restored.status_code == 200
    assert user_data.load_favorites() == ["/silence", "/captions"]


def test_workflow_delete_creates_restorable_tombstone(client, csrf_token, monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)
    workflow = {"name": "Rough Cut", "steps": [{"endpoint": "/silence", "label": "Silence"}]}
    user_data.save_workflows([workflow])

    rejected = client.delete(
        "/workflow/delete",
        json={"name": "Rough Cut"},
        headers=_headers(csrf_token),
    )
    assert rejected.status_code == 409
    assert user_data.load_workflows()[0]["name"] == "Rough Cut"

    preview = client.delete(
        "/workflow/delete",
        json={"name": "Rough Cut", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()
    assert preview.status_code == 200
    assert preview_data["destructive_plan"]["records"][0]["key"] == "Rough Cut"

    response = client.delete(
        "/workflow/delete",
        json={"name": "Rough Cut", "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    tombstone_id = response.get_json()["tombstone"]["id"]
    assert response.status_code == 200
    assert user_data.load_workflows() == []

    restored = client.post(
        "/settings/tombstones/restore",
        json={"id": tombstone_id},
        headers=_headers(csrf_token),
    )

    assert restored.status_code == 200
    assert user_data.load_workflows()[0]["name"] == "Rough Cut"


def test_settings_workflow_delete_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)
    workflow = {"name": "Template Cut", "steps": [{"endpoint": "/silence"}]}
    user_data.save_workflows([workflow])

    preview = client.post(
        "/workflows/delete",
        json={"name": "Template Cut", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["destructive_plan"]["records"][0]["path"].endswith("workflows.json")

    missing_token = client.post(
        "/workflows/delete",
        json={"name": "Template Cut"},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert user_data.load_workflows()[0]["name"] == "Template Cut"

    confirmed = client.post(
        "/workflows/delete",
        json={"name": "Template Cut", "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["deleted"] == "Template Cut"
    assert user_data.load_workflows() == []


def test_assistant_dismiss_clear_creates_restorable_tombstone(client, csrf_token, monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)
    user_data.save_assistant_dismissed("sequence-a", ["silence-dead-air", "generate-chapters"])

    response = client.post(
        "/assistant/dismiss-clear",
        json={"sequence_key": "sequence-a"},
        headers=_headers(csrf_token),
    )

    tombstone_id = response.get_json()["tombstone"]["id"]
    assert response.status_code == 200
    assert user_data.load_assistant_dismissed("sequence-a") == []

    restored = client.post(
        "/settings/tombstones/restore",
        json={"id": tombstone_id},
        headers=_headers(csrf_token),
    )

    assert restored.status_code == 200
    assert user_data.load_assistant_dismissed("sequence-a") == [
        "silence-dead-air",
        "generate-chapters",
    ]


def test_tombstone_store_is_capped(monkeypatch, tmp_path):
    user_data = _isolate_user_data(monkeypatch, tmp_path)

    for index in range(105):
        user_data.create_user_tombstone(
            "preset",
            f"preset-{index}",
            {"settings": {"index": index}},
            source_file="user_presets.json",
        )

    entries = user_data.list_user_tombstones()
    assert len(entries) == user_data.USER_TOMBSTONE_MAX_COUNT
    assert entries[0]["key"] == "preset-5"
    assert entries[-1]["key"] == "preset-104"
