"""Regression tests for model/cache clear preview plans."""

from __future__ import annotations

import os

# Each one moves a cache these routes read.
_CACHE_ENV = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME")


def _headers(token: str) -> dict[str, str]:
    return {"X-OpenCut-Token": token}


def _isolate_whisper_caches(monkeypatch, tmp_path):
    import opencut.routes.system_whisper_routes as system_routes

    for name in _CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    local_appdata = tmp_path / "localappdata"
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda value: str(home) if value == "~" else value,
    )
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    hf_whisper = home / ".cache" / "huggingface" / "hub" / "models--openai--whisper-base"
    hf_other = home / ".cache" / "huggingface" / "hub" / "models--not-speech"
    whisper = home / ".cache" / "whisper"
    opencut_models = local_appdata / "OpenCut" / "models"
    for directory in (hf_whisper, hf_other, whisper, opencut_models):
        directory.mkdir(parents=True)
    (hf_whisper / "model.bin").write_bytes(b"whisper")
    (hf_other / "model.bin").write_bytes(b"other")
    (whisper / "base.pt").write_bytes(b"cache")
    (opencut_models / "base.bin").write_bytes(b"local")
    return system_routes, {
        "hf_whisper": hf_whisper,
        "hf_other": hf_other,
        "whisper": whisper,
        "opencut_models": opencut_models,
    }


def test_whisper_clear_cache_dry_run_returns_plan_without_deleting(client, csrf_token, monkeypatch, tmp_path):
    _system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)

    response = client.post(
        "/whisper/clear-cache",
        json={"dry_run": True},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 200
    data = response.get_json()
    planned_paths = {entry["path"] for entry in data["plan"]}
    assert data["dry_run"] is True
    assert data["cleared"] == []
    assert str(paths["hf_whisper"]) in planned_paths
    assert str(paths["whisper"]) in planned_paths
    assert str(paths["opencut_models"]) in planned_paths
    assert str(paths["hf_other"]) not in planned_paths
    assert paths["hf_whisper"].exists()
    assert paths["whisper"].exists()
    assert paths["opencut_models"].exists()


def test_whisper_clear_cache_plans_the_hub_cache_hf_hub_cache_names(client, csrf_token, monkeypatch, tmp_path):
    """It always planned ~/.cache/huggingface/hub, so with HF_HUB_CACHE set the
    button left the model faster-whisper loads and offered someone else's."""
    _system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)
    hub = tmp_path / "custom-hub"
    loaded = hub / "models--Systran--faster-whisper-base"
    other = hub / "models--bert--base"
    for directory in (loaded, other):
        directory.mkdir(parents=True)
        (directory / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    response = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))

    planned_paths = {entry["path"] for entry in response.get_json()["plan"]}
    assert str(loaded) in planned_paths
    assert str(other) not in planned_paths
    assert str(paths["hf_whisper"]) not in planned_paths


def test_whisper_clear_cache_skips_opencut_models_without_localappdata(client, csrf_token, monkeypatch, tmp_path):
    """Off Windows LOCALAPPDATA is unset, and the joined path was relative to
    whatever folder the server started in."""
    _isolate_whisper_caches(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    cwd = tmp_path / "cwd"
    (cwd / "OpenCut" / "models").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    response = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))

    plan = response.get_json()["plan"]
    assert [entry["path"] for entry in plan if entry["category"] == "opencut-models"] == []
    assert (cwd / "OpenCut" / "models").exists()


def test_models_list_and_delete_follow_hf_hub_cache_and_torch_home(client, csrf_token, monkeypatch, tmp_path):
    """/models/list read ~/.cache/huggingface/hub and $TORCH_HOME/checkpoints,
    and /models/delete refused anything outside ~/.cache, so a moved cache
    listed nothing and its models couldn't be deleted."""
    import opencut.routes.system_model_routes as model_routes

    _isolate_whisper_caches(monkeypatch, tmp_path)
    hub = tmp_path / "hub"
    torch_home = tmp_path / "torch-home"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    repo = hub / "models--org--speech"
    repo.mkdir(parents=True)
    (repo / "model.bin").write_bytes(b"model")
    checkpoint = torch_home / "hub" / "checkpoints" / "depth.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(model_routes._models_cache, "data", None)

    listed = client.get("/models/list", headers=_headers(csrf_token)).get_json()["models"]

    listed_paths = {model["path"] for model in listed}
    assert str(repo) in listed_paths
    assert str(checkpoint) in listed_paths
    for target in (repo, checkpoint):
        response = client.post(
            "/models/delete",
            json={"path": str(target), "dry_run": True},
            headers=_headers(csrf_token),
        )
        assert response.status_code == 200, response.get_json()
        assert target.exists()


def test_whisper_clear_cache_reports_per_path_delete_errors(client, csrf_token, monkeypatch, tmp_path):
    system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)
    preview = client.post(
        "/whisper/clear-cache",
        json={"dry_run": True},
        headers=_headers(csrf_token),
    )

    def deny_delete(path):
        return False, f"{path}: denied"

    monkeypatch.setattr(system_routes, "_delete_cache_target", deny_delete)
    response = client.post(
        "/whisper/clear-cache",
        json={"confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    data = response.get_json()
    assert response.status_code == 200
    assert data["success"] is False
    assert data["cleared"] == []
    assert len(data["errors"]) == len(data["plan"])
    assert paths["hf_whisper"].exists()


def test_whisper_clear_cache_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    _system_routes, _paths = _isolate_whisper_caches(monkeypatch, tmp_path)

    response = client.post(
        "/whisper/clear-cache",
        json={},
        headers=_headers(csrf_token),
    )

    data = response.get_json()
    assert response.status_code == 409
    assert data["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert data["plan"]["confirm_token"]


def test_models_delete_dry_run_preserves_target(client, csrf_token, monkeypatch, tmp_path):
    import opencut.routes.system_model_routes as system_routes

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "model.bin"
    model_file.write_bytes(b"model")
    monkeypatch.setattr(system_routes, "WHISPER_MODELS_DIR", str(model_dir))

    response = client.post(
        "/models/delete",
        json={"path": str(model_file), "dry_run": True},
        headers=_headers(csrf_token),
    )

    data = response.get_json()
    assert response.status_code == 200
    assert data["success"] is True
    assert data["dry_run"] is True
    assert data["deleted"] == []
    assert data["plan"][0]["path"] == str(model_file)
    assert data["plan"][0]["bytes"] == len(b"model")
    assert model_file.exists()


def test_models_delete_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    import opencut.routes.system_model_routes as system_routes

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "model.bin"
    model_file.write_bytes(b"model")
    monkeypatch.setattr(system_routes, "WHISPER_MODELS_DIR", str(model_dir))

    response = client.post(
        "/models/delete",
        json={"path": str(model_file)},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 409
    assert response.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"


def test_models_delete_accepts_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    import opencut.routes.system_model_routes as system_routes

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "model.bin"
    model_file.write_bytes(b"model")
    monkeypatch.setattr(system_routes, "WHISPER_MODELS_DIR", str(model_dir))
    preview = client.post(
        "/models/delete",
        json={"path": str(model_file), "dry_run": True},
        headers=_headers(csrf_token),
    )

    response = client.post(
        "/models/delete",
        json={"path": str(model_file), "confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert not model_file.exists()
