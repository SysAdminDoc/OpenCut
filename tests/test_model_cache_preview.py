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


def test_whisper_clear_cache_keeps_the_crisperwhisper_add_on(client, csrf_token, monkeypatch, tmp_path):
    """Any Hub folder with "whisper" in its name was planned, so clearing
    Whisper's cache also deleted CrisperWhisper, a separate add-on."""
    _system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)
    crisper = paths["hf_whisper"].parent / "models--nyrahealth--CrisperWhisper"
    crisper.mkdir()
    (crisper / "model.safetensors").write_bytes(b"crisper")
    preview = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))
    assert str(crisper) not in {entry["path"] for entry in preview.get_json()["plan"]}

    response = client.post(
        "/whisper/clear-cache",
        json={"confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert not paths["hf_whisper"].exists()
    assert crisper.exists()


def test_whisper_clear_cache_skips_relative_paths_from_an_empty_xdg_cache_home(
    client, csrf_token, monkeypatch, tmp_path,
):
    """XDG_CACHE_HOME="" made both caches cwd-relative ("whisper" and
    "huggingface/hub"), so the clear planned folders under the server's cwd."""
    _isolate_whisper_caches(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    cwd = tmp_path / "cwd"
    stray_whisper = cwd / "whisper"
    stray_hub_repo = cwd / "huggingface" / "hub" / "models--openai--whisper-base"
    for directory in (stray_whisper, stray_hub_repo):
        directory.mkdir(parents=True)
        (directory / "model.bin").write_bytes(b"x")
    monkeypatch.chdir(cwd)

    response = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))

    assert response.get_json()["plan"] == []


def test_whisper_clear_cache_plans_a_hub_cache_rooted_on_the_current_drive(
    client, csrf_token, monkeypatch, tmp_path,
):
    """HF_HUB_CACHE=/hf/hub has no drive, so 3.13's os.path.isabs calls it
    relative, but the Hub puts it on the current drive and loads from it."""
    _isolate_whisper_caches(monkeypatch, tmp_path)
    repo = tmp_path / "hub" / "models--Systran--faster-whisper-base"
    repo.mkdir(parents=True)
    (repo / "model.bin").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)  # same drive as tmp_path
    monkeypatch.setenv("HF_HUB_CACHE", os.path.splitdrive(str(tmp_path / "hub"))[1])

    response = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))

    planned = {os.path.normcase(os.path.abspath(entry["path"])) for entry in response.get_json()["plan"]}
    assert os.path.normcase(str(repo)) in planned


def test_models_list_skips_a_hub_cache_that_follows_the_cwd(client, csrf_token, monkeypatch, tmp_path):
    """With XDG_CACHE_HOME="" the list showed models under the server's cwd
    that /models/delete then refused."""
    import opencut.routes.system_model_routes as model_routes

    _isolate_whisper_caches(monkeypatch, tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    cwd = tmp_path / "cwd"
    (cwd / "huggingface" / "hub" / "models--org--speech").mkdir(parents=True)
    (cwd / "torch" / "hub" / "checkpoints").mkdir(parents=True)
    (cwd / "torch" / "hub" / "checkpoints" / "depth.pth").write_bytes(b"x")
    monkeypatch.chdir(cwd)
    monkeypatch.setitem(model_routes._models_cache, "data", None)

    listed = client.get("/models/list", headers=_headers(csrf_token)).get_json()["models"]

    assert [model["path"] for model in listed if model["source"] in ("huggingface", "torch")] == []


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


def test_models_delete_refuses_a_cache_folder_itself(client, csrf_token, monkeypatch, tmp_path):
    """A cache root counted as inside itself, so one call could delete the
    whole Hub cache, every torch checkpoint, or the Whisper models folder."""
    import opencut.routes.system_model_routes as model_routes

    _isolate_whisper_caches(monkeypatch, tmp_path)
    hub = tmp_path / "hub"
    torch_home = tmp_path / "torch-home"
    whisper_models = tmp_path / "whisper-models"
    home_hf = tmp_path / "home" / ".cache" / "huggingface"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setattr(model_routes, "WHISPER_MODELS_DIR", str(whisper_models))
    repo = hub / "models--org--speech"
    folders = [hub, torch_home / "hub", torch_home / "hub" / "checkpoints", whisper_models, home_hf, home_hf / "hub"]
    for folder in [*folders, repo]:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.bin").write_bytes(b"x")

    for folder in folders:
        response = client.post(
            "/models/delete",
            json={"path": str(folder), "dry_run": True},
            headers=_headers(csrf_token),
        )
        assert response.status_code == 403, (str(folder), response.get_json())
    response = client.post(
        "/models/delete",
        json={"path": str(repo), "dry_run": True},
        headers=_headers(csrf_token),
    )
    assert response.status_code == 200, response.get_json()


def test_models_delete_refuses_the_cwd_when_xdg_cache_home_is_empty(client, csrf_token, monkeypatch, tmp_path):
    """XDG_CACHE_HOME="" made the Hub and torch roots relative, which resolved
    against the server's cwd and let /models/delete reach folders there."""
    _isolate_whisper_caches(monkeypatch, tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    cwd = tmp_path / "cwd"
    stray = cwd / "huggingface" / "hub" / "models--org--speech"
    stray.mkdir(parents=True)
    (stray / "model.bin").write_bytes(b"x")
    monkeypatch.chdir(cwd)

    response = client.post(
        "/models/delete",
        json={"path": str(stray), "dry_run": True},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 403, response.get_json()
    assert stray.exists()


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


def test_whisper_clear_cache_deletes_past_a_locked_file(client, csrf_token, monkeypatch, tmp_path):
    """rmtree stopped at the first file it couldn't remove, so one model a job
    still had open left the rest of the folder behind."""
    _system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)
    snapshot = paths["hf_whisper"] / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    locked = snapshot / "a-locked.bin"
    locked.write_bytes(b"open")
    rest = [paths["hf_whisper"] / "b-config.json", snapshot / "c.bin"]
    for path in rest:
        path.write_bytes(b"x")
    real_unlink = os.unlink

    def unlink(target, *args, **kwargs):
        if os.path.basename(str(target)) == locked.name:
            raise PermissionError(13, "The process cannot access the file", str(target))
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)
    preview = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))

    response = client.post(
        "/whisper/clear-cache",
        json={"confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    data = response.get_json()
    assert data["success"] is False
    assert [path for path in rest if path.exists()] == []
    assert locked.exists()
    hub_errors = [error for error in data["errors"] if error.startswith(str(paths["hf_whisper"]))]
    assert len(hub_errors) == 1
    # The folders it leaves behind aren't failures of their own.
    assert "1 item(s)" in hub_errors[0] and locked.name in hub_errors[0], hub_errors[0]
    assert not paths["whisper"].exists()


def test_models_list_drops_what_a_partial_delete_removed(client, csrf_token, monkeypatch, tmp_path):
    """A delete that left one locked file answered 500 without resetting the
    list cache, so the model list showed the full old size for 5 minutes."""
    import opencut.routes.system_model_routes as model_routes

    _isolate_whisper_caches(monkeypatch, tmp_path)
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    repo = hub / "models--org--speech"
    repo.mkdir(parents=True)
    locked = repo / "a-locked.bin"
    locked.write_bytes(b"open")
    weights = repo / "b-weights.bin"
    weights.write_bytes(b"x" * (2 * 1024 * 1024))
    monkeypatch.setitem(model_routes._models_cache, "data", None)

    def listed_size():
        models = client.get("/models/list", headers=_headers(csrf_token)).get_json()["models"]
        return [model["size_mb"] for model in models if model["path"] == str(repo)]

    assert listed_size() == [2.0]
    real_unlink = os.unlink

    def unlink(target, *args, **kwargs):
        if os.path.basename(str(target)) == locked.name:
            raise PermissionError(13, "The process cannot access the file", str(target))
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)
    preview = client.post("/models/delete", json={"path": str(repo), "dry_run": True}, headers=_headers(csrf_token))
    response = client.post(
        "/models/delete",
        json={"path": str(repo), "confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert response.status_code == 500, response.get_json()
    assert locked.exists() and not weights.exists()
    assert listed_size() == [0.0]


def test_whisper_clear_cache_drops_cleared_models_from_the_list(client, csrf_token, monkeypatch, tmp_path):
    """Clear cache deletes Hub folders the model list shows, but never reset
    the list cache, so they stayed listed for 5 minutes."""
    import opencut.routes.system_model_routes as model_routes

    _system_routes, paths = _isolate_whisper_caches(monkeypatch, tmp_path)
    monkeypatch.setitem(model_routes._models_cache, "data", None)

    def listed_paths():
        models = client.get("/models/list", headers=_headers(csrf_token)).get_json()["models"]
        return {model["path"] for model in models}

    assert str(paths["hf_whisper"]) in listed_paths()
    preview = client.post("/whisper/clear-cache", json={"dry_run": True}, headers=_headers(csrf_token))
    response = client.post(
        "/whisper/clear-cache",
        json={"confirm_token": preview.get_json()["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert response.get_json()["success"] is True, response.get_json()
    after = listed_paths()
    assert str(paths["hf_whisper"]) not in after
    assert str(paths["hf_other"]) in after


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
