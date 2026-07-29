"""Server-owned fixed targets for ``POST /system/open-path``.

The CEP panel used to hold ``--enable-nodejs`` purely so it could build a log
path and spawn an OS process itself. These targets move that ownership to the
server: the panel sends an opaque name and never learns a filesystem path.
"""

from __future__ import annotations

import pytest

from opencut.routes.system import _OPEN_PATH_FIXED_TARGETS, resolve_fixed_open_target


@pytest.fixture()
def opened(monkeypatch, tmp_path):
    """Redirect the target roots at tmp_path and capture launches."""
    from opencut import user_data
    from opencut.routes import system as system_module
    from opencut.routes import system_workspace_routes as workspace

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(system_module, "_user_data_dir", lambda: str(tmp_path))
    monkeypatch.setitem(
        _OPEN_PATH_FIXED_TARGETS, "log_dir", lambda: str(tmp_path)
    )

    launches: list[list[str]] = []

    class _FakePopen:
        def __init__(self, args, **_kwargs):
            launches.append([str(a) for a in args])

    monkeypatch.setattr(workspace._sp, "Popen", _FakePopen)
    monkeypatch.setattr(workspace.os, "startfile", lambda p: launches.append(["startfile", p]), raising=False)
    return tmp_path, launches


def _post(client, csrf_token, body):
    return client.post(
        "/system/open-path",
        json=body,
        headers={"X-OpenCut-Token": csrf_token},
    )


def test_every_declared_target_resolves_to_an_absolute_path():
    for name in _OPEN_PATH_FIXED_TARGETS:
        resolved = resolve_fixed_open_target(name)
        assert resolved
        assert resolved == resolve_fixed_open_target(name.upper())


def test_unknown_target_raises_rather_than_falling_back():
    with pytest.raises(KeyError):
        resolve_fixed_open_target("../../etc/passwd")


def test_server_log_target_launches_without_a_client_supplied_path(client, csrf_token, opened):
    tmp_path, launches = opened
    (tmp_path / "server.log").write_text("hello", encoding="utf-8")

    resp = _post(client, csrf_token, {"target": "server_log", "mode": "open"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["path"].endswith("server.log")
    assert launches, "expected the server to launch the OS handler"


def test_log_dir_target_opens_the_folder(client, csrf_token, opened):
    tmp_path, launches = opened

    resp = _post(client, csrf_token, {"target": "log_dir", "mode": "reveal"})

    assert resp.status_code == 200
    assert resp.get_json()["path"] == str(tmp_path)
    assert launches


def test_missing_diagnostic_file_is_reported_not_launched(client, csrf_token, opened):
    _tmp_path, launches = opened

    resp = _post(client, csrf_token, {"target": "crash_log", "mode": "open"})

    assert resp.status_code == 404
    assert launches == []


def test_unknown_target_is_rejected(client, csrf_token, opened):
    _tmp_path, launches = opened

    resp = _post(client, csrf_token, {"target": "etc_passwd", "mode": "open"})

    assert resp.status_code == 400
    assert "Allowed targets" in resp.get_json()["error"]
    assert launches == []


def test_target_and_path_are_mutually_exclusive(client, csrf_token, opened):
    _tmp_path, launches = opened

    resp = _post(
        client,
        csrf_token,
        {"target": "server_log", "path": "C:\\Windows\\System32\\cmd.exe", "mode": "open"},
    )

    assert resp.status_code == 400
    assert launches == []


def test_caller_supplied_path_still_requires_an_allowlisted_extension(client, csrf_token, tmp_path):
    executable = tmp_path / "payload.exe"
    executable.write_text("", encoding="utf-8")

    resp = _post(client, csrf_token, {"path": str(executable), "mode": "open"})

    assert resp.status_code == 403
