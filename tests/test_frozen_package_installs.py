"""A packaged server installs add-on packages it can actually load.

The frozen build runs a bundled CPython and imports add-on packages only from
``~/.opencut/packages``. Issue #8's server died importing native modules built
for another Python, so whatever fills that directory has to match the bundled
minor version, and an install anywhere else is one the server never sees.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import Future

import pytest

from opencut import security
from tests.conftest import csrf_headers

TAG = f"{sys.version_info[0]}.{sys.version_info[1]}"
OTHER_TAG = f"{sys.version_info[0]}.{sys.version_info[1] - 1}"
MATCHING = r"C:\Python\match\python.exe"
MISMATCHED = rf"C:\Python{OTHER_TAG.replace('.', '')}\python.exe"
LAUNCHER = r"C:\Windows\py.exe"


def _machine(monkeypatch, *, launcher_version=None, path_python_version=OTHER_TAG):
    """PATH holds ``python`` at ``path_python_version`` and maybe a ``py`` launcher."""
    found = {"python": MISMATCHED if path_python_version == OTHER_TAG else MATCHING}
    if launcher_version:
        found["py"] = LAUNCHER
    monkeypatch.setattr(security.shutil, "which", lambda name: found.get(name))

    def fake_run(cmd, **_kwargs):
        if cmd[0] == LAUNCHER:
            if cmd[1] != f"-{TAG}" or launcher_version != TAG:
                return subprocess.CompletedProcess(cmd, 103, "", "No suitable Python runtime found")
            return subprocess.CompletedProcess(cmd, 0, f"{MATCHING}\n{TAG}\n", "")
        executable = found["python"]
        return subprocess.CompletedProcess(cmd, 0, f"{executable}\n{path_python_version}\n", "")

    monkeypatch.setattr(security.subprocess, "run", fake_run)


def test_a_different_python_first_on_path_is_skipped_for_the_matching_one(monkeypatch):
    _machine(monkeypatch, launcher_version=TAG, path_python_version=OTHER_TAG)
    assert security._find_system_python() == MATCHING


def test_no_matching_python_means_no_installer_rather_than_a_wrong_one(monkeypatch):
    _machine(monkeypatch, launcher_version=None, path_python_version=OTHER_TAG)
    assert security._find_system_python() is None


def test_a_matching_python_on_path_is_used_without_a_launcher(monkeypatch):
    _machine(monkeypatch, launcher_version=None, path_python_version=TAG)
    assert security._find_system_python() == MATCHING


def test_source_installs_keep_their_fallback_order(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    base = ["python", "-m", "pip", "install", "x"]
    assert security.pip_install_variants(base, "T") == [
        base,
        [*base, "--user"],
        [*base, "--target", "T"],
    ]
    assert security.install_python() == sys.executable


@pytest.fixture
def frozen(monkeypatch):
    """A packaged build; conftest already points ``~`` at an isolated home."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return os.path.join(os.path.expanduser("~"), ".opencut", "packages")


def test_safe_pip_install_targets_only_the_packages_dir_with_the_matching_python(monkeypatch, frozen):
    commands = []
    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)
    monkeypatch.setattr(
        security.subprocess,
        "run",
        lambda cmd, **_kwargs: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    monkeypatch.setattr(security, "_verify_package_importable", lambda *_args: True)

    security.safe_pip_install("rembg")

    assert len(commands) == 1
    command = commands[0]
    assert command[0] == MATCHING
    assert command[command.index("--target") + 1] == frozen
    assert "--upgrade" in command
    assert "--user" not in command


def test_safe_pip_install_names_the_python_it_needs(monkeypatch, frozen):
    monkeypatch.setattr(security, "_find_system_python", lambda: None)
    with pytest.raises(RuntimeError, match=rf"needs Python {TAG}"):
        security.safe_pip_install("rembg")


class _FakeProc:
    def __init__(self, cmd):
        self.args = cmd
        self.stdout = iter(["Successfully installed thing-1.0\n"])
        self.returncode = 0
        self.pid = 4242

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def inline_jobs(monkeypatch):
    class InlinePool:
        def submit(self, job_id, fn):
            future = Future()
            fn()
            future.set_result(None)
            return future

    monkeypatch.setattr("opencut.workers.get_pool", lambda: InlinePool())
    monkeypatch.setattr("opencut.jobs._persist_job", lambda *args, **kwargs: None)


@pytest.fixture
def recorded(monkeypatch, client):
    """Every process the route starts is faked, so the install job may run for real."""
    monkeypatch.setitem(client.application.config, "OPENCUT_RUN_INSTALLS_IN_TESTS", True)
    calls = {"popen": [], "run": []}

    def fake_popen(cmd, *args, **kwargs):
        calls["popen"].append(cmd)
        return _FakeProc(cmd)

    def fake_run(cmd, *args, **kwargs):
        calls["run"].append((cmd, kwargs.get("env")))
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _job(client, csrf_token, route, payload):
    from opencut import jobs
    from opencut.jobs import _get_job_copy

    response = client.post(route, json=payload, headers=csrf_headers(csrf_token))
    assert response.status_code == 200, response.get_json()
    job_id = response.get_json()["job_id"]
    job = _get_job_copy(job_id)
    assert job is not None, (job_id, list(jobs.jobs))
    return job


def _every_command(calls):
    return list(calls["popen"]) + [cmd for cmd, _env in calls["run"]]


def test_whisper_install_in_a_packaged_build_reaches_the_server(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    job = _job(client, csrf_token, "/install-whisper", {"backend": "openai-whisper"})

    assert job["status"] == "complete", job.get("error")
    assert job["result"]["installed"] is True
    installs = [cmd for cmd in recorded["popen"] if "install" in cmd]
    assert installs, recorded
    for cmd in installs:
        assert cmd[0] == MATCHING
        assert cmd[cmd.index("--target") + 1] == frozen
        assert "--upgrade" in cmd and "--user" not in cmd
    assert all(cmd[0] != sys.executable for cmd in _every_command(recorded))
    verify_envs = [env for cmd, env in recorded["run"] if "-c" in cmd]
    assert verify_envs and all(frozen in env["PYTHONPATH"] for env in verify_envs)


def test_whisper_install_without_a_matching_python_says_which_one(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: None)

    job = _job(client, csrf_token, "/install-whisper", {"backend": "openai-whisper"})

    assert job["status"] == "error"
    assert f"needs Python {TAG}" in job["error"]
    commands = _every_command(recorded)
    assert [cmd for cmd in commands if "pip" in cmd] == []
    # Never the server exe itself: ``OpenCut-Server.exe -m pip`` starts a second server.
    assert all(cmd[0] != sys.executable for cmd in commands)


def test_whisper_reinstall_in_a_packaged_build_leaves_the_users_python_alone(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    job = _job(client, csrf_token, "/whisper/reinstall", {"backend": "whisperx"})

    assert job["status"] == "complete", job.get("error")
    commands = _every_command(recorded)
    assert not [cmd for cmd in commands if "uninstall" in cmd]
    installs = [cmd for cmd in recorded["popen"] if "install" in cmd]
    assert len(installs) == 1
    assert installs[0][installs[0].index("--target") + 1] == frozen
    verify = [(cmd, env) for cmd, env in recorded["run"] if "-c" in cmd]
    assert verify and "import whisperx" in verify[-1][0][-1]
    assert frozen in verify[-1][1]["PYTHONPATH"]
