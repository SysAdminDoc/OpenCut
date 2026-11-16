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
import sysconfig
from concurrent.futures import Future

import pytest

from opencut import security
from tests.conftest import csrf_headers

TAG = f"{sys.version_info[0]}.{sys.version_info[1]}"
OTHER_TAG = f"{sys.version_info[0]}.{sys.version_info[1] - 1}"
MATCHING = r"C:\Python\match\python.exe"
MISMATCHED = rf"C:\Python{OTHER_TAG.replace('.', '')}\python.exe"
LAUNCHER = r"C:\Windows\py.exe"
ABI = security.runtime_abi()


def _machine(monkeypatch, *, launcher_version=None, path_python_version=OTHER_TAG, path_abi=None):
    """PATH holds ``python`` at ``path_python_version`` and maybe a ``py`` launcher."""
    found = {"python": MISMATCHED if path_python_version == OTHER_TAG else MATCHING}
    if launcher_version:
        found["py"] = LAUNCHER
    monkeypatch.setattr(security.shutil, "which", lambda name: found.get(name))

    def fake_run(cmd, **_kwargs):
        if cmd[0] == LAUNCHER:
            if cmd[1] != f"-{TAG}" or launcher_version != TAG:
                return subprocess.CompletedProcess(cmd, 103, "", "No suitable Python runtime found")
            return subprocess.CompletedProcess(cmd, 0, f"{MATCHING}\n{TAG}\n{ABI}\n", "")
        executable = found["python"]
        abi = path_abi or ABI
        return subprocess.CompletedProcess(
            cmd, 0, f"{executable}\n{path_python_version}\n{abi}\n", ""
        )

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


def test_a_same_version_python_with_another_abi_is_skipped(monkeypatch):
    """A 32-bit Python of the same minor picks win32 wheels the server can't load."""
    other_bits = ABI.replace("64bit", "32bit") if "64bit" in ABI else ABI.replace("32bit", "64bit")
    _machine(monkeypatch, path_python_version=TAG, path_abi=other_bits)
    assert security._find_system_python() is None


def test_the_probe_prints_this_interpreters_abi():
    """The real probe, so its ABI string and runtime_abi() can't drift apart."""
    probed = security._probe_interpreter([sys.executable])
    assert probed is not None
    executable, version, abi = probed
    assert version == sys.version_info[:2]
    assert abi == ABI
    assert os.path.isfile(executable)


def test_an_arm64_python_does_not_pass_for_the_x64_one(monkeypatch, tmp_path):
    """Windows on ARM: an ARM64 3.13 is 64-bit too, but pip picks win_arm64 wheels
    for it and mediapipe ships none. The real probe has to tell them apart."""
    other = "win-arm64" if sysconfig.get_platform() != "win-arm64" else "win-amd64"
    shim = tmp_path / "reports_another_platform.py"
    shim.write_text(
        f"import sys, sysconfig\nsysconfig.get_platform = lambda: {other!r}\nexec(sys.argv[2])\n",
        encoding="utf-8",
    )
    probed = security._probe_interpreter([sys.executable, str(shim)])
    assert probed is not None and probed[1] == sys.version_info[:2]
    assert probed[2] != ABI

    found = {"py": LAUNCHER, "python": MATCHING}
    monkeypatch.setattr(security.shutil, "which", lambda name: found.get(name))
    monkeypatch.setattr(
        security,
        "_probe_interpreter",
        lambda cmd: probed if cmd[0] == LAUNCHER else (MATCHING, sys.version_info[:2], ABI),
    )
    assert security._find_system_python() == MATCHING


@pytest.mark.parametrize("io_encoding", [None, "utf-8"])
def test_a_python_under_a_non_ascii_user_folder_comes_back_as_a_real_path(
    monkeypatch, tmp_path, io_encoding,
):
    """The installer runs the server with PYTHONIOENCODING=utf-8, which every probed
    Python inherits. Read as the code page, an accented folder name came back garbled."""
    folder = tmp_path / "Jos\u00e9" / "Python313"
    folder.mkdir(parents=True)
    claimed = str(folder / "python.exe")
    shim = tmp_path / "claims_a_non_ascii_executable.py"
    shim.write_text(
        f"import sys\nsys.executable = {claimed!r}\nexec(sys.argv[2])\n", encoding="utf-8"
    )
    if io_encoding:
        monkeypatch.setenv("PYTHONIOENCODING", io_encoding)
    else:
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)

    probed = security._probe_interpreter([sys.executable, str(shim)])

    assert probed is not None
    assert probed[0] == claimed
    assert probed[2] == ABI


@pytest.mark.parametrize(
    "platform, name",
    [("win-amd64", "64-bit (x64)"), ("win-arm64", "ARM64"), ("win32", "32-bit")],
)
def test_the_missing_python_message_names_the_python_org_installer(monkeypatch, platform, name):
    monkeypatch.setattr(security.sysconfig, "get_platform", lambda: platform)
    assert f"needs {name} Python {TAG}. Install it from python.org" in security.frozen_install_requirement()


def test_the_packages_folder_follows_the_data_folder(monkeypatch, tmp_path):
    from opencut import helpers

    monkeypatch.setattr(helpers, "OPENCUT_DIR", str(tmp_path))
    assert security.user_packages_dir() == str(tmp_path / "packages")


def test_source_installs_use_this_interpreter(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert security.install_python() == sys.executable


@pytest.fixture
def frozen(monkeypatch):
    """A packaged build; conftest already points ``~`` at an isolated home."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return os.path.join(os.path.expanduser("~"), ".opencut", "packages")


def _staged_under_the_data_folder(cmd, packages):
    """pip fills a staging folder next to the packages folder, never the folder itself."""
    staging = cmd[cmd.index("--target") + 1]
    assert os.path.dirname(staging) == os.path.dirname(packages)
    assert os.path.basename(staging).startswith(".") and "-staging-" in staging
    return staging


def test_safe_pip_install_stages_with_the_matching_python_then_imports_it(monkeypatch, frozen):
    from opencut.core import package_store

    commands = []

    def fake_run(cmd, timeout, on_start=None):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)
    monkeypatch.setattr(package_store, "_run", fake_run)

    security.safe_pip_install("rembg")

    installs = [cmd for cmd in commands if "pip" in cmd]
    assert len(installs) == 1
    command = installs[0]
    assert command[0] == MATCHING
    staging = _staged_under_the_data_folder(command, frozen)
    assert "--upgrade" not in command and "--user" not in command
    checks = [cmd for cmd in commands if "--import-check" in cmd]
    assert checks == [[sys.executable, "--import-check", "rembg", "--import-path", staging]]


def test_safe_pip_install_names_the_python_it_needs(monkeypatch, frozen):
    monkeypatch.setattr(security, "_find_system_python", lambda: None)
    with pytest.raises(RuntimeError, match=rf"needs .+ Python {TAG}\. Install it from python\.org"):
        security.safe_pip_install("rembg")


class _FakeProc:
    def __init__(self, cmd):
        self.args = cmd
        self.stdout = iter(["Successfully installed thing-1.0\n"])
        self.returncode = 0
        self.pid = 4242

    def communicate(self, timeout=None):
        return "Successfully installed thing-1.0\n", ""

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


def _pip_commands(calls):
    return [cmd for cmd in _every_command(calls) if "pip" in cmd]


def test_whisper_install_in_a_packaged_build_reaches_the_server(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    job = _job(client, csrf_token, "/install-whisper", {"backend": "openai-whisper"})

    assert job["status"] == "complete", job.get("error")
    assert job["result"]["installed"] is True
    installs = [cmd for cmd in recorded["popen"] if "pip" in cmd and "install" in cmd]
    assert len(installs) == 1, recorded
    assert installs[0][0] == MATCHING
    staging = _staged_under_the_data_folder(installs[0], frozen)
    assert "--upgrade" not in installs[0] and "--user" not in installs[0]
    # The server exe itself imports it; ``OpenCut-Server.exe -m pip`` would start a second server.
    checks = [cmd for cmd in recorded["popen"] if "--import-check" in cmd]
    assert checks == [[sys.executable, "--import-check", "whisper", "--import-path", staging]]
    assert all(cmd[0] != sys.executable for cmd in _pip_commands(recorded))


def test_whisper_install_without_a_matching_python_says_which_one(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: None)

    job = _job(client, csrf_token, "/install-whisper", {"backend": "openai-whisper"})

    assert job["status"] == "error"
    assert f" Python {TAG}. Install it" in job["error"]
    assert _pip_commands(recorded) == []


@pytest.mark.parametrize("packaged", [True, False], ids=["packaged", "source"])
def test_the_upgrade_pip_strategy_runs_a_command_pip_accepts(
    monkeypatch, request, inline_jobs, recorded, client, csrf_token, packaged,
):
    if packaged:
        request.getfixturevalue("frozen")
        monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    def tokenizers_strategies_fail(cmd, *args, **kwargs):
        recorded["popen"].append(cmd)
        proc = _FakeProc(cmd)
        if "tokenizers" in cmd or any(arg.startswith("tokenizers>=") for arg in cmd):
            proc.returncode = 1
        return proc

    monkeypatch.setattr(subprocess, "Popen", tokenizers_strategies_fail)

    job = _job(client, csrf_token, "/install-whisper", {"backend": "faster-whisper"})

    assert job["status"] == "complete", job.get("error")
    pip_installs = [cmd for cmd in _pip_commands(recorded) if "install" in cmd]
    # "pip setuptools wheel" as one argument is an invalid requirement to pip.
    assert [arg for cmd in pip_installs for arg in cmd if " " in arg] == []
    upgrades = [cmd for cmd in pip_installs if "setuptools" in cmd]
    if packaged:
        # The borrowed Python's own pip stays as it is; openai-whisper is the fallback.
        assert upgrades == []
        assert job["result"]["backend"] == "openai-whisper"
    else:
        assert upgrades and all({"pip", "wheel", "--upgrade"} <= set(cmd) for cmd in upgrades)
        assert all("--target" not in cmd for cmd in upgrades)
        assert job["result"]["backend"] == "faster-whisper"


def test_whisper_reinstall_in_a_packaged_build_leaves_the_users_python_alone(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token,
):
    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    job = _job(client, csrf_token, "/whisper/reinstall", {"backend": "openai-whisper"})

    assert job["status"] == "complete", job.get("error")
    commands = _every_command(recorded)
    assert not [cmd for cmd in commands if "uninstall" in cmd]
    installs = [cmd for cmd in recorded["popen"] if "pip" in cmd and "install" in cmd]
    assert len(installs) == 1
    staging = _staged_under_the_data_folder(installs[0], frozen)
    assert "--no-cache-dir" in installs[0]
    checks = [cmd for cmd in recorded["popen"] if "--import-check" in cmd]
    assert checks == [[sys.executable, "--import-check", "whisper", "--import-path", staging]]


def test_whisper_reinstall_clears_whisper_from_the_hub_cache_the_loaders_use(
    monkeypatch, tmp_path, inline_jobs, recorded, client, csrf_token,
):
    """It cleared ~/.cache/huggingface/hub whatever HF_HUB_CACHE said, so the
    corrupt model a reinstall is for survived it, and a whisper repo another
    tool kept in the default cache was deleted."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    # The old code read the default cache through expanduser, so the decoy
    # has to sit where expanduser points.
    monkeypatch.setattr(os.path, "expanduser", lambda value: str(home) if value == "~" else value)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    corrupt = tmp_path / "hub" / "models--Systran--faster-whisper-base"
    unrelated = tmp_path / "hub" / "models--bert--base"
    elsewhere = home / ".cache" / "huggingface" / "hub" / "models--openai--whisper-tiny"
    openai_cache = tmp_path / "xdg" / "whisper"
    local_models = tmp_path / "localappdata" / "OpenCut" / "models"
    for directory in (corrupt, unrelated, elsewhere, openai_cache, local_models):
        directory.mkdir(parents=True)
        (directory / "model.bin").write_bytes(b"x")

    job = _job(client, csrf_token, "/whisper/reinstall", {"backend": "faster-whisper"})

    assert job["status"] == "complete", job.get("error")
    assert not corrupt.exists()
    assert not openai_cache.exists()
    assert not local_models.exists()
    assert (unrelated / "model.bin").exists()
    assert (elsewhere / "model.bin").exists()


@pytest.mark.parametrize("route", ["/install-whisper", "/whisper/reinstall"])
def test_whisperx_gets_the_dependency_policy_reason_instead_of_a_pip_run(
    monkeypatch, frozen, inline_jobs, recorded, client, csrf_token, route,
):
    """WhisperX pins a Torch stack OpenCut can't run; the old route installed it anyway."""
    from opencut.dependency_support import UNSUPPORTED_DEPENDENCIES

    monkeypatch.setattr(security, "_find_system_python", lambda: MATCHING)

    job = _job(client, csrf_token, route, {"backend": "whisperx"})

    assert job["status"] == "error"
    assert UNSUPPORTED_DEPENDENCIES["whisperx"][:40] in job["error"]
    # Nothing runs first either: the reinstall used to clear caches before failing.
    assert _pip_commands(recorded) == []
