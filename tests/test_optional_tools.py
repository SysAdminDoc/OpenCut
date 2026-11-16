"""Optional tools land where the server loads them, and only once they work there.

Discussion #9: the WPF installer pip-installed auto-editor, edge-tts and
mediapipe into whatever system Python it found, which the packaged server
never imports from, and auto-editor's pip release is a frozen 29.x. Now
``OpenCut-Server.exe --install-tools`` does it: the pinned native auto-editor
into ``<data folder>/bin``, the pip tools through an ABI-matched Python into
a staging folder that the server itself has to import before it's merged.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from opencut import helpers, security
from opencut.core import optional_tools, package_store

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_CS = (
    REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Services" / "DependencyInstaller.cs"
)
CRASH_EXIT = 0xC0000005  # what a native module built for another ABI does to the server


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(helpers, "OPENCUT_DIR", str(data))
    return data


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------
def test_pip_tool_bounds_are_the_ones_pyproject_declares():
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]
    declared = {req for group in extras.values() for req in group}
    for requirement, _module in optional_tools.PIP_TOOLS.values():
        assert requirement in declared, requirement


def test_the_installer_offers_exactly_the_tools_the_server_installs():
    source = INSTALLER_CS.read_text(encoding="utf-8")
    match = re.search(r"SupportedTools\s*=\s*\[([^\]]*)\]", source)
    assert match, "DependencyInstaller.SupportedTools not found"
    assert tuple(re.findall(r'"([^"]+)"', match.group(1))) == optional_tools.TOOL_NAMES


def test_the_installer_no_longer_runs_pip_itself():
    source = INSTALLER_CS.read_text(encoding="utf-8")
    assert '"pip"' not in source
    assert '"--install-tools"' in source
    assert not (INSTALLER_CS.parent / "PythonResolver.cs").exists()


def test_the_auto_editor_pin_is_one_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+", optional_tools.AUTO_EDITOR_VERSION)
    assert f"/download/{optional_tools.AUTO_EDITOR_VERSION}/" in optional_tools.AUTO_EDITOR_URL
    assert optional_tools.AUTO_EDITOR_URL.startswith("https://github.com/WyattBlue/auto-editor/")
    assert re.fullmatch(r"[0-9a-f]{64}", optional_tools.AUTO_EDITOR_SHA256)


# ---------------------------------------------------------------------------
# auto-editor
# ---------------------------------------------------------------------------
def _python_version_line():
    return subprocess.run(
        [sys.executable, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_validator_runs_the_staged_file_it_hashed(monkeypatch):
    """A real executable stands in: its hash, then its ``--version`` output."""
    monkeypatch.setattr(optional_tools, "AUTO_EDITOR_SHA256", _sha(sys.executable))
    monkeypatch.setattr(optional_tools, "AUTO_EDITOR_VERSION", _python_version_line())

    assert optional_tools._check_auto_editor(sys.executable) is True


def test_the_validator_refuses_other_bytes_before_running_them(monkeypatch, tmp_path):
    staged = tmp_path / ".auto-editor.exe.x.part"
    staged.write_bytes(b"MZ not the pinned build")
    monkeypatch.setattr(
        optional_tools, "_run", lambda *a, **k: pytest.fail("ran an unverified download")
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        optional_tools._check_auto_editor(str(staged))


def test_the_validator_refuses_a_build_that_reports_another_version(monkeypatch):
    monkeypatch.setattr(optional_tools, "AUTO_EDITOR_SHA256", _sha(sys.executable))
    monkeypatch.setattr(optional_tools, "AUTO_EDITOR_VERSION", "31.8.0")

    with pytest.raises(RuntimeError, match="didn't run here"):
        optional_tools._check_auto_editor(sys.executable)


@pytest.mark.skipif(os.name != "nt", reason="the pinned build is the Windows binary")
def test_auto_editor_downloads_into_the_data_folder_bin(monkeypatch, data_dir):
    calls = []

    def fake_download(url, destination, **kwargs):
        calls.append((url, destination, kwargs))

    monkeypatch.setattr("opencut.core.url_safety.transactional_download", fake_download)

    path = optional_tools.install_auto_editor(report=lambda _line: None)

    expected = str(data_dir / "bin" / "auto-editor.exe")
    assert path == expected
    [(url, destination, kwargs)] = calls
    assert url == optional_tools.AUTO_EDITOR_URL
    assert destination == expected
    assert kwargs["validator"] is optional_tools._check_auto_editor
    assert kwargs["max_bytes"] >= 45_473_280


@pytest.mark.skipif(os.name != "nt", reason="the pinned build is the Windows binary")
def test_the_pinned_build_already_in_place_is_not_downloaded_again(monkeypatch, data_dir):
    existing = data_dir / "bin" / "auto-editor.exe"
    existing.parent.mkdir()
    existing.write_bytes(b"pinned bytes")
    monkeypatch.setattr(optional_tools, "AUTO_EDITOR_SHA256", _sha(existing))
    monkeypatch.setattr(
        "opencut.core.url_safety.transactional_download",
        lambda *a, **k: pytest.fail("downloaded the build that was already there"),
    )

    assert optional_tools.install_auto_editor(report=lambda _line: None) == str(existing)


# ---------------------------------------------------------------------------
# pip tools
# ---------------------------------------------------------------------------
class _Machine:
    """pip writes a fake edge_tts into --target; the child server's import is scripted."""

    def __init__(self, import_exit=0, pip_exit=0):
        self.import_exit = import_exit
        self.pip_exit = pip_exit
        self.commands = []

    def __call__(self, cmd, timeout, on_start=None):
        self.commands.append(cmd)
        if "pip" in cmd:
            target = Path(cmd[cmd.index("--target") + 1])
            (target / "edge_tts").mkdir()
            (target / "edge_tts" / "__init__.py").write_text("", encoding="utf-8")
            (target / "edge_tts-7.3.0.dist-info").mkdir()
            (target / "bin").mkdir()
            (target / "bin" / "edge-tts.exe").write_bytes(b"script")
            return subprocess.CompletedProcess(cmd, self.pip_exit, "", "pip said no")
        assert "--import-check" in cmd
        return subprocess.CompletedProcess(cmd, self.import_exit, "", "Fatal Python error")


@pytest.fixture
def machine(monkeypatch, data_dir):
    monkeypatch.setattr(security, "install_python", lambda: "C:/Python313/python.exe")
    packages = data_dir / "packages"
    packages.mkdir()
    (packages / "edge_tts-6.1.9.dist-info").mkdir()
    (packages / "edge_tts").mkdir()
    (packages / "edge_tts" / "old.py").write_text("", encoding="utf-8")
    (packages / "rembg-2.0.0.dist-info").mkdir()
    (packages / "bin").mkdir()
    (packages / "bin" / "rembg.exe").write_bytes(b"other tool")

    def install(**kwargs):
        fake = _Machine(**kwargs)
        monkeypatch.setattr(package_store, "_run", fake)
        return fake

    return packages, install


def _leftover_staging(data_dir):
    return [p.name for p in data_dir.iterdir() if "staging" in p.name]


def _data_entries(data_dir):
    """The data folder, less the store's lock file and its inventory marker."""
    return sorted(
        p.name for p in data_dir.iterdir() if p.name not in ("packages.lock", "packages-inventory.json")
    )


def test_a_pip_tool_is_merged_only_after_the_server_imports_it(machine, data_dir):
    packages, install = machine
    fake = install()

    optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    pip, check = fake.commands
    assert pip[:4] == ["C:/Python313/python.exe", "-m", "pip", "install"]
    assert "edge-tts>=6.1,<8" in pip
    staging = pip[pip.index("--target") + 1]
    assert os.path.dirname(staging) == str(data_dir)
    assert check[check.index("--import-check") + 1] == "edge_tts"
    assert check[check.index("--import-path") + 1] == staging

    entries = sorted(p.name for p in packages.iterdir())
    assert entries == ["bin", "edge_tts", "edge_tts-7.3.0.dist-info", "rembg-2.0.0.dist-info"]
    assert not (packages / "edge_tts" / "old.py").exists()
    assert sorted(p.name for p in (packages / "bin").iterdir()) == ["edge-tts.exe", "rembg.exe"]
    assert _leftover_staging(data_dir) == []


def test_a_tool_that_crashes_the_server_never_reaches_the_packages_folder(machine, data_dir):
    packages, install = machine
    install(import_exit=CRASH_EXIT)
    before = sorted(str(p.relative_to(packages)) for p in packages.rglob("*"))

    with pytest.raises(RuntimeError, match=r"couldn't import edge_tts \(exit code"):
        optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    assert sorted(str(p.relative_to(packages)) for p in packages.rglob("*")) == before
    assert _leftover_staging(data_dir) == []


def test_a_failed_pip_run_is_not_import_checked(machine, data_dir):
    _packages, install = machine
    fake = install(pip_exit=1)

    with pytest.raises(RuntimeError, match="pip said no"):
        optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    assert len(fake.commands) == 1
    assert _leftover_staging(data_dir) == []


def _tree(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def test_a_merge_that_stops_partway_puts_the_old_install_back(machine, data_dir, monkeypatch):
    packages, install = machine
    install()
    before = _tree(packages)
    real_replace = os.replace

    def replace(src, dst):
        # bin/edge-tts.exe and edge_tts are already in by now (staged entries go in sorted)
        if os.path.basename(dst) == "edge_tts-7.3.0.dist-info":
            raise PermissionError(13, "The process cannot access the file", dst)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(RuntimeError, match=r"The copy that was there is back\. If OpenCut is running"):
        optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    assert _tree(packages) == before
    assert _data_entries(data_dir) == ["packages"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows won't move a folder with an open file in it")
def test_a_file_the_running_server_holds_open_leaves_the_old_install_whole(machine, data_dir):
    packages, install = machine
    install()
    before = _tree(packages)

    with open(packages / "edge_tts" / "old.py", "rb"):
        with pytest.raises(RuntimeError, match="close it and Premiere Pro"):
            optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    assert _tree(packages) == before
    assert _data_entries(data_dir) == ["packages"]


def test_folders_a_killed_install_left_behind_are_cleared_once_stale(machine, data_dir):
    _packages, install = machine
    install()
    two_hours_ago = time.time() - 2 * 3600
    for name in (".edge-tts-staging-k1ll3d", ".mediapipe-staging-0ld", ".packages-replaced-k1ll3d"):
        (data_dir / name / "edge_tts").mkdir(parents=True)
        (data_dir / name / "edge_tts" / "__init__.py").write_text("", encoding="utf-8")
        os.utime(data_dir / name, (two_hours_ago, two_hours_ago))
    (data_dir / ".edge-tts-staging-running").mkdir()  # another run that's still going
    (data_dir / "models").mkdir()
    os.utime(data_dir / "models", (two_hours_ago, two_hours_ago))

    optional_tools.install_pip_tool("edge-tts", report=lambda _line: None)

    assert _data_entries(data_dir) == [
        ".edge-tts-staging-running", "models", "packages",
    ]


def test_no_matching_python_names_the_one_to_install(monkeypatch, data_dir):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(security, "_find_system_python", lambda: None)
    monkeypatch.setattr(package_store, "_run", lambda *a, **k: pytest.fail("ran pip"))

    with pytest.raises(RuntimeError, match=r"needs .+ Python \d+\.\d+\. Install it from python\.org"):
        optional_tools.install_pip_tool("mediapipe", report=lambda _line: None)


# ---------------------------------------------------------------------------
# The child server's import check, for real
# ---------------------------------------------------------------------------
@pytest.fixture
def child_env(monkeypatch, data_dir):
    monkeypatch.setenv("OPENCUT_HOME", str(data_dir))
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.delattr(sys, "frozen", raising=False)


def test_the_server_imports_a_module_only_the_staging_folder_has(child_env, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "oc_staged_only_probe.py").write_text("VALUE = 1\n", encoding="utf-8")

    optional_tools.smoke_import("oc_staged_only_probe", str(staging))

    with pytest.raises(RuntimeError, match=r"exit code 1\).*No module named 'oc_not_staged_probe'"):
        optional_tools.smoke_import("oc_not_staged_probe", str(staging))


def test_the_import_check_refuses_anything_but_a_module_name():
    assert optional_tools.run_import_check("json; import os", None) == 2
    assert optional_tools.run_import_check("", None) == 2


FROZEN_SERVER = REPO_ROOT / "dist" / "OpenCut-Server" / (
    "OpenCut-Server.exe" if os.name == "nt" else "OpenCut-Server"
)


@pytest.mark.skipif(not FROZEN_SERVER.is_file(), reason="no dist/OpenCut-Server build present")
@pytest.mark.parametrize("module", ["plistlib", "zoneinfo", "mailbox", "dbm", "graphlib"])
def test_the_built_server_has_the_standard_library_add_ons_import(module, data_dir):
    """mediapipe's matplotlib imports plistlib, which the bundle used to leave out."""
    result = subprocess.run(
        [str(FROZEN_SERVER), "--import-check", module],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env={**os.environ, "OPENCUT_HOME": str(data_dir)},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# --install-tools
# ---------------------------------------------------------------------------
def test_unknown_tools_are_refused_before_anything_installs(monkeypatch):
    lines = []
    monkeypatch.setattr(optional_tools, "install_auto_editor", lambda r: pytest.fail("installed"))

    assert optional_tools.install_tools(["auto-editor", "whisperx"], lines.append) == 2
    assert lines == ["[failed] whisperx: choose from auto-editor, edge-tts, mediapipe"]


def test_each_tool_ends_with_one_status_line_and_a_failure_does_not_stop_the_rest(monkeypatch):
    lines = []
    monkeypatch.setattr(optional_tools, "install_auto_editor", lambda r: "C:/data/bin/auto-editor.exe")

    def pip_tool(name, report):
        if name == "mediapipe":
            raise RuntimeError("no wheel")

    monkeypatch.setattr(optional_tools, "install_pip_tool", pip_tool)

    code = optional_tools.install_tools([" Edge-TTS", "mediapipe", "auto-editor", "edge-tts"], lines.append)

    assert code == 1
    assert lines == [
        "[ok] edge-tts is ready",
        "[failed] mediapipe: no wheel",
        f"[ok] auto-editor {optional_tools.AUTO_EDITOR_VERSION} is in C:/data/bin/auto-editor.exe",
    ]


def test_a_failure_with_a_multi_line_reason_stays_on_its_status_line(monkeypatch):
    lines = []

    def pip_tool(name, report):
        raise RuntimeError(
            "pip couldn't install mediapipe: ERROR: Could not find a version\r\n"
            "  that satisfies the requirement\nERROR: No matching distribution"
        )

    monkeypatch.setattr(optional_tools, "install_pip_tool", pip_tool)

    assert optional_tools.install_tools(["mediapipe"], lines.append) == 1
    assert lines == [
        "[failed] mediapipe: pip couldn't install mediapipe: ERROR: Could not find a version"
        " that satisfies the requirement ERROR: No matching distribution"
    ]
