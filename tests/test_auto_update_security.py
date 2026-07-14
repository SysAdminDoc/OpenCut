"""Security and install-origin contracts for the OpenCut updater."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__ = lambda value: value
    response.__exit__ = MagicMock(return_value=False)
    return response


def test_release_check_uses_canonical_repository_and_keeps_asset_digest():
    from opencut.core.auto_update import GITHUB_API_URL, get_latest_release

    payload = {
        "tag_name": "v2.0.0",
        "assets": [
            {
                "name": "OpenCut-Setup-2.0.0.exe",
                "browser_download_url": "https://github.com/SysAdminDoc/OpenCut/releases/x",
                "digest": "sha256:" + "a" * 64,
            }
        ],
    }
    with patch("opencut.core.auto_update.urlopen", return_value=_response(payload)) as opened:
        release = get_latest_release()

    assert GITHUB_API_URL == "https://api.github.com/repos/SysAdminDoc/OpenCut/releases"
    assert opened.call_args.args[0].full_url == f"{GITHUB_API_URL}/latest"
    assert release.assets[0]["digest"] == "sha256:" + "a" * 64


def test_packaged_origin_disables_unverified_automatic_update():
    from opencut.core.auto_update import detect_install_origin

    with patch.object(sys, "frozen", True, create=True):
        origin = detect_install_origin()

    assert origin.kind == "packaged"
    assert origin.update_supported is False
    assert "signed installer feed" in origin.reason


def test_source_origin_requires_canonical_remote(tmp_path):
    from opencut.core.auto_update import detect_install_origin

    (tmp_path / ".git").mkdir()
    with (
        patch("opencut.core.auto_update._repo_root", return_value=tmp_path),
        patch("opencut.core.auto_update._git_origin", return_value="https://evil.test/fork"),
    ):
        origin = detect_install_origin()

    assert origin.kind == "source"
    assert origin.update_supported is False
    assert "not the canonical" in origin.reason


def test_source_origin_accepts_canonical_ssh_remote(tmp_path):
    from opencut.core.auto_update import detect_install_origin

    (tmp_path / ".git").mkdir()
    with (
        patch("opencut.core.auto_update._repo_root", return_value=tmp_path),
        patch(
            "opencut.core.auto_update._git_origin",
            return_value="git@github.com:SysAdminDoc/OpenCut.git",
        ),
    ):
        origin = detect_install_origin()

    assert origin.kind == "source"
    assert origin.update_method == "git"


def test_trigger_refuses_method_that_does_not_match_origin():
    from opencut.core.auto_update import InstallOrigin, trigger_update

    origin = InstallOrigin("source", "git", repo_root="C:/repo")
    with (
        patch("opencut.core.auto_update.detect_install_origin", return_value=origin),
        patch("opencut.core.auto_update.get_latest_release") as release,
    ):
        result = trigger_update("pip")

    assert result.success is False
    assert "can only update via 'git'" in result.message
    release.assert_not_called()


def test_trigger_refuses_downgrade_or_reinstall():
    from opencut.core.auto_update import InstallOrigin, ReleaseInfo, trigger_update

    with (
        patch(
            "opencut.core.auto_update.detect_install_origin",
            return_value=InstallOrigin("wheel", "pip"),
        ),
        patch("opencut.core.auto_update._read_installed_version", return_value="2.0.0"),
        patch(
            "opencut.core.auto_update.get_latest_release",
            return_value=ReleaseInfo(version="1.9.0"),
        ),
        patch("opencut.core.auto_update._update_via_pip") as update,
    ):
        result = trigger_update("auto")

    assert result.success is False
    assert "Refusing downgrade" in result.message
    update.assert_not_called()


def test_pip_update_downloads_exact_distribution_and_verifies_digest():
    from opencut.core.auto_update import PYPI_DISTRIBUTION, _update_via_pip

    wheel_bytes = b"signed-by-pypi-metadata"
    wheel_name = "opencut_ppro-2.0.0-py3-none-any.whl"
    wheel_hash = hashlib.sha256(wheel_bytes).hexdigest()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "download" in command:
            destination = Path(command[command.index("--dest") + 1])
            (destination / wheel_name).write_bytes(wheel_bytes)
        return subprocess.CompletedProcess(command, 0, "", "")

    with (
        patch(
            "opencut.core.auto_update._fetch_pypi_release_hashes",
            return_value={wheel_name: wheel_hash},
        ),
        patch("opencut.core.auto_update.subprocess.run", side_effect=run),
        patch("opencut.core.auto_update._read_installed_version", return_value="2.0.0"),
    ):
        result = _update_via_pip("2.0.0")

    assert result.success is True
    assert commands[0][-1] == f"{PYPI_DISTRIBUTION}==2.0.0"
    assert commands[0][3:6] == ["download", "--disable-pip-version-check", "--no-deps"]
    assert commands[1][3] == "install"
    assert commands[1][-1].endswith(wheel_name)


def test_pip_update_never_installs_digest_mismatch():
    from opencut.core.auto_update import _update_via_pip

    wheel_name = "opencut_ppro-2.0.0-py3-none-any.whl"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index("--dest") + 1])
        (destination / wheel_name).write_bytes(b"tampered")
        return subprocess.CompletedProcess(command, 0, "", "")

    with (
        patch(
            "opencut.core.auto_update._fetch_pypi_release_hashes",
            return_value={wheel_name: "0" * 64},
        ),
        patch("opencut.core.auto_update.subprocess.run", side_effect=run),
    ):
        result = _update_via_pip("2.0.0")

    assert result.success is False
    assert "SHA-256 verification failed" in result.message
    assert len(commands) == 1


def test_git_update_refuses_dirty_checkout(tmp_path):
    from opencut.core.auto_update import _update_via_git

    (tmp_path / ".git").mkdir()
    dirty = subprocess.CompletedProcess([], 0, " M opencut/core/file.py\n", "")
    with (
        patch(
            "opencut.core.auto_update._git_origin",
            return_value="https://github.com/SysAdminDoc/OpenCut.git",
        ),
        patch("opencut.core.auto_update.subprocess.run", return_value=dirty) as run,
    ):
        result = _update_via_git(tmp_path, "2.0.0")

    assert result.success is False
    assert "must be committed" in result.message
    assert run.call_count == 1
