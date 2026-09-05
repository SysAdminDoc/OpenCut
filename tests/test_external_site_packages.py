"""A frozen build must not adopt a foreign interpreter's site-packages.

The packaged server used to execute the first ``python`` on PATH and append
whatever site-packages it reported. In issue #8 a Python 3.13 build adopted
``C:\\Python312`` and then died with no traceback, which is what a native
extension module built for the wrong CPython minor version does: it aborts the
process rather than raising ImportError.

It was also an unreviewed code-execution ingress. Any writable directory on
PATH holding a ``python.exe`` got run at startup, and anything in its
site-packages could shadow a bundled module.
"""

from __future__ import annotations

import os
import sys

import pytest

from opencut import server as server_module
from opencut.server import EXTERNAL_SITE_PACKAGES_ENV, _setup_system_site_packages


@pytest.fixture()
def frozen(monkeypatch):
    """Pretend to be a PyInstaller build without being one."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(EXTERNAL_SITE_PACKAGES_ENV, raising=False)


def _forbid_subprocess(monkeypatch):
    """Fail loudly if anything shells out to an interpreter."""
    def _boom(*args, **kwargs):
        raise AssertionError(f"the frozen build executed an interpreter: {args!r}")

    monkeypatch.setattr(server_module._sp, "run", _boom)


def test_frozen_build_runs_no_interpreter_by_default(frozen, monkeypatch):
    """The regression: this used to execute whatever PATH offered."""
    _forbid_subprocess(monkeypatch)
    before = list(sys.path)
    assert _setup_system_site_packages() == []
    assert sys.path == before


def test_source_checkout_is_untouched(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    _forbid_subprocess(monkeypatch)
    assert _setup_system_site_packages() == []


def test_mismatched_minor_version_is_refused(frozen, monkeypatch, tmp_path, caplog):
    """A 3.12 interpreter must never be adopted by a 3.13 build."""
    fake = tmp_path / "python.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(EXTERNAL_SITE_PACKAGES_ENV, str(fake))

    other = (sys.version_info[0], sys.version_info[1] - 1)
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()

    def _fake_run(command, **kwargs):
        import subprocess

        probe = command[-1]
        if "version_info" in probe:
            return subprocess.CompletedProcess(command, 0, stdout=f"{other[0]}.{other[1]}\n", stderr="")
        # If we get here the version gate did not stop it.
        return subprocess.CompletedProcess(
            command, 0, stdout=f'["{site_dir.as_posix()}"]\n', stderr=""
        )

    monkeypatch.setattr(server_module._sp, "run", _fake_run)

    before = list(sys.path)
    with caplog.at_level("WARNING"):
        assert _setup_system_site_packages() == []
    assert sys.path == before, "a mismatched-ABI site-packages reached sys.path"
    assert str(site_dir) not in sys.path
    assert EXTERNAL_SITE_PACKAGES_ENV in caplog.text
    assert f"{other[0]}.{other[1]}" in caplog.text, "the rejected candidate was not reported"


def test_matching_interpreter_is_adopted_when_opted_in(frozen, monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(EXTERNAL_SITE_PACKAGES_ENV, str(fake))

    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    ours = sys.version_info[:2]

    def _fake_run(command, **kwargs):
        import subprocess

        probe = command[-1]
        if "version_info" in probe:
            return subprocess.CompletedProcess(command, 0, stdout=f"{ours[0]}.{ours[1]}\n", stderr="")
        if "getusersitepackages" in probe:
            return subprocess.CompletedProcess(command, 0, stdout="[]\n", stderr="")
        return subprocess.CompletedProcess(
            command, 0, stdout=f'["{site_dir.as_posix()}"]\n', stderr=""
        )

    monkeypatch.setattr(server_module._sp, "run", _fake_run)

    assert _setup_system_site_packages() == [str(fake)]
    assert any(os.path.samefile(entry, site_dir) for entry in sys.path if os.path.isdir(entry))


def test_an_explicit_path_does_not_fall_back_to_path_search(frozen, monkeypatch, tmp_path):
    """Naming an interpreter that is gone must not silently pick another."""
    missing = tmp_path / "gone" / "python.exe"
    monkeypatch.setenv(EXTERNAL_SITE_PACKAGES_ENV, str(missing))
    _forbid_subprocess(monkeypatch)

    before = list(sys.path)
    assert _setup_system_site_packages() == []
    assert sys.path == before


def test_auto_searches_path_but_still_gates_on_version(frozen, monkeypatch, tmp_path):
    import shutil

    monkeypatch.setenv(EXTERNAL_SITE_PACKAGES_ENV, "auto")
    candidate = tmp_path / "python.exe"
    candidate.write_text("", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: str(candidate))

    def _fake_run(command, **kwargs):
        import subprocess

        return subprocess.CompletedProcess(command, 0, stdout="2.7\n", stderr="")

    monkeypatch.setattr(server_module._sp, "run", _fake_run)

    before = list(sys.path)
    assert _setup_system_site_packages() == []
    assert sys.path == before
