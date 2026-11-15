"""F462: an optional module that raises anything on import is contained.

Discussion #10 showed a Torch DLL failure raising ``FileNotFoundError``
(WinError 206) through ``_try_import`` into ``/health``. torch 2.15 also
raises on a second ``import torch`` after a failed first one, so the
verdict has to be remembered instead of re-probed every health poll.
"""

from __future__ import annotations

import json
import sys

import pytest

from opencut import checks, helpers
from tests.conftest import csrf_headers

_BROKEN_INIT = """\
import pathlib
_marker = pathlib.Path(__file__).with_name("attempts.txt")
with _marker.open("a", encoding="utf-8") as fh:
    fh.write("x")
raise {exc}
"""


def _broken_package(root, name, exc):
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_BROKEN_INIT.format(exc=exc), encoding="utf-8")
    return pkg / "attempts.txt"


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    return site


@pytest.mark.parametrize(
    "exc, exc_name",
    [
        ('OSError("[WinError 206] The filename or extension is too long")', "OSError"),
        ('FileNotFoundError("[WinError 206] c10.dll")', "FileNotFoundError"),
        ('RuntimeError("only one import is allowed")', "RuntimeError"),
        ('ImportError("DLL load failed while importing _C")', "ImportError"),
        ("ModuleNotFoundError(\"No module named 'numpy'\", name='numpy')", "ModuleNotFoundError"),
    ],
)
def test_any_import_failure_is_a_cached_verdict(fake_site, exc, exc_name):
    attempts = _broken_package(fake_site, "f462_broken_pkg", exc)

    assert helpers._try_import("f462_broken_pkg") is None
    assert helpers._try_import("f462_broken_pkg") is None
    assert helpers._try_import_from("f462_broken_pkg", "anything") is None

    assert attempts.read_text(encoding="utf-8") == "x"
    verdict = helpers.import_failures()["f462_broken_pkg"]
    assert verdict["exception"] == exc_name
    assert verdict["message"]
    listed = checks.get_check_failures()["import:f462_broken_pkg"]
    assert listed["exception"] == exc_name


def test_a_module_that_is_not_installed_is_not_recorded_and_is_seen_once_installed(fake_site):
    assert helpers._try_import("f462_later_pkg") is None
    assert "f462_later_pkg" not in helpers.import_failures()

    (fake_site / "f462_later_pkg.py").write_text("VALUE = 7\n", encoding="utf-8")
    import importlib

    importlib.invalidate_caches()
    try:
        assert helpers._try_import("f462_later_pkg").VALUE == 7
    finally:
        sys.modules.pop("f462_later_pkg", None)


def test_an_install_retries_import_errors_but_not_broken_native_modules(fake_site):
    _broken_package(fake_site, "f462_missing_dep", "ModuleNotFoundError(\"No module named 'dep'\", name='dep')")
    _broken_package(fake_site, "f462_bad_dll", 'OSError("[WinError 126] bad.dll")')
    assert helpers._try_import("f462_missing_dep") is None
    assert helpers._try_import("f462_bad_dll") is None

    from opencut.routes import system

    system.invalidate_caps_cache()  # what a finished dependency install calls

    remaining = helpers.import_failures()
    assert "f462_missing_dep" not in remaining
    assert remaining["f462_bad_dll"]["retryable"] is False


def test_health_survives_a_torch_that_raises_oserror_and_never_reimports_it(fake_site, client, csrf_token):
    attempts = _broken_package(
        fake_site, "torch", 'OSError("[WinError 206] The filename or extension is too long")'
    )
    from opencut.routes import system

    def poll():
        system.invalidate_caps_cache()
        resp = client.get("/health")
        assert resp.status_code == 200
        caps = resp.get_json()["capabilities"]
        assert caps["silero_vad"] is False
        assert caps["video_ai"]["rvm"] is False
        return attempts.read_text(encoding="utf-8")

    after_first = poll()
    # A package importing torch for the first time (ctranslate2 does) can try
    # it during that first poll too. After it, no poll imports torch again.
    assert after_first
    assert poll() == after_first
    assert poll() == after_first

    failures = client.get("/system/check-failures", headers=csrf_headers(csrf_token)).get_json()["failures"]
    assert failures["import:torch"]["exception"] == "OSError"
    assert "WinError 206" in failures["import:torch"]["message"]


_STARTUP_PROBE = """\
import json
import pathlib
import sys

from opencut.config import OpenCutConfig
from opencut.routes import system
from opencut.server import create_app

attempts = pathlib.Path(sys.argv[1])


def count():
    return len(attempts.read_text(encoding="utf-8")) if attempts.exists() else 0


client = create_app(config=OpenCutConfig(), testing=True).test_client()
started = count()
codes = []
counts = []
for _ in range(3):
    system.invalidate_caps_cache()
    codes.append(client.get("/health").status_code)
    counts.append(count())
print("RESULT " + json.dumps([started, codes, counts]))
"""


def test_server_starts_with_a_torch_that_raises_oserror(fake_site, tmp_path):
    """gpu.py imports torch when the server starts. An OSError there used to
    stop the server before it served anything."""
    import os
    import subprocess

    attempts = _broken_package(fake_site, "torch", 'OSError("[WinError 206] c10.dll")')
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(fake_site), repo]),
        "HOME": str(home),
        "USERPROFILE": str(home),
    })
    env.pop("OPENCUT_HOME", None)
    proc = subprocess.run(
        [sys.executable, "-c", _STARTUP_PROBE, str(attempts)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    started, codes, counts = json.loads(result[len("RESULT "):])
    assert started == 1
    assert codes == [200, 200, 200]
    assert counts[1] == counts[0] and counts[2] == counts[0]
