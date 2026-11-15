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


_HEALTH_PROBED = ("whisperx", "kokoro", "pedalboard", "df", "cv2", "mediapipe", "insightface")


def test_health_contains_every_optional_probe_it_runs(fake_site, client, csrf_token, monkeypatch):
    """The availability checks in captions_enhanced, voice_gen, audio_pro and
    face_tools caught only ImportError, so a broken native library escaped,
    was imported again on every poll, and a broken kokoro hid edge_tts."""
    attempts = {}
    for name in _HEALTH_PROBED:
        monkeypatch.delitem(sys.modules, name, raising=False)
        attempts[name] = _broken_package(fake_site, name, f'OSError("[WinError 126] {name}.dll")')
    monkeypatch.delitem(sys.modules, "edge_tts", raising=False)
    (fake_site / "edge_tts.py").write_text("VALUE = 1\n", encoding="utf-8")
    from opencut.routes import system

    caps = None
    for _ in range(3):
        system.invalidate_caps_cache()
        resp = client.get("/health")
        assert resp.status_code == 200
        caps = resp.get_json()["capabilities"]

    assert caps["edge_tts"] is True
    assert caps["kokoro"] is False
    assert caps["whisperx"] is False
    assert caps["pedalboard"] is False
    assert caps["face_tools"]["opencv"] is False
    assert caps["face_tools"]["mediapipe"] is False
    assert caps["face_tools"]["insightface"] is False
    failures = client.get("/system/check-failures", headers=csrf_headers(csrf_token)).get_json()["failures"]
    for name in _HEALTH_PROBED:
        assert attempts[name].read_text(encoding="utf-8") == "x", name
        assert failures[f"import:{name}"]["exception"] == "OSError", name


_SLOW_BROKEN_INIT = """\
import pathlib
import time
_marker = pathlib.Path(__file__).with_name("attempts.txt")
with _marker.open("a", encoding="utf-8") as fh:
    fh.write("x")
time.sleep(0.3)
raise OSError("[WinError 126] slow.dll")
"""


def test_concurrent_first_probes_import_a_broken_module_once(fake_site):
    import threading

    pkg = fake_site / "f462_slow_broken"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_SLOW_BROKEN_INIT, encoding="utf-8")
    barrier = threading.Barrier(4)
    results = []

    def probe():
        barrier.wait()
        results.append(helpers._try_import("f462_slow_broken"))

    threads = [threading.Thread(target=probe) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert results == [None, None, None, None]
    assert (pkg / "attempts.txt").read_text(encoding="utf-8") == "x"


def test_a_probe_during_another_threads_import_gets_the_finished_module(fake_site):
    import importlib
    import threading
    import time

    (fake_site / "f462_slow_ok.py").write_text("import time\ntime.sleep(0.6)\nREADY = True\n", encoding="utf-8")
    loader = threading.Thread(target=importlib.import_module, args=("f462_slow_ok",))
    loader.start()
    try:
        deadline = time.monotonic() + 10
        while "f462_slow_ok" not in sys.modules and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "f462_slow_ok" in sys.modules

        module = helpers._try_import("f462_slow_ok")

        assert module is not None
        assert module.READY is True
    finally:
        loader.join(timeout=30)
        sys.modules.pop("f462_slow_ok", None)


def test_modules_that_probe_each_other_from_two_threads_both_finish(fake_site):
    """A module that calls _try_import while it is itself being imported must
    not hold a second per-name lock: the other thread can be waiting on
    Python's module lock for the first, and that cycle never resolved."""
    import threading

    (fake_site / "f462_cycle_a.py").write_text(
        "import time\n"
        "from opencut import helpers\n"
        "time.sleep(0.4)\n"
        "B = helpers._try_import('f462_cycle_b')\n"
        "VALUE = 'a'\n",
        encoding="utf-8",
    )
    (fake_site / "f462_cycle_b.py").write_text(
        "import time\ntime.sleep(0.4)\nimport f462_cycle_a\nVALUE = 'b'\n",
        encoding="utf-8",
    )
    barrier = threading.Barrier(2)
    results = {}

    def probe(name):
        barrier.wait()
        results[name] = helpers._try_import(name)

    threads = [
        threading.Thread(target=probe, args=(name,), daemon=True)
        for name in ("f462_cycle_a", "f462_cycle_b")
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        assert not any(t.is_alive() for t in threads), "the two probes deadlocked"
        assert set(results) == {"f462_cycle_a", "f462_cycle_b"}
    finally:
        sys.modules.pop("f462_cycle_a", None)
        sys.modules.pop("f462_cycle_b", None)


def test_a_module_that_imports_after_a_retryable_failure_is_handed_back(fake_site):
    """ensure_package installs a missing dependency and then imports the
    module itself; the earlier ImportError record must not hide it."""
    import importlib

    (fake_site / "f462_needs_dep.py").write_text(
        "import f462_the_dep\nVALUE = 5\n", encoding="utf-8"
    )
    try:
        assert helpers._try_import("f462_needs_dep") is None
        assert helpers.import_failures()["f462_needs_dep"]["retryable"] is True

        (fake_site / "f462_the_dep.py").write_text("X = 1\n", encoding="utf-8")
        importlib.invalidate_caches()
        importlib.import_module("f462_needs_dep")

        module = helpers._try_import("f462_needs_dep")
        assert module is not None and module.VALUE == 5
        assert "f462_needs_dep" not in helpers.import_failures()
    finally:
        sys.modules.pop("f462_needs_dep", None)
        sys.modules.pop("f462_the_dep", None)


def test_resolve_probe_records_a_scripting_module_that_raises(fake_site, monkeypatch):
    from opencut.core import resolve_bridge

    monkeypatch.setattr(resolve_bridge, "_ensure_resolve_path", lambda: None)
    (fake_site / "DaVinciResolveScript.py").write_text(
        "import pathlib\n"
        "p = pathlib.Path(__file__).with_name('attempts.txt')\n"
        "p.write_text((p.read_text() if p.exists() else '') + 'x')\n"
        "raise OSError('[WinError 126] fusionscript.dll')\n",
        encoding="utf-8",
    )

    assert [checks.check_resolve_available() for _ in range(3)] == [False, False, False]

    assert (fake_site / "attempts.txt").read_text(encoding="utf-8") == "x"
    assert helpers.import_failures()["DaVinciResolveScript"]["exception"] == "OSError"
    assert "import:DaVinciResolveScript" in checks.get_check_failures()
