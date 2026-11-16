"""F447: optional capability probes run in a child process.

``/health`` used to import Torch and the other optional native stacks in the
server, so a library that aborts on import took the server down before it
answered (discussions #9 and #10). These tests install packages that really
crash their process, then check that the server keeps answering and reports
the probe as unavailable with a reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from opencut import crash_reporter
from opencut.core import capability_probe
from tests.conftest import csrf_headers

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_MARK = """\
import pathlib
_marker = pathlib.Path(__file__).with_name("attempts.txt")
with _marker.open("a", encoding="utf-8") as fh:
    fh.write("x")
"""

# What each broken torch does after it leaves its mark.
_TORCH_BODIES = {
    # A thread started at address 0 faults with STATUS_ACCESS_VIOLATION, the
    # way a native initializer built for another runtime takes a process down.
    "access_violation": (
        "import ctypes, sys, time\n"
        "if sys.platform == 'win32':\n"
        "    ctypes.windll.kernel32.CreateThread(None, 0, None, None, 0, None)\n"
        "    time.sleep(60)\n"
        "ctypes.string_at(0)\n"
    ),
    "abort": "import os\nos.abort()\n",
    "exit": "import os\nos._exit(5)\n",
    "hang": "import time\ntime.sleep(120)\n",
    "oserror": 'raise OSError("[WinError 206] The filename or extension is too long")\n',
}

if os.name == "nt":
    _EXIT_CODES = {"access_violation": 0xC0000005, "abort": 3, "exit": 5}
    _EXIT_TEXT = {
        "access_violation": "exit code 0xC0000005 (access violation)",
        "abort": "exit code 3 (abort)",
        "exit": "exit code 5",
    }
else:  # pragma: no cover - the suite runs on Windows
    _EXIT_CODES = {"access_violation": -11, "abort": -6, "exit": 5}
    _EXIT_TEXT = {"access_violation": "SIGSEGV", "abort": "SIGABRT", "exit": "exit code 5"}


def _package(root, name, body):
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_MARK + body, encoding="utf-8")
    return pkg / "attempts.txt"


def _attempts(marker) -> str:
    return marker.read_text(encoding="utf-8") if marker.exists() else ""


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """A folder ahead of site-packages; the probe child gets the same path."""
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    # opentimelineio stands in for a healthy optional package.
    (site / "opentimelineio.py").write_text(_MARK, encoding="utf-8")
    return site


@pytest.fixture
def crash_log(tmp_path, monkeypatch):
    folder = tmp_path / "server-data"
    folder.mkdir()
    path = folder / "crash.log"
    monkeypatch.setattr(crash_reporter, "OPENCUT_DIR", str(folder))
    monkeypatch.setattr(crash_reporter, "CRASH_LOG", str(path))
    return path


@pytest.fixture
def child_home(tmp_path):
    home = tmp_path / "child-home"
    (home / ".opencut").mkdir(parents=True)
    return home


@pytest.fixture
def make_discovery(tmp_path, child_home):
    """Fresh discoveries with their own cache, installed for the routes."""
    made = []

    def make(probes, *, install=True, cache=None, **kwargs):
        discovery = capability_probe.CapabilityDiscovery(
            cache_path=str(cache or tmp_path / "capability_probes.json"),
            probes=probes,
            env={
                "HOME": str(child_home),
                "USERPROFILE": str(child_home),
                "OPENCUT_HOME": str(child_home / ".opencut"),
            },
            **kwargs,
        )
        previous = capability_probe.set_discovery(discovery) if install else None
        made.append((discovery, install, previous))
        return discovery

    yield make
    for discovery, install, previous in reversed(made):
        discovery.close()
        if install:
            capability_probe.set_discovery(previous)


@pytest.fixture
def discovering_app(app, csrf_token, monkeypatch):
    """The test app with discovery switched on, as a real server has it.

    The CSRF token is fetched first: a /health call with discovery on and no
    test discovery installed would start the default one over every probe.
    """
    monkeypatch.setitem(app.config, "OPENCUT_CAPABILITY_DISCOVERY", True)
    return app


def _health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    return resp.get_json()


@pytest.mark.parametrize("kind", ["access_violation", "abort", "exit"])
def test_a_probe_that_kills_its_process_leaves_the_server_answering(
    kind, fake_site, crash_log, make_discovery, discovering_app, csrf_token
):
    torch_attempts = _package(fake_site, "torch", _TORCH_BODIES[kind])
    discovery = make_discovery(("silero_vad", "otio"))
    client = discovering_app.test_client()

    first = _health(client)
    assert "silero_vad" not in first["capabilities"]
    assert set(first["capability_discovery"]["pending"]) == {"silero_vad", "otio"}
    assert first["capabilities"]["ffmpeg"] is True

    assert discovery.wait(240), "discovery never finished"
    after = _health(client)

    caps = after["capabilities"]
    assert caps["silero_vad"] is False
    assert caps["otio"] is True  # a fresh child ran the probe after the crash
    failed = after["capability_discovery"]["failed"]
    assert set(failed) == {"silero_vad"}
    assert failed["silero_vad"]["state"] == "crashed"
    assert failed["silero_vad"]["exit_code"] == _EXIT_CODES[kind]
    assert _EXIT_TEXT[kind] in failed["silero_vad"]["reason"]
    assert after["capability_discovery"]["state"] == "ready"
    assert discovery.spawn_count == 2
    # Only the child ever imported the broken torch.
    assert _attempts(torch_attempts) == "x"

    failures = client.get("/system/check-failures", headers=csrf_headers(csrf_token)).get_json()["failures"]
    assert failures["probe:silero_vad"]["state"] == "crashed"
    assert failures["probe:silero_vad"]["exception"] == "ProbeCrashed"
    assert "probe:otio" not in failures

    record = crash_log.read_text(encoding="utf-8")
    assert "capability probe crashed" in record
    assert "probe: silero_vad" in record
    assert _EXIT_TEXT[kind] in record


def test_a_probe_that_hangs_is_killed_and_health_never_waits_for_it(
    fake_site, crash_log, make_discovery, discovering_app
):
    torch_attempts = _package(fake_site, "torch", _TORCH_BODIES["hang"])
    discovery = make_discovery(("silero_vad", "otio"), probe_timeout=8)
    client = discovering_app.test_client()

    _health(client)
    deadline = time.monotonic() + 120
    while not _attempts(torch_attempts) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _attempts(torch_attempts) == "x", "the probe child never reached torch"

    # The child is stuck inside the import right now.
    started = time.monotonic()
    during = _health(client)
    assert time.monotonic() - started < 4
    assert during["capability_discovery"]["state"] == "running"
    assert "silero_vad" in during["capability_discovery"]["pending"]

    assert discovery.wait(240), "discovery never finished"
    after = _health(client)
    assert after["capabilities"]["silero_vad"] is False
    assert after["capabilities"]["otio"] is True
    failed = after["capability_discovery"]["failed"]["silero_vad"]
    assert failed["state"] == "timeout"
    assert failed["reason"] == "no answer within 8 seconds"
    assert "capability probe timeout" in crash_log.read_text(encoding="utf-8")


def test_concurrent_health_requests_share_one_probe_run(fake_site, make_discovery, discovering_app):
    otio_attempts = fake_site / "attempts.txt"
    discovery = make_discovery(("otio",))
    barrier = threading.Barrier(8)
    codes = []

    def poll():
        client = discovering_app.test_client()
        barrier.wait()
        codes.append(client.get("/health").status_code)

    threads = [threading.Thread(target=poll) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert codes == [200] * 8
    assert discovery.wait(240)
    client = discovering_app.test_client()
    for _ in range(3):
        assert _health(client)["capabilities"]["otio"] is True
    assert discovery.spawn_count == 1
    assert _attempts(otio_attempts) == "x"


def test_saved_results_are_reused_until_the_installed_packages_change(fake_site, tmp_path, make_discovery):
    cache = tmp_path / "saved.json"
    first = make_discovery(("otio",), install=False, cache=cache)
    assert first.ensure_started() is True
    assert first.wait(240)
    assert first.capabilities() == {"otio": True}
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["fingerprint"] == capability_probe.runtime_fingerprint()

    # A restart with the same runtime and packages reuses the answer.
    second = make_discovery(("otio",), install=False, cache=cache)
    assert second.ensure_started() is False
    assert second.capabilities() == {"otio": True}
    assert second.spawn_count == 0

    # Installing a package changes the fingerprint, so everything is probed again.
    (fake_site / "newly_installed-1.0.dist-info").mkdir()
    second._fingerprint_checked = 0.0
    assert second.ensure_started() is True
    assert second.wait(240)
    assert second.spawn_count == 1
    assert second.capabilities() == {"otio": True}


def test_health_reports_a_torch_import_error_from_the_child_and_never_imports_it(
    fake_site, make_discovery, discovering_app, csrf_token
):
    """Discussion #10: a Torch DLL failure raising OSError. The verdict comes
    from the probe child, and polls after it don't import torch again."""
    attempts = _package(fake_site, "torch", _TORCH_BODIES["oserror"])
    discovery = make_discovery(("silero_vad", "video_ai"))
    client = discovering_app.test_client()

    _health(client)
    assert discovery.wait(240)
    for _ in range(3):
        caps = _health(client)["capabilities"]
        assert caps["silero_vad"] is False
        assert caps["video_ai"]["rvm"] is False
    assert _attempts(attempts) == "x"
    assert discovery.spawn_count == 1

    failures = client.get("/system/check-failures", headers=csrf_headers(csrf_token)).get_json()["failures"]
    assert failures["import:torch"]["exception"] == "OSError"
    assert "WinError 206" in failures["import:torch"]["message"]
    # An import error is an answer, not a crash.
    assert "probe:silero_vad" not in failures


def test_every_probe_contains_a_broken_native_module(fake_site, make_discovery, discovering_app, csrf_token):
    """The checks in captions_enhanced, voice_gen, audio_pro and face_tools
    caught only ImportError, and a broken kokoro hid edge_tts. cv2 is left
    out: it's a core dependency the server itself imports at startup."""
    modules = ["whisperx", "kokoro", "pedalboard", "df", "mediapipe", "insightface"]
    attempts = {
        name: _package(fake_site, name, f'raise OSError("[WinError 126] {name}.dll")\n') for name in modules
    }
    (fake_site / "edge_tts.py").write_text("VALUE = 1\n", encoding="utf-8")
    discovery = make_discovery(("whisperx", "kokoro", "pedalboard", "deepfilter", "face_tools", "edge_tts"))
    client = discovering_app.test_client()

    _health(client)
    assert discovery.wait(240)
    caps = _health(client)["capabilities"]

    assert caps["edge_tts"] is True
    assert caps["kokoro"] is False
    assert caps["whisperx"] is False
    assert caps["pedalboard"] is False
    assert caps["deepfilter"] is False
    assert caps["face_tools"]["mediapipe"] is False
    assert caps["face_tools"]["insightface"] is False
    failures = client.get("/system/check-failures", headers=csrf_headers(csrf_token)).get_json()["failures"]
    for name in modules:
        assert _attempts(attempts[name]) == "x", name
        assert failures[f"import:{name}"]["exception"] == "OSError", name


_STARTUP = """\
import ctypes
import json
import pathlib
import sys

if sys.platform == "win32":
    # No crash dialog if this process dies; the test reads the exit code.
    ctypes.windll.kernel32.SetErrorMode(0x8003)

from opencut.config import OpenCutConfig
from opencut.server import create_app

attempts = pathlib.Path(sys.argv[1])
client = create_app(config=OpenCutConfig(), testing=True).test_client()
codes = [client.get("/health").status_code for _ in range(3)]
print("RESULT " + json.dumps([codes, attempts.exists()]))
"""


@pytest.mark.parametrize("kind", ["access_violation", "oserror"])
def test_the_server_starts_and_answers_with_a_broken_torch_installed(kind, fake_site, tmp_path):
    """gpu.py imported torch as the server started, so a torch that crashed
    natively on import kept the server from ever answering."""
    attempts = _package(fake_site, "torch", _TORCH_BODIES[kind])
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
        [sys.executable, "-c", _STARTUP, str(attempts)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        creationflags=_NO_WINDOW,
    )
    assert proc.returncode == 0, f"exit {proc.returncode:#x}: {proc.stderr[-2000:]}"
    result = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    codes, imported = json.loads(result[len("RESULT "):])
    assert codes == [200, 200, 200]
    assert imported is False


def test_describe_exit_names_windows_crash_codes():
    assert capability_probe.describe_exit(3221225477) == "exit code 0xC0000005 (access violation)"
    assert capability_probe.describe_exit(-1073741819) == "exit code 0xC0000005 (access violation)"
    assert capability_probe.describe_exit(0xC0000409) == "exit code 0xC0000409 (fast fail (stack buffer overrun))"
    assert capability_probe.describe_exit(5) == "exit code 5"
    assert capability_probe.describe_exit(None) == "no exit code"


# A probe import that starts a long-lived helper process handing it the
# child's real stdout, the way a library might start a compile server.
_HELPER_BODY = """\
import subprocess, sys
_helper = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(90)"],
    stdin=subprocess.DEVNULL, stdout=sys.__stdout__.fileno(), stderr=subprocess.DEVNULL,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
pathlib.Path(__file__).with_name("helper.pid").write_text(str(_helper.pid), encoding="utf-8")
"""


def _alive(pid: int) -> bool:
    if os.name != "nt":  # pragma: no cover - the suite runs on Windows
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return k32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def _dies(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def helper_pid(fake_site):
    """The helper's PID once the probe started it; killed if it outlives the test."""
    path = fake_site / "helper.pid"
    read = []

    def get():
        deadline = time.monotonic() + 60
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        pid = int(path.read_text(encoding="utf-8"))
        read.append(pid)
        return pid

    yield get
    for pid in read:
        if _alive(pid):
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, creationflags=_NO_WINDOW)


def test_a_helper_holding_the_probe_pipe_doesnt_stall_discovery(fake_site, make_discovery, helper_pid):
    """The child finishes every probe and exits, but the helper keeps its
    stdout open, so EOF never came: discovery sat out the probe timeout and
    then blocked closing the pipe until the helper quit."""
    (fake_site / "opentimelineio.py").write_text(_MARK + _HELPER_BODY, encoding="utf-8")
    (fake_site / "edge_tts.py").write_text("VALUE = 1\n", encoding="utf-8")
    discovery = make_discovery(("otio", "edge_tts"), install=False, probe_timeout=60)

    started = time.monotonic()
    assert discovery.ensure_started()
    assert discovery.wait(45), "discovery stalled behind the helper"
    assert time.monotonic() - started < 30
    snap = discovery.snapshot()
    assert snap["state"] == "ready"
    assert snap["failed"] == {}
    assert discovery.capabilities()["otio"] is True
    assert discovery.spawn_count == 1
    assert _dies(helper_pid()), "the helper outlived its probe child"


def test_a_hung_probe_and_the_helper_it_started_are_killed_together(
    fake_site, crash_log, make_discovery, helper_pid
):
    (fake_site / "opentimelineio.py").write_text(
        _MARK + _HELPER_BODY + "import time\ntime.sleep(120)\n", encoding="utf-8",
    )
    discovery = make_discovery(("otio",), install=False, probe_timeout=6)

    started = time.monotonic()
    assert discovery.ensure_started()
    assert discovery.wait(45), "discovery stalled behind the helper"
    assert time.monotonic() - started < 30
    failed = discovery.snapshot()["failed"]["otio"]
    assert failed["state"] == "timeout"
    assert failed["reason"] == "no answer within 6 seconds"
    assert _dies(helper_pid()), "killing the hung child left its helper running"


def test_a_probe_that_prints_without_a_newline_still_reports(fake_site, make_discovery):
    """Output with no trailing newline ran into the result line, so the
    result was lost and the probe read as crashed with exit code 0."""
    (fake_site / "opentimelineio.py").write_text(
        _MARK + "import os\nprint('printed', end='')\nos.write(1, b'written to fd 1')\n",
        encoding="utf-8",
    )
    discovery = make_discovery(("otio",), install=False)

    assert discovery.ensure_started()
    assert discovery.wait(60)
    assert discovery.snapshot()["failed"] == {}
    assert discovery.capabilities()["otio"] is True


def test_shutdown_stops_a_running_probe_child(app, csrf_token, make_discovery, monkeypatch):
    """/shutdown ends the server with os._exit, which skips the atexit hook
    that kills the probe child, so a hung probe outlived the server."""
    discovery = make_discovery(("otio",))
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
    )
    try:
        with discovery._lock:
            discovery._active = child
        exited = threading.Event()
        monkeypatch.setattr(os, "_exit", lambda code: exited.set())

        resp = app.test_client().post("/shutdown", headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        assert exited.wait(10), "the shutdown thread never reached os._exit"
        assert _dies(child.pid), "the probe child outlived the server"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(10)
