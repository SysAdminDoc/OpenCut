"""A dying server must leave evidence behind.

``~/.opencut/crash.log`` had three readers and no writer: ``crash_packet.py``,
``issue_report.py`` and the ``/logs/crash`` download all tail a file nothing
ever produced. In issue #8 the server stopped mid-session and the log just
ended, so there was no way to tell a Python exception from a native abort.

The subprocess tests here are deliberately real processes: a segfault cannot be
simulated in-process, and faulthandler's whole value is that it works when no
Python-level hook can run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from opencut import crash_reporter


@pytest.fixture()
def crash_log(tmp_path, monkeypatch):
    """Point the reporter at a temp file and reset its install latch."""
    path = tmp_path / "crash.log"
    monkeypatch.setattr(crash_reporter, "OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(crash_reporter, "CRASH_LOG", str(path))
    monkeypatch.setattr(crash_reporter, "_installed", False)
    return path


def test_write_crash_record_captures_the_traceback_and_provenance(crash_log):
    try:
        raise ValueError("boom")
    except ValueError:
        crash_reporter.write_crash_record("unhandled exception", *sys.exc_info())

    text = crash_log.read_text(encoding="utf-8")
    assert "unhandled exception" in text
    assert "ValueError: boom" in text
    assert "sys.path" in text, "path provenance is what tells you where a bad module came from"
    assert "frozen:" in text
    assert "external_site_packages:" in text


def test_thread_excepthook_names_the_thread(crash_log, monkeypatch):
    import threading

    crash_reporter.install_crash_handlers(enable_faulthandler=False)

    def _boom():
        raise RuntimeError("worker died")

    worker = threading.Thread(target=_boom, name="opencut-worker-7")
    worker.start()
    worker.join()

    text = crash_log.read_text(encoding="utf-8")
    assert "unhandled thread exception" in text
    assert "opencut-worker-7" in text
    assert "RuntimeError: worker died" in text


def test_keyboard_interrupt_is_not_recorded_as_a_crash(crash_log):
    crash_reporter.install_crash_handlers(enable_faulthandler=False)
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())
    assert not crash_log.exists() or "KeyboardInterrupt" not in crash_log.read_text(encoding="utf-8")


def test_crash_log_is_trimmed_when_oversized(crash_log, monkeypatch):
    monkeypatch.setattr(crash_reporter, "MAX_CRASH_LOG_BYTES", 4096)
    crash_log.write_text("x" * 20000, encoding="utf-8")
    try:
        raise ValueError("late")
    except ValueError:
        crash_reporter.write_crash_record("unhandled exception", *sys.exc_info())

    assert crash_log.stat().st_size < 20000
    text = crash_log.read_text(encoding="utf-8")
    assert "truncated" in text
    assert "ValueError: late" in text, "the new record survived the trim"


def _run_child(tmp_path, body: str) -> subprocess.CompletedProcess:
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parents[1]).replace(chr(92), '/')!r})
            from opencut import crash_reporter
            crash_reporter.OPENCUT_DIR = {str(tmp_path).replace(chr(92), '/')!r}
            crash_reporter.CRASH_LOG = {str(tmp_path / "crash.log").replace(chr(92), '/')!r}
            crash_reporter.install_crash_handlers()
            {body}
            """
        ),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )


def test_native_abort_leaves_a_faulthandler_traceback(tmp_path):
    """The failure no Python hook can see: only an armed faulthandler writes."""
    result = _run_child(tmp_path, "import faulthandler; faulthandler._sigsegv()")
    assert result.returncode != 0, "the child was supposed to die"

    log = tmp_path / "crash.log"
    assert log.exists(), "a native abort left no crash record"
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "faulthandler armed" in text
    assert "Current thread" in text or "Stack (most recent call first)" in text


def test_unhandled_exception_in_a_child_is_recorded(tmp_path):
    result = _run_child(tmp_path, "raise SystemError('fatal in main')")
    assert result.returncode != 0

    text = (tmp_path / "crash.log").read_text(encoding="utf-8", errors="replace")
    assert "unhandled exception" in text
    assert "SystemError: fatal in main" in text
    # The original traceback still reaches stderr; the record is an addition.
    assert "SystemError" in result.stderr
