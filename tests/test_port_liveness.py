"""A live listener must beat every socket-option tolerance.

``_check_port`` used to set ``SO_REUSEADDR`` before binding, with a comment
about TIME_WAIT that is only true on POSIX. On Windows ``SO_REUSEADDR`` also
permits binding over a socket another process is *actively* holding, so the
probe reported a busy port as free, OpenCut started a second server on it, the
second overwrote the PID file, and the panel saw intermittent "service
unavailable" (issue #8: pid 26164 and pid 10352 both claimed port 5679).

The regression test here is deliberately a real listening socket rather than a
mock: the defect lived in the kernel's interpretation of a socket option, which
a mocked socket cannot reproduce.
"""

from __future__ import annotations

import socket

import pytest

from opencut import pid as pid_module
from opencut.pid import _bind_probe, _check_port, _port_has_listener

HOST = "127.0.0.1"


@pytest.fixture()
def listening_port():
    """Bind and listen on an ephemeral port, yielding it while it stays live."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, 0))
    server.listen(1)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.fixture()
def free_port():
    """Return a port number that nothing is listening on."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((HOST, 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def test_check_port_rejects_a_live_listener(listening_port):
    """The regression: this returned True on Windows before the fix."""
    assert _check_port(HOST, listening_port) is False


def test_check_port_accepts_a_genuinely_free_port(free_port):
    assert _check_port(HOST, free_port) is True


def test_port_has_listener_distinguishes_live_from_free(listening_port, free_port):
    assert _port_has_listener(HOST, listening_port) is True
    assert _port_has_listener(HOST, free_port) is False


def test_reuseaddr_bind_probe_alone_is_not_a_liveness_test(listening_port):
    """Document why the old implementation was wrong, and keep it wrong.

    On Windows the tolerant bind probe succeeds against a live listener; on
    POSIX it does not. Either way it must never be the sole basis for deciding
    a port is free, which is what this asserts by checking ``_check_port``
    disagrees with it whenever it is permissive.
    """
    tolerant = _bind_probe(HOST, listening_port, tolerate_time_wait=True)
    if tolerant:
        assert _check_port(HOST, listening_port) is False, (
            "the tolerant bind probe accepted a live listener and _check_port "
            "agreed with it -- the issue #8 defect is back"
        )


def test_wait_for_port_does_not_report_a_live_listener_as_free(listening_port):
    assert pid_module._wait_for_port(HOST, listening_port, timeout=0.5) is False


def test_check_port_tolerates_a_recently_closed_connection(free_port):
    """A TIME_WAIT remnant from a killed server stays reusable.

    Asserting only that ``_check_port`` returns True would pass whether or not
    a TIME_WAIT socket existed, so it could not detect the tolerance
    regressing. Drive the two probes directly and require that the tolerant one
    is what rescues the port when the exclusive one refuses it.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, free_port))
    server.listen(1)
    client = socket.create_connection((HOST, free_port), timeout=2)
    accepted, _ = server.accept()
    accepted.close()
    client.close()
    server.close()

    assert _port_has_listener(HOST, free_port) is False, "nothing should still be listening"
    exclusive = _bind_probe(HOST, free_port, tolerate_time_wait=False)
    tolerant = _bind_probe(HOST, free_port, tolerate_time_wait=True)
    assert tolerant is True, "a TIME_WAIT remnant must stay reusable"
    if not exclusive:
        # The interesting case: the strict probe refused and only the tolerant
        # one succeeded, which is exactly the fallback _check_port relies on.
        assert _check_port(HOST, free_port) is True
    assert _check_port(HOST, free_port) is True


def test_a_failed_kill_leaves_the_pid_file_intact(tmp_path, monkeypatch):
    """The PID file must survive a kill that did not work.

    ``_nuke_old_servers`` removed it right after ``_kill_via_pid`` regardless of
    whether the process died. When the kill failed, the caller then reported
    "an unknown PID" for the live server it was trying to name, and nothing
    could find that server afterwards.
    """
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("4242\n5679\n", encoding="utf-8")
    monkeypatch.setattr(pid_module, "PID_FILE", str(pid_file))

    monkeypatch.setattr(pid_module, "_kill_via_shutdown_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(pid_module, "_kill_via_pid", lambda *a, **k: False)
    monkeypatch.setattr(pid_module, "_kill_via_netstat", lambda *a, **k: False)
    # The port never frees: every strategy fails.
    monkeypatch.setattr(pid_module, "_wait_for_port", lambda *a, **k: False)
    monkeypatch.setattr(pid_module, "_check_port", lambda *a, **k: False)

    assert pid_module._nuke_old_servers(HOST, 5679) is False
    assert pid_file.exists(), "a failed kill deleted the live server's PID file"
    assert pid_module._read_pid() == (4242, 5679)


def test_a_successful_kill_still_clears_the_pid_file(tmp_path, monkeypatch):
    """Positive control: the cleanup must still happen when the kill works."""
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("4242\n5679\n", encoding="utf-8")
    monkeypatch.setattr(pid_module, "PID_FILE", str(pid_file))

    monkeypatch.setattr(pid_module, "_kill_via_shutdown_endpoint", lambda *a, **k: False)
    monkeypatch.setattr(pid_module, "_kill_via_pid", lambda *a, **k: True)

    calls = {"n": 0}

    def _wait(*_args, **_kwargs):
        calls["n"] += 1
        return calls["n"] > 1  # first call (after shutdown endpoint) fails

    monkeypatch.setattr(pid_module, "_wait_for_port", _wait)

    assert pid_module._nuke_old_servers(HOST, 5679) is True
    assert not pid_file.exists()
