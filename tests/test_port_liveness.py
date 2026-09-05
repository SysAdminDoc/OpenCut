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
    """A TIME_WAIT remnant from a killed server stays reusable."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, free_port))
    server.listen(1)
    client = socket.create_connection((HOST, free_port), timeout=2)
    accepted, _ = server.accept()
    accepted.close()
    client.close()
    server.close()

    assert _check_port(HOST, free_port) is True
