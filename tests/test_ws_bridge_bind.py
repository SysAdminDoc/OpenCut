"""Starting the WebSocket bridge must not claim success before it binds.

``WebSocketBridge.start()`` set ``_running = True``, spawned a thread and
returned, so ``/ws/start`` answered "running on port 5680" while that thread was
still on its way to an OSError. The bind failure was logged and dropped, which
left ``/ws/status`` reporting a generic stopped state: a port collision looked
exactly like a bridge nobody had started. The CEP panel then dialled a hardcoded
5680 regardless of which port the backend had actually taken.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from opencut.core.ws_bridge import WebSocketBridge

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((HOST, 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


pytest.importorskip("websockets", reason="the bridge needs the websockets package")


@pytest.fixture()
def bridge():
    made: list[WebSocketBridge] = []

    def _make(port: int) -> WebSocketBridge:
        instance = WebSocketBridge(host=HOST, port=port)
        made.append(instance)
        return instance

    yield _make
    for instance in made:
        instance.stop()


# ---------------------------------------------------------------------------
# The happy path reports the port it actually bound
# ---------------------------------------------------------------------------

def test_start_returns_only_after_the_socket_is_bound(bridge):
    instance = bridge(_free_port())
    result = instance.start(timeout=10)

    assert result["bound"] is True
    assert result["error"] is None
    assert instance.is_running is True

    # The port is claimable by a client the instant start() returns; before the
    # fix start() could return while the socket did not exist yet.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        assert probe.connect_ex((HOST, result["port"])) == 0


def test_bind_result_reports_the_real_port_not_the_requested_one(bridge):
    """Port 0 asks the OS to choose, which is the general case of a retry."""
    instance = bridge(0)
    result = instance.start(timeout=10)

    assert result["bound"] is True
    assert result["requested_port"] == 0
    assert result["port"] != 0, "the bound port was never read back off the socket"
    assert 1024 <= result["port"] <= 65535


# ---------------------------------------------------------------------------
# Failures keep their reason
# ---------------------------------------------------------------------------

def test_a_port_collision_fails_the_start_and_keeps_the_reason(bridge):
    """The reported case: something already holds the port."""
    port = _free_port()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind((HOST, port))
    blocker.listen(1)
    try:
        instance = bridge(port)
        result = instance.start(timeout=10)

        assert result["bound"] is False, "start() claimed a bind that could not have happened"
        assert instance.is_running is False
        assert result["error"], "the OSError was dropped, as it was before"
        assert str(port) in result["error"]
        # And it is still there afterwards, which is what /ws/status reads.
        assert instance.last_error == result["error"]
    finally:
        blocker.close()


def test_a_missing_websockets_package_is_reported_not_silently_stopped(bridge, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "websockets" or name.startswith("websockets."):
            raise ImportError("No module named 'websockets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    instance = bridge(_free_port())
    result = instance.start(timeout=10)

    assert result["bound"] is False
    assert "websockets" in (result["error"] or "")


def test_start_does_not_hang_forever_when_the_thread_never_settles(bridge, monkeypatch):
    """A bind that never resolves must time out with a stated reason."""
    instance = bridge(_free_port())
    monkeypatch.setattr(instance, "_run_server", lambda: time.sleep(30))

    started = time.time()
    result = instance.start(timeout=0.5)
    elapsed = time.time() - started

    assert result["bound"] is False
    assert "within" in (result["error"] or "")
    assert elapsed < 5, "start() blocked well past its timeout"


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------

def test_a_bridge_can_be_stopped_and_started_again(bridge):
    instance = bridge(_free_port())
    assert instance.start(timeout=10)["bound"] is True
    instance.stop()
    assert instance.is_running is False

    again = instance.start(timeout=10)
    assert again["bound"] is True
    assert again["error"] is None


def test_a_failed_start_clears_the_previous_error_on_retry(bridge):
    port = _free_port()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind((HOST, port))
    blocker.listen(1)
    instance = bridge(port)
    try:
        assert instance.start(timeout=10)["bound"] is False
        assert instance.last_error
    finally:
        blocker.close()

    # Give the OS a moment to release, then retry on a definitely-free port.
    instance.port = _free_port()
    retry = instance.start(timeout=10)
    assert retry["bound"] is True
    assert retry["error"] is None
    assert instance.last_error is None


# ---------------------------------------------------------------------------
# The panel must use the reported port
# ---------------------------------------------------------------------------

def test_cep_panel_does_not_hardcode_the_bridge_port():
    """The CEP panel dialled 5680 whatever the backend reported."""
    source = (REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "var port = 5680;" not in source, "the panel is back to a hardcoded bridge port"
    assert "_wsBridgePort" in source
    assert "rememberWsBridgePort" in source


def test_cep_panel_records_the_port_from_both_endpoints():
    source = (REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js").read_text(
        encoding="utf-8", errors="replace"
    )
    # Reading it from only one of the two leaves the other path on the default.
    start_block = source.split('api("POST", "/ws/start"', 1)[1][:600]
    status_block = source.split('api("GET", "/ws/status"', 1)[1][:400]
    assert "rememberWsBridgePort" in start_block
    assert "rememberWsBridgePort" in status_block


def test_concurrent_start_calls_do_not_double_bind(bridge):
    """Two panels pressing connect at once must not fight over the socket."""
    instance = bridge(_free_port())
    results = []

    def _start():
        results.append(instance.start(timeout=10))

    threads = [threading.Thread(target=_start) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert len(results) == 3
    bound_ports = {result["port"] for result in results if result["bound"]}
    assert len(bound_ports) <= 1, f"the bridge bound more than one port: {bound_ports}"
    assert instance.is_running is True


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

def test_ws_start_route_reports_failure_instead_of_a_false_success(monkeypatch):
    """`/ws/start` answered running=True before the socket existed."""
    from opencut.routes import system_realtime_routes as routes

    class _FailedBridge:
        port = 5680
        is_running = False
        client_count = 0
        last_error = "Cannot bind 127.0.0.1:5680: address in use"

        def bind_result(self):
            return {"bound": False, "port": 5680, "error": self.last_error}

    monkeypatch.setattr(routes, "_select_ws_bridge_port", lambda preferred: 5680)

    import opencut.core.ws_bridge as ws_bridge

    monkeypatch.setattr(ws_bridge, "get_bridge", lambda: None)
    monkeypatch.setattr(ws_bridge, "init_bridge", lambda **kwargs: _FailedBridge())
    monkeypatch.setattr(ws_bridge, "check_websocket_available", lambda: True)

    from opencut.server import _get_app

    app = _get_app()
    with app.test_client() as client:
        health = client.get("/health")
        token = (health.get_json() or {}).get("csrf_token", "")
        response = client.post("/ws/start", json={}, headers={"X-OpenCut-Token": token})

    assert response.status_code == 503, "a failed bind still answered 2xx"
    body = response.get_json()
    assert body["success"] is False
    assert body["running"] is False
    assert "address in use" in (body.get("error") or "")


def test_ws_status_route_retains_the_bind_error(monkeypatch):
    from opencut.routes import system_realtime_routes as routes  # noqa: F401
    import opencut.core.ws_bridge as ws_bridge

    class _FailedBridge:
        port = 5681
        is_running = False
        client_count = 0
        last_error = "Cannot bind 127.0.0.1:5681: address in use"

        def bind_result(self):
            return {"bound": False, "port": 5681, "error": self.last_error}

    monkeypatch.setattr(ws_bridge, "get_bridge", lambda: _FailedBridge())

    from opencut.server import _get_app

    app = _get_app()
    with app.test_client() as client:
        body = client.get("/ws/status").get_json()

    assert body["running"] is False
    assert "address in use" in (body.get("error") or ""), (
        "a port collision is still indistinguishable from a bridge nobody started"
    )
