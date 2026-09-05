"""Which server handles loopback traffic, and does it starve under load.

Issue #8's reporter pasted "WARNING: This is a development server. Do not use it
in a production deployment." into a bug report about an unrelated crash. That
line comes from Werkzeug's ``run_simple`` banner, alongside a duplicate of the
address OpenCut has already printed itself.

The server behind a loopback bind did not change -- it is reachable only from
the machine it runs on, and the panel streams SSE through it. What changed is
that the banner is gone and the choice is reported, so a future bug report can
say which server was serving.
"""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from werkzeug.serving import make_server

import opencut.server as server_module


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


# ---------------------------------------------------------------------------
# The choice
# ---------------------------------------------------------------------------

def test_remote_binds_still_use_waitress():
    assert server_module._should_use_production_wsgi(host="0.0.0.0", debug=False)


def test_loopback_stays_on_the_werkzeug_threaded_server():
    assert not server_module._should_use_production_wsgi(host="127.0.0.1", debug=False)


def test_loopback_serving_reports_itself(monkeypatch):
    """`/system/status` must be able to name the server that is running."""
    served = {}

    class _FakeServer:
        def serve_forever(self):
            served["ran"] = True

        def server_close(self):
            served["closed"] = True

    monkeypatch.setattr(
        "werkzeug.serving.make_server",
        lambda *args, **kwargs: _FakeServer(),
    )
    monkeypatch.setattr(server_module, "_ACTIVE_WSGI_SERVER", "not started")

    server_module._serve_wsgi_app(object(), host="127.0.0.1", port=1, debug=False)

    assert served == {"ran": True, "closed": True}
    reported = server_module.active_wsgi_server()
    assert "werkzeug" in reported and "loopback" in reported


def test_remote_serving_reports_waitress(monkeypatch):
    import types

    calls = []
    waitress = types.ModuleType("waitress")
    waitress.serve = lambda app, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(server_module.sys.modules, "waitress", waitress)
    monkeypatch.setattr(server_module, "_ACTIVE_WSGI_SERVER", "not started")

    server_module._serve_wsgi_app(object(), host="0.0.0.0", port=1, debug=False)

    assert calls, "waitress was not used for a remote bind"
    assert "waitress" in server_module.active_wsgi_server()


def test_the_loopback_path_does_not_print_the_development_warning(monkeypatch, capsys):
    """The banner is what put a scary line into someone's bug report."""
    class _FakeServer:
        def serve_forever(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr("werkzeug.serving.make_server", lambda *a, **k: _FakeServer())
    server_module._serve_wsgi_app(object(), host="127.0.0.1", port=1, debug=False)

    output = capsys.readouterr()
    combined = output.out + output.err
    assert "development server" not in combined
    assert "Serving Flask app" not in combined


# ---------------------------------------------------------------------------
# Concurrency: a stream must not starve ordinary requests
# ---------------------------------------------------------------------------

@pytest.fixture()
def streaming_server():
    """A real threaded Werkzeug server with one slow streaming endpoint."""
    from flask import Flask, Response

    app = Flask(__name__)

    @app.route("/stream")
    def stream():
        def generate():
            for _ in range(20):
                yield "data: tick\n\n"
                time.sleep(0.05)

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/quick")
    def quick():
        return {"ok": True}

    port = _free_port()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_open_streams_do_not_starve_other_requests(streaming_server):
    """Three concurrent SSE readers plus ordinary traffic on the same server.

    The threaded server has to keep answering while streams are held open. If
    it serialised, the quick calls would queue behind a one-second stream.
    """
    import urllib.request

    port = streaming_server
    base = f"http://127.0.0.1:{port}"

    def read_stream():
        with urllib.request.urlopen(f"{base}/stream", timeout=15) as resp:
            return len(resp.read())

    def quick_call():
        started = time.time()
        with urllib.request.urlopen(f"{base}/quick", timeout=15) as resp:
            resp.read()
        return time.time() - started

    with ThreadPoolExecutor(max_workers=8) as pool:
        streams = [pool.submit(read_stream) for _ in range(3)]
        time.sleep(0.15)  # let the streams get established and stay open
        quick = [pool.submit(quick_call) for _ in range(5)]

        durations = [future.result() for future in quick]
        payloads = [future.result() for future in streams]

    assert all(size > 0 for size in payloads), "a stream returned nothing"
    slowest = max(durations)
    assert slowest < 1.0, (
        f"a plain request waited {slowest:.2f}s behind open streams; the server "
        "is serialising requests"
    )
