"""Safety coverage for OpenCut's PID and port lifecycle helpers."""
from __future__ import annotations


def test_netstat_does_not_kill_an_unverified_port_holder(monkeypatch):
    import opencut.pid as pid

    calls = []

    monkeypatch.setattr(pid.sys, "platform", "win32")
    monkeypatch.setattr(pid, "_is_opencut_on_port", lambda *_args: False)

    def unexpected_subprocess(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unverified listener must not be enumerated or killed")

    monkeypatch.setattr(pid._sp, "run", unexpected_subprocess)

    assert pid._kill_via_netstat("127.0.0.1", 5679) is False
    assert calls == []
