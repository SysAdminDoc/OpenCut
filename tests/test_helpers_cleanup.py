"""Regression tests for lazy deferred-cleanup worker startup."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def _run_helper_probe(code: str):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_helpers_import_does_not_start_temp_cleanup_thread():
    data = _run_helper_probe(
        """
        import json
        import threading

        before = [thread.name for thread in threading.enumerate()]
        import opencut.helpers  # noqa: F401
        after = [thread.name for thread in threading.enumerate()]
        print(json.dumps({"before": before, "after": after}))
        """
    )

    assert "opencut-temp-cleanup" not in data["before"]
    assert "opencut-temp-cleanup" not in data["after"]


def test_temp_cleanup_thread_starts_on_first_schedule(tmp_path):
    target = tmp_path / "scheduled.tmp"
    data = _run_helper_probe(
        f"""
        import json
        import threading

        from opencut import helpers

        target = {str(target)!r}
        open(target, "w", encoding="utf-8").write("x")
        helpers._schedule_temp_cleanup(target, delay=60.0)
        names = [thread.name for thread in threading.enumerate()]
        print(json.dumps({{"thread_names": names}}))
        """
    )

    assert "opencut-temp-cleanup" in data["thread_names"]
