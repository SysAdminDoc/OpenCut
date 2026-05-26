"""
Tests for scripts/lint_subprocess_timeouts.py (RESEARCH_FEATURE_PLAN_2026-05-25 E4).

The linter is an AST scanner that asserts every ``subprocess.run`` and
``subprocess.Popen`` call (and the ``_sp`` aliases) under ``opencut/`` is
bounded by a timeout. ``Popen`` is bounded via downstream
``.wait(timeout=N)`` or ``.communicate(timeout=N)`` within the same
function body.

These tests:
  - Confirm the live tree is currently clean.
  - Confirm the linter correctly flags bare ``subprocess.run``.
  - Confirm the linter correctly flags ``Popen`` with no downstream timeout.
  - Confirm allow-listed file-manager spawns (``explorer``, ``open``)
    do not produce hits.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "lint_subprocess_timeouts.py"


def _load_module():
    name = "scripts_lint_subprocess_timeouts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLintSelfChecks(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def _scan_source(self, code: str):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "mod.py").write_text(code, encoding="utf-8")
            self.mod.ROOT = tmp
            self.mod.SCAN_DIRS = (tmp,)
            return self.mod.collect_hits()

    def test_live_tree_is_clean(self):
        # Re-load to reset ROOT.
        sys.modules.pop("scripts_lint_subprocess_timeouts", None)
        mod = _load_module()
        hits = mod.collect_hits()
        self.assertEqual(
            hits, [],
            "Live opencut/ tree must have zero subprocess timeout violations. "
            f"Got: {[(h.path, h.line, h.reason) for h in hits]}",
        )

    def test_bare_subprocess_run_is_flagged(self):
        code = (
            "import subprocess\n"
            "def foo():\n"
            "    subprocess.run(['echo', 'hi'])\n"
        )
        hits = self._scan_source(code)
        self.assertEqual(len(hits), 1)
        self.assertIn("subprocess.run", hits[0].reason)

    def test_popen_without_downstream_timeout_is_flagged(self):
        code = (
            "import subprocess as _sp\n"
            "def bar():\n"
            "    p = _sp.Popen(['sleep', '99'])\n"
            "    p.wait()\n"  # no timeout
        )
        hits = self._scan_source(code)
        self.assertEqual(len(hits), 1)
        self.assertIn("Popen", hits[0].reason)

    def test_popen_with_communicate_timeout_is_clean(self):
        code = (
            "import subprocess as _sp\n"
            "def baz():\n"
            "    p = _sp.Popen(['sleep', '99'])\n"
            "    p.communicate(timeout=10)\n"
        )
        hits = self._scan_source(code)
        self.assertEqual(hits, [])

    def test_file_manager_spawn_is_allowlisted(self):
        code = (
            "import subprocess as _sp\n"
            "def reveal():\n"
            "    _sp.Popen(['explorer', '/select,', 'x'])\n"
            "    _sp.Popen(['open', '-R', 'y'])\n"
            "    _sp.Popen(['xdg-open', 'z'])\n"
        )
        hits = self._scan_source(code)
        self.assertEqual(hits, [], f"unexpected hits on file-manager spawns: {hits}")

    def test_subprocess_run_with_timeout_is_clean(self):
        code = (
            "import subprocess\n"
            "def ok():\n"
            "    subprocess.run(['echo', 'hi'], timeout=5)\n"
        )
        hits = self._scan_source(code)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
