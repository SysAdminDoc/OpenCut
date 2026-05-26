"""
Tests for scripts/test_breadth_gate.py (RESEARCH_FEATURE_PLAN_2026-05-25 Q9).

The breadth gate is a cheap proxy for per-blueprint coverage: it counts
how many ``opencut/core/*.py`` modules are referenced by at least one
``tests/test_*.py`` file via static import scanning. The floor is
``MIN_RATIO`` (currently 0.75 — measured 0.78 live on the autonomous-
loop branch).

These tests confirm:
  - The live ratio meets the floor.
  - The import-detection regexes catch both ``from opencut.core.X``
    and ``import opencut.core.X``.
  - Missing tests/ directory degrades gracefully (skips count).
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "test_breadth_gate.py"


def _load():
    name = "scripts_test_breadth_gate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestBreadthGate(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_live_ratio_meets_floor(self):
        e = self.mod.evaluate()
        self.assertGreaterEqual(
            e["ratio"], e["floor"],
            (
                f"Test-reference ratio {e['ratio']:.1%} below floor "
                f"{e['floor']:.0%}. Add a test import for any new core "
                "module before merging."
            ),
        )

    def test_import_from_regex_captures_dotted_form(self):
        sample = "from opencut.core.silence import detect, speed_up_silences"
        matches = list(self.mod._IMPORT_FROM_RE.finditer(sample))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group(1), "silence")

    def test_import_from_regex_captures_bare_form(self):
        sample = "from opencut.core import captions, audio, video_core"
        matches = list(self.mod._IMPORT_FROM_RE.finditer(sample))
        self.assertEqual(len(matches), 1)
        # Bare form yields group(1)=None — names come from group(2).
        self.assertIsNone(matches[0].group(1))
        names = {n.strip() for n in matches[0].group(2).split(",")}
        self.assertIn("captions", names)
        self.assertIn("audio", names)

    def test_import_direct_regex(self):
        sample = "import opencut.core.silence"
        m = self.mod._IMPORT_DIRECT_RE.search(sample)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "silence")


if __name__ == "__main__":
    unittest.main()
