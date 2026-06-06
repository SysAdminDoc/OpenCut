"""
Tests for scripts/check_doc_sizes.py (RESEARCH_FEATURE_PLAN_2026-05-25 E1).

The script enforces that project documentation size/count claims stay within
±15% of the live filesystem or generated manifests. The tests:
  - exercise the regex against both the bare ``(~N lines)`` form and the
    dated ``(~N lines as of YYYY; was ~M lines through v1.9.x)`` form;
  - confirm ``--check`` passes against the current in-sync tree;
  - confirm drift is detected when we mutate a documented number.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_doc_sizes.py"


def _load_module():
    """Load scripts/check_doc_sizes.py as a module.

    The module uses ``@dataclass`` decorators which require the module to be
    registered in ``sys.modules`` *before* executing — otherwise dataclasses
    crashes with ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    """
    name = "scripts_check_doc_sizes"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestDocSizeRegex(unittest.TestCase):
    """The CEP main.js regex must match both legacy and dated forms."""

    def setUp(self):
        self.mod = _load_module()
        self.cep_re = next(
            t.regex for t in self.mod.TARGETS
            if t.label == "CEP client/main.js lines"
        )

    def test_matches_bare_form(self):
        m = self.cep_re.search("- `client/main.js` (~7730 lines) - Frontend controller.")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "7730")

    def test_matches_dated_form(self):
        text = "- `client/main.js` (~15263 lines as of 2026-05-25; was ~7,730 lines through v1.9.x) - …"
        m = self.cep_re.search(text)
        self.assertIsNotNone(m)
        # Must capture the *current* (first) number, not the historic one.
        self.assertEqual(m.group(1), "15263")

    def test_readme_route_count_regex_matches_feature_overview(self):
        route_re = next(
            t.regex for t in self.mod.TARGETS
            if t.label == "README feature overview API routes"
        )
        m = route_re.search("OpenCut v1.32.0 includes **1,523 API routes**, **8 panel tabs**")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "1,523")


class TestDocSizeCLI(unittest.TestCase):
    def test_check_passes_against_live_tree(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode, 0,
            f"--check must pass in-sync tree. stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_check_fails_on_mutated_doc(self):
        """Copy CLAUDE.md to a temp file, mutate the main.js line count, run --check via importable mod."""
        mod = _load_module()
        target = next(
            t for t in mod.TARGETS if t.label == "CEP client/main.js lines"
        )
        live = target.live()
        # Replace the current number with one 99% smaller — guaranteed >15% drift.
        bogus_claim = max(1, int(live * 0.01))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            (tmp_root / "extension" / "com.opencut.panel" / "client").mkdir(parents=True)
            (tmp_root / "extension" / "com.opencut.uxp").mkdir(parents=True)
            # Live targets point inside ROOT; copy the panel files so live() works.
            shutil.copy2(
                REPO_ROOT / "extension/com.opencut.panel/client/main.js",
                tmp_root / "extension/com.opencut.panel/client/main.js",
            )
            shutil.copy2(
                REPO_ROOT / "extension/com.opencut.uxp/main.js",
                tmp_root / "extension/com.opencut.uxp/main.js",
            )
            # Minimal docs to satisfy the regex with a deliberately wrong number.
            claude = (
                "# OpenCut - CLAUDE.md\n"
                "### Frontend (CEP Panel)\n"
                f"- `extension/com.opencut.panel/client/main.js` (~{bogus_claim} lines) "
                "- Frontend controller stub.\n"
            )
            (tmp_root / "CLAUDE.md").write_text(claude, encoding="utf-8")
            # PROJECT_CONTEXT.md absent — script tolerates missing docs.

            # Run the script with REPO_ROOT replaced via cwd-relative paths.
            script_copy = tmp_root / "check_doc_sizes.py"
            shutil.copy2(SCRIPT, script_copy)
            # Patch the ROOT constant by running with PYTHONPATH and re-importing.
            patched = script_copy.read_text(encoding="utf-8").replace(
                "ROOT = Path(__file__).resolve().parent.parent",
                f"ROOT = Path(r'{tmp_root}')",
                1,
            )
            script_copy.write_text(patched, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script_copy), "--check"],
                cwd=tmp_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 1,
                f"--check should fail on mutated doc. stdout: {result.stdout}\nstderr: {result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
