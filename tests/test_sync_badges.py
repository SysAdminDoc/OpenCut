"""
Tests for scripts/sync_badges.py (closes RESEARCH_FEATURE_PLAN_2026-05-25 Q4/E8).

The README ships hand-edited "API Routes" and "Tests" badges that historically
drifted from the live route manifest. The badge sync script binds the badges
to ``route_manifest.json::total_routes`` (truth) and a floor-to-100 test count.

These tests cover:
  - Regex correctness against the current README badge syntax.
  - Replacement function returns idempotent output when value already matches.
  - --check exits 0 against the live (in-sync) README, exits 1 if we mutate it.
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
SCRIPT = REPO_ROOT / "scripts" / "sync_badges.py"


def _copy_minimal_sync_tree(tmp_path: Path) -> None:
    tmp_path.mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "opencut" / "_generated").mkdir(parents=True)
    (tmp_path / "opencut" / "core").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "README.md", tmp_path / "README.md")
    shutil.copy2(SCRIPT, tmp_path / "scripts" / "sync_badges.py")
    shutil.copy2(REPO_ROOT / "opencut" / "__init__.py", tmp_path / "opencut" / "__init__.py")
    shutil.copy2(
        REPO_ROOT / "opencut" / "core" / "caption_styles.py",
        tmp_path / "opencut" / "core" / "caption_styles.py",
    )
    shutil.copy2(
        REPO_ROOT / "opencut" / "_generated" / "route_manifest.json",
        tmp_path / "opencut" / "_generated" / "route_manifest.json",
    )


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_badges", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBadgeRegex(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_route_regex_matches_current_format(self):
        sample = "![Routes](https://img.shields.io/badge/API%20Routes-1499-orange)"
        m = self.mod.ROUTE_BADGE_RE.search(sample)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "1499")

    def test_test_regex_matches_current_format(self):
        sample = "![Tests](https://img.shields.io/badge/Tests-7700+-brightgreen)"
        m = self.mod.TEST_BADGE_RE.search(sample)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "7700+")

    def test_replace_badge_no_op_when_in_sync(self):
        text = "![Routes](https://img.shields.io/badge/API%20Routes-1499-orange)"
        out, changed = self.mod._replace_badge(text, self.mod.ROUTE_BADGE_RE, "1499")
        self.assertFalse(changed)
        self.assertEqual(out, text)

    def test_replace_badge_writes_when_drifted(self):
        text = "![Routes](https://img.shields.io/badge/API%20Routes-1234-orange)"
        out, changed = self.mod._replace_badge(text, self.mod.ROUTE_BADGE_RE, "1499")
        self.assertTrue(changed)
        self.assertIn("1499", out)
        self.assertNotIn("1234", out)


class TestBadgeCheckCLI(unittest.TestCase):
    """End-to-end: --check exits 0 in-sync, 1 when README is mutated."""

    def test_check_passes_on_live_readme(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode, 0,
            f"--check should pass against in-sync README. stderr: {result.stderr}",
        )

    def test_check_fails_on_mutated_readme(self):
        """Copy the repo to a temp dir, mutate the badge, confirm --check fails."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "opencut"
            _copy_minimal_sync_tree(tmp_path)
            # tests/ optional but its presence affects test-count math; skip.
            # Mutate the route badge to a wrong number. Derive the current
            # count from the README itself so this stays correct as routes are
            # added (hardcoding it makes the test re-stale on every route bump).
            readme_path = tmp_path / "README.md"
            readme = readme_path.read_text(encoding="utf-8")
            match = _load_module().ROUTE_BADGE_RE.search(readme)
            self.assertIsNotNone(match, "Route badge not found in README (format changed?)")
            current = match.group(2)
            self.assertNotEqual(current, "99", "Test sentinel collides with live route count")
            mutated = readme.replace(
                f"/API%20Routes-{current}-orange",
                "/API%20Routes-99-orange",
            )
            self.assertNotEqual(mutated, readme, "Could not mutate badge (README format changed?)")
            readme_path.write_text(mutated, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(tmp_path / "scripts" / "sync_badges.py"), "--check"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 1,
                f"--check should fail on mutated badge. stderr: {result.stderr}",
            )
            self.assertIn("drift", result.stderr.lower())

    def test_check_fails_on_mutated_product_fact(self):
        """Non-badge product facts are guarded by the same --check path."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "opencut"
            _copy_minimal_sync_tree(tmp_path)

            readme_path = tmp_path / "README.md"
            readme = readme_path.read_text(encoding="utf-8")
            mutated = readme.replace("| 55 Caption Styles |", "| 19 Caption Styles |")
            self.assertNotEqual(mutated, readme, "Could not mutate caption style fact")
            readme_path.write_text(mutated, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(tmp_path / "scripts" / "sync_badges.py"), "--check"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Caption feature style count", result.stderr)


if __name__ == "__main__":
    unittest.main()
