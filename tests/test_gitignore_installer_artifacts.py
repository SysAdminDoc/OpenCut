"""
Guard test for RESEARCH_FEATURE_PLAN_2026-05-25 E2.

The .NET WPF installer used to track its ``bin/`` and ``obj/`` build
output in git (951 files of DLLs, PDBs, BAML, generated C#, NuGet caches,
etc). The cleanup commit moved them to .gitignore and ``git rm --cached``
'd them. This test guards against accidental re-introduction by:

  1. Asserting the four patterns are present in the repo's .gitignore.
  2. Asserting ``git ls-files`` produces zero matches under the ignored
     directories. Skipped when running outside a git work-tree (e.g. on
     a source tarball).
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = REPO_ROOT / ".gitignore"


class TestInstallerGitignore(unittest.TestCase):
    def test_gitignore_blocks_installer_bin_and_obj(self):
        text = GITIGNORE.read_text(encoding="utf-8")
        for pattern in (
            "installer/src/*/obj/",
            "installer/src/*/bin/",
            "installer/tests/*/obj/",
            "installer/tests/*/bin/",
        ):
            self.assertIn(
                pattern, text,
                f".gitignore must list {pattern!r} (E2 guardrail)",
            )

    def test_no_installer_build_artifacts_in_git_ls_files(self):
        git = shutil.which("git")
        if not git:
            self.skipTest("git executable not on PATH")
        try:
            result = subprocess.run(
                [git, "ls-files"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            self.skipTest("git invocation failed")
        if result.returncode != 0:
            self.skipTest("not a git work-tree")
        offenders = [
            line for line in result.stdout.splitlines()
            if "/bin/" in line or "/obj/" in line
            if line.startswith("installer/src/") or line.startswith("installer/tests/")
        ]
        self.assertEqual(
            offenders, [],
            "installer/(src|tests)/*/(bin|obj)/ files must not be tracked. "
            "Re-run: git rm --cached -r installer/src/*/bin installer/src/*/obj",
        )


if __name__ == "__main__":
    unittest.main()
