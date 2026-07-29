"""A clean clone must be able to follow its own docs and build entry points.

Tracked entry points had drifted from the maintained toolchain: ``BUILD.bat``
invoked a deprecated builder that hardcoded version 0.6.5 and produced an Inno
artifact, and tracked docs linked to files that are deliberately untracked, so
someone cloning the repository could not reproduce the documented topology.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_build_entry_point_uses_the_maintained_builder():
    build_bat = _read("BUILD.bat")
    assert "installer\\InstallerBuilder.ps1" in build_bat
    # The deprecated root builder hardcoded a stale version and built Inno.
    assert not (REPO_ROOT / "InstallerBuilder.ps1").exists()
    assert "0.6.5" not in build_bat


def test_maintained_builder_derives_the_current_artifact_name():
    builder = _read("installer/InstallerBuilder.ps1")
    # Version comes from the single source of truth, not a literal.
    assert "__version__" in builder
    assert 'OpenCut-Setup-$Version.exe' in builder


def test_build_entry_point_propagates_failure():
    """A build wrapper that swallows the exit code hides broken releases."""
    assert "exit /b %ERRORLEVEL%" in _read("BUILD.bat")


def test_tracked_docs_do_not_link_to_untracked_files():
    tracked = _tracked_files()
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    offenders: list[str] = []

    for rel in sorted(f for f in tracked if f.endswith(".md")):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        base = (REPO_ROOT / rel).parent
        for match in link_re.finditer(source):
            target = match.group(1).split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (base / target).resolve()
            try:
                rel_target = resolved.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            # Directories are fine as long as something under them is tracked.
            if resolved.is_dir():
                if any(t.startswith(rel_target + "/") for t in tracked):
                    continue
            if rel_target not in tracked:
                offenders.append(f"{rel} -> {target}")

    assert offenders == [], f"tracked docs link to files absent from a clone: {offenders}"


def test_pyinstaller_hidden_imports_are_source_derived():
    spec = _read("opencut_server.spec")
    # Derived from the _try_import call sites rather than hand-maintained, so
    # a new optional backend does not silently miss the frozen build.
    assert "_discover_lazy_imports" in spec
    assert "_try_import" in spec
    assert "collect_submodules('opencut')" in spec
    # The old list is gone; these were maintained by hand and had drifted.
    assert "'transnetv2'," not in spec
    assert "'resemble_enhance'," not in spec


def test_pyinstaller_spec_discovery_finds_real_backends():
    """Guard against a regex that silently matches nothing."""
    pattern = re.compile("_try_import\\(\\s*[\"']([A-Za-z0-9_.]+)[\"']")
    found: set[str] = set()
    for path in (REPO_ROOT / "opencut").rglob("*.py"):
        found.update(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))

    assert len(found) > 50, "lazy-import discovery collapsed"
    # Specific names the hand-maintained list had wrong.
    assert "transnetv2_pytorch" in found
    assert "auto_editor" in found


def test_installer_policy_does_not_gate_releases_on_signing():
    """OpenCut ships unsigned by policy; signing must not read as a gate."""
    policy = _read("docs/INSTALLER_POLICY.md")
    lowered = policy.lower()
    assert "signed wpf release" not in lowered
    assert "signed-release verification" not in lowered
    assert "signing cert expiry" not in lowered
    assert "unsigned" in lowered
