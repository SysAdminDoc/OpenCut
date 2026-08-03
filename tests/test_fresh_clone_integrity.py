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


# A markdown link is not the only way to point a reader at a file: the docs
# refer to paths in backticks and in quotes far more often than in links, and
# the original check matched only `[text](target)`.
_PATH_REF_RE = re.compile(
    r"[`\"']([A-Za-z0-9_.][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]{1,6})[`\"']"
)

# Build outputs the docs legitimately name as *products* of a documented
# command. They are absent from a clone by design, not by oversight.
_GENERATED_PREFIXES = (
    "ffmpeg/",
    "dist/",
    "build/",
    "node_modules/",
    "installer/publish/",
    "installer/dist/",
)


def _is_generated(target: str) -> bool:
    return target.startswith(_GENERATED_PREFIXES) or "/client/dist/" in target


def test_tracked_docs_do_not_name_untracked_repo_files():
    """Backticked and quoted path references must survive a clone too."""
    tracked = _tracked_files()
    offenders: list[str] = []

    for rel in sorted(f for f in tracked if f.endswith(".md")):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for match in _PATH_REF_RE.finditer(source):
            target = match.group(1).replace("\\", "/").removeprefix("./")
            if "/" not in target or _is_generated(target):
                # A bare filename is usually prose ("edit config.json"), not a
                # repository path; only rooted references are checkable.
                continue
            candidate = REPO_ROOT / target
            # Only an existing-but-untracked file is a clone hazard. A path that
            # does not exist at all is either prose or a separate bug class.
            if candidate.is_file() and target not in tracked:
                offenders.append(f"{rel} -> {target}")

    assert offenders == [], (
        "tracked docs name files that exist locally but are absent from a clone: "
        f"{sorted(set(offenders))}"
    )


_ASSIGNMENT_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)\s*=\s*((?:REPO_ROOT|[A-Z][A-Z0-9_]*)(?:\s*/\s*\"[^\"]+\")+)\s*$",
    re.MULTILINE,
)
_CHAIN_RE = re.compile(r"(REPO_ROOT|[A-Z][A-Z0-9_]*)((?:\s*/\s*\"[^\"]+\")+)")
_SEGMENT_RE = re.compile(r"\"([^\"]+)\"")


def _repo_paths_referenced_by(source: str) -> set[Path]:
    """Resolve ``REPO_ROOT / "a" / "b"`` chains, including via named constants."""
    names: dict[str, Path] = {"REPO_ROOT": REPO_ROOT}
    # Constants may be defined in terms of earlier constants; a few passes
    # settle the chain without needing a real interpreter.
    for _ in range(4):
        for name, expr in _ASSIGNMENT_RE.findall(source):
            head = _CHAIN_RE.match(expr)
            if head is None:
                continue
            base = names.get(head.group(1))
            if base is None:
                continue
            names[name] = base.joinpath(*_SEGMENT_RE.findall(head.group(2)))

    referenced = {path for name, path in names.items() if name != "REPO_ROOT"}
    for head, tail in _CHAIN_RE.findall(source):
        base = names.get(head)
        if base is not None:
            referenced.add(base.joinpath(*_SEGMENT_RE.findall(tail)))
    return referenced


def test_the_test_suite_only_reads_files_a_clone_actually_has():
    """`pytest` on a fresh clone must not fail on maintainer-only files.

    Nine modules used to read `docs/*.md` files that `.gitignore` excluded, so
    the advertised green baseline was reproducible only on the maintainer's
    machine. A path that exists here but is untracked is exactly that bug; a
    path that exists nowhere is an "assert this stays deleted" reference and is
    left alone.
    """
    tracked = _tracked_files()
    offenders: list[str] = []

    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        for target in sorted(_repo_paths_referenced_by(source)):
            try:
                rel_target = target.resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            if target.is_dir():
                continue
            if not target.exists():
                continue
            if rel_target in tracked:
                continue
            offenders.append(f"{path.name} -> {rel_target}")

    assert offenders == [], (
        "tests read files that exist locally but are absent from a clone: "
        f"{sorted(set(offenders))}"
    )


def test_clone_hazard_scanner_still_resolves_real_paths():
    """Guard against a regex that silently matches nothing."""
    resolved = _repo_paths_referenced_by(
        (REPO_ROOT / "tests" / "test_installer_policy.py").read_text(
            encoding="utf-8", errors="replace"
        )
    )
    assert any(p.name == "OpenCut.iss" for p in resolved), resolved


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
