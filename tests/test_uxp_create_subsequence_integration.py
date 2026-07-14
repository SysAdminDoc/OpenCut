"""F254 UXP createSubsequence integration guardrails."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"
PANEL_DIR = REPO_ROOT / "extension" / "com.opencut.panel"
UXP_DIR = REPO_ROOT / "extension" / "com.opencut.uxp"
PANEL_PACKAGE = PANEL_DIR / "package.json"
UXP_ESLINT_CONFIG = UXP_DIR / "eslint.config.mjs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f254_beta_snapshot_still_tracks_create_subsequence_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.85"
    assert "26.3.0-beta.85" in data["tracked_versions"]


def test_f254_uxp_bridge_sets_range_before_creating_subsequence():
    source = _read(MAIN_JS)

    assert "async function createSubsequenceFromRange(payload)" in source
    assert "ppro?.TickTime?.createWithSeconds" in source
    assert "seq.createSetInPointAction" in source
    assert "seq.createSetOutPointAction" in source
    assert "seq.createSubsequence(ignoreTrackTargeting)" in source
    assert "OpenCut set subsequence range" in source
    assert "OpenCut restore sequence range" in source


def test_f254_range_restore_uses_project_transaction():
    source = _read(MAIN_JS)

    helper = source[
        source.index("async function _executeSequenceRangeActions"):
        source.index("async function _readSequenceRange")
    ]
    assert "typeof context.proj.lockedAccess === \"function\"" in helper
    assert "context.proj.lockedAccess(() => {" in helper
    assert "context.proj.executeTransaction" in helper
    assert "seq.createSetInPointAction(inPoint)" in helper
    assert "seq.createSetOutPointAction(outPoint)" in helper
    assert "async " not in helper[helper.index("context.proj.lockedAccess(() => {"):]
    assert "await " not in helper[helper.index("context.proj.lockedAccess(() => {"):]
    assert "_executeProjectActions" not in source
    assert "finally" in source
    assert "rangeVerification" in source
    assert "_sequenceRangeMatches" in source
    assert "Premiere did not restore the original sequence range." in source
    assert "A valid start/end range in seconds is required." in source


def test_f254_adobe_linter_accepts_bridge_and_rejects_unlocked_factory():
    package = json.loads(PANEL_PACKAGE.read_text(encoding="utf-8"))
    dependencies = package["devDependencies"]
    assert dependencies["@adobe/eslint-plugin-premierepro"] == "26.3.0"
    assert dependencies["@adobe/premierepro"] == "26.3.0"
    assert UXP_ESLINT_CONFIG.is_file()

    eslint_name = "eslint.cmd" if os.name == "nt" else "eslint"
    eslint = PANEL_DIR / "node_modules" / ".bin" / eslint_name
    assert eslint.is_file(), "run npm install in extension/com.opencut.panel"

    accepted = subprocess.run(
        [str(eslint), "main.js"],
        cwd=UXP_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr

    fixture_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".js",
            prefix="opencut-unlocked-action-",
            dir=UXP_DIR,
            delete=False,
        ) as fixture:
            fixture.write(
                "const action = sequence.createSetInPointAction(inPoint);\n"
                "project.executeTransaction((compoundAction) => {\n"
                "  compoundAction.addAction(action);\n"
                "}, 'Unsafe action scope');\n"
            )
            fixture_path = Path(fixture.name)
        rejected = subprocess.run(
            [str(eslint), fixture_path.name],
            cwd=UXP_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if fixture_path is not None:
            fixture_path.unlink(missing_ok=True)

    output = rejected.stdout + rejected.stderr
    assert rejected.returncode != 0, output
    assert "@adobe/premierepro/require-action-lock-scope" in output


def test_f254_export_dispatch_passes_subsequence_to_encoder():
    source = _read(MAIN_JS)

    assert "const subsequence = await createSubsequenceFromRange(parsed)" in source
    assert "const exportResult = await exportSubsequenceWithEncoder(subsequence.sequence, parsed)" in source
    assert "sequenceName: subsequence.sequenceName" in source
    assert "ignoreTrackTargeting: subsequence.ignoreTrackTargeting" in source
    assert "createSubsequenceFromRange" in source


def test_f254_release_smoke_runs_subsequence_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_create_subsequence_integration.py" in source
