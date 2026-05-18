#!/usr/bin/env python3
"""Release smoke matrix runner (F098).

Chains the cheap-to-run release gates into a single command. Each step is
intentionally small and idempotent so we can fail fast — if any of these
trip, the release is not safe to ship.

Steps (in order):

1. ``bootstrap`` — `scripts/bootstrap_check.py --json --metadata-only`
2. ``version-sync`` — `scripts/sync_version.py --check`
3. ``route-manifest`` — generated route manifest drift check
4. ``api-aliases`` — generated /api alias manifest drift check
5. ``feature-readiness`` — generated route/check readiness drift check
6. ``model-cards`` — generated model/license card drift check
7. ``license-gate`` — model/dependency license allow-list check
8. ``roadmap-lint`` — roadmap citation sanity check
9. ``text-shaping`` — FFmpeg/libass HarfBuzz/FriBidi + renderer capability gate
10. ``ruff`` — lint the python package
11. ``pytest-fast`` — focused test ids covering release gates
12. ``pip-audit`` — Python dependency advisories (skipped if not installed)
13. ``npm-advisory`` — CEP panel allow-list check with machine-readable JSON assertion
14. ``panel-source`` — CEP panel source tree smoke

Each step records ``status`` (``ok|fail|skipped``), an exit code, a duration
in ms, and a short message. The script exits with code 1 if any non-skipped
step failed.

The orchestrator is deliberately stdlib-only so it can run inside a fresh
environment before `pip install -e .` has happened.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO_ROOT / "extension" / "com.opencut.panel"


@dataclass
class StepResult:
    name: str
    status: str  # ok | fail | skipped | warn
    exit_code: int = 0
    duration_ms: int = 0
    message: str = ""
    skipped_reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_line(self) -> str:
        symbol = {
            "ok": "PASS",
            "fail": "FAIL",
            "skipped": "SKIP",
            "warn": "WARN",
        }.get(self.status, "????")
        return f"[{symbol}] {self.name} ({self.duration_ms} ms) — {self.message or self.skipped_reason}"


@dataclass
class StepDefinition:
    name: str
    runner: Callable[[argparse.Namespace], StepResult]
    description: str = ""


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _tail(text: str, lines: int = 12) -> str:
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_bootstrap(args: argparse.Namespace) -> StepResult:
    start = time.time()
    script = REPO_ROOT / "scripts" / "bootstrap_check.py"
    if not script.exists():
        return StepResult(
            "bootstrap",
            "skipped",
            skipped_reason="scripts/bootstrap_check.py missing",
            duration_ms=int((time.time() - start) * 1000),
        )
    result = _run(
        [sys.executable, str(script), "--json", "--metadata-only"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "bootstrap",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="bootstrap_check passed" if status == "ok" else "bootstrap_check failed",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_route_manifest(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.dump_route_manifest", "--check", "--quiet"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "route-manifest",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="manifest in sync" if status == "ok" else "route manifest drift",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_api_aliases(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.dump_api_aliases", "--check"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "api-aliases",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="alias manifest in sync" if status == "ok" else "alias manifest drifted",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_feature_readiness(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.dump_feature_readiness", "--check"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "feature-readiness",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="readiness manifest in sync" if status == "ok" else "readiness manifest drift",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_esbuild_pin(_args: argparse.Namespace) -> StepResult:
    """F131 — every resolved esbuild instance must be >=0.25.0.

    Skips gracefully if Node or the panel's `node_modules` tree is not
    present (the dev VM and many CI matrix legs don't run npm).
    """
    start = time.time()
    node = shutil.which("node")
    panel_root = REPO_ROOT / "extension" / "com.opencut.panel"
    script = panel_root / "scripts" / "check-esbuild-pin.mjs"
    node_modules = panel_root / "node_modules"
    if node is None:
        return StepResult(
            "esbuild-pin",
            "skipped",
            skipped_reason="node executable not on PATH",
            duration_ms=int((time.time() - start) * 1000),
        )
    if not script.is_file():
        return StepResult(
            "esbuild-pin",
            "skipped",
            skipped_reason="check-esbuild-pin.mjs missing",
            duration_ms=int((time.time() - start) * 1000),
        )
    if not node_modules.is_dir():
        return StepResult(
            "esbuild-pin",
            "skipped",
            skipped_reason="panel node_modules not installed",
            duration_ms=int((time.time() - start) * 1000),
        )
    result = _run([node, str(script), "--json"], cwd=panel_root)
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "esbuild-pin",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message=(
            "esbuild >=0.25 across the resolved tree"
            if status == "ok"
            else "esbuild below 0.25 in the resolved tree (F131)"
        ),
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_roadmap_mirror(_args: argparse.Namespace) -> StepResult:
    """F184 — `docs/ROADMAP*.md` must stay short pointer stubs.

    The canonical roadmap is at the repo root. The `docs/` copies used
    to be parallel files that drifted (different content, different
    F-number coverage, stale March-2026 timestamps). F184 collapsed
    them to pointer stubs; this step asserts they never grow back.

    Thresholds: pointer stubs are < 60 lines and must mention the
    canonical ROADMAP path. A grown-back parallel ledger trips here.
    """
    start = time.time()
    docs_dir = REPO_ROOT / "docs"
    stubs = [
        docs_dir / "ROADMAP.md",
        docs_dir / "ROADMAP-COMPLETED.md",
    ]
    violations: List[str] = []
    for stub in stubs:
        if not stub.is_file():
            # Missing stub is fine — the docs/ copy is optional.
            continue
        text = stub.read_text(encoding="utf-8", errors="replace")
        line_count = text.count("\n") + 1
        if line_count > 60:
            violations.append(
                f"{stub.relative_to(REPO_ROOT)} grew to {line_count} lines "
                f"(F184 limit: 60)"
            )
        if "ROADMAP.md" not in text or "Moved" not in text:
            violations.append(
                f"{stub.relative_to(REPO_ROOT)} is missing the F184 pointer "
                f"language (requires both 'Moved' and 'ROADMAP.md')"
            )
    duration = int((time.time() - start) * 1000)
    if violations:
        return StepResult(
            "roadmap-mirror",
            "fail",
            duration_ms=duration,
            message="docs/ROADMAP*.md drifted from F184 pointer-stub shape",
            stderr_tail="\n".join(violations)[-1000:],
        )
    return StepResult(
        "roadmap-mirror",
        "ok",
        duration_ms=duration,
        message="docs/ROADMAP*.md stay short F184 pointer stubs",
    )


def step_mcp_registry(_args: argparse.Namespace) -> StepResult:
    """F147 — committed MCP registry manifest must match live tool catalogue."""
    start = time.time()
    result = _run(
        [
            sys.executable,
            "-m",
            "opencut.tools.dump_mcp_registry_manifest",
            "--check",
        ],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "mcp-registry",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message=(
            "MCP registry manifest in sync"
            if status == "ok"
            else "MCP registry manifest drift"
        ),
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_adobe_premierepro_versions(_args: argparse.Namespace) -> StepResult:
    """F251 — surface @adobe/premierepro npm drift as a notification.

    Drift is *informational* — Adobe shipping a new beta is a signal
    to file F-numbers, not a release blocker. We translate the tool's
    exit code (2 = drift, 0 = sync) into an ``ok``/``warn`` status the
    release-smoke runner reports without failing closed.

    Offline runs (no network on this host) also surface as ``warn``
    rather than ``fail`` so the release gate stays green when there
    is no internet, which is the dev VM's default state.
    """
    start = time.time()
    result = _run(
        [
            sys.executable,
            "-m",
            "opencut.tools.adobe_premierepro_versions",
            "--check",
            "--json",
        ],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    if result.returncode == 0:
        status = "ok"
        message = "no drift"
    elif result.returncode == 2:
        status = "warn"
        message = "drift detected (informational; file F-numbers if Adobe shipped APIs)"
    else:
        status = "warn"
        message = f"could not probe registry (exit {result.returncode})"
    return StepResult(
        "adobe-premierepro-versions",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message=message,
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_model_cards(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.dump_model_cards", "--check"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "model-cards",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="model cards in sync" if status == "ok" else "model card drift",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_license_gate(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.license_gate", "--json"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "license-gate",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="all licenses on allowlist or waived" if status == "ok" else "license gate failed",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_roadmap_lint(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.lint_roadmap_sources"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "roadmap-lint",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="roadmap citations resolve" if status == "ok" else "roadmap citation drift",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_version_sync(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    script = REPO_ROOT / "scripts" / "sync_version.py"
    if not script.exists():
        return StepResult(
            "version-sync",
            "skipped",
            skipped_reason="scripts/sync_version.py missing",
            duration_ms=int((time.time() - start) * 1000),
        )
    result = _run([sys.executable, str(script), "--check"], cwd=REPO_ROOT)
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "version-sync",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="version surfaces aligned" if status == "ok" else "version drift detected",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_text_shaping(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    result = _run(
        [sys.executable, "-m", "opencut.tools.text_shaping_gate", "--json"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return StepResult(
            "text-shaping",
            "fail",
            exit_code=result.returncode,
            duration_ms=duration,
            message=f"text shaping gate did not emit parseable JSON: {exc}",
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )

    summary = payload.get("summary", {})
    status = "ok" if result.returncode == 0 and payload.get("status") == "ok" else "fail"
    warning_count = int(summary.get("warnings") or 0)
    message = (
        f"hard shaping gates passed ({warning_count} advisory warnings)"
        if status == "ok"
        else "text shaping gate failed"
    )
    return StepResult(
        "text-shaping",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message=message,
        stdout_tail=_tail(json.dumps(payload, indent=2)),
        stderr_tail=_tail(result.stderr),
    )


def step_ruff(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    if shutil.which("ruff") is None:
        return StepResult(
            "ruff",
            "skipped",
            skipped_reason="ruff not installed",
            duration_ms=int((time.time() - start) * 1000),
        )
    result = _run(
        ["ruff", "check", "opencut/", "--select", "E,F,I", "--ignore", "E501,E402"],
        cwd=REPO_ROOT,
    )
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "ruff",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="lint clean" if status == "ok" else "ruff reported findings",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


# Release-gate test ids that exercise the most fragile invariants. Keep the
# list small enough to finish in seconds; the full suite still runs in CI.
RELEASE_GATE_TESTS: List[str] = [
    "tests/test_bootstrap_check.py",
    "tests/test_node_advisories.py",
    "tests/test_seed_github_issues.py",
    "tests/test_route_manifest.py",
    "tests/test_route_collisions.py",
    "tests/test_openapi_contract.py",
    "tests/test_api_aliases.py",
    "tests/test_feature_readiness_generator.py",
    "tests/test_feature_registry.py",
    "tests/test_mcp_server.py",
    "tests/test_macos_notarization.py",
    "tests/test_release_sbom.py",
    "tests/test_sbom_completeness.py",
    "tests/test_ffmpeg_installer_manifest.py",
    "tests/test_text_shaping_gate.py",
    "tests/test_srt_encoding.py",
    "tests/test_caption_language_confidence.py",
    "tests/test_captions_translate_srt.py",
    "tests/test_uxp_macos_http.py",
    "tests/test_adobe_premierepro_versions.py",
    "tests/test_mcp_registry_manifest.py",
    "tests/test_mcp_sdk_pin.py",
    "tests/test_otio_aaf_adapter_pin.py",
    "tests/test_audioop_shim.py",
    "tests/test_ffmpeg_filter_regression.py",
    "tests/test_roadmap_mirror.py",
    "tests/test_eval_datasets.py",
    "tests/test_download_eval_dataset.py",
    "tests/test_installer_policy.py",
    "tests/test_inno_installer_smoke.py",
    "tests/test_launcher_scripts.py",
    "tests/test_uxp_backend_client_contract.py",
    "tests/test_esbuild_pin.py",
    "tests/test_caption_qc.py",
    "tests/test_caption_reading_profiles.py",
    "tests/test_caption_display_settings.py",
    "tests/test_loudness_standards.py",
    "tests/test_local_auth.py",
    "tests/test_model_cards.py",
    "tests/test_roadmap_lint.py",
    "tests/test_capability_profile.py",
    "tests/test_marker_import.py",
    "tests/test_review_bundle.py",
    "tests/test_c2pa_sidecar.py",
    "tests/test_plugin_manifest.py",
    "tests/test_marker_metadata.py",
    "tests/test_ai_eval_harness.py",
    "tests/test_fcp_transitions.py",
    "tests/test_ocio_validate.py",
    "tests/test_windows_arm64_doc.py",
    "tests/test_project_health.py",
    "tests/test_crash_packet.py",
    "tests/test_job_diagnostics.py",
    "tests/test_license_gate.py",
    "tests/test_hardening.py::test_uxp_engine_registry_escapes_dynamic_attribute_values",
    "tests/test_hardening.py::test_uxp_fetch_wrapper_clears_backend_timeout_timers",
    "tests/test_config_and_userdata.py::test_server_main_rejects_remote_bind_without_opt_in",
    "tests/test_config_and_userdata.py::test_server_main_allows_remote_bind_with_explicit_opt_in",
    "tests/test_platform_ux.py::TestWebUIUpload::test_upload_sanitizes_windows_paths_and_markup",
    "tests/test_platform_ux.py::TestWebUIOperationCatalog::test_serve_web_ui_uses_text_nodes_for_dynamic_content",
]


def step_pytest_fast(args: argparse.Namespace) -> StepResult:
    start = time.time()
    if shutil.which("pytest") is None and not (REPO_ROOT / "pyproject.toml").exists():
        return StepResult(
            "pytest-fast",
            "skipped",
            skipped_reason="pytest unavailable",
            duration_ms=int((time.time() - start) * 1000),
        )
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", *RELEASE_GATE_TESTS]
    if args.pytest_extra:
        cmd.extend(args.pytest_extra)
    result = _run(cmd, cwd=REPO_ROOT)
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "pytest-fast",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message=(
            f"{len(RELEASE_GATE_TESTS)} gate tests passed"
            if status == "ok"
            else "release-gate tests failed"
        ),
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_pip_audit(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    # Prefer the module entrypoint so we don't depend on PATH ordering.
    cmd = [sys.executable, "-m", "pip_audit", "-r", "requirements.txt", "--format", "json"]
    result = _run(cmd, cwd=REPO_ROOT)
    duration = int((time.time() - start) * 1000)
    if "No module named" in (result.stderr or ""):
        return StepResult(
            "pip-audit",
            "skipped",
            skipped_reason="pip-audit not installed",
            duration_ms=duration,
        )
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "pip-audit",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="no advisories" if status == "ok" else "pip-audit reported advisories",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def _node_command(*node_args: str, cwd: Path) -> StepResult:
    start = time.time()
    candidates = ["node", "node.exe"]
    node_bin = next((shutil.which(name) for name in candidates if shutil.which(name)), None)
    if not node_bin:
        return StepResult(
            "panel-script",
            "skipped",
            skipped_reason="node not on PATH",
            duration_ms=int((time.time() - start) * 1000),
        )
    result = _run([node_bin, *node_args], cwd=cwd)
    duration = int((time.time() - start) * 1000)
    status = "ok" if result.returncode == 0 else "fail"
    return StepResult(
        "panel-script",
        status,
        exit_code=result.returncode,
        duration_ms=duration,
        message="ok" if status == "ok" else "node script failed",
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def step_npm_advisory(_args: argparse.Namespace) -> StepResult:
    start = time.time()
    script = PANEL_DIR / "scripts" / "check-advisories.mjs"
    if not script.exists():
        return StepResult(
            "npm-advisory",
            "skipped",
            skipped_reason="check-advisories.mjs missing",
            duration_ms=int((time.time() - start) * 1000),
        )
    if not (PANEL_DIR / "node_modules").exists():
        return StepResult(
            "npm-advisory",
            "skipped",
            skipped_reason="node_modules absent; run `npm ci` first",
            duration_ms=int((time.time() - start) * 1000),
        )
    node_bin = next((shutil.which(name) for name in ("node", "node.exe") if shutil.which(name)), None)
    if not node_bin:
        return StepResult(
            "npm-advisory",
            "skipped",
            skipped_reason="node not on PATH",
            duration_ms=int((time.time() - start) * 1000),
        )

    result = _run([node_bin, "scripts/check-advisories.mjs", "--json"], cwd=PANEL_DIR)
    duration = int((time.time() - start) * 1000)

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return StepResult(
            "npm-advisory",
            "fail",
            exit_code=result.returncode,
            duration_ms=duration,
            message=f"advisory checker did not emit parseable JSON: {exc}",
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )

    summary = payload.get("summary", {})
    allowed = int(summary.get("allowed") or 0)
    unwaived = int(summary.get("unwaived") or 0)
    status = payload.get("status")
    ok = result.returncode == 0 and status == "ok" and unwaived == 0
    message = (
        f"advisories on allow-list ({allowed} allowed)"
        if ok and allowed
        else "no advisories reported by npm audit"
        if ok
        else f"unwaived advisories detected ({unwaived})"
    )
    return StepResult(
        "npm-advisory",
        "ok" if ok else "fail",
        exit_code=result.returncode,
        duration_ms=duration,
        message=message,
        stdout_tail=_tail(json.dumps(payload, indent=2)),
        stderr_tail=_tail(result.stderr),
    )


def step_panel_source(_args: argparse.Namespace) -> StepResult:
    if not (PANEL_DIR / "scripts" / "verify-build.mjs").exists():
        return StepResult(
            "panel-source",
            "skipped",
            skipped_reason="verify-build.mjs missing",
        )
    res = _node_command("scripts/verify-build.mjs", cwd=PANEL_DIR)
    res.name = "panel-source"
    if res.status == "ok":
        res.message = "panel source tree intact"
    return res


STEPS: List[StepDefinition] = [
    StepDefinition("bootstrap", step_bootstrap, "Run scripts/bootstrap_check.py"),
    StepDefinition("version-sync", step_version_sync, "Check version surfaces"),
    StepDefinition("route-manifest", step_route_manifest, "Check route manifest is in sync"),
    StepDefinition("api-aliases", step_api_aliases, "Check /api alias manifest is in sync"),
    StepDefinition("feature-readiness", step_feature_readiness, "Check route/check readiness manifest is in sync"),
    StepDefinition("mcp-registry", step_mcp_registry, "Check MCP server registry manifest is in sync (F147)"),
    StepDefinition("model-cards", step_model_cards, "Check generated model cards in sync"),
    StepDefinition("license-gate", step_license_gate, "Run the license allowlist gate over model cards"),
    StepDefinition("roadmap-lint", step_roadmap_lint, "Lint ROADMAP source appendix"),
    StepDefinition("roadmap-mirror", step_roadmap_mirror, "Verify docs/ROADMAP*.md stay F184 pointer stubs"),
    StepDefinition("text-shaping", step_text_shaping, "Check FFmpeg/libass and renderer text shaping support"),
    StepDefinition("ruff", step_ruff, "Lint the Python package"),
    StepDefinition("pytest-fast", step_pytest_fast, "Run release-gate pytest ids"),
    StepDefinition("pip-audit", step_pip_audit, "Audit requirements.txt"),
    StepDefinition("npm-advisory", step_npm_advisory, "Run npm advisory allow-list gate"),
    StepDefinition("esbuild-pin", step_esbuild_pin, "Verify resolved esbuild >= 0.25 (F131)"),
    StepDefinition("panel-source", step_panel_source, "Smoke the CEP panel source tree"),
    StepDefinition(
        "adobe-premierepro-versions",
        step_adobe_premierepro_versions,
        "Notify on @adobe/premierepro npm registry drift (informational, F251)",
    ),
]


# ---------------------------------------------------------------------------
# Public API used by tests + the CLI entrypoint
# ---------------------------------------------------------------------------


def run_release_smoke(
    args: Optional[argparse.Namespace] = None,
    *,
    only: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
) -> List[StepResult]:
    ns = args or argparse.Namespace(pytest_extra=[])
    if not hasattr(ns, "pytest_extra"):
        ns.pytest_extra = []

    selected: List[StepDefinition] = []
    only_set = {name for name in (only or [])}
    skip_set = {name for name in (skip or [])}
    for step in STEPS:
        if only_set and step.name not in only_set:
            continue
        if step.name in skip_set:
            continue
        selected.append(step)

    results: List[StepResult] = []
    for step in selected:
        result = step.runner(ns)
        results.append(result)
    return results


def overall_status(results: List[StepResult]) -> str:
    if any(r.status == "fail" for r in results):
        return "fail"
    if not results:
        return "skipped"
    if all(r.status == "skipped" for r in results):
        return "skipped"
    return "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    parser.add_argument("--only", action="append", default=[], help="run only the named step (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="skip the named step (repeatable)")
    parser.add_argument(
        "--pytest-extra",
        nargs="*",
        default=[],
        help="extra args appended to the pytest-fast invocation",
    )
    parser.add_argument("--list", action="store_true", help="list available steps and exit")
    args = parser.parse_args(argv)

    if args.list:
        for step in STEPS:
            print(f"{step.name:<14} {step.description}")
        return 0

    results = run_release_smoke(args, only=args.only or None, skip=args.skip or None)
    status = overall_status(results)

    if args.json:
        payload = {"status": status, "steps": [asdict(r) for r in results]}
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            print(r.as_line())
            if r.status == "fail" and r.stderr_tail:
                for line in r.stderr_tail.splitlines():
                    print(f"        {line}")
        print(f"\nresult: {status.upper()}")

    return 0 if status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
