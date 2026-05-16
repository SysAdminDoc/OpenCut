#!/usr/bin/env python3
"""Release smoke matrix runner (F098).

Chains the cheap-to-run release gates into a single command. Each step is
intentionally small and idempotent so we can fail fast — if any of these
trip, the release is not safe to ship.

Steps (in order):

1. ``bootstrap`` — `scripts/bootstrap_check.py --json --metadata-only`
2. ``version-sync`` — `scripts/sync_version.py --check`
3. ``ruff`` — lint the python package
4. ``pytest-fast`` — focused test ids covering release gates
5. ``pip-audit`` — Python dependency advisories (skipped if not installed)
6. ``npm-advisory`` — CEP panel allow-list check
7. ``panel-source`` — CEP panel source tree smoke

Each step records ``status`` (``ok|fail|skipped``), an exit code, a duration
in ms, and a short message. The script exits with code 1 if any non-skipped
step failed.

The orchestrator is deliberately stdlib-only so it can run inside a fresh
environment before `pip install -e .` has happened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO_ROOT / "extension" / "com.opencut.panel"


@dataclass
class StepResult:
    name: str
    status: str  # ok | fail | skipped
    exit_code: int = 0
    duration_ms: int = 0
    message: str = ""
    skipped_reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_line(self) -> str:
        symbol = {"ok": "PASS", "fail": "FAIL", "skipped": "SKIP"}.get(self.status, "????")
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
    "tests/test_feature_registry.py",
    "tests/test_caption_qc.py",
    "tests/test_local_auth.py",
    "tests/test_model_cards.py",
    "tests/test_roadmap_lint.py",
    "tests/test_capability_profile.py",
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
    if not (PANEL_DIR / "scripts" / "check-advisories.mjs").exists():
        return StepResult(
            "npm-advisory",
            "skipped",
            skipped_reason="check-advisories.mjs missing",
        )
    if not (PANEL_DIR / "node_modules").exists():
        return StepResult(
            "npm-advisory",
            "skipped",
            skipped_reason="node_modules absent; run `npm ci` first",
        )
    res = _node_command("scripts/check-advisories.mjs", cwd=PANEL_DIR)
    res.name = "npm-advisory"
    if res.status == "ok":
        res.message = "advisories on allow-list"
    return res


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
    StepDefinition("model-cards", step_model_cards, "Check generated model cards in sync"),
    StepDefinition("roadmap-lint", step_roadmap_lint, "Lint ROADMAP source appendix"),
    StepDefinition("ruff", step_ruff, "Lint the Python package"),
    StepDefinition("pytest-fast", step_pytest_fast, "Run release-gate pytest ids"),
    StepDefinition("pip-audit", step_pip_audit, "Audit requirements.txt"),
    StepDefinition("npm-advisory", step_npm_advisory, "Run npm advisory allow-list gate"),
    StepDefinition("panel-source", step_panel_source, "Smoke the CEP panel source tree"),
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
