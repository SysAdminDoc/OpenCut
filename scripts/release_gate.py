#!/usr/bin/env python3
"""Fail-closed local release verification and promotion receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = REPO_ROOT / "build" / "release-receipt.json"
RECEIPT_SCHEMA_VERSION = 1
DEFAULT_MAX_AGE_SECONDS = 2 * 60 * 60
LOCAL_STATE_PATHS = frozenset({"ROADMAP.md", "RESEARCH.md", "Roadmap_Blocked.md"})
REQUIRED_STEPS = frozenset(
    {
        "bootstrap",
        "version-sync",
        "generated-docs",
        "route-manifest",
        "api-aliases",
        "feature-readiness",
        "mcp-registry",
        "model-cards",
        "license-gate",
        "release-lock",
        "ffmpeg-provenance",
        "dependency-matrix",
        "text-shaping",
        "caption-unicode",
        "contrast-audit",
        "ruff",
        "pytest-fast",
        "pip-audit",
        "panel-unit",
        "panel-rendered",
        "npm-advisory",
        "esbuild-pin",
        "panel-source",
    }
)


class ReleaseGateError(RuntimeError):
    """Raised when a release action is not authorized by fresh evidence."""


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _git(*args: str) -> str:
    result = _run(["git", *args], timeout=60)
    if result.returncode != 0:
        raise ReleaseGateError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _parse_status_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        path = raw.strip('"').replace("\\", "/")
        if path and path not in LOCAL_STATE_PATHS:
            paths.append(path)
    return sorted(set(paths))


def _status_paths() -> list[str]:
    result = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], timeout=60)
    if result.returncode != 0:
        raise ReleaseGateError(result.stderr.strip() or "git status failed")
    return _parse_status_paths(result.stdout)


def current_source_state() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty_paths": _status_paths(),
    }


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ReleaseGateError("release receipt has an invalid generated_at timestamp") from exc
    if parsed.tzinfo is None:
        raise ReleaseGateError("release receipt timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseGateError(f"release receipt is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseGateError(f"release receipt is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseGateError("release receipt root must be an object")
    return payload


def validate_receipt(
    path: Path,
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: datetime | None = None,
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _read_receipt(path)
    if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ReleaseGateError("release receipt schema is unsupported")
    if payload.get("status") != "ok" or payload.get("strict") is not True:
        raise ReleaseGateError("release receipt does not record a successful strict gate")

    generated_at = _parse_utc(str(payload.get("generated_at") or ""))
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_seconds = (clock - generated_at).total_seconds()
    if age_seconds < -60 or age_seconds > max_age_seconds:
        raise ReleaseGateError(
            f"release receipt is stale ({max(0, int(age_seconds))} seconds old; maximum is {max_age_seconds})"
        )

    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ReleaseGateError("release receipt has no step results")
    by_name = {str(step.get("name")): step for step in steps if isinstance(step, dict) and step.get("name")}
    missing = sorted(REQUIRED_STEPS - by_name.keys())
    skipped = sorted(name for name in REQUIRED_STEPS if name in by_name and by_name[name].get("status") != "ok")
    if missing:
        raise ReleaseGateError("release receipt is missing required steps: " + ", ".join(missing))
    if skipped:
        raise ReleaseGateError("required release steps did not pass: " + ", ".join(skipped))

    current = source_state or current_source_state()
    recorded = payload.get("source") or {}
    if current.get("dirty_paths"):
        raise ReleaseGateError(
            "release-sensitive worktree changes invalidate the receipt: " + ", ".join(current["dirty_paths"])
        )
    if current.get("branch") != "main" or recorded.get("branch") != "main":
        raise ReleaseGateError("release receipts are valid only for the main branch")
    if recorded.get("commit") != current.get("commit"):
        raise ReleaseGateError("release receipt was generated for a different commit")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_verification(receipt_path: Path) -> dict[str, Any]:
    source = current_source_state()
    if source["dirty_paths"]:
        raise ReleaseGateError(
            "commit or revert release-sensitive changes before verification: " + ", ".join(source["dirty_paths"])
        )
    result = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "release_smoke.py"), "--strict", "--json"],
        timeout=3600,
    )
    try:
        smoke = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseGateError("release smoke did not emit valid JSON") from exc
    if result.returncode != 0 or smoke.get("status") != "ok":
        raise ReleaseGateError("release smoke failed; no receipt was written")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "ok",
        "strict": True,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": "python scripts/release_gate.py verify",
        "source": source,
        "steps": smoke.get("steps") or [],
    }
    _write_json(receipt_path, receipt)
    validate_receipt(receipt_path, source_state=source)
    return receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_artifact(artifact: Path, artifact_kind: str) -> dict[str, Any]:
    if not artifact.is_file():
        raise ReleaseGateError(f"release artifact does not exist: {artifact}")
    script_name = f"smoke_{artifact_kind}_installer.ps1"
    env = dict(os.environ)
    env["OPENCUT_INSTALLER_SMOKE"] = "1"
    result = _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts" / script_name),
            "-InstallerPath",
            str(artifact.resolve()),
        ],
        env=env,
        timeout=1800,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ReleaseGateError(f"artifact smoke failed: {detail}")
    return {
        "name": artifact.name,
        "path": str(artifact.resolve()),
        "sha256": _sha256(artifact),
        "size_bytes": artifact.stat().st_size,
        "smoke": script_name,
        "status": "ok",
    }


def promote_artifact(
    receipt_path: Path,
    artifact: Path,
    promotion_receipt: Path,
    *,
    artifact_kind: str,
    tag: str = "",
) -> dict[str, Any]:
    source_receipt = validate_receipt(receipt_path)
    artifact_result = _smoke_artifact(artifact, artifact_kind)
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source_receipt["source"],
        "source_receipt_sha256": _sha256(receipt_path),
        "artifact": artifact_result,
        "tag": tag,
    }

    if tag:
        version = (REPO_ROOT / "opencut" / "__init__.py").read_text(encoding="utf-8")
        expected = tag.removeprefix("v")
        if f'__version__ = "{expected}"' not in version:
            raise ReleaseGateError(f"tag {tag} does not match the synchronized project version")
        if _git("tag", "--list", tag):
            raise ReleaseGateError(f"tag already exists: {tag}")
        result = _run(["git", "tag", "-a", tag, "-m", f"OpenCut {tag}"], timeout=60)
        if result.returncode != 0:
            raise ReleaseGateError(result.stderr.strip() or f"could not create tag {tag}")
    _write_json(promotion_receipt, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    verify = subparsers.add_parser("verify", help="run every local release gate and write a receipt")
    verify.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)

    validate = subparsers.add_parser("validate", help="validate a fresh receipt against this checkout")
    validate.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    validate.add_argument("--max-age-seconds", type=int, default=DEFAULT_MAX_AGE_SECONDS)

    promote = subparsers.add_parser("promote", help="smoke an artifact and optionally create its local tag")
    promote.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    promote.add_argument("--artifact", type=Path, required=True)
    promote.add_argument("--artifact-kind", choices=("wpf", "inno"), required=True)
    promote.add_argument("--promotion-receipt", type=Path, required=True)
    promote.add_argument("--tag", default="")

    args = parser.parse_args(argv)
    try:
        if args.action == "verify":
            payload = run_verification(args.receipt)
        elif args.action == "validate":
            payload = validate_receipt(args.receipt, max_age_seconds=args.max_age_seconds)
        else:
            payload = promote_artifact(
                args.receipt,
                args.artifact,
                args.promotion_receipt,
                artifact_kind=args.artifact_kind,
                tag=args.tag,
            )
    except (ReleaseGateError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1

    print(json.dumps({"status": "ok", "result": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
