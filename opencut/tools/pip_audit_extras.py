"""Run pip-audit against OpenCut's committed and optional dependencies.

F263 exists because the release smoke gate used to audit only
``requirements.txt``. That misses the committed lockfile and the heavy optional
install surface in ``pyproject.toml`` -- especially the ``[all]`` extra that
users choose when they want every AI/video/audio backend available.

This tool keeps the release gate structured and deterministic from OpenCut's
side: it builds temporary requirements files for each requested target, invokes
``python -m pip_audit`` for each one, and emits one JSON status object per
target. The vulnerability data itself remains live pip-audit output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
LOCKFILE_PATH = REPO_ROOT / "requirements-lock.txt"
DEFAULT_EXTRAS = ("all",)


@dataclass(frozen=True)
class AuditTarget:
    name: str
    kind: str
    requirements: List[str]
    extra: str = ""
    source: str = ""
    no_deps: bool = False


@dataclass(frozen=True)
class AllowedAdvisory:
    package: str
    aliases: tuple[str, ...]
    reason: str
    docs: str = "docs/PYTHON_ADVISORIES.md"


ALLOWED_ADVISORIES: dict[str, AllowedAdvisory] = {
    "CVE-2024-27763": AllowedAdvisory(
        package="basicsr",
        aliases=("GHSA-86w8-vhw6-q9qq",),
        reason=(
            "BasicSR has no fixed release; the reported path is a contrived "
            "local SLURM_NODELIST/scontrol execution scenario behind optional "
            "local RealESRGAN/GFPGAN usage."
        ),
    ),
    "CVE-2026-1839": AllowedAdvisory(
        package="transformers",
        aliases=("GHSA-69w3-r845-3855",),
        reason=(
            "Transformers 5.x is blocked by WhisperX's huggingface-hub<1 pin; "
            "OpenCut does not use transformers.Trainer checkpoint resume, and "
            "the resolved pyproject[all] Torch stack is 2.8."
        ),
    ),
}


def _clean_requirement_lines(lines: Iterable[str]) -> List[str]:
    requirements: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


def _dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def load_pyproject_requirements(pyproject_path: Path, extra: str) -> List[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    base = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    if extra not in optional:
        choices = ", ".join(sorted(optional))
        raise ValueError(f"unknown pyproject extra {extra!r}; expected one of: {choices}")
    return _dedupe_preserving_order([*base, *optional[extra]])


def load_requirements(requirements_path: Path) -> List[str]:
    return _clean_requirement_lines(requirements_path.read_text(encoding="utf-8").splitlines())


def build_targets(
    *,
    pyproject_path: Path = PYPROJECT_PATH,
    requirements_path: Path = REQUIREMENTS_PATH,
    lockfile_path: Path = LOCKFILE_PATH,
    extras: Iterable[str] = DEFAULT_EXTRAS,
    include_requirements: bool = True,
    include_lockfile: bool = True,
) -> List[AuditTarget]:
    targets: List[AuditTarget] = []
    if include_requirements:
        targets.append(
            AuditTarget(
                name="requirements.txt",
                kind="requirements",
                requirements=load_requirements(requirements_path),
                source=str(requirements_path.relative_to(REPO_ROOT)),
            )
        )
    if include_lockfile:
        targets.append(
            AuditTarget(
                name="requirements-lock.txt",
                kind="lockfile",
                requirements=load_requirements(lockfile_path),
                source=str(lockfile_path.relative_to(REPO_ROOT)),
                no_deps=True,
            )
        )

    for extra in extras:
        targets.append(
            AuditTarget(
                name=f"pyproject[{extra}]",
                kind="pyproject-extra",
                extra=extra,
                requirements=load_pyproject_requirements(pyproject_path, extra),
                source=str(pyproject_path.relative_to(REPO_ROOT)),
            )
        )
    return targets


def _tail(text: str, lines: int = 12) -> str:
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _json_from_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("pip-audit did not emit JSON")
    return json.loads(stdout[start:])


def _run(cmd: List[str], cwd: Path, timeout: int, env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def _vulnerability_ids(vulnerability: dict) -> set[str]:
    ids: set[str] = set()
    primary = str(vulnerability.get("id") or "").strip()
    if primary:
        ids.add(primary)
    for alias in vulnerability.get("aliases") or []:
        cleaned = str(alias).strip()
        if cleaned:
            ids.add(cleaned)
    return ids


def _allowed_advisory_for(package: str, vulnerability: dict) -> tuple[str, AllowedAdvisory] | tuple[None, None]:
    package_key = package.lower()
    found_ids = {item.lower() for item in _vulnerability_ids(vulnerability)}
    for advisory_id, advisory in ALLOWED_ADVISORIES.items():
        allowed_ids = {advisory_id.lower(), *(alias.lower() for alias in advisory.aliases)}
        if package_key == advisory.package.lower() and found_ids.intersection(allowed_ids):
            return advisory_id, advisory
    return None, None


def _collect_vulnerabilities(dependencies: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for dependency in dependencies:
        package = str(dependency.get("name") or "")
        version = str(dependency.get("version") or "")
        for vulnerability in dependency.get("vulns", []) or []:
            advisory_id, advisory = _allowed_advisory_for(package, vulnerability)
            finding = {
                "package": package,
                "version": version,
                "id": vulnerability.get("id", ""),
                "aliases": vulnerability.get("aliases") or [],
                "fix_versions": vulnerability.get("fix_versions") or [],
                "allowed": advisory is not None,
            }
            if advisory:
                finding["waiver"] = {
                    "id": advisory_id,
                    "reason": advisory.reason,
                    "docs": advisory.docs,
                }
            findings.append(finding)
    return findings


def _audit_target(
    target: AuditTarget,
    *,
    timeout: int,
    process_timeout: int,
    vulnerability_service: str,
) -> dict:
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="opencut-pip-audit-") as tmp:
        req_path = Path(tmp) / "requirements.txt"
        cache_path = Path(tmp) / "pip-cache"
        cache_path.mkdir()
        req_path.write_text("\n".join(target.requirements) + "\n", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "pip_audit",
            "-r",
            str(req_path),
            "--format",
            "json",
            "--progress-spinner",
            "off",
            "--timeout",
            str(timeout),
            "--cache-dir",
            str(cache_path),
        ]
        if vulnerability_service:
            cmd.extend(["--vulnerability-service", vulnerability_service])
        if target.no_deps:
            cmd.append("--no-deps")
        env = {**os.environ, "PIP_CACHE_DIR": str(cache_path)}
        try:
            result = _run(cmd, cwd=REPO_ROOT, timeout=process_timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.time() - start) * 1000)
            return {
                "name": target.name,
                "kind": target.kind,
                "extra": target.extra,
                "source": target.source,
                "status": "fail",
                "exit_code": -1,
                "duration_ms": duration_ms,
                "requirement_count": len(target.requirements),
                "no_deps": target.no_deps,
                "resolved_dependency_count": 0,
                "vulnerability_count": 0,
                "allowed_vulnerability_count": 0,
                "unallowed_vulnerability_count": 0,
                "vulnerabilities": [],
                "parse_error": f"pip-audit timed out after {process_timeout}s",
                "stdout_tail": _tail((exc.stdout or "") if isinstance(exc.stdout, str) else ""),
                "stderr_tail": _tail((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
            }

    duration_ms = int((time.time() - start) * 1000)
    payload: dict = {}
    parse_error = ""
    try:
        payload = _json_from_stdout(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = str(exc)

    dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else []
    vulnerabilities = _collect_vulnerabilities(dependencies)
    vulnerability_count = len(vulnerabilities)
    allowed_vulnerability_count = sum(1 for vuln in vulnerabilities if vuln["allowed"])
    unallowed_vulnerability_count = vulnerability_count - allowed_vulnerability_count
    collection_failed = parse_error or result.returncode not in (0, 1) or (result.returncode == 1 and vulnerability_count == 0)
    status = "ok" if not collection_failed and unallowed_vulnerability_count == 0 else "fail"
    return {
        "name": target.name,
        "kind": target.kind,
        "extra": target.extra,
        "source": target.source,
        "status": status,
        "exit_code": result.returncode,
        "duration_ms": duration_ms,
        "requirement_count": len(target.requirements),
        "no_deps": target.no_deps,
        "resolved_dependency_count": len(dependencies),
        "vulnerability_count": vulnerability_count,
        "allowed_vulnerability_count": allowed_vulnerability_count,
        "unallowed_vulnerability_count": unallowed_vulnerability_count,
        "vulnerabilities": vulnerabilities,
        "parse_error": parse_error,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def run_audits(
    targets: Iterable[AuditTarget],
    *,
    timeout: int = 15,
    process_timeout: int = 300,
    vulnerability_service: str = "pypi",
) -> dict:
    if importlib.util.find_spec("pip_audit") is None:
        return {
            "status": "skipped",
            "message": "pip-audit not installed",
            "targets": [],
            "target_count": 0,
            "vulnerability_count": 0,
            "allowed_vulnerability_count": 0,
            "unallowed_vulnerability_count": 0,
        }

    results = [
        _audit_target(
            target,
            timeout=timeout,
            process_timeout=process_timeout,
            vulnerability_service=vulnerability_service,
        )
        for target in targets
    ]
    vulnerability_count = sum(result["vulnerability_count"] for result in results)
    allowed_vulnerability_count = sum(result["allowed_vulnerability_count"] for result in results)
    unallowed_vulnerability_count = sum(result["unallowed_vulnerability_count"] for result in results)
    failed = [result for result in results if result["status"] != "ok"]
    status = "fail" if failed else "ok"
    return {
        "status": status,
        "message": (
            "no unallowed advisories"
            if status == "ok"
            else f"{len(failed)} pip-audit target(s) reported unallowed advisories or collection errors"
        ),
        "targets": results,
        "target_count": len(results),
        "vulnerability_count": vulnerability_count,
        "allowed_vulnerability_count": allowed_vulnerability_count,
        "unallowed_vulnerability_count": unallowed_vulnerability_count,
    }


def _optional_extra_names(pyproject_path: Path) -> List[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra",
        action="append",
        dest="extras",
        help="pyproject optional dependency extra to audit; repeatable (default: all)",
    )
    parser.add_argument(
        "--all-extras",
        action="store_true",
        help="audit every optional dependency extra from pyproject.toml",
    )
    parser.add_argument(
        "--no-requirements",
        action="store_true",
        help="skip the legacy requirements.txt audit target",
    )
    parser.add_argument(
        "--no-lockfile",
        action="store_true",
        help="skip the committed requirements-lock.txt audit target",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="skip pyproject optional dependency audit targets",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="pip-audit network timeout in seconds per request",
    )
    parser.add_argument(
        "--process-timeout",
        type=int,
        default=300,
        help="maximum seconds to allow each pip-audit subprocess to resolve and audit",
    )
    parser.add_argument(
        "--vulnerability-service",
        choices=("pypi", "osv"),
        default="pypi",
        help="pip-audit vulnerability service",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON only",
    )
    args = parser.parse_args(argv)

    if args.no_extras and (args.extras or args.all_extras):
        parser.error("--no-extras cannot be combined with --extra or --all-extras")
    extras = [] if args.no_extras else (_optional_extra_names(PYPROJECT_PATH) if args.all_extras else (args.extras or list(DEFAULT_EXTRAS)))
    targets = build_targets(
        extras=extras,
        include_requirements=not args.no_requirements,
        include_lockfile=not args.no_lockfile,
    )
    payload = run_audits(
        targets,
        timeout=args.timeout,
        process_timeout=args.process_timeout,
        vulnerability_service=args.vulnerability_service,
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"[pip-audit-extras] {payload['status'].upper()} — {payload['message']}")
        for target in payload.get("targets", []):
            print(
                "  "
                f"{target['name']}: {target['status']} "
                f"({target['unallowed_vulnerability_count']} unallowed, "
                f"{target['allowed_vulnerability_count']} allowed, "
                f"{target['resolved_dependency_count']} resolved deps)"
            )
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(cli())
