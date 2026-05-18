#!/usr/bin/env python3
"""
Verify the local OpenCut development bootstrap.

This is intentionally stdlib-only so it can run before project dependencies are
installed. It validates the environment shape and prints actionable failures
instead of crashing on the first missing dependency.

Usage:
    python scripts/bootstrap_check.py
    python scripts/bootstrap_check.py --json
    python scripts/bootstrap_check.py --metadata-only
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 11)


def _resolve_python_for_subprocess() -> str:
    """Return a Python executable that can actually spawn a child process.

    F181 — `sys.executable` is normally the right answer, but UV-style
    trampolines (and some Windows redirected installs) point at a
    binary whose target was removed without the venv being cleaned up.
    Spawning the trampoline raises ``FileNotFoundError`` /
    ``OSError(errno=2)``. Detect that case once at bootstrap time and
    fall back to ``shutil.which("python")`` / ``python3`` / ``py``.

    Returns ``sys.executable`` on the happy path so non-trampoline runs
    keep their existing behaviour.
    """
    import shutil

    # Probe sys.executable with a no-op so we fail fast on broken
    # trampolines instead of failing inside the real check.
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0:
            return sys.executable
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Fallback hunt — same order as opencut.security._find_system_python.
    for name in ("python", "python3", "py"):
        candidate = shutil.which(name)
        if not candidate:
            continue
        if Path(candidate).resolve() == Path(sys.executable).resolve():
            # Same broken trampoline; skip.
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "import sys; sys.exit(0)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue
    # Last resort — return sys.executable anyway. Callers will see the
    # actual error from the failed subprocess.run.
    return sys.executable

RUNTIME_IMPORTS: Dict[str, str] = {
    "click": "click",
    "rich": "rich",
    "flask": "flask",
    "flask-cors": "flask_cors",
    "python-json-logger": "pythonjsonlogger",
    "psutil": "psutil",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail)


def _fail(name: str, detail: str, hint: str = "") -> CheckResult:
    return CheckResult(name=name, ok=False, detail=detail, hint=hint)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _path_is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_python_version() -> CheckResult:
    current = sys.version_info[:3]
    current_text = ".".join(str(part) for part in current)
    required_text = ".".join(str(part) for part in MIN_PYTHON)
    if current >= MIN_PYTHON:
        return _pass("python-version", f"Python {current_text} satisfies >= {required_text}")
    return _fail(
        "python-version",
        f"Python {current_text} is below required >= {required_text}",
        "Install Python 3.11 or newer before running OpenCut.",
    )


def check_repo_import(repo_root: Path = REPO_ROOT) -> CheckResult:
    root_text = str(repo_root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True

    try:
        module = importlib.import_module("opencut")
        module_path = Path(getattr(module, "__file__", "")).resolve()
    except Exception as exc:
        return _fail(
            "repo-import",
            f"Could not import opencut from the repository: {exc}",
            "Run this script from a checkout and install the package with python -m pip install -e .",
        )
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass

    if _path_is_relative_to(module_path, repo_root.resolve()):
        return _pass("repo-import", f"opencut imports from {module_path.relative_to(repo_root)}")

    return _fail(
        "repo-import",
        f"opencut imports from {module_path}, not this checkout",
        "Activate the intended virtualenv or run python -m pip install -e . from the repo root.",
    )


def check_runtime_imports(imports: Dict[str, str] = RUNTIME_IMPORTS) -> CheckResult:
    missing = [package for package, module_name in imports.items() if importlib.util.find_spec(module_name) is None]
    if not missing:
        return _pass("runtime-imports", "Required runtime imports are available")
    return _fail(
        "runtime-imports",
        "Missing required runtime imports: " + ", ".join(sorted(missing)),
        'Install core dependencies with python -m pip install -e ".[dev]".',
    )


def check_server_import(repo_root: Path = REPO_ROOT) -> CheckResult:
    root_text = str(repo_root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            module = importlib.import_module("opencut.server")
            create_app = getattr(module, "create_app", None)
    except Exception as exc:
        return _fail(
            "server-import",
            f"Could not import opencut.server: {exc}",
            'Install core dependencies with python -m pip install -e ".[dev]".',
        )
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass

    if callable(create_app):
        return _pass("server-import", "opencut.server.create_app is importable")
    return _fail("server-import", "opencut.server imported but create_app is not callable")


def check_version_sync(repo_root: Path = REPO_ROOT) -> CheckResult:
    script = repo_root / "scripts" / "sync_version.py"
    if not script.exists():
        return _fail("version-sync", "scripts/sync_version.py is missing")

    python = _resolve_python_for_subprocess()
    try:
        completed = subprocess.run(
            [python, str(script), "--check"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        # F181 — UV / Windows trampoline whose target Python is missing.
        return _fail(
            "version-sync",
            f"Could not spawn Python to run sync_version.py: {exc}",
            "Your venv looks like a UV trampoline pointing at a missing "
            "interpreter. Recreate the venv (`uv venv` or `python -m venv .venv`) "
            "or run the script through a system Python.",
        )
    except OSError as exc:
        return _fail(
            "version-sync",
            f"Could not run sync_version.py: {exc}",
            "Check that the Python interpreter is functional and the script is "
            "readable.",
        )
    except subprocess.TimeoutExpired:
        return _fail(
            "version-sync",
            "sync_version.py timed out after 60 seconds",
            "Investigate any process blocking sync_version.py and rerun.",
        )
    if completed.returncode == 0:
        return _pass("version-sync", "Version targets are in sync")

    output = (completed.stdout + completed.stderr).strip().splitlines()
    summary = "; ".join(line.strip() for line in output if "MISMATCH" in line)
    if not summary:
        summary = (completed.stdout + completed.stderr).strip() or "version check failed"
    return _fail(
        "version-sync",
        summary,
        "Run python scripts/sync_version.py and commit all touched version targets.",
    )


def _active_requirement_lines(lines: Iterable[str]) -> List[str]:
    active: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            active.append(line)
    return active


def check_lockfile(repo_root: Path = REPO_ROOT) -> CheckResult:
    lockfile = repo_root / "requirements-lock.txt"
    if not lockfile.exists():
        return _fail("requirements-lock", "requirements-lock.txt is missing")

    active = _active_requirement_lines(_read_text(lockfile).splitlines())
    self_entries = [
        line for line in active
        if re.match(r"^opencut(?:\s*(==|>=|<=|~=|>|<|@)|\s*$)", line, flags=re.IGNORECASE)
    ]
    if self_entries:
        return _fail(
            "requirements-lock",
            "Lockfile contains self-package entry: " + ", ".join(self_entries),
            "Regenerate the lockfile from dependencies only; do not pin opencut from PyPI.",
        )

    return _pass("requirements-lock", f"requirements-lock.txt has {len(active)} auditable dependency lines")


def run_checks(metadata_only: bool = False, repo_root: Path = REPO_ROOT) -> List[CheckResult]:
    checks = [
        check_python_version(),
        check_repo_import(repo_root),
        check_version_sync(repo_root),
        check_lockfile(repo_root),
    ]
    if not metadata_only:
        checks.extend([
            check_runtime_imports(),
            check_server_import(repo_root),
        ])
    return checks


def _print_text(results: Sequence[CheckResult]) -> None:
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"{prefix} {result.name}: {result.detail}")
        if result.hint:
            print(f"  hint: {result.hint}")


def _print_json(results: Sequence[CheckResult]) -> None:
    payload = {
        "ok": all(result.ok for result in results),
        "checks": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify OpenCut bootstrap readiness")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip runtime dependency and server import checks",
    )
    args = parser.parse_args(argv)

    results = run_checks(metadata_only=args.metadata_only)
    if args.json:
        _print_json(results)
    else:
        _print_text(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
