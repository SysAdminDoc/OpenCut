#!/usr/bin/env python3
"""Validate and optionally resolve OpenCut's supported dependency matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencut.dependency_support import (  # noqa: E402
    EXTRA_SUPPORT,
    PLATFORMS,
    PYTHON_REQUIRES,
    PYTHON_VERSIONS,
    assert_extra_names,
    extra_support,
    normalise_platform,
    python_supported,
)

UV_PLATFORMS = {"win32": "windows", "linux": "linux", "darwin": "macos"}


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def validate_contract() -> dict:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["requires-python"] != PYTHON_REQUIRES:
        raise RuntimeError(
            f"pyproject requires-python is {project['requires-python']!r}; expected {PYTHON_REQUIRES!r}"
        )
    assert_extra_names(project["optional-dependencies"])
    if not python_supported():
        detected = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(f"OpenCut supports Python 3.11-3.14; detected Python {detected}.")
    return project


def matrix_lanes() -> list[tuple[str, str]]:
    return [
        (python_version, platform_name)
        for platform_name in PLATFORMS
        for python_version in PYTHON_VERSIONS
    ]


def resolve_extras(
    selected: list[str],
    *,
    python_version: str | None = None,
    platform_name: str | None = None,
) -> list[dict]:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for dependency-matrix resolution; install uv 0.11.16 or newer.")
    results: list[dict] = []
    target_platform = normalise_platform(platform_name)
    target_python = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    version_tuple = tuple(int(part) for part in target_python.split("."))
    with tempfile.TemporaryDirectory(prefix="opencut-dependency-matrix-") as temp_dir:
        venv = Path(temp_dir) / "venv"
        create = subprocess.run(
            [uv, "venv", "--python", sys.executable, "--no-project", str(venv)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if create.returncode != 0:
            raise RuntimeError(f"could not create clean resolver environment: {create.stderr.strip()}")
        python_exe = _venv_python(venv)
        for extra in selected:
            support = extra_support(
                extra,
                version=version_tuple,
                platform_name=target_platform,
            )
            if not support["supported"]:
                results.append(
                    {
                        "extra": extra,
                        "python": target_python,
                        "platform": target_platform,
                        "status": "unsupported",
                        "reason": support["reason"],
                    }
                )
                continue
            command = [
                uv,
                "pip",
                "install",
                "--dry-run",
                "--python",
                str(python_exe),
                "--python-version",
                target_python,
                "--python-platform",
                UV_PLATFORMS[target_platform],
                f".[{extra}]",
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"resolver failed for Python {target_python}/"
                    f"{target_platform}/opencut-ppro[{extra}]: {detail}"
                )
            results.append(
                {
                    "extra": extra,
                    "python": target_python,
                    "platform": target_platform,
                    "status": "resolved",
                    "reason": "",
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="run uv against every supported extra")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="resolve every supported extra for all 12 OS/Python target lanes",
    )
    parser.add_argument("--extra", action="append", default=[], help="limit resolution to one extra")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_contract()
        selected = args.extra or sorted(EXTRA_SUPPORT)
        unknown = sorted(set(selected) - set(EXTRA_SUPPORT))
        if unknown:
            raise RuntimeError(f"unknown extras: {', '.join(unknown)}")
        if args.matrix:
            results = []
            for python_version, platform_name in matrix_lanes():
                results.extend(
                    resolve_extras(
                        selected,
                        python_version=python_version,
                        platform_name=platform_name,
                    )
                )
        elif args.resolve:
            results = resolve_extras(selected)
        else:
            results = [
            {"extra": name, "status": "supported", "reason": ""}
            for name in selected
            ]
        payload = {
            "status": "ok",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": normalise_platform(),
            "results": results,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        payload = {"status": "error", "error": str(exc)}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload["status"] == "ok":
        resolved = sum(item["status"] == "resolved" for item in payload["results"])
        print(
            f"dependency matrix valid for Python {payload['python']}/{payload['platform']}: "
            f"{len(payload['results'])} extras ({resolved} resolved)"
        )
    else:
        print(f"dependency matrix failed: {payload['error']}", file=sys.stderr)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
