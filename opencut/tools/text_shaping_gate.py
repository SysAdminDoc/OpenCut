"""Text shaping capability gate for caption burn-in/rendering (F241).

The caption stack has three render paths with different dependency surfaces:

* FFmpeg/libass for ASS/SRT/VTT burn-in.
* Pillow for transparent styled-caption overlays.
* Skia when the optional skia-python fast path is installed.

FFmpeg/libass is a hard release requirement because subtitle burn-in is a
core caption export path. Pillow RAQM and Skia shaping are reported by default
and can be promoted to hard failures in packaging CI with the strict flags.

Usage::

    python -m opencut.tools.text_shaping_gate --json
    python -m opencut.tools.text_shaping_gate --require-pillow-raqm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

FFMPEG_REQUIRED_FLAGS = ("libass", "libharfbuzz", "libfribidi")
FFMPEG_REQUIRED_FILTERS = ("ass", "subtitles")
PILLOW_FEATURES = ("raqm", "harfbuzz", "fribidi", "freetype2")

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


@dataclass
class CapabilityCheck:
    name: str
    status: str
    required: bool
    message: str
    details: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_ffmpeg_binary(
    explicit: Optional[str] = None,
    *,
    env: Optional[dict] = None,
    repo_root: Path = REPO_ROOT,
) -> Optional[str]:
    """Resolve the FFmpeg binary used by the gate."""
    if explicit:
        return explicit

    env_map = env or os.environ
    env_value = env_map.get("OPENCUT_FFMPEG") or env_map.get("FFMPEG_BINARY")
    if env_value:
        return env_value

    found = shutil.which("ffmpeg")
    if found:
        return found

    bundled_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = repo_root / "ffmpeg" / bundled_name
    if bundled.exists():
        return str(bundled)
    return None


def _run(cmd: Sequence[str], runner: Optional[Runner] = None) -> subprocess.CompletedProcess:
    if runner is not None:
        return runner(cmd)
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        check=False,
    )


def parse_ffmpeg_configuration(output: str) -> Dict[str, bool]:
    """Return required configure-flag presence from ``ffmpeg -version`` text."""
    return {
        name: f"--enable-{name}" in output
        for name in FFMPEG_REQUIRED_FLAGS
    }


def parse_ffmpeg_filters(output: str) -> Dict[str, bool]:
    """Return exact filter-name presence from ``ffmpeg -filters`` text."""
    names = set()
    for raw in output.splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        # Filter lines are "<flags> <name> <in-out> <description>". Do not
        # substring-match, because "greyedge assumption" contains "ass".
        names.add(parts[1])
    return {name: name in names for name in FFMPEG_REQUIRED_FILTERS}


def inspect_ffmpeg_text_shaping(
    ffmpeg_path: Optional[str] = None,
    *,
    runner: Optional[Runner] = None,
    env: Optional[dict] = None,
    repo_root: Path = REPO_ROOT,
) -> CapabilityCheck:
    """Verify the FFmpeg/libass path has complex-script shaping support."""
    resolved = resolve_ffmpeg_binary(ffmpeg_path, env=env, repo_root=repo_root)
    if not resolved:
        return CapabilityCheck(
            name="ffmpeg-libass",
            status="fail",
            required=True,
            message="ffmpeg was not found in PATH, OPENCUT_FFMPEG, or the bundled ffmpeg directory",
        )

    version_result = _run([resolved, "-hide_banner", "-version"], runner)
    if version_result.returncode != 0:
        return CapabilityCheck(
            name="ffmpeg-libass",
            status="fail",
            required=True,
            message="ffmpeg -version failed",
            details={
                "ffmpeg": resolved,
                "exit_code": version_result.returncode,
                "stderr": (version_result.stderr or "").strip()[-1000:],
            },
        )

    filters_result = _run([resolved, "-hide_banner", "-filters"], runner)
    if filters_result.returncode != 0:
        return CapabilityCheck(
            name="ffmpeg-libass",
            status="fail",
            required=True,
            message="ffmpeg -filters failed",
            details={
                "ffmpeg": resolved,
                "exit_code": filters_result.returncode,
                "stderr": (filters_result.stderr or "").strip()[-1000:],
            },
        )

    flags = parse_ffmpeg_configuration(version_result.stdout + "\n" + version_result.stderr)
    filters = parse_ffmpeg_filters(filters_result.stdout + "\n" + filters_result.stderr)
    missing_flags = [name for name, present in flags.items() if not present]
    missing_filters = [name for name, present in filters.items() if not present]
    if missing_flags or missing_filters:
        return CapabilityCheck(
            name="ffmpeg-libass",
            status="fail",
            required=True,
            message="ffmpeg/libass is missing required text-shaping support",
            details={
                "ffmpeg": resolved,
                "required_config_flags": flags,
                "required_filters": filters,
                "missing_config_flags": missing_flags,
                "missing_filters": missing_filters,
            },
        )

    first_line = (version_result.stdout or version_result.stderr or "").splitlines()[0:1]
    return CapabilityCheck(
        name="ffmpeg-libass",
        status="ok",
        required=True,
        message="ffmpeg/libass has HarfBuzz, FriBidi, ASS, and subtitles support",
        details={
            "ffmpeg": resolved,
            "version": first_line[0] if first_line else "",
            "required_config_flags": flags,
            "required_filters": filters,
        },
    )


def inspect_pillow_text_shaping(*, require_raqm: bool = False) -> CapabilityCheck:
    """Report whether Pillow was built with RAQM/HarfBuzz/FriBidi support."""
    try:
        from PIL import Image, features
    except ImportError:
        return CapabilityCheck(
            name="pillow-raqm",
            status="fail" if require_raqm else "skipped",
            required=require_raqm,
            message="Pillow is not installed",
        )

    checks: Dict[str, bool] = {}
    for feature in PILLOW_FEATURES:
        try:
            checks[feature] = bool(features.check(feature))
        except Exception:
            checks[feature] = False

    has_complex_shaping = checks.get("raqm") and checks.get("harfbuzz") and checks.get("fribidi")
    if has_complex_shaping:
        status = "ok"
        message = "Pillow has RAQM, HarfBuzz, and FriBidi support"
    elif require_raqm:
        status = "fail"
        message = "Pillow is installed but lacks RAQM/HarfBuzz/FriBidi complex-script shaping"
    else:
        status = "warning"
        message = "Pillow overlay rendering lacks RAQM/HarfBuzz/FriBidi; FFmpeg/libass remains the shaped burn-in path"

    return CapabilityCheck(
        name="pillow-raqm",
        status=status,
        required=require_raqm,
        message=message,
        details={
            "version": getattr(Image, "__version__", ""),
            "features": checks,
        },
    )


def inspect_skia_text_shaping(*, require_skia: bool = False) -> CapabilityCheck:
    """Report whether optional skia-python exposes a shaping-capable surface."""
    try:
        import skia  # type: ignore
    except ImportError:
        return CapabilityCheck(
            name="skia-shaping",
            status="fail" if require_skia else "skipped",
            required=require_skia,
            message="skia-python is not installed; OpenCut will use Pillow for styled overlays",
        )

    has_shaper = bool(getattr(skia, "Shaper", None))
    has_textlayout = bool(getattr(skia, "textlayout", None))
    if has_shaper or has_textlayout:
        status = "ok"
        message = "skia-python exposes shaping/textlayout APIs"
    elif require_skia:
        status = "fail"
        message = "skia-python is installed but no shaping/textlayout API was detected"
    else:
        status = "warning"
        message = "skia-python is installed but no shaping/textlayout API was detected"

    return CapabilityCheck(
        name="skia-shaping",
        status=status,
        required=require_skia,
        message=message,
        details={
            "version": getattr(skia, "__version__", ""),
            "has_shaper": has_shaper,
            "has_textlayout": has_textlayout,
        },
    )


def build_text_shaping_report(
    *,
    ffmpeg_path: Optional[str] = None,
    require_pillow_raqm: bool = False,
    require_skia: bool = False,
    runner: Optional[Runner] = None,
) -> dict:
    """Build a machine-readable shaping capability report."""
    checks = [
        inspect_ffmpeg_text_shaping(ffmpeg_path, runner=runner),
        inspect_pillow_text_shaping(require_raqm=require_pillow_raqm),
        inspect_skia_text_shaping(require_skia=require_skia),
    ]
    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warning"]
    skipped = [check for check in checks if check.status == "skipped"]
    return {
        "status": "fail" if failures else "ok",
        "summary": {
            "ok": sum(1 for check in checks if check.status == "ok"),
            "warnings": len(warnings),
            "skipped": len(skipped),
            "failures": len(failures),
        },
        "policy": {
            "ffmpeg_libass": "required",
            "pillow_raqm": "required when --require-pillow-raqm is set; warning otherwise",
            "skia_shaping": "required when --require-skia is set; skipped/warning otherwise",
        },
        "checks": [check.as_dict() for check in checks],
    }


def _print_human(report: dict) -> None:
    print(f"[text-shaping] {str(report['status']).upper()}")
    for check in report["checks"]:
        required = "required" if check["required"] else "advisory"
        print(f"- {check['name']}: {check['status']} ({required}) - {check['message']}")


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument("--ffmpeg", default=None, help="explicit ffmpeg binary path")
    parser.add_argument(
        "--require-pillow-raqm",
        action="store_true",
        help="fail if Pillow is absent or lacks RAQM/HarfBuzz/FriBidi",
    )
    parser.add_argument(
        "--require-skia",
        action="store_true",
        help="fail if skia-python is absent or lacks a shaping/textlayout API",
    )
    args = parser.parse_args(argv)

    report = build_text_shaping_report(
        ffmpeg_path=args.ffmpeg,
        require_pillow_raqm=args.require_pillow_raqm,
        require_skia=args.require_skia,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human(report)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(cli())
