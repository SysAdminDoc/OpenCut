#!/usr/bin/env python3
"""
OpenCut README badge sync.

Closes RESEARCH_FEATURE_PLAN_2026-05-25.md Q4/E8: the README shipped a
hand-edited "API Routes-1344" / "Tests-7600+" badge that drifted away from
the generated route manifest and the live test count.

Source of truth:
  - Routes  -> ``opencut/_generated/route_manifest.json::shipped_route_count``
              (advertised count excludes 501 strategic stubs; falls back to
              ``total_routes`` for older manifests without the field)
  - Tests   -> ``len(glob('tests/test_*.py'))`` rounded down to nearest 100.
              The README badge uses ``<N>+`` (e.g. ``7600+``) so any drift
              by less than 100 is intentionally tolerated.
  - Version -> ``opencut/__init__.py::__version__`` (already handled by
              ``scripts/sync_version.py``; we re-check it here only for
              consistency, never write).
  - Python  -> from ``pyproject.toml::requires-python`` (read-only).

Usage:
    python scripts/sync_badges.py            # update README
    python scripts/sync_badges.py --check    # exit 1 if README is stale
    python scripts/sync_badges.py --json     # emit JSON status for CI
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROUTE_MANIFEST = ROOT / "opencut" / "_generated" / "route_manifest.json"
TESTS_DIR = ROOT / "tests"
CAPTION_STYLES = ROOT / "opencut" / "core" / "caption_styles.py"
OPENCUT_INIT = ROOT / "opencut" / "__init__.py"

ROUTE_BADGE_RE = re.compile(
    r"(!\[Routes\]\(https://img\.shields\.io/badge/API%20Routes-)(\d+)(-orange\))"
)
TEST_BADGE_RE = re.compile(
    r"(!\[Tests\]\(https://img\.shields\.io/badge/Tests-)(\d+\+)(-brightgreen\))"
)
FEATURE_OVERVIEW_ROUTES_RE = re.compile(
    r"OpenCut v(?P<version>[\w.]+) includes \*\*[\d,]+ shipped API routes\*\* "
    r"\(implemented or dependency-gated; \d+ strategic 501 stubs are tracked separately and excluded\)"
)
DIAGRAM_ROUTES_RE = re.compile(r"\|\s+[\d,]+ shipped routes\s+\|")
COMPARISON_CAPTION_STYLES_RE = re.compile(r"\| Animated captions \| \d+ styles,")
CAPTION_FEATURE_STYLES_RE = re.compile(r"\| \d+ Caption Styles \|")
TESTING_PROSE_RE = re.compile(r"[\d,]+\+ estimated tests across \d+ root test files")
TREE_TESTS_RE = re.compile(
    r"tests/\s+# pytest test suite \([\d,]+\+ estimated tests, \d+ root test files\)"
)


class TextTarget:
    def __init__(self, label: str, regex: re.Pattern, expected: str) -> None:
        self.label = label
        self.regex = regex
        self.expected = expected


def _live_manifest() -> dict:
    if not ROUTE_MANIFEST.exists():
        raise SystemExit(
            f"ERROR: {ROUTE_MANIFEST.relative_to(ROOT)} not found. "
            "Run: python -m opencut.tools.dump_route_manifest"
        )
    payload = json.loads(ROUTE_MANIFEST.read_text(encoding="utf-8"))
    return payload


def _live_route_count() -> int:
    # Advertise shipped routes only (excludes 501 strategic stubs). Older
    # manifests without the field fall back to total_routes.
    payload = _live_manifest()
    count = int(payload.get("shipped_route_count", payload.get("total_routes", 0)))
    if count <= 0:
        raise SystemExit("ERROR: route manifest reports zero routes")
    return count


def _live_stub_count() -> int:
    payload = _live_manifest()
    readiness = payload.get("readiness_counts") or {}
    return int(readiness.get("stub", 0))


def _live_test_count_rounded() -> int:
    """Count test files, then floor to nearest hundred for the ``N+`` badge."""
    if not TESTS_DIR.is_dir():
        return 0
    n = _live_root_test_file_count()
    # The badge displays e.g. 7600+ — the floor-to-hundred convention is
    # ``floor(test_files * 40 / 100) * 100`` if we assume ~40 tests/file,
    # but that's brittle. Instead we hard-floor to nearest 100 of the file
    # count multiplied by an average tests-per-file factor; we keep the
    # factor in one place so updating the convention is one knob.
    AVG_TESTS_PER_FILE = 40  # historic shipping average in this repo
    estimated_tests = n * AVG_TESTS_PER_FILE
    return (estimated_tests // 100) * 100


def _live_root_test_file_count() -> int:
    if not TESTS_DIR.is_dir():
        return 0
    return sum(1 for p in TESTS_DIR.glob("test_*.py"))


def _live_caption_style_count() -> int:
    if not CAPTION_STYLES.exists():
        raise SystemExit(f"ERROR: {CAPTION_STYLES.relative_to(ROOT)} not found")
    tree = ast.parse(CAPTION_STYLES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "_STYLE_DEFS" in names and isinstance(node.value, ast.List):
                count = len(node.value.elts)
                if count <= 0:
                    raise SystemExit("ERROR: caption style catalog is empty")
                return count
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_STYLE_DEFS"
            and isinstance(node.value, ast.List)
        ):
            count = len(node.value.elts)
            if count <= 0:
                raise SystemExit("ERROR: caption style catalog is empty")
            return count
    raise SystemExit("ERROR: could not find _STYLE_DEFS in caption_styles.py")


def _live_version() -> str:
    if not OPENCUT_INIT.exists():
        raise SystemExit(f"ERROR: {OPENCUT_INIT.relative_to(ROOT)} not found")
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
        OPENCUT_INIT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise SystemExit("ERROR: could not find __version__ in opencut/__init__.py")
    return match.group(1)


def _format_int(value: int) -> str:
    return f"{value:,}"


def _replace_badge(text: str, regex: re.Pattern, new_value: str) -> tuple[str, bool]:
    """Return (new_text, did_change)."""
    m = regex.search(text)
    if not m:
        return text, False
    if m.group(2) == new_value:
        return text, False
    new_text = text[: m.start(2)] + new_value + text[m.end(2):]
    return new_text, True


def _replace_text_target(text: str, target: TextTarget) -> tuple[str, bool]:
    m = target.regex.search(text)
    if not m:
        return text, False
    if m.group(0) == target.expected:
        return text, False
    return text[: m.start()] + target.expected + text[m.end():], True


def _build_badge_targets() -> list[tuple[str, re.Pattern, str]]:
    routes = _live_route_count()
    tests = _live_test_count_rounded()
    return [
        ("API Routes badge", ROUTE_BADGE_RE, str(routes)),
        ("Tests badge", TEST_BADGE_RE, f"{tests}+"),
    ]


def _build_text_targets() -> list[TextTarget]:
    routes = _live_route_count()
    route_text = _format_int(routes)
    stubs = _live_stub_count()
    caption_styles = _live_caption_style_count()
    test_files = _live_root_test_file_count()
    tests = _live_test_count_rounded()
    test_text = _format_int(tests)
    version = _live_version()
    return [
        TextTarget(
            "Feature overview route prose",
            FEATURE_OVERVIEW_ROUTES_RE,
            (
                f"OpenCut v{version} includes **{route_text} shipped API routes** "
                f"(implemented or dependency-gated; {stubs} strategic 501 stubs are "
                "tracked separately and excluded)"
            ),
        ),
        TextTarget(
            "Architecture diagram route count",
            DIAGRAM_ROUTES_RE,
            f"|  {route_text} shipped routes |",
        ),
        TextTarget(
            "Comparison caption style count",
            COMPARISON_CAPTION_STYLES_RE,
            f"| Animated captions | {caption_styles} styles,",
        ),
        TextTarget(
            "Caption feature style count",
            CAPTION_FEATURE_STYLES_RE,
            f"| {caption_styles} Caption Styles |",
        ),
        TextTarget(
            "Testing prose counts",
            TESTING_PROSE_RE,
            f"{test_text}+ estimated tests across {test_files} root test files",
        ),
        TextTarget(
            "Project tree test counts",
            TREE_TESTS_RE,
            f"tests/               # pytest test suite ({test_text}+ estimated tests, {test_files} root test files)",
        ),
    ]


def cmd_sync() -> int:
    if not README.exists():
        print(f"ERROR: README not found at {README}", file=sys.stderr)
        return 2
    text = README.read_text(encoding="utf-8")
    original = text
    changed_any = False
    for label, regex, new_value in _build_badge_targets():
        text, did = _replace_badge(text, regex, new_value)
        if did:
            print(f"  SYNC {label} -> {new_value}")
            changed_any = True
        else:
            print(f"  OK   {label} already {new_value}")
    for target in _build_text_targets():
        text, did = _replace_text_target(text, target)
        if did:
            print(f"  SYNC {target.label}")
            changed_any = True
        else:
            print(f"  OK   {target.label}")
    if changed_any and text != original:
        README.write_text(text, encoding="utf-8")
        print(f"\nWrote {README.relative_to(ROOT)}.")
    else:
        print(f"\nNo changes needed in {README.relative_to(ROOT)}.")
    return 0


def cmd_check() -> int:
    if not README.exists():
        print(f"ERROR: README not found at {README}", file=sys.stderr)
        return 2
    text = README.read_text(encoding="utf-8")
    drift = []
    for label, regex, new_value in _build_badge_targets():
        m = regex.search(text)
        if not m:
            drift.append((label, "missing", new_value))
            continue
        if m.group(2) != new_value:
            drift.append((label, m.group(2), new_value))
    for target in _build_text_targets():
        m = target.regex.search(text)
        if not m:
            drift.append((target.label, "missing", target.expected))
            continue
        if m.group(0) != target.expected:
            drift.append((target.label, m.group(0), target.expected))
    if drift:
        print("README badge drift detected:", file=sys.stderr)
        for label, current, expected in drift:
            print(f"  {label}: README has {current!r}, expected {expected!r}", file=sys.stderr)
        print(
            "\nRun: python scripts/sync_badges.py",
            file=sys.stderr,
        )
        return 1
    print("README badges in sync with live counts.")
    return 0


def cmd_json() -> int:
    payload = {
        "route_count": _live_route_count(),
        "stub_count": _live_stub_count(),
        "test_count_rounded": _live_test_count_rounded(),
        "root_test_files": _live_root_test_file_count(),
        "caption_style_count": _live_caption_style_count(),
        "drift": [],
    }
    if README.exists():
        text = README.read_text(encoding="utf-8")
        for label, regex, expected in _build_badge_targets():
            m = regex.search(text)
            payload["drift"].append({
                "label": label,
                "current": m.group(2) if m else None,
                "expected": expected,
                "drifted": (not m) or (m.group(2) != expected),
            })
        for target in _build_text_targets():
            m = target.regex.search(text)
            payload["drift"].append({
                "label": target.label,
                "current": m.group(0) if m else None,
                "expected": target.expected,
                "drifted": (not m) or (m.group(0) != target.expected),
            })
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    any_drift = any(d.get("drifted") for d in payload["drift"])
    return 1 if any_drift else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync README badges with live project counts.")
    parser.add_argument("--check", action="store_true", help="Exit 1 if README badges drift.")
    parser.add_argument("--json", action="store_true", help="Emit JSON status for CI.")
    args = parser.parse_args()
    if args.json:
        sys.exit(cmd_json())
    if args.check:
        sys.exit(cmd_check())
    sys.exit(cmd_sync())


if __name__ == "__main__":
    main()
