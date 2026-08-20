"""Generate the canonical route/feature manifest (F099).

The README and roadmap quote route counts that drift with every release.
This module is the single source of truth: it walks
:class:`~flask.Flask.url_map`, categorises by blueprint and first-party
surface, and writes a deterministic JSON manifest to
``opencut/_generated/route_manifest.json``.

Use it three ways:

* ``python -m opencut.tools.dump_route_manifest`` — rewrites the manifest.
* ``python -m opencut.tools.dump_route_manifest --check`` — fails when
  the committed manifest disagrees with the live Flask app (used in CI).
* ``opencut.tools.dump_route_manifest.build_manifest()`` — Python API
  used by tests and other generators.

The manifest is keyed by route rule so re-runs are stable across Python
dict orderings, and methods are sorted alphabetically with the standard
``HEAD``/``OPTIONS`` Flask additions stripped.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "route_manifest.json"
FEATURE_READINESS_MANIFEST_PATH = (
    REPO_ROOT / "opencut" / "_generated" / "feature_readiness.json"
)
MANIFEST_VERSION = 4
SURFACE_MANIFEST_VERSION = 1

SURFACE_CLASSES = (
    "panel",
    "palette",
    "cli",
    "mcp",
    "integration-only",
)
_SURFACE_PRECEDENCE = SURFACE_CLASSES
_ROUTE_STRING_RE = re.compile(r'''(?P<quote>["'`])(?P<route>/[^"'`\r\n]*)(?P=quote)''')

_STD_METHODS = {"HEAD", "OPTIONS"}

# Route readiness tiers. Only ``stub`` routes are excluded from the advertised
# (shipped) route count — they return HTTP 501 and have no real implementation.
# ``dependency-gated`` routes are fully implemented but require an optional
# dependency to be installed; they ship and are counted.
READINESS_IMPLEMENTED = "implemented"
READINESS_DEPENDENCY_GATED = "dependency-gated"
READINESS_STUB = "stub"

# Source-level markers. 501 ROUTE_STUBBED / _stub_501 are strategic stubs with
# no implementation; _stub_503 / missing_dependency mark optional-dependency
# gates on otherwise-real handlers.
_STUB_MARKERS = ("_stub_501(", "ROUTE_STUBBED")
_DEPENDENCY_MARKERS = ("_stub_503(", "missing_dependency(", "MISSING_DEPENDENCY")

# Handlers that delegate to an ``opencut.core`` adapter whose entrypoint is
# still a terminal ``raise NotImplementedError`` stub (e.g. the Wave Q/R/S
# ``raise RuntimeError(INSTALL_HINT)`` pattern over asr_parakeet.transcribe)
# are stubs too: installing the dependency only swaps the error class.
#
# A route that calls a stub entrypoint *defensively* — inside a ``try`` whose
# handler catches the stub's own error and falls back to a working engine
# (e.g. ``video_ai_upscale`` prefers SeedVR2 then falls back to Real-ESRGAN) —
# is NOT a stub; only an unguarded stub call with no fallback makes the whole
# route terminal.
_CORE_IMPORT_RE = re.compile(
    r"^\s*from opencut\.core import (?P<names>[\w ,]+)$", re.MULTILINE
)
_FALLBACK_EXC_NAMES = {"NotImplementedError", "RuntimeError", "Exception", "BaseException"}


def _imported_core_modules(source: str) -> set:
    modules = set()
    for match in _CORE_IMPORT_RE.finditer(source):
        for module in (name.strip() for name in match.group("names").split(",")):
            if module:
                modules.add(module)
    return modules


def _handler_catches_fallback(try_node: ast.Try) -> bool:
    """True when a ``try`` block catches a stub-style error (so it can fall back)."""
    for handler in try_node.handlers:
        exc = handler.type
        if exc is None:  # bare ``except:``
            return True
        names = exc.elts if isinstance(exc, ast.Tuple) else [exc]
        for name in names:
            if isinstance(name, ast.Name) and name.id in _FALLBACK_EXC_NAMES:
                return True
    return False


def _delegates_to_stub_entrypoint(source: str) -> bool:
    """True when *source* calls a terminal-stub core entrypoint with no fallback.

    A stub call guarded by a ``try/except`` that catches its error is treated as
    a defensive delegation with a fallback, not a terminal stub.
    """
    from opencut.core.stub_scan import stub_functions

    modules = _imported_core_modules(source)
    stub_calls = {
        f"{module}.{fn}"
        for module in modules
        for fn in stub_functions(module)
    }
    if not stub_calls:
        return False

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        # Fall back to the coarse textual check when the source can't be parsed.
        return any(
            re.search(rf"\b{re.escape(call.split('.')[0])}\.{call.split('.')[1]}\s*\(", source)
            for call in stub_calls
        )

    guarded_lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _handler_catches_fallback(node):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    guarded_lines.add(getattr(child, "lineno", -1))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        if f"{func.value.id}.{func.attr}" in stub_calls and node.lineno not in guarded_lines:
            return True
    return False


def _return_expression_has_501_status(value: Optional[ast.expr]) -> bool:
    """Return whether a direct handler return encodes HTTP status 501.

    OpenCut handlers normally use ``error_response(..., status=501)``.  The
    tuple and ``make_response`` forms are recognised too so the readiness gate
    follows the HTTP contract rather than one helper's spelling.
    """
    if value is None:
        return False
    if isinstance(value, ast.Tuple):
        return any(
            isinstance(item, ast.Constant) and item.value == 501
            for item in value.elts[1:]
        )
    if not isinstance(value, ast.Call):
        return False

    for keyword in value.keywords:
        if (
            keyword.arg in {"status", "status_code"}
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 501
        ):
            return True

    func_name = ""
    if isinstance(value.func, ast.Name):
        func_name = value.func.id
    elif isinstance(value.func, ast.Attribute):
        func_name = value.func.attr
    if func_name == "make_response":
        return any(
            isinstance(arg, ast.Constant) and arg.value == 501
            for arg in value.args[1:]
        )
    return False


def _has_unconditional_501_return(source: str) -> bool:
    """Return whether the handler's reached success path terminates at 501.

    Only returns directly in the function body count.  A conditional 501 is
    not an unconditional success-path stub, while a top-level return after an
    optional-dependency guard is.  This distinction prevents an earlier 503
    marker from making an impossible-to-use route look dependency-gated.
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False

    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if function is None:
        return False
    return any(
        isinstance(statement, ast.Return)
        and _return_expression_has_501_status(statement.value)
        for statement in function.body
    )


def _classify_readiness(view_func) -> str:
    """Classify a Flask view function as implemented / dependency-gated / stub.

    Uses static inspection of the (unwrapped) handler source so the manifest
    can report which advertised routes are real vs. 501 strategic stubs. When
    the source is unavailable (e.g. generated backend routes), the route is a
    real implementation, so it defaults to ``implemented``.
    """
    if view_func is None:
        return READINESS_IMPLEMENTED
    try:
        target = inspect.unwrap(view_func)
        source = inspect.getsource(target)
    except (OSError, TypeError, ValueError):
        return READINESS_IMPLEMENTED

    if any(marker in source for marker in _STUB_MARKERS):
        return READINESS_STUB
    if _delegates_to_stub_entrypoint(source):
        return READINESS_STUB
    if _has_unconditional_501_return(source):
        return READINESS_STUB
    if any(marker in source for marker in _DEPENDENCY_MARKERS):
        return READINESS_DEPENDENCY_GATED
    return READINESS_IMPLEMENTED


@dataclass
class RouteEntry:
    rule: str
    methods: List[str]
    endpoint: str
    blueprint: str
    readiness: str = READINESS_IMPLEMENTED
    workflow: Optional[Dict[str, str]] = None

    def as_tuple(self):
        return (self.rule, tuple(self.methods), self.endpoint, self.blueprint,
                self.readiness, self.workflow)


@dataclass
class Manifest:
    version: int
    generated_at: str
    total_routes: int
    shipped_route_count: int
    blueprint_count: int
    method_counts: Dict[str, int]
    readiness_counts: Dict[str, int]
    blueprints: Dict[str, Dict[str, object]]
    routes: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "total_routes": self.total_routes,
            "shipped_route_count": self.shipped_route_count,
            "blueprint_count": self.blueprint_count,
            "method_counts": dict(sorted(self.method_counts.items())),
            "readiness_counts": dict(sorted(self.readiness_counts.items())),
            "blueprints": {
                name: {
                    "route_count": meta["route_count"],
                    "method_counts": dict(sorted(meta["method_counts"].items())),
                    "sample_rules": meta["sample_rules"],
                }
                for name, meta in sorted(self.blueprints.items())
            },
            "routes": sorted(self.routes, key=lambda r: (r["rule"], tuple(r["methods"]))),
        }


def _collect_routes(app) -> List[RouteEntry]:
    from opencut.core.workflow import get_workflow_step_metadata

    entries: List[RouteEntry] = []
    for rule in app.url_map.iter_rules():
        methods = sorted(m for m in rule.methods if m not in _STD_METHODS)
        if not methods:
            continue
        endpoint = rule.endpoint
        blueprint = endpoint.split(".", 1)[0] if "." in endpoint else "<app>"
        view_func = app.view_functions.get(endpoint)
        workflow = None
        if "POST" in methods:
            workflow = get_workflow_step_metadata(view_func)
        entries.append(
            RouteEntry(
                rule=rule.rule,
                methods=methods,
                endpoint=endpoint,
                blueprint=blueprint,
                readiness=_classify_readiness(view_func),
                workflow=workflow,
            )
        )
    return entries


def _summarise(entries: Iterable[RouteEntry]) -> Manifest:
    method_counts: Counter = Counter()
    readiness_counts: Counter = Counter()
    by_blueprint: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"route_count": 0, "method_counts": Counter(), "sample_rules": []}
    )
    routes_payload: List[Dict[str, object]] = []

    for entry in entries:
        for method in entry.methods:
            method_counts[method] += 1
        readiness_counts[entry.readiness] += 1
        bucket = by_blueprint[entry.blueprint]
        bucket["route_count"] += 1
        bucket["method_counts"].update(entry.methods)
        if len(bucket["sample_rules"]) < 3:
            bucket["sample_rules"].append(entry.rule)
        route_payload = {
            "rule": entry.rule,
            "methods": list(entry.methods),
            "endpoint": entry.endpoint,
            "blueprint": entry.blueprint,
            "readiness": entry.readiness,
        }
        if entry.workflow:
            route_payload["workflow"] = dict(entry.workflow)
        routes_payload.append(route_payload)

    blueprints_payload = {
        name: {
            "route_count": meta["route_count"],
            "method_counts": dict(meta["method_counts"]),
            "sample_rules": sorted(meta["sample_rules"]),
        }
        for name, meta in by_blueprint.items()
    }

    shipped = len(routes_payload) - readiness_counts.get(READINESS_STUB, 0)

    return Manifest(
        version=MANIFEST_VERSION,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_routes=len(routes_payload),
        shipped_route_count=shipped,
        blueprint_count=len(blueprints_payload),
        method_counts=dict(method_counts),
        readiness_counts=dict(readiness_counts),
        blueprints=blueprints_payload,
        routes=routes_payload,
    )


def _surface_source_paths() -> Dict[str, List[Path]]:
    """Return the first-party source files used by the route surface scan.

    The scan deliberately uses the same literal-reference boundary as the
    research audit. Generated bundles, route manifests, and backend route
    implementations are excluded so a backend route is not mistaken for a
    user-facing surface merely because it exists.
    """

    uxp_root = REPO_ROOT / "extension" / "com.opencut.uxp"
    panel_sources = [
        REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js",
        *sorted(uxp_root.glob("*.js")),
    ]
    return {
        "panel": sorted(set(panel_sources)),
        "palette": [REPO_ROOT / "opencut" / "core" / "command_palette.py"],
        "cli": [REPO_ROOT / "opencut" / "cli.py"],
        "mcp": [REPO_ROOT / "opencut" / "mcp_server.py"],
    }


def _literal_surface_references(route_rules: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    """Map route rules to source files containing exact route literals."""

    rules = {str(rule) for rule in route_rules if str(rule).startswith("/")}
    references: Dict[str, Dict[str, List[str]]] = {rule: {} for rule in rules}
    for surface, paths in _surface_source_paths().items():
        for path in paths:
            if not path.is_file():
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            literals = {
                match.group("route")
                for match in _ROUTE_STRING_RE.finditer(source)
                if match.group("route") in rules
            }
            relative = path.relative_to(REPO_ROOT).as_posix()
            for rule in literals:
                references[rule].setdefault(surface, []).append(relative)

    for rule in references:
        for surface in references[rule]:
            references[rule][surface] = sorted(set(references[rule][surface]))
    return references


def validate_surface_coverage(routes: Iterable[dict]) -> List[str]:
    """Return gate errors for routes without a valid declared surface class."""

    errors: List[str] = []
    allowed = set(SURFACE_CLASSES)
    for route in routes:
        if not isinstance(route, dict):
            errors.append("route entry is not an object")
            continue
        rule = str(route.get("rule") or "<unknown>")
        primary = route.get("surface_class")
        classes = route.get("surface_classes")
        evidence = route.get("surface_evidence")
        if primary not in allowed:
            errors.append(f"{rule} has no valid surface_class")
        if not isinstance(classes, list) or not classes or not set(classes).issubset(allowed):
            errors.append(f"{rule} has no valid surface_classes")
        elif primary not in classes:
            errors.append(f"{rule} surface_class is absent from surface_classes")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{rule} has no surface evidence")
    return errors


def _apply_surface_metadata(manifest: dict) -> dict:
    """Attach route surface declarations and the classification gate."""

    routes = manifest.get("routes", [])
    references = _literal_surface_references(
        route.get("rule", "") for route in routes if isinstance(route, dict)
    )
    primary_counts = {surface: 0 for surface in SURFACE_CLASSES}
    assigned_counts = {surface: 0 for surface in SURFACE_CLASSES}
    direct_routes = 0
    multi_surface_routes = 0
    shipped_routes = 0

    for route in routes:
        if not isinstance(route, dict):
            continue
        rule = str(route.get("rule") or "")
        matched = references.get(rule, {})
        classes = [surface for surface in _SURFACE_PRECEDENCE if surface in matched]
        if not classes:
            classes = ["integration-only"]
            evidence = [
                "no literal reference in first-party panel, palette, CLI, or curated MCP sources"
            ]
        else:
            evidence = [
                source
                for surface in classes
                for source in matched.get(surface, [])
            ]
            direct_routes += 1
        if len(classes) > 1:
            multi_surface_routes += 1
        primary = classes[0]
        route["surface_class"] = primary
        route["surface_classes"] = classes
        route["surface_evidence"] = evidence
        primary_counts[primary] += 1
        for surface in classes:
            assigned_counts[surface] += 1
        if route.get("readiness") != READINESS_STUB:
            shipped_routes += 1

    errors = validate_surface_coverage(routes)
    shipped_integration_only = sum(
        1
        for route in routes
        if route.get("readiness") != READINESS_STUB
        and route.get("surface_class") == "integration-only"
    )
    shipped_direct = shipped_routes - shipped_integration_only
    manifest["surface_coverage"] = {
        "version": SURFACE_MANIFEST_VERSION,
        "sources": {
            surface: [
                path.relative_to(REPO_ROOT).as_posix()
                for path in paths
                if path.is_file()
            ]
            for surface, paths in _surface_source_paths().items()
        },
        "classes": list(SURFACE_CLASSES),
        "summary": {
            "shipped_routes": shipped_routes,
            "direct_surface_routes": shipped_direct,
            "integration_only_routes": shipped_integration_only,
            "coverage_percent": round((shipped_direct / shipped_routes) * 100, 1)
            if shipped_routes
            else 0.0,
            "multi_surface_routes": multi_surface_routes,
            "primary_counts": primary_counts,
            "assigned_counts": assigned_counts,
        },
        "gate": {
            "passes": not errors,
            "errors": errors,
            "unclassified_routes": [
                str(route.get("rule") or "<unknown>")
                for route in routes
                if route.get("surface_class") not in SURFACE_CLASSES
            ],
        },
    }
    return manifest


def build_manifest(app=None) -> dict:
    """Return the manifest dict computed from a live Flask app.

    ``app`` defaults to a freshly created OpenCut server so call sites
    such as tests can pass in a stripped-down fixture for unit tests.
    """
    if app is None:
        # Import lazily so unit tests can avoid the full server boot.
        from opencut.server import create_app

        # Describing the route graph must not mutate the machine being
        # described. A production boot here swept temp files, started the
        # disk-monitor thread, ran credential migrations, and imported
        # whatever plugins were installed locally.
        app = create_app(introspection=True)

    entries = _collect_routes(app)
    return _apply_surface_metadata(_summarise(entries).as_dict())


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_signature(manifest: dict) -> dict:
    """Drop the timestamp so diff checks don't rebuild on every run."""
    return {k: v for k, v in manifest.items() if k != "generated_at"}


def diff_manifests(expected: dict, live: dict) -> List[str]:
    """Return human-readable diff lines, empty when identical (ignoring time)."""
    diffs: List[str] = []
    expected_sig = _manifest_signature(expected)
    live_sig = _manifest_signature(live)

    for key in (
        "version",
        "total_routes",
        "shipped_route_count",
        "blueprint_count",
        "surface_coverage",
    ):
        if expected_sig.get(key) != live_sig.get(key):
            diffs.append(
                f"{key}: committed={expected_sig.get(key)!r} live={live_sig.get(key)!r}"
            )

    expected_routes = {
        (r["rule"], tuple(r["methods"])): r
        for r in expected_sig.get("routes", [])
        if isinstance(r, dict)
    }
    live_routes = {
        (r["rule"], tuple(r["methods"])): r
        for r in live_sig.get("routes", [])
        if isinstance(r, dict)
    }
    expected_keys = set(expected_routes)
    live_keys = set(live_routes)
    only_in_committed = sorted(expected_keys - live_keys)
    only_in_live = sorted(live_keys - expected_keys)
    for rule, methods in only_in_committed:
        diffs.append(f"removed: {','.join(methods)} {rule}")
    for rule, methods in only_in_live:
        diffs.append(f"new:     {','.join(methods)} {rule}")
    for rule, methods in sorted(expected_keys & live_keys):
        expected_route = expected_routes[(rule, methods)]
        live_route = live_routes[(rule, methods)]
        if expected_route != live_route:
            fields = sorted(
                field for field in set(expected_route) | set(live_route)
                if expected_route.get(field) != live_route.get(field)
            )
            diffs.append(f"changed: {','.join(methods)} {rule} fields={','.join(fields)}")

    if expected_sig.get("blueprints", {}) != live_sig.get("blueprints", {}):
        expected_bps = set(expected_sig.get("blueprints", {}).keys())
        live_bps = set(live_sig.get("blueprints", {}).keys())
        for name in sorted(expected_bps - live_bps):
            diffs.append(f"blueprint removed: {name}")
        for name in sorted(live_bps - expected_bps):
            diffs.append(f"blueprint added: {name}")

    return diffs


def _build_feature_readiness_manifest(route_manifest: dict) -> dict:
    """Build the companion readiness artifact from this exact route snapshot."""
    from opencut.tools.dump_feature_readiness import build_manifest

    return build_manifest(route_manifest=route_manifest)


def _feature_readiness_diffs(live: dict) -> List[str]:
    """Return drift in the companion artifact checked by this same gate."""
    from opencut.tools.dump_feature_readiness import diff_manifests, load_manifest

    existing = load_manifest(FEATURE_READINESS_MANIFEST_PATH)
    if existing is None:
        return [
            "no committed companion manifest at "
            f"{FEATURE_READINESS_MANIFEST_PATH.relative_to(REPO_ROOT)}"
        ]
    return diff_manifests(existing, live)


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed manifest is out of sync with the live app",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress info logging",
    )
    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    manifest = build_manifest()
    feature_readiness = _build_feature_readiness_manifest(manifest)

    if args.check:
        existing = load_manifest()
        if existing is None:
            print("[route-manifest] FAIL — no committed manifest at "
                  f"{MANIFEST_PATH.relative_to(REPO_ROOT)} (run without --check first)")
            return 1
        diffs = diff_manifests(existing, manifest)
        readiness_diffs = _feature_readiness_diffs(feature_readiness)
        if readiness_diffs:
            diffs.extend(
                f"feature readiness: {line}" for line in readiness_diffs
            )
        if diffs:
            print(
                "[route-manifest] FAIL — route/readiness manifests are out of "
                "sync with the live app:"
            )
            for line in diffs[:25]:
                print(f"  - {line}")
            if len(diffs) > 25:
                print(f"  - ... {len(diffs) - 25} more diff lines (re-run without --check to rebuild)")
            return 1
        print(
            f"[route-manifest] OK — {manifest['total_routes']} routes across "
            f"{manifest['blueprint_count']} blueprints"
        )
        return 0

    out = write_manifest(manifest)
    from opencut.tools.dump_feature_readiness import write_manifest as write_readiness

    readiness_out = write_readiness(
        feature_readiness,
        FEATURE_READINESS_MANIFEST_PATH,
    )
    print(
        f"[route-manifest] wrote {out.relative_to(REPO_ROOT)} — "
        f"{manifest['total_routes']} routes across {manifest['blueprint_count']} blueprints"
    )
    print(
        "[route-manifest] wrote companion "
        f"{readiness_out.relative_to(REPO_ROOT)} — "
        f"{feature_readiness['total_records']} readiness records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
