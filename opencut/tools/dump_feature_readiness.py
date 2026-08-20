"""Generate the F191 route-derived feature readiness manifest.

The curated registry in :mod:`opencut.registry` carries human-reviewed labels,
install hints, docs links, and stub/experimental judgement. This generator adds
the mechanical part: it scans route functions for calls to known
``opencut.checks.check_*`` probes, joins those endpoints to the live route
manifest, and writes ``opencut/_generated/feature_readiness.json``.

Use it three ways:

* ``python -m opencut.tools.dump_feature_readiness`` - rewrites the manifest.
* ``python -m opencut.tools.dump_feature_readiness --check`` - fails on drift.
* ``build_manifest()`` - Python API used by tests and release smoke.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from opencut import checks
from opencut.dependency_support import dependency_support
from opencut.model_cards import CARDS
from opencut.registry import (
    GENERATED_FEATURE_READINESS_PATH,
    NON_AI_CHECKS,
    STATE_AVAILABLE,
)
from opencut.tools.dump_route_manifest import (
    READINESS_DEPENDENCY_GATED,
    READINESS_IMPLEMENTED,
    READINESS_STUB,
)
from opencut.tools.dump_route_manifest import (
    build_manifest as build_route_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTES_DIR = REPO_ROOT / "opencut" / "routes"
CORE_DIR = REPO_ROOT / "opencut" / "core"
CHECKS_PATH = REPO_ROOT / "opencut" / "checks.py"
MANIFEST_PATH = GENERATED_FEATURE_READINESS_PATH
MANIFEST_VERSION = 1


def _all_probe_names() -> Set[str]:
    return {
        name
        for name, value in inspect.getmembers(checks)
        if name.startswith("check_") and callable(value)
    }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _probe_aliases(probes: Set[str]) -> Dict[str, str]:
    """Map core-level check helpers back to the public checks.py probe.

    Several Wave K routes call helpers such as
    ``tts_gptsovits.check_gptsovits_available()`` while checks.py exposes the
    lightweight public wrapper as ``check_gptsovits()``. This alias table lets
    generated records use the public wrapper and avoid importing heavy model
    modules at registry import time.
    """
    aliases = {name: name for name in probes}
    try:
        tree = ast.parse(CHECKS_PATH.read_text(encoding="utf-8"))
    except OSError:
        return aliases

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in probes:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name = _call_name(sub.func)
            if name.startswith("check_"):
                aliases.setdefault(name, node.name)
    return aliases


def _blueprint_assignments(tree: ast.Module) -> Dict[str, str]:
    blueprints: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                local_name = imported.asname or imported.name
                if imported.name.endswith("_bp"):
                    blueprints[local_name] = imported.name.removesuffix("_bp")
            continue
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_blueprint = (
            isinstance(func, ast.Name)
            and func.id == "Blueprint"
            or isinstance(func, ast.Attribute)
            and func.attr == "Blueprint"
        )
        if not is_blueprint or not node.value.args:
            continue
        name_arg = node.value.args[0]
        if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                blueprints[target.id] = name_arg.value
    return blueprints


def _route_endpoints(node: ast.FunctionDef | ast.AsyncFunctionDef, blueprints: Dict[str, str]) -> List[str]:
    endpoints: List[str] = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if (
            not isinstance(func, ast.Attribute)
            or func.attr != "route"
            or not isinstance(func.value, ast.Name)
            or func.value.id not in blueprints
        ):
            continue
        endpoint_name = node.name
        for keyword in decorator.keywords:
            if (
                keyword.arg == "endpoint"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                endpoint_name = keyword.value.value
        endpoints.append(f"{blueprints[func.value.id]}.{endpoint_name}")
    return endpoints


def _called_probes(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: Dict[str, str],
) -> List[str]:
    probes: Set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        name = _call_name(sub.func)
        if name in aliases:
            probes.add(aliases[name])
    return sorted(probes)


def route_endpoint_probes(
    routes_dir: Path = ROUTES_DIR,
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    aliases = aliases or _probe_aliases(_all_probe_names())
    endpoint_probes: Dict[str, List[str]] = {}
    for path in sorted(routes_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        blueprints = _blueprint_assignments(tree)
        if not blueprints:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            endpoints = _route_endpoints(node, blueprints)
            if not endpoints:
                continue
            probes = _called_probes(node, aliases)
            if not probes:
                continue
            for endpoint in endpoints:
                endpoint_probes[endpoint] = probes
    return endpoint_probes


def _slug_probe(probe: str) -> str:
    slug = probe
    if slug.startswith("check_"):
        slug = slug[len("check_") :]
    if slug.endswith("_available"):
        slug = slug[: -len("_available")]
    return slug.replace("_", "-")


def _label_probe(probe: str) -> str:
    return _slug_probe(probe).replace("-", " ").title()


def _route_category(routes: Iterable[str]) -> str:
    for route in sorted(routes):
        parts = [part for part in route.split("/") if part and not part.startswith("<")]
        if parts:
            head = parts[0]
            if head in {"tts", "adr"}:
                return "audio"
            if head in {"caption", "captions", "subtitle"}:
                return "captions"
            if head in {"llm", "agent"}:
                return "llm"
            return head
    return "generated"


def _minimum_vram_mb(hardware: str) -> int:
    """Return the minimum VRAM requirement encoded in a model-card hardware string."""
    match = re.search(r">=\s*(\d+(?:\.\d+)?)\s*(GB|MB)\s+VRAM", hardware, re.IGNORECASE)
    if not match:
        return 0
    amount = float(match.group(1))
    unit = match.group(2).lower()
    return int(amount if unit == "mb" else amount * 1024)


def _requires_gpu(hardware: str) -> bool:
    return hardware.strip().lower().startswith("gpu")


_ROUTE_READINESS_TO_FEATURE_STATE = {
    READINESS_STUB: "stub",
    READINESS_DEPENDENCY_GATED: "missing_dependency",
    READINESS_IMPLEMENTED: STATE_AVAILABLE,
}


def _derive_feature_state(
    routes: Iterable[str],
    route_readiness: Dict[str, str],
) -> str:
    """Derive a feature state from its routes' readiness classifications.

    If ALL routes are stubs the feature is ``stub``; if ALL are
    dependency-gated (and none implemented) it is ``missing_dependency``;
    otherwise at least one route is implemented so the feature is ``available``.
    """
    levels = set()
    for route in routes:
        levels.add(route_readiness.get(route, READINESS_IMPLEMENTED))
    if not levels:
        return STATE_AVAILABLE
    if levels == {READINESS_STUB}:
        return "stub"
    if levels <= {READINESS_STUB, READINESS_DEPENDENCY_GATED}:
        return "missing_dependency"
    return STATE_AVAILABLE


def _capability_routes(routes: Iterable[str]) -> List[str]:
    """Prefer executable feature routes over metadata-only companions.

    An implemented ``/info`` or ``/models`` endpoint can describe a feature
    whose actual operation is still a 501 stub.  It must not promote that
    feature to available.  If a probe is only used by metadata endpoints, keep
    those routes so infrastructure dashboards retain their existing state.
    """
    routes_list = list(routes)
    executable = [
        route
        for route in routes_list
        if not route.rstrip("/").endswith(("/info", "/models"))
    ]
    return executable or routes_list


_IMPL_MODULES_BY_PROBE: Optional[Dict[str, List[str]]] = None


def _core_module_stems(node: ast.AST) -> List[str]:
    """``opencut.core.<stem>`` modules imported anywhere under *node*."""
    modules: List[str] = []

    def _add(stem: str) -> None:
        if stem and stem not in modules:
            modules.append(stem)

    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.module:
            # `from opencut.core import music_acestep` names the module itself.
            if child.module == "opencut.core":
                for alias in child.names:
                    if alias.name != "*":
                        _add(alias.name)
                continue
            names = [child.module]
        elif isinstance(child, ast.Import):
            names = [alias.name for alias in child.names]
        else:
            continue
        for name in names:
            if not name.startswith("opencut.core."):
                continue
            _add(name[len("opencut.core."):].split(".", 1)[0])
    return modules


def _module_level_core_bindings(tree: ast.Module) -> Dict[str, str]:
    """Map names bound at module level to the ``opencut.core`` module behind them.

    Route files commonly do ``from opencut.core import music_acestep`` (or
    ``import opencut.core.x as y``) at the top and reference the module inside
    the handler, so a function-body-only scan sees no adapter at all.
    """
    bindings: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "opencut.core":
                for alias in node.names:
                    if alias.name != "*":
                        bindings[alias.asname or alias.name] = alias.name
            elif node.module.startswith("opencut.core."):
                stem = node.module[len("opencut.core."):].split(".", 1)[0]
                for alias in node.names:
                    bindings[alias.asname or alias.name] = stem
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("opencut.core."):
                    continue
                stem = alias.name[len("opencut.core."):].split(".", 1)[0]
                bindings[alias.asname or alias.name.split(".")[-1]] = stem
    return bindings


def _referenced_core_stems(
    node: ast.AST,
    bindings: Dict[str, str],
) -> List[str]:
    """Module-level ``opencut.core`` bindings this function actually uses."""
    if not bindings:
        return []
    stems: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in bindings:
            stem = bindings[child.id]
            if stem not in stems:
                stems.append(stem)
    return stems


def _impl_modules_from_checks() -> Dict[str, List[str]]:
    """Adapter modules each ``checks.check_*`` probe imports directly."""
    index: Dict[str, List[str]] = {}
    try:
        source = Path(inspect.getsourcefile(checks) or "").read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, TypeError, ValueError):  # pragma: no cover - defensive
        logging.getLogger("opencut").warning(
            "feature readiness: could not index opencut.checks adapters"
        )
        return index

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("check_"):
            continue
        modules = _core_module_stems(node)
        if modules:
            index[node.name] = modules
    return index


def _impl_modules_from_routes(
    routes_dir: Path = ROUTES_DIR,
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Adapter modules imported by the route functions each probe gates.

    Most probes test a third-party binary or package directly
    (``check_auto_editor_available`` calls ``shutil.which``), so scanning
    ``opencut.checks`` alone finds no adapter for them. The implementation the
    probe actually gates lives in the route handler, which imports
    ``opencut.core.<adapter>`` before calling it — that is the module the stub
    scanner needs to inspect.
    """
    aliases = aliases or _probe_aliases(_all_probe_names())
    index: Dict[str, List[str]] = defaultdict(list)

    for path in sorted(routes_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        blueprints = _blueprint_assignments(tree)
        if not blueprints:
            continue
        bindings = _module_level_core_bindings(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _route_endpoints(node, blueprints):
                continue
            probes = _called_probes(node, aliases)
            if not probes:
                continue
            modules = _core_module_stems(node)
            for stem in _referenced_core_stems(node, bindings):
                if stem not in modules:
                    modules.append(stem)
            if not modules:
                continue
            for probe in probes:
                for stem in modules:
                    if stem not in index[probe]:
                        index[probe].append(stem)
    return dict(index)


def _build_impl_module_index() -> Dict[str, List[str]]:
    """Map each ``checks.check_*`` probe to the adapter modules behind it.

    Generated records used to carry no implementation identity at all, which
    made them structurally incapable of being graded as stubs: with no module
    to scan, a terminal ``NotImplementedError`` adapter whose optional
    dependency happened to be installed was advertised as available.

    The mapping is derived rather than hand-maintained because the naming is
    not always mechanical — ``check_searaft`` lives in ``flow_searaft`` — so
    guessing ``opencut.core.<probe stem>`` silently misses adapters. Two
    sources are merged: what the probe itself imports, and what the routes it
    gates import.
    """
    index: Dict[str, List[str]] = {
        probe: list(modules) for probe, modules in _impl_modules_from_checks().items()
    }
    for probe, modules in _impl_modules_from_routes().items():
        merged = index.setdefault(probe, [])
        for stem in modules:
            if stem not in merged:
                merged.append(stem)

    # Last resort for probes whose only route is the dependency dashboard:
    # a module named exactly after the probe. Verified against the filesystem,
    # so it resolves a real adapter or contributes nothing.
    for probe in _all_probe_names():
        if index.get(probe):
            continue
        stem = _slug_probe(probe).replace("-", "_")
        if (CORE_DIR / f"{stem}.py").is_file():
            index[probe] = [stem]

    return index


def impl_modules_for_probe(probe: str) -> List[str]:
    """Adapter module stems backing ``probe``; empty when none can be derived."""
    global _IMPL_MODULES_BY_PROBE
    if _IMPL_MODULES_BY_PROBE is None:
        _IMPL_MODULES_BY_PROBE = _build_impl_module_index()
    return list(_IMPL_MODULES_BY_PROBE.get(probe, ()))


def _record_for_probe(
    probe: str,
    routes: Iterable[str],
    route_readiness: Optional[Dict[str, str]] = None,
) -> dict:
    routes_sorted = sorted(set(routes))
    cards = {card.check_name: card for card in CARDS}
    card = cards.get(probe)
    if card is not None:
        feature_id = card.feature_id
        label = card.label
        category = card.category
        install_hint = card.install_hint
        docs = "docs/MODELS.md"
        hardware = card.hardware
    elif probe in NON_AI_CHECKS:
        feature_id = f"auto.{_slug_probe(probe)}"
        label = _label_probe(probe)
        category = _route_category(routes_sorted)
        install_hint = "bundled or system dependency; see route response"
        docs = "docs/MODELS.md#excluded-infrastructure-checks"
        hardware = ""
    else:
        feature_id = f"auto.{_slug_probe(probe)}"
        label = _label_probe(probe)
        category = _route_category(routes_sorted)
        install_hint = "see route response for dependency hint"
        docs = "docs/MODELS.md"
        hardware = ""

    support = dependency_support(f"{feature_id} {label} {install_hint}")
    if not support["supported"]:
        install_hint = (
            "Unavailable in OpenCut's supported dependency matrix: "
            f"{support['reason']}"
        )

    state = _derive_feature_state(
        _capability_routes(routes_sorted),
        route_readiness or {},
    )

    return {
        "feature_id": feature_id,
        "label": label,
        "category": category,
        "state": state,
        "install_hint": install_hint,
        "docs": docs,
        "routes": routes_sorted,
        "check_name": probe,
        "impl_module": ",".join(impl_modules_for_probe(probe)),
        "source": "generated",
        "notes": "Auto-derived from route functions that call this check probe.",
        "hardware": hardware,
        "requires_gpu": _requires_gpu(hardware),
        "minimum_vram_mb": _minimum_vram_mb(hardware),
    }


def unproven_available_records(manifest: dict) -> List[str]:
    """Records claiming ``available`` that name no adapter the stub scanner can read.

    A record with an empty ``impl_module`` is structurally incapable of being
    graded as a stub — that blind spot is what let three terminal-stub adapters
    advertise as available. An adapter stem that does not resolve to a real
    ``opencut/core/<stem>.py`` is the same blind spot with extra steps.
    """
    offenders: List[str] = []
    for record in manifest.get("records", []):
        if record.get("state") != STATE_AVAILABLE:
            continue
        stems = [s.strip() for s in str(record.get("impl_module") or "").split(",") if s.strip()]
        if not stems:
            offenders.append(
                f"{record.get('feature_id')} ({record.get('check_name')}): no impl_module"
            )
            continue
        missing = [stem for stem in stems if not (CORE_DIR / f"{stem}.py").is_file()]
        if missing:
            offenders.append(
                f"{record.get('feature_id')} ({record.get('check_name')}): "
                f"impl_module does not resolve: {', '.join(missing)}"
            )
    return offenders


def build_manifest(route_manifest: Optional[dict] = None) -> dict:
    route_manifest = route_manifest or build_route_manifest()
    endpoint_probes = route_endpoint_probes()
    routes_by_probe: Dict[str, Set[str]] = defaultdict(set)

    route_readiness: Dict[str, str] = {}
    for route in route_manifest.get("routes", []):
        rule = str(route.get("rule") or "")
        if rule:
            route_readiness[rule] = str(
                route.get("readiness") or READINESS_IMPLEMENTED
            )
        endpoint = str(route.get("endpoint") or "")
        for probe in endpoint_probes.get(endpoint, []):
            routes_by_probe[probe].add(rule)

    records = [
        _record_for_probe(probe, routes, route_readiness)
        for probe, routes in sorted(routes_by_probe.items())
        if routes
    ]
    return {
        "version": MANIFEST_VERSION,
        # When the route generator supplies its in-memory snapshot, preserve
        # that timestamp so both committed artifacts visibly share one clock.
        "generated_at": route_manifest.get("generated_at")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "opencut/_generated/route_manifest.json",
        "total_records": len(records),
        "total_routes": sum(len(record["routes"]) for record in records),
        "records": records,
    }


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


def _signature(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "generated_at"}


def diff_manifests(expected: dict, live: dict) -> List[str]:
    diffs: List[str] = []
    expected_sig = _signature(expected)
    live_sig = _signature(live)
    for key in ("version", "total_records", "total_routes"):
        if expected_sig.get(key) != live_sig.get(key):
            diffs.append(f"{key}: committed={expected_sig.get(key)!r} live={live_sig.get(key)!r}")

    expected_records = {record["feature_id"]: record for record in expected_sig.get("records", [])}
    live_records = {record["feature_id"]: record for record in live_sig.get("records", [])}
    for feature_id in sorted(set(expected_records) - set(live_records)):
        diffs.append(f"removed feature: {feature_id}")
    for feature_id in sorted(set(live_records) - set(expected_records)):
        diffs.append(f"new feature: {feature_id}")
    for feature_id in sorted(set(expected_records) & set(live_records)):
        if expected_records[feature_id] != live_records[feature_id]:
            diffs.append(f"changed feature: {feature_id}")
    return diffs


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed manifest differs from the live route/check graph",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = build_manifest()

    unproven = unproven_available_records(manifest)
    if unproven:
        print("[feature-readiness] FAIL - available records with no proven implementation:")
        for line in unproven[:25]:
            print(f"  - {line}")
        if len(unproven) > 25:
            print(f"  - ... {len(unproven) - 25} more")
        return 1

    if args.check:
        existing = load_manifest()
        if existing is None:
            print(
                "[feature-readiness] FAIL - no committed manifest at "
                f"{MANIFEST_PATH.relative_to(REPO_ROOT)}"
            )
            return 1
        diffs = diff_manifests(existing, manifest)
        if diffs:
            print("[feature-readiness] FAIL - committed manifest is out of sync:")
            for line in diffs[:25]:
                print(f"  - {line}")
            if len(diffs) > 25:
                print(f"  - ... {len(diffs) - 25} more diff lines")
            return 1
        print(
            f"[feature-readiness] OK - {manifest['total_records']} generated "
            f"records / {manifest['total_routes']} route bindings"
        )
        return 0

    out = write_manifest(manifest)
    print(
        f"[feature-readiness] wrote {out.relative_to(REPO_ROOT)} - "
        f"{manifest['total_records']} generated records / "
        f"{manifest['total_routes']} route bindings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
