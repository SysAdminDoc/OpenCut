"""Route-readiness enrichment for command/search feature entries."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROUTE_MANIFEST_PATH = PACKAGE_ROOT / "_generated" / "route_manifest.json"
FEATURE_READINESS_PATH = PACKAGE_ROOT / "_generated" / "feature_readiness.json"

READINESS_IMPLEMENTED = "implemented"
READINESS_DEPENDENCY_GATED = "dependency-gated"
READINESS_STUB = "stub"
READINESS_MISSING_ROUTE = "missing_route"
READINESS_NO_ROUTE = "no_route"

RUNNABLE_READINESS = {
    READINESS_IMPLEMENTED,
    READINESS_DEPENDENCY_GATED,
}


def _clean_route(route: object) -> str:
    value = str(route or "").strip()
    if not value:
        return ""
    return value.split("?", 1)[0].rstrip("/") or "/"


@lru_cache(maxsize=1)
def _route_manifest_by_rule() -> dict[str, dict]:
    try:
        payload = json.loads(ROUTE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[str, dict] = {}
    for record in payload.get("routes") or []:
        if not isinstance(record, Mapping):
            continue
        rule = _clean_route(record.get("rule"))
        if rule:
            out[rule] = dict(record)
    return out


@lru_cache(maxsize=1)
def _feature_readiness_by_route() -> dict[str, dict]:
    try:
        payload = json.loads(FEATURE_READINESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    out: dict[str, dict] = {}
    for record in payload.get("records") or []:
        if not isinstance(record, Mapping):
            continue
        for route in record.get("routes") or []:
            rule = _clean_route(route)
            if rule and rule not in out:
                out[rule] = dict(record)
    return out


def route_readiness(route: object) -> dict:
    """Return readiness metadata for a route advertised by a feature entry."""
    rule = _clean_route(route)
    if not rule:
        return {
            "route": "",
            "readiness": READINESS_NO_ROUTE,
            "route_valid": False,
            "runnable": False,
            "route_methods": [],
            "readiness_reason": "No backend route is attached to this feature.",
        }

    manifest_record = _route_manifest_by_rule().get(rule)
    feature_record = _feature_readiness_by_route().get(rule)
    if manifest_record is None:
        return {
            "route": rule,
            "readiness": READINESS_MISSING_ROUTE,
            "route_valid": False,
            "runnable": False,
            "route_methods": [],
            "readiness_reason": (
                "This feature is not backed by a live route in "
                "opencut/_generated/route_manifest.json."
            ),
        }

    readiness = str(manifest_record.get("readiness") or READINESS_IMPLEMENTED)
    methods = [str(method) for method in manifest_record.get("methods") or []]
    payload = {
        "route": rule,
        "readiness": readiness,
        "route_valid": True,
        "runnable": readiness in RUNNABLE_READINESS,
        "route_methods": methods,
        "readiness_reason": _readiness_reason(readiness, feature_record),
    }
    if feature_record:
        payload["feature_state"] = str(feature_record.get("state") or "")
        payload["install_hint"] = str(feature_record.get("install_hint") or "")
        payload["readiness_docs"] = str(feature_record.get("docs") or "")
    return payload


def _readiness_reason(readiness: str, feature_record: Mapping | None) -> str:
    if readiness == READINESS_IMPLEMENTED:
        return "Live backend route is implemented."
    if readiness == READINESS_DEPENDENCY_GATED:
        hint = str((feature_record or {}).get("install_hint") or "").strip()
        if hint:
            return f"Live backend route exists, but optional setup may be required: {hint}"
        return "Live backend route exists, but optional dependencies may be required."
    if readiness == READINESS_STUB:
        return "Route is registered as a strategic stub and is not runnable yet."
    return f"Route readiness is {readiness}."


def enrich_feature_entry(entry: Mapping) -> dict:
    """Return a feature entry with route/readiness fields attached."""
    out: MutableMapping[str, object] = dict(entry)
    readiness = route_readiness(out.get("route"))
    out.update(readiness)
    return dict(out)
