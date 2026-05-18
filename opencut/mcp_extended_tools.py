"""Auto-generated extended MCP route tools (F194).

The curated MCP catalogue in :mod:`opencut.mcp_server` stays intentionally
small. This module backs the opt-in, lower-priority route-level catalogue
generated from ``opencut/_generated/route_manifest.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote, urlencode

PACKAGE_ROOT = Path(__file__).resolve().parent
ROUTE_MANIFEST_PATH = PACKAGE_ROOT / "_generated" / "route_manifest.json"
API_ALIAS_MANIFEST_PATH = PACKAGE_ROOT / "_generated" / "api_aliases.json"
EXTENDED_MANIFEST_PATH = PACKAGE_ROOT / "_generated" / "mcp_extended_tools.json"

EXTENDED_MCP_ENV = "OPENCUT_MCP_EXTENDED_TOOLS"
EXTENDED_TOOL_PREFIX = "opencut_route_"
EXTENDED_MANIFEST_VERSION = 1
LOWER_PRIORITY_NOTE = (
    "Auto-generated lower-priority route tool. Prefer a curated opencut_* "
    "tool when one exists for the same workflow."
)

_FLASK_PARAM_RE = re.compile(
    r"<(?:(?P<converter>[A-Za-z_][A-Za-z0-9_]*):)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)>"
)
_NON_NAME_RE = re.compile(r"[^0-9A-Za-z]+")
_TRUE_VALUES = {"1", "true", "yes", "on", "extended"}
_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_CONVERTER_SCHEMA_TYPES = {
    "int": "integer",
    "float": "number",
}


@dataclass(frozen=True)
class ExtendedRouteTool:
    name: str
    method: str
    rule: str
    endpoint: str
    blueprint: str
    path_params: tuple[dict, ...] = field(default_factory=tuple)
    response_schema: str = ""

    def as_dict(self) -> dict:
        data = asdict(self)
        data["path_params"] = [dict(param) for param in self.path_params]
        return data


def extended_tools_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True when the generated route-level MCP tools are enabled."""

    if env is None:
        env = os.environ
    return str(env.get(EXTENDED_MCP_ENV, "")).strip().lower() in _TRUE_VALUES


def load_route_manifest(path: Path = ROUTE_MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_alias_manifest(path: Path = API_ALIAS_MANIFEST_PATH) -> dict:
    if not path.is_file():
        return {"aliases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _response_schema_lookup() -> dict[str, str]:
    try:
        from opencut.openapi_registry import build_endpoint_schema_map
    except Exception:
        return {}
    return {
        rule: schema.__name__
        for rule, schema in build_endpoint_schema_map().items()
    }


def _alias_rules_to_skip(api_alias_manifest: Mapping[str, object]) -> set[str]:
    aliases = api_alias_manifest.get("aliases", [])
    if not isinstance(aliases, list):
        return set()
    return {
        str(alias.get("alias_rule"))
        for alias in aliases
        if isinstance(alias, dict) and alias.get("alias_rule")
    }


def _path_params(rule: str) -> tuple[dict, ...]:
    params: list[dict] = []
    for match in _FLASK_PARAM_RE.finditer(rule):
        params.append(
            {
                "name": match.group("name"),
                "converter": (match.group("converter") or "string").lower(),
            }
        )
    return tuple(params)


def _slug_for_route(method: str, rule: str) -> str:
    def replace_param(match: re.Match) -> str:
        return f"by_{match.group('name')}"

    raw = _FLASK_PARAM_RE.sub(replace_param, rule.strip("/") or "root")
    slug = _NON_NAME_RE.sub("_", raw).strip("_").lower()
    return f"{method.lower()}_{slug or 'root'}"[:180].strip("_")


def _tool_name(method: str, rule: str) -> str:
    return EXTENDED_TOOL_PREFIX + _slug_for_route(method, rule)


def _iter_route_tools(
    route_manifest: Mapping[str, object],
    *,
    excluded_route_keys: Iterable[tuple[str, str]] = (),
    api_alias_manifest: Mapping[str, object] | None = None,
    response_schema_rules: Mapping[str, str] | None = None,
) -> list[ExtendedRouteTool]:
    excluded = {(method.upper(), rule) for method, rule in excluded_route_keys}
    alias_rules = _alias_rules_to_skip(api_alias_manifest or {})
    response_schemas = dict(response_schema_rules or {})
    seen_names: set[str] = set()
    tools: list[ExtendedRouteTool] = []

    routes = route_manifest.get("routes", [])
    if not isinstance(routes, list):
        return []

    for route in routes:
        if not isinstance(route, dict):
            continue
        rule = str(route.get("rule", ""))
        if not rule or rule.startswith("/static/") or rule in alias_rules:
            continue
        methods = route.get("methods", [])
        if not isinstance(methods, list):
            continue
        for method_raw in methods:
            method = str(method_raw).upper()
            if method not in _METHODS or (method, rule) in excluded:
                continue
            name = _tool_name(method, rule)
            if name in seen_names:
                endpoint_slug = _NON_NAME_RE.sub("_", str(route.get("endpoint", ""))).strip("_").lower()
                name = f"{name}_{endpoint_slug}"[:220].strip("_")
            seen_names.add(name)
            tools.append(
                ExtendedRouteTool(
                    name=name,
                    method=method,
                    rule=rule,
                    endpoint=str(route.get("endpoint", "")),
                    blueprint=str(route.get("blueprint", "")),
                    path_params=_path_params(rule),
                    response_schema=response_schemas.get(rule, ""),
                )
            )
    return sorted(tools, key=lambda tool: tool.name)


def tool_definition(entry: ExtendedRouteTool) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for param in entry.path_params:
        name = str(param["name"])
        converter = str(param.get("converter") or "string")
        properties[name] = {
            "type": _CONVERTER_SCHEMA_TYPES.get(converter, "string"),
            "description": f"Path parameter `{name}` for `{entry.rule}`.",
        }
        required.append(name)

    if entry.method == "GET":
        properties["query"] = {
            "type": "object",
            "description": "Optional query-string parameters to append to the route.",
            "additionalProperties": True,
        }
    else:
        properties["body"] = {
            "type": "object",
            "description": "Optional JSON request body for the backend route.",
            "additionalProperties": True,
        }

    description = f"{LOWER_PRIORITY_NOTE} Route: {entry.method} {entry.rule}."
    if entry.response_schema:
        description += f" Response schema: {entry.response_schema}."

    return {
        "name": entry.name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "annotations": {
            "title": f"{entry.method} {entry.rule}",
            "readOnlyHint": entry.method == "GET",
            "destructiveHint": entry.method == "DELETE",
        },
        "metadata": {
            "generated": True,
            "priority": "extended",
            "method": entry.method,
            "path": entry.rule,
            "endpoint": entry.endpoint,
            "blueprint": entry.blueprint,
            "path_params": [dict(param) for param in entry.path_params],
            "response_schema": entry.response_schema,
        },
    }


def build_manifest(
    *,
    route_manifest: Mapping[str, object] | None = None,
    api_alias_manifest: Mapping[str, object] | None = None,
    excluded_route_keys: Iterable[tuple[str, str]] = (),
    response_schema_rules: Mapping[str, str] | None = None,
) -> dict:
    """Build the committed extended MCP tool manifest."""

    route_manifest = route_manifest or load_route_manifest()
    api_alias_manifest = api_alias_manifest or load_api_alias_manifest()
    response_schema_rules = response_schema_rules or _response_schema_lookup()
    entries = _iter_route_tools(
        route_manifest,
        excluded_route_keys=excluded_route_keys,
        api_alias_manifest=api_alias_manifest,
        response_schema_rules=response_schema_rules,
    )
    tools = [tool_definition(entry) for entry in entries]
    by_method: dict[str, int] = {}
    for entry in entries:
        by_method[entry.method] = by_method.get(entry.method, 0) + 1
    typed = sum(1 for entry in entries if entry.response_schema)
    return {
        "manifest_version": EXTENDED_MANIFEST_VERSION,
        "source": "opencut/_generated/route_manifest.json",
        "api_alias_source": "opencut/_generated/api_aliases.json",
        "enabled_by": f"{EXTENDED_MCP_ENV}=1 or opencut-mcp-server --extended-tools",
        "tool_prefix": EXTENDED_TOOL_PREFIX,
        "tool_count": len(tools),
        "response_schema_count": typed,
        "method_counts": dict(sorted(by_method.items())),
        "tools": tools,
    }


def load_extended_manifest(path: Path = EXTENDED_MANIFEST_PATH) -> dict:
    if not path.is_file():
        return build_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def load_extended_tools(path: Path = EXTENDED_MANIFEST_PATH) -> list[dict]:
    manifest = load_extended_manifest(path)
    tools = manifest.get("tools", [])
    return list(tools) if isinstance(tools, list) else []


def extended_tool_routes(tools: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    routes: dict[str, dict] = {}
    for tool in tools:
        name = str(tool.get("name", ""))
        metadata = tool.get("metadata", {})
        if name and isinstance(metadata, dict) and metadata.get("generated"):
            routes[name] = dict(metadata)
    return routes


_DEFAULT_EXTENDED_TOOLS = load_extended_tools()
_DEFAULT_EXTENDED_ROUTES = extended_tool_routes(_DEFAULT_EXTENDED_TOOLS)


def get_extended_tools() -> list[dict]:
    return list(_DEFAULT_EXTENDED_TOOLS)


def is_extended_tool(tool_name: str) -> bool:
    return tool_name in _DEFAULT_EXTENDED_ROUTES


def _render_path(rule: str, arguments: Mapping[str, object], path_params: Sequence[Mapping[str, object]]) -> str:
    values: dict[str, str] = {}
    for param in path_params:
        name = str(param.get("name", ""))
        value = arguments.get(name)
        if value is None or value == "":
            raise ValueError(f"Missing path parameter `{name}`")
        values[name] = quote(str(value), safe="")

    def replace(match: re.Match) -> str:
        return values[match.group("name")]

    return _FLASK_PARAM_RE.sub(replace, rule)


def _remaining_arguments(
    arguments: Mapping[str, object],
    path_params: Sequence[Mapping[str, object]],
) -> dict:
    reserved = {"body", "query"} | {str(param.get("name", "")) for param in path_params}
    return {key: value for key, value in arguments.items() if key not in reserved}


def invoke_extended_tool(
    tool_name: str,
    arguments: Mapping[str, object],
    api_call: Callable[[str, str, object | None], object],
) -> object:
    """Dispatch an opt-in generated route tool to the Flask backend."""

    route = _DEFAULT_EXTENDED_ROUTES.get(tool_name)
    if route is None:
        return {"error": f"Unknown extended MCP tool: {tool_name}"}
    if not isinstance(arguments, dict):
        return {"error": "`arguments` must be a JSON object"}

    method = str(route["method"])
    rule = str(route["path"])
    path_params = route.get("path_params", [])
    try:
        path = _render_path(rule, arguments, path_params if isinstance(path_params, list) else [])
    except ValueError as exc:
        return {"error": str(exc)}

    extra = _remaining_arguments(arguments, path_params if isinstance(path_params, list) else [])
    if method == "GET":
        query = arguments.get("query", {})
        if query is None:
            query = {}
        if not isinstance(query, dict):
            return {"error": "`query` must be a JSON object"}
        query_payload = {**query, **extra}
        if query_payload:
            path = f"{path}?{urlencode(query_payload, doseq=True)}"
        return api_call(method, path, None)

    body = arguments.get("body")
    if body is None:
        body = extra or None
    elif not isinstance(body, dict):
        return {"error": "`body` must be a JSON object"}
    elif extra:
        body = {**body, **extra}
    return api_call(method, path, body)
