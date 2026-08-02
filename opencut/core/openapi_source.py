"""The single OpenAPI schema source.

OpenCut used to grow one generator per surface: a 3.0.3 spec at
``/openapi.json`` with dataclass-derived responses, a schema-free 3.1.0
skeleton at ``/api/openapi.json``, and a third adapter under
``/architecture/openapi``. They disagreed about versions, operation ids,
security, request bodies, and error shapes, so a client had to read the source
to find out what an operation actually accepts.

This module builds one 3.1.1 document and everything else adapts from it:

* **Responses** come from the dataclass registry in :mod:`opencut.openapi_registry`
  and are emitted once under ``components/schemas`` and ``$ref``-ed, instead of
  being inlined per operation.
* **Requests** come from the curated MCP tool ``inputSchema`` definitions.
  Those are hand-authored, already reviewed, and already the contract the MCP
  clients use — so REST and MCP describe the same operation the same way
  rather than drifting apart.
* **Errors** come from :mod:`opencut.errors`, which is what the routes actually
  return, so a documented error is a real one.
* ``downgrade_to_30`` produces the 3.0.3 compatibility document served at the
  legacy endpoint. It is an explicit adapter over this source, not a
  second generator.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from opencut import __version__
from opencut.openapi import (
    _JOB_ENDPOINTS,
    _MUTATING_METHODS,
    _SKIP_METHODS,
    _dataclass_to_schema,
    _flask_rule_to_openapi_path,
    _operation_id,
)
from opencut.openapi_registry import build_endpoint_schema_map

OPENAPI_VERSION = "3.1.1"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
DEFAULT_SERVER_URL = "http://127.0.0.1:5679"

#: Response codes every mutating operation can really produce.
_MUTATING_ERROR_RESPONSES = ("400", "403", "429", "500")
_READ_ERROR_RESPONSES = ("400", "429", "500")

_ERROR_RESPONSE_NAMES = {
    "400": "ValidationError",
    "403": "CsrfRejected",
    "404": "NotFound",
    "429": "RateLimited",
    "500": "ServerError",
}

_ERROR_DESCRIPTIONS = {
    "400": "Invalid or missing parameters",
    "403": "Missing or invalid CSRF token",
    "404": "Resource not found",
    "429": "Rate limited or job queue full",
    "500": "Unhandled server error",
}


def error_schema() -> dict:
    """The structured error body every route returns via ``opencut.errors``.

    ``code`` is deliberately an open string. The taxonomy grows with the
    feature surface, and an enum that lags reality would make a valid error
    fail validation — worse than documenting the shape and leaving the value
    open.
    """
    return {
        "type": "object",
        "required": ["error"],
        "properties": {
            "error": {
                "type": "string",
                "description": "Human-readable failure message.",
            },
            "code": {
                "type": "string",
                "pattern": "^[A-Z][A-Z0-9_]*$",
                "description": (
                    "Stable machine-readable error code, e.g. INVALID_INPUT, "
                    "FILE_NOT_FOUND, RATE_LIMITED, QUEUE_FULL."
                ),
            },
            "suggestion": {
                "type": "string",
                "description": "Recovery hint the panels surface to the user.",
            },
        },
        "additionalProperties": True,
    }


# ---------------------------------------------------------------------------
# Request bodies, sourced from the curated MCP tool schemas
# ---------------------------------------------------------------------------
def request_schema_map() -> Dict[tuple, dict]:
    """``{(METHOD, flask_rule): json_schema}`` for curated MCP operations.

    The curated MCP catalogue is the only place in the tree where request
    payloads are described by hand for a large set of routes. Reusing it means
    one schema serves both protocols; a fix in either is a fix in both.
    """
    from opencut import mcp_server

    by_route: Dict[tuple, dict] = {}
    tool_schemas = {tool["name"]: tool.get("inputSchema") for tool in mcp_server.MCP_TOOLS}
    for tool_name, (method, path) in mcp_server._TOOL_ROUTES.items():
        schema = tool_schemas.get(tool_name)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            continue
        if not schema.get("properties"):
            continue
        key = (str(method).upper(), str(path))
        # A route reachable from two tools keeps the first (registry order is
        # stable), rather than silently merging two different contracts.
        by_route.setdefault(key, schema)
    return by_route


def _request_body(schema: dict, tool_name: str) -> dict:
    body = dict(schema)
    body.setdefault("additionalProperties", True)
    return {
        "required": bool(schema.get("required")),
        "description": f"Typed payload shared with the `{tool_name}` MCP tool.",
        "content": {"application/json": {"schema": body}},
    }


def _query_parameters(schema: dict) -> List[dict]:
    """Render an object schema as individual query parameters."""
    required = set(schema.get("required") or [])
    parameters = []
    for name, prop in (schema.get("properties") or {}).items():
        parameters.append({
            "name": name,
            "in": "query",
            "required": name in required,
            "schema": {k: v for k, v in prop.items() if k != "description"},
            "description": prop.get("description", ""),
        })
    return parameters


# ---------------------------------------------------------------------------
# Spec assembly
# ---------------------------------------------------------------------------
def _tag_for(endpoint: str) -> str:
    return endpoint.split(".")[0] if "." in endpoint else "default"


def _components(response_schemas: Dict[str, type]) -> dict:
    schemas: Dict[str, Any] = {"OpenCutError": error_schema()}
    for cls in response_schemas.values():
        name = cls.__name__
        if name not in schemas:
            schemas[name] = _dataclass_to_schema(cls)

    responses = {
        name: {
            "description": _ERROR_DESCRIPTIONS[status],
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/OpenCutError"}
                }
            },
        }
        for status, name in _ERROR_RESPONSE_NAMES.items()
    }

    return {
        "schemas": schemas,
        "responses": responses,
        "securitySchemes": {
            "CSRFToken": {
                "type": "apiKey",
                "in": "header",
                "name": "X-OpenCut-Token",
                "description": "Token issued by GET /health; required on every mutation.",
            }
        },
    }


def _route_uses_csrf(view_func) -> bool:
    """Reuse the existing detector so the two documents agree on security."""
    from opencut.core.openapi_spec import _route_uses_csrf as detect
    return detect(view_func)


def build_spec(app, *, server_url: str = DEFAULT_SERVER_URL) -> dict:
    """Build the canonical OpenAPI 3.1.1 document for *app*."""
    response_schemas = build_endpoint_schema_map()
    request_schemas = request_schema_map()
    tool_names = {
        (str(method).upper(), str(path)): name
        for name, (method, path) in _import_tool_routes().items()
    }

    paths: Dict[str, dict] = {}
    tags: Dict[str, dict] = {}

    for rule in app.url_map.iter_rules():
        raw_path = rule.rule
        if raw_path.startswith("/static"):
            continue
        methods = sorted(rule.methods - _SKIP_METHODS)
        if not methods:
            continue

        path, path_parameters = _flask_rule_to_openapi_path(raw_path)
        view_func = app.view_functions.get(rule.endpoint)
        docstring = (view_func.__doc__ or "").strip() if view_func else ""
        tag = _tag_for(rule.endpoint)
        tags.setdefault(tag, {"name": tag})

        path_item = paths.setdefault(path, {})
        if path_parameters:
            existing = path_item.setdefault("parameters", [])
            known = {param.get("name") for param in existing}
            for parameter in path_parameters:
                if parameter["name"] not in known:
                    existing.append(parameter)

        for method in methods:
            operation: Dict[str, Any] = {
                "summary": docstring.split("\n")[0] if docstring else rule.endpoint,
                "operationId": _operation_id(rule.endpoint, method, path),
                "tags": [tag],
                "responses": {},
            }
            if docstring:
                operation["description"] = docstring

            # --- success -------------------------------------------------
            schema_cls = response_schemas.get(raw_path)
            if schema_cls is not None:
                success = {"$ref": f"#/components/schemas/{schema_cls.__name__}"}
            elif raw_path in _JOB_ENDPOINTS and method == "POST":
                success = {"$ref": "#/components/schemas/JobResponse"}
            else:
                success = {"type": "object"}
            operation["responses"]["200"] = {
                "description": "Successful response",
                "content": {"application/json": {"schema": success}},
            }

            # --- request -------------------------------------------------
            request_key = (method, raw_path)
            request_schema = request_schemas.get(request_key)
            if request_schema is not None:
                if method in _MUTATING_METHODS:
                    operation["requestBody"] = _request_body(
                        request_schema, tool_names.get(request_key, "opencut")
                    )
                else:
                    query_params = _query_parameters(request_schema)
                    if query_params:
                        operation["parameters"] = query_params

            # --- errors --------------------------------------------------
            statuses = (
                _MUTATING_ERROR_RESPONSES
                if method in _MUTATING_METHODS
                else _READ_ERROR_RESPONSES
            )
            for status in statuses:
                operation["responses"][status] = {
                    "$ref": f"#/components/responses/{_ERROR_RESPONSE_NAMES[status]}"
                }

            if method in _MUTATING_METHODS and _route_uses_csrf(view_func):
                operation["security"] = [{"CSRFToken": []}]

            path_item[method.lower()] = operation

    spec: Dict[str, Any] = {
        "openapi": OPENAPI_VERSION,
        "jsonSchemaDialect": JSON_SCHEMA_DIALECT,
        "info": {
            "title": "OpenCut API",
            "description": "Premiere Pro video editing automation backend",
            "version": __version__,
        },
        "paths": paths,
        "tags": [tags[name] for name in sorted(tags)],
        "components": _components(response_schemas),
    }
    if server_url:
        spec["servers"] = [{"url": server_url}]
    return spec


def _import_tool_routes() -> dict:
    from opencut import mcp_server
    return dict(mcp_server._TOOL_ROUTES)


# ---------------------------------------------------------------------------
# 3.0.3 compatibility adapter
# ---------------------------------------------------------------------------
_TYPE_ARRAY_RE = re.compile(r".*")


def _downgrade_schema(node: Any) -> Any:
    """Rewrite 2020-12 constructs 3.0.3 validators reject."""
    if isinstance(node, list):
        return [_downgrade_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$schema", "jsonSchemaDialect", "const"):
            if key == "const":
                # 3.0.3 has no `const`; a single-value enum is the equivalent.
                out["enum"] = [value]
            continue
        if key == "type" and isinstance(value, list):
            # `["string", "null"]` is 3.1 only; 3.0.3 spells it `nullable`.
            non_null = [item for item in value if item != "null"]
            out["type"] = non_null[0] if non_null else "string"
            if len(non_null) != len(value):
                out["nullable"] = True
            continue
        if key in ("exclusiveMinimum", "exclusiveMaximum") and isinstance(value, bool):
            continue
        if key == "prefixItems":
            out["items"] = _downgrade_schema(value[0]) if value else {}
            continue
        out[key] = _downgrade_schema(value)
    return out


def downgrade_to_30(spec: dict) -> dict:
    """Return the 3.0.3 rendering of a canonical 3.1.1 *spec*."""
    downgraded = _downgrade_schema(spec)
    downgraded["openapi"] = "3.0.3"
    downgraded.pop("jsonSchemaDialect", None)
    return downgraded


# ---------------------------------------------------------------------------
# Contract coverage — the ratchet input
# ---------------------------------------------------------------------------
def contract_coverage(spec: Optional[dict] = None, app=None) -> dict:
    """Count typed operations so a regression cannot land unnoticed."""
    if spec is None:
        if app is None:  # pragma: no cover - callers always pass one
            raise ValueError("contract_coverage needs a spec or an app")
        spec = build_spec(app)

    operations = 0
    typed_requests = 0
    typed_responses = 0
    error_typed = 0
    csrf_documented = 0

    for path_item in spec.get("paths", {}).values():
        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            operations += 1
            if operation.get("requestBody") or any(
                param.get("in") == "query" for param in operation.get("parameters", [])
            ):
                typed_requests += 1
            success = (
                operation.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            if "$ref" in success:
                typed_responses += 1
            if all(
                "$ref" in operation.get("responses", {}).get(status, {})
                for status in ("400", "500")
            ):
                error_typed += 1
            if operation.get("security"):
                csrf_documented += 1

    return {
        "openapi": spec.get("openapi", ""),
        "operations": operations,
        "typed_requests": typed_requests,
        "typed_responses": typed_responses,
        "error_typed_operations": error_typed,
        "csrf_documented_operations": csrf_documented,
        "component_schemas": len(spec.get("components", {}).get("schemas", {})),
    }


__all__ = [
    "OPENAPI_VERSION",
    "JSON_SCHEMA_DIALECT",
    "build_spec",
    "downgrade_to_30",
    "contract_coverage",
    "error_schema",
    "request_schema_map",
]
