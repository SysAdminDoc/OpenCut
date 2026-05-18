"""
OpenCut OpenAPI Spec Generator

Introspects the Flask app's url_map to produce an OpenAPI 3.0.x JSON spec.
Response schemas are discovered from dataclasses registered through
``opencut.openapi_registry``.
"""

import re
import types
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

from opencut import __version__
from opencut.openapi_registry import (
    build_endpoint_schema_map,
    schema_extra_properties,
)
from opencut.schemas import JobResponse

# ---------------------------------------------------------------------------
# Endpoint -> schema mapping (discovered dataclass response schemas)
# ---------------------------------------------------------------------------
_ENDPOINT_SCHEMAS: Dict[str, type] = build_endpoint_schema_map()

# Endpoints that return a JobResponse (async job-based routes)
_JOB_ENDPOINTS = {
    "/silence", "/search/index", "/timeline/export-from-markers",
    "/install-whisper", "/whisper/reinstall",
    "/video/color-match", "/video/auto-zoom", "/video/multicam-cuts",
    "/audio/loudness-match", "/audio/beat-markers",
    "/captions/chapters", "/captions/repeat-detect",
    "/workflow/run",
    "/video/ai/upscale", "/video/ai/rembg", "/video/ai/interpolate",
    "/video/ai/denoise", "/video/shorts-pipeline",
    "/video/depth/map", "/video/depth/bokeh", "/video/depth/parallax",
    "/video/broll-plan",
    "/video/face/blur", "/video/face/enhance", "/video/face/swap",
    "/video/style/apply", "/video/style/arbitrary",
    "/video/remove/watermark", "/video/upscale/run",
    "/audio/denoise", "/audio/normalize", "/audio/enhance",
    "/audio/separate", "/audio/tts/generate",
}

# Methods to skip (HEAD is auto-added by Flask; OPTIONS for CORS)
_SKIP_METHODS = {"HEAD", "OPTIONS"}
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

_FLASK_PATH_PARAM_RE = re.compile(
    r"<(?:(?P<converter>[A-Za-z_][A-Za-z0-9_]*):)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)>"
)
_OPENAPI_OPERATION_ID_RE = re.compile(r"[^0-9A-Za-z_]+")
_FLASK_CONVERTER_TYPES = {
    "string": "string",
    "path": "string",
    "uuid": "string",
    "int": "integer",
    "float": "number",
}


def _flask_rule_to_openapi_path(rule: str) -> tuple[str, List[dict]]:
    """Convert Flask path converter syntax to OpenAPI path syntax."""
    parameters: List[dict] = []

    def _replace(match: re.Match) -> str:
        converter = (match.group("converter") or "string").lower()
        name = match.group("name")
        parameters.append({
            "name": name,
            "in": "path",
            "required": True,
            "schema": {
                "type": _FLASK_CONVERTER_TYPES.get(converter, "string"),
            },
            "description": f"Flask converter: {converter}",
        })
        return "{" + name + "}"

    return _FLASK_PATH_PARAM_RE.sub(_replace, rule), parameters


def _operation_id(endpoint: str, method: str, openapi_path: str) -> str:
    """Build a stable OpenAPI operationId that remains unique for aliases."""
    path_part = openapi_path.strip("/") or "root"
    raw = f"{endpoint}_{method.lower()}_{path_part}"
    safe = _OPENAPI_OPERATION_ID_RE.sub("_", raw)
    return re.sub(r"_+", "_", safe).strip("_")


def _python_type_to_json(tp, _seen: set[type] | None = None) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(tp)
    args = get_args(tp)
    seen = _seen or set()

    if tp is Any:
        return {"type": "object", "additionalProperties": True}
    if tp is str or tp is Path:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is type(None):
        return {"type": "string", "nullable": True}
    if is_dataclass(tp):
        return _dataclass_to_schema(tp, seen)
    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and len(non_none_args) != len(args):
            schema = dict(_python_type_to_json(non_none_args[0], seen))
            schema["nullable"] = True
            return schema
        if non_none_args:
            return {
                "oneOf": [_python_type_to_json(arg, seen) for arg in non_none_args],
            }
    if tp is dict or tp is Dict or origin is dict:
        return {"type": "object"}
    if origin in (list, List, tuple):
        if args and args[0] is not Ellipsis:
            return {"type": "array", "items": _python_type_to_json(args[0], seen)}
        return {"type": "array", "items": {}}
    if tp is list:
        return {"type": "array", "items": {}}
    return {"type": "string"}


def _dataclass_to_schema(cls, _seen: set[type] | None = None) -> dict:
    """Convert a dataclass to an OpenAPI schema object."""
    seen = set(_seen or set())
    if cls in seen:
        return {"type": "object"}
    seen.add(cls)

    props = {}
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}
    for f in dc_fields(cls):
        props[f.name] = _python_type_to_json(type_hints.get(f.name, f.type), seen)
    props.update(schema_extra_properties(cls))
    return {
        "type": "object",
        "properties": props,
    }


def generate_openapi_spec(app) -> dict:
    """Build an OpenAPI 3.0.3 spec dict from a Flask app's url_map."""
    paths: Dict[str, dict] = {}

    for rule in app.url_map.iter_rules():
        raw_path = rule.rule
        path, path_parameters = _flask_rule_to_openapi_path(raw_path)
        # Skip static endpoint
        if raw_path.startswith("/static"):
            continue

        methods = sorted(rule.methods - _SKIP_METHODS)
        if not methods:
            continue

        # Look up the view function for docstring extraction
        view_func = app.view_functions.get(rule.endpoint)
        docstring = (view_func.__doc__ or "").strip() if view_func else ""

        path_item = paths.setdefault(path, {})
        if path_parameters:
            existing_parameters = path_item.setdefault("parameters", [])
            existing_names = {param.get("name") for param in existing_parameters}
            for parameter in path_parameters:
                if parameter["name"] not in existing_names:
                    existing_parameters.append(parameter)

        for method in methods:
            operation: Dict[str, Any] = {
                "summary": docstring.split("\n")[0] if docstring else rule.endpoint,
                "operationId": _operation_id(rule.endpoint, method, path),
                "tags": [rule.endpoint.split(".")[0] if "." in rule.endpoint else "default"],
                "responses": {},
            }

            if docstring:
                operation["description"] = docstring

            # Build response schema
            schema_cls = _ENDPOINT_SCHEMAS.get(raw_path)
            if schema_cls is not None:
                response_schema = _dataclass_to_schema(schema_cls)
            elif raw_path in _JOB_ENDPOINTS and method == "POST":
                response_schema = _dataclass_to_schema(JobResponse)
            else:
                response_schema = {"type": "object"}

            operation["responses"]["200"] = {
                "description": "Successful response",
                "content": {
                    "application/json": {"schema": response_schema}
                },
            }

            # Add common mutating-route error responses. GET endpoints can still
            # define their own richer errors later through typed schemas.
            if method in _MUTATING_METHODS:
                operation["responses"]["400"] = {
                    "description": "Validation error",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"error": {"type": "string"}},
                            }
                        }
                    },
                }
                operation["responses"]["403"] = {
                    "description": "Missing or invalid CSRF token",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"error": {"type": "string"}},
                            }
                        }
                    },
                }

            path_item[method.lower()] = operation

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "OpenCut API",
            "description": "Premiere Pro video editing automation backend",
            "version": __version__,
        },
        "servers": [{"url": "http://127.0.0.1:5679"}],
        "paths": paths,
    }
