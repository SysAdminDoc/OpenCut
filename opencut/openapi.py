"""
OpenCut OpenAPI Spec Generator

Introspects the Flask app's url_map to produce an OpenAPI 3.0.x JSON spec.
Schema classes from opencut.schemas are mapped to known endpoints for
response definitions.
"""

from dataclasses import fields as dc_fields
from typing import Any, Dict, List, Optional, get_args, get_origin

from opencut import __version__
from opencut.schemas import (
    AutoZoomResult,
    BeatMarkersResult,
    ChaptersResult,
    ColorMatchResult,
    DeliverableResult,
    ExportMarkersResult,
    IndexResult,
    JobResponse,
    LoudnessMatchResult,
    MulticamResult,
    RepeatDetectResult,
    SearchResult,
    SilenceResult,
    UpdateCheckResult,
)

# ---------------------------------------------------------------------------
# Endpoint -> schema mapping (known response schemas)
# ---------------------------------------------------------------------------
_ENDPOINT_SCHEMAS: Dict[str, type] = {
    "/health": None,                       # ad-hoc dict
    "/system/update-check": UpdateCheckResult,
    "/search/footage": SearchResult,
    "/search/index": IndexResult,
    "/deliverables/vfx-sheet": DeliverableResult,
    "/deliverables/adr-list": DeliverableResult,
    "/deliverables/music-cue-sheet": DeliverableResult,
    "/deliverables/asset-list": DeliverableResult,
    "/timeline/export-from-markers": ExportMarkersResult,
    "/silence": SilenceResult,
    "/audio/loudness-match": LoudnessMatchResult,
    "/audio/beat-markers": BeatMarkersResult,
    "/video/color-match": ColorMatchResult,
    "/video/auto-zoom": AutoZoomResult,
    "/video/multicam-cuts": MulticamResult,
    "/captions/chapters": ChaptersResult,
    "/captions/repeat-detect": RepeatDetectResult,
}

# Endpoints that return a JobResponse (async job-based routes)
_JOB_ENDPOINTS = {
    "/silence", "/search/index", "/timeline/export-from-markers",
    "/install-whisper", "/whisper/reinstall",
    "/video/color-match", "/video/auto-zoom", "/video/multicam-cuts",
    "/audio/loudness-match", "/audio/beat-markers",
    "/captions/chapters", "/captions/repeat-detect",
    "/workflow/run",
}

# Methods to skip (HEAD is auto-added by Flask; OPTIONS for CORS)
_SKIP_METHODS = {"HEAD", "OPTIONS"}


def _python_type_to_json(tp) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = get_origin(tp)

    if tp is str or tp is Optional[str]:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is dict or tp is Dict or origin is dict:
        return {"type": "object"}
    if origin is list or origin is List:
        args = get_args(tp)
        if args:
            return {"type": "array", "items": _python_type_to_json(args[0])}
        return {"type": "array", "items": {}}
    if tp is list:
        return {"type": "array", "items": {}}
    if tp is type(None):
        return {"type": "string", "nullable": True}
    # Optional[X] -> X with nullable
    if origin is type(None) or str(tp).startswith("typing.Optional"):
        args = get_args(tp)
        if args:
            schema = _python_type_to_json(args[0])
            schema["nullable"] = True
            return schema
    return {"type": "string"}


def _dataclass_to_schema(cls) -> dict:
    """Convert a dataclass to an OpenAPI schema object."""
    props = {}
    for f in dc_fields(cls):
        props[f.name] = _python_type_to_json(f.type)
    return {
        "type": "object",
        "properties": props,
    }


def generate_openapi_spec(app) -> dict:
    """Build an OpenAPI 3.0.3 spec dict from a Flask app's url_map."""
    paths: Dict[str, dict] = {}

    for rule in app.url_map.iter_rules():
        path = rule.rule
        # Skip static endpoint
        if path.startswith("/static"):
            continue

        methods = sorted(rule.methods - _SKIP_METHODS)
        if not methods:
            continue

        # Look up the view function for docstring extraction
        view_func = app.view_functions.get(rule.endpoint)
        docstring = (view_func.__doc__ or "").strip() if view_func else ""

        path_item = paths.setdefault(path, {})

        for method in methods:
            operation: Dict[str, Any] = {
                "summary": docstring.split("\n")[0] if docstring else rule.endpoint,
                "operationId": f"{rule.endpoint}_{method.lower()}",
                "tags": [rule.endpoint.split(".")[0] if "." in rule.endpoint else "default"],
                "responses": {},
            }

            if docstring:
                operation["description"] = docstring

            # Build response schema
            schema_cls = _ENDPOINT_SCHEMAS.get(path)
            if schema_cls is not None:
                response_schema = _dataclass_to_schema(schema_cls)
            elif path in _JOB_ENDPOINTS and method == "POST":
                response_schema = _dataclass_to_schema(JobResponse)
            else:
                response_schema = {"type": "object"}

            operation["responses"]["200"] = {
                "description": "Successful response",
                "content": {
                    "application/json": {"schema": response_schema}
                },
            }

            # Add 400/403 for POST endpoints
            if method == "POST":
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
