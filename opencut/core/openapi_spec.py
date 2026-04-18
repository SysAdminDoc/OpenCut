"""
OpenAPI 3.1 spec generator — walks Flask's ``url_map``.

A first-class machine-readable API catalogue for OpenCut's 1,200+
routes.  Unlocks:

- third-party / plugin-developer discoverability (``GET /api/openapi.json``),
- LLM-agent orchestration (Claude / GPT / Gemini clients consume
  OpenAPI natively),
- API clients auto-generated via ``openapi-generator``, ``orval``,
  etc.,
- Swagger / ReDoc UIs hosted at ``/api/docs``.

Design
------
- **Pure stdlib** — no ``flask-smorest`` / ``apispec`` dependency.
  Walks ``app.url_map`` + introspects Flask's view functions /
  docstrings.
- **Adapters-free** — the spec carries only what we can derive from
  route metadata. Rich request/response schemas can be added
  incrementally without changing this module — callers providing a
  ``schema_hints`` dict supplement the auto-derived skeleton.
- **OpenAPI 3.1 compatible** — the spec renders in Swagger UI,
  ReDoc, and Scalar without adjustment.

Gotchas
-------
- **HEAD / OPTIONS methods are stripped** from the spec so the
  catalogue stays clean. Flask adds them automatically to GET routes.
- **CSRF-protected routes** are flagged via the ``X-OpenCut-Token``
  header security scheme. Non-mutation routes skip it.
- **Path parameters** are converted from Flask's ``<int:foo>`` /
  ``<foo>`` syntax to OpenAPI's ``{foo}``.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# Methods we document. HEAD / OPTIONS are implementation noise.
_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

# Flask path-converter syntax → OpenAPI type
_CONVERTER_TYPES = {
    "string": "string",
    "int": "integer",
    "float": "number",
    "path": "string",
    "uuid": "string",
    # Custom converters fall back to string
}

_FLASK_PATH_PARAM_RE = re.compile(r"<(?:(?P<conv>[a-zA-Z_]+):)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)>")


def _flask_to_openapi_path(rule: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert a Flask rule to (openapi_path, parameter_objects)."""
    params: List[Dict[str, Any]] = []
    out_segments: List[str] = []
    last_end = 0
    for m in _FLASK_PATH_PARAM_RE.finditer(rule):
        out_segments.append(rule[last_end:m.start()])
        name = m.group("name")
        conv = (m.group("conv") or "string").lower()
        ptype = _CONVERTER_TYPES.get(conv, "string")
        out_segments.append("{" + name + "}")
        params.append({
            "name": name,
            "in": "path",
            "required": True,
            "schema": {"type": ptype},
            "description": f"Flask converter: {conv}",
        })
        last_end = m.end()
    out_segments.append(rule[last_end:])
    return "".join(out_segments), params


def _docstring_summary(view_func) -> Tuple[str, str]:
    """Extract ``(summary, description)`` from a view's docstring.

    Summary is the first line; description is the body (if any).
    """
    doc = (inspect.getdoc(view_func) or "").strip()
    if not doc:
        return "", ""
    lines = doc.splitlines()
    summary = lines[0].strip().rstrip(".")[:120]
    description = "\n".join(lines[1:]).strip()
    return summary, description


def _route_uses_csrf(view_func) -> bool:
    """Detect ``@require_csrf`` wrapping by walking closure cells.

    ``@require_csrf`` sets a ``__wrapped__`` chain; if we find the
    marker name anywhere in the chain the route mutates state and
    requires the ``X-OpenCut-Token`` header.
    """
    seen = set()
    cur = view_func
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = getattr(cur, "__name__", "") or ""
        qual = getattr(cur, "__qualname__", "") or ""
        if "require_csrf" in name or "require_csrf" in qual:
            return True
        cur = getattr(cur, "__wrapped__", None)
    return False


def _infer_tag(blueprint_name: str, rule: str) -> str:
    """Derive an OpenAPI tag from blueprint name or URL prefix."""
    if blueprint_name and blueprint_name not in ("static",):
        return blueprint_name
    # Fall back to the first URL segment
    seg = rule.strip("/").split("/", 1)[0] or "root"
    return seg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_spec(
    app,
    *,
    title: str = "OpenCut API",
    version: Optional[str] = None,
    description: str = "",
    server_url: str = "",
    schema_hints: Optional[Dict[str, Dict[str, Any]]] = None,
    include_deprecated: bool = True,
) -> Dict[str, Any]:
    """Build an OpenAPI 3.1 spec dict from a Flask app.

    Args:
        app: Configured Flask application (after ``register_blueprints``).
        title: Spec title displayed in Swagger UI.
        version: Spec version. Defaults to ``opencut.__version__``.
        description: Optional Markdown description shown at the top of
            Swagger UI.
        server_url: Base URL advertised in the spec (e.g.
            ``"http://localhost:5679"``). Empty string → omit.
        schema_hints: Per-endpoint override dict keyed by
            ``"{METHOD} {path}"``. Each entry is a partial OpenAPI
            operation object that gets shallow-merged over the
            auto-derived skeleton.  Use this to attach requestBody /
            response schemas without editing every route.
        include_deprecated: Keep deprecated routes in the spec.

    Returns:
        Dict — ready to ``json.dumps`` and serve.
    """
    if version is None:
        try:
            from opencut import __version__ as _ver
            version = _ver
        except Exception:  # noqa: BLE001
            version = "0.0.0"

    paths: Dict[str, Dict[str, Any]] = {}
    tag_set: Dict[str, Dict[str, str]] = {}
    hints = schema_hints or {}

    for rule in app.url_map.iter_rules():
        methods = set(rule.methods or set()) & _HTTP_METHODS
        if not methods:
            continue
        openapi_path, path_params = _flask_to_openapi_path(str(rule.rule))
        view_func = app.view_functions.get(rule.endpoint)
        if view_func is None:
            continue

        summary, long_desc = _docstring_summary(view_func)
        is_mutating = bool(methods & {"POST", "PUT", "PATCH", "DELETE"})
        requires_csrf = is_mutating and _route_uses_csrf(view_func)

        bp_name = (rule.endpoint.rsplit(".", 1)[0]
                   if "." in rule.endpoint else "")
        tag = _infer_tag(bp_name, str(rule.rule))
        tag_set.setdefault(tag, {"name": tag})

        path_entry = paths.setdefault(openapi_path, {})
        if path_params:
            # PathItem-level parameters (apply to every method on this path)
            existing = path_entry.setdefault("parameters", [])
            seen_names = {p.get("name") for p in existing}
            for p in path_params:
                if p["name"] not in seen_names:
                    existing.append(p)

        for method in sorted(methods):
            op: Dict[str, Any] = {
                "summary": summary,
                "description": long_desc,
                "tags": [tag],
                "operationId": f"{method.lower()}_{rule.endpoint}".replace(".", "_"),
                "responses": {
                    "200": {"description": "Success"},
                    "400": {"description": "Invalid input"},
                    "429": {"description": "Rate limited / too many jobs"},
                    "500": {"description": "Internal error"},
                },
            }
            if requires_csrf:
                op["security"] = [{"CSRFToken": []}]
                op["parameters"] = op.get("parameters", []) + [{
                    "name": "X-OpenCut-Token",
                    "in": "header",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "CSRF token from `GET /health`.",
                }]
            if is_mutating:
                op["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"},
                        },
                    },
                }
            # Merge caller-supplied hints (method-path keyed)
            hint_key = f"{method} {openapi_path}"
            if hint_key in hints:
                # Shallow merge — caller wins on conflicting keys
                for k, v in hints[hint_key].items():
                    op[k] = v

            if not include_deprecated and op.get("deprecated"):
                continue

            path_entry[method.lower()] = op

    spec: Dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {
            "title": title,
            "version": str(version),
            "description": description or (
                "Auto-generated catalogue of OpenCut routes. "
                "Descriptions are extracted from view-function docstrings."
            ),
        },
        "paths": paths,
        "tags": sorted(tag_set.values(), key=lambda t: t["name"]),
        "components": {
            "securitySchemes": {
                "CSRFToken": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-OpenCut-Token",
                    "description": (
                        "CSRF token returned from ``GET /health``. "
                        "Sent as a header on every mutating request."
                    ),
                },
            },
        },
    }
    if server_url:
        spec["servers"] = [{"url": server_url.rstrip("/")}]
    return spec


# ---------------------------------------------------------------------------
# Swagger UI — inlined HTML
# ---------------------------------------------------------------------------

_SWAGGER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>html,body{{margin:0;padding:0;background:#101014;}}</style>
</head>
<body>
<div id="swagger"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
window.onload = function() {{
    window.ui = SwaggerUIBundle({{
        url: "{spec_url}",
        dom_id: "#swagger",
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        layout: "BaseLayout",
        docExpansion: "none",
        tryItOutEnabled: false
    }});
}};
</script>
</body>
</html>
"""


def swagger_ui_html(spec_url: str, title: str = "OpenCut API") -> str:
    """Render a self-contained Swagger UI HTML page for ``spec_url``."""
    return _SWAGGER_TEMPLATE.format(
        title=title.replace("<", "&lt;")[:80],
        spec_url=spec_url,
    )
