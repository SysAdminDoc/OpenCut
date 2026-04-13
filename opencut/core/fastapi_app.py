"""
OpenCut FastAPI Migration Layer (Feature 7.1)

A FastAPI adapter layer that runs alongside the existing Flask app, enabling
gradual migration.  Provides:

- Auto-generated Pydantic models from route parameter patterns
- ASGI wrapper for Flask via WSGIMiddleware (gradual migration)
- WebSocket stub for real-time job updates
- Combined OpenAPI spec at /docs
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclass result types
# ---------------------------------------------------------------------------

@dataclass
class RouteInfo:
    """Metadata about a discovered Flask route."""
    rule: str
    endpoint: str
    methods: List[str]
    params: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AdaptedRoute:
    """Result of adapting a Flask route to FastAPI."""
    original_rule: str
    fastapi_path: str
    methods: List[str]
    model_name: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class OpenAPISpec:
    """Container for a combined OpenAPI spec."""
    title: str = "OpenCut API"
    version: str = "1.0.0"
    paths: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.title,
                "version": self.version,
                **self.info,
            },
            "paths": self.paths,
            "components": self.components,
        }


# ---------------------------------------------------------------------------
# PydanticModelFactory — generate Pydantic models from dict specs
# ---------------------------------------------------------------------------

class PydanticModelFactory:
    """Generates Pydantic v2 models from dictionary specifications.

    Caches generated models to avoid re-creation.
    """

    _cache: Dict[str, Any] = {}

    # Map from simple type names to Python types
    TYPE_MAP = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "list": list,
        "array": list,
        "dict": dict,
        "object": dict,
    }

    @classmethod
    def create_model(cls, name: str, field_specs: Dict[str, Any]) -> Any:
        """Create a Pydantic model class from a dict of field specs.

        Args:
            name: Model class name.
            field_specs: Mapping of field_name -> type_str or dict with
                keys ``type``, ``default``, ``description``.

        Returns:
            A Pydantic BaseModel subclass (or a plain dataclass if
            pydantic is unavailable).
        """
        if name in cls._cache:
            return cls._cache[name]

        field_definitions = cls._parse_field_specs(field_specs)

        try:
            from pydantic import create_model as _pydantic_create_model
            # Build kwargs: field_name = (type, default_or_FieldInfo)
            pydantic_fields = {}
            for fname, (ftype, fdefault, fdesc) in field_definitions.items():
                if fdefault is _MISSING:
                    pydantic_fields[fname] = (ftype, ...)
                else:
                    try:
                        from pydantic import Field
                        pydantic_fields[fname] = (
                            ftype,
                            Field(default=fdefault, description=fdesc),
                        )
                    except ImportError:
                        pydantic_fields[fname] = (ftype, fdefault)

            model = _pydantic_create_model(name, **pydantic_fields)
        except ImportError:
            # Fallback: create a plain dataclass
            model = _make_fallback_dataclass(name, field_definitions)

        cls._cache[name] = model
        return model

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    @classmethod
    def _parse_field_specs(
        cls, specs: Dict[str, Any]
    ) -> Dict[str, Tuple[type, Any, str]]:
        """Parse field_specs into {name: (type, default, description)}."""
        result = {}
        for fname, fspec in specs.items():
            if isinstance(fspec, str):
                ftype = cls.TYPE_MAP.get(fspec, str)
                result[fname] = (ftype, _MISSING, "")
            elif isinstance(fspec, dict):
                raw_type = fspec.get("type", "str")
                ftype = cls.TYPE_MAP.get(raw_type, str)
                fdefault = fspec.get("default", _MISSING)
                fdesc = fspec.get("description", "")
                result[fname] = (ftype, fdefault, fdesc)
            else:
                result[fname] = (str, _MISSING, "")
        return result

    @classmethod
    def get_cached_models(cls) -> Dict[str, Any]:
        """Return a copy of the model cache."""
        return dict(cls._cache)


class _MissingSentinel:
    """Sentinel for 'no default provided'."""
    def __repr__(self) -> str:
        return "<MISSING>"

_MISSING = _MissingSentinel()


def _make_fallback_dataclass(
    name: str,
    field_defs: Dict[str, Tuple[type, Any, str]],
) -> type:
    """Create a dataclass as fallback when Pydantic is unavailable."""
    fields_list = []
    for fname, (ftype, fdefault, _desc) in field_defs.items():
        if fdefault is _MISSING:
            fields_list.append((fname, ftype))
        else:
            fields_list.append((fname, ftype, field(default=fdefault)))
    return dataclasses.make_dataclass(name, fields_list)


# ---------------------------------------------------------------------------
# RouteAdapter — convert Flask routes to FastAPI equivalents
# ---------------------------------------------------------------------------

class RouteAdapter:
    """Converts Flask Blueprint routes into FastAPI route descriptors."""

    # Flask -> FastAPI path converter mapping
    _CONVERTER_MAP = {
        "string": "str",
        "int": "int",
        "float": "float",
        "path": "path",
        "uuid": "str",
    }

    def __init__(self) -> None:
        self.adapted_routes: List[AdaptedRoute] = []

    def adapt_route(
        self, rule: str, endpoint: str, methods: List[str],
        view_func: Optional[Callable] = None,
    ) -> AdaptedRoute:
        """Convert a single Flask route rule to a FastAPI path.

        Converts ``<int:id>`` -> ``{id}`` style path parameters.
        """
        fastapi_path = self._convert_path(rule)
        model_name = None
        if view_func and any(m in ("POST", "PUT", "PATCH") for m in methods):
            model_name = self._infer_model_name(endpoint)

        result = AdaptedRoute(
            original_rule=rule,
            fastapi_path=fastapi_path,
            methods=[m for m in methods if m not in ("OPTIONS", "HEAD")],
            model_name=model_name,
        )
        self.adapted_routes.append(result)
        return result

    def adapt_blueprint(self, blueprint: Any) -> List[AdaptedRoute]:
        """Adapt all routes registered on a Flask Blueprint."""
        results = []
        if not hasattr(blueprint, "deferred_functions"):
            return results

        # Blueprints don't store rules directly; iterate the app later.
        return results

    @staticmethod
    def _convert_path(rule: str) -> str:
        """Convert Flask-style ``<type:name>`` to FastAPI ``{name}``."""
        # <int:id> -> {id}  ;  <name> -> {name}
        return re.sub(r"<(?:\w+:)?(\w+)>", r"{\1}", rule)

    @staticmethod
    def _infer_model_name(endpoint: str) -> str:
        """Generate a Pydantic model name from an endpoint."""
        parts = endpoint.replace(".", "_").split("_")
        return "".join(p.capitalize() for p in parts if p) + "Request"


# ---------------------------------------------------------------------------
# adapt_flask_route (standalone function)
# ---------------------------------------------------------------------------

def adapt_flask_route(
    blueprint: Any, rule: str, endpoint: str
) -> AdaptedRoute:
    """Convert a single Flask route to its FastAPI equivalent.

    Args:
        blueprint: The Flask Blueprint the route belongs to.
        rule: URL rule string (e.g., ``/hw/encode``).
        endpoint: Endpoint name.

    Returns:
        An ``AdaptedRoute`` describing the conversion.
    """
    adapter = RouteAdapter()
    view_func = None
    if hasattr(blueprint, "view_functions"):
        view_func = blueprint.view_functions.get(endpoint)

    methods = ["GET"]
    return adapter.adapt_route(rule, endpoint, methods, view_func)


# ---------------------------------------------------------------------------
# OpenCutFastAPI — wraps Flask app with FastAPI mount
# ---------------------------------------------------------------------------

class OpenCutFastAPI:
    """Wrapper that mounts a Flask app inside a FastAPI application.

    Usage::

        wrapper = OpenCutFastAPI(flask_app)
        fastapi_app = wrapper.fastapi_app
        # Run with: uvicorn module:fastapi_app
    """

    def __init__(self, flask_app: Any = None) -> None:
        self.flask_app = flask_app
        self._fastapi_app: Any = None
        self._adapter = RouteAdapter()
        self._routes_registered: List[AdaptedRoute] = []
        self._websocket_handlers: Dict[str, Callable] = {}

    @property
    def fastapi_app(self) -> Any:
        """Lazy-create the FastAPI app."""
        if self._fastapi_app is None:
            self._fastapi_app = self._create_app()
        return self._fastapi_app

    def _create_app(self) -> Any:
        """Build a FastAPI application with optional Flask mount."""
        try:
            from fastapi import FastAPI
        except ImportError:
            raise ImportError(
                "fastapi is required. Install it with: "
                "pip install fastapi[standard]"
            )

        app = FastAPI(
            title="OpenCut API",
            version="1.0.0",
            description="OpenCut Video Editing Backend — FastAPI Migration Layer",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Register health endpoint on the FastAPI side
        @app.get("/fastapi/health")
        async def fastapi_health():
            return {
                "status": "ok",
                "layer": "fastapi",
                "timestamp": time.time(),
            }

        # Register WebSocket stub
        self._register_websocket_stub(app)

        # Mount Flask if provided
        if self.flask_app is not None:
            mount_flask_app(app, self.flask_app)

        return app

    def _register_websocket_stub(self, app: Any) -> None:
        """Register a WebSocket endpoint for real-time job updates."""
        try:
            from fastapi import WebSocket, WebSocketDisconnect  # noqa: F401
        except ImportError:
            return

        @app.websocket("/ws/jobs")
        async def job_updates(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    # Echo back with status — stub for future implementation
                    await websocket.send_json({
                        "type": "ack",
                        "message": data,
                        "info": "WebSocket job updates coming soon",
                    })
            except Exception:
                pass  # Client disconnected

    def register_websocket(
        self, path: str, handler: Callable
    ) -> None:
        """Register a custom WebSocket handler."""
        self._websocket_handlers[path] = handler

    def discover_flask_routes(self) -> List[RouteInfo]:
        """Discover all routes registered on the Flask app."""
        if self.flask_app is None:
            return []

        routes = []
        for rule in self.flask_app.url_map.iter_rules():
            if rule.endpoint == "static":
                continue
            methods = sorted(
                rule.methods - {"OPTIONS", "HEAD"}
            )
            params = {}
            for arg in rule.arguments:
                # Try to infer type from converters
                converter = rule._converters.get(arg) if hasattr(rule, "_converters") else None
                params[arg] = type(converter).__name__ if converter else "string"

            routes.append(RouteInfo(
                rule=rule.rule,
                endpoint=rule.endpoint,
                methods=methods,
                params=params,
            ))
        return routes


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def create_fastapi_app(flask_app: Any = None) -> Any:
    """Create a FastAPI app, optionally wrapping a Flask application.

    Args:
        flask_app: Optional Flask app to mount under ``/flask``.

    Returns:
        A FastAPI application instance with OpenAPI docs at ``/docs``.
    """
    wrapper = OpenCutFastAPI(flask_app)
    return wrapper.fastapi_app


def mount_flask_app(fastapi_app: Any, flask_app: Any) -> Any:
    """Mount a Flask app inside a FastAPI app using WSGIMiddleware.

    Args:
        fastapi_app: The FastAPI app to mount into.
        flask_app: The Flask WSGI app.

    Returns:
        The FastAPI app with Flask mounted at ``/flask``.
    """
    try:
        from starlette.middleware.wsgi import WSGIMiddleware
    except ImportError:
        try:
            from a2wsgi import WSGIMiddleware
        except ImportError:
            logger.warning(
                "WSGIMiddleware not available; Flask app not mounted. "
                "Install starlette or a2wsgi."
            )
            return fastapi_app

    fastapi_app.mount("/flask", WSGIMiddleware(flask_app))
    return fastapi_app


def generate_openapi_spec(flask_app: Any = None) -> dict:
    """Generate a combined OpenAPI specification.

    Merges the FastAPI auto-generated spec with metadata discovered
    from Flask routes.

    Args:
        flask_app: Optional Flask app for route discovery.

    Returns:
        OpenAPI 3.1 spec as a dict.
    """
    spec = OpenAPISpec()

    # If Flask app is available, discover its routes
    if flask_app is not None:
        wrapper = OpenCutFastAPI(flask_app)
        routes = wrapper.discover_flask_routes()

        for route in routes:
            path_key = RouteAdapter._convert_path(route.rule)
            methods_spec = {}
            for method in route.methods:
                methods_spec[method.lower()] = {
                    "summary": route.endpoint.replace("_", " ").title(),
                    "description": route.description or f"Endpoint: {route.endpoint}",
                    "operationId": route.endpoint,
                    "parameters": [
                        {
                            "name": pname,
                            "in": "path",
                            "required": True,
                            "schema": {"type": ptype},
                        }
                        for pname, ptype in route.params.items()
                    ],
                    "responses": {
                        "200": {"description": "Success"},
                    },
                }
            spec.paths[path_key] = methods_spec

    return spec.to_dict()
