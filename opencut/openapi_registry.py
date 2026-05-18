"""Dataclass-backed OpenAPI response schema registry.

F193 moves response schema selection away from the legacy
``opencut.openapi`` endpoint table. Schema classes now carry their route
metadata, and this module discovers those dataclasses from safe, stdlib-only
schema modules.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, is_dataclass
from importlib import import_module
from typing import Any, Iterable, Mapping

OPENAPI_ROUTES_ATTR = "__openapi_routes__"
OPENAPI_EXTRA_PROPERTIES_ATTR = "__openapi_extra_properties__"

DEFAULT_RESPONSE_SCHEMA_MODULES = (
    "opencut.schemas",
    "opencut.core.audio_description",
    "opencut.core.c2pa_sidecar",
    "opencut.core.crash_packet",
    "opencut.core.delivery_transfer",
    "opencut.core.eval_datasets",
    "opencut.core.marker_import",
    "opencut.core.ocio_validate",
    "opencut.core.project_health",
    "opencut.core.review_bundle",
)


@dataclass(frozen=True)
class ResponseSchemaRecord:
    """One route-to-dataclass OpenAPI response schema binding."""

    route: str
    schema_class: type
    module: str


def _normalise_routes(routes: Iterable[str]) -> tuple[str, ...]:
    normalised: list[str] = []
    for route in routes:
        value = str(route or "").strip()
        if not value:
            raise ValueError("OpenAPI schema route cannot be empty")
        if not value.startswith("/"):
            raise ValueError(f"OpenAPI schema route must start with '/': {value!r}")
        normalised.append(value)
    if not normalised:
        raise ValueError("At least one OpenAPI schema route is required")
    return tuple(dict.fromkeys(normalised))


def register_response_schema(
    schema_class: type,
    *routes: str,
    extra_properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> type:
    """Attach OpenAPI route metadata to a dataclass schema class."""

    setattr(schema_class, OPENAPI_ROUTES_ATTR, _normalise_routes(routes))
    if extra_properties:
        setattr(
            schema_class,
            OPENAPI_EXTRA_PROPERTIES_ATTR,
            {str(name): dict(schema) for name, schema in extra_properties.items()},
        )
    return schema_class


def openapi_response_schema(
    *routes: str,
    extra_properties: Mapping[str, Mapping[str, Any]] | None = None,
):
    """Class decorator for route response dataclasses."""

    def _decorate(schema_class: type) -> type:
        return register_response_schema(
            schema_class,
            *routes,
            extra_properties=extra_properties,
        )

    return _decorate


def schema_routes(schema_class: type) -> tuple[str, ...]:
    """Return route metadata attached to a dataclass schema class."""

    routes = getattr(schema_class, OPENAPI_ROUTES_ATTR, ())
    if isinstance(routes, str):
        return _normalise_routes((routes,))
    return _normalise_routes(routes) if routes else ()


def schema_extra_properties(schema_class: type) -> dict[str, dict]:
    """Return computed/custom OpenAPI properties for a schema class."""

    extras = getattr(schema_class, OPENAPI_EXTRA_PROPERTIES_ATTR, {})
    return {str(name): dict(schema) for name, schema in dict(extras).items()}


def _iter_module_schema_classes(module) -> Iterable[type]:
    for _, value in inspect.getmembers(module, inspect.isclass):
        if value.__module__ != module.__name__:
            continue
        if is_dataclass(value) and schema_routes(value):
            yield value


def discover_response_schemas(
    module_names: Iterable[str] = DEFAULT_RESPONSE_SCHEMA_MODULES,
) -> list[ResponseSchemaRecord]:
    """Discover route response schemas from dataclasses in configured modules."""

    records: list[ResponseSchemaRecord] = []
    seen_routes: dict[str, str] = {}
    for module_name in module_names:
        module = import_module(module_name)
        for schema_class in _iter_module_schema_classes(module):
            for route in schema_routes(schema_class):
                existing = seen_routes.get(route)
                schema_name = f"{schema_class.__module__}.{schema_class.__name__}"
                if existing and existing != schema_name:
                    raise ValueError(
                        f"Duplicate OpenAPI schema route {route!r}: "
                        f"{existing} and {schema_name}"
                    )
                seen_routes[route] = schema_name
                records.append(
                    ResponseSchemaRecord(
                        route=route,
                        schema_class=schema_class,
                        module=module_name,
                    )
                )
    return sorted(records, key=lambda record: (record.route, record.schema_class.__name__))


def build_endpoint_schema_map(
    module_names: Iterable[str] = DEFAULT_RESPONSE_SCHEMA_MODULES,
) -> dict[str, type]:
    """Return the legacy endpoint -> schema map from discovered dataclasses."""

    return {
        record.route: record.schema_class
        for record in discover_response_schemas(module_names)
    }

