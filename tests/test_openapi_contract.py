"""Contract tests for the generated OpenAPI catalogues (F208)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from opencut.openapi import _flask_rule_to_openapi_path

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
MUTATING_METHODS = {"post", "put", "patch", "delete"}
VALID_SCHEMA_TYPES = {"array", "boolean", "integer", "number", "object", "string"}
PATH_PARAM_RE = re.compile(r"{([^{}]+)}")
RAW_FLASK_PARAM_RE = re.compile(r"<[^>]+>")
OPERATION_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _expected_operations(app) -> set[tuple[str, str]]:
    operations: set[tuple[str, str]] = set()
    for rule in app.url_map.iter_rules():
        raw_rule = str(rule.rule)
        if raw_rule.startswith("/static"):
            continue
        path = re.sub(
            r"<(?:[A-Za-z_][A-Za-z0-9_]*:)?([A-Za-z_][A-Za-z0-9_]*)>",
            r"{\1}",
            raw_rule,
        )
        for method in sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}):
            operations.add((path, method.lower()))
    return operations


def _documented_operations(spec: dict[str, Any]) -> set[tuple[str, str]]:
    operations: set[tuple[str, str]] = set()
    for path, path_item in spec["paths"].items():
        for method in path_item:
            if method in HTTP_METHODS:
                operations.add((path, method))
    return operations


def _operation_items(spec: dict[str, Any]):
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method in HTTP_METHODS:
                yield path, method, operation


def _schema_nodes(schema: dict[str, Any]):
    yield schema
    for value in schema.get("properties", {}).values():
        if isinstance(value, dict):
            yield from _schema_nodes(value)
    items = schema.get("items")
    if isinstance(items, dict):
        yield from _schema_nodes(items)


def _assert_response_schema(response: dict[str, Any], pointer: str) -> None:
    assert isinstance(response, dict), pointer
    assert response.get("description"), pointer
    content = response.get("content", {})
    assert "application/json" in content, pointer
    schema = content["application/json"].get("schema")
    assert isinstance(schema, dict), pointer

    for node in _schema_nodes(schema):
        node_type = node.get("type")
        if isinstance(node_type, list):
            assert set(node_type) <= VALID_SCHEMA_TYPES, pointer
        elif node_type is not None:
            assert node_type in VALID_SCHEMA_TYPES, pointer
        if "properties" in node:
            assert isinstance(node["properties"], dict), pointer
        if node_type == "array":
            assert "items" in node, pointer


def test_flask_rule_to_openapi_path_converts_parameters():
    path, parameters = _flask_rule_to_openapi_path(
        "/journal/<int:entry_id>/asset/<path:asset_path>/<unknown:token>"
    )

    assert path == "/journal/{entry_id}/asset/{asset_path}/{token}"
    assert [param["name"] for param in parameters] == [
        "entry_id",
        "asset_path",
        "token",
    ]
    assert [param["schema"]["type"] for param in parameters] == [
        "integer",
        "string",
        "string",
    ]
    assert all(param["in"] == "path" for param in parameters)
    assert all(param["required"] is True for param in parameters)


def test_root_openapi_documents_every_live_flask_route(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.get_json()

    assert spec["openapi"] == "3.0.3"
    assert spec["info"]["title"] == "OpenCut API"
    assert len(spec["paths"]) >= 1300

    documented = _documented_operations(spec)
    expected = _expected_operations(client.application)

    assert documented == expected
    assert not [path for path in spec["paths"] if RAW_FLASK_PARAM_RE.search(path)]

    for path, path_item in spec["paths"].items():
        expected_param_names = set(PATH_PARAM_RE.findall(path))
        if not expected_param_names:
            continue
        documented_param_names = {
            parameter.get("name")
            for parameter in path_item.get("parameters", [])
            if parameter.get("in") == "path"
        }
        assert documented_param_names == expected_param_names, path


def test_root_openapi_operations_have_valid_ids_and_responses(client):
    spec = client.get("/openapi.json").get_json()
    operation_ids = []

    for path, method, operation in _operation_items(spec):
        pointer = f"{method.upper()} {path}"
        operation_id = operation.get("operationId")
        assert operation_id, pointer
        assert OPERATION_ID_RE.match(operation_id), pointer
        operation_ids.append(operation_id)

        assert operation.get("summary"), pointer
        assert operation.get("tags"), pointer

        responses = operation.get("responses")
        assert isinstance(responses, dict), pointer
        assert "200" in responses, pointer
        _assert_response_schema(responses["200"], f"{pointer} 200")

        if method in MUTATING_METHODS:
            for status_code in ("400", "403"):
                assert status_code in responses, pointer
                _assert_response_schema(
                    responses[status_code],
                    f"{pointer} {status_code}",
                )

    counts = Counter(operation_ids)
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    assert not duplicates


def test_api_openapi_catalogue_uses_openapi_path_parameters(client):
    response = client.get("/api/openapi.json?refresh=1")
    assert response.status_code == 200
    spec = response.get_json()

    assert spec["openapi"] == "3.1.0"
    assert len(spec["paths"]) >= 1300
    assert not [path for path in spec["paths"] if RAW_FLASK_PARAM_RE.search(path)]
