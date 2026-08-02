"""The OpenAPI contract must be one document, typed, and non-regressing.

Before this, OpenCut published three disagreeing specs: a 3.0.3 document with
dataclass responses, a schema-free 3.1.0 skeleton, and a third adapter under
`/architecture`. Clients could not tell what an operation accepted without
reading route source.

These tests pin the consolidation:

* both HTTP endpoints describe the *same* operations, one as 3.1.1 and one as
  a 3.0.3 compatibility rendering of it;
* request schemas are the ones the curated MCP tools already use, so REST and
  MCP cannot drift;
* every operation carries typed, `$ref`-ed error responses;
* generated valid payloads validate and deliberately invalid ones do not;
* a committed manifest ratchets typed coverage so it cannot silently fall.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pytest

from opencut.core import openapi_source
from opencut.tools import dump_openapi_contract

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    from opencut.server import create_app
    return create_app()


@pytest.fixture(scope="module")
def spec(app):
    return openapi_source.build_spec(app)


# ---------------------------------------------------------------------------
# One source, explicit adapters
# ---------------------------------------------------------------------------
def test_canonical_endpoint_serves_the_source_document(app, spec):
    body = app.test_client().get("/api/openapi.json?refresh=1").get_json()
    assert body["openapi"] == openapi_source.OPENAPI_VERSION
    assert body["jsonSchemaDialect"] == openapi_source.JSON_SCHEMA_DIALECT
    assert set(body["paths"]) == set(spec["paths"])


def test_legacy_endpoint_is_a_downgrade_of_the_same_document(app, spec):
    body = app.test_client().get("/openapi.json").get_json()
    assert body["openapi"] == "3.0.3"
    # Same operations, different rendering — that is what makes it an adapter
    # rather than a second generator.
    assert set(body["paths"]) == set(spec["paths"])
    for path, path_item in spec["paths"].items():
        assert set(path_item) == set(body["paths"][path]), path


def test_downgrade_removes_constructs_30_cannot_express(spec):
    downgraded = openapi_source.downgrade_to_30(spec)
    assert "jsonSchemaDialect" not in downgraded

    def walk(node):
        if isinstance(node, list):
            for item in node:
                yield from walk(item)
        elif isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)

    for node in walk(downgraded):
        assert "const" not in node, node
        assert "prefixItems" not in node, node
        assert not isinstance(node.get("type"), list), node


def test_downgrade_maps_nullable_unions():
    source = {"paths": {}, "components": {"schemas": {
        "Thing": {"type": "object", "properties": {
            "maybe": {"type": ["string", "null"]},
            "fixed": {"const": "only"},
        }},
    }}}
    out = openapi_source.downgrade_to_30(source)
    maybe = out["components"]["schemas"]["Thing"]["properties"]["maybe"]
    assert maybe["type"] == "string"
    assert maybe["nullable"] is True
    assert out["components"]["schemas"]["Thing"]["properties"]["fixed"]["enum"] == ["only"]


# ---------------------------------------------------------------------------
# Typed operations
# ---------------------------------------------------------------------------
def test_request_schemas_come_from_the_curated_mcp_tools():
    from opencut import mcp_server

    request_map = openapi_source.request_schema_map()
    assert request_map, "no typed request schemas were discovered"

    tool_routes = {
        (str(method).upper(), str(path))
        for method, path in mcp_server._TOOL_ROUTES.values()
    }
    # Every typed request must trace back to a curated tool — a schema with no
    # MCP owner would be a second, unreviewed source.
    assert set(request_map) <= tool_routes


def test_typed_request_bodies_are_object_schemas_with_properties(spec):
    typed = 0
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            body = operation.get("requestBody")
            if not body:
                continue
            typed += 1
            schema = body["content"]["application/json"]["schema"]
            assert schema["type"] == "object", f"{method} {path}"
            assert schema["properties"], f"{method} {path}"
    assert typed >= 50, f"only {typed} operations carry a typed request body"


def test_every_operation_has_typed_error_responses(spec):
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            responses = operation["responses"]
            pointer = f"{method.upper()} {path}"
            for status in ("400", "429", "500"):
                assert status in responses, pointer
                assert "$ref" in responses[status], pointer
            if method in ("post", "put", "patch", "delete"):
                assert "$ref" in responses.get("403", {}), pointer


def test_error_responses_all_point_at_the_shared_error_schema(spec):
    components = spec["components"]
    for name, response in components["responses"].items():
        ref = response["content"]["application/json"]["schema"]["$ref"]
        assert ref == "#/components/schemas/OpenCutError", name
    error = components["schemas"]["OpenCutError"]
    assert error["required"] == ["error"]
    assert set(error["properties"]) >= {"error", "code", "suggestion"}


def test_csrf_protected_operations_declare_the_security_scheme(app, spec):
    from opencut.core.openapi_spec import _route_uses_csrf

    documented = {
        f"{method.upper()} {path}"
        for path, path_item in spec["paths"].items()
        for method, operation in path_item.items()
        if isinstance(operation, dict) and operation.get("security")
    }
    assert documented, (
        "no operation declares CSRF; the detector is broken again"
    )
    # Spot-check against the live decorator rather than trusting the spec.
    for rule in app.url_map.iter_rules():
        if rule.rule != "/silence":
            continue
        assert _route_uses_csrf(app.view_functions[rule.endpoint])
    assert "POST /silence" in documented


def test_security_scheme_names_the_real_header(spec):
    scheme = spec["components"]["securitySchemes"]["CSRFToken"]
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-OpenCut-Token"


def test_operation_ids_are_unique(spec):
    ids = [
        operation["operationId"]
        for path_item in spec["paths"].values()
        for method, operation in path_item.items()
        if method != "parameters" and isinstance(operation, dict)
    ]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Generated payloads
# ---------------------------------------------------------------------------
def _sample_for(schema: dict):
    """Smallest payload that satisfies a request schema's required fields."""
    sample = {}
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        prop = properties.get(name, {})
        kind = prop.get("type")
        if kind == "string":
            sample[name] = prop.get("enum", ["sample"])[0]
        elif kind == "integer":
            sample[name] = int(prop.get("minimum", 1))
        elif kind == "number":
            sample[name] = float(prop.get("minimum", 1))
        elif kind == "boolean":
            sample[name] = True
        elif kind == "array":
            sample[name] = []
        elif kind == "object":
            sample[name] = {}
        else:
            sample[name] = "sample"
    return sample


def test_generated_payloads_validate_against_their_own_schemas(spec):
    jsonschema = pytest.importorskip("jsonschema")

    checked = 0
    for path, path_item in spec["paths"].items():
        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            body = operation.get("requestBody")
            if not body:
                continue
            schema = body["content"]["application/json"]["schema"]
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(_sample_for(schema), schema)
            checked += 1
    assert checked >= 50


def test_invalid_payloads_are_rejected_by_their_own_schemas(spec):
    jsonschema = pytest.importorskip("jsonschema")

    rejected = 0
    for path_item in spec["paths"].values():
        for method, operation in path_item.items():
            if method == "parameters" or not isinstance(operation, dict):
                continue
            body = operation.get("requestBody")
            if not body:
                continue
            schema = body["content"]["application/json"]["schema"]
            required = schema.get("required") or []
            if not required:
                continue
            # Wrong type on a required field must fail; a schema that accepts
            # anything is not a contract.
            broken = _sample_for(schema)
            prop = schema["properties"].get(required[0], {})
            broken[required[0]] = {} if prop.get("type") != "object" else "nope"
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(broken, schema)
            rejected += 1
    assert rejected >= 20, f"only {rejected} request schemas reject bad input"


def test_component_schemas_are_valid_2020_12(spec):
    jsonschema = pytest.importorskip("jsonschema")

    for name, schema in spec["components"]["schemas"].items():
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:  # pragma: no cover
            raise AssertionError(f"{name} is not valid 2020-12: {exc}") from exc


def test_a_real_error_response_matches_the_documented_error_schema(app, spec):
    jsonschema = pytest.importorskip("jsonschema")

    # A mutation without the CSRF header is the cheapest real error path.
    response = app.test_client().post("/silence", json={"filepath": "/nope.mp4"})
    assert response.status_code in (400, 403), response.get_json()
    jsonschema.validate(
        response.get_json(), spec["components"]["schemas"]["OpenCutError"]
    )


# ---------------------------------------------------------------------------
# Coverage ratchet
# ---------------------------------------------------------------------------
class TestContractManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.committed = dump_openapi_contract.load_manifest()
        cls.live = dump_openapi_contract.build_manifest()

    def test_committed_manifest_matches_live_app(self):
        diffs = dump_openapi_contract.diff_manifests(self.committed, self.live)
        self.assertFalse(diffs, (
            "openapi_contract.json is out of sync. Run "
            "`python -m opencut.tools.dump_openapi_contract` and commit.\n  - "
            + "\n  - ".join(diffs[:10])
        ))

    def test_ratchet_counters_are_present_and_positive(self):
        coverage = self.committed["coverage"]
        for key in dump_openapi_contract.RATCHET_KEYS:
            with self.subTest(key=key):
                self.assertGreater(coverage[key], 0)

    def test_ratchet_detects_a_regression(self):
        weakened = json.loads(json.dumps(self.committed))
        weakened["coverage"]["typed_responses"] -= 1
        diffs = dump_openapi_contract.diff_manifests(self.committed, weakened)
        self.assertTrue(any(line.startswith("RATCHET:") for line in diffs), diffs)

    def test_manifest_records_both_renderings(self):
        self.assertEqual(
            self.committed["canonical_openapi"], openapi_source.OPENAPI_VERSION
        )
        self.assertEqual(self.committed["compatibility_openapi"], "3.0.3")

    def test_manifest_is_committed_under_generated(self):
        self.assertTrue(dump_openapi_contract.MANIFEST_PATH.is_file())
        self.assertEqual(
            dump_openapi_contract.MANIFEST_PATH.parent.name, "_generated"
        )
