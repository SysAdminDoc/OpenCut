"""Conformance matrix across the 2026-07-28 MCP protocol boundary.

OpenCut has to serve two eras at once: pre-2026 clients that open with
`initialize` and expect a bare result, and 2026-07-28 clients that are
stateless, state their version per request in `_meta`, and require
`resultType`, server identity, and cache hints on every result.

These tests pin both eras and — just as importantly — pin what OpenCut does
*not* claim. An advertised capability nobody implemented is worse than an
absent one, so the capability report is asserted closed.
"""
from __future__ import annotations

import json
import unittest

from opencut import __version__, mcp_server


def rpc(method: str, params: dict | None = None, msg_id=1):
    return mcp_server.dispatch_jsonrpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": method,
        "params": params or {},
    })


def modern(extra: dict | None = None) -> dict:
    meta = {mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION}
    meta.update(extra or {})
    return {"_meta": meta}


def legacy() -> dict:
    return {"_meta": {mcp_server.META_PROTOCOL_VERSION: mcp_server.LEGACY_PROTOCOL_VERSION}}


class TestDiscovery(unittest.TestCase):
    """`server/discover` replaces the handshake as the entry point."""

    def test_discover_is_implemented(self):
        result = rpc("server/discover")["result"]
        self.assertIn(mcp_server.LATEST_PROTOCOL_VERSION, result["protocolVersions"])
        self.assertEqual(result["serverInfo"]["name"], "opencut")
        self.assertEqual(result["serverInfo"]["version"], __version__)

    def test_discover_needs_no_prior_handshake(self):
        # No initialize, no session id, no state — this is the whole point.
        self.assertIn("result", rpc("server/discover"))

    def test_discover_advertises_the_json_schema_dialect(self):
        result = rpc("server/discover")["result"]
        self.assertEqual(result["schemaDialect"], mcp_server.JSON_SCHEMA_DIALECT)
        self.assertIn("2020-12", result["schemaDialect"])

    def test_latest_version_is_first_and_legacy_is_still_offered(self):
        versions = rpc("server/discover")["result"]["protocolVersions"]
        self.assertEqual(versions[0], mcp_server.LATEST_PROTOCOL_VERSION)
        self.assertIn(mcp_server.LEGACY_PROTOCOL_VERSION, versions)


class TestCapabilityHonesty(unittest.TestCase):
    """Optional extensions must not be claimed without an implementation."""

    def setUp(self):
        self.capabilities = mcp_server.server_capabilities()

    def test_tools_are_claimed(self):
        self.assertIn("tools", self.capabilities)

    def test_subscriptions_are_not_claimed(self):
        self.assertFalse(self.capabilities["resources"]["subscribe"])
        # `subscriptions/listen` is unimplemented, so it must not be routable.
        self.assertEqual(rpc("subscriptions/listen")["error"]["code"], -32601)

    def test_tasks_extension_is_not_claimed(self):
        self.assertEqual(self.capabilities["extensions"], {})
        for method in ("tasks/get", "tasks/update"):
            with self.subTest(method=method):
                self.assertEqual(rpc(method)["error"]["code"], -32601)

    def test_removed_methods_are_not_served(self):
        # 2026-07-28 removed ping and logging/setLevel outright.
        for method in ("ping", "logging/setLevel", "resources/subscribe"):
            with self.subTest(method=method):
                self.assertEqual(rpc(method)["error"]["code"], -32601)


class TestModernResultEnvelope(unittest.TestCase):
    def test_results_carry_result_type_complete(self):
        for method in ("server/discover", "tools/list", "prompts/list", "resources/list"):
            with self.subTest(method=method):
                result = rpc(method, modern())["result"]
                self.assertEqual(result["resultType"], "complete")

    def test_results_carry_server_identity(self):
        result = rpc("tools/list", modern())["result"]
        info = result["_meta"][mcp_server.META_SERVER_INFO]
        self.assertEqual(info["name"], "opencut")
        self.assertEqual(info["version"], __version__)

    def test_list_results_carry_cache_hints(self):
        for method in ("tools/list", "prompts/list", "resources/list",
                       "resources/templates/list"):
            with self.subTest(method=method):
                result = rpc(method, modern())["result"]
                self.assertGreater(result["ttlMs"], 0)
                self.assertIn(result["cacheScope"], ("public", "private"))

    def test_tool_call_is_not_given_cache_hints(self):
        result = rpc("tools/call", modern({}) | {
            "name": "definitely_not_a_tool", "arguments": {},
        })["result"]
        self.assertEqual(result["resultType"], "complete")
        self.assertNotIn("ttlMs", result)

    def test_trace_context_is_propagated_back(self):
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        result = rpc("tools/list", modern({"traceparent": traceparent}))["result"]
        self.assertEqual(result["_meta"]["traceparent"], traceparent)

    def test_unknown_meta_is_not_echoed(self):
        result = rpc("tools/list", modern({"totally-made-up": "x"}))["result"]
        self.assertNotIn("totally-made-up", result["_meta"])


class TestLegacyEra(unittest.TestCase):
    """Pre-2026 clients must see exactly what they saw before."""

    def test_initialize_still_works(self):
        result = rpc("initialize", {"protocolVersion": mcp_server.LEGACY_PROTOCOL_VERSION})["result"]
        self.assertEqual(result["protocolVersion"], mcp_server.LEGACY_PROTOCOL_VERSION)
        self.assertEqual(result["serverInfo"]["name"], "opencut")

    def test_initialize_echoes_a_supported_requested_version(self):
        result = rpc("initialize", {
            "protocolVersion": mcp_server.LATEST_PROTOCOL_VERSION
        })["result"]
        self.assertEqual(result["protocolVersion"], mcp_server.LATEST_PROTOCOL_VERSION)

    def test_initialize_falls_back_for_an_unknown_requested_version(self):
        result = rpc("initialize", {"protocolVersion": "1999-01-01"})["result"]
        self.assertEqual(result["protocolVersion"], mcp_server.LEGACY_PROTOCOL_VERSION)

    def test_legacy_results_omit_the_modern_fields(self):
        for params in ({}, legacy()):
            with self.subTest(params=params):
                result = rpc("tools/list", params)["result"]
                self.assertNotIn("resultType", result)
                self.assertNotIn("_meta", result)
                self.assertNotIn("ttlMs", result)
                self.assertIn("tools", result)

    def test_initialized_notification_is_still_swallowed(self):
        self.assertIsNone(mcp_server.dispatch_jsonrpc({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        }))

    def test_a_message_without_an_id_never_gets_a_reply(self):
        self.assertIsNone(mcp_server.dispatch_jsonrpc({
            "jsonrpc": "2.0", "method": "tools/list",
        }))


class TestVersionNegotiation(unittest.TestCase):
    def test_unsupported_version_is_rejected_not_guessed(self):
        error = rpc("tools/list", {
            "_meta": {mcp_server.META_PROTOCOL_VERSION: "2019-01-01"}
        })["error"]
        self.assertEqual(error["code"], mcp_server.ERROR_UNSUPPORTED_PROTOCOL_VERSION)
        self.assertIn(mcp_server.LATEST_PROTOCOL_VERSION, error["data"]["supported"])

    def test_error_code_is_inside_the_specification_range(self):
        # -32020..-32099 is reserved for the specification.
        self.assertTrue(
            -32099 <= mcp_server.ERROR_UNSUPPORTED_PROTOCOL_VERSION <= -32020
        )

    def test_absent_version_is_treated_as_the_legacy_era(self):
        self.assertEqual(
            mcp_server.negotiated_protocol_version({}),
            mcp_server.LEGACY_PROTOCOL_VERSION,
        )

    def test_every_advertised_version_is_actually_served(self):
        for version in mcp_server.SUPPORTED_PROTOCOL_VERSIONS:
            with self.subTest(version=version):
                response = rpc("tools/list", {
                    "_meta": {mcp_server.META_PROTOCOL_VERSION: version}
                })
                self.assertNotIn("error", response)


class TestToolCatalogue(unittest.TestCase):
    def test_tools_list_is_deterministic(self):
        first = [tool["name"] for tool in rpc("tools/list", modern())["result"]["tools"]]
        second = [tool["name"] for tool in rpc("tools/list", modern())["result"]["tools"]]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)), "duplicate tool names")

    def test_input_schemas_are_json_schema_2020_12_compatible(self):
        """No draft-04-only keywords that a 2020-12 validator would reject."""
        banned = {"exclusiveMinimum_boolean", "id", "definitions"}
        for tool in mcp_server.get_mcp_tools(include_extended=False):
            schema = tool.get("inputSchema") or {}
            with self.subTest(tool=tool["name"]):
                self.assertEqual(schema.get("type"), "object")
                self.assertIsInstance(schema.get("properties", {}), dict)
                self.assertFalse(banned & set(schema))

    def test_tool_call_result_is_json_serialisable(self):
        result = rpc("tools/call", modern() | {
            "name": "definitely_not_a_tool", "arguments": {},
        })["result"]
        json.dumps(result)
        self.assertEqual(result["content"][0]["type"], "text")


class TestBridgeRouteReportsProtocol(unittest.TestCase):
    """The panels read their counts and protocol from the backend."""

    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.client = create_app().test_client()

    def test_mcp_info_reports_generated_counts_and_protocol(self):
        body = self.client.get("/mcp/info").get_json()
        self.assertEqual(body["latest_protocol_version"], mcp_server.LATEST_PROTOCOL_VERSION)
        self.assertIn(mcp_server.LEGACY_PROTOCOL_VERSION, body["protocol_versions"])
        self.assertEqual(body["schema_dialect"], mcp_server.JSON_SCHEMA_DIALECT)
        self.assertEqual(
            body["curated_count"],
            len(mcp_server.get_mcp_tools(include_extended=False)),
        )
        self.assertGreater(body["extended_count"], 0)
        self.assertFalse(body["capabilities"]["resources"]["subscribe"])


class TestPanelsDoNotHardcodeCounts(unittest.TestCase):
    def test_uxp_mcp_hint_has_no_baked_in_tool_counts(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / "extension" / "com.opencut.uxp"
        for name in ("index.html", "locales/en.json", "locales/es.json"):
            text = (root / name).read_text(encoding="utf-8")
            hint_lines = [
                line for line in text.splitlines() if "mcp_bridge_hint" in line
            ]
            for line in hint_lines:
                with self.subTest(file=name, line=line.strip()[:80]):
                    self.assertNotRegex(
                        line, r"\b\d{2,}[,\d]*\s+(curated|opt-in|extended|herramientas)"
                    )


if __name__ == "__main__":
    unittest.main()
