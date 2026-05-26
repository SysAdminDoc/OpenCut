"""
Tests for the F146 UXP MCP bridge (RESEARCH_FEATURE_PLAN_2026-05-25 carry-forward).

The bridge proxies the MCP sidecar's tool catalogue + dispatcher onto
the main Flask app on :5679 so UXP panels can keep using MCP after
Adobe's CEP EOL (~Sept 2026). Three routes:

  GET  /mcp/tools     curated + opt-in extended catalogue
  POST /mcp/call      invoke a tool (CSRF-protected, rate-limited)
  GET  /mcp/info      capability + counts

These tests:
  * confirm the tool catalogue surfaces curated + extended counts;
  * confirm /mcp/call rejects unknown tools, missing arguments, and
    non-dict arguments;
  * confirm /mcp/call routes a real tool name through handle_tool_call
    (with the underlying call mocked so the test stays hermetic);
  * confirm rate-limit acquire/release is paired (no slot leak after
    a failing call);
  * confirm /mcp/info ships the version + endpoints + transport tag.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMcpBridgeRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.token = cls.client.get("/health").get_json().get("csrf_token", "")

    def test_info_endpoint(self):
        resp = self.client.get("/mcp/info")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertGreater(body["curated_count"], 0)
        self.assertGreaterEqual(body["extended_count"], 0)
        self.assertEqual(body["transport"], "uxp-bridge")
        self.assertIn("/mcp/call", body["endpoints"])
        self.assertIn("/mcp/tools", body["endpoints"])

    def test_tools_curated_only(self):
        resp = self.client.get("/mcp/tools?include_extended=false")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertFalse(body["include_extended"])
        # Curated count must match /mcp/info.
        info = self.client.get("/mcp/info").get_json()
        self.assertEqual(body["count"], info["curated_count"])

    def test_tools_with_extended(self):
        resp = self.client.get("/mcp/tools?include_extended=true")
        body = json.loads(resp.data.decode("utf-8"))
        info = self.client.get("/mcp/info").get_json()
        # With extended on, count = curated + extended.
        self.assertEqual(body["count"], info["curated_count"] + info["extended_count"])

    def test_call_rejects_unknown_tool(self):
        resp = self.client.post(
            "/mcp/call",
            json={"tool": "definitely_not_a_real_tool", "arguments": {}},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertIn("unknown tool", body["error"])

    def test_call_rejects_missing_tool(self):
        resp = self.client.post(
            "/mcp/call",
            json={"arguments": {}},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_call_rejects_non_dict_arguments(self):
        # Pick a real tool name so we get past the allowlist guard.
        tool_name = self.client.get("/mcp/tools?include_extended=false") \
            .get_json()["tools"][0]["name"]
        resp = self.client.post(
            "/mcp/call",
            json={"tool": tool_name, "arguments": "not a dict"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_call_routes_through_handle_tool_call(self):
        """The bridge must dispatch through opencut.mcp_server.handle_tool_call."""
        tool_name = self.client.get("/mcp/tools?include_extended=false") \
            .get_json()["tools"][0]["name"]
        sentinel = {"sentinel": True, "echo": tool_name}
        with patch("opencut.mcp_server.handle_tool_call", return_value=sentinel) as m:
            resp = self.client.post(
                "/mcp/call",
                json={"tool": tool_name, "arguments": {"filepath": "/tmp/x.mp4"}},
                headers={"X-OpenCut-Token": self.token},
            )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(body["tool"], tool_name)
        self.assertEqual(body["result"], sentinel)
        self.assertGreaterEqual(body["duration_ms"], 0)
        m.assert_called_once_with(tool_name, {"filepath": "/tmp/x.mp4"})

    def test_call_csrf_required(self):
        tool_name = self.client.get("/mcp/tools?include_extended=false") \
            .get_json()["tools"][0]["name"]
        # No X-OpenCut-Token header → CSRF should reject.
        resp = self.client.post(
            "/mcp/call",
            json={"tool": tool_name, "arguments": {}},
        )
        # CSRF rejection is 403 in this codebase.
        self.assertEqual(resp.status_code, 403)

    def test_call_rate_limit_release_after_exception(self):
        """If the wrapped tool call raises, the rate-limit slot must release."""
        from opencut.security import rate_limit, rate_limit_release
        tool_name = self.client.get("/mcp/tools?include_extended=false") \
            .get_json()["tools"][0]["name"]
        rl_key = f"mcp_bridge::{tool_name}"

        # First confirm the key isn't held.
        first = rate_limit(rl_key)
        if first:
            rate_limit_release(rl_key)

        with patch("opencut.mcp_server.handle_tool_call", side_effect=RuntimeError("boom")):
            resp = self.client.post(
                "/mcp/call",
                json={"tool": tool_name, "arguments": {}},
                headers={"X-OpenCut-Token": self.token},
            )
        # Whatever the response shape, the slot must be free again.
        again = rate_limit(rl_key)
        self.assertTrue(again, "Rate-limit slot leaked after exception in tool call")
        rate_limit_release(rl_key)
        # And the response must be an error (500-class) not a hang.
        self.assertGreaterEqual(resp.status_code, 400)


class TestToolIndex(unittest.TestCase):
    def test_tool_index_keyed_by_name(self):
        from opencut.routes.mcp_bridge_routes import _tool_index
        idx = _tool_index()
        self.assertGreater(len(idx), 30)
        # Every value is the original tool dict.
        for name, tool in idx.items():
            self.assertEqual(tool["name"], name)


if __name__ == "__main__":
    unittest.main()
