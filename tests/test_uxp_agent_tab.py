"""
Tests for the UXP Agent tab wiring (RESEARCH_FEATURE_PLAN_2026-05-25 F252 follow-on).

The Agent tab is the UXP-only surface that exposes the four backends
shipped in the 2026-05-25 autonomous loop:

  - F143 chat conductor (plan / self-review)
  - Q3 one-click Enhance
  - Q7 / F273 Sequence Index
  - Q8 shorts A/B variants
  - F146 MCP bridge

These tests assert that:
  * every button rendered in index.html has a matching event handler
    in main.js (no dead-button regressions);
  * every endpoint hit by the Agent tab is registered in the Flask app
    (no typo regressions);
  * the parity ledger annotates ``agent`` as UXP-only.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UXP_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
PARITY_LEDGER = REPO_ROOT / "extension" / "PANEL_PARITY.json"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


# Buttons declared in the Agent panel of index.html.
AGENT_BUTTON_IDS = (
    "agentChatPlanBtn",
    "agentChatReviewBtn",
    "agentChatClearBtn",
    "enhanceDryRunBtn",
    "enhanceRunBtn",
    "variantsDryRunBtn",
    "variantsRunBtn",
    "sequenceIndexBuildBtn",
    "sequenceIndexInfoBtn",
    "mcpBridgeInfoBtn",
    "mcpBridgeListBtn",
)

# Endpoints the Agent tab hits via BackendClient.
AGENT_ENDPOINTS = (
    "/agent/chat/plan",
    "/agent/chat/review",
    "/enhance/auto/dry-run",
    "/enhance/auto",
    "/shorts/variants/dry-run",
    "/shorts/variants",
    "/timeline/sequence-index",
    "/timeline/sequence-index/info",
    "/mcp/info",
    "/mcp/tools",
)


class TestUxpAgentTabHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read(UXP_HTML)

    def test_agent_tab_button_present(self):
        self.assertIn('data-tab="agent"', self.html)
        self.assertIn('id="tabBtnAgent"', self.html)

    def test_agent_tab_panel_present(self):
        self.assertIn('id="tab-agent"', self.html)

    def test_all_buttons_declared(self):
        for btn_id in AGENT_BUTTON_IDS:
            with self.subTest(button=btn_id):
                self.assertIn(f'id="{btn_id}"', self.html,
                              f"Agent panel HTML missing button id={btn_id!r}")


class TestUxpAgentTabWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = _read(UXP_JS)

    def test_init_agent_tab_function_exists(self):
        self.assertIn("function initAgentTab()", self.js)

    def test_init_agent_tab_called_from_initapp(self):
        # The initApp body must call initAgentTab() before "Ready" log.
        self.assertIn("initAgentTab();", self.js)

    def test_every_button_has_event_handler(self):
        """Each declared button must be referenced as an event-handler target.
        We accept either ``addEventListener`` on ``$("id")`` or the
        ``$("id")?.addEventListener`` shorthand."""
        for btn_id in AGENT_BUTTON_IDS:
            with self.subTest(button=btn_id):
                # Look for either `getElementById("id")` (via $) or a string
                # match on the id followed shortly by addEventListener.
                pattern = re.compile(
                    re.escape(f'"{btn_id}"') + r".{0,200}?addEventListener",
                    re.DOTALL,
                )
                self.assertRegex(
                    self.js, pattern,
                    f"No event handler wired for button id={btn_id!r}",
                )

    def test_every_endpoint_referenced(self):
        for endpoint in AGENT_ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                self.assertIn(endpoint, self.js,
                              f"Agent tab JS missing endpoint reference {endpoint!r}")


class TestUxpAgentEndpointsRegistered(unittest.TestCase):
    """Every endpoint the Agent tab hits must exist on the live Flask app."""

    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.rules = {r.rule for r in cls.app.url_map.iter_rules()}

    def test_all_agent_endpoints_present_on_live_app(self):
        # Drop the optional query string from /mcp/tools.
        normalized = [e.split("?", 1)[0] for e in AGENT_ENDPOINTS]
        for endpoint in normalized:
            with self.subTest(endpoint=endpoint):
                self.assertIn(
                    endpoint, self.rules,
                    f"Live Flask app does not register {endpoint!r} "
                    "— UXP Agent tab will see 404",
                )


class TestPanelParityLedger(unittest.TestCase):
    """The UXP-only ``agent`` tab must be annotated in PANEL_PARITY.json."""

    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(_read(PARITY_LEDGER))

    def test_agent_in_uxp_only(self):
        self.assertIn("agent", self.ledger.get("uxp_only", {}))

    def test_agent_has_justification(self):
        agent = self.ledger["uxp_only"]["agent"]
        self.assertTrue(agent.get("justification"))
        self.assertIn("F143", agent["justification"])
        self.assertIn("F146", agent["justification"])


if __name__ == "__main__":
    unittest.main()
