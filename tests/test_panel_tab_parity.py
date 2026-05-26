"""
CEP <-> UXP tab parity gate (RESEARCH_FEATURE_PLAN_2026-05-25 Q5).

Adobe CEP end-of-life lands around 2026-09. The CEP -> UXP migration
plan (F252) assumes panel parity that is not actually true at the tab
level — CEP has 'export' and 'nlp' tabs that UXP doesn't, and UXP has
'search' and 'deliverables' tabs that CEP doesn't.

This test parses both panels' ``index.html`` for their declared tab IDs
(``data-nav="..."`` for CEP, ``data-tab="..."`` for UXP) and asserts the
divergence is exactly annotated in ``extension/PANEL_PARITY.json``.
Adding or removing a tab without updating the ledger fails the gate.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CEP_INDEX = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
UXP_INDEX = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
PARITY_LEDGER = REPO_ROOT / "extension" / "PANEL_PARITY.json"

CEP_TAB_RE = re.compile(r'data-nav="([^"]+)"')
UXP_TAB_RE = re.compile(r'data-tab="([^"]+)"')


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_tabs(text: str, regex: re.Pattern) -> set[str]:
    return set(regex.findall(text))


class TestPanelTabParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cep_tabs = _extract_tabs(_read(CEP_INDEX), CEP_TAB_RE)
        cls.uxp_tabs = _extract_tabs(_read(UXP_INDEX), UXP_TAB_RE)
        cls.ledger = json.loads(PARITY_LEDGER.read_text(encoding="utf-8"))

    def test_ledger_lists_every_common_tab(self):
        common = self.cep_tabs & self.uxp_tabs
        ledger_common = set(self.ledger.get("common_tabs", []))
        missing_in_ledger = common - ledger_common
        unexpected_in_ledger = ledger_common - common
        self.assertEqual(
            missing_in_ledger, set(),
            f"PANEL_PARITY.json common_tabs is missing tabs present in both panels: {missing_in_ledger}",
        )
        self.assertEqual(
            unexpected_in_ledger, set(),
            f"PANEL_PARITY.json common_tabs lists tabs not present in both panels: {unexpected_in_ledger}",
        )

    def test_ledger_annotates_every_cep_only_tab(self):
        cep_only = self.cep_tabs - self.uxp_tabs
        ledger_cep_only = set(self.ledger.get("cep_only", {}).keys())
        missing_in_ledger = cep_only - ledger_cep_only
        unexpected_in_ledger = ledger_cep_only - cep_only
        self.assertEqual(
            missing_in_ledger, set(),
            f"CEP has tabs not annotated in PANEL_PARITY.json cep_only: {missing_in_ledger}. "
            "Add an entry with a justification field, or add the tab to UXP and remove it from this list.",
        )
        self.assertEqual(
            unexpected_in_ledger, set(),
            f"PANEL_PARITY.json cep_only annotates tabs that no longer exist in CEP: {unexpected_in_ledger}",
        )

    def test_ledger_annotates_every_uxp_only_tab(self):
        uxp_only = self.uxp_tabs - self.cep_tabs
        ledger_uxp_only = set(self.ledger.get("uxp_only", {}).keys())
        missing_in_ledger = uxp_only - ledger_uxp_only
        unexpected_in_ledger = ledger_uxp_only - uxp_only
        self.assertEqual(
            missing_in_ledger, set(),
            f"UXP has tabs not annotated in PANEL_PARITY.json uxp_only: {missing_in_ledger}. "
            "Add an entry with a justification field, or add the tab to CEP and remove it from this list.",
        )
        self.assertEqual(
            unexpected_in_ledger, set(),
            f"PANEL_PARITY.json uxp_only annotates tabs that no longer exist in UXP: {unexpected_in_ledger}",
        )

    def test_every_divergence_has_justification_field(self):
        for side in ("cep_only", "uxp_only"):
            for tab_id, entry in self.ledger.get(side, {}).items():
                self.assertIsInstance(
                    entry, dict,
                    f"PANEL_PARITY.json[{side}][{tab_id}] must be an object, got {type(entry).__name__}",
                )
                self.assertIn(
                    "justification", entry,
                    f"PANEL_PARITY.json[{side}][{tab_id}] missing required 'justification' field",
                )
                self.assertTrue(
                    str(entry["justification"]).strip(),
                    f"PANEL_PARITY.json[{side}][{tab_id}].justification is empty",
                )


if __name__ == "__main__":
    unittest.main()
