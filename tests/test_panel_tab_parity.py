"""
CEP <-> UXP tab parity gate (RESEARCH_FEATURE_PLAN_2026-05-25 Q5).

Adobe's November 2025 guidance plans one calendar year of dual CEP/UXP
support after Premiere Pro 25.6. The CEP -> UXP migration plan (F252)
assumes panel parity that is not actually true at the tab
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
README = REPO_ROOT / "README.md"
UXP_MIGRATION = REPO_ROOT / "docs" / "UXP_MIGRATION.md"

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

    def test_adobe_cep_timeline_has_primary_source_and_retrieval_date(self):
        timeline = self.ledger.get("$adobe_cep_eol")
        self.assertIsInstance(timeline, dict)
        self.assertEqual(timeline.get("planning_horizon"), "approximately 2026-11")
        self.assertIn("not announced an exact removal date", timeline.get("planning_note", ""))
        self.assertEqual(
            timeline.get("statement"),
            "the plan is to support both CEP and UXP for a calendar year, after which we will remove support for CEP extensibilty",
        )
        self.assertEqual(
            timeline.get("source_url"),
            "https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md",
        )
        self.assertEqual(timeline.get("retrieved"), "2026-08-12")

    def test_cep_only_tabs_remain_maintained_through_the_corrected_horizon(self):
        for tab_id in ("export", "nlp"):
            justification = self.ledger["cep_only"][tab_id]["justification"].lower()
            self.assertIn("maintained", justification)
            self.assertIn("security", justification)
            self.assertIn("reliability", justification)
            self.assertIn("november 2026", justification)
            self.assertNotIn("do not invest further", justification)

        serialized = json.dumps(self.ledger)
        self.assertNotIn("2026-09", serialized)

    def test_public_guidance_uses_the_sourced_planning_horizon(self):
        for path in (README, UXP_MIGRATION):
            source = _read(path)
            self.assertIn("November 2026", source, path.name)
            self.assertIn(
                "https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md",
                source,
                path.name,
            )
            self.assertNotIn("September 2026", source, path.name)


if __name__ == "__main__":
    unittest.main()
