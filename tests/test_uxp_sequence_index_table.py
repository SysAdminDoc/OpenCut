"""Guardrails for the UXP Sequence Index working table.

The Sequence Index card used to POST a *summary* payload (track counts, not
tracks) and then throw the returned rows away, so the backend's filter/sort
surface and the host locators were unreachable from UXP. These tests pin the
pieces that make it an actual table:

  * a real timeline walk in PProBridge (not ``getSequenceInfo``);
  * table/grid semantics with sortable headers and keyboard navigation;
  * the filter, media/effects facet, result count, and empty state;
  * column visibility that survives a reload;
  * host jump and filtered CSV export.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_CSS = REPO_ROOT / "extension" / "com.opencut.uxp" / "style.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class TestSequenceIndexMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read(UXP_HTML)

    def test_table_has_grid_semantics(self):
        self.assertIn('id="sequenceIndexTable"', self.html)
        self.assertIn('role="grid"', self.html)
        self.assertIn('id="sequenceIndexHeadRow"', self.html)
        self.assertIn('id="sequenceIndexBody"', self.html)

    def test_table_has_an_accessible_caption(self):
        self.assertIn("oc-visually-hidden", self.html)
        self.assertIn('data-i18n="uxp.agent.index_table_caption"', self.html)

    def test_search_and_facet_controls_exist(self):
        for control_id in (
            "sequenceIndexSearch",
            "sequenceIndexTrackType",
            "sequenceIndexEffects",
            "sequenceIndexOffline",
            "sequenceIndexMinRating",
        ):
            with self.subTest(control=control_id):
                self.assertIn(f'id="{control_id}"', self.html)

    def test_every_control_has_a_label(self):
        for control_id in (
            "sequenceIndexSearch",
            "sequenceIndexTrackType",
            "sequenceIndexEffects",
            "sequenceIndexOffline",
            "sequenceIndexMinRating",
        ):
            with self.subTest(control=control_id):
                self.assertIn(f'for="{control_id}"', self.html)

    def test_result_count_and_empty_state_are_present(self):
        self.assertIn('id="sequenceIndexCount"', self.html)
        self.assertIn('id="sequenceIndexEmpty"', self.html)
        # The count line is a live region so filtering announces itself.
        count_tag = re.search(r"<[^>]*id=\"sequenceIndexCount\"[^>]*>", self.html)
        self.assertIsNotNone(count_tag)
        self.assertIn('aria-live="polite"', count_tag.group(0))

    def test_column_toggle_and_export_controls_exist(self):
        self.assertIn('id="sequenceIndexColumns"', self.html)
        self.assertIn('id="sequenceIndexExportBtn"', self.html)


class TestSequenceIndexController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.js = _read(UXP_JS)

    def test_bridge_walks_the_timeline_instead_of_the_summary(self):
        self.assertIn("async function getSequenceIndexPayload()", self.js)
        self.assertIn("getSequenceIndexPayload,", self.js)
        # The build handler must use the walk, not the track-count summary.
        block = self.js[
            self.js.index('$("sequenceIndexBuildBtn")?.addEventListener'):
            self.js.index('$("sequenceIndexExportBtn")?.addEventListener')
        ]
        self.assertIn("PProBridge.getSequenceIndexPayload()", block)
        self.assertNotIn("PProBridge.getSequenceInfo()", block)

    def test_walk_emits_the_fields_the_backend_indexes(self):
        block = self.js[
            self.js.index("async function _sequenceIndexTrack("):
            self.js.index("async function getSequenceInfo()")
        ]
        for field in ("name:", "path:", "start,", "end:", "effects:", "offline:", "nodeId:"):
            with self.subTest(field=field):
                self.assertIn(field, block)

    def test_filter_and_export_endpoints_are_called(self):
        self.assertIn("/timeline/sequence-index/filter", self.js)
        self.assertIn("/timeline/sequence-index/export-csv", self.js)

    def test_filter_payload_carries_every_facet(self):
        block = self.js[
            self.js.index("function sequenceIndexFilterPayload()"):
            self.js.index("function sequenceIndexRenderHead()")
        ]
        for key in ("query", "track_type", "min_rating", "sort_key", "descending",
                    "has_effects", "offline"):
            with self.subTest(key=key):
                self.assertIn(key, block)

    def test_sortable_headers_set_aria_sort(self):
        block = self.js[
            self.js.index("function sequenceIndexRenderHead()"):
            self.js.index("function sequenceIndexFocusCell(")
        ]
        self.assertIn('setAttribute("aria-sort"', block)
        self.assertIn('"ascending"', block)
        self.assertIn('"descending"', block)
        self.assertIn('setAttribute("scope", "col")', block)

    def test_grid_keyboard_navigation_is_wired(self):
        self.assertIn("function sequenceIndexHandleGridKey(", self.js)
        for key in ("ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Home", "End",
                    "PageUp", "PageDown"):
            with self.subTest(key=key):
                self.assertIn(f'case "{key}"', self.js)
        self.assertIn('$("sequenceIndexBody")?.addEventListener("keydown", sequenceIndexHandleGridKey)',
                      self.js)

    def test_rows_use_roving_tabindex(self):
        block = self.js[
            self.js.index("function sequenceIndexRenderRows()"):
            self.js.index("function sequenceIndexRenderCount(")
        ]
        self.assertIn('setAttribute("role", "gridcell")', block)
        self.assertIn("td.tabIndex = (rowIdx === 0 && colIdx === 0) ? 0 : -1;", block)

    def test_cells_are_written_as_text_not_markup(self):
        block = self.js[
            self.js.index("function sequenceIndexRenderRows()"):
            self.js.index("function sequenceIndexRenderCount(")
        ]
        self.assertIn("td.textContent =", block)
        self.assertNotIn("innerHTML", block)

    def test_host_jump_moves_the_playhead(self):
        self.assertIn("async function sequenceIndexJumpToRow(", self.js)
        self.assertIn("PProBridge.setSequencePlayhead({ seconds })", self.js)

    def test_column_visibility_persists(self):
        self.assertIn("opencut.uxp.sequenceIndex.columns", self.js)
        self.assertIn("function sequenceIndexPersistColumns()", self.js)
        self.assertIn("function sequenceIndexReadStoredColumns()", self.js)
        # Identity columns must stay in every export.
        self.assertIn("SEQUENCE_INDEX_PINNED_COLUMNS", self.js)

    def test_large_indexes_are_paged_and_reported_honestly(self):
        self.assertIn("SEQUENCE_INDEX_PAGE_SIZE", self.js)
        self.assertIn("uxp.agent.index_count_truncated", self.js)

    def test_loading_state_is_exposed(self):
        self.assertIn("function sequenceIndexSetBusy(", self.js)
        self.assertIn('setAttribute("aria-busy"', self.js)


class TestSequenceIndexStyles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = _read(UXP_CSS)

    def test_table_styles_exist(self):
        for selector in (".oc-table", ".oc-table-scroll", ".oc-table-sort",
                         ".oc-column-toggles", ".oc-visually-hidden"):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css)

    def test_sort_direction_is_not_conveyed_by_color_alone(self):
        self.assertIn('th[aria-sort="ascending"]', self.css)
        self.assertIn('th[aria-sort="descending"]', self.css)

    def test_forced_colors_keeps_the_grid_readable(self):
        self.assertIn("forced-colors: active", self.css)


class TestSequenceIndexRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.rules = {r.rule for r in cls.app.url_map.iter_rules()}

    def test_every_sequence_index_endpoint_is_registered(self):
        for rule in (
            "/timeline/sequence-index",
            "/timeline/sequence-index/filter",
            "/timeline/sequence-index/export-csv",
            "/timeline/sequence-index/info",
        ):
            with self.subTest(rule=rule):
                self.assertIn(rule, self.rules)


if __name__ == "__main__":
    unittest.main()
