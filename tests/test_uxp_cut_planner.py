"""F349 — the UXP cut path must prefer the typed 26.3 remove-items action.

`Sequence.rippleDelete()` is absent from the 26.3 typings and is reported to
return success while changing nothing on that host, so the cut path now routes
through `SequenceEditor.createRemoveItemsAction`. That action removes whole
track items, so these tests pin the boundary rule that decides when a cut range
can be expressed that way, and the wiring that keeps ripple delete as fallback.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_DIR = REPO_ROOT / "extension" / "com.opencut.uxp"
PLANNER = UXP_DIR / "uxp-cut-planner.js"
MAIN_JS = UXP_DIR / "main.js"


@pytest.fixture(scope="module")
def node_bin() -> str:
    found = shutil.which("node") or shutil.which("node.exe")
    if not found:
        pytest.skip("node not on PATH")
    return found


def _plan(node_bin: str, items, start, end, tolerance=None) -> dict:
    args = json.dumps([items, start, end] + ([tolerance] if tolerance is not None else []))
    program = textwrap.dedent(
        f"""
        const url = require('url').pathToFileURL({json.dumps(str(PLANNER))}).href;
        import(url).then((mod) => {{
            const plan = mod.planCutRemoval(...{args});
            process.stdout.write(JSON.stringify({{
                contained: plan.contained.map((i) => i.id),
                straddling: plan.straddling.map((i) => i.id),
                unreadable: plan.unreadable.map((i) => i.id),
                removable: plan.removable,
                reason: plan.reason,
                tolerance: mod.CUT_BOUNDARY_TOLERANCE_SECONDS,
            }}));
        }}).catch((e) => {{ console.error(e); process.exit(1); }});
        """
    )
    result = subprocess.run(
        [node_bin, "-e", program],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError("node exited non-zero: " + (result.stderr or result.stdout))
    return json.loads(result.stdout or "{}")


def _item(id_, start, end):
    return {"id": id_, "start": start, "end": end}


def test_fully_contained_items_are_removable(node_bin):
    plan = _plan(node_bin, [_item("a", 2.0, 4.0), _item("b", 4.0, 5.0)], 1.0, 6.0)

    assert plan["removable"] is True
    assert plan["contained"] == ["a", "b"]
    assert plan["straddling"] == []


def test_items_outside_the_range_are_ignored(node_bin):
    items = [_item("before", 0.0, 1.0), _item("inside", 2.0, 3.0), _item("after", 5.0, 6.0)]
    plan = _plan(node_bin, items, 1.5, 4.0)

    assert plan["contained"] == ["inside"]
    assert plan["straddling"] == []
    assert plan["removable"] is True


def test_an_item_crossing_the_start_blocks_the_typed_path(node_bin):
    """Removing it whole would delete media before the cut."""
    plan = _plan(node_bin, [_item("long", 0.0, 3.0)], 1.0, 2.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["long"]
    assert "cross a cut boundary" in plan["reason"]


def test_an_item_crossing_the_end_blocks_the_typed_path(node_bin):
    plan = _plan(node_bin, [_item("tail", 1.5, 9.0)], 1.0, 2.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["tail"]


def test_an_item_spanning_the_whole_range_blocks_the_typed_path(node_bin):
    """The silence-inside-one-clip case, which needs a razor the API lacks."""
    plan = _plan(node_bin, [_item("clip", 0.0, 60.0)], 10.0, 11.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["clip"]


def test_one_straddling_item_blocks_the_whole_cut(node_bin):
    items = [_item("ok", 2.0, 3.0), _item("bad", 2.5, 9.0)]
    plan = _plan(node_bin, items, 1.0, 4.0)

    assert plan["removable"] is False
    assert plan["contained"] == ["ok"]
    assert plan["straddling"] == ["bad"]


def test_boundaries_within_tolerance_count_as_contained(node_bin):
    """Frame-boundary rounding must not force the legacy path."""
    plan = _plan(node_bin, [_item("snug", 0.9999, 2.0001)], 1.0, 2.0)

    assert plan["removable"] is True
    assert plan["contained"] == ["snug"]


def test_touching_items_do_not_overlap(node_bin):
    """An item ending exactly at the cut start is untouched, not removed."""
    items = [_item("ends_at_start", 0.0, 1.0), _item("starts_at_end", 2.0, 3.0)]
    plan = _plan(node_bin, items, 1.0, 2.0)

    assert plan["contained"] == []
    assert plan["straddling"] == []
    assert plan["removable"] is False
    assert "No track item overlaps" in plan["reason"]


def test_unreadable_boundaries_block_rather_than_being_assumed_outside(node_bin):
    plan = _plan(node_bin, [_item("mystery", None, None)], 1.0, 2.0)

    assert plan["removable"] is False
    assert plan["unreadable"] == ["mystery"]
    assert "unreadable boundaries" in plan["reason"]


def test_an_empty_timeline_is_not_removable(node_bin):
    plan = _plan(node_bin, [], 1.0, 2.0)

    assert plan["removable"] is False
    assert "No track item overlaps" in plan["reason"]


def test_a_zero_length_or_reversed_range_is_refused(node_bin):
    for start, end in ((2.0, 2.0), (3.0, 1.0)):
        plan = _plan(node_bin, [_item("a", 0.0, 5.0)], start, end)
        assert plan["removable"] is False
        assert "forward interval" in plan["reason"]


def test_tolerance_is_a_millisecond_by_default(node_bin):
    plan = _plan(node_bin, [], 1.0, 2.0)

    assert plan["tolerance"] == pytest.approx(0.001)


class TestCutPathWiring:
    """The typed action has to be reached the way Premiere requires."""

    def _source(self) -> str:
        return MAIN_JS.read_text(encoding="utf-8", errors="replace")

    def test_main_imports_the_planner(self):
        assert 'from "./uxp-cut-planner.js"' in self._source()

    def test_cut_path_uses_the_typed_editor_action(self):
        source = self._source()
        assert "SequenceEditor?.getEditor" in source
        assert "createRemoveItemsAction" in source

    def test_removal_runs_inside_a_project_transaction(self):
        """Outside executeTransaction the edit would not be undoable."""
        source = self._source()
        start = source.index("async function _removeItemsWithEditor")
        body = source[start:source.index("async function _applyOneCut")]
        assert "executeTransaction" in body
        assert "compoundAction.addAction" in body
        assert "lockedAccess" in body

    def test_selection_is_built_through_the_documented_factory(self):
        source = self._source()
        assert "TrackItemSelection.createEmptySelection" in source
        assert "selection.addItem" in source

    def test_ripple_delete_survives_as_fallback(self):
        source = self._source()
        assert "_rippleDeleteFallback" in source
        assert "seq.rippleDelete(" in source

    def test_fallback_reason_reaches_the_verification_payload(self):
        """A silent downgrade to the broken path would be invisible."""
        source = self._source()
        assert "fallback_reasons" in source
        assert "methods: methodCounts" in source

    def test_ticks_go_through_the_shared_converter(self):
        """The hand-inlined 254016000000 literal was a second source of truth."""
        source = self._source()
        start = source.index("async function _rippleDeleteFallback")
        body = source[start:start + 600]
        assert "_secondsToTicks(cut.start)" in body
        assert "254016000000" not in body
