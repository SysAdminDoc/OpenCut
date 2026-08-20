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


class TestNonDestructiveCutMode:
    """F339 — reviewed cuts must have a reversible outcome."""

    def _source(self) -> str:
        return MAIN_JS.read_text(encoding="utf-8", errors="replace")

    def test_apply_cuts_accepts_a_mode(self):
        source = self._source()
        assert 'async function applyCuts(cuts, mode = "delete")' in source
        assert 'String(mode || "delete").toLowerCase() === "disable"' in source

    def test_disable_uses_the_undoable_set_disabled_action(self):
        source = self._source()
        assert "createSetDisabledAction(true)" in source
        start = source.index("async function _disableItemsWithEditor")
        body = source[start:source.index("async function _applyOneCut")]
        assert "executeTransaction" in body
        assert "compoundAction.addAction" in body

    def test_disable_mode_never_falls_back_to_deleting(self):
        """A fallback that removes media is not a degraded form of 'keep it'."""
        source = self._source()
        start = source.index("async function _applyOneCut")
        body = source[start:source.index("async function _rippleDeleteFallback")]
        disable_branch = body[body.index("if (disableMode) {"):body.index("if (plan.removable) {")]
        assert "_rippleDeleteFallback" not in disable_branch
        assert '"skipped"' in disable_branch

    def test_a_boundary_crossing_range_is_skipped_not_deleted(self):
        source = self._source()
        start = source.index("async function _applyOneCut")
        body = source[start:source.index("async function _rippleDeleteFallback")]
        assert "if (!plan.removable) return { method: \"skipped\", note: plan.reason };" in body

    def test_mode_reaches_the_host_dispatch(self):
        source = self._source()
        start = source.index('case "ocApplySequenceCuts"')
        assert "mode" in source[start:start + 200]

    def test_disabled_state_is_part_of_the_read_back_fingerprint(self):
        """Otherwise disabling reads back as 'nothing changed' and fails closed."""
        verifier = (UXP_DIR / "uxp-host-write-verification.js").read_text(encoding="utf-8")
        assert "isDisabled" in verifier
        assert "${disabled}" in verifier

    def test_skipped_cuts_are_not_counted_as_written(self):
        source = self._source()
        assert "reported: sorted.length - methodCounts.skipped" in source


class TestCepNonDestructiveCutMode:
    def _source(self) -> str:
        return (
            REPO_ROOT / "extension" / "com.opencut.panel" / "host" / "index.jsx"
        ).read_text(encoding="utf-8", errors="replace")

    def test_host_accepts_both_the_legacy_array_and_a_mode_object(self):
        source = self._source()
        assert "parsedPayload.length === undefined && parsedPayload.cuts" in source
        assert 'var disableOnly = cutMode === "disable";' in source

    def test_disable_sets_the_clip_flag_instead_of_removing(self):
        source = self._source()
        assert "if (disableOnly) { vClip.disabled = true; } else { vClip.remove(false, true); }" in source
        assert "if (disableOnly) { aClip.disabled = true; } else { aClip.remove(false, true); }" in source

    def test_clip_fingerprint_carries_disabled_state(self):
        source = self._source()
        assert "disabledState" in source

    def test_panel_exposes_the_mode_and_refuses_the_interchange_path(self):
        client = (
            REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
        ).read_text(encoding="utf-8", errors="replace")
        html = (
            REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"
        ).read_text(encoding="utf-8", errors="replace")
        assert 'id="timelineCutMode"' in html
        assert "function getTimelineCutMode()" in client
        # The interchange path re-imports a razored timeline and cannot express
        # "leave the clip in place but disabled", so it must refuse, not delete.
        assert "timeline.disable_mode_interchange" in client
