"""The cut path in both panels: which clips a range touches, and what it does to them.

`Sequence.rippleDelete()` is absent from the 26.3 typings and is reported to
return success while changing nothing on that host, so the UXP cut path routes
through `SequenceEditor.createRemoveItemsAction` (F349). That action removes
whole track items, so F351 added boundary trims for the clips a range only
partly covers, and F353 carried the same rule into the CEP host, which had the
same gap. These tests pin the boundary rule, the trim arithmetic, the check
that a trim did not turn out to be a move, and the agreement between the two
panels — a reviewed cut has to produce one timeline, not two.
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


def _run_node(node_bin: str, body: str) -> dict:
    program = textwrap.dedent(
        f"""
        const url = require('url').pathToFileURL({json.dumps(str(PLANNER))}).href;
        import(url).then((mod) => {{
            {body}
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


def _plan(node_bin: str, items, start, end, tolerance=None) -> dict:
    args = json.dumps([items, start, end] + ([tolerance] if tolerance is not None else []))
    return _run_node(
        node_bin,
        f"""
        const plan = mod.planCutRemoval(...{args});
        process.stdout.write(JSON.stringify({{
            contained: plan.contained.map((i) => i.id),
            straddling: plan.straddling.map((i) => i.id),
            unreadable: plan.unreadable.map((i) => i.id),
            trims: plan.trims.map((t) => ({{
                id: t.item.id, kind: t.kind, to: t.to, source: t.source,
            }})),
            blocked: plan.blocked.map((b) => ({{id: b.item.id, reason: b.reason}})),
            removable: plan.removable,
            reason: plan.reason,
            expected: mod.expectedPostState(plan),
            tolerance: mod.CUT_BOUNDARY_TOLERANCE_SECONDS,
        }}));
        """,
    )


def _verify(node_bin: str, expected, observed) -> list:
    args = json.dumps([expected, observed])
    return _run_node(
        node_bin,
        f"""
        process.stdout.write(JSON.stringify({{
            mismatches: mod.verifyPostState(...{args}),
        }}));
        """,
    )["mismatches"]


def _item(id_, start, end):
    return {"id": id_, "start": start, "end": end}


def _clip(id_, start, end, in_point=0.0, speed=1.0):
    """A track item whose source points are readable, so a trim can be planned."""
    return {
        "id": id_,
        "start": start,
        "end": end,
        "inPoint": in_point,
        "outPoint": in_point + (end - start),
        "speed": speed,
    }


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


def test_an_item_enclosing_the_range_blocks_the_typed_path(node_bin):
    """Cutting a hole in the middle of one item needs a razor 26.3 lacks."""
    plan = _plan(node_bin, [_clip("long", 0.0, 3.0)], 1.0, 2.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["long"]
    assert "cross a cut boundary" in plan["reason"]
    assert "razor" in plan["blocked"][0]["reason"]


def test_an_item_spanning_the_whole_range_blocks_the_typed_path(node_bin):
    """The silence-inside-one-clip case, which needs a razor the API lacks."""
    plan = _plan(node_bin, [_clip("clip", 0.0, 60.0)], 10.0, 11.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["clip"]


def test_a_straddling_item_without_source_points_blocks_the_cut(node_bin):
    """A trim whose new in/out cannot be stated cannot be verified either."""
    plan = _plan(node_bin, [_item("tail", 1.5, 9.0)], 1.0, 2.0)

    assert plan["removable"] is False
    assert plan["straddling"] == ["tail"]
    assert plan["trims"] == []
    assert "source in/out points" in plan["blocked"][0]["reason"]


def test_one_untrimmable_item_blocks_the_whole_cut(node_bin):
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


class TestBoundaryTrims:
    """F351 — an item crossing one boundary is trimmed, not left uncut.

    `createRemoveItemsAction` takes whole items, so before F351 any range that
    touched part of a clip fell through to `rippleDelete`, which the 26.3 host
    reports applying while changing nothing. Trimming the item back to the
    boundary and pulling its source point in by the same amount expresses the
    cut through typed actions without touching the media the user kept.
    """

    def test_an_item_running_past_the_end_is_trimmed_forward(self, node_bin):
        plan = _plan(node_bin, [_clip("tail", 1.5, 9.0, in_point=5.0)], 1.0, 2.0)

        assert plan["removable"] is True
        assert plan["blocked"] == []
        trim = plan["trims"][0]
        assert trim["kind"] == "tail"
        assert trim["to"] == {"start": 2.0, "end": 9.0}
        # The clip loses its first half-second, so playback resumes half a
        # second later in the source.
        assert trim["source"]["inPoint"] == pytest.approx(5.5)
        assert trim["source"]["outPoint"] == pytest.approx(12.5)

    def test_an_item_starting_before_the_range_is_trimmed_back(self, node_bin):
        plan = _plan(node_bin, [_clip("head", 0.0, 3.0, in_point=10.0)], 2.0, 5.0)

        assert plan["removable"] is True
        trim = plan["trims"][0]
        assert trim["kind"] == "head"
        assert trim["to"] == {"start": 0.0, "end": 2.0}
        assert trim["source"]["inPoint"] == pytest.approx(10.0)
        assert trim["source"]["outPoint"] == pytest.approx(12.0)

    def test_a_trim_preserves_duration_against_source_range(self, node_bin):
        """Off-by-one here plays the wrong frames rather than failing loudly."""
        for clip, start, end in (
            (_clip("tail", 1.5, 9.0, in_point=5.0), 1.0, 2.0),
            (_clip("head", 0.0, 3.0, in_point=10.0), 2.0, 5.0),
        ):
            trim = _plan(node_bin, [clip], start, end)["trims"][0]
            sequence_span = trim["to"]["end"] - trim["to"]["start"]
            source_span = trim["source"]["outPoint"] - trim["source"]["inPoint"]
            assert source_span == pytest.approx(sequence_span)

    def test_trims_and_removals_can_share_one_cut(self, node_bin):
        items = [_clip("whole", 2.0, 3.0), _clip("tail", 3.0, 9.0, in_point=0.0)]
        plan = _plan(node_bin, items, 1.0, 4.0)

        assert plan["removable"] is True
        assert plan["contained"] == ["whole"]
        assert [t["id"] for t in plan["trims"]] == ["tail"]

    def test_a_retimed_clip_is_refused(self, node_bin):
        """Source points do not advance a second per second on a speed ramp."""
        plan = _plan(node_bin, [_clip("fast", 1.5, 9.0, in_point=5.0, speed=2.0)], 1.0, 2.0)

        assert plan["removable"] is False
        assert "retimed" in plan["blocked"][0]["reason"]

    def test_an_unidentifiable_item_is_not_trimmed(self, node_bin):
        """A trim leaves the item behind, so the read-back has to find it again."""
        anonymous = _clip("", 1.5, 9.0, in_point=5.0)
        plan = _plan(node_bin, [anonymous], 1.0, 2.0)

        assert plan["removable"] is False
        assert "cannot be identified" in plan["blocked"][0]["reason"]

    def test_the_expected_post_state_names_removals_and_trim_boundaries(self, node_bin):
        items = [_clip("whole", 2.0, 3.0), _clip("tail", 3.0, 9.0)]
        expected = _plan(node_bin, items, 1.0, 4.0)["expected"]

        assert expected["removed"] == ["whole"]
        assert expected["trimmed"] == [
            {"id": "tail", "kind": "tail", "start": 4.0, "end": 9.0}
        ]

    def test_a_plan_that_landed_exactly_reports_no_mismatch(self, node_bin):
        expected = {"removed": ["whole"], "trimmed": [{"id": "tail", "start": 4.0, "end": 9.0}]}
        observed = [{"id": "tail", "start": 4.0, "end": 9.0}, {"id": "elsewhere", "start": 20.0, "end": 21.0}]

        assert _verify(node_bin, expected, observed) == []

    def test_a_move_is_caught_where_a_trim_was_expected(self, node_bin):
        """The failure this check exists for: setStart shifting instead of trimming.

        A moved item keeps its duration, so its end lands where a trim never
        would. Without the end in the comparison the two are indistinguishable.
        """
        expected = {"removed": [], "trimmed": [{"id": "tail", "start": 4.0, "end": 9.0}]}
        moved = [{"id": "tail", "start": 4.0, "end": 10.0}]

        mismatches = _verify(node_bin, expected, moved)

        assert [m["kind"] for m in mismatches] == ["trim_landed_elsewhere"]
        assert mismatches[0]["observed"] == {"start": 4.0, "end": 10.0}

    def test_an_item_that_survived_removal_is_reported(self, node_bin):
        expected = {"removed": ["whole"], "trimmed": []}

        mismatches = _verify(node_bin, expected, [{"id": "whole", "start": 2.0, "end": 3.0}])

        assert [m["kind"] for m in mismatches] == ["not_removed"]

    def test_a_trimmed_item_that_disappeared_is_reported(self, node_bin):
        """Trimming must never remove the item; that is the media being kept."""
        expected = {"removed": [], "trimmed": [{"id": "tail", "start": 4.0, "end": 9.0}]}

        mismatches = _verify(node_bin, expected, [])

        assert [m["kind"] for m in mismatches] == ["trimmed_item_vanished"]

    def test_frame_rounding_does_not_read_as_a_failed_trim(self, node_bin):
        expected = {"removed": [], "trimmed": [{"id": "tail", "start": 4.0, "end": 9.0}]}
        rounded = [{"id": "tail", "start": 4.0005, "end": 8.9995}]

        assert _verify(node_bin, expected, rounded) == []


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
        body = self._write_body()
        assert "executeTransaction" in body
        assert "compoundAction.addAction" in body
        assert "lockedAccess" in body

    def _write_body(self) -> str:
        source = self._source()
        start = source.index("async function _writeCutWithEditor")
        return source[start:source.index("function _describeMismatches")]

    def test_trims_and_removals_share_one_transaction(self):
        """Two transactions could leave the timeline trimmed but not cut."""
        body = self._write_body()
        transaction = body[body.index("const run = () =>"):body.index("let accepted = false;")]
        assert "createSetEndAction" in transaction
        assert "createSetStartAction" in transaction
        assert "createRemoveItemsAction" in transaction

    def test_a_trim_writes_both_the_boundary_and_the_source_point(self):
        body = self._write_body()
        assert "createSetOutPointAction" in body
        assert "createSetInPointAction" in body

    def test_removal_does_not_ripple(self):
        """A rippling removal would shift the very items the read-back checks,
        and the CEP host's Clip.remove(false, true) does not ripple either."""
        body = self._write_body()
        assert "createRemoveItemsAction(selection, false, mediaType)" in body

    def test_the_write_is_checked_against_the_plan_it_promised(self):
        body = self._write_body()
        assert "verifyPostState(expectedPostState(plan)" in body
        assert "verified: false" in body

    def test_a_disproved_plan_does_not_retry_through_ripple_delete(self):
        """The timeline has already been edited; repeating the range compounds it."""
        source = self._source()
        start = source.index("async function _applyOneCut")
        body = source[start:source.index("async function _rippleDeleteFallback")]
        assert 'if (typed.verified === false) return { method: "failed", note: typed.reason };' in body

    def test_a_failed_cut_stops_the_batch(self):
        source = self._source()
        start = source.index("async function applyCuts")
        assert 'if (outcome.method === "failed") break;' in source[start:start + 3000]

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

    def test_skipped_and_failed_cuts_are_not_counted_as_written(self):
        source = self._source()
        assert "const written = sorted.length - methodCounts.skipped - methodCounts.failed;" in source
        assert "{ ok: status !== \"failed\", applied: written }" in source

    def test_a_boundary_crossing_item_is_not_disabled_whole(self):
        """Disabling it would mute media outside the range; trimming would delete it."""
        source = self._source()
        start = source.index("async function _applyOneCut")
        body = source[start:source.index("async function _rippleDeleteFallback")]
        disable_branch = body[body.index("if (disableMode) {"):body.index("if (plan.removable) {")]
        assert "if (plan.trims.length) {" in disable_branch


CEP_HOST = REPO_ROOT / "extension" / "com.opencut.panel" / "host" / "index.jsx"


def _extract_function(source: str, name: str) -> str:
    """Lift one function out of the ExtendScript host by brace matching.

    The host only runs inside Premiere, but its cut planning is pure, so the
    shipped source can be exercised directly instead of asserted against as a
    string.
    """
    start = source.index(f"function {name}(")
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _run_cep(node_bin: str, functions, body: str) -> dict:
    source = CEP_HOST.read_text(encoding="utf-8", errors="replace")
    lifted = "\n".join(_extract_function(source, name) for name in functions)
    program = textwrap.dedent(
        """
        function _ocLog() {}
        function Time() { this.seconds = null; this.ticks = null; }
        var OC_CUT_TOLERANCE_SECONDS = 0.01;
        %s
        %s
        """
    ) % (lifted, body)
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


def _cep_clip(start, end, in_point=0.0, speed=1.0, name="clip"):
    return {
        "name": name,
        "start": {"seconds": start},
        "end": {"seconds": end},
        "inPoint": {"seconds": in_point},
        "outPoint": {"seconds": in_point + (end - start)},
        "speed": speed,
    }


class TestCepBoundaryTrims:
    """F353 — the CEP host has to cut the same ranges the UXP path does.

    It previously removed only clips whose bounds fell entirely inside the
    range, so the same reviewed cut produced two different timelines depending
    on which panel was open, and the older, more widely installed one left the
    cut uncut without saying so.
    """

    def _plan(self, node_bin, clip, start, end):
        return _run_cep(
            node_bin,
            ["_ocPlanClipCut"],
            f"process.stdout.write(JSON.stringify(_ocPlanClipCut({json.dumps(clip)}, {start}, {end})));",
        )

    def test_a_covered_clip_is_removed(self, node_bin):
        assert self._plan(node_bin, _cep_clip(2.0, 3.0), 1.0, 4.0)["action"] == "remove"

    def test_a_clip_outside_the_range_is_untouched(self, node_bin):
        assert self._plan(node_bin, _cep_clip(5.0, 6.0), 1.0, 4.0)["action"] == "none"

    def test_a_clip_running_past_the_end_is_trimmed_forward(self, node_bin):
        plan = self._plan(node_bin, _cep_clip(1.5, 9.0, in_point=5.0), 1.0, 2.0)

        assert plan["action"] == "trim_tail"
        assert plan["start"] == pytest.approx(2.0)
        assert plan["end"] == pytest.approx(9.0)
        assert plan["inPoint"] == pytest.approx(5.5)

    def test_a_clip_starting_before_the_range_is_trimmed_back(self, node_bin):
        plan = self._plan(node_bin, _cep_clip(0.0, 3.0, in_point=10.0), 2.0, 5.0)

        assert plan["action"] == "trim_head"
        assert plan["start"] == pytest.approx(0.0)
        assert plan["end"] == pytest.approx(2.0)
        assert plan["outPoint"] == pytest.approx(12.0)

    def test_a_trim_preserves_duration_against_source_range(self, node_bin):
        for clip, start, end in (
            (_cep_clip(1.5, 9.0, in_point=5.0), 1.0, 2.0),
            (_cep_clip(0.0, 3.0, in_point=10.0), 2.0, 5.0),
        ):
            plan = self._plan(node_bin, clip, start, end)
            assert plan["outPoint"] - plan["inPoint"] == pytest.approx(plan["end"] - plan["start"])

    def test_a_clip_enclosing_the_range_is_refused(self, node_bin):
        plan = self._plan(node_bin, _cep_clip(0.0, 60.0), 10.0, 11.0)

        assert plan["action"] == "blocked"
        assert "razor" in plan["reason"]

    def test_a_retimed_clip_is_refused(self, node_bin):
        plan = self._plan(node_bin, _cep_clip(1.5, 9.0, speed=2.0), 1.0, 2.0)

        assert plan["action"] == "blocked"
        assert "retimed" in plan["reason"]

    def test_unreadable_source_points_are_refused(self, node_bin):
        clip = _cep_clip(1.5, 9.0)
        clip["inPoint"] = None
        plan = self._plan(node_bin, clip, 1.0, 2.0)

        assert plan["action"] == "blocked"
        assert "source in/out points" in plan["reason"]

    def test_frame_rounding_keeps_a_snug_clip_on_the_removal_path(self, node_bin):
        assert self._plan(node_bin, _cep_clip(0.995, 2.005), 1.0, 2.0)["action"] == "remove"

    def test_the_two_panels_agree_on_every_case(self, node_bin):
        """The whole point of F353: one reviewed cut, one outcome."""
        cases = [
            ((2.0, 3.0, 0.0, 1.0), 1.0, 4.0, "remove"),
            ((5.0, 6.0, 0.0, 1.0), 1.0, 4.0, "none"),
            ((1.5, 9.0, 5.0, 1.0), 1.0, 2.0, "trim_tail"),
            ((0.0, 3.0, 10.0, 1.0), 2.0, 5.0, "trim_head"),
            ((0.0, 60.0, 0.0, 1.0), 10.0, 11.0, "blocked"),
            ((1.5, 9.0, 5.0, 2.0), 1.0, 2.0, "blocked"),
        ]
        for (c_start, c_end, in_point, speed), start, end, expected in cases:
            cep = self._plan(
                node_bin, _cep_clip(c_start, c_end, in_point=in_point, speed=speed), start, end
            )["action"]
            uxp_plan = _plan(
                node_bin,
                [_clip("x", c_start, c_end, in_point=in_point, speed=speed)],
                start,
                end,
            )
            if expected == "remove":
                uxp = "remove" if uxp_plan["contained"] == ["x"] else "?"
            elif expected == "none":
                uxp = "none" if not uxp_plan["contained"] and not uxp_plan["straddling"] else "?"
            elif expected == "blocked":
                uxp = "blocked" if uxp_plan["blocked"] else "?"
            else:
                uxp = f"trim_{uxp_plan['trims'][0]['kind']}" if uxp_plan["trims"] else "?"
            assert cep == expected, (c_start, c_end, start, end)
            assert uxp == expected, (c_start, c_end, start, end)

    def test_a_move_is_caught_where_a_trim_was_expected(self, node_bin):
        """A moved clip keeps its duration, so its end gives it away."""
        result = _run_cep(
            node_bin,
            ["_ocVerifyCutPlans"],
            """
            var trims = [{id: "video|0||clip", start: 4.0, end: 9.0}];
            process.stdout.write(JSON.stringify({
                landed: _ocVerifyCutPlans(trims, [], {"video|0||clip": {start: 4.0, end: 9.0}}),
                moved: _ocVerifyCutPlans(trims, [], {"video|0||clip": {start: 4.0, end: 10.0}}),
                vanished: _ocVerifyCutPlans(trims, [], {}),
                survived: _ocVerifyCutPlans([], ["video|0||gone"], {"video|0||gone": {start: 0, end: 1}}),
                unreadable_timeline: _ocVerifyCutPlans(trims, [], null)
            }));
            """,
        )

        assert result["landed"] == []
        assert "read back 4.000-10.000s" in result["moved"][0]
        assert "disappeared" in result["vanished"][0]
        assert "still present after removal" in result["survived"][0]
        # Nothing readable is not the same as something wrong.
        assert result["unreadable_timeline"] == []


class TestCepNonDestructiveCutMode:
    def _source(self) -> str:
        return CEP_HOST.read_text(encoding="utf-8", errors="replace")

    def test_host_accepts_both_the_legacy_array_and_a_mode_object(self):
        source = self._source()
        assert "parsedPayload.length === undefined && parsedPayload.cuts" in source
        assert 'var disableOnly = cutMode === "disable";' in source

    def test_disable_sets_the_clip_flag_instead_of_removing(self):
        source = self._source()
        assert "clip.disabled = true;" in source
        assert "clip.remove(false, true);" in source

    def test_disable_mode_refuses_a_boundary_crossing_clip(self):
        """Trimming it would delete the media disable mode exists to keep."""
        source = self._source()
        start = source.index("function ocApplySequenceCuts")
        body = source[start:source.index("var afterClips = _ocSequenceClipSnapshot(seq);", start)]
        assert "disabling it whole would mute media outside the range" in body

    def test_removal_still_does_not_ripple(self):
        """Both panels leave the cut range empty without shifting anything."""
        assert "clip.remove(false, true);" in self._source()

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
