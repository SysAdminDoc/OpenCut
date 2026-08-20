"""F314 — a caption edit must not force a whole-timeline re-render.

The planning half is pure so the boundary arithmetic is testable without
media; the render half is exercised against real FFmpeg when it is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import caption_burnin as cb  # noqa: E402


def _cue(start, end, text):
    return {"start": start, "end": end, "text": text}


# ---------------------------------------------------------------------------
# caption_change_ranges
# ---------------------------------------------------------------------------


def test_identical_captions_change_nothing():
    cues = [_cue(1, 2, "a"), _cue(5, 6, "b")]
    assert cb.caption_change_ranges(cues, list(cues)) == []


def test_edited_text_marks_the_cue_span():
    ranges = cb.caption_change_ranges([_cue(10, 12, "old")], [_cue(10, 12, "new")])

    assert len(ranges) == 1
    start, end = ranges[0]
    assert start < 10 and end > 12


def test_added_cue_is_a_change():
    ranges = cb.caption_change_ranges([], [_cue(3, 4, "hello")])

    assert len(ranges) == 1
    assert ranges[0][0] < 3 < ranges[0][1]


def test_removed_cue_is_a_change():
    """The old pixels still have to be painted over."""
    ranges = cb.caption_change_ranges([_cue(3, 4, "bye")], [])

    assert len(ranges) == 1
    assert ranges[0][0] < 3 < ranges[0][1]


def test_retimed_cue_covers_both_positions():
    ranges = cb.caption_change_ranges([_cue(3, 4, "x")], [_cue(20, 21, "x")])

    assert len(ranges) == 2
    assert ranges[0][0] < 3 and ranges[1][1] > 21


def test_untouched_cues_do_not_widen_the_change():
    old = [_cue(1, 2, "keep"), _cue(30, 31, "edit me")]
    new = [_cue(1, 2, "keep"), _cue(30, 31, "edited")]
    ranges = cb.caption_change_ranges(old, new)

    assert len(ranges) == 1
    assert ranges[0][0] > 2


def test_adjacent_edits_merge_into_one_range():
    old = [_cue(10, 11, "a"), _cue(11, 12, "b")]
    new = [_cue(10, 11, "A"), _cue(11, 12, "B")]

    assert len(cb.caption_change_ranges(old, new)) == 1


def test_duplicate_cues_are_counted_not_deduplicated():
    """Two identical cues dropping to one is still an edit."""
    ranges = cb.caption_change_ranges([_cue(1, 2, "x"), _cue(1, 2, "x")], [_cue(1, 2, "x")])

    assert len(ranges) == 1


# ---------------------------------------------------------------------------
# build_incremental_plan
# ---------------------------------------------------------------------------

KEYS = [float(k) for k in range(0, 21, 2)]


def test_plan_snaps_outward_to_keyframes():
    plan = cb.build_incremental_plan(20.0, [(11.9, 14.1)], KEYS)

    assert plan.incremental is True
    assert plan.encode_ranges == [(10.0, 16.0)]
    assert plan.copy_ranges == [(0.0, 10.0), (16.0, 20.0)]


def test_plan_copy_and_encode_tile_the_whole_timeline():
    plan = cb.build_incremental_plan(20.0, [(5.0, 6.0), (15.0, 16.0)], KEYS)
    covered = sorted(plan.encode_ranges + plan.copy_ranges)

    assert covered[0][0] == 0.0
    assert covered[-1][1] == 20.0
    for (_, end), (nxt, _) in zip(covered, covered[1:]):
        assert end == pytest.approx(nxt)
    assert plan.encode_duration + plan.copy_duration == pytest.approx(20.0)


def test_unchanged_captions_fall_back_rather_than_rendering_nothing():
    plan = cb.build_incremental_plan(20.0, [], KEYS)

    assert plan.incremental is False
    assert "unchanged" in plan.fallback_reason


def test_a_source_without_keyframes_falls_back():
    plan = cb.build_incremental_plan(20.0, [(5.0, 6.0)], [0.0])

    assert plan.incremental is False
    assert "keyframes" in plan.fallback_reason


def test_unknown_duration_falls_back():
    plan = cb.build_incremental_plan(0.0, [(5.0, 6.0)], KEYS)

    assert plan.incremental is False
    assert "duration" in plan.fallback_reason


def test_widespread_edits_fall_back_to_a_whole_file_render():
    """Segmenting is only worth it when most of the file survives."""
    changes = [(t + 0.5, t + 1.5) for t in range(0, 20, 2)]
    plan = cb.build_incremental_plan(20.0, changes, KEYS)

    assert plan.incremental is False
    assert "cheaper" in plan.fallback_reason


def test_an_edit_past_the_last_keyframe_ends_at_the_duration():
    plan = cb.build_incremental_plan(20.0, [(19.5, 19.9)], KEYS)

    assert plan.encode_ranges[-1][1] == 20.0


def test_plan_serialises_for_reporting():
    payload = cb.build_incremental_plan(20.0, [(11.9, 14.1)], KEYS).as_dict()

    assert payload["incremental"] is True
    assert payload["encode_ranges"] == [[10.0, 16.0]]
    assert payload["fallback_reason"] == ""


def test_missing_previous_render_is_reported_not_guessed(tmp_path):
    plan = cb.plan_incremental_reburn(
        str(tmp_path / "source.mp4"),
        [_cue(1, 2, "a")],
        [_cue(1, 2, "b")],
        previous_render="",
        duration=20.0,
    )

    assert plan.incremental is False
    assert "previous render" in plan.fallback_reason


# ---------------------------------------------------------------------------
# Real render
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_fixture(path, seconds=20):
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:size=320x240:rate=10:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-g", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )


def _frame_hashes(path):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
         "-map", "0:v", "-f", "framehash", "-hash", "md5", str(path) + ".hash"],
        check=True, capture_output=True,
    )
    rows = []
    with open(str(path) + ".hash", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 6:
                rows.append(parts[-1])
    return rows


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/ffprobe unavailable")
def test_reburn_leaves_untouched_regions_bit_identical(tmp_path):
    source = tmp_path / "source.mp4"
    _make_fixture(source)
    old = [_cue(2.0, 4.0, "ONE"), _cue(12.0, 14.0, "TWO")]
    new = [_cue(2.0, 4.0, "ONE"), _cue(12.0, 14.0, "TWO EDITED")]

    first = tmp_path / "v1.mp4"
    cb.burnin_segments(str(source), old, output_path=str(first))

    report = {}
    second = tmp_path / "v2.mp4"
    cb.burnin_segments(
        str(source), new,
        output_path=str(second),
        previous_render=str(first),
        previous_segments=old,
        render_report=report,
    )

    assert report["incremental"] is True, report
    assert report["encode_ranges"] == [[10.0, 16.0]]

    before, after = _frame_hashes(first), _frame_hashes(second)
    # Frames 0-10s and 16s-end are copied, so they must match exactly.
    assert before[:100] == after[:100]
    assert before[-40:] == after[-40:]
    # And the edited cue must actually differ.
    assert before[115:140] != after[115:140]


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/ffprobe unavailable")
def test_burnin_output_carries_keyframes_a_later_edit_can_cut_on(tmp_path):
    """x264 defaults can leave a short render with one keyframe."""
    from opencut.core.smart_render import _get_keyframes

    source = tmp_path / "source.mp4"
    _make_fixture(source)
    burned = tmp_path / "burned.mp4"
    cb.burnin_segments(str(source), [_cue(2.0, 4.0, "ONE")], output_path=str(burned))

    keyframes = _get_keyframes(str(burned))
    assert len(keyframes) >= 5, keyframes


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/ffprobe unavailable")
def test_unchanged_captions_take_the_whole_file_path_and_say_so(tmp_path):
    source = tmp_path / "source.mp4"
    _make_fixture(source)
    cues = [_cue(2.0, 4.0, "ONE")]

    first = tmp_path / "v1.mp4"
    cb.burnin_segments(str(source), cues, output_path=str(first))

    report = {}
    second = tmp_path / "v2.mp4"
    cb.burnin_segments(
        str(source), cues,
        output_path=str(second),
        previous_render=str(first),
        previous_segments=cues,
        render_report=report,
    )

    assert report["incremental"] is False
    assert "unchanged" in report["fallback_reason"]
    assert os.path.isfile(second)
