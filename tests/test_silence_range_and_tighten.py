"""F325 — scope silence work to in/out points, and tighten instead of delete.

`detect_silences` took no time range, so silence work always spanned the whole
file and a sequence selection could not be honoured; the only outcomes were hard
cut or speed-up. The most-voted open Premiere idea in this area asks for exactly
range scoping plus "shorten pauses to N seconds", and Adobe's answer ("use
Text-Based Editing") was rejected by the requester.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core.silence import (  # noqa: E402
    TimeSegment,
    clamp_segments_to_range,
    tighten_silences,
)


def _segs(*pairs):
    return [TimeSegment(start=a, end=b, label="silence") for a, b in pairs]


class TestRangeScoping:
    def test_no_range_returns_everything_unchanged(self):
        segs = _segs((1.0, 2.0), (5.0, 6.0))
        assert [(s.start, s.end) for s in clamp_segments_to_range(segs)] == [
            (1.0, 2.0), (5.0, 6.0)
        ]

    def test_segments_outside_the_range_are_dropped(self):
        segs = _segs((1.0, 2.0), (5.0, 6.0), (9.0, 10.0))
        out = clamp_segments_to_range(segs, 4.0, 7.0)
        assert [(s.start, s.end) for s in out] == [(5.0, 6.0)]

    def test_a_straddling_segment_is_trimmed_not_dropped(self):
        """The part inside the selection is still silence the user asked about."""
        out = clamp_segments_to_range(_segs((3.0, 8.0)), 5.0, 7.0)
        assert [(s.start, s.end) for s in out] == [(5.0, 7.0)]

    def test_timestamps_stay_absolute_to_the_source(self):
        """Re-basing to the selection would misplace every cut on the timeline."""
        out = clamp_segments_to_range(_segs((12.0, 14.0)), 10.0, 20.0)
        assert out[0].start == 12.0 and out[0].end == 14.0

    def test_open_ended_ranges_work(self):
        segs = _segs((1.0, 2.0), (9.0, 10.0))
        assert len(clamp_segments_to_range(segs, 5.0, None)) == 1
        assert len(clamp_segments_to_range(segs, None, 5.0)) == 1

    def test_reversed_bounds_are_normalised(self):
        out = clamp_segments_to_range(_segs((5.0, 6.0)), 7.0, 4.0)
        assert [(s.start, s.end) for s in out] == [(5.0, 6.0)]

    def test_a_zero_width_range_selects_nothing(self):
        assert clamp_segments_to_range(_segs((1.0, 9.0)), 5.0, 5.0) == []

    def test_labels_survive_clamping(self):
        out = clamp_segments_to_range(_segs((1.0, 9.0)), 2.0, 3.0)
        assert out[0].label == "silence"

    def test_empty_input_is_safe(self):
        assert clamp_segments_to_range([], 1.0, 2.0) == []


class TestTightenMode:
    def test_a_long_pause_is_shortened_not_removed(self):
        plan = tighten_silences(_segs((10.0, 14.0)), target_duration=0.5)
        entry = plan[0]
        assert entry["keep_duration"] == 0.5
        assert entry["trim"] == 3.5
        # The kept part sits at the head so speech before it never moves.
        assert entry["trim_start"] == 10.5
        assert entry["trim_end"] == 14.0

    def test_a_pause_already_short_enough_is_untouched(self):
        plan = tighten_silences(_segs((3.0, 3.2)), target_duration=0.5)
        assert plan[0]["trim"] == 0.0
        assert round(plan[0]["keep_duration"], 6) == 0.2

    def test_a_zero_target_is_equivalent_to_removal(self):
        plan = tighten_silences(_segs((1.0, 3.0)), target_duration=0.0)
        assert plan[0]["trim"] == 2.0
        assert plan[0]["keep_duration"] == 0.0

    def test_a_negative_target_is_clamped_not_inverted(self):
        plan = tighten_silences(_segs((1.0, 3.0)), target_duration=-5.0)
        assert plan[0]["keep_duration"] == 0.0
        assert plan[0]["trim"] == 2.0

    def test_every_silence_gets_a_plan_entry(self):
        plan = tighten_silences(_segs((1.0, 2.0), (5.0, 9.0), (11.0, 11.1)), 0.5)
        assert len(plan) == 3
        assert [round(e["trim"], 3) for e in plan] == [0.5, 3.5, 0.0]

    def test_trim_span_never_extends_past_the_pause(self):
        for start, end in ((0.0, 1.0), (4.0, 4.25), (100.0, 130.0)):
            entry = tighten_silences(_segs((start, end)), 0.75)[0]
            assert start <= entry["trim_start"] <= entry["trim_end"] == end

    def test_empty_input_is_safe(self):
        assert tighten_silences([], 0.5) == []


class TestDetectorsAcceptTheRange:
    def test_detect_silences_exposes_the_range(self):
        import inspect

        from opencut.core.silence import detect_silences, detect_silences_vad, detect_speech

        for fn in (detect_silences, detect_silences_vad, detect_speech):
            params = inspect.signature(fn).parameters
            assert "range_start" in params, fn.__name__
            assert "range_end" in params, fn.__name__
            assert params["range_start"].default is None, fn.__name__

    def test_the_route_reads_in_out_points(self):
        source = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "opencut", "routes", "audio.py",
        )
        with open(source, "r", encoding="utf-8") as fh:
            text = fh.read()
        assert 'data.get("range_start")' in text
        assert "range_start=range_start" in text
