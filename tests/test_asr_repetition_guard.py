"""F326 — guard long-file ASR repetition loops.

Whisper-family decoders degrade on hour-plus audio into looping one phrase for
the remainder of the file. The transcript still looks plausible, and the next
stage deletes footage based on it, so an undetected loop is a data-safety
problem rather than a quality one. Detection flags; it never discards.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core.captions import (  # noqa: E402
    DECODER_COMPRESSION_RATIO_THRESHOLD,
    DECODER_NO_SPEECH_THRESHOLD,
    REPETITION_REVIEW_REASON,
    CaptionSegment,
    flag_repetition_loops,
)


def _segments(texts):
    return [
        CaptionSegment(text=t, start=float(i), end=float(i) + 1.0)
        for i, t in enumerate(texts)
    ]


class TestLoopDetection:
    def test_a_loop_is_detected_and_flagged(self):
        segs = _segments(["intro"] + ["thanks for watching"] * 6 + ["outro"])
        summary = flag_repetition_loops(segs)

        assert summary["runs"] == 1
        assert summary["longest_run"] == 6
        looped = [s for s in segs if s.human_review_recommended]
        assert len(looped) == 6
        assert all(REPETITION_REVIEW_REASON in s.review_reasons for s in looped)

    def test_clean_speech_is_untouched(self):
        segs = _segments(["one", "two", "three", "four"])
        assert flag_repetition_loops(segs)["runs"] == 0
        assert not any(s.human_review_recommended for s in segs)

    def test_short_deliberate_repetition_is_not_a_loop(self):
        """"no, no" twice is speech; flagging it would cry wolf."""
        segs = _segments(["no", "no", "and then we left"])
        assert flag_repetition_loops(segs)["runs"] == 0

    def test_detection_ignores_case_punctuation_and_spacing(self):
        segs = _segments(["Thanks for watching.", "thanks for watching", "  THANKS FOR WATCHING!  "])
        assert flag_repetition_loops(segs)["runs"] == 1

    def test_two_separate_loops_are_both_reported(self):
        segs = _segments(["a"] * 3 + ["middle"] + ["b"] * 4)
        summary = flag_repetition_loops(segs)
        assert summary["runs"] == 2
        assert summary["longest_run"] == 4
        assert summary["segments_flagged"] == 7

    def test_a_loop_at_the_very_end_is_caught(self):
        """The classic failure: it degrades and never recovers."""
        segs = _segments(["real speech", "more speech"] + ["um"] * 5)
        assert flag_repetition_loops(segs)["runs"] == 1
        assert segs[-1].human_review_recommended
        assert not segs[0].human_review_recommended

    def test_blank_segments_are_not_treated_as_a_loop(self):
        segs = _segments(["", "", "", ""])
        assert flag_repetition_loops(segs)["runs"] == 0


class TestDetectionNeverDestroys:
    def test_no_segment_is_removed_or_rewritten(self):
        texts = ["intro"] + ["looped line"] * 5
        segs = _segments(texts)
        flag_repetition_loops(segs)

        assert [s.text for s in segs] == texts
        assert len(segs) == len(texts)

    def test_existing_review_reasons_are_preserved(self):
        segs = _segments(["dup"] * 4)
        segs[0].review_reasons = ["low_confidence"]
        flag_repetition_loops(segs)

        assert "low_confidence" in segs[0].review_reasons
        assert REPETITION_REVIEW_REASON in segs[0].review_reasons

    def test_reason_is_not_duplicated_on_a_second_pass(self):
        segs = _segments(["dup"] * 4)
        flag_repetition_loops(segs)
        flag_repetition_loops(segs)
        assert segs[0].review_reasons.count(REPETITION_REVIEW_REASON) == 1


class TestEdgeCases:
    def test_empty_and_none_are_safe(self):
        assert flag_repetition_loops([])["runs"] == 0
        assert flag_repetition_loops(None)["runs"] == 0

    def test_a_nonsense_threshold_disables_detection(self):
        segs = _segments(["dup"] * 10)
        assert flag_repetition_loops(segs, run_threshold=1)["runs"] == 0
        assert not any(s.human_review_recommended for s in segs)


class TestDecoderThresholds:
    def test_thresholds_are_explicit_values_not_left_to_the_wrapper(self):
        assert DECODER_COMPRESSION_RATIO_THRESHOLD > 0
        assert 0 < DECODER_NO_SPEECH_THRESHOLD < 1
