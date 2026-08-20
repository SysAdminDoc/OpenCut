"""F341 — multicam needs a cutting grammar, not just a minimum duration.

`generate_multicam_cuts` exposed one knob. The category leader for Premiere
multicam is distrusted for exactly the things a knob cannot express: it cuts
only between talking heads, it mishandles overlapping speech, and it needs one
isolated audio track per speaker. OpenCut drives switching from diarization, so
a single mixed track works — which is the differentiator these tests pin.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core.multicam import generate_multicam_cuts  # noqa: E402

# One microphone, three speakers, overlapping in places — the shape AutoPod
# cannot switch from and the one podcast recordings actually have.
MIXED_TRACK_DIARIZATION = [
    {"speaker": "host", "start": 0.0, "end": 12.0},
    {"speaker": "guest_a", "start": 12.0, "end": 25.0},
    {"speaker": "host", "start": 24.0, "end": 26.0},      # interjects over guest_a
    {"speaker": "guest_b", "start": 26.0, "end": 40.0},
    {"speaker": "host", "start": 40.0, "end": 52.0},
]


class TestDefaultsUnchanged:
    def test_no_grammar_options_behaves_as_before(self):
        result = generate_multicam_cuts(MIXED_TRACK_DIARIZATION)
        assert result["wide_cuts"] == 0
        assert all("shot" not in cut for cut in result["cuts"])
        assert result["total_cuts"] > 0

    def test_the_effective_grammar_is_reported(self):
        """A caller has to be able to show what rules actually ran."""
        grammar = generate_multicam_cuts(MIXED_TRACK_DIARIZATION)["grammar"]
        assert grammar["min_cut_duration"] == 1.0
        assert grammar["wide_track"] is None
        assert grammar["cut_on_interruption"] is True


class TestMixedTrackIsTheDifferentiator:
    def test_cuts_generate_from_a_single_mixed_audio_track(self):
        """No per-speaker isolated tracks anywhere in the input."""
        result = generate_multicam_cuts(MIXED_TRACK_DIARIZATION)

        assert result["total_cuts"] >= 4
        assert len(set(result["speaker_to_track"].values())) == 3
        for cut in result["cuts"]:
            assert cut["track"] == result["speaker_to_track"][cut["speaker"]]

    def test_each_speaker_keeps_one_camera_for_the_whole_edit(self):
        result = generate_multicam_cuts(MIXED_TRACK_DIARIZATION)
        seen = {}
        for cut in result["cuts"]:
            seen.setdefault(cut["speaker"], cut["track"])
            assert seen[cut["speaker"]] == cut["track"]


class TestInterruptionRule:
    def test_interruptions_are_cut_to_by_default(self):
        result = generate_multicam_cuts(MIXED_TRACK_DIARIZATION, min_cut_duration=0.5)
        assert any(abs(cut["time"] - 24.0) < 0.01 for cut in result["cuts"])

    def test_disabling_it_suppresses_the_overlapping_switch(self):
        """Cutting to a two-second interjection is where this looks worst."""
        result = generate_multicam_cuts(
            MIXED_TRACK_DIARIZATION, min_cut_duration=0.5, cut_on_interruption=False
        )
        assert not any(abs(cut["time"] - 24.0) < 0.01 for cut in result["cuts"])
        assert result["grammar"]["cut_on_interruption"] is False

    def test_non_overlapping_speech_is_unaffected_by_the_rule(self):
        clean = [
            {"speaker": "a", "start": 0.0, "end": 10.0},
            {"speaker": "b", "start": 10.0, "end": 20.0},
        ]
        with_rule = generate_multicam_cuts(clean, cut_on_interruption=False)
        without = generate_multicam_cuts(clean, cut_on_interruption=True)
        assert with_rule["total_cuts"] == without["total_cuts"]


class TestWideShotCadence:
    def test_a_wide_is_inserted_on_a_count_cadence(self):
        result = generate_multicam_cuts(
            MIXED_TRACK_DIARIZATION,
            wide_track=9,
            wide_every_n_cuts=2,
            wide_duration=2.0,
        )
        wides = [c for c in result["cuts"] if c.get("shot") == "wide"]
        assert wides
        assert result["wide_cuts"] == len(wides)
        assert all(c["track"] == 9 for c in wides)

    def test_the_timeline_stays_contiguous_after_insertion(self):
        """A wide must take time from the shot it splits, not add time.

        Uses non-overlapping input: the mixed-track fixture has speakers talking
        over each other by design, so its cuts are not contiguous to begin with.
        """
        clean = [
            {"speaker": "a", "start": 0.0, "end": 10.0},
            {"speaker": "b", "start": 10.0, "end": 20.0},
            {"speaker": "a", "start": 20.0, "end": 30.0},
        ]
        result = generate_multicam_cuts(
            clean, wide_track=9, wide_every_n_cuts=1, wide_duration=2.0
        )
        cuts = result["cuts"]
        assert result["wide_cuts"] > 0
        for previous, following in zip(cuts, cuts[1:]):
            end = previous["time"] + previous["duration"]
            assert abs(end - following["time"]) < 0.01, (previous, following)

    def test_a_short_shot_is_never_split(self):
        """Cutting away from a one-second line and back reads as a glitch."""
        short = [
            {"speaker": "a", "start": 0.0, "end": 2.0},
            {"speaker": "b", "start": 2.0, "end": 4.0},
        ]
        result = generate_multicam_cuts(
            short, wide_track=9, wide_every_n_cuts=1, wide_duration=2.0
        )
        assert result["wide_cuts"] == 0

    def test_cadence_requires_a_wide_track(self):
        result = generate_multicam_cuts(
            MIXED_TRACK_DIARIZATION, wide_every_n_cuts=1, wide_duration=2.0
        )
        assert result["wide_cuts"] == 0

    def test_a_time_cadence_also_triggers(self):
        result = generate_multicam_cuts(
            MIXED_TRACK_DIARIZATION,
            wide_track=9,
            wide_every_seconds=10.0,
            wide_duration=2.0,
        )
        assert result["wide_cuts"] > 0


class TestPerSpeakerFloors:
    def test_a_speaker_override_raises_that_speaker_floor_only(self):
        segments = [
            {"speaker": "host", "start": 0.0, "end": 2.0},
            {"speaker": "guest", "start": 2.0, "end": 4.0},
        ]
        result = generate_multicam_cuts(
            segments, min_cut_duration=0.5, speaker_min_duration={"host": 5.0}
        )
        speakers = {cut["speaker"] for cut in result["cuts"]}
        assert "host" not in speakers
        assert "guest" in speakers

    def test_an_unlisted_speaker_uses_the_global_floor(self):
        segments = [{"speaker": "guest", "start": 0.0, "end": 0.2}]
        result = generate_multicam_cuts(
            segments, min_cut_duration=1.0, speaker_min_duration={"host": 5.0}
        )
        assert result["total_cuts"] == 0


class TestEdgeCases:
    def test_empty_input_reports_the_grammar_anyway(self):
        result = generate_multicam_cuts([])
        assert result["total_cuts"] == 0
        assert result["wide_cuts"] == 0
        assert "grammar" in result

    def test_everything_filtered_out_is_not_a_crash(self):
        result = generate_multicam_cuts(
            [{"speaker": "a", "start": 0.0, "end": 0.1}], min_cut_duration=10.0
        )
        assert result["cuts"] == []
        assert "grammar" in result

    def test_nonsense_cadence_values_are_clamped(self):
        result = generate_multicam_cuts(
            MIXED_TRACK_DIARIZATION,
            wide_track=9,
            wide_every_n_cuts=-5,
            wide_every_seconds=-1.0,
            wide_duration=-2.0,
        )
        assert result["wide_cuts"] == 0
