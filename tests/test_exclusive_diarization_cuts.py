"""Fixture-backed coverage for pyannote exclusive speaker cut boundaries."""

from dataclasses import dataclass

import pytest


@dataclass
class _Turn:
    start: float
    end: float


class _Annotation:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, *, yield_label=False):
        assert yield_label is True
        for start, end, speaker in self._turns:
            yield _Turn(start, end), None, speaker


class _PyannoteOutput:
    def __init__(self, regular, exclusive):
        self.speaker_diarization = regular
        self.exclusive_speaker_diarization = exclusive


def test_exclusive_annotation_wins_and_cuts_land_on_boundary():
    """Overlapping regular turns must resolve to the exclusive boundary."""
    from opencut.core.diarize import annotation_to_segments, select_diarization_annotation
    from opencut.core.multicam import generate_multicam_cuts

    regular = _Annotation([
        (0.0, 2.2, "SPEAKER_A"),
        (1.8, 4.0, "SPEAKER_B"),
    ])
    exclusive = _Annotation([
        (0.0, 2.0, "SPEAKER_A"),
        (2.0, 4.0, "SPEAKER_B"),
    ])

    annotation, source = select_diarization_annotation(
        _PyannoteOutput(regular, exclusive)
    )
    segments = annotation_to_segments(annotation)
    cuts = generate_multicam_cuts(
        [
            {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
            for segment in segments
        ],
        speaker_to_track={"SPEAKER_A": 0, "SPEAKER_B": 1},
        min_cut_duration=0.1,
    )["cuts"]

    assert source == "exclusive_speaker_diarization"
    assert [cut["speaker"] for cut in cuts] == ["SPEAKER_A", "SPEAKER_B"]
    assert cuts[1]["time"] == pytest.approx(2.0, abs=1 / 30)
    assert cuts[0]["time"] + cuts[0]["duration"] <= cuts[1]["time"]


def test_mapping_output_uses_exclusive_annotation_when_present():
    """The pyannote output adapter also accepts serialized-style mappings."""
    from opencut.core.diarize import annotation_to_segments, select_diarization_annotation

    regular = _Annotation([(0.0, 2.2, "A")])
    exclusive = _Annotation([(0.0, 2.0, "A")])
    annotation, source = select_diarization_annotation({
        "speaker_diarization": regular,
        "exclusive_speaker_diarization": exclusive,
    })

    assert source == "exclusive_speaker_diarization"
    assert annotation_to_segments(annotation)[0].end == pytest.approx(2.0)


def test_multicam_route_preserves_diarization_boundary_source():
    """The route adapter keeps provenance for response/UI reporting."""
    from opencut.core.diarize import DiarizationResult, SpeakerSegment
    from opencut.routes.video_editing import _diarization_result_to_multicam_segments

    segments, source = _diarization_result_to_multicam_segments(DiarizationResult(
        segments=[SpeakerSegment("SPEAKER_A", 0.0, 2.0)],
        speakers=["SPEAKER_A"],
        num_speakers=1,
        boundary_source="exclusive_speaker_diarization",
    ))

    assert source == "exclusive_speaker_diarization"
    assert segments == [{
        "start": 0.0,
        "end": 2.0,
        "text": "",
        "speaker": "SPEAKER_A",
    }]


def test_multicam_route_uses_pyannote_result_before_asr(monkeypatch):
    """The filepath route branch consumes the selected exclusive result."""
    import importlib

    diarize_module = importlib.import_module("opencut.core.diarize")
    video_editing = importlib.import_module("opencut.routes.video_editing")
    exclusive_result = type("Result", (), {
        "segments": [
            type("Segment", (), {"start": 0.0, "end": 2.0, "speaker": "A"})(),
            type("Segment", (), {"start": 2.0, "end": 4.0, "speaker": "B"})(),
        ],
        "boundary_source": "exclusive_speaker_diarization",
    })()
    calls = {}

    monkeypatch.setattr(diarize_module, "check_pyannote_available", lambda: True)

    def fake_diarize(filepath, config):
        calls["filepath"] = filepath
        calls["num_speakers"] = config.num_speakers
        return exclusive_result

    monkeypatch.setattr(diarize_module, "diarize", fake_diarize)

    segments, source = video_editing._pyannote_segments_for_multicam(
        "episode.wav", {"num_speakers": "2"}
    )

    assert calls == {"filepath": "episode.wav", "num_speakers": 2}
    assert source == "exclusive_speaker_diarization"
    assert [segment["start"] for segment in segments] == [0.0, 2.0]
