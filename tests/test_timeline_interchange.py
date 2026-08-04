"""Regression coverage for large Premiere timeline interchange passes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch

from opencut.export.premiere import (
    cut_ranges_to_segments,
    export_premiere_xml_from_cuts,
)
from opencut.utils.media import AudioStream, MediaInfo, VideoStream


def _two_audio_fixture() -> MediaInfo:
    return MediaInfo(
        path="C:/media/source.mp4",
        filename="source.mp4",
        duration=2000.0,
        video=VideoStream(width=1920, height=1080, fps=25.0, duration=2000.0),
        audio=AudioStream(sample_rate=48000, channels=2, bit_depth=24, duration=2000.0),
    )


def _clip_ranges(track):
    return [
        (
            int(item.findtext("start")),
            int(item.findtext("end")),
            int(item.findtext("in")),
            int(item.findtext("out")),
        )
        for item in track.findall("clipitem")
    ]


def test_cut_ranges_invert_and_merge_deterministically():
    normalized, kept = cut_ranges_to_segments(
        [
            {"start": 2, "end": 3},
            {"start": 2.5, "end": 4},
            {"start": 5, "end": 6, "duration": "not used when end is present"},
            {"start": "8", "end": "9"},
            {"start": 10, "end": 10},
        ],
        12,
    )

    assert normalized == [
        {"start": 2.0, "end": 4.0},
        {"start": 5.0, "end": 6.0},
        {"start": 8.0, "end": 9.0},
    ]
    assert [(segment.start, segment.end) for segment in kept] == [
        (0.0, 2.0),
        (4.0, 5.0),
        (6.0, 8.0),
        (9.0, 12.0),
    ]


def test_thousand_cut_interchange_preserves_video_and_two_audio_tracks(tmp_path):
    cuts = [{"start": index * 2.0, "end": index * 2.0 + 1.0} for index in range(1000)]
    output_path = tmp_path / "large-pass.xml"

    with patch("opencut.export.premiere.probe", return_value=_two_audio_fixture()):
        result = export_premiere_xml_from_cuts(
            "C:/media/source.mp4",
            cuts,
            str(output_path),
            sequence_name="Large OpenCut Pass",
        )

    assert result["requested_cuts"] == 1000
    assert result["normalized_cuts"] == 1000
    assert result["kept_segments"] == 1000
    assert result["audio_tracks"] == 2

    root = ET.parse(output_path).getroot()
    sequence = root.find("sequence")
    assert sequence is not None
    video_tracks = sequence.findall("./media/video/track")
    audio_tracks = sequence.findall("./media/audio/track")
    assert len(video_tracks) == 1
    assert len(audio_tracks) == 2
    assert sequence.findtext("./media/audio/outputs/group/numchannels") == "2"
    assert len(sequence.findall("./media/audio/outputs/group/channel")) == 2

    video_ranges = _clip_ranges(video_tracks[0])
    audio_ranges = [_clip_ranges(track) for track in audio_tracks]
    assert len(video_ranges) == 1000
    assert all(len(ranges) == 1000 for ranges in audio_ranges)
    assert audio_ranges[0] == video_ranges
    assert audio_ranges[1] == video_ranges
    assert all(start <= end and source_in <= source_out for start, end, source_in, source_out in video_ranges)

    # Every clip participates in the complete V1/A1/A2 link group. This is
    # what keeps the source in/out decisions synchronized after import.
    first_video_links = {
        link.findtext("linkclipref")
        for link in video_tracks[0].find("clipitem").findall("link")
    }
    assert first_video_links == {
        "clipitem-video-0-1",
        "clipitem-audio-0-1",
        "clipitem-audio-1-1",
    }


def test_premiere_interchange_route_reports_requested_cuts(tmp_path):
    from opencut.server import app

    app.config["TESTING"] = True
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"fixture")
    exporter_result = {
        "output_path": str(tmp_path / "source_opencut_interchange.xml"),
        "requested_cuts": 101,
        "normalized_cuts": 101,
        "kept_segments": 101,
        "audio_tracks": 2,
    }

    with patch(
        "opencut.export.premiere.export_premiere_xml_from_cuts",
        return_value=exporter_result,
    ) as exporter:
        with app.test_client() as client:
            csrf_token = client.get("/health").get_json()["csrf_token"]
            response = client.post(
                "/timeline/export-premiere-interchange",
                json={
                    "filepath": str(source_path),
                    "cuts": [{"start": 1, "end": 2}] * 101,
                    "output_dir": str(tmp_path),
                },
                headers={"X-OpenCut-Token": csrf_token},
            )

    assert response.status_code == 200
    assert response.get_json()["requested_cuts"] == 101
    exporter.assert_called_once()
