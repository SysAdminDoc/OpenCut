"""Regression coverage for Kdenlive/Shotcut MLT interchange."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from opencut.core.silence import TimeSegment
from opencut.export.mlt_export import export_mlt, export_mlt_from_cuts
from opencut.utils.media import AudioStream, MediaInfo, VideoStream


REFERENCE = Path(__file__).parent / "fixtures" / "mlt_reference.mlt"


def _media_fixture() -> MediaInfo:
    return MediaInfo(
        path="C:/media/source.mp4",
        filename="source.mp4",
        duration=12.0,
        video=VideoStream(width=1920, height=1080, fps=25.0, duration=12.0),
        audio=AudioStream(sample_rate=48000, channels=2, duration=12.0),
    )


def _properties(node: ET.Element) -> dict[str, str]:
    return {
        prop.attrib["name"]: prop.text or ""
        for prop in node.findall("property")
    }


def _signature(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    profile = root.find("profile")
    assert profile is not None
    playlist = root.find("playlist[@id='playlist0']")
    assert playlist is not None
    sources = []
    for node in root:
        if node.tag not in {"chain", "producer"}:
            continue
        props = _properties(node)
        sources.append(
            (
                node.tag,
                node.attrib["id"],
                node.attrib["in"],
                node.attrib["out"],
                props.get("mlt_service"),
                props.get("resource"),
                props.get("opencut:source_in"),
                props.get("opencut:source_out"),
                props.get("opencut:output_frames"),
                props.get("opencut:speed"),
            )
        )
    entries = []
    filters = []
    for entry in playlist.findall("entry"):
        entries.append((entry.attrib["producer"], entry.attrib["in"], entry.attrib["out"]))
        for filter_node in entry.findall("filter"):
            props = _properties(filter_node)
            filters.append(
                (
                    filter_node.attrib["in"],
                    filter_node.attrib["out"],
                    props.get("mlt_service"),
                    props.get("level"),
                )
            )
    tractor = root.find("tractor[@id='tractor0']")
    assert tractor is not None
    return {
        "profile": {
            key: profile.attrib[key]
            for key in ("width", "height", "frame_rate_num", "frame_rate_den")
        },
        "sources": sources,
        "entries": entries,
        "filters": filters,
        "tractor": (tractor.attrib["in"], tractor.attrib["out"]),
    }


def test_mlt_reference_preserves_speed_volume_and_frame_boundaries(tmp_path):
    output_path = tmp_path / "generated.mlt"
    segments = [
        {"start": 0.0, "end": 2.0, "volume_keyframes": [(0.0, 1.0), (1.0, 0.5)]},
        {"start": 4.0, "end": 7.0, "speed": 2.0},
    ]

    with (
        patch("opencut.export.mlt_export.probe", return_value=_media_fixture()),
        patch("opencut.export.mlt_export._file_to_url", return_value="file:///C:/media/source.mp4"),
    ):
        result = export_mlt(
            "C:/media/source.mp4",
            segments,
            output_path,
            sequence_name="Reference Edit",
        )

    assert result == {
        "output_path": str(output_path),
        "format": "mlt",
        "segments": 2,
        "duration_frames": 88,
        "framerate": 25.0,
        "speed_changes": 1,
        "volume_keyframes": 2,
    }
    assert _signature(output_path) == _signature(REFERENCE)


def test_mlt_export_accepts_time_segments_and_single_mapping_keyframe(tmp_path):
    output_path = tmp_path / "segments.mlt"
    with patch("opencut.export.mlt_export.probe", return_value=_media_fixture()):
        result = export_mlt(
            "C:/media/source.mp4",
            [
                TimeSegment(1.0, 2.0, "speech"),
                {"start": 3.0, "duration": 1.0, "volume": {"time": 0.0, "db": -6}},
            ],
            output_path,
        )

    assert result["segments"] == 2
    root = ET.parse(output_path).getroot()
    assert root.find(".//property[@name='level']").text == "-6"


def test_mlt_export_from_cuts_reports_inverted_ranges(tmp_path):
    output_path = tmp_path / "cuts.mlt"
    with patch("opencut.export.mlt_export.probe", return_value=_media_fixture()):
        result = export_mlt_from_cuts(
            "C:/media/source.mp4",
            [{"start": 2.0, "end": 3.0}],
            output_path,
            total_duration=5.0,
        )

    assert result["requested_cuts"] == 1
    assert result["normalized_cuts"] == 1
    assert result["kept_segments"] == 2
    assert result["segments"] == 2


def _route_client():
    from opencut.server import app

    app.config["TESTING"] = True
    client = app.test_client()
    token = client.get("/health").get_json()["csrf_token"]
    return client, {"X-OpenCut-Token": token, "Content-Type": "application/json"}


def test_mlt_route_exports_cuts_to_validated_output_path(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    client, headers = _route_client()
    expected = {
        "output_path": str(tmp_path / "source_opencut.mlt"),
        "format": "mlt",
        "segments": 2,
        "duration_frames": 50,
    }
    with patch(
        "opencut.export.mlt_export.export_mlt_from_cuts",
        return_value=expected,
    ) as exporter:
        response = client.post(
            "/timeline/export-mlt",
            headers=headers,
            json={
                "filepath": str(source),
                "cuts": [{"start": 1, "end": 2}],
                "output_dir": str(tmp_path),
            },
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["format"] == "mlt"
    assert body["mode"] == "cuts"
    assert body["output_path"] == expected["output_path"]
    exporter.assert_called_once()


def test_mlt_route_rejects_unknown_mode(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    client, headers = _route_client()
    response = client.post(
        "/timeline/export-mlt",
        headers=headers,
        json={"filepath": str(source), "mode": "markers", "markers": []},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "mode must be cuts or segments"


def test_silence_cli_exports_mlt(tmp_path):
    from opencut import cli as cli_module

    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    summary = {
        "original_formatted": "00:10",
        "kept_formatted": "00:02",
        "removed_formatted": "00:08",
        "reduction_percent": 80.0,
        "segments_count": 1,
    }
    with (
        patch("opencut.core.silence.detect_speech", return_value=[TimeSegment(0, 2, "speech")]),
        patch("opencut.core.silence.get_edit_summary", return_value=summary),
        patch(
            "opencut.export.mlt_export.export_mlt",
            return_value={"output_path": str(tmp_path / "source_opencut.mlt")},
        ) as exporter,
    ):
        result = CliRunner().invoke(
            cli_module.cli,
            ["silence", str(source), "--format", "mlt"],
        )

    assert result.exit_code == 0, result.output
    assert "Kdenlive or Shotcut" in result.output
    exporter.assert_called_once()
    assert exporter.call_args.args[2] == str(tmp_path / "source_opencut.mlt")
