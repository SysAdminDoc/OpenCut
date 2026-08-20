"""Real OpenTimelineIO compatibility, downgrade, and route contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

otio = pytest.importorskip("opentimelineio")

from opencut.core.silence import TimeSegment  # noqa: E402
from opencut.export.otio_compat import (  # noqa: E402
    OTIOPreflightError,
    get_otio_capabilities,
    preflight_otio_timeline,
    write_otio_timeline,
)
from opencut.export.otio_diff import diff_timelines  # noqa: E402
from opencut.export.otio_export import export_otio  # noqa: E402


def _fixture_timeline():
    fps = 24
    timeline = otio.schema.Timeline(name="Compatibility fixture")
    timeline.global_start_time = otio.opentime.RationalTime(1001, fps)
    timeline.metadata["fixture"] = {"purpose": "roundtrip", "revision": 3}
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for index, start in enumerate((48, 240)):
        clip = otio.schema.Clip(
            name=f"Clip {index + 1}",
            media_reference=otio.schema.ExternalReference(
                target_url=f"file:///media/clip-{index + 1}.mov"
            ),
            source_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(start, fps),
                otio.opentime.RationalTime(96, fps),
            ),
            metadata={"camera": f"A{index + 1}"},
        )
        if index == 0:
            clip.markers.append(otio.schema.Marker(
                name="Review",
                marked_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(12, fps),
                    otio.opentime.RationalTime(1, fps),
                ),
                color=otio.schema.MarkerColor.YELLOW,
                metadata={"owner": "editor"},
            ))
        track.append(clip)
        if index == 0:
            track.append(otio.schema.Transition(
                name="Dissolve",
                transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
                in_offset=otio.opentime.RationalTime(12, fps),
                out_offset=otio.opentime.RationalTime(12, fps),
                metadata={"intent": "soft"},
            ))
    timeline.tracks.append(track)
    return timeline


def _assert_fixture_roundtrip(path, expected_schema):
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["metadata"]["opencut_export"]["schema_target"] == expected_schema
    recovered = otio.adapters.read_from_file(str(path))
    assert recovered.global_start_time.value == pytest.approx(1001)
    assert recovered.global_start_time.rate == pytest.approx(24)
    assert recovered.metadata["fixture"]["revision"] == 3
    items = list(recovered.tracks[0])
    assert [item.name for item in items] == ["Clip 1", "Dissolve", "Clip 2"]
    assert items[0].source_range.start_time.value == pytest.approx(48)
    assert items[0].markers[0].name == "Review"
    assert items[0].markers[0].metadata["owner"] == "editor"
    assert items[1].in_offset.value == pytest.approx(12)
    assert items[1].out_offset.value == pytest.approx(12)
    assert items[1].metadata["intent"] == "soft"


def test_capabilities_discover_versions_and_adapter_features():
    capabilities = get_otio_capabilities()
    native = next(item for item in capabilities["adapters"] if item["name"] == "otio_json")
    target_ids = {item["id"] for item in capabilities["schema_targets"]}

    assert native["read"] is True
    assert native["write"] is True
    assert native["schema_targeting"] is True
    assert native["version"] == capabilities["runtime_version"]
    assert {"current", "OTIO_CORE:0.14.0"} <= target_ids


@pytest.mark.parametrize("schema_target", ["current", "OTIO_CORE:0.14.0"])
def test_current_and_legacy_fixtures_roundtrip_without_loss(tmp_path, schema_target):
    output = tmp_path / f"fixture-{schema_target.replace(':', '-')}.otio"
    report = write_otio_timeline(
        _fixture_timeline(),
        str(output),
        schema_target=schema_target,
    )

    assert report["ready"] is True
    assert report["lossy"] is False
    assert report["written"] is True
    _assert_fixture_roundtrip(output, schema_target)


@pytest.mark.parametrize("schema_target", ["current", "OTIO_CORE:0.14.0"])
def test_export_otio_emits_transitions_that_roundtrip(tmp_path, schema_target):
    output = tmp_path / f"transition-{schema_target.replace(':', '-')}.otio"
    report = export_otio(
        "C:/media/source.mp4",
        [
            TimeSegment(0.0, 2.0, "speech"),
            TimeSegment(2.0, 4.0, "speech"),
        ],
        str(output),
        schema_target=schema_target,
        return_report=True,
        transitions=[
            {
                "after": 0,
                "name": "Soft dissolve",
                "type": "dissolve",
                "in_offset": 0.5,
                "out_offset": 0.5,
                "metadata": {"owner": "editor"},
            },
        ],
    )

    assert report["written"] is True
    recovered = otio.adapters.read_from_file(str(output))
    track = list(recovered.tracks[0])
    assert [type(item) for item in track] == [otio.schema.Clip, otio.schema.Transition, otio.schema.Clip]
    transition = track[1]
    assert transition.name == "Soft dissolve"
    assert transition.in_offset.value == pytest.approx(12)
    assert transition.out_offset.value == pytest.approx(12)
    assert transition.metadata["owner"] == "editor"
    assert transition.metadata["opencut"]["transition_type"] == "dissolve"


def test_otio_diff_reports_transition_changes(tmp_path):
    left_timeline = _fixture_timeline()
    del left_timeline.tracks[0][1]
    left = tmp_path / "left.otio"
    right = tmp_path / "right.otio"
    write_otio_timeline(left_timeline, str(left))
    write_otio_timeline(_fixture_timeline(), str(right))

    result = diff_timelines(str(left), str(right))

    assert result.summary["transitions_added"] == 1
    assert result.summary["transitions_removed"] == 0
    assert result.transition_diffs[0].kind == "added"
    assert result.to_dict()["transition_diffs"][0]["right"]["name"] == "Dissolve"


def test_preflight_reports_multireference_loss_before_write(tmp_path):
    timeline = _fixture_timeline()
    clip = timeline.tracks[0][0]
    clip.set_media_references({
        "proxy": otio.schema.ExternalReference(target_url="file:///media/proxy.mov"),
        "camera": otio.schema.ExternalReference(target_url="file:///media/camera.mov"),
    }, "camera")
    output = tmp_path / "lossy.otio"

    report = preflight_otio_timeline(
        timeline,
        schema_target="OTIO_CORE:0.14.0",
        output_path=str(output),
    )
    assert report["ready"] is True
    assert report["lossy"] is True
    assert any("media_references" in path for path in report["lossy_fields"])
    assert not output.exists()

    with pytest.raises(OTIOPreflightError) as exc_info:
        write_otio_timeline(
            timeline,
            str(output),
            schema_target="OTIO_CORE:0.14.0",
        )
    assert exc_info.value.report["lossy"] is True
    assert not output.exists()

    accepted = write_otio_timeline(
        timeline,
        str(output),
        schema_target="OTIO_CORE:0.14.0",
        accept_lossy=True,
    )
    assert accepted["written"] is True
    assert output.exists()


def test_preflight_reports_unavailable_and_incompatible_adapters():
    missing = preflight_otio_timeline(_fixture_timeline(), adapter_name="not-installed")
    assert missing["ready"] is False
    assert missing["adapter"]["installed"] is False
    assert "not installed" in missing["errors"][0]

    capabilities = get_otio_capabilities()
    non_native = next(
        (item for item in capabilities["adapters"] if item["write"] and not item["schema_targeting"]),
        None,
    )
    if non_native:
        report = preflight_otio_timeline(
            _fixture_timeline(),
            adapter_name=non_native["name"],
            schema_target="OTIO_CORE:0.14.0",
        )
        assert report["ready"] is False
        assert "cannot target" in " ".join(report["errors"])


def test_current_target_overrides_environment_downgrade(tmp_path, monkeypatch):
    output = tmp_path / "current.otio"
    monkeypatch.setenv("OTIO_DEFAULT_TARGET_VERSION_FAMILY_LABEL", "OTIO_CORE:0.14.0")

    report = write_otio_timeline(_fixture_timeline(), str(output), schema_target="current")
    raw = json.loads(output.read_text(encoding="utf-8"))
    first_clip = raw["tracks"]["children"][0]["children"][0]

    assert report["schema_target"] == "current"
    assert first_clip["OTIO_SCHEMA"] == "Clip.2"


def test_invalid_schema_target_is_a_structured_preflight_failure(tmp_path):
    output = tmp_path / "invalid.otio"
    with pytest.raises(OTIOPreflightError) as exc_info:
        write_otio_timeline(
            _fixture_timeline(),
            str(output),
            schema_target="OTIO_CORE:99.0.0",
        )
    assert exc_info.value.report["ready"] is False
    assert "Unknown OTIO schema target" in exc_info.value.report["errors"][0]
    assert not output.exists()


def _route_client():
    from opencut.server import app

    app.config["TESTING"] = True
    client = app.test_client()
    token = client.get("/health").get_json()["csrf_token"]
    return client, {"X-OpenCut-Token": token, "Content-Type": "application/json"}


def test_export_route_passes_transition_specs_to_writer(tmp_path):
    client, headers = _route_client()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    transition = {
        "after": 0,
        "name": "Dissolve",
        "type": "SMPTE_Dissolve",
        "in_offset": 0.25,
        "out_offset": 0.25,
    }
    result = {"output_path": str(tmp_path / "source_opencut.otio"), "written": True}
    with (
        patch("opencut.utils.media.probe", return_value=SimpleNamespace(fps=24.0, duration=4.0)),
        patch("opencut.export.otio_export.export_otio", return_value=result) as exporter,
    ):
        response = client.post(
            "/timeline/export-otio",
            headers=headers,
            json={
                "filepath": str(source),
                "output_dir": str(tmp_path),
                "mode": "segments",
                "segments": [{"start": 0, "end": 2}, {"start": 2, "end": 4}],
                "transitions": [transition],
            },
        )

    assert response.status_code == 200
    assert exporter.call_args.kwargs["transitions"] == [transition]


def test_capabilities_route_and_legacy_preflight_do_not_write(tmp_path):
    client, headers = _route_client()
    capabilities = client.get("/timeline/otio-capabilities")
    assert capabilities.status_code == 200
    assert any(item["name"] == "otio_json" for item in capabilities.get_json()["adapters"])

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "source_opencut.otio"
    with patch(
        "opencut.utils.media.probe",
        return_value=SimpleNamespace(fps=24.0, duration=10.0),
    ):
        response = client.post("/timeline/export-otio", headers=headers, json={
            "filepath": str(source),
            "output_dir": str(tmp_path),
            "mode": "segments",
            "segments": [{"start": 0, "end": 2}],
            "schema_target": "OTIO_CORE:0.14.0",
            "adapter_name": "otio_json",
            "preflight_only": True,
        })
    assert response.status_code == 200
    report = response.get_json()["preflight"]
    assert report["schema_target"] == "OTIO_CORE:0.14.0"
    assert report["written"] is False
    assert not output.exists()


def test_export_route_returns_versioned_preflight_and_rejects_missing_adapter(tmp_path):
    client, headers = _route_client()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    payload = {
        "filepath": str(source),
        "output_dir": str(tmp_path),
        "mode": "segments",
        "segments": [{"start": 0, "end": 2}],
        "schema_target": "current",
        "adapter_name": "otio_json",
    }
    with patch(
        "opencut.utils.media.probe",
        return_value=SimpleNamespace(fps=24.0, duration=10.0),
    ):
        response = client.post("/timeline/export-otio", headers=headers, json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["preflight"]["written"] is True
    assert body["preflight"]["adapter"]["version"] == otio.__version__
    assert body["preflight"]["schema_target"] == "current"
    assert (tmp_path / "source_opencut.otio").exists()

    payload["adapter_name"] = "not-installed"
    with patch(
        "opencut.utils.media.probe",
        return_value=SimpleNamespace(fps=24.0, duration=10.0),
    ):
        unavailable = client.post("/timeline/export-otio", headers=headers, json=payload)
    assert unavailable.status_code == 400
    unavailable_body = unavailable.get_json()
    assert unavailable_body["code"] == "OTIO_PREFLIGHT_FAILED"
    assert unavailable_body["preflight"]["adapter"]["installed"] is False


def test_silence_cli_exposes_otio_schema_preflight(tmp_path):
    from opencut import cli as cli_module
    from opencut.core.silence import TimeSegment

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
    ):
        result = CliRunner().invoke(cli_module.cli, [
            "silence",
            str(source),
            "--format", "otio",
            "--otio-schema", "OTIO_CORE:0.14.0",
            "--otio-preflight",
        ])

    assert result.exit_code == 0, result.output
    assert "schema OTIO_CORE:0.14.0" in result.output
    assert "Preflight passed" in result.output
    assert not (tmp_path / "source_opencut.otio").exists()


# ---------------------------------------------------------------------------
# F331 — bundle contract
# ---------------------------------------------------------------------------


def test_capabilities_report_names_the_verified_range():
    """"Installed" and "tested against" are different facts."""
    from opencut.export.otio_compat import (
        VERIFIED_OTIO_RANGE,
        _within_verified_range,
        get_otio_capabilities,
    )

    report = get_otio_capabilities()
    assert report["verified_range"] == list(VERIFIED_OTIO_RANGE)
    assert report["runtime_is_verified"] is _within_verified_range(
        report["runtime_version"]
    )


def test_otioz_bundle_layout_and_media_policy(tmp_path):
    """Pin the bundle contract so an untested OTIO minor fails loudly.

    The project has already been bitten by an OTIO minor renaming
    MediaReferencePolicy enum members between 0.15 and 0.17, which raised
    AttributeError at runtime in a shipped export path.
    """
    import zipfile

    from opentimelineio.file_bundle_utils import MediaReferencePolicy

    from opencut.export.otio_export import export_timeline_otioz

    for name in ("ErrorIfNotFile", "MissingIfNotFile", "AllMissing"):
        assert hasattr(MediaReferencePolicy, name), (
            f"OTIO {otio.__version__} renamed MediaReferencePolicy.{name}; "
            "the otioz export path depends on these members"
        )

    media = tmp_path / "clip.mov"
    media.write_bytes(b"not real media, but a resolvable local file")
    clips = [{"path": str(media), "start": 0.0, "end": 1.0, "name": "clip"}]
    bundle = tmp_path / "out.otioz"

    export_timeline_otioz(clips, str(bundle), bundle_media=False)

    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert "content.otio" in names, names
    # bundle_media=False must not copy the media in.
    assert not any(name.startswith("media/") for name in names), names
