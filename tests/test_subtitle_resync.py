import json
from pathlib import Path

import pytest
from click.testing import CliRunner


def _write_srt(path: Path, cues):
    lines = []
    for index, (start, end, text) in enumerate(cues, 1):
        def tc(value):
            millis = round(value * 1000)
            hours, remainder = divmod(millis, 3_600_000)
            minutes, remainder = divmod(remainder, 60_000)
            seconds, millis = divmod(remainder, 1_000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        lines.extend([str(index), f"{tc(start)} --> {tc(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _reference_cues():
    return [
        {"start": 0.50, "end": 1.80, "text": "Welcome to OpenCut."},
        {"start": 3.20, "end": 4.60, "text": "This is the timing fixture."},
        {"start": 7.00, "end": 8.40, "text": "The final cue stays aligned."},
    ]


def test_constant_offset_preview_and_apply(tmp_path):
    from opencut.core.subtitle_resync import (
        parse_srt_file,
        resync_subtitles,
        write_resynced_srt,
    )

    reference = _reference_cues()
    offset = 1.25
    source = tmp_path / "offset.srt"
    _write_srt(
        source,
        [(cue["start"] + offset, cue["end"] + offset, cue["text"]) for cue in reference],
    )
    output = tmp_path / "offset_resynced.srt"

    preview = resync_subtitles(source, reference_segments=reference, fps=30)

    assert preview["fit_mode"] == "affine"
    assert preview["offset_seconds"] == pytest.approx(-offset, abs=0.001)
    assert preview["rate"] == pytest.approx(1.0, abs=0.000001)
    assert preview["within_one_frame"] is True
    assert not output.exists(), "preview must not write the output file"

    write_resynced_srt(preview, output)
    corrected = parse_srt_file(output)
    assert [cue.start for cue in corrected] == pytest.approx(
        [cue["start"] for cue in reference], abs=0.001
    )
    assert [cue.end for cue in corrected] == pytest.approx(
        [cue["end"] for cue in reference], abs=0.001
    )


def test_affine_drift_realigns_to_reference_within_one_frame(tmp_path):
    from opencut.core.subtitle_resync import parse_srt_file, resync_subtitles

    reference = _reference_cues()
    source = tmp_path / "drift.srt"
    source_rate = 1.08
    source_offset = 0.75
    _write_srt(
        source,
        [
            (
                source_rate * cue["start"] + source_offset,
                source_rate * cue["end"] + source_offset,
                cue["text"],
            )
            for cue in reference
        ],
    )

    preview = resync_subtitles(source, reference_segments=reference, fps=24)
    corrected = parse_srt_file(source)

    assert preview["fit_mode"] == "affine"
    assert preview["rate"] == pytest.approx(1.0 / source_rate, abs=0.000001)
    assert preview["offset_seconds"] == pytest.approx(
        -source_offset / source_rate, abs=0.000001
    )
    assert preview["within_one_frame"] is True
    for output_cue, reference_cue in zip(preview["cues"], reference):
        assert output_cue["start"] == pytest.approx(reference_cue["start"], abs=1 / 24)
        assert output_cue["end"] == pytest.approx(reference_cue["end"], abs=1 / 24)
    assert [cue.start for cue in corrected] != pytest.approx(
        [cue["start"] for cue in reference], abs=0.01
    )


def test_video_reference_uses_existing_transcription_contract(tmp_path):
    from opencut.core.subtitle_resync import resync_subtitles

    reference = _reference_cues()
    source = tmp_path / "captions.srt"
    video = tmp_path / "video.mp4"
    _write_srt(source, [(cue["start"] + 0.5, cue["end"] + 0.5, cue["text"]) for cue in reference])
    video.write_bytes(b"fixture")

    seen = []

    def fake_transcriber(path):
        seen.append(path)
        return {"segments": reference}

    preview = resync_subtitles(source, video_path=video, transcriber=fake_transcriber)

    assert seen == [str(video)]
    assert preview["reference_source"] == "transcription"
    assert preview["offset_seconds"] == pytest.approx(-0.5, abs=0.001)


def test_resync_route_requires_preview_confirmation_before_apply(client, csrf_token, tmp_path):
    reference = _reference_cues()
    source = tmp_path / "route.srt"
    output = tmp_path / "route_resynced.srt"
    _write_srt(source, [(cue["start"] + 1.0, cue["end"] + 1.0, cue["text"]) for cue in reference])
    headers = {"X-OpenCut-Token": csrf_token, "Content-Type": "application/json"}
    payload = {
        "srt_path": str(source),
        "output_path": str(output),
        "reference_segments": reference,
        "fps": 30,
    }

    preview_response = client.post("/subtitle/resync", json=payload, headers=headers)
    assert preview_response.status_code == 200
    preview = preview_response.get_json()
    assert preview["preview"] is True
    assert preview["applied"] is False
    assert preview["result"]["within_one_frame"] is True
    assert not output.exists()
    original = source.read_text(encoding="utf-8")

    apply_payload = {
        **payload,
        "apply": True,
        "confirm_token": preview["plan"]["confirm_token"],
    }
    apply_response = client.post("/subtitle/resync", json=apply_payload, headers=headers)
    assert apply_response.status_code == 200
    assert apply_response.get_json()["applied"] is True
    assert output.exists()
    assert source.read_text(encoding="utf-8") == original


def test_cli_resync_is_read_only_until_apply(tmp_path):
    from opencut.cli import cli

    reference = _reference_cues()
    source = tmp_path / "cli.srt"
    reference_json = tmp_path / "reference.json"
    output = tmp_path / "cli_resynced.srt"
    _write_srt(source, [(cue["start"] + 0.75, cue["end"] + 0.75, cue["text"]) for cue in reference])
    reference_json.write_text(json.dumps({"segments": reference}), encoding="utf-8")
    runner = CliRunner()

    preview_result = runner.invoke(
        cli,
        [
            "subtitle-resync",
            str(source),
            "--reference-json",
            str(reference_json),
        ],
    )
    assert preview_result.exit_code == 0, preview_result.output
    assert "Preview only" in preview_result.output
    assert not output.exists()

    apply_result = runner.invoke(
        cli,
        [
            "subtitle-resync",
            str(source),
            "--reference-json",
            str(reference_json),
            "--output",
            str(output),
            "--apply",
        ],
    )
    assert apply_result.exit_code == 0, apply_result.output
    assert output.exists()
