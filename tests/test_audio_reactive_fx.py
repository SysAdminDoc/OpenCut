"""Coverage for the deterministic audio-reactive renderer and route contract."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from opencut.core import audio_reactive_fx as fx
from opencut.helpers import get_ffmpeg_path, get_ffprobe_path
from tests.conftest import csrf_headers


def test_presets_and_parameter_validation_are_allowlisted():
    presets = fx.list_presets()
    assert [preset["name"] for preset in presets] == list(fx.PRESETS)
    assert fx.validate_request_payload({}) is None
    assert fx.validate_request_payload({"preset": "missing"})
    assert fx.validate_request_payload(
        {"preset": "boom", "custom_params": {"unknown": 1}}
    )
    assert fx.validate_request_payload(
        {"preset": "boom", "custom_params": {"zoom_pulse": 2.1}}
    )
    assert fx.validate_request_payload(
        {"preset": "boom", "custom_params": {"strobe_on_beat": 1}}
    )
    assert fx.validate_request_payload(
        {"preset": "boom", "custom_params": {"zoom_pulse": "nan"}}
    )


@pytest.mark.parametrize("preset", sorted(fx.PRESETS))
def test_each_documented_preset_builds_a_filter(preset):
    keyframes = [
        {"time": 0.0, "drive": 0.8, "beat": True},
        {"time": 0.05, "drive": 0.2, "beat": False},
    ]
    video_filter = fx.build_video_filter(keyframes, fx.PRESETS[preset], 160, 120)
    assert video_filter != "null"
    assert "eq=" in video_filter


def test_filter_is_bounded_and_zero_parameters_are_a_noop():
    assert fx.build_video_filter([], fx.PRESETS["boom"], 160, 120) == "null"
    assert fx.build_video_filter(
        [{"time": 0.0, "drive": 1.0, "beat": False}],
        {name: 0 for name in fx.MAX_PARAMETER_VALUES} | {"strobe_on_beat": False},
        160,
        120,
    ) == "null"

    keyframes = [
        {"time": index / 20.0, "drive": (index % 10) / 10.0, "beat": index % 5 == 0}
        for index in range(fx.MAX_KEYFRAMES + 100)
    ]
    video_filter = fx.build_video_filter(keyframes, fx.PRESETS["boom"], 160, 120)
    # The same bounded drive expression is reused by several FFmpeg options;
    # count the serialized occurrences rather than assuming one occurrence per
    # sampled point.
    assert video_filter.count("between(") <= fx.MAX_FILTER_POINTS * 8


def test_render_returns_metadata_and_replaces_output_atomically(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    audio = tmp_path / "track.wav"
    output = tmp_path / "rendered.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    output.write_bytes(b"old output")

    analysis_calls = []
    commands = []

    def fake_analyze(path, **kwargs):
        analysis_calls.append((path, kwargs))
        return {
            "amplitude": [0.1, 0.4, 0.2],
            "rms": [0.05, 0.2, 0.1],
            "beats": [0.0, 1.0, 0.0],
        }

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[-1]).write_bytes(b"encoded output")

    from opencut.core import rhythm_effects

    monkeypatch.setattr(fx, "_audio_stream_available", lambda _path: True)
    monkeypatch.setattr(fx, "get_video_info", lambda _path: {
        "width": 160,
        "height": 120,
        "duration": 0.15,
    })
    monkeypatch.setattr(rhythm_effects, "analyze_audio_features", fake_analyze)
    monkeypatch.setattr(fx, "run_ffmpeg", fake_run)

    progress = []
    result = fx.render(
        str(video),
        str(audio),
        preset="snare",
        custom_params={"zoom_pulse": 0.2},
        output=str(output),
        on_progress=lambda percent, message: progress.append((percent, message)),
    )

    assert output.read_bytes() == b"encoded output"
    assert result.output == str(output)
    assert result.preset == "snare"
    assert result.beat_count == 1
    assert len(result.keyframes) == 3
    assert result.analysis["engine"] == "pcm_onset"
    assert result.analysis["frame_count"] == 3
    assert result.capabilities["network_required"] is False
    assert progress[-1][0] == 100
    assert analysis_calls[0][0] == str(audio)
    assert commands[0][1]["job_id"] == ""
    assert commands[0][0].count("-i") == 2
    assert "1:a:0?" in commands[0][0]
    assert not list(tmp_path.glob(".*.part.mp4"))


def test_render_cleans_partial_output_and_preserves_existing_file(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    audio = tmp_path / "track.wav"
    output = tmp_path / "rendered.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    output.write_bytes(b"keep this")

    from opencut.core import rhythm_effects

    monkeypatch.setattr(fx, "_audio_stream_available", lambda _path: True)
    monkeypatch.setattr(fx, "get_video_info", lambda _path: {"width": 160, "height": 120})
    monkeypatch.setattr(
        rhythm_effects,
        "analyze_audio_features",
        lambda *_args, **_kwargs: {"amplitude": [], "rms": [], "beats": []},
    )

    def fail_after_writing(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(fx, "run_ffmpeg", fail_after_writing)
    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        fx.render(str(video), str(audio), output=str(output))

    assert output.read_bytes() == b"keep this"
    assert not list(tmp_path.glob(".*.part.mp4"))


def test_render_cancellation_is_observed_before_analysis(monkeypatch, tmp_path):
    video = tmp_path / "input.mp4"
    audio = tmp_path / "track.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    analyze_called = False

    from opencut.core import rhythm_effects

    def fail_if_called(*_args, **_kwargs):
        nonlocal analyze_called
        analyze_called = True
        raise AssertionError("analysis should not start after cancellation")

    monkeypatch.setattr(rhythm_effects, "analyze_audio_features", fail_if_called)
    with pytest.raises(fx.AudioReactiveCancelled):
        fx.render(
            str(video),
            str(audio),
            is_cancelled=lambda: True,
        )
    assert analyze_called is False


def test_render_without_audio_stream_preserves_video(monkeypatch, tmp_path):
    video = tmp_path / "silent.mp4"
    output = tmp_path / "rendered.mp4"
    video.write_bytes(b"video")
    commands = []

    monkeypatch.setattr(fx, "_audio_stream_available", lambda _path: False)
    monkeypatch.setattr(fx, "get_video_info", lambda _path: {"width": 160, "height": 120})

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"video only")

    monkeypatch.setattr(fx, "run_ffmpeg", fake_run)
    result = fx.render(str(video), str(video), output=str(output))

    assert output.read_bytes() == b"video only"
    assert result.keyframes == []
    assert result.analysis["audio_stream"] is False
    assert "no audio stream" in " ".join(result.notes).lower()
    assert commands[0].count("-i") == 1


def _ffmpeg_available() -> bool:
    try:
        return Path(get_ffmpeg_path()).is_file() and Path(get_ffprobe_path()).is_file()
    except (OSError, RuntimeError):
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/FFprobe unavailable")
def test_render_produces_a_real_media_artifact(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "reactive.mp4"
    built = subprocess.run(
        [
            get_ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=0.8:size=160x120:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.8:sample_rate=48000",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr[-2000:]

    result = fx.render(str(source), str(source), preset="bass_drop", output=str(output))
    assert output.is_file() and output.stat().st_size > 0
    assert result.backend == "pcm_onset"
    assert result.analysis["frame_count"] > 0

    probe = subprocess.run(
        [
            get_ffprobe_path(),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    details = json.loads(probe.stdout)
    assert float(details["format"]["duration"]) > 0
    assert {stream["codec_type"] for stream in details["streams"]} >= {"video", "audio"}


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/FFprobe unavailable")
def test_render_accepts_variable_frame_rate_video_with_external_audio(tmp_path):
    from tests.media_corpus import build_corpus

    media = build_corpus(tmp_path, ["vfr", "cfr_25"])
    output = tmp_path / "vfr-reactive.mp4"
    result = fx.render(
        media["vfr"],
        media["cfr_25"],
        preset="chill",
        output=str(output),
    )
    assert output.is_file() and output.stat().st_size > 0
    assert result.analysis["frame_count"] > 0

    probe = subprocess.run(
        [
            get_ffprobe_path(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert {stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]} >= {
        "video",
        "audio",
    }


def test_route_is_a_durable_job_and_uses_the_available_renderer():
    from opencut.routes.wave_k_routes import route_audio_reactive_fx

    assert getattr(route_audio_reactive_fx, "_opencut_async_job", False) is True
    assert route_audio_reactive_fx._opencut_job_type == "audio_reactive_fx"
    assert fx.check_audio_reactive_available() is True


def test_route_rejects_invalid_preset_before_creating_a_job(client, csrf_token, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    response = client.post(
        "/video/audio-reactive",
        json={"video_path": str(video), "preset": "not-a-preset"},
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 400
    assert response.get_json()["code"] == "INVALID_INPUT"


def test_route_creates_and_completes_a_durable_job(monkeypatch, client, csrf_token, tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    def fake_render(*_args, **_kwargs):
        return fx.AudioReactiveResult(
            output=str(tmp_path / "rendered.mp4"),
            preset="boom",
            beat_count=2,
            keyframes=[{"time": 0.0, "drive": 1.0}],
            analysis={"engine": "pcm_onset", "frame_count": 1},
            capabilities={"deterministic_renderer": True},
            notes=[],
        )

    monkeypatch.setattr(fx, "render", fake_render)
    response = client.post(
        "/video/audio-reactive",
        json={"video_path": str(video), "preset": "boom"},
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    from opencut.jobs import _get_job_copy

    deadline = time.monotonic() + 5
    job = _get_job_copy(job_id)
    while job and job.get("status") not in {"complete", "error", "cancelled"}:
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
        job = _get_job_copy(job_id)

    assert job is not None
    assert job["status"] == "complete"
    assert job["result"]["analysis"]["engine"] == "pcm_onset"
    assert job["result"]["capabilities"]["deterministic_renderer"] is True
