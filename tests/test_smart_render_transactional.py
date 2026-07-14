"""Transactional, stream-integrity, and fallback contracts for smart render."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _profile(*streams, duration=10.0):
    from opencut.core.smart_render import MediaProfile

    return MediaProfile(
        duration=duration,
        start_time=0.0,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        streams=list(streams),
    )


def _stream(index, kind, codec, **kwargs):
    from opencut.core.smart_render import MediaStreamProfile

    return MediaStreamProfile(
        index=index,
        codec_type=kind,
        codec_name=codec,
        duration=kwargs.pop("duration", 10.0),
        **kwargs,
    )


def _plan():
    from opencut.core.smart_render import ChangedSegment, SmartRenderPlan

    return SmartRenderPlan(
        total_duration=10.0,
        changed_segments=[ChangedSegment(2.0, 4.0)],
        copy_segments=[(0.0, 2.0), (4.0, 10.0)],
        encode_duration=2.0,
        copy_duration=8.0,
        estimated_speedup=5.0,
        keyframes=[0.0, 2.0, 4.0, 8.0],
    )


def test_preflight_allows_codec_compatible_cfr_av():
    from opencut.core.smart_render import _build_preflight

    source = _profile(
        _stream(0, "video", "h264", avg_frame_rate="24/1", r_frame_rate="24/1"),
        _stream(1, "audio", "aac"),
    )
    with patch("opencut.core.smart_render._probe_media", return_value=source):
        preflight, returned = _build_preflight("source.mp4", _plan(), "libx264")

    assert returned is source
    assert preflight.eligible is True
    assert preflight.requested_codec == "h264"
    assert preflight.keyframe_count == 4
    assert [item["type"] for item in preflight.stream_map] == ["video", "audio"]


def test_preflight_routes_vfr_subtitles_and_codec_mismatch_to_fallback():
    from opencut.core.smart_render import _build_preflight

    source = _profile(
        _stream(0, "video", "hevc", avg_frame_rate="24000/1001", r_frame_rate="30/1"),
        _stream(1, "audio", "pcm_s16le"),
        _stream(2, "subtitle", "mov_text", duration=4.0),
    )
    with patch("opencut.core.smart_render._probe_media", return_value=source):
        preflight, _ = _build_preflight("vfr-subtitle.mov", _plan(), "libx264")

    reasons = " ".join(preflight.fallback_reasons)
    assert preflight.eligible is False
    assert "copied video is hevc" in reasons
    assert "not uniformly AAC" in reasons
    assert "subtitle streams" in reasons
    assert "variable-frame-rate" in reasons


def test_timestamp_validation_rejects_non_monotonic_packets():
    from opencut.core.smart_render import (
        SmartRenderValidationError,
        _validate_packet_timestamps,
    )

    profile = _profile(_stream(0, "video", "h264"), duration=3.0)
    with patch(
        "opencut.core.smart_render._sample_packet_timestamps",
        return_value={0: [0.0, 0.5, 0.25]},
    ):
        with pytest.raises(SmartRenderValidationError, match="not monotonic"):
            _validate_packet_timestamps("bad.mp4", profile, 0.25)


def test_validation_checks_streams_codec_duration_packets_and_decode():
    from opencut.core.smart_render import _validate_render

    source = _profile(_stream(0, "video", "h264"), _stream(1, "audio", "aac"))
    output = _profile(_stream(0, "video", "h264"), _stream(1, "audio", "aac"))
    with (
        patch("opencut.core.smart_render._probe_media", return_value=output),
        patch("opencut.core.smart_render._validate_packet_timestamps", return_value=321),
        patch("opencut.core.smart_render._decode_smoke") as decode,
    ):
        result = _validate_render("staged.mp4", source, "h264")

    assert result["stream_counts"] == {"video": 1, "audio": 1, "subtitle": 0}
    assert result["sampled_packets"] == 321
    assert result["decode_smoke"] == "passed"
    decode.assert_called_once_with("staged.mp4", 10.0)


def test_partial_validation_failure_falls_back_before_atomic_replace(tmp_path):
    from opencut.core.smart_render import (
        SmartRenderPreflight,
        SmartRenderValidationError,
        smart_render,
    )

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    final_path = tmp_path / "final.mp4"
    final_path.write_bytes(b"old-final")
    source = _profile(_stream(0, "video", "h264"))
    preflight = SmartRenderPreflight(
        True, 10.0, 0.0, "mp4", [], 4, "libx264", "h264"
    )

    def partial(_source, staged, *_args):
        Path(staged).write_bytes(b"invalid-partial")
        return 3

    def fallback(_source, staged, *_args):
        Path(staged).write_bytes(b"validated-fallback")

    with (
        patch("opencut.core.smart_render.detect_changed_segments", return_value=_plan()),
        patch(
            "opencut.core.smart_render._build_preflight",
            return_value=(preflight, source),
        ),
        patch("opencut.core.smart_render._render_partial", side_effect=partial),
        patch("opencut.core.smart_render._render_full_transcode", side_effect=fallback),
        patch(
            "opencut.core.smart_render._validate_render",
            side_effect=[SmartRenderValidationError("bad timestamps"), {"decode_smoke": "passed"}],
        ),
    ):
        result = smart_render(str(source_path), [{}], str(final_path))

    assert final_path.read_bytes() == b"validated-fallback"
    assert source_path.read_bytes() == b"source"
    assert result["fallback_used"] is True
    assert "bad timestamps" in " ".join(result["fallback_reasons"])
    assert list(tmp_path.glob(".final.opencut-*")) == []


def test_failed_fallback_preserves_existing_final_and_cleans_staging(tmp_path):
    from opencut.core.smart_render import SmartRenderPreflight, smart_render

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    final_path = tmp_path / "final.mp4"
    final_path.write_bytes(b"old-final")
    source = _profile(_stream(0, "video", "hevc"))
    preflight = SmartRenderPreflight(
        False,
        10.0,
        0.0,
        "mp4",
        [],
        4,
        "libx264",
        "h264",
        ["codec mismatch"],
    )
    with (
        patch("opencut.core.smart_render.detect_changed_segments", return_value=_plan()),
        patch(
            "opencut.core.smart_render._build_preflight",
            return_value=(preflight, source),
        ),
        patch(
            "opencut.core.smart_render._render_full_transcode",
            side_effect=RuntimeError("encoder failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="encoder failed"):
            smart_render(str(source_path), [{}], str(final_path))

    assert final_path.read_bytes() == b"old-final"
    assert source_path.read_bytes() == b"source"
    assert list(tmp_path.glob(".final.opencut-*")) == []


def test_output_cannot_replace_source(tmp_path):
    from opencut.core.smart_render import SmartRenderPreflight, smart_render

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    source = _profile(_stream(0, "video", "h264"))
    preflight = SmartRenderPreflight(
        True, 10.0, 0.0, "mp4", [], 4, "libx264", "h264"
    )
    with (
        patch("opencut.core.smart_render.detect_changed_segments", return_value=_plan()),
        patch(
            "opencut.core.smart_render._build_preflight",
            return_value=(preflight, source),
        ),
    ):
        with pytest.raises(ValueError, match="cannot replace the source"):
            smart_render(str(source_path), [{}], str(source_path))

    assert source_path.read_bytes() == b"source"


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_av_fixture(path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x90:rate=24:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=3",
        "-c:v",
        "libx264",
        "-g",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/ffprobe unavailable")
def test_real_render_promotes_only_media_validated_output(tmp_path):
    from opencut.core.smart_render import smart_render

    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _make_av_fixture(source)

    result = smart_render(
        str(source),
        [{"start": 0.8, "end": 1.6, "type": "effect"}],
        str(output),
        preset="ultrafast",
    )

    assert output.is_file() and output.stat().st_size > 0
    assert source.is_file() and source.stat().st_size > 0
    assert result["validation"]["decode_smoke"] == "passed"
    assert result["validation"]["stream_counts"] == {
        "video": 1,
        "audio": 1,
        "subtitle": 0,
    }
    assert result["preflight"]["requested_codec"] == "h264"


@pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg/ffprobe unavailable")
def test_real_subtitle_fixture_uses_fallback_and_preserves_track(tmp_path):
    from opencut.core.smart_render import smart_render

    base = tmp_path / "base.mp4"
    source = tmp_path / "subtitle.mp4"
    output = tmp_path / "subtitle-output.mp4"
    subtitles = tmp_path / "captions.srt"
    _make_av_fixture(base)
    subtitles.write_text(
        "1\n00:00:00,250 --> 00:00:01,750\nSmart render subtitle\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(base),
            "-i",
            str(subtitles),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            "mov_text",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    result = smart_render(
        str(source),
        [{"start": 0.8, "end": 1.6, "type": "effect"}],
        str(output),
        preset="ultrafast",
    )

    assert result["fallback_used"] is True
    assert "subtitle streams" in " ".join(result["fallback_reasons"])
    assert result["validation"]["stream_counts"]["subtitle"] == 1
