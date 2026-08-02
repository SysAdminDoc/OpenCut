"""Release-gated media conformance over the synthetic corpus.

`tests/test_integration_ffmpeg.py` only ever exercises one media shape:
24 fps CFR H.264 + stereo AAC starting at PTS 0. These tests run the same
operations over the timing, channel, colour, stream, and path combinations
that actually break automated edits (see :mod:`tests.media_corpus`), and they
assert measurable properties — duration, timecode, A/V sync, stream and
channel metadata, error accounting, VMAF model identity — rather than "the
job said complete".

Run with:
    python -m pytest tests/test_media_conformance.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from opencut.helpers import get_ffmpeg_path
from tests.conftest import csrf_headers
from tests.media_corpus import (
    CORPUS,
    CORPUS_BY_NAME,
    DURATION,
    DURATION_TOLERANCE_S,
    SYNC_TOLERANCE_S,
    build_corpus,
    count_decode_errors,
    ffmpeg_available,
    format_duration,
    probe,
    streams_of,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="FFmpeg not installed"),
]


# ---------------------------------------------------------------------------
# Session corpus
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, str]:
    return build_corpus(tmp_path_factory.mktemp("media_corpus"))


def poll_job(client, job_id, csrf_token, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}", headers={"X-OpenCut-Token": csrf_token})
        data = resp.get_json()
        if data["status"] in ("complete", "error", "cancelled"):
            return data
        time.sleep(0.25)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def run_trim(client, csrf_token, filepath, start, end, **extra):
    payload = {"filepath": filepath, "start": start, "end": end}
    payload.update(extra)
    resp = client.post("/video/trim", data=json.dumps(payload),
                       headers=csrf_headers(csrf_token))
    assert resp.status_code == 200, resp.get_json()
    return poll_job(client, resp.get_json()["job_id"], csrf_token)


def video_stream(path) -> dict:
    streams = streams_of(path, "video")
    assert streams, f"{path} has no video stream"
    return streams[0]


def _rotation_of(stream: dict) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(side_data["rotation"])
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        return int(float(tags["rotate"]))
    return 0


def _timecode_of(path) -> str:
    info = probe(path)
    for stream in info.get("streams", []):
        tags = stream.get("tags") or {}
        if tags.get("timecode"):
            return tags["timecode"]
    return (info.get("format", {}).get("tags") or {}).get("timecode", "")


# ---------------------------------------------------------------------------
# The corpus must actually carry what it claims
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", CORPUS, ids=[f.name for f in CORPUS])
def test_fixture_carries_its_declared_properties(corpus, fixture):
    path = corpus[fixture.name]
    assert os.path.isfile(path), f"{fixture.name} was not built"
    info = probe(path)
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    for key, expected in fixture.expect.items():
        if key == "avg_frame_rate":
            assert video[0]["avg_frame_rate"] == expected
        elif key == "channels":
            assert audio and int(audio[0]["channels"]) == expected
        elif key == "pix_fmt":
            assert video[0]["pix_fmt"] == expected
        elif key in ("color_transfer", "color_space"):
            assert video[0].get(key) == expected
        elif key == "audio_streams":
            assert len(audio) == expected
        elif key == "subtitle_streams":
            assert len([s for s in streams if s.get("codec_type") == "subtitle"]) == expected
        elif key == "attachment_streams":
            assert len([s for s in streams if s.get("codec_type") == "attachment"]) == expected
        elif key == "rotation":
            assert _rotation_of(video[0]) == expected
        elif key == "timecode":
            assert _timecode_of(path) == expected
        elif key == "start_time_min":
            assert float(info["format"]["start_time"]) >= expected
        elif key == "decode_errors_min":
            assert count_decode_errors(path) >= expected
        elif key == "variable_frame_rate":
            assert _frame_interval_count(path) > 1
        elif key in ("width", "height"):
            assert int(video[0][key]) == expected
        else:  # pragma: no cover - a new expectation needs a comparison here
            raise AssertionError(f"unhandled expectation key {key!r}")


def _frame_interval_count(path) -> int:
    """Distinct rounded frame intervals — >1 means the stream is VFR."""
    from opencut.helpers import get_ffprobe_path
    result = subprocess.run(
        [get_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    times = [
        float(token.strip().rstrip(","))
        for token in result.stdout.split()
        if token.strip().rstrip(",")
    ]
    deltas = {round(b - a, 3) for a, b in zip(times, times[1:])}
    return len(deltas)


# ---------------------------------------------------------------------------
# Probing: fractional rates must not be rounded away
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected_fps",
    [
        ("cfr_25", 25.0),
        ("cfr_23976", 24000 / 1001),
        ("cfr_2997_dropframe", 30000 / 1001),
        ("cfr_5994", 60000 / 1001),
    ],
)
def test_probe_reports_fractional_rates_exactly(corpus, name, expected_fps):
    from opencut.helpers import get_video_info

    info = get_video_info(corpus[name])
    assert abs(float(info["fps"]) - expected_fps) < 0.01, info
    # A 23.976 source reported as 24 would drift a frame every ~42 s.
    assert abs(float(info["duration"]) - DURATION) < DURATION_TOLERANCE_S


def test_probe_survives_a_video_only_source(corpus):
    from opencut.helpers import get_video_info

    info = get_video_info(corpus["no_audio"])
    assert float(info["duration"]) > 0
    assert streams_of(corpus["no_audio"], "audio") == []


# ---------------------------------------------------------------------------
# Trim: duration, sync, and timing edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["cfr_25", "cfr_23976", "cfr_2997_dropframe", "cfr_5994", "vfr"]
)
def test_trim_duration_holds_across_frame_rates(client, csrf_token, corpus, name):
    result = run_trim(client, csrf_token, corpus[name], "0:00:00.5", "0:00:01.5")
    assert result["status"] == "complete", result.get("error")
    output = result["result"]["output_path"]
    assert abs(format_duration(output) - 1.0) <= DURATION_TOLERANCE_S


def test_trim_of_a_delayed_pts_source_uses_relative_time(client, csrf_token, corpus):
    """A 3.5 s start offset must not be added to the requested in-point."""
    source = corpus["delayed_pts"]
    assert float(probe(source)["format"]["start_time"]) >= 3.0

    result = run_trim(client, csrf_token, source, "0:00:00.5", "0:00:01.5")
    assert result["status"] == "complete", result.get("error")
    output = result["result"]["output_path"]
    duration = format_duration(output)
    assert abs(duration - 1.0) <= DURATION_TOLERANCE_S, (
        f"delayed-PTS trim produced {duration:.3f}s; the source offset leaked "
        "into the requested range"
    )


@pytest.mark.parametrize("name", ["cfr_25", "cfr_23976", "multichannel"])
def test_trim_keeps_audio_and_video_in_sync(client, csrf_token, corpus, name):
    result = run_trim(client, csrf_token, corpus[name], "0:00:00.25", "0:00:01.75")
    assert result["status"] == "complete", result.get("error")
    output = result["result"]["output_path"]

    video = streams_of(output, "video")[0]
    audio = streams_of(output, "audio")
    assert audio, "trim dropped the audio stream"
    v_start = float(video.get("start_time") or 0.0)
    a_start = float(audio[0].get("start_time") or 0.0)
    assert abs(v_start - a_start) <= SYNC_TOLERANCE_S

    v_dur = float(video.get("duration") or 0.0)
    a_dur = float(audio[0].get("duration") or 0.0)
    if v_dur and a_dur:
        assert abs(v_dur - a_dur) <= SYNC_TOLERANCE_S


def test_trim_preserves_channel_layout(client, csrf_token, corpus):
    result = run_trim(client, csrf_token, corpus["multichannel"], "0:00:00", "0:00:01")
    assert result["status"] == "complete", result.get("error")
    audio = streams_of(result["result"]["output_path"], "audio")[0]
    assert int(audio["channels"]) == 6, "5.1 was silently downmixed"


def test_trim_of_a_mono_source_stays_mono(client, csrf_token, corpus):
    result = run_trim(client, csrf_token, corpus["mono"], "0:00:00", "0:00:01")
    assert result["status"] == "complete", result.get("error")
    audio = streams_of(result["result"]["output_path"], "audio")[0]
    assert int(audio["channels"]) == 1


def test_trim_of_a_video_only_source_succeeds(client, csrf_token, corpus):
    result = run_trim(client, csrf_token, corpus["no_audio"], "0:00:00", "0:00:01")
    assert result["status"] == "complete", result.get("error")
    output = result["result"]["output_path"]
    assert streams_of(output, "video")
    assert streams_of(output, "audio") == []


def test_trim_accepts_unicode_and_spaced_paths(client, csrf_token, corpus):
    source = corpus["unicode_path"]
    assert any(ord(ch) > 127 for ch in os.path.basename(source))
    result = run_trim(client, csrf_token, source, "0:00:00", "0:00:01")
    assert result["status"] == "complete", result.get("error")
    assert os.path.isfile(result["result"]["output_path"])


# ---------------------------------------------------------------------------
# Metadata preservation through a stream copy
# ---------------------------------------------------------------------------
def _stream_copy(source: str, dest: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [get_ffmpeg_path(), "-hide_banner", "-nostdin", "-y",
         "-i", source, "-map", "0", "-c", "copy", *extra, dest],
        capture_output=True, text=True, check=False,
    )


def test_hdr_colour_metadata_survives_a_stream_copy(corpus, tmp_path):
    out = tmp_path / "hdr_copy.mp4"
    assert _stream_copy(corpus["hdr10"], str(out)).returncode == 0
    stream = video_stream(out)
    assert stream["pix_fmt"] == "yuv420p10le"
    assert stream.get("color_transfer") == "smpte2084"
    assert stream.get("color_space") == "bt2020nc"
    assert stream.get("color_primaries") == "bt2020"


def test_rotation_survives_a_stream_copy(corpus, tmp_path):
    out = tmp_path / "rotated_copy.mp4"
    assert _stream_copy(corpus["rotated"], str(out)).returncode == 0
    assert _rotation_of(video_stream(out)) == 90


def _timecode_to_frames(timecode: str, fps_num: int = 30000, fps_den: int = 1001) -> int:
    """Absolute frame index for an SMPTE timecode, drop-frame aware.

    ``;`` (or ``.``) before the frame field marks drop-frame, where two frame
    numbers are skipped at the top of every minute except every tenth.
    """
    drop = ";" in timecode or "." in timecode
    hours, minutes, seconds, frames = (
        int(part) for part in timecode.replace(";", ":").replace(".", ":").split(":")
    )
    nominal = round(fps_num / fps_den)  # 30 for 29.97, 60 for 59.94
    index = ((hours * 60 + minutes) * 60 + seconds) * nominal + frames
    if drop:
        dropped_per_minute = 2 if nominal == 30 else 4
        total_minutes = hours * 60 + minutes
        index -= dropped_per_minute * (total_minutes - total_minutes // 10)
    return index


def test_drop_frame_timecode_position_survives_a_stream_copy(corpus, tmp_path):
    """A remux may re-express the timecode, but not move it.

    FFmpeg's mov muxer writes the copied ``tmcd`` track without the
    drop-frame flag, so ``00:59:50;00`` comes back as the non-drop
    ``00:59:46:12``. That is the *same* absolute frame — the label changes,
    the position must not. Anything that shifts the position is a real
    conformance failure.
    """
    out = tmp_path / "df_copy.mov"
    assert _stream_copy(corpus["cfr_2997_dropframe"], str(out)).returncode == 0

    source_tc = _timecode_of(corpus["cfr_2997_dropframe"])
    copied_tc = _timecode_of(out)
    assert source_tc == "00:59:50;00"
    assert copied_tc, "the copy lost its timecode entirely"
    assert _timecode_to_frames(copied_tc) == _timecode_to_frames(source_tc), (
        f"timecode moved: {source_tc} -> {copied_tc}"
    )


def test_drop_frame_source_keeps_its_drop_frame_flag(corpus):
    """The source itself must be drop-frame, or the fixture proves nothing."""
    assert ";" in _timecode_of(corpus["cfr_2997_dropframe"])


def test_subtitle_stream_survives_a_stream_copy(corpus, tmp_path):
    out = tmp_path / "subs_copy.mp4"
    assert _stream_copy(corpus["with_subtitles"], str(out)).returncode == 0
    assert len(streams_of(out, "subtitle")) == 1


def test_attachment_stream_survives_a_stream_copy(corpus, tmp_path):
    out = tmp_path / "attach_copy.mkv"
    assert _stream_copy(corpus["with_attachment"], str(out)).returncode == 0
    assert len(streams_of(out, "attachment")) == 1


def test_drop_frame_source_carries_a_timecode_data_stream(corpus):
    """The QuickTime timecode track is a data stream, not just a tag."""
    data_streams = streams_of(corpus["cfr_2997_dropframe"], "data")
    assert data_streams, "expected a timecode data stream"
    assert (data_streams[0].get("tags") or {}).get("timecode") == "00:59:50;00"


# ---------------------------------------------------------------------------
# Error accounting
# ---------------------------------------------------------------------------
def test_corrupt_source_decodes_with_errors_but_does_not_stall(corpus):
    errors = count_decode_errors(corpus["corrupt"])
    assert errors >= 1, "the corrupt fixture stopped being corrupt"
    assert count_decode_errors(corpus["cfr_25"]) == 0


def test_corrupt_source_reports_a_terminal_status_not_a_hang(client, csrf_token, corpus):
    result = run_trim(client, csrf_token, corpus["corrupt"], "0:00:00", "0:00:01")
    assert result["status"] in ("complete", "error")
    if result["status"] == "error":
        # A failure must be classified, not a bare traceback string.
        assert result.get("error"), result
    else:
        assert os.path.isfile(result["result"]["output_path"])


def test_missing_input_is_rejected_before_a_job_is_created(client, csrf_token, tmp_path):
    resp = client.post(
        "/video/trim",
        data=json.dumps({
            "filepath": str(tmp_path / "does-not-exist.mp4"),
            "start": "0:00:00",
            "end": "0:00:01",
        }),
        headers=csrf_headers(csrf_token),
    )
    body = resp.get_json()
    assert resp.status_code >= 400, body
    assert body.get("code") in ("FILE_NOT_FOUND", "INVALID_INPUT"), body


# ---------------------------------------------------------------------------
# Proxy / original parity and VMAF receipts
# ---------------------------------------------------------------------------
def test_proxy_shares_the_originals_timing_grid(corpus):
    original = video_stream(corpus["cfr_25"])
    proxy = video_stream(corpus["proxy"])
    assert proxy["avg_frame_rate"] == original["avg_frame_rate"]
    assert abs(format_duration(corpus["proxy"]) - format_duration(corpus["cfr_25"])) <= 0.1
    # A proxy is smaller on purpose; that is the only difference allowed here.
    assert int(proxy["width"]) < int(original["width"])


def test_quality_receipt_identifies_the_vmaf_model(corpus):
    from opencut.core.quality_metrics import (
        VMAF_MODEL,
        check_vmaf_available,
        compare_videos,
    )

    report = compare_videos(corpus["proxy"], corpus["cfr_25"])
    # SSIM/PSNR need matching dimensions; the proxy comparison proves the
    # distorted input is scaled to the reference rather than erroring out.
    assert report.ssim is not None, report.notes
    assert report.psnr is not None, report.notes

    if not check_vmaf_available():
        assert report.vmaf is None
        assert any("libvmaf" in note for note in report.notes), report.notes
        return

    assert report.vmaf is not None, report.notes
    assert report.vmaf_model == VMAF_MODEL
    assert report.vmaf_version, "receipt must name the libvmaf build"
    assert report.vmaf_scaling
    assert report.vmaf_mode == "hd"


def test_identical_inputs_score_as_a_perfect_match(corpus):
    from opencut.core.quality_metrics import check_vmaf_available, compare_videos

    report = compare_videos(corpus["cfr_25"], corpus["cfr_25"])
    assert report.ssim == pytest.approx(1.0, abs=1e-6), report.notes
    # A lossless comparison reports PSNR as infinity; it must not read as a
    # parse failure.
    assert report.psnr == float("inf"), report.notes
    if check_vmaf_available():
        assert report.vmaf is not None and report.vmaf > 95.0, report.notes


# ---------------------------------------------------------------------------
# Corpus bookkeeping
# ---------------------------------------------------------------------------
def test_every_corpus_entry_declares_an_expectation_and_a_description():
    for fixture in CORPUS:
        assert fixture.expect, f"{fixture.name} asserts nothing"
        assert fixture.description, f"{fixture.name} has no description"
    assert len(CORPUS_BY_NAME) == len(CORPUS), "duplicate corpus names"
