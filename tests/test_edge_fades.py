"""F335 — cut-boundary fades.

Removing silence rebuilds the timeline by trimming segments and concatenating
them, which splices two unrelated waveforms sample-to-sample. Unless both sides
sit at a zero crossing the step is audible as a click on every edit. These tests
pin the clamp, the geometry of the emitted filter, and that the render paths
actually apply it at splices and only at splices.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import wave  # noqa: E402

import pytest  # noqa: E402

from opencut.helpers import (  # noqa: E402
    EDGE_FADE_DEFAULT_MS,
    EDGE_FADE_MAX_MS,
    build_edge_fade_filter,
    edge_fade_ms,
    get_ffmpeg_path,
)


class TestEdgeFadeClamp:
    def test_default_when_unspecified(self):
        assert edge_fade_ms(None) == EDGE_FADE_DEFAULT_MS

    def test_zero_disables(self):
        assert edge_fade_ms(0) == 0.0

    def test_negative_disables_rather_than_inverting(self):
        assert edge_fade_ms(-5) == 0.0

    def test_absurd_request_is_capped(self):
        assert edge_fade_ms(10_000) == EDGE_FADE_MAX_MS

    def test_junk_disables_instead_of_raising(self):
        # These arrive from request JSON, so they must never reach FFmpeg.
        for junk in ("abc", None if False else "", [], {}, float("nan"), float("inf")):
            assert edge_fade_ms(junk) == 0.0, junk

    def test_numeric_string_is_accepted(self):
        assert edge_fade_ms("12") == 12.0


class TestEdgeFadeFilter:
    def test_both_edges_by_default(self):
        out = build_edge_fade_filter(10.0, 5.0)
        assert "afade=t=in:st=0:d=0.005000" in out
        assert "afade=t=out:st=9.995000:d=0.005000" in out

    def test_fade_out_lands_inside_the_segment(self):
        """A fade-out starting past the end would never be applied."""
        duration = 2.0
        out = build_edge_fade_filter(duration, 10.0)
        start = float(out.split("afade=t=out:st=")[1].split(":")[0])
        assert 0 < start < duration
        assert abs((duration - start) - 0.010) < 1e-6

    def test_single_edge_requests(self):
        assert build_edge_fade_filter(10.0, 5.0, fade_out=False) == (
            "afade=t=in:st=0:d=0.005000"
        )
        assert build_edge_fade_filter(10.0, 5.0, fade_in=False).startswith("afade=t=out")

    def test_disabled_fade_is_empty(self):
        assert build_edge_fade_filter(10.0, 0) == ""

    def test_no_edges_requested_is_empty(self):
        assert build_edge_fade_filter(10.0, 5.0, fade_in=False, fade_out=False) == ""

    def test_short_segment_is_left_alone(self):
        """Fading both ends of a very short segment would shape its content."""
        assert build_edge_fade_filter(0.01, 5.0) == ""

    def test_nonsense_duration_is_empty_not_a_crash(self):
        for bad in (0, -1, "abc", None, float("nan")):
            assert build_edge_fade_filter(bad, 5.0) == "", bad


class TestRenderPathsDeClickTheirSplices:
    def _filter_for(self, monkeypatch, module, call):
        """Capture the filter_complex a render path hands to FFmpeg."""
        seen = {}

        def _fake_run(cmd, *a, **kw):
            for i, part in enumerate(cmd):
                if part in ("-filter_complex", "-filter_complex_script"):
                    seen["fc"] = cmd[i + 1]
            return ""

        monkeypatch.setattr(module, "run_ffmpeg", _fake_run)
        call()
        return seen.get("fc", "")

    def test_transcript_cut_render_fades_interior_joins_only(self, monkeypatch):
        from opencut.core import transcript_timeline_edit as tte

        cuts = [
            tte.CutEntry(source_start=0.0, source_end=5.0),
            tte.CutEntry(source_start=20.0, source_end=25.0),
            tte.CutEntry(source_start=40.0, source_end=45.0),
        ]
        fc = self._filter_for(
            monkeypatch, tte, lambda: tte._concat_segments("in.mp4", cuts, "out.mp4")
        )

        segments = [chunk for chunk in fc.split(";") if chunk.startswith("[0:a]")]
        assert len(segments) == 3
        # First segment opens the file: no fade-in. Last closes it: no fade-out.
        assert "afade=t=in" not in segments[0]
        assert "afade=t=out" in segments[0]
        assert "afade=t=in" in segments[1] and "afade=t=out" in segments[1]
        assert "afade=t=in" in segments[2]
        assert "afade=t=out" not in segments[2]

    def test_transcript_cut_render_honours_opt_out(self, monkeypatch):
        from opencut.core import transcript_timeline_edit as tte

        cuts = [
            tte.CutEntry(source_start=0.0, source_end=5.0),
            tte.CutEntry(source_start=20.0, source_end=25.0),
        ]
        fc = self._filter_for(
            monkeypatch,
            tte,
            lambda: tte._concat_segments("in.mp4", cuts, "out.mp4", fade_ms=0),
        )
        assert "afade" not in fc

    def test_single_segment_render_adds_no_fade(self, monkeypatch):
        """One segment has no splices, so nothing should be shaped."""
        from opencut.core import transcript_timeline_edit as tte

        cuts = [tte.CutEntry(source_start=0.0, source_end=5.0)]
        fc = self._filter_for(
            monkeypatch, tte, lambda: tte._concat_segments("in.mp4", cuts, "out.mp4")
        )
        assert "afade" not in fc


def _ffmpeg_or_skip() -> str:
    path = get_ffmpeg_path() or shutil.which("ffmpeg")
    if not path or not os.path.exists(path):
        pytest.skip("ffmpeg is not available")
    return path


def _ffmpeg(binary: str, *args: str) -> None:
    result = subprocess.run(
        [binary, "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"ffmpeg failed: {result.stderr[:800]}")


def _read_mono_pcm(path) -> list[int]:
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return list(struct.unpack(f"<{len(raw) // 2}h", raw))


def _largest_step(samples: list[int], centre: int, radius: int) -> int:
    """Largest absolute sample-to-sample jump in a window."""
    low = max(1, centre - radius)
    high = min(len(samples), centre + radius)
    return max(abs(samples[i] - samples[i - 1]) for i in range(low, high))


class TestTheFadeActuallyRemovesTheStep:
    """F365 — measure the splice, do not just assert the filter string.

    The tests above prove the render path emits this filter at interior joins
    and nowhere else. They cannot show that the filter does anything audible.
    This renders a real cut with FFmpeg and measures the sample-level
    discontinuity at the join, which is the click, using the same segment
    geometry `_concat_segments` builds.

    The source is a chirp so the instantaneous phase at the two splice points
    differs, which is what makes the join step rather than happening to line
    up at a zero crossing. The yardstick is the waveform's own steepest slope
    away from the join: a discontinuity worth hearing is a jump the signal
    would never make on its own.
    """

    SAMPLE_RATE = 48000
    SEGMENTS = ((0.0, 1.0), (2.0, 3.0))

    def _render(self, binary, source, target, fade_ms):
        chains, labels = [], []
        for index, (start, end) in enumerate(self.SEGMENTS):
            chain = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
            fade = build_edge_fade_filter(
                end - start,
                fade_ms,
                fade_in=index > 0,
                fade_out=index < len(self.SEGMENTS) - 1,
            )
            if fade:
                chain += "," + fade
            chains.append(f"[0:a]{chain}[a{index}]")
            labels.append(f"[a{index}]")
        graph = (
            ";".join(chains)
            + ";"
            + "".join(labels)
            + f"concat=n={len(self.SEGMENTS)}:v=0:a=1[out]"
        )
        _ffmpeg(
            binary,
            "-i", str(source),
            "-filter_complex", graph,
            "-map", "[out]",
            "-ac", "1",
            "-ar", str(self.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(target),
        )
        return _read_mono_pcm(target)

    def test_the_join_steps_without_the_fade_and_not_with_it(self, tmp_path):
        binary = _ffmpeg_or_skip()
        source = tmp_path / "chirp.wav"
        _ffmpeg(
            binary,
            "-f", "lavfi",
            "-i", f"aevalsrc=0.8*sin(2*PI*(300*t+150*t*t)):s={self.SAMPLE_RATE}:d=4",
            "-ac", "1",
            "-ar", str(self.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(source),
        )

        join = int(self.SEGMENTS[0][1] * self.SAMPLE_RATE)
        clicking = self._render(binary, source, tmp_path / "nofade.wav", 0)
        faded = self._render(binary, source, tmp_path / "faded.wav", 5.0)

        # Steepest slope the signal reaches well away from the join, measured
        # on the un-faded render so the fade cannot flatter the reference.
        natural = _largest_step(clicking, join // 2, 200)
        assert natural > 0

        clicking_step = _largest_step(clicking, join, 3)
        faded_step = _largest_step(faded, join, 3)

        assert clicking_step > natural, (
            f"the un-faded splice should jump further than the waveform ever "
            f"does on its own: join step {clicking_step} vs natural {natural}"
        )
        assert faded_step < natural, (
            f"the faded splice should disappear into the waveform's own "
            f"motion: join step {faded_step} vs natural {natural}"
        )
        assert faded_step * 4 < clicking_step, (
            f"the fade should remove most of the step: {clicking_step} -> {faded_step}"
        )
