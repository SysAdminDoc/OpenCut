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

from opencut.helpers import (  # noqa: E402
    EDGE_FADE_DEFAULT_MS,
    EDGE_FADE_MAX_MS,
    build_edge_fade_filter,
    edge_fade_ms,
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
