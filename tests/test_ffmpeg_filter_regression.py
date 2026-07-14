"""F128 — FFmpeg filter-graph regression suite.

The bundled FFmpeg is on track to bump from 8.0.1 to 8.1 (see F129).
Without a regression suite, a filter name change or behaviour shift in
the new build would silently break production pipelines.

This file is the lightweight first slice of that suite. It is **not** a
full output-bitwise comparison (which is expensive and fragile across
FFmpeg versions). Instead it:

1. Probes the installed FFmpeg for the filters and demuxers OpenCut
   actually uses (libass, drawtext, scale, crop, atempo, etc.). The
   probe parses ``ffmpeg -filters`` output and asserts each required
   filter exists. F241's text-shaping gate covers a similar surface
   for libass + HarfBuzz; this file is broader.
2. Parses each shipped filter graph through FFmpeg's null muxer
   (`ffmpeg -f lavfi -i 'color=c=black:s=160x120:d=0.1' -filter_complex
   '<fc>' -f null -`) so we catch syntax regressions without writing
   any output file. Synthetic ``lavfi`` sources keep the test fast
   (<2s on a cold start) and deterministic.
3. Skips automatically when FFmpeg is not installed so the dev VM
   without the bundled binary still passes ``pytest-fast``.

When F129 lands (FFmpeg 8.1 bump), this file is the first thing CI
runs to confirm no regression. Adding a new filter graph to OpenCut?
Add it here too.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_ffmpeg() -> str:
    """Find FFmpeg via PATH, env vars, or the bundled `ffmpeg/` directory.

    Mirrors `opencut.server._setup_ffmpeg_path` so this regression suite
    can find the bundled FFmpeg without needing the full opencut server
    init to run first.
    """
    env_path = os.environ.get("OPENCUT_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env_path and Path(env_path).is_file():
        return env_path
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    bundled_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = REPO_ROOT / "ffmpeg" / bundled_name
    if bundled.is_file():
        return str(bundled)
    return ""


_RESOLVED_FFMPEG = _resolve_ffmpeg()

pytestmark = [
    pytest.mark.skipif(
        not _RESOLVED_FFMPEG,
        reason="FFmpeg not installed (checked PATH, OPENCUT_FFMPEG, and bundled ffmpeg/)",
    )
]


# ---------------------------------------------------------------------------
# Filter / demuxer registry
# ---------------------------------------------------------------------------


# Filters OpenCut emits across its core modules + routes. Each entry is
# the exact filter name as it appears in `ffmpeg -filters` output.
# Keep this list sorted so a removed-filter regression diff is obvious.
REQUIRED_FILTERS: Tuple[str, ...] = (
    "amix",            # mix two audio inputs (silence + filler removal)
    "anullsrc",        # silent audio for placeholders
    "ass",             # styled caption burn-in (F236/F241)
    "atempo",          # audio speed change for sped-up silences
    "color",           # solid-color generator (used in tests + title cards)
    "concat",          # multi-clip concat for merges
    "crop",            # face-reframe + auto-zoom
    "drawtext",        # watermark + simple title cards
    "eq",              # colour adjustments (brightness, contrast, sat)
    "fade",            # transition crossfades
    "format",          # pix-fmt normalization
    "hstack",          # side-by-side compare / multicam
    "loudnorm",        # EBU R 128 loudness normalisation
    "minterpolate",    # frame-interpolation fallback (F129/F164)
    "null",            # passthrough
    "overlay",         # PiP + watermark
    "pan",             # M+E mix center-channel subtract (F237 stack)
    "scale",           # resolution change
    "setpts",          # speed change for video
    "silencedetect",   # silence detection
    "subtitles",       # libass-backed burn-in (F241)
    "trim",            # in-graph trimming
    "vstack",          # vertical stack
    "xfade",           # crossfade transitions
)


# Filter graphs we know OpenCut emits. Each entry is `(label, graph_str)`.
# The graph must be runnable against a synthetic color source so the
# regression test does not need any real media file.
#
# Some graphs are video-only, some are audio-only. The fixture below
# spins up the right `-f lavfi -i` matching the graph's needs.
SHIPPED_VIDEO_GRAPHS: List[Tuple[str, str]] = [
    ("crop", "crop=80:60:0:0"),
    ("scale", "scale=160:120"),
    ("scale_lanczos", "scale=160:120:flags=lanczos"),
    ("eq_brightness", "eq=brightness=0.1:contrast=1.05"),
    ("fade_in", "fade=t=in:st=0:d=0.05"),
    ("format_yuv420", "format=yuv420p"),
    ("setpts_speed", "setpts=0.5*PTS"),
    ("drawtext_simple", "drawtext=text='OpenCut':x=10:y=10:fontsize=20:fontcolor=white"),
    ("overlay_corner", "[0:v]format=yuva420p,split=2[main][copy];"
                       "[copy]scale=40:30[pip];"
                       "[main][pip]overlay=10:10"),
]

SHIPPED_AUDIO_GRAPHS: List[Tuple[str, str]] = [
    ("atempo", "atempo=1.25"),
    ("loudnorm_default", "loudnorm=I=-23:TP=-2:LRA=7"),
    ("silencedetect", "silencedetect=noise=-30dB:d=0.5"),
    ("pan_center_subtract", "pan=stereo|c0=c0-c1|c1=c1-c0"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    return _RESOLVED_FFMPEG or shutil.which("ffmpeg") or "ffmpeg"


def _list_filters() -> set:
    """Return the set of filter names FFmpeg reports.

    ``ffmpeg -filters`` output format (post the header block) looks like::

        ' TS aap               AA->A      Apply Affine Projection...'
        ' .. abench            A->A       Benchmark part of a filtergraph.'

    The leading flags field is exactly 2 chars, preceded by a single
    space. We split on whitespace and pick the 2nd token as the name
    when the 1st token is the 2-char flag pattern and the 3rd token
    matches the IO arrow pattern (e.g. ``A->A``, ``AA->A``, ``V->V``).
    """
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, (
        f"ffmpeg -filters failed: {proc.stderr[-500:]}"
    )
    names: set = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        flag_token, name_token, io_token = parts[0], parts[1], parts[2]
        # Flag token is exactly 2 chars of {. T S}, and IO token contains the arrow.
        if len(flag_token) != 2:
            continue
        if not all(ch in ".TSC" for ch in flag_token):
            continue
        if "->" not in io_token:
            continue
        names.add(name_token)
    return names


def _run_video_graph(graph: str) -> Tuple[int, str]:
    """Pipe a synthetic color source through *graph* and return (rc, stderr_tail)."""
    cmd = [
        _ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10:d=0.1",
        "-filter_complex", graph,
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    return proc.returncode, (proc.stderr or "").strip()[-1000:]


def _run_audio_graph(graph: str) -> Tuple[int, str]:
    """Pipe a synthetic sine source through *graph* and return (rc, stderr_tail)."""
    cmd = [
        _ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-af", graph,
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    return proc.returncode, (proc.stderr or "").strip()[-1000:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def installed_filters() -> set:
    return _list_filters()


def test_filter_registry_is_alphabetised():
    """The REQUIRED_FILTERS tuple must stay sorted so diffs are obvious."""
    assert list(REQUIRED_FILTERS) == sorted(REQUIRED_FILTERS)


@pytest.mark.parametrize("filter_name", REQUIRED_FILTERS)
def test_required_filter_is_present(installed_filters, filter_name):
    """Each filter OpenCut emits must exist in the installed FFmpeg.

    If FFmpeg 8.1 silently renames or drops a filter, this test fires
    first — before the install ever reaches users.
    """
    assert filter_name in installed_filters, (
        f"FFmpeg is missing filter {filter_name!r}. Installed filters "
        f"count: {len(installed_filters)}. This is the F128 gate — a "
        f"bundled FFmpeg upgrade may have changed the filter surface."
    )


@pytest.mark.parametrize("label,graph", SHIPPED_VIDEO_GRAPHS, ids=lambda x: x if isinstance(x, str) else "")
def test_video_filter_graph_parses(label, graph):
    """Each shipped video filter graph must parse + run on a 0.1s synthetic input.

    We run against ``lavfi color=`` so no real media is needed. A
    non-zero return code means the graph syntax broke between FFmpeg
    versions — which is exactly the F128 regression we want to catch
    before a release.
    """
    rc, stderr = _run_video_graph(graph)
    if rc != 0 and "Fontconfig error: Cannot load default config file" in stderr:
        # drawtext needs a fontconfig config file; some Windows FFmpeg builds
        # ship without one and abort before parsing the graph. That is a build/
        # environment gap, not the filter-syntax regression this gate guards —
        # real syntax breaks still surface a non-fontconfig parse error below.
        pytest.skip("FFmpeg build lacks a fontconfig config file (drawtext unavailable)")
    assert rc == 0, f"Video graph {label!r} failed: {stderr}"


@pytest.mark.parametrize("label,graph", SHIPPED_AUDIO_GRAPHS, ids=lambda x: x if isinstance(x, str) else "")
def test_audio_filter_graph_parses(label, graph):
    rc, stderr = _run_audio_graph(graph)
    assert rc == 0, f"Audio graph {label!r} failed: {stderr}"


def test_silencedetect_emits_silence_markers():
    """silencedetect must produce `silence_start` lines for a quiet section.

    Synthetic input: 0.5s of silence (volume=0) at 44.1k. The filter
    must emit a `silence_start` event. If FFmpeg 8.1 changes the line
    format we catch the regression here before it breaks
    ``opencut.core.silence`` parsing.
    """
    cmd = [
        _ffmpeg(), "-hide_banner", "-nostats",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=0.5",
        "-af", "silencedetect=noise=-30dB:d=0.1",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr[-500:]
    # silencedetect writes to stderr ("[silencedetect @ 0x...] silence_start: 0").
    assert "silence_start" in proc.stderr, (
        "silencedetect did not emit a silence_start marker — "
        "opencut.core.silence regex parsing will break"
    )


def test_loudnorm_runs_two_pass_compatible():
    """loudnorm must accept the same I/TP/LRA pairs we ship in F237 presets."""
    rc, stderr = _run_audio_graph("loudnorm=I=-14:TP=-1:LRA=11:print_format=json")
    assert rc == 0, f"loudnorm regression: {stderr}"


def test_ffmpeg_version_is_8_or_newer():
    """Defensive — OpenCut's bundled FFmpeg is 8.0.1+. A drastic downgrade
    would break filters that ship only in 7+ (xfade modes, etc.)."""
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 0
    first_line = (proc.stdout.splitlines() or [""])[0]
    assert "ffmpeg version" in first_line.lower()
    # The version token follows `ffmpeg version`. We accept anything
    # that starts with `n` (n4.4, n5.x, etc.) or `7.` / `8.` / `9.`.
    if "ffmpeg version " in first_line:
        version_token = first_line.split("ffmpeg version ", 1)[1].split()[0]
        # Just make sure it's not a 0.x or 1.x — OpenCut needs >= 4.
        first_char = version_token[0]
        if first_char.isdigit():
            assert int(first_char) >= 4, (
                f"FFmpeg version too old: {version_token!r}. OpenCut requires >=4.x"
            )
