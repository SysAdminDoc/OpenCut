"""Deterministic synthetic media corpus for release-gated conformance tests.

The existing FFmpeg integration tests only ever see one shape of media:
24 fps CFR H.264 with stereo AAC starting at PTS 0. The combinations that
actually break automated edits — drop-frame rates, variable frame rate,
delayed PTS, decode errors mid-file, channel layouts, extra stream types,
10-bit HDR metadata, rotation, and non-ASCII paths — were never generated,
so nothing asserted what OpenCut does with them.

Everything here is synthesized from ``lavfi`` sources, so the corpus is
deterministic, tiny, and carries no third-party licence obligations. Each
entry declares what it *should* probe as; the conformance tests compare the
declaration against the real ffprobe output and against operation results, so
a fixture that silently fails to acquire its property is a test failure
rather than a vacuous pass.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

# Small and short: the corpus is about shape, not throughput.
SIZE = "160x120"
DURATION = 2.0

# Tolerances are declared once so a test cannot quietly widen its own.
DURATION_TOLERANCE_S = 0.35
SYNC_TOLERANCE_S = 0.10
TIMECODE_TOLERANCE_FRAMES = 2


def ffmpeg_available() -> bool:
    return bool(get_ffmpeg_path()) and bool(get_ffprobe_path())


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _ffmpeg(*args: str) -> None:
    cmd = [get_ffmpeg_path(), "-hide_banner", "-nostdin", "-y", *args]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"corpus build failed ({' '.join(args[-3:])}):\n{result.stderr[-2000:]}"
        )


def probe(path: str | Path) -> dict:
    """Full ffprobe JSON for *path* (streams + format)."""
    result = _run([
        get_ffprobe_path(), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout or "{}")


def streams_of(path: str | Path, codec_type: Optional[str] = None) -> list[dict]:
    streams = probe(path).get("streams", [])
    if codec_type is None:
        return streams
    return [s for s in streams if s.get("codec_type") == codec_type]


def format_duration(path: str | Path) -> float:
    return float(probe(path).get("format", {}).get("duration", 0.0) or 0.0)


def count_decode_errors(path: str | Path) -> int:
    """Number of decode errors FFmpeg reports while reading the whole file."""
    result = _run([
        get_ffmpeg_path(), "-hide_banner", "-nostdin",
        "-err_detect", "ignore_err", "-i", str(path), "-f", "null", "-",
    ])
    haystack = result.stderr.lower()
    return sum(
        haystack.count(marker)
        for marker in ("error while decoding", "corrupt", "invalid nal", "no frame")
    )


@dataclass
class Fixture:
    """One corpus entry plus the properties it is built to carry."""

    name: str
    filename: str
    build: Callable[[Path], None]
    expect: dict = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _video_audio(out: Path, *, rate: str, extra: Optional[list[str]] = None,
                 audio: str = "stereo") -> None:
    args = [
        "-f", "lavfi", "-i", f"testsrc2=duration={DURATION}:size={SIZE}:rate={rate}",
    ]
    if audio != "none":
        layout = {"mono": "mono", "stereo": "stereo", "5.1": "5.1"}[audio]
        args += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={DURATION}:sample_rate=48000",
            "-filter_complex", f"[1:a]aformat=channel_layouts={layout}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:a", "aac", "-b:a", "64k",
        ]
    else:
        args += ["-map", "0:v"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p"]
    args += list(extra or [])
    args += [str(out)]
    _ffmpeg(*args)


def _build_cfr(rate: str, audio: str = "stereo", extra=None):
    def _build(out: Path) -> None:
        _video_audio(out, rate=rate, audio=audio, extra=extra)
    return _build


def _build_vfr(out: Path) -> None:
    """Concatenate two different frame rates into one passthrough stream."""
    tmp_dir = out.parent / "_vfr_parts"
    tmp_dir.mkdir(exist_ok=True)
    part_a = tmp_dir / "a.mkv"
    part_b = tmp_dir / "b.mkv"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=duration=1:size={SIZE}:rate=24",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", str(part_a))
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=duration=1:size={SIZE}:rate=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", str(part_b))
    listing = tmp_dir / "list.txt"
    listing.write_text(
        f"file '{part_a.as_posix()}'\nfile '{part_b.as_posix()}'\n", encoding="utf-8"
    )
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing),
            "-fps_mode", "passthrough", "-c", "copy", str(out))


def _build_delayed_pts(out: Path) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=duration={DURATION}:size={SIZE}:rate=25",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}:sample_rate=48000",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-output_ts_offset", "3.5", "-muxdelay", "0", "-muxpreload", "0",
        str(out),
    )


def _build_corrupt(out: Path) -> None:
    """A structurally valid file whose payload is damaged mid-stream."""
    clean = out.parent / "_corrupt_source.mp4"
    if not clean.exists():
        _video_audio(clean, rate="25")
    data = bytearray(clean.read_bytes())
    # Damage a band in the middle of the media data, leaving the container
    # header and index intact so the file still opens and then errors.
    start = len(data) // 3
    end = min(len(data) - 1024, start + 2048)
    for index in range(start, end):
        data[index] ^= 0xFF
    out.write_bytes(bytes(data))


def _build_no_audio(out: Path) -> None:
    _video_audio(out, rate="25", audio="none")


def _build_subtitles(out: Path) -> None:
    srt = out.parent / "_captions.srt"
    srt.write_text(
        "1\n00:00:00,200 --> 00:00:01,000\nfirst cue\n\n"
        "2\n00:00:01,100 --> 00:00:01,900\nsecond cue\n",
        encoding="utf-8",
    )
    base = out.parent / "_subtitle_base.mp4"
    if not base.exists():
        _video_audio(base, rate="25")
    _ffmpeg("-i", str(base), "-i", str(srt),
            "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", str(out))


def _build_attachment(out: Path) -> None:
    note = out.parent / "_attachment.txt"
    note.write_text("opencut corpus attachment\n", encoding="utf-8")
    base = out.parent / "_attachment_base.mp4"
    if not base.exists():
        _video_audio(base, rate="25")
    _ffmpeg("-i", str(base), "-attach", str(note),
            "-metadata:s:t", "mimetype=text/plain",
            "-map", "0", "-c", "copy", str(out))


def _build_hdr10(out: Path) -> None:
    # The container-level colour options alone do not reach the H.264 VUI;
    # x264 has to be told as well, or the file probes as "unknown transfer"
    # and the fixture would assert nothing.
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=duration={DURATION}:size={SIZE}:rate=25",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-color_range", "tv",
        "-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        str(out),
    )


def _build_rotated(out: Path) -> None:
    base = out.parent / "_rotate_base.mp4"
    if not base.exists():
        _video_audio(base, rate="25")
    _ffmpeg("-display_rotation", "90", "-i", str(base), "-c", "copy", str(out))


def _build_proxy(out: Path) -> None:
    original = out.parent / CORPUS_BY_NAME["cfr_25"].filename
    _ffmpeg("-i", str(original), "-vf", "scale=80:60",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "34",
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(out))


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
CORPUS: tuple[Fixture, ...] = (
    Fixture(
        "cfr_25", "cfr_25.mp4", _build_cfr("25"),
        expect={"avg_frame_rate": "25/1", "channels": 2, "pix_fmt": "yuv420p"},
        description="Plain CFR baseline the other fixtures are compared against.",
    ),
    Fixture(
        "cfr_23976", "cfr_23976.mp4", _build_cfr("24000/1001"),
        expect={"avg_frame_rate": "24000/1001"},
        description="23.976 — the rate that turns naive frame math into drift.",
    ),
    Fixture(
        "cfr_2997_dropframe", "cfr_2997_df.mov",
        _build_cfr("30000/1001", extra=["-timecode", "00:59:50;00"]),
        expect={"avg_frame_rate": "30000/1001", "timecode": "00:59:50;00"},
        description="29.97 with a drop-frame start timecode.",
    ),
    Fixture(
        "cfr_5994", "cfr_5994.mp4", _build_cfr("60000/1001"),
        expect={"avg_frame_rate": "60000/1001"},
        description="59.94 — double-rate delivery.",
    ),
    Fixture(
        "vfr", "vfr.mkv", _build_vfr,
        expect={"variable_frame_rate": True},
        description="Two frame rates in one passthrough stream.",
    ),
    Fixture(
        "delayed_pts", "delayed_pts.mp4", _build_delayed_pts,
        expect={"start_time_min": 3.0},
        description="Non-zero start PTS, as screen recorders and cameras emit.",
    ),
    Fixture(
        "corrupt", "corrupt.mp4", _build_corrupt,
        expect={"decode_errors_min": 1},
        description="Valid container, damaged payload — decoding must continue.",
    ),
    Fixture(
        "mono", "mono.mp4", _build_cfr("25", audio="mono"),
        expect={"channels": 1},
        description="Single-channel audio.",
    ),
    Fixture(
        "multichannel", "multichannel.mp4", _build_cfr("25", audio="5.1"),
        expect={"channels": 6},
        description="5.1 audio that must not be silently downmixed.",
    ),
    Fixture(
        "no_audio", "no_audio.mp4", _build_no_audio,
        expect={"audio_streams": 0},
        description="Video-only source; audio filters must not be assumed.",
    ),
    Fixture(
        "with_subtitles", "with_subtitles.mp4", _build_subtitles,
        expect={"subtitle_streams": 1},
        description="Embedded timed-text subtitle stream.",
    ),
    Fixture(
        "with_attachment", "with_attachment.mkv", _build_attachment,
        expect={"attachment_streams": 1},
        description="Attachment stream, as font-carrying deliverables use.",
    ),
    Fixture(
        "hdr10", "hdr10.mp4", _build_hdr10,
        expect={
            "pix_fmt": "yuv420p10le",
            "color_transfer": "smpte2084",
            "color_space": "bt2020nc",
        },
        description="10-bit HDR colour metadata that must survive a copy.",
    ),
    Fixture(
        "rotated", "rotated.mp4", _build_rotated,
        expect={"rotation": 90},
        description="Display-matrix rotation, as phone footage carries.",
    ),
    Fixture(
        "unicode_path", "clip ünïcode — テスト.mp4",
        _build_cfr("25"),
        expect={"channels": 2},
        description="Spaces plus non-ASCII in the filename.",
    ),
    Fixture(
        "proxy", "proxy.mp4", _build_proxy,
        expect={"width": 80, "height": 60},
        description="Reduced-resolution proxy of cfr_25 for parity checks.",
    ),
)

CORPUS_BY_NAME: dict[str, Fixture] = {entry.name: entry for entry in CORPUS}

# `proxy` reads its source from disk, so it must build after `cfr_25`.
BUILD_ORDER: tuple[str, ...] = tuple(
    [name for name in CORPUS_BY_NAME if name != "proxy"] + ["proxy"]
)


def build_corpus(root: Path, names: Optional[list[str]] = None) -> dict[str, str]:
    """Build (or reuse) the corpus under *root* and return ``{name: path}``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    wanted = list(names) if names else list(BUILD_ORDER)
    order = [name for name in BUILD_ORDER if name in wanted]
    paths: dict[str, str] = {}
    for name in order:
        entry = CORPUS_BY_NAME[name]
        out = root / entry.filename
        if not out.exists() or out.stat().st_size == 0:
            entry.build(out)
        paths[name] = str(out)
    return paths


def build_one(root: Path, name: str) -> str:
    return build_corpus(root, [name])[name]
