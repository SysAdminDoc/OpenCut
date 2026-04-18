"""
Atheris fuzz tests for OpenCut's text-format parsers.

Atheris (https://github.com/google/atheris, Apache-2) drives
parser entry points with structured random input to surface crashes,
unhandled exceptions, and DoS-able inputs before attackers /
malformed-file-handlers do.

Target parsers (pick targets at runtime via the ``target`` arg):

1. ``srt_time`` — ``caption_compliance._parse_srt_time`` SRT timestamp
   parser.  Rejecting malformed timestamps without raising is the
   security-critical contract; fuzz verifies no un-caught exceptions
   escape.
2. ``srt_file`` — the full ``caption_compliance._parse_srt`` file
   parser. Fuzzes an on-disk .srt with randomly-structured content.
3. ``cube_lut`` — ``lut_library._parse_cube`` (.cube LUT file parser).
   ``.cube`` files are user-supplied; a malformed one must not crash
   the server.
4. ``voice_grammar`` — ``voice_command_grammar.parse``. Must be total
   — it's on the hot request path and mustn't raise.
5. ``event_moments_spikes`` — ``event_moments._find_spikes``. Must
   tolerate any float envelope without `IndexError` / `ZeroDivisionError`.

Running
-------

Default ``pytest`` invocations skip these targets because they're
infinite loops by design. Opt in with ``RUN_FUZZ=1`` and pass the
target via env or pytest `-k`::

    RUN_FUZZ=1 FUZZ_TARGET=srt_time pytest tests/fuzz/ -s

Atheris + libFuzzer flags pass through positionally after ``--`` on
the command line, e.g. ``-max_total_time=30``.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

ATHERIS_FLAG = os.environ.get("RUN_FUZZ", "").lower() in ("1", "true", "yes")

FUZZ_TARGETS = (
    "srt_time",
    "srt_file",
    "cube_lut",
    "voice_grammar",
    "event_moments_spikes",
)


def _atheris():
    """Lazy import so importing this test module doesn't hard-require atheris."""
    import atheris
    return atheris


# ---------------------------------------------------------------------------
# Target harnesses
# ---------------------------------------------------------------------------

def _fuzz_srt_time(data: bytes) -> None:
    from opencut.core.caption_compliance import _parse_srt_time
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return
    try:
        _parse_srt_time(text)
    except (ValueError, TypeError):
        # Expected rejection paths — the parser is allowed to say no.
        return


def _fuzz_srt_file(data: bytes) -> None:
    from opencut.core.caption_compliance import _parse_srt
    # The parser reads from disk, so write the fuzz bytes to a temp .srt
    fd, path = tempfile.mkstemp(suffix=".srt", prefix="opencut_fuzz_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            _parse_srt(path)
        except (ValueError, UnicodeDecodeError, OSError):
            return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _fuzz_cube_lut(data: bytes) -> None:
    from opencut.core.lut_library import _parse_cube
    fd, path = tempfile.mkstemp(suffix=".cube", prefix="opencut_fuzz_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            _parse_cube(path)
        except (ValueError, OSError, UnicodeDecodeError):
            return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _fuzz_voice_grammar(data: bytes) -> None:
    from opencut.core.voice_command_grammar import parse
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return
    # ``parse`` advertises "never raises" — any exception surfaced here
    # is a real defect.
    parse(text)


def _fuzz_event_moments_spikes(data: bytes) -> None:
    """Feed random-length float envelopes into ``_find_spikes``."""
    from opencut.core.event_moments import _find_spikes
    if not data:
        return
    # Convert each byte to a pseudo-RMS value in [0, 1]. Truncate to
    # at most 10_000 samples to keep individual fuzz iterations fast.
    env = [(b & 0xFF) / 255.0 for b in data[:10_000]]
    hop = 0.1
    min_spacing = ((data[0] & 0x7F) or 1) * 0.1
    k_sigma = 0.5 + (data[-1] & 0x0F) * 0.25
    _find_spikes(env, hop, min_spacing, k_sigma)


TARGETS = {
    "srt_time": _fuzz_srt_time,
    "srt_file": _fuzz_srt_file,
    "cube_lut": _fuzz_cube_lut,
    "voice_grammar": _fuzz_voice_grammar,
    "event_moments_spikes": _fuzz_event_moments_spikes,
}


# ---------------------------------------------------------------------------
# Atheris entry point
# ---------------------------------------------------------------------------

def run_fuzz(target: str) -> None:
    """Configure Atheris for the named target and hand off to libFuzzer.

    This function returns only when libFuzzer's stop condition is hit
    (e.g. ``-max_total_time=30`` passed after ``--``). In normal CI
    usage the caller sets a short time budget.
    """
    if target not in TARGETS:
        raise SystemExit(
            f"Unknown fuzz target {target!r}. Valid: {sorted(TARGETS)}"
        )
    atheris = _atheris()
    atheris.Setup(sys.argv, TARGETS[target])
    atheris.Fuzz()


# ---------------------------------------------------------------------------
# Pytest wrapper — skipped by default
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ATHERIS_FLAG, reason="set RUN_FUZZ=1 to enable")
@pytest.mark.parametrize("target", list(TARGETS.keys()))
def test_fuzz_target_smoke(target):
    """Single-input sanity check so CI knows the harness loads.

    With ``RUN_FUZZ=1``, each target is called with a handful of
    deterministic payloads — this is NOT full fuzzing, just a smoke
    guard to catch "import broke the fuzz target" regressions.  Run
    ``python -m tests.fuzz.test_parser_fuzz <target> -max_total_time=N``
    for real libFuzzer runs.
    """
    payloads = [
        b"",
        b"00:00:01,000",
        b"1\n00:00:00,000 --> 00:00:01,500\nhello\n\n",
        b"# comment\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n",
        b"cut here",
        b"speed up one point five x",
        b"\x00\x01\x02\x03" * 64,
        bytes(range(256)),
    ]
    for p in payloads:
        TARGETS[target](p)


if __name__ == "__main__":
    # CLI mode: python -m tests.fuzz.test_parser_fuzz <target>
    if len(sys.argv) < 2:
        print(
            "Usage: python -m tests.fuzz.test_parser_fuzz <target> "
            "[libFuzzer flags]\n"
            f"Targets: {', '.join(sorted(TARGETS))}"
        )
        sys.exit(2)
    _target = sys.argv.pop(1)
    run_fuzz(_target)
