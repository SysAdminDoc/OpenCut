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
6. ``validate_path`` — ``security.validate_path`` / ``validate_filepath``.
   User-supplied file paths must reject traversal, device, UNC, and invalid
   paths without surfacing unexpected platform exceptions.
7. ``otio_parse`` — OTIO JSON / adapter parsing. Uses OpenTimelineIO when
   installed and a lightweight JSON fallback otherwise.
8. ``fcpxml_parse`` — FCP XML parsing via ElementTree. Malformed XML should
   fail as a parse error, not crash the harness.
9. ``marker_import`` — marker CSV / Premiere CSV / EDL import parsers.
10. ``c2pa_sidecar`` — C2PA sidecar JSON verifier.
11. ``plugin_manifest`` — plugin manifest schema + lock validator.
12. ``webhook_signature`` — HMAC signature normalisation / verification.
13. ``safe_pip_install`` — package-spec validation used by safe_pip_install;
    this target never invokes pip.

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

import csv
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ATHERIS_FLAG = os.environ.get("RUN_FUZZ", "").lower() in ("1", "true", "yes")


def _atheris():
    """Lazy import so importing this test module doesn't hard-require atheris."""
    import atheris
    return atheris


def _decode_text(data: bytes, *, limit: int = 50_000) -> str:
    return data[:limit].decode("utf-8", errors="replace")


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


def _fuzz_validate_path(data: bytes) -> None:
    from opencut.security import validate_filepath, validate_path
    text = _decode_text(data, limit=4096)
    with tempfile.TemporaryDirectory(prefix="opencut_fuzz_paths_") as root:
        safe_file = os.path.join(root, "probe.txt")
        with open(safe_file, "wb") as fh:
            fh.write(b"ok")
        for candidate in (text, os.path.join(root, text[:240]), safe_file):
            try:
                validate_path(candidate, allowed_base=root)
            except (ValueError, OSError, TypeError):
                pass
            try:
                validate_filepath(candidate)
            except (ValueError, OSError, TypeError):
                pass


def _fuzz_otio_parse(data: bytes) -> None:
    """Exercise OTIO-compatible JSON parsing without requiring OTIO in CI."""
    text = _decode_text(data)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        tracks = payload.get("tracks") or payload.get("children") or []
        if isinstance(tracks, list):
            for track in tracks[:32]:
                if isinstance(track, dict):
                    _ = track.get("name", "")

    try:
        from opencut.export.otio_diff import check_otio_diff_available
    except ImportError:
        return
    if not check_otio_diff_available():
        return

    fd, path = tempfile.mkstemp(suffix=".otio", prefix="opencut_fuzz_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data[:100_000])
        try:
            import opentimelineio as otio
            otio.adapters.read_from_file(path)
        except Exception:  # noqa: BLE001
            # OTIO adapters raise a broad set of parser-specific exceptions for
            # malformed timelines; those are expected rejection paths.
            return
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _fuzz_fcpxml_parse(data: bytes) -> None:
    text = _decode_text(data)
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, UnicodeDecodeError, ValueError):
        return
    if root.tag.lower().endswith("fcpxml"):
        version = root.attrib.get("version", "")
        _ = version[:16]
    for idx, elem in enumerate(root.iter()):
        if idx > 100:
            break
        _ = elem.tag
        _ = list(elem.attrib.items())[:8]


def _fuzz_marker_import(data: bytes) -> None:
    from opencut.core.marker_import import import_markers, parse_csv, parse_edl, parse_premiere_csv
    text = _decode_text(data)
    fps = 1.0 + ((data[0] if data else 29) % 120)
    for parser in (parse_csv, parse_premiere_csv, parse_edl):
        try:
            parser(text, fps=fps)
        except (ValueError, TypeError, csv.Error):
            pass
    for fmt in ("csv", "premiere_csv", "edl"):
        try:
            import_markers(text=text, fps=fps, format=fmt)
        except (ValueError, TypeError, csv.Error):
            pass


def _fuzz_c2pa_sidecar(data: bytes) -> None:
    from opencut.core.c2pa_sidecar import verify_sidecar
    with tempfile.TemporaryDirectory(prefix="opencut_fuzz_c2pa_") as root:
        sidecar = Path(root) / "asset.mp4.c2pa.json"
        sidecar.write_bytes(data[:100_000])
        try:
            verify_sidecar(str(sidecar))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError):
            return


def _fuzz_plugin_manifest(data: bytes) -> None:
    from opencut.core.plugin_manifest import MANIFEST_FILENAME, validate_manifest_schema, validate_plugin_manifest
    text = _decode_text(data)
    try:
        manifest = json.loads(text)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
        manifest = None
    if manifest is not None:
        validate_manifest_schema(manifest)

    with tempfile.TemporaryDirectory(prefix="opencut_fuzz_plugin_") as root:
        plugin_dir = Path(root)
        (plugin_dir / MANIFEST_FILENAME).write_bytes(data[:100_000])
        try:
            validate_plugin_manifest(plugin_dir)
        except (OSError, UnicodeDecodeError, ValueError):
            return


def _fuzz_webhook_signature(data: bytes) -> None:
    from opencut.core.webhook_signature import sign_webhook_body, verify_webhook_signature
    secret_len = (data[0] % 64) if data else 0
    secret = _decode_text(data[1:1 + secret_len], limit=64) or "opencut-fuzz-secret"
    body = data[1 + secret_len:100_000]
    try:
        signature = sign_webhook_body(secret, body)
    except (TypeError, ValueError):
        return
    verify_webhook_signature(secret, body, signature)
    verify_webhook_signature(secret, body, _decode_text(data, limit=256))


def _fuzz_safe_pip_install(data: bytes) -> None:
    from opencut.security import validate_safe_pip_package
    text = _decode_text(data, limit=512)
    try:
        validate_safe_pip_package(text)
    except ValueError:
        return


TARGETS = {
    "srt_time": _fuzz_srt_time,
    "srt_file": _fuzz_srt_file,
    "cube_lut": _fuzz_cube_lut,
    "voice_grammar": _fuzz_voice_grammar,
    "event_moments_spikes": _fuzz_event_moments_spikes,
    "validate_path": _fuzz_validate_path,
    "otio_parse": _fuzz_otio_parse,
    "fcpxml_parse": _fuzz_fcpxml_parse,
    "marker_import": _fuzz_marker_import,
    "c2pa_sidecar": _fuzz_c2pa_sidecar,
    "plugin_manifest": _fuzz_plugin_manifest,
    "webhook_signature": _fuzz_webhook_signature,
    "safe_pip_install": _fuzz_safe_pip_install,
}

FUZZ_TARGETS = tuple(TARGETS)


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
        b'{"tracks": [{"name": "V1", "children": []}]}',
        b'<?xml version="1.0"?><fcpxml version="1.11"><library /></fcpxml>',
        b"timecode,name,duration,color,comment\n00:00:01:00,Intro,1,green,ok\n",
        b'{"asset": {"title": "asset.mp4", "sha256": "0", "bytes": 0}}',
        b'{"name": "demo", "version": "1", "description": "d", "api_version": 1, "capabilities": []}',
        b"requests==2.32.0",
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
