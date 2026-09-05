"""Static guards for local release provenance documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"
RELEASE_PROVENANCE_DOC = REPO_ROOT / "docs" / "RELEASE_PROVENANCE.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_release_provenance_docs_name_local_manifest_commands():
    docs = _read(RELEASE_PROVENANCE_DOC)

    assert "python scripts/release_smoke.py --json" in docs
    assert "python scripts/sbom.py --format json --output dist/opencut-declared-sbom.cyclonedx.json" in docs
    assert "python scripts/verify_ffmpeg_provenance.py --manifest dist/ffmpeg-provenance.json" in docs
    assert "scripts/verify_embedded_media_provenance.py" in docs
    assert "dist/embedded-media-provenance.json" in docs
    assert "gh release create" in docs
    assert "gh release upload" in docs
    assert ".github/workflows" not in docs
    assert "GitHub Actions" not in docs


def test_release_provenance_docs_explain_embedded_decoder_policy():
    docs = _read(RELEASE_PROVENANCE_DOC)

    assert "OpenCV and PyAV each carry their own FFmpeg libraries" in docs
    assert "opencv_videoio_ffmpeg" in docs
    assert "FFmpeg 8.1.2 library floor" in docs


def test_release_provenance_docs_include_bundled_ffmpeg_pin_and_hash():
    """The document must state the pin the installers actually fetch.

    This used to assert the literal strings ``8.1.2-essentials_build-www.gyan.dev``,
    ``>= 8.1.1`` and ``>= 2026-06-10``. Every one of them had gone stale: the
    installers fetch a 2026-08-03 git snapshot, ``RELEASE_FLOOR`` is 8.1.3, the
    release lane is closed outright, and ``SNAPSHOT_FLOOR_DATE`` moved to
    2026-07-06. Because the assertions named the old values rather than reading
    the current ones, they kept passing while the public document became wrong
    in four places. Read the facts from the code instead.
    """
    from opencut.tools.check_provenance_docs import find_divergences

    divergences = find_divergences()
    assert not divergences, "\n".join(
        f"{item['document']}: {item['field']} -- {item['problem']}, "
        f"expected {item['expected']!r}"
        for item in divergences
    )


def test_release_provenance_docs_do_not_advertise_a_closed_lane():
    """A closed release lane must not read as an option."""
    from opencut.core import ffmpeg_provenance as provenance

    docs = _read(RELEASE_PROVENANCE_DOC)
    if not provenance.RELEASE_LANE_OPEN:
        assert "Release lane is closed" in docs
        assert "acceptable on **either** lane" not in docs, (
            "the document offers a release lane the code refuses"
        )


def test_provenance_doc_guard_can_actually_fail(tmp_path, monkeypatch):
    """Positive control: point the checker at a document that omits a fact."""
    from opencut.tools import check_provenance_docs

    empty = tmp_path / "RELEASE_PROVENANCE.md"
    empty.write_text("nothing useful here\n", encoding="utf-8")
    monkeypatch.setattr(check_provenance_docs, "RELEASE_PROVENANCE_DOC", empty)

    divergences = check_provenance_docs.find_divergences()
    assert divergences, "the guard passed against a document stating none of the facts"
    assert any("ffmpeg_provenance" in item["field"] for item in divergences)


def test_release_smoke_runs_release_provenance_guard():
    smoke = _read(RELEASE_SMOKE)

    assert "tests/test_release_provenance_attestation.py" in smoke
