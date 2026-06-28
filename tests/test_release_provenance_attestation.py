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
    assert "gh release create" in docs
    assert "gh release upload" in docs
    assert ".github/workflows" not in docs
    assert "GitHub Actions" not in docs


def test_release_provenance_docs_include_bundled_ffmpeg_pin_and_hash():
    docs = _read(RELEASE_PROVENANCE_DOC)

    assert "8.1.2-essentials_build-www.gyan.dev" in docs
    assert "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" in docs
    assert "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec" in docs
    assert "Release lane" in docs
    assert ">= 8.1.1" in docs
    assert ">= 2026-06-10" in docs


def test_release_smoke_runs_release_provenance_guard():
    smoke = _read(RELEASE_SMOKE)

    assert "tests/test_release_provenance_attestation.py" in smoke
