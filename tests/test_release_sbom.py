import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SBOM_PATH = "dist/opencut-declared-sbom.cyclonedx.json"


def _metadata_properties(bom: dict) -> dict[str, str]:
    return {prop["name"]: prop["value"] for prop in bom["metadata"].get("properties", [])}


def test_sbom_generator_emits_cyclonedx_json(tmp_path):
    output = tmp_path / "opencut-declared-sbom.cyclonedx.json"
    result = subprocess.run(
        [sys.executable, "scripts/sbom.py", "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    bom = json.loads(output.read_text(encoding="utf-8"))
    assert "SBOM written" in result.stdout
    assert bom["bomFormat"] == "CycloneDX"
    assert bom["specVersion"] == "1.5"
    assert bom["metadata"]["component"]["name"] == "opencut-ppro"
    properties = _metadata_properties(bom)
    assert properties["opencut:sbom:fidelity"] == "declared-only"
    assert "requirements-lock.txt" in properties["opencut:sbom:vulnerability-audit-targets"]
    assert any(component["name"] == "flask" for component in bom["components"])


def test_local_release_docs_and_smoke_cover_declared_sbom():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")
    docs = (REPO_ROOT / "docs" / "RELEASE_PROVENANCE.md").read_text(encoding="utf-8")

    assert "tests/test_release_sbom.py" in smoke
    assert f"python scripts/sbom.py --format json --output {SBOM_PATH}" in docs
    assert "dist/opencut-sbom.cyclonedx.json" not in docs
