import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SBOM_PATH = "dist/opencut-sbom.cyclonedx.json"


def test_sbom_generator_emits_cyclonedx_json(tmp_path):
    output = tmp_path / "opencut-sbom.cyclonedx.json"
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
    assert bom["metadata"]["component"]["name"] == "opencut"
    assert any(component["name"] == "flask" for component in bom["components"])


def test_release_workflow_generates_and_uploads_sbom():
    workflow = (REPO_ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")

    assert "Generate release SBOM" in workflow
    assert "Archive release SBOM" in workflow
    assert "Upload SBOM to release" in workflow
    assert f"python scripts/sbom.py --format json --output {SBOM_PATH}" in workflow
    assert f"gh release upload ${{{{ github.ref_name }}}} {SBOM_PATH} --clobber" in workflow
    assert "OpenCut-SBOM-CycloneDX" in workflow
