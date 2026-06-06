"""Static guards for release artifact provenance attestations."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"
RELEASE_PROVENANCE_DOC = REPO_ROOT / "docs" / "RELEASE_PROVENANCE.md"

ATTEST_ACTION = "actions/attest@281a49d4cbb0a72c9575a50d18f6deb515a11deb # v4"
EXPECTED_SUBJECT_PATHS = {
    "release-upload-artifacts/server/*",
    "release-artifacts/OpenCut-Linux-Desktop-Packages/*",
    "release-artifacts/OpenCut-Setup-Windows/*",
    "release-artifacts/OpenCut-Declared-Dependency-SBOM-CycloneDX/opencut-declared-sbom.cyclonedx.json",
}


def _workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8").replace("\r\n", "\n")


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing job block {job_name!r}"
    return match.group("body")


def _step_block(job: str, step_name: str) -> str:
    match = re.search(
        rf"^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - name: |\Z)",
        job,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing step {step_name!r}"
    return match.group("body")


def _release_upload_job() -> str:
    return _job_block(_workflow_text(), "release-upload")


def test_release_upload_job_has_attestation_permissions():
    job = _release_upload_job()
    match = re.search(
        r"^    permissions:\n(?P<body>(?:      [A-Za-z-]+: (?:read|write)\n)+)\n    steps:",
        job,
        flags=re.MULTILINE,
    )
    assert match, "release-upload job permissions must be explicit"
    permissions = dict(line.strip().split(": ", maxsplit=1) for line in match.group("body").splitlines())

    assert permissions["contents"] == "write"
    assert permissions["id-token"] == "write"
    assert permissions["attestations"] == "write"


def test_attestation_happens_after_packaging_and_before_uploads():
    job = _release_upload_job()

    assert job.index("Download build artifacts") < job.index("Package server artifacts for release")
    assert job.index("Package server artifacts for release") < job.index("Generate release artifact attestations")
    assert job.index("Generate release artifact attestations") < job.index("Upload server artifacts to release")
    assert job.index("Generate release artifact attestations") < job.index("Upload Linux desktop packages to release")
    assert job.index("Generate release artifact attestations") < job.index("Upload installer to release")
    assert job.index("Generate release artifact attestations") < job.index("Upload SBOM to release")


def test_release_attestation_covers_uploaded_artifact_paths():
    attest_step = _step_block(_release_upload_job(), "Generate release artifact attestations")

    assert f"uses: {ATTEST_ACTION}" in attest_step
    assert "subject-path: |" in attest_step
    for subject_path in EXPECTED_SUBJECT_PATHS:
        assert subject_path in attest_step


def test_server_release_upload_uses_pre_attested_packaged_artifacts():
    job = _release_upload_job()
    package_step = _step_block(job, "Package server artifacts for release")
    upload_step = _step_block(job, "Upload server artifacts to release")

    assert 'mkdir -p release-upload-artifacts/server' in package_step
    assert 'cp "$artifact_dir/OpenCut-Server-macOS.zip" "release-upload-artifacts/server/OpenCut-Server-macOS.zip"' in package_step
    assert 'tar -C "$artifact_dir" -czf "release-upload-artifacts/server/$artifact_name.tar.gz" OpenCut-Server' in package_step
    assert "artifacts=(release-upload-artifacts/server/*)" in upload_step
    assert "gh release upload ${{ github.ref_name }} \"${artifacts[@]}\" --clobber" in upload_step
    assert "tar -C" not in upload_step


def test_release_provenance_docs_include_attestation_verification_commands():
    docs = RELEASE_PROVENANCE_DOC.read_text(encoding="utf-8")

    assert "gh attestation verify OpenCut-Server-Linux.tar.gz -R SysAdminDoc/OpenCut" in docs
    assert "gh attestation verify opencut-declared-sbom.cyclonedx.json -R SysAdminDoc/OpenCut" in docs
    assert "release-upload-artifacts/server/*" in docs


def test_release_smoke_runs_release_provenance_guard():
    smoke = RELEASE_SMOKE.read_text(encoding="utf-8")

    assert "tests/test_release_provenance_attestation.py" in smoke
