"""F203 Windows Authenticode signing tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sign_windows_artifacts.ps1"
DOC = REPO_ROOT / "docs" / "WINDOWS_CODESIGNING.md"
POLICY_DOC = REPO_ROOT / "docs" / "INSTALLER_POLICY.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_codesigning_script_exists_and_uses_signtool():
    text = _read(SCRIPT)

    assert "F203" in text
    assert "signtool.exe" in text
    assert "signtool sign" in text
    assert "signtool verify" in text
    assert "/fd SHA256" in text
    assert "/td SHA256" in text


def test_codesigning_script_uses_secret_backed_pfx_and_timestamp():
    text = _read(SCRIPT)

    for token in (
        "WINDOWS_CODESIGN_PFX_BASE64",
        "WINDOWS_CODESIGN_PFX_PASSWORD",
        "WINDOWS_CODESIGN_TIMESTAMP_URL",
        "http://timestamp.digicert.com",
    ):
        assert token in text
    assert "FromBase64String" in text
    assert "Remove-Item -LiteralPath $pfxPath" in text


def test_codesigning_script_has_renewal_warning_gate():
    text = _read(SCRIPT)

    assert "WINDOWS_CODESIGN_CERT_EXPIRES_AT" in text
    assert "RenewalWarningDays = 90" in text
    assert "FailOnExpiringCert" in text
    assert "renew before release" in text


def test_codesigning_doc_and_policy_reference_f203():
    doc = _read(DOC)
    policy = _read(POLICY_DOC)

    for token in (
        "WINDOWS_CODESIGN_PFX_BASE64",
        "WINDOWS_CODESIGN_PFX_PASSWORD",
        "WINDOWS_CODESIGN_CERT_EXPIRES_AT",
        "90-day renewal window",
        "signtool verify",
    ):
        assert token in doc
    assert "Required Environment Variables" in doc
    assert "GitHub Actions" not in doc
    assert ".github/workflows" not in doc
    assert "F203 status" in policy
    assert "docs/WINDOWS_CODESIGNING.md" in policy
