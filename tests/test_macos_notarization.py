from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notarization_script_uses_notarytool_and_hardened_runtime():
    script = (REPO_ROOT / "scripts" / "notarize_macos.sh").read_text(encoding="utf-8")

    assert "xcrun notarytool submit" in script
    assert "--options runtime" in script
    assert "--timestamp" in script
    assert "altool" not in script
    assert "MACOS_CERTIFICATE_P12_BASE64" in script
    assert "APPLE_API_PRIVATE_KEY" in script


def test_notarization_docs_name_required_secrets_and_apple_docs():
    docs = (REPO_ROOT / "docs" / "MACOS_NOTARIZATION.md").read_text(encoding="utf-8")

    for secret in (
        "MACOS_CERTIFICATE_P12_BASE64",
        "MACOS_CERTIFICATE_PASSWORD",
        "APPLE_API_KEY_ID",
        "APPLE_API_ISSUER_ID",
        "APPLE_API_PRIVATE_KEY",
    ):
        assert secret in docs

    assert "developer.apple.com/documentation/security/notarizing-macos-software-before-distribution" in docs
    assert "notarytool" in docs
    assert "scripts/notarize_macos.sh --check-env" in docs
    assert "GitHub Actions" not in docs
    assert ".github/workflows" not in docs
