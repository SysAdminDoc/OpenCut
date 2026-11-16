"""F467: the release build signs the CEP panel and ships the signed contents.

These run installer/CepSigning.ps1 under Windows PowerShell 5.1 against a
throwaway panel and a throwaway self-signed certificate, so they need Windows
and Adobe's ZXPSignCmd. Signing skips the timestamp server to stay offline.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "installer" / "CepSigning.ps1"
_FOUND_POWERSHELL = shutil.which("powershell")
POWERSHELL = _FOUND_POWERSHELL or "powershell"
DSIG = "{http://www.w3.org/2000/09/xmldsig#}"
CERT_PASSWORD = "OpenCutTestOnly2026"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or _FOUND_POWERSHELL is None,
    reason="CEP signing runs under Windows PowerShell",
)

MANIFEST = """<?xml version="1.0" encoding="UTF-8"?>
<ExtensionManifest Version="7.0" ExtensionBundleId="com.opencut.panel" ExtensionBundleVersion="0.0.1">
  <ExtensionList><Extension Id="com.opencut.panel.main" Version="0.0.1" /></ExtensionList>
</ExtensionManifest>
"""


def _ps(body: str, **env: str) -> subprocess.CompletedProcess[str]:
    script = ". $env:OC_MODULE\n$ErrorActionPreference = 'Stop'\n" + body
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        env={**os.environ, "OC_MODULE": str(MODULE), **env},
        timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@pytest.fixture(scope="module")
def zxpsigncmd() -> str:
    result = _ps("try { Resolve-ZxpSignCmd } catch { 'MISSING'; exit 0 }")
    tool = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "MISSING"
    if result.returncode != 0 or tool == "MISSING":
        pytest.skip("ZXPSignCmd.exe is not installed")
    return tool


@pytest.fixture(scope="module")
def certificate(tmp_path_factory, zxpsigncmd) -> Path:
    cert = tmp_path_factory.mktemp("cert") / "test.p12"
    subprocess.run(
        [zxpsigncmd, "-selfSignedCert", "US", "NA", "OpenCutTest", "OpenCutTest", CERT_PASSWORD, str(cert)],
        check=True,
        capture_output=True,
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return cert


def _make_panel(root: Path) -> Path:
    panel = root / "com.opencut.panel"
    (panel / "CSXS").mkdir(parents=True)
    (panel / "CSXS" / "manifest.xml").write_text(MANIFEST, encoding="utf-8")
    (panel / "client").mkdir()
    (panel / "client" / "index.html").write_text("<!doctype html><title>OpenCut</title>\n", encoding="utf-8")
    (panel / "client" / "main.js").write_text("console.log('panel');\n", encoding="utf-8")
    return panel


def _sign(panel: Path, zxp: Path, certificate: Path) -> subprocess.CompletedProcess[str]:
    return _ps(
        "try { Invoke-CepPanelSigning -ExtensionDir $env:OC_EXT -ZxpPath $env:OC_ZXP "
        "-Certificate $env:OC_CERT -Password $env:OC_PWD -SkipTimestamp | Out-Null; 'SIGNED' } "
        "catch { 'ERROR: ' + $_.Exception.Message; exit 3 }",
        OC_EXT=str(panel),
        OC_ZXP=str(zxp),
        OC_CERT=str(certificate),
        OC_PWD=CERT_PASSWORD,
    )


def _verify(path: Path, require_timestamp: bool = False) -> subprocess.CompletedProcess[str]:
    flag = " -RequireTimestamp" if require_timestamp else ""
    return _ps(
        f"try {{ Assert-CepPanelSignature -Path $env:OC_PATH{flag} | Out-Null; 'VERIFIED' }} "
        "catch { 'ERROR: ' + $_.Exception.Message; exit 3 }",
        OC_PATH=str(path),
    )


@pytest.fixture()
def signed_panel(tmp_path, certificate) -> tuple[Path, Path, dict[str, bytes]]:
    panel = _make_panel(tmp_path)
    original = {p.relative_to(panel).as_posix(): p.read_bytes() for p in panel.rglob("*") if p.is_file()}
    zxp = tmp_path / "dist" / "OpenCut-Panel-0.0.1.zxp"
    result = _sign(panel, zxp, certificate)
    assert "SIGNED" in result.stdout, result.stdout + result.stderr
    return panel, zxp, original


def test_signed_panel_ships_the_package_contents_and_verifies(signed_panel):
    panel, zxp, original = signed_panel

    assert zxp.is_file()
    assert (panel / "META-INF" / "signatures.xml").is_file()
    assert (panel / "mimetype").is_file()
    for rel, data in original.items():
        assert (panel / rel).read_bytes() == data, rel
    assert "VERIFIED" in _verify(panel).stdout
    assert "VERIFIED" in _verify(zxp).stdout


def test_signature_manifest_covers_every_shipped_file(signed_panel):
    panel, _zxp, _original = signed_panel
    tree = ET.parse(panel / "META-INF" / "signatures.xml")
    manifest = next(m for m in tree.iter(f"{DSIG}Manifest") if m.get("Id") == "PackageContents")
    digests = {
        ref.get("URI"): ref.findtext(f"{DSIG}DigestValue")
        for ref in manifest.iter(f"{DSIG}Reference")
    }
    shipped = {
        p.relative_to(panel).as_posix()
        for p in panel.rglob("*")
        if p.is_file() and p.relative_to(panel).as_posix() != "META-INF/signatures.xml"
    }

    assert set(digests) == shipped
    for rel, digest in digests.items():
        expected = base64.b64encode(hashlib.sha256((panel / rel).read_bytes()).digest()).decode()
        assert digest == expected, rel


@pytest.mark.parametrize("change", ["tamper", "extra_file", "remove_file"])
def test_verify_rejects_a_changed_panel(signed_panel, change):
    panel, _zxp, _original = signed_panel
    if change == "tamper":
        with (panel / "client" / "main.js").open("a", encoding="utf-8") as handle:
            handle.write("fetch('https://example.invalid');\n")
    elif change == "extra_file":
        (panel / "client" / "injected.js").write_text("alert(1);\n", encoding="utf-8")
    else:
        (panel / "client" / "index.html").unlink()

    result = _verify(panel)
    assert result.returncode == 3
    assert "CEP signature check failed" in result.stdout


def test_require_timestamp_rejects_an_untimestamped_signature(signed_panel):
    panel, _zxp, _original = signed_panel
    result = _verify(panel, require_timestamp=True)
    assert result.returncode == 3
    assert "no valid timestamp" in result.stdout


def _fake_zxpsigncmd(tmp_path: Path, timestamp_line: str) -> Path:
    # ZXPSignCmd 4.1.3 prints one of three states after "Timestamp: ". A DigiCert
    # timestamped panel verified on 2026-11-15 printed the first one.
    tool = tmp_path / "fake-zxpsigncmd.cmd"
    tool.write_text(
        f"@echo off\r\necho Timestamp: {timestamp_line}\r\necho Signature verified successfully\r\n",
        encoding="ascii",
    )
    return tool


@pytest.mark.parametrize(
    ("timestamp_line", "accepted"),
    [
        ("Valid and within certificate validity dates at time of signing", True),
        ("Valid timestamp, but outside certificate validity dates", False),
        ("Invalid timestamp", False),
    ],
)
def test_require_timestamp_accepts_only_an_in_validity_timestamp(tmp_path, timestamp_line, accepted):
    tool = _fake_zxpsigncmd(tmp_path, timestamp_line)
    result = _ps(
        "try { Assert-CepPanelSignature -Path $env:OC_PATH -RequireTimestamp -ZxpSignCmd $env:OC_TOOL | Out-Null; "
        "'VERIFIED' } catch { 'ERROR: ' + $_.Exception.Message; exit 3 }",
        OC_PATH=str(tmp_path),
        OC_TOOL=str(tool),
    )
    if accepted:
        assert "VERIFIED" in result.stdout, result.stdout + result.stderr
    else:
        assert result.returncode == 3, result.stdout + result.stderr
        assert "no valid timestamp" in result.stdout


def test_signing_refuses_the_debug_port_file(tmp_path, certificate):
    panel = _make_panel(tmp_path)
    (panel / ".debug").write_text("<ExtensionList/>\n", encoding="utf-8")

    result = _sign(panel, tmp_path / "out.zxp", certificate)
    assert result.returncode == 3
    assert "must not be signed" in result.stdout
    assert not (tmp_path / "out.zxp").exists()


def test_signing_refuses_links(tmp_path, certificate):
    import _winapi

    panel = _make_panel(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    _winapi.CreateJunction(str(outside), str(panel / "client" / "linked"))

    result = _sign(panel, tmp_path / "out.zxp", certificate)
    assert result.returncode == 3
    assert "contains links" in result.stdout


def test_missing_certificate_stops_before_signing(tmp_path):
    panel = _make_panel(tmp_path)
    result = _sign(panel, tmp_path / "out.zxp", tmp_path / "absent.p12")
    assert result.returncode == 3
    assert "CEP signing certificate not found" in result.stdout
    assert not (panel / "META-INF").exists()
