import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_VERSION = "2026-08-03-git-01a25f74cc-full_build-www.gyan.dev"


def test_wpf_installer_constants_pin_bundled_ffmpeg_version():
    constants = (
        REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "AppConstants.cs"
    ).read_text(encoding="utf-8")
    config = (
        REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "InstallConfig.cs"
    ).read_text(encoding="utf-8")

    assert f'BundledFfmpegVersion = "{FFMPEG_VERSION}"' in constants
    assert f'BundledFfprobeVersion = "{FFMPEG_VERSION}"' in constants
    assert "BundledFfmpegSecurityFloor" in constants
    assert "BundledFfmpegSecurityCve" in constants
    assert "BundledFfmpegSecurityFixCommits" in constants
    assert "BundledFfmpegSourceCommit" in constants
    assert "BundledFfmpegSourceSha256" in constants
    assert "BundledFfmpegPackageSha256" in constants
    assert "BundledFfmpegPackageUrl" in constants
    assert 'InstallerManifestFile = "installer.json"' in constants
    assert "InstallerManifestPath" in config
    assert "SpecialFolder.UserProfile" in config


def test_wpf_installer_writes_ffmpeg_manifest():
    engine = (
        REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Services" / "InstallEngine.cs"
    ).read_text(encoding="utf-8")

    assert "WriteInstallerManifest" in engine
    assert "bundled_ffmpeg_version" in engine
    assert "bundled_ffprobe_version" in engine
    assert "bundled_ffmpeg_security_floor" in engine
    assert "bundled_ffmpeg_security_cve" in engine
    assert "bundled_ffmpeg_security_fix_commits" in engine
    assert "bundled_ffmpeg_source_commit" in engine
    assert "bundled_ffmpeg_source_sha256" in engine
    assert "bundled_ffmpeg_package_sha256" in engine
    assert "bundled_ffmpeg_package_url" in engine
    assert "VerifyPayload(ffmpegSrc)" in engine
    assert "installer_kind" in engine
    assert "JsonSerializer.Serialize" in engine


def test_inno_installer_writes_ffmpeg_manifest():
    inno = (REPO_ROOT / "OpenCut.iss").read_text(encoding="utf-8")

    assert f'#define BundledFfmpegVersion "{FFMPEG_VERSION}"' in inno
    assert f'#define BundledFfprobeVersion "{FFMPEG_VERSION}"' in inno
    assert "procedure WriteInstallerManifest" in inno
    assert "bundled_ffmpeg_version" in inno
    assert "bundled_ffprobe_version" in inno
    assert "bundled_ffmpeg_security_floor" in inno
    assert "bundled_ffmpeg_security_cve" in inno
    assert "bundled_ffmpeg_security_fix_commits" in inno
    assert "bundled_ffmpeg_source_commit" in inno
    assert "bundled_ffmpeg_source_sha256" in inno
    assert "bundled_ffmpeg_package_sha256" in inno
    assert "bundled_ffmpeg_package_url" in inno
    assert "VerifyInstalledFfmpeg();" in inno
    assert "SW_HIDE" in inno
    assert "installer.json" in inno
    assert "WriteInstallerManifest();" in inno


def test_pinned_installer_version_matches_provenance_module():
    """The installer pins and the Python provenance floor must agree."""
    from opencut.core import ffmpeg_provenance as fp

    assert fp.PINNED_INSTALLER_VERSION == FFMPEG_VERSION
    assert fp.RELEASE_FLOOR == (8, 1, 3)
    assert "CVE-2026-8461" in fp.JUNE_2026_CVES
    assert set(("CVE-2026-64832", "CVE-2026-64833", "CVE-2026-64835", "CVE-2026-66041")) <= set(fp.SECURITY_CVES)
    # A literal count here failed on every legitimate grading pass. What the
    # pins actually need is that the list is exactly its two parts with no
    # duplicates, and that each entry is a real full-length commit hash.
    assert len(fp.SECURITY_FIX_COMMITS) == (
        len(fp.MAGICYUV_FIX_COMMITS) + len(fp.JULY_2026_FIX_COMMITS)
    )
    assert len(set(fp.SECURITY_FIX_COMMITS)) == len(fp.SECURITY_FIX_COMMITS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", c) for c in fp.SECURITY_FIX_COMMITS)
