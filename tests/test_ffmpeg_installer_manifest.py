from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_VERSION = "8.1.2-essentials_build-www.gyan.dev"


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
    assert "installer.json" in inno
    assert "WriteInstallerManifest();" in inno


def test_pinned_installer_version_matches_provenance_module():
    """The installer pins and the Python provenance floor must agree."""
    from opencut.core import ffmpeg_provenance as fp

    assert fp.PINNED_INSTALLER_VERSION == FFMPEG_VERSION
    # Release floor is 8.1.1 (the version the installers bundle).
    assert fp.RELEASE_FLOOR == (8, 1, 1)
