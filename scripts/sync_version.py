#!/usr/bin/env python3
"""
OpenCut Version Sync Script

Reads __version__ from opencut/__init__.py and updates all other files
that contain version strings to match.

Usage:
    python scripts/sync_version.py              # Sync current version to all files
    python scripts/sync_version.py --set 1.3.0  # Set new version, then sync
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_PY = ROOT / "opencut" / "__init__.py"

# Each entry: (relative path, regex pattern, replacement template)
# The replacement template uses {v} for the version string and may also use
# computed release-series placeholders from version_tokens().
TARGETS = [
    (
        "pyproject.toml",
        r'^(version\s*=\s*")[^"]+(")',
        r'\g<1>{v}\g<2>',
    ),
    (
        "opencut/server.py",
        r'(OpenCut Backend Server v)[0-9]+\.[0-9]+\.[0-9]+',
        r'\g<1>{v}',
    ),
    (
        "extension/com.opencut.panel/client/index.html",
        r'(<span class="settings-value">)[0-9]+\.[0-9]+\.[0-9]+(</span>)',
        r'\g<1>{v}\g<2>',
    ),
    (
        "extension/com.opencut.panel/client/main.js",
        r'(OpenCut CEP Panel - Main Controller v)[0-9]+\.[0-9]+\.[0-9]+',
        r'\g<1>{v}',
    ),
    (
        "extension/com.opencut.panel/client/style.css",
        r'(OpenCut CEP Panel v)[0-9]+\.[0-9]+\.[0-9]+',
        r'\g<1>{v}',
    ),
    # CEP manifest (both bundle and extension version)
    (
        "extension/com.opencut.panel/CSXS/manifest.xml",
        r'(ExtensionBundleVersion=")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    (
        "extension/com.opencut.panel/CSXS/manifest.xml",
        r'(<Extension Id="com\.opencut\.panel\.main" Version=")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # Installer C# project
    (
        "installer/src/OpenCut.Installer/OpenCut.Installer.csproj",
        r'(<Version>)[0-9]+\.[0-9]+\.[0-9]+(</Version>)',
        r'\g<1>{v}\g<2>',
    ),
    (
        "installer/src/OpenCut.Installer/OpenCut.Installer.csproj",
        r'(<FileVersion>)[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(</FileVersion>)',
        r'\g<1>{v}.0\g<2>',
    ),
    # Installer AppConstants.cs
    (
        "installer/src/OpenCut.Installer/Models/AppConstants.cs",
        r'(Version\s*=\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # Installer app.manifest
    (
        "installer/src/OpenCut.Installer/Properties/app.manifest",
        r'(assemblyIdentity version=")[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}.0\g<2>',
    ),
    # Install.ps1 dev installer banner
    (
        "Install.ps1",
        r'(Installer v)[0-9]+\.[0-9]+\.[0-9]+',
        r'\g<1>{v}',
    ),
    # install.py dev installer
    (
        "install.py",
        r'(VERS\s*=\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # requirements.txt header comment
    (
        "requirements.txt",
        r'(# OpenCut v)[0-9]+\.[0-9]+\.[0-9]+',
        r'\g<1>{v}',
    ),
    # Inno Setup
    (
        "OpenCut.iss",
        r'(#define MyAppVersion\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # CEP package.json
    (
        "extension/com.opencut.panel/package.json",
        r'("version":\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # CEP package-lock.json root package metadata. Keep these patterns scoped
    # to the package root so dependency versions such as lightningcss 1.32.0
    # are not rewritten as OpenCut releases.
    (
        "extension/com.opencut.panel/package-lock.json",
        r'(^\s*"version":\s*")[0-9]+\.[0-9]+\.[0-9]+(",\s*$)',
        r'\g<1>{v}\g<2>',
    ),
    (
        "extension/com.opencut.panel/package-lock.json",
        r'("":\s*\{\s*"name":\s*"opencut-panel",\s*"version":\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # UXP manifest.json
    (
        "extension/com.opencut.uxp/manifest.json",
        r'("version":\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # UXP main.js VERSION constant
    (
        "extension/com.opencut.uxp/main.js",
        r'(const VERSION\s*=\s*")[0-9]+\.[0-9]+\.[0-9]+(")',
        r'\g<1>{v}\g<2>',
    ),
    # UXP index.html version display
    (
        "extension/com.opencut.uxp/index.html",
        r'(<span class="oc-version"[^>]*>v)[0-9]+\.[0-9]+\.[0-9]+(</span>)',
        r'\g<1>{v}\g<2>',
    ),
    (
        "extension/com.opencut.uxp/index.html",
        r'(<span id="uxpVersionDisplay">)[0-9]+\.[0-9]+\.[0-9]+( \(UXP\)</span>)',
        r'\g<1>{v}\g<2>',
    ),
    # Security support policy tracks minor series, not full patch releases.
    (
        "SECURITY.md",
        r'(latest minor\*\* \(`)[0-9]+\.[0-9]+\.x(`\) and the one immediately preceding it \(`)[0-9]+\.[0-9]+\.x(`\))',
        r'\g<1>{series}\g<2>{previous_series}\g<3>',
    ),
    (
        "SECURITY.md",
        r'(\| )[0-9]+\.[0-9]+\.x(\s+\|[^\n]*Active[^\n]*\|\s*)[^\n|]+(\s*\|)',
        r'\g<1>{series}\g<2>—                    \g<3>',
    ),
    (
        "SECURITY.md",
        r'(\| )[0-9]+\.[0-9]+\.x(\s+\|[^\n]*Previous[^\n]*\|\s*)\+90 days after [0-9]+\.[0-9]+(\s*\|)',
        r'\g<1>{previous_series}\g<2>+90 days after {latest_minor}\g<3>',
    ),
    (
        "SECURITY.md",
        r'(\| )[0-9]+\.[0-9]+\.x(\s+\|[^\n]*Critical only[^\n]*\|\s*)\+30 days after [0-9]+\.[0-9]+(\s*\|)',
        r'\g<1>{critical_series}\g<2>+30 days after {latest_minor}\g<3>',
    ),
    (
        "SECURITY.md",
        r'(\| )[≤<=>\s]*[0-9]+\.[0-9]+(\s+\|[^\n]*End of life[^\n]*\|\s*)n/a(\s*\|)',
        r'\g<1>≤ {eol_minor}\g<2>n/a\g<3>',
    ),
    (
        "opencut/core/c2pa_sidecar.py",
        r'(CLAIM_GENERATOR_DEFAULT\s*=\s*"OpenCut/)[0-9]+\.[0-9]+\.[0-9]+( \(sidecar; c2pa-spec 2\.3\)")',
        r'\g<1>{v}\g<2>',
    ),
]


def version_tokens(version: str) -> dict[str, str]:
    """Return replacement tokens derived from an X.Y.Z version string."""
    major_s, minor_s, patch_s = version.split(".")
    major = int(major_s)
    minor = int(minor_s)

    def minor_at(offset: int) -> int:
        return max(minor + offset, 0)

    return {
        "v": version,
        "major": str(major),
        "minor": str(minor),
        "patch": patch_s,
        "series": f"{major}.{minor}.x",
        "previous_series": f"{major}.{minor_at(-1)}.x",
        "critical_series": f"{major}.{minor_at(-2)}.x",
        "latest_minor": f"{major}.{minor}",
        "previous_minor": f"{major}.{minor_at(-1)}",
        "critical_minor": f"{major}.{minor_at(-2)}",
        "eol_minor": f"{major}.{minor_at(-3)}",
    }


def render_replacement(replacement: str, version: str) -> str:
    """Expand sync_version replacement placeholders for a target."""
    rendered = replacement
    for key, value in version_tokens(version).items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def read_version() -> str:
    """Read __version__ from opencut/__init__.py."""
    text = INIT_PY.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        print(f"ERROR: Could not find __version__ in {INIT_PY}")
        sys.exit(1)
    return m.group(1)


def set_version(new_ver: str) -> None:
    """Write __version__ into opencut/__init__.py."""
    text = INIT_PY.read_text(encoding="utf-8")
    updated = re.sub(
        r'(__version__\s*=\s*")[^"]+(")',
        rf'\g<1>{new_ver}\g<2>',
        text,
    )
    INIT_PY.write_text(updated, encoding="utf-8")
    print(f"  SET  {INIT_PY.relative_to(ROOT)}  ->  {new_ver}")


def check_file(rel_path: str, pattern: str, replacement: str, version: str) -> bool:
    """Check if a file's version matches. Returns True if in sync."""
    fpath = ROOT / rel_path
    if not fpath.exists():
        return True  # Missing files are OK (optional targets)

    text = fpath.read_text(encoding="utf-8")
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return True  # Pattern not found — nothing to check

    matched = m.group(0)
    repl = render_replacement(replacement, version)
    expected = re.sub(pattern, repl, matched, count=1, flags=re.MULTILINE)
    if expected == matched:
        return True

    print(f"  MISMATCH {rel_path}  (expected {version})")
    return False


def sync_file(rel_path: str, pattern: str, replacement: str, version: str) -> bool:
    """Update a single version occurrence in a file. Returns True if changed."""
    fpath = ROOT / rel_path
    if not fpath.exists():
        print(f"  SKIP {rel_path}  (file not found)")
        return False

    text = fpath.read_text(encoding="utf-8")
    repl = render_replacement(replacement, version)
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)

    if count == 0:
        print(f"  SKIP {rel_path}  (pattern not matched)")
        return False

    fpath.write_text(updated, encoding="utf-8")
    print(f"  SYNC {rel_path}  ->  {version}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync OpenCut version across all files")
    parser.add_argument("--set", dest="new_version", metavar="X.Y.Z",
                        help="Set a new version in __init__.py before syncing")
    parser.add_argument("--check", action="store_true",
                        help="Check all files are in sync (exit 1 if not). Does not modify files.")
    args = parser.parse_args()

    if args.new_version:
        if not re.match(r'^\d+\.\d+\.\d+$', args.new_version):
            print(f"ERROR: Invalid version format '{args.new_version}' (expected X.Y.Z)")
            sys.exit(1)
        set_version(args.new_version)

    version = read_version()

    if args.check:
        print(f"\nChecking version {version} across project files:\n")
        all_ok = True
        for rel_path, pattern, replacement in TARGETS:
            if not check_file(rel_path, pattern, replacement, version):
                all_ok = False
        if all_ok:
            print(f"\nAll files in sync at v{version}.")
        else:
            print("\nVersion mismatch detected! Run: python scripts/sync_version.py")
            sys.exit(1)
        return

    print(f"\nSyncing version {version} across project files:\n")

    changed = 0
    for rel_path, pattern, replacement in TARGETS:
        if sync_file(rel_path, pattern, replacement, version):
            changed += 1

    print(f"\nDone. {changed} file(s) updated to v{version}.")


if __name__ == "__main__":
    main()
