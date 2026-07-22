"""F200 — Windows installer policy invariants.

The repo ships two parallel Windows installer paths (the recommended
WPF / .NET 10 path under `installer/` and the deprecated-but-supported
Inno Setup fallback at `OpenCut.iss`). Both must agree on the user-
visible install-tree state. These tests pin the §3 invariants from
`docs/INSTALLER_POLICY.md`:

* Bundled FFmpeg version string (F207).
* Display name / Program Files default root.
* CEP extension folder layout.

Future divergence either trips the test (so the lockstep gets
re-established) or requires updating both the policy doc and these
asserts in the same PR.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INNO_SCRIPT = REPO_ROOT / "OpenCut.iss"
WPF_CONSTANTS = (
    REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Models" / "AppConstants.cs"
)
POLICY_DOC = REPO_ROOT / "docs" / "INSTALLER_POLICY.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Policy doc presence
# ---------------------------------------------------------------------------


def test_policy_doc_exists_and_mentions_both_installers():
    assert POLICY_DOC.is_file(), "F200 — docs/INSTALLER_POLICY.md must exist"
    text = _read(POLICY_DOC)
    assert "WPF" in text, "policy doc must reference the WPF installer path"
    assert "Inno" in text, "policy doc must reference the Inno Setup fallback"
    assert "recommended" in text.lower()
    # The policy doc must name F201 / F203 / F207 / F213 so the F-number chain
    # stays discoverable.
    for fid in ("F201", "F203", "F207", "F213"):
        assert fid in text, f"policy doc must reference {fid}"


def test_policy_doc_mentions_lockstep_invariants():
    text = _read(POLICY_DOC)
    for invariant in (
        "FFmpeg",
        "display name",
        "Program Files",
        "Adobe\\CEP",
    ):
        assert invariant.lower() in text.lower(), (
            f"policy doc must list lockstep invariant {invariant!r}"
        )


# ---------------------------------------------------------------------------
# Lockstep invariants (the actual policy contract)
# ---------------------------------------------------------------------------


def _wpf_constant(name: str) -> str:
    """Read a `public const string <name> = "...";` from AppConstants.cs."""
    assert WPF_CONSTANTS.is_file(), (
        f"WPF installer constants missing at {WPF_CONSTANTS}"
    )
    text = _read(WPF_CONSTANTS)
    # Handle both regular strings and verbatim @"..." strings.
    pattern = rf'public\s+const\s+string\s+{re.escape(name)}\s*=\s*(@?"([^"]*)")'
    match = re.search(pattern, text)
    assert match, f"WPF AppConstants.cs missing const {name!r}"
    return match.group(2)


def _inno_define(name: str) -> str:
    """Read a `#define name "value"` directive from OpenCut.iss."""
    assert INNO_SCRIPT.is_file(), f"OpenCut.iss missing at {INNO_SCRIPT}"
    text = _read(INNO_SCRIPT)
    pattern = rf'#define\s+{re.escape(name)}\s+"([^"]*)"'
    match = re.search(pattern, text)
    assert match, f"OpenCut.iss missing #define {name!r}"
    return match.group(1)


def test_bundled_ffmpeg_version_matches_across_installers():
    """F207 — both installers must report the same bundled FFmpeg version."""
    wpf_version = _wpf_constant("BundledFfmpegVersion")
    inno_version = _inno_define("BundledFfmpegVersion")
    assert wpf_version, "WPF BundledFfmpegVersion is empty"
    assert wpf_version == inno_version, (
        f"F200 §3.1 lockstep — WPF reports BundledFfmpegVersion={wpf_version!r} "
        f"but OpenCut.iss reports {inno_version!r}. Mirror the change."
    )


def test_inno_script_writes_installer_json_with_ffmpeg_version():
    """F207 — Inno installer must write ~/.opencut/installer.json with the
    bundled FFmpeg version so the running server can surface it."""
    text = _read(INNO_SCRIPT)
    assert "installer.json" in text, (
        "OpenCut.iss must write ~/.opencut/installer.json (F207)"
    )
    assert "bundled_ffmpeg_version" in text, (
        "OpenCut.iss must record bundled_ffmpeg_version in installer.json"
    )
    assert "GetParameterValue(" in text
    assert "'/USERDATADIR='" in text
    assert "DeleteFile(ConfigDir + '\\installer.json')" in text


def test_inno_production_remains_elevated_and_dev_payload_is_excluded():
    text = _read(INNO_SCRIPT)
    assert '#define InstallerPrivilegesRequired "admin"' in text
    assert "PrivilegesRequired={#InstallerPrivilegesRequired}" in text
    for excluded in ("node_modules\\*", "tests\\*", "test-results\\*", "playwright-report\\*"):
        assert excluded in text


def test_install_root_is_program_files_opencut():
    """F200 §3.3 — install root default must be C:\\Program Files\\OpenCut
    in both installers."""
    wpf_text = _read(WPF_CONSTANTS)
    inno_text = _read(INNO_SCRIPT)
    assert r"Program Files\OpenCut" in wpf_text, (
        "WPF AppConstants.cs must default to Program Files\\OpenCut"
    )
    # Inno Setup uses {pf}\OpenCut, {commonpf64}\OpenCut, {autopf}\OpenCut,
    # or the macro form {autopf}\{#MyAppName} when AppName is defined.
    has_pf = (
        any(token in inno_text for token in (
            r"{pf}\OpenCut",
            r"{commonpf}\OpenCut",
            r"{commonpf64}\OpenCut",
            r"{pf64}\OpenCut",
            r"{autopf}\OpenCut",
            r"Program Files\OpenCut",
        ))
        or (
            re.search(r"#define\s+MyAppName\s+\"OpenCut\"", inno_text)
            and re.search(r"DefaultDirName\s*=\s*\{(?:auto)?pf(?:64)?\}\\\{#MyAppName\}", inno_text)
        )
    )
    assert has_pf, (
        "OpenCut.iss must default to a Program Files\\OpenCut variant"
    )


def test_inno_script_lays_ffmpeg_under_app_ffmpeg():
    """F200 §3.4 prep — bundled FFmpeg lands at {app}\\ffmpeg."""
    text = _read(INNO_SCRIPT)
    assert r"{app}\ffmpeg" in text, (
        "OpenCut.iss must install FFmpeg under {app}\\ffmpeg"
    )


# ---------------------------------------------------------------------------
# Retirement-plan gates
# ---------------------------------------------------------------------------


def test_policy_doc_names_retirement_trigger_not_a_calendar_date():
    """F200 must not commit to a calendar date for Inno retirement —
    the gate is the WPF coverage milestone (F212 close)."""
    text = _read(POLICY_DOC)
    assert "F212" in text, "F200 retirement gate must reference F212"
    assert "retirement" in text.lower() or "retire" in text.lower()
    # Defensive: the policy must explicitly state the retirement is
    # gated on a milestone, not a calendar date.
    assert "not a calendar date" in text or "milestone" in text.lower(), (
        "F200 doc must clarify Inno retirement is milestone-gated"
    )
