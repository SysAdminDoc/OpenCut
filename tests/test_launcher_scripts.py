"""F211 — cross-platform launcher script smoke tests.

The repo ships five launcher entry points:

* ``OpenCut-Server.bat`` — Windows console launcher.
* ``OpenCut-Server.vbs`` / ``OpenCut-Launcher.vbs`` — Windows hidden
  launcher (no console window).
* ``OpenCut-Server.command`` — macOS double-click launcher (F261).
* ``OpenCut-Server.sh`` — Linux shell launcher (F261).

These tests pin the **shape** of each script so a future refactor
cannot regress a launcher into a state that:

* Fails to start `python -m opencut.server`.
* Crashes when the install path contains spaces (Program Files).
* Stops sourcing the bundled FFmpeg directory onto PATH.

Pinning the contents instead of actually running the launcher means
the tests are cross-platform (the Linux CI matrix leg can validate
the Windows `.bat` shape, and vice versa) and don't need any
Python install on the launch path.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BAT_LAUNCHER = REPO_ROOT / "OpenCut-Server.bat"
VBS_LAUNCHER = REPO_ROOT / "OpenCut-Server.vbs"
VBS_HELPER = REPO_ROOT / "OpenCut-Launcher.vbs"
COMMAND_LAUNCHER = REPO_ROOT / "OpenCut-Server.command"
SH_LAUNCHER = REPO_ROOT / "OpenCut-Server.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_windows_bat_launcher_exists():
    assert BAT_LAUNCHER.is_file(), f"Windows BAT launcher missing at {BAT_LAUNCHER}"


def test_windows_vbs_launchers_exist():
    assert VBS_LAUNCHER.is_file(), f"Windows VBS launcher missing at {VBS_LAUNCHER}"
    assert VBS_HELPER.is_file(), f"Windows VBS helper missing at {VBS_HELPER}"


def test_macos_command_launcher_exists():
    """F261 macOS launcher must ship."""
    assert COMMAND_LAUNCHER.is_file(), (
        f"macOS .command launcher missing at {COMMAND_LAUNCHER}"
    )


def test_linux_sh_launcher_exists():
    """F261 Linux launcher must ship."""
    assert SH_LAUNCHER.is_file(), (
        f"Linux .sh launcher missing at {SH_LAUNCHER}"
    )


# ---------------------------------------------------------------------------
# Shebang / encoding shape
# ---------------------------------------------------------------------------


def test_macos_launcher_starts_with_shebang():
    text = _read(COMMAND_LAUNCHER)
    first_line = (text.splitlines() or [""])[0]
    assert first_line.startswith("#!"), (
        f".command launcher must start with a shebang; got {first_line!r}"
    )
    assert "sh" in first_line or "bash" in first_line, (
        f".command launcher must invoke sh or bash; got {first_line!r}"
    )


def test_linux_launcher_starts_with_shebang():
    text = _read(SH_LAUNCHER)
    first_line = (text.splitlines() or [""])[0]
    assert first_line.startswith("#!"), (
        f".sh launcher must start with a shebang; got {first_line!r}"
    )
    assert "sh" in first_line or "bash" in first_line


def test_posix_launchers_use_lf_line_endings():
    """F261 launchers must keep LF endings — CRLF breaks the shebang
    parser on macOS/Linux and produces an opaque 'env: bad interpreter'
    failure."""
    for path in (COMMAND_LAUNCHER, SH_LAUNCHER):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, (
            f"{path.name} must use LF line endings (not CRLF). "
            f"Check .gitattributes for the eol=lf rule."
        )


# ---------------------------------------------------------------------------
# Behavioural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("launcher", [
    BAT_LAUNCHER,
    COMMAND_LAUNCHER,
    SH_LAUNCHER,
])
def test_launcher_starts_opencut_server(launcher):
    """Every launcher must end up invoking `python -m opencut.server`
    (the canonical backend entry point). ``.command`` is allowed to
    delegate to ``.sh`` via ``exec`` — the wrapper is a one-liner so
    macOS users get a double-clickable entry point that mirrors the
    Linux path."""
    if not launcher.is_file():
        pytest.skip(f"launcher {launcher.name} not present on this platform")
    text = _read(launcher)
    delegates = re.search(r"\bexec\b.*OpenCut-Server\.sh\b", text) is not None
    if delegates:
        # Confirm the delegate target exists.
        assert SH_LAUNCHER.is_file(), (
            f"{launcher.name} delegates to OpenCut-Server.sh which is missing"
        )
        return
    # Allow either `python -m opencut.server` or `python -m opencut`
    # (the package `__main__` proxies through). Reject all other entry
    # points so a future refactor cannot silently switch us to a
    # different module without updating this test.
    assert re.search(r"\bopencut(?:\.server)?\b", text), (
        f"{launcher.name} must invoke opencut(.server) module entry point"
    )
    assert "-m" in text, (
        f"{launcher.name} must use `python -m` so the package import path "
        f"resolves correctly when the launcher is double-clicked from an "
        f"arbitrary CWD"
    )


@pytest.mark.parametrize("launcher", [
    BAT_LAUNCHER,
    COMMAND_LAUNCHER,
    SH_LAUNCHER,
])
def test_launcher_handles_paths_with_spaces(launcher):
    """Launchers must keep working when installed under `C:\\Program
    Files\\OpenCut` (Windows) or `/Applications/OpenCut.app` (macOS)
    — both contain a space that breaks naive un-quoted invocation."""
    if not launcher.is_file():
        pytest.skip(f"launcher {launcher.name} not present on this platform")
    text = _read(launcher)
    if launcher.suffix == ".bat":
        # cmd.exe requires double-quoted paths when they contain spaces.
        # The bat launcher must show evidence of that handling (either
        # explicit quoting around %~dp0/%CD% style refs, OR `set "VAR=…"`
        # style assignment that lets later expansions stay quoted, OR
        # the `pushd "%~dp0"` trick that lets the rest of the script
        # use bare paths).
        has_quoting = (
            'pushd "' in text
            or 'cd /d "' in text
            or '"%~dp0' in text
            or 'set "OPENCUT_HOME=%~dp0' in text
            or 'set "OPENCUT_HOME=%~dp0"' in text
            or re.search(r'set\s+"[A-Z_]+=%~dp0', text) is not None
        )
        assert has_quoting, (
            f"{launcher.name} must quote path expansions to handle "
            f"spaces in the install root (Program Files)"
        )
    else:
        # POSIX shells need quotes around variable expansions for the
        # same reason ("$SCRIPT_DIR", "$HOME/...").
        # Allow either explicit quotes around $VAR or a `cd "$VAR"`
        # / `set -e; cd ...` defensive opener.
        has_quoting = (
            '"$' in text
            or 'cd "' in text
            or 'cd "$(' in text
        )
        assert has_quoting, (
            f"{launcher.name} must double-quote shell variable expansions "
            f"to survive spaces in the install path"
        )


def _launcher_text_resolved(path: Path) -> str:
    """Return the launcher's effective text — for the .command wrapper,
    follow its `exec OpenCut-Server.sh` delegation and return the .sh
    content. The wrapper itself is intentionally a 4-liner; the env
    setup lives in the script it exec's."""
    text = _read(path)
    if re.search(r"\bexec\b.*OpenCut-Server\.sh\b", text):
        return _read(SH_LAUNCHER)
    return text


def test_posix_launchers_export_opencut_home():
    """F261 launchers must export OPENCUT_HOME so the backend can find
    bundled FFmpeg + model dirs regardless of CWD. The .command wrapper
    delegates to .sh, so we follow the exec chain to find the env setup.
    """
    for path in (COMMAND_LAUNCHER, SH_LAUNCHER):
        text = _launcher_text_resolved(path)
        assert "OPENCUT_HOME" in text, (
            f"{path.name} (or its delegate) must export OPENCUT_HOME (F261)"
        )


def test_posix_launchers_propagate_bundled_ffmpeg():
    """F261 launchers must put the bundled ffmpeg/ dir on PATH if it
    exists locally. This is the same precedence opencut.server uses
    when probing for the bundled binary."""
    for path in (COMMAND_LAUNCHER, SH_LAUNCHER):
        text = _launcher_text_resolved(path)
        path_munged = (
            "PATH=" in text and "ffmpeg" in text.lower()
        ) or (
            "OPENCUT_FFMPEG" in text or "FFMPEG_BINARY" in text
        )
        assert path_munged, (
            f"{path.name} (or its delegate) must expose the bundled "
            f"ffmpeg/ dir to the server (either via PATH or OPENCUT_FFMPEG)"
        )


@pytest.mark.parametrize("launcher", [BAT_LAUNCHER, VBS_LAUNCHER, SH_LAUNCHER])
def test_source_launchers_reject_unsupported_python_before_import(launcher):
    """Every source launcher must enforce the canonical Python floor."""
    text = _read(launcher).lower()
    assert "3.11" in text
    assert "detected" in text
    assert "require" in text
    assert "python.org/downloads" in text


def test_packaged_hidden_launcher_does_not_depend_on_host_python():
    """The installed launcher starts the frozen server, so no floor check applies."""
    text = _read(VBS_HELPER).lower()
    assert "opencut-server.exe" in text
    assert "python" not in text


# ---------------------------------------------------------------------------
# Filesystem mode
# ---------------------------------------------------------------------------


def test_posix_launchers_are_marked_executable_in_git():
    """`.command` and `.sh` files must be committed with the executable
    bit set, otherwise users have to `chmod +x` after cloning.

    The Windows VM running this test exposes Z: via VMware Shared
    Folders, which doesn't preserve POSIX mode bits — so this test
    falls back to `git ls-files --stage` when the on-disk bit is
    unreadable. The git index is the source of truth either way.
    """
    import subprocess

    for path in (COMMAND_LAUNCHER, SH_LAUNCHER):
        st = path.stat()
        on_disk_exec = bool(st.st_mode & stat.S_IXUSR)
        if on_disk_exec:
            continue
        # Fallback — check the git index for mode 100755.
        try:
            result = subprocess.run(
                ["git", "ls-files", "--stage", path.name],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, OSError):
            pytest.skip(f"git not available to verify {path.name} executable bit")
            return
        if result.returncode != 0:
            pytest.skip(f"git ls-files failed for {path.name}: {result.stderr}")
            return
        # ls-files --stage output: "<mode> <hash> <stage>\t<path>"
        # We need the first column to be 100755 (executable).
        first_field = (result.stdout.split("\t", 1)[0] or "").split()
        mode = first_field[0] if first_field else ""
        assert mode == "100755", (
            f"{path.name} must be marked executable in git "
            f"(mode 100755); current index mode is {mode!r}. "
            f"Run `git update-index --chmod=+x {path.name}` to fix."
        )
