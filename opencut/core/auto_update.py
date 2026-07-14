"""
OpenCut Auto-Update Mechanism (Feature 7.3)

Checks GitHub Releases API for the latest version, compares with the
currently running version, and reports whether an update is available
along with the changelog.
"""

import hashlib
import importlib.metadata
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_REPO = "SysAdminDoc/OpenCut"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
PYPI_DISTRIBUTION = "opencut-ppro"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_DISTRIBUTION}"
UPDATE_CHECK_TIMEOUT = 15  # seconds
USER_AGENT = "OpenCut-AutoUpdate/1.0"

# Cap on GitHub Releases responses. A real release payload is ~10-50 KB; this
# cap is defence-in-depth in case the host header points elsewhere.
_MAX_GITHUB_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_PYPI_RESPONSE_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ReleaseInfo:
    """Information about a single GitHub release."""
    tag_name: str = ""
    version: str = ""
    name: str = ""
    body: str = ""
    published_at: str = ""
    html_url: str = ""
    prerelease: bool = False
    assets: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InstallOrigin:
    """Detected install shape and its only permitted update mechanism."""

    kind: str
    update_method: Optional[str] = None
    reason: str = ""
    repo_root: str = ""

    @property
    def update_supported(self) -> bool:
        return bool(self.update_method)

    def to_dict(self) -> dict:
        return asdict(self) | {"update_supported": self.update_supported}


@dataclass
class UpdateCheckResult:
    """Result of an update check."""
    current_version: str = ""
    latest_version: str = ""
    update_available: bool = False
    release_info: Optional[ReleaseInfo] = None
    changelog: str = ""
    error: Optional[str] = None
    install_origin: str = "unknown"
    update_supported: bool = False
    update_method: Optional[str] = None
    update_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.release_info:
            d["release_info"] = self.release_info.to_dict()
        return d


@dataclass
class UpdateResult:
    """Result of triggering an update."""
    success: bool = False
    method: str = ""
    message: str = ""
    new_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Version comparison helpers
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:[-.]?(alpha|beta|rc)\.?(\d*))?", re.IGNORECASE)


def _parse_version(version_str: str) -> tuple:
    """Parse a version string into a comparable tuple.

    Returns (major, minor, patch, prerelease_rank, prerelease_num).
    Stable releases get prerelease_rank=99 so they sort higher than alpha/beta/rc.
    """
    m = _VERSION_RE.match(version_str.strip())
    if not m:
        return (0, 0, 0, 0, 0)
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    pre_type = (m.group(4) or "").lower()
    pre_num = int(m.group(5)) if m.group(5) else 0
    pre_rank = {"alpha": 1, "beta": 2, "rc": 3}.get(pre_type, 99)
    return (major, minor, patch, pre_rank, pre_num)


def _version_is_newer(latest: str, current: str) -> bool:
    """Return True if *latest* is a strictly newer version than *current*."""
    return _parse_version(latest) > _parse_version(current)


def _is_valid_version(version: str) -> bool:
    """Return whether *version* is entirely represented by our parser."""
    return bool(_VERSION_RE.fullmatch((version or "").strip()))


# ---------------------------------------------------------------------------
# Install-origin detection
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_canonical_git_remote(remote: str) -> bool:
    """Accept only GitHub transports that identify the canonical repository."""
    value = (remote or "").strip().lower().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value in {
        "https://github.com/sysadmindoc/opencut",
        "ssh://git@github.com/sysadmindoc/opencut",
        "git@github.com:sysadmindoc/opencut",
    }


def _git_origin(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(repo_root),
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def detect_install_origin() -> InstallOrigin:
    """Detect the running install without allowing callers to choose a riskier path."""
    if getattr(sys, "frozen", False):
        return InstallOrigin(
            kind="packaged",
            reason=(
                "Packaged installs require a signed installer feed; automatic updates "
                "are disabled until that feed is available."
            ),
        )

    root = _repo_root()
    if (root / ".git").exists():
        try:
            remote = _git_origin(root)
        except (OSError, subprocess.SubprocessError) as exc:
            return InstallOrigin(
                kind="source",
                reason=f"Cannot verify the source checkout's origin remote: {exc}",
                repo_root=str(root),
            )
        if not _is_canonical_git_remote(remote):
            shown = remote or "missing"
            return InstallOrigin(
                kind="source",
                reason=(
                    f"Source checkout origin is {shown!r}, not the canonical "
                    f"https://github.com/{GITHUB_REPO}."
                ),
                repo_root=str(root),
            )
        return InstallOrigin(
            kind="source",
            update_method="git",
            reason="Canonical source checkout detected.",
            repo_root=str(root),
        )

    try:
        dist = importlib.metadata.distribution(PYPI_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return InstallOrigin(
            kind="unknown",
            reason=(
                f"Neither a canonical source checkout nor the {PYPI_DISTRIBUTION!r} "
                "distribution was detected."
            ),
        )

    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            direct = json.loads(direct_url)
        except (TypeError, json.JSONDecodeError):
            direct = {}
        if direct.get("dir_info", {}).get("editable") or direct.get("vcs_info"):
            return InstallOrigin(
                kind="source",
                reason=(
                    "An editable/VCS install outside a verified canonical checkout "
                    "cannot be updated safely."
                ),
            )

    return InstallOrigin(
        kind="wheel",
        update_method="pip",
        reason=f"Installed {PYPI_DISTRIBUTION!r} wheel detected.",
    )


# ---------------------------------------------------------------------------
# GitHub API interaction
# ---------------------------------------------------------------------------
def get_latest_release(include_prerelease: bool = False) -> ReleaseInfo:
    """Fetch the latest release from GitHub Releases API.

    Args:
        include_prerelease: If False (default), skip pre-release tags.

    Returns:
        ReleaseInfo with the latest release data.

    Raises:
        ConnectionError: If the API request fails.
    """
    url = GITHUB_API_URL + ("" if include_prerelease else "/latest")
    req = Request(url, headers={
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": USER_AGENT,
    })
    try:
        with urlopen(req, timeout=UPDATE_CHECK_TIMEOUT) as resp:
            raw = resp.read(_MAX_GITHUB_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_GITHUB_RESPONSE_BYTES:
                raise ConnectionError(
                    f"GitHub API response exceeds {_MAX_GITHUB_RESPONSE_BYTES} byte cap"
                )
            data = json.loads(raw.decode("utf-8"))
    except (URLError, OSError) as exc:
        raise ConnectionError(f"Failed to reach GitHub API: {exc}") from exc

    if isinstance(data, list):
        # /releases endpoint returns a list -- pick the first non-prerelease
        for release in data:
            if include_prerelease or not release.get("prerelease", False):
                data = release
                break
        else:
            if data:
                data = data[0]
            else:
                return ReleaseInfo()

    tag = data.get("tag_name", "")
    version = tag.lstrip("vV")
    assets = []
    for a in data.get("assets", []):
        assets.append({
            "name": a.get("name", ""),
            "size": a.get("size", 0),
            "download_url": a.get("browser_download_url", ""),
            "content_type": a.get("content_type", ""),
            "digest": a.get("digest", ""),
        })

    return ReleaseInfo(
        tag_name=tag,
        version=version,
        name=data.get("name", ""),
        body=data.get("body", ""),
        published_at=data.get("published_at", ""),
        html_url=data.get("html_url", ""),
        prerelease=data.get("prerelease", False),
        assets=assets,
    )


def parse_changelog(release_data: ReleaseInfo) -> str:
    """Extract a human-readable changelog from release body markdown.

    Strips HTML tags and normalizes whitespace for plain-text display.
    """
    body = release_data.body or ""
    # Strip HTML tags
    body = re.sub(r"<[^>]+>", "", body)
    # Normalize blank lines
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_for_updates(
    current_version: Optional[str] = None,
    include_prerelease: bool = False,
    on_progress: Optional[Callable] = None,
) -> UpdateCheckResult:
    """Check whether a newer version of OpenCut is available.

    Args:
        current_version: The version to compare against.  If None,
            reads ``opencut.__version__``.
        include_prerelease: Consider pre-release versions.
        on_progress: Optional progress callback.

    Returns:
        UpdateCheckResult with comparison details.
    """
    if current_version is None:
        from opencut import __version__
        current_version = __version__

    origin = detect_install_origin()

    if on_progress:
        on_progress({"step": "checking", "message": "Contacting GitHub..."})

    try:
        latest = get_latest_release(include_prerelease=include_prerelease)
    except ConnectionError as exc:
        return UpdateCheckResult(
            current_version=current_version,
            error=str(exc),
            install_origin=origin.kind,
            update_supported=origin.update_supported,
            update_method=origin.update_method,
            update_reason=origin.reason,
        )

    if on_progress:
        on_progress({"step": "comparing", "message": "Comparing versions..."})

    changelog = parse_changelog(latest)
    update_available = _version_is_newer(latest.version, current_version)

    return UpdateCheckResult(
        current_version=current_version,
        latest_version=latest.version,
        update_available=update_available,
        release_info=latest,
        changelog=changelog,
        install_origin=origin.kind,
        update_supported=origin.update_supported,
        update_method=origin.update_method,
        update_reason=origin.reason,
    )


def trigger_update(method: str = "auto", on_progress: Optional[Callable] = None) -> UpdateResult:
    """Trigger an in-place update of OpenCut.

    Args:
        method: ``"auto"`` (recommended), or the detected origin's exact method.
        on_progress: Optional progress callback.

    Returns:
        UpdateResult indicating success/failure.
    """
    if not isinstance(method, str):
        method = ""
    method = method.lower().strip()
    if method not in ("auto", "pip", "git", "download"):
        return UpdateResult(
            success=False,
            method=method,
            message=f"Unsupported update method: {method!r}. Use 'auto'.",
        )

    origin = detect_install_origin()
    if not origin.update_supported:
        return UpdateResult(success=False, method=method, message=origin.reason)

    selected_method = origin.update_method or ""
    if method == "download":
        return UpdateResult(
            success=False,
            method=method,
            message=(
                "Direct installer updates are disabled because no signed installer "
                "feed is configured. Use the verified mechanism for this install origin."
            ),
        )
    if method != "auto" and method != selected_method:
        return UpdateResult(
            success=False,
            method=method,
            message=(
                f"A {origin.kind} install can only update via {selected_method!r}; "
                f"refusing caller-selected {method!r}."
            ),
        )

    current_version = _read_installed_version()
    try:
        latest = get_latest_release()
    except Exception as exc:
        return UpdateResult(
            success=False,
            method=selected_method,
            message=f"Cannot resolve the canonical release: {exc}",
        )
    if not _is_valid_version(current_version) or not _is_valid_version(latest.version):
        return UpdateResult(
            success=False,
            method=selected_method,
            message=(
                f"Cannot safely compare installed version {current_version!r} with "
                f"release version {latest.version!r}."
            ),
        )
    if not _version_is_newer(latest.version, current_version):
        return UpdateResult(
            success=False,
            method=selected_method,
            message=(
                f"Refusing downgrade or reinstall: installed {current_version}, "
                f"canonical release {latest.version}."
            ),
            new_version=current_version,
        )

    if on_progress:
        on_progress({"step": "updating", "method": selected_method})

    if selected_method == "pip":
        return _update_via_pip(latest.version)
    return _update_via_git(Path(origin.repo_root), latest.version)


# ---------------------------------------------------------------------------
# Update strategies
# ---------------------------------------------------------------------------
def _fetch_pypi_release_hashes(version: str) -> Dict[str, str]:
    """Return published wheel SHA-256 hashes for one exact PyPI release."""
    url = f"{PYPI_JSON_URL}/{version}/json"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=UPDATE_CHECK_TIMEOUT) as response:
            raw = response.read(_MAX_PYPI_RESPONSE_BYTES + 1)
    except (URLError, OSError) as exc:
        raise ConnectionError(f"Failed to reach PyPI metadata: {exc}") from exc
    if len(raw) > _MAX_PYPI_RESPONSE_BYTES:
        raise ConnectionError("PyPI metadata response exceeds the byte cap")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConnectionError(f"Invalid PyPI metadata response: {exc}") from exc

    hashes = {}
    for artifact in data.get("urls", []):
        filename = artifact.get("filename", "")
        digest = artifact.get("digests", {}).get("sha256", "")
        if filename.endswith(".whl") and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            hashes[filename] = digest.lower()
    if not hashes:
        raise ConnectionError(
            f"PyPI release {PYPI_DISTRIBUTION}=={version} has no digest-pinned wheel."
        )
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_via_pip(expected_version: str) -> UpdateResult:
    """Download, verify, then install an exact canonical PyPI wheel."""
    try:
        trusted_hashes = _fetch_pypi_release_hashes(expected_version)
        with tempfile.TemporaryDirectory(prefix="opencut-update-") as temp_dir:
            download = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--dest",
                    temp_dir,
                    f"{PYPI_DISTRIBUTION}=={expected_version}",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if download.returncode != 0:
                return UpdateResult(
                    success=False,
                    method="pip",
                    message=f"pip download failed: {download.stderr[:500]}",
                )
            wheels = list(Path(temp_dir).glob("*.whl"))
            if len(wheels) != 1:
                return UpdateResult(
                    success=False,
                    method="pip",
                    message=f"Expected one wheel, found {len(wheels)}; nothing was installed.",
                )
            wheel = wheels[0]
            expected_digest = trusted_hashes.get(wheel.name)
            actual_digest = _sha256_file(wheel)
            if not expected_digest or actual_digest != expected_digest:
                return UpdateResult(
                    success=False,
                    method="pip",
                    message=(
                        f"SHA-256 verification failed for {wheel.name}; nothing was installed."
                    ),
                )
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    str(wheel),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if install.returncode != 0:
                return UpdateResult(
                    success=False,
                    method="pip",
                    message=f"pip install failed: {install.stderr[:500]}",
                )
        new_ver = _read_installed_version()
        if new_ver != expected_version:
            return UpdateResult(
                success=False,
                method="pip",
                message=(
                    f"Update command completed but installed version is {new_ver!r}, "
                    f"expected {expected_version!r}."
                ),
                new_version=new_ver,
            )
        return UpdateResult(
            success=True,
            method="pip",
            message="Installed the SHA-256-verified PyPI wheel.",
            new_version=new_ver,
        )
    except subprocess.TimeoutExpired:
        return UpdateResult(success=False, method="pip", message="pip update timed out (300s).")
    except Exception as exc:
        return UpdateResult(success=False, method="pip", message=str(exc))


def _update_via_git(repo_root: Path, expected_version: str) -> UpdateResult:
    """Fast-forward a clean canonical checkout."""
    if not (repo_root / ".git").exists():
        return UpdateResult(
            success=False,
            method="git",
            message="Not a git repository -- cannot update via git.",
        )
    try:
        remote = _git_origin(repo_root)
        if not _is_canonical_git_remote(remote):
            return UpdateResult(
                success=False,
                method="git",
                message="The checkout's origin is no longer the canonical OpenCut repository.",
            )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root),
        )
        if status.returncode != 0 or status.stdout.strip():
            return UpdateResult(
                success=False,
                method="git",
                message="Tracked checkout changes must be committed or reverted before updating.",
            )
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            new_ver = _read_installed_version()
            if new_ver != expected_version:
                return UpdateResult(
                    success=False,
                    method="git",
                    message=(
                        f"Canonical checkout updated but reports {new_ver!r}; "
                        f"expected release {expected_version!r}."
                    ),
                    new_version=new_ver,
                )
            return UpdateResult(
                success=True,
                method="git",
                message=f"Git pull successful. {result.stdout.strip()[:200]}",
                new_version=new_ver,
            )
        return UpdateResult(
            success=False,
            method="git",
            message=f"git pull failed: {result.stderr[:500]}",
        )
    except Exception as exc:
        return UpdateResult(success=False, method="git", message=str(exc))


def _read_installed_version() -> str:
    """Re-read the installed version of the opencut package."""
    try:
        return importlib.metadata.version(PYPI_DISTRIBUTION)
    except Exception:
        try:
            from opencut import __version__
            return __version__
        except Exception:
            return "unknown"
