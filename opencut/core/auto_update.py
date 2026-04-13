"""
OpenCut Auto-Update Mechanism (Feature 7.3)

Checks GitHub Releases API for the latest version, compares with the
currently running version, and reports whether an update is available
along with the changelog.
"""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_REPO = "opencut/opencut"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
UPDATE_CHECK_TIMEOUT = 15  # seconds
USER_AGENT = "OpenCut-AutoUpdate/1.0"


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


@dataclass
class UpdateCheckResult:
    """Result of an update check."""
    current_version: str = ""
    latest_version: str = ""
    update_available: bool = False
    release_info: Optional[ReleaseInfo] = None
    changelog: str = ""
    error: Optional[str] = None

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
            data = json.loads(resp.read().decode("utf-8"))
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

    if on_progress:
        on_progress({"step": "checking", "message": "Contacting GitHub..."})

    try:
        latest = get_latest_release(include_prerelease=include_prerelease)
    except ConnectionError as exc:
        return UpdateCheckResult(
            current_version=current_version,
            error=str(exc),
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
    )


def trigger_update(method: str = "pip", on_progress: Optional[Callable] = None) -> UpdateResult:
    """Trigger an in-place update of OpenCut.

    Args:
        method: Update method.  Supported: ``"pip"``, ``"git"``, ``"download"``.
        on_progress: Optional progress callback.

    Returns:
        UpdateResult indicating success/failure.
    """
    method = method.lower().strip()
    if method not in ("pip", "git", "download"):
        return UpdateResult(
            success=False,
            method=method,
            message=f"Unsupported update method: {method!r}. Use 'pip', 'git', or 'download'.",
        )

    if on_progress:
        on_progress({"step": "updating", "method": method})

    if method == "pip":
        return _update_via_pip()
    elif method == "git":
        return _update_via_git()
    else:
        return _update_via_download()


# ---------------------------------------------------------------------------
# Update strategies
# ---------------------------------------------------------------------------
def _update_via_pip() -> UpdateResult:
    """Update OpenCut via ``pip install --upgrade``."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "opencut"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            # Re-read version after upgrade
            new_ver = _read_installed_version()
            return UpdateResult(
                success=True,
                method="pip",
                message="Successfully updated via pip.",
                new_version=new_ver,
            )
        return UpdateResult(
            success=False,
            method="pip",
            message=f"pip upgrade failed: {result.stderr[:500]}",
        )
    except subprocess.TimeoutExpired:
        return UpdateResult(success=False, method="pip", message="pip upgrade timed out (300s).")
    except Exception as exc:
        return UpdateResult(success=False, method="pip", message=str(exc))


def _update_via_git() -> UpdateResult:
    """Update OpenCut via ``git pull`` in the repo directory."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    git_dir = os.path.join(repo_root, ".git")
    if not os.path.isdir(git_dir):
        return UpdateResult(
            success=False,
            method="git",
            message="Not a git repository -- cannot update via git.",
        )
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
        if result.returncode == 0:
            new_ver = _read_installed_version()
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


def _update_via_download() -> UpdateResult:
    """Placeholder for download-based update (e.g. installer)."""
    try:
        latest = get_latest_release()
        if not latest.assets:
            return UpdateResult(
                success=False,
                method="download",
                message="No downloadable assets found in the latest release.",
            )
        download_url = latest.assets[0].get("download_url", "")
        return UpdateResult(
            success=True,
            method="download",
            message=f"Download available at: {download_url}",
            new_version=latest.version,
        )
    except Exception as exc:
        return UpdateResult(success=False, method="download", message=str(exc))


def _read_installed_version() -> str:
    """Re-read the installed version of the opencut package."""
    try:
        import importlib

        import opencut
        importlib.reload(opencut)
        return opencut.__version__
    except Exception:
        return "unknown"
