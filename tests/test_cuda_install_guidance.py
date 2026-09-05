"""The documented GPU install lane must cover the hardware OpenCut claims.

README.md, requirements.txt, install.py and upscale_flashvsr.py each restated
the CUDA wheel index, and all four said cu121 -- an index with no sm_120
kernels. Every RTX 50-series user who followed the documentation got the
failure reported as issue #7. One source now, and this keeps it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from opencut.gpu import (
    CUDA_INDEX_MAX_CAPABILITY,
    CUDA_WHEEL_INDEX,
    CUDA_WHEEL_INDEX_URL,
    NEWEST_CONSUMER_CAPABILITY,
    supported_capability_range,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every file that tells a user how to install a CUDA-capable torch.
DOCUMENTED_INSTALL_SITES = (
    "README.md",
    "requirements.txt",
    "install.py",
    "opencut/core/upscale_flashvsr.py",
)

_INDEX_RE = re.compile(r"download\.pytorch\.org/whl/(cu\d+)")


def test_documented_index_covers_the_newest_supported_hardware():
    """The check the roadmap item asks for: index versus declared range."""
    covered = CUDA_INDEX_MAX_CAPABILITY[CUDA_WHEEL_INDEX]
    assert covered >= NEWEST_CONSUMER_CAPABILITY, (
        f"the documented index {CUDA_WHEEL_INDEX} tops out at compute capability "
        f"{covered[0]}.{covered[1]}, but OpenCut claims support through "
        f"{NEWEST_CONSUMER_CAPABILITY[0]}.{NEWEST_CONSUMER_CAPABILITY[1]}. "
        "Either raise the index or lower the claim."
    )


def test_supported_range_is_ordered():
    oldest, newest = supported_capability_range()
    assert oldest < newest


@pytest.mark.parametrize("relative", DOCUMENTED_INSTALL_SITES)
def test_every_documented_site_names_the_canonical_index(relative):
    path = REPO_ROOT / relative
    assert path.is_file(), f"{relative} moved; update DOCUMENTED_INSTALL_SITES"
    text = path.read_text(encoding="utf-8", errors="replace")
    found = set(_INDEX_RE.findall(text))
    assert found, f"{relative} no longer documents a CUDA wheel index"
    assert found == {CUDA_WHEEL_INDEX}, (
        f"{relative} names {sorted(found)} but opencut.gpu.CUDA_WHEEL_INDEX is "
        f"{CUDA_WHEEL_INDEX}. Keep the documented lane in one place."
    )


def test_no_stale_index_survives_anywhere_in_the_shipped_docs():
    """Catch a fifth site appearing that DOCUMENTED_INSTALL_SITES does not list."""
    stale = {name for name in CUDA_INDEX_MAX_CAPABILITY if name != CUDA_WHEEL_INDEX}
    skip_dirs = {".git", "build", "dist", "node_modules", "__pycache__", "release_licenses"}
    # *.egg-info is gitignored build output whose PKG-INFO is generated from
    # README.md, which this module already gates directly. A stale local copy
    # would fail this for a file nobody edits.
    # Files that legitimately discuss the old index: history, planning, and the
    # tests that assert the fix.
    skip_files = {"CHANGELOG.md", "ROADMAP.md", "RESEARCH.md", "Roadmap_Blocked.md", "CLAUDE.md"}

    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        # PKG-INFO has no suffix and still ships in the sdist and wheel, where
        # it renders on PyPI. A suffix-only filter could not see it.
        if path.suffix not in {".md", ".txt", ".py", ".ps1", ".iss"} and path.name != "PKG-INFO":
            continue
        # Only directory components, so a *file* named "build" is still checked.
        directories = path.relative_to(REPO_ROOT).parts[:-1]
        if set(directories) & skip_dirs:
            continue
        if any(name.endswith(".egg-info") for name in directories):
            continue
        if path.name in skip_files or path.name.startswith("requirements-release"):
            continue
        if path.name.startswith("test_cuda_install_guidance") or path.name.startswith("test_gpu_"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in _INDEX_RE.findall(text):
            if name in stale and path.name != "gpu.py":
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")

    assert not offenders, (
        "these files still point users at a CUDA index that cannot run current "
        "hardware:\n" + "\n".join(f"  {item}" for item in offenders)
    )


def test_install_command_is_built_from_the_canonical_url():
    from opencut.gpu import TORCH_GPU_INSTALL_COMMAND

    assert CUDA_WHEEL_INDEX_URL in TORCH_GPU_INSTALL_COMMAND
    assert "torch" in TORCH_GPU_INSTALL_COMMAND
