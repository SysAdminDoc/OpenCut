"""
OpenCut Proxy-to-Full-Res Swap on Export (60.2)

Check timeline clip paths against the proxy manifest, resolve to original
high-resolution files, verify originals exist, and report swap status.

Uses the proxy map JSON maintained by proxy_gen.py.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

PROXY_METADATA_FILE = ".opencut_proxy_map.json"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class SwapEntry:
    """Status of a single proxy-to-original swap."""
    proxy_path: str = ""
    original_path: str = ""
    status: str = "unknown"  # "swapped", "original_missing", "not_proxy", "already_original"
    original_exists: bool = False
    proxy_exists: bool = False
    original_size: int = 0
    proxy_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SwapResult:
    """Complete proxy swap check result."""
    entries: List[SwapEntry] = field(default_factory=list)
    total_clips: int = 0
    swapped: int = 0
    original_missing: int = 0
    already_original: int = 0
    not_in_manifest: int = 0
    all_originals_available: bool = False

    def to_dict(self) -> dict:
        return {
            "total_clips": self.total_clips,
            "swapped": self.swapped,
            "original_missing": self.original_missing,
            "already_original": self.already_original,
            "not_in_manifest": self.not_in_manifest,
            "all_originals_available": self.all_originals_available,
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# Proxy Map Loading
# ---------------------------------------------------------------------------
def load_proxy_map(proxy_dirs: List[str]) -> Dict[str, str]:
    """
    Load proxy-to-original mapping from one or more proxy directories.

    Args:
        proxy_dirs: List of directories that may contain proxy maps.

    Returns:
        Dict mapping absolute proxy path -> absolute original path.
    """
    combined = {}
    for d in proxy_dirs:
        map_path = os.path.join(d, PROXY_METADATA_FILE)
        if os.path.isfile(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                combined.update(mapping)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read proxy map %s: %s", map_path, e)
    return combined


def _is_proxy_path(path: str) -> bool:
    """Check if a path looks like a proxy file (contains _proxy suffix)."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.endswith("_proxy")


# ---------------------------------------------------------------------------
# Swap Check
# ---------------------------------------------------------------------------
def check_proxy_swap(
    clip_paths: List[str],
    proxy_dirs: Optional[List[str]] = None,
    proxy_map: Optional[Dict[str, str]] = None,
    on_progress: Optional[Callable] = None,
) -> SwapResult:
    """
    Check timeline clip paths against proxy manifest and resolve to originals.

    For each clip path:
    - If it's a known proxy: resolve to original, check original exists
    - If it's already an original: mark as such
    - If not in manifest: mark as unknown/not_proxy

    Args:
        clip_paths: List of clip paths from the timeline.
        proxy_dirs: Directories containing proxy map files.
        proxy_map: Pre-loaded proxy mapping (overrides proxy_dirs if given).
        on_progress: Callback(pct, msg).

    Returns:
        SwapResult with per-clip status.
    """
    if on_progress:
        on_progress(5, "Loading proxy manifest...")

    # Load or use provided map
    if proxy_map is None:
        if proxy_dirs:
            proxy_map = load_proxy_map(proxy_dirs)
        else:
            # Try to discover proxy dirs from clip paths
            discovered = set()
            for cp in clip_paths:
                parent = os.path.dirname(os.path.abspath(cp))
                discovered.add(parent)
                # Also check a "proxies" subdirectory nearby
                proxies_dir = os.path.join(os.path.dirname(parent), "proxies")
                if os.path.isdir(proxies_dir):
                    discovered.add(proxies_dir)
            proxy_map = load_proxy_map(list(discovered))

    # Build reverse map: original -> proxy
    original_set = set(proxy_map.values())

    result = SwapResult(total_clips=len(clip_paths))

    if on_progress:
        on_progress(15, f"Checking {len(clip_paths)} clips against manifest...")

    for i, cp in enumerate(clip_paths):
        abs_cp = os.path.abspath(cp)
        entry = SwapEntry(proxy_path=cp)

        if abs_cp in proxy_map:
            # This path is a known proxy
            original = proxy_map[abs_cp]
            entry.original_path = original
            entry.proxy_exists = os.path.isfile(abs_cp)
            entry.original_exists = os.path.isfile(original)

            if entry.original_exists:
                entry.status = "swapped"
                entry.original_size = os.path.getsize(original)
                result.swapped += 1
            else:
                entry.status = "original_missing"
                result.original_missing += 1

            if entry.proxy_exists:
                entry.proxy_size = os.path.getsize(abs_cp)

        elif abs_cp in original_set:
            # Already using the original file
            entry.status = "already_original"
            entry.original_path = abs_cp
            entry.original_exists = os.path.isfile(abs_cp)
            if entry.original_exists:
                entry.original_size = os.path.getsize(abs_cp)
            result.already_original += 1

        else:
            # Not in proxy manifest
            entry.status = "not_proxy"
            entry.original_path = abs_cp
            entry.original_exists = os.path.isfile(abs_cp)
            result.not_in_manifest += 1

        result.entries.append(entry)

        if on_progress and (i + 1) % 10 == 0:
            pct = 15 + int((i / len(clip_paths)) * 80)
            on_progress(pct, f"Checking clip {i + 1}/{len(clip_paths)}...")

    result.all_originals_available = result.original_missing == 0

    if on_progress:
        on_progress(100, f"Swap check complete: {result.swapped} swappable, "
                         f"{result.original_missing} missing originals")

    return result


def get_swap_paths(swap_result: SwapResult) -> Dict[str, str]:
    """
    Extract a proxy->original mapping from a SwapResult.

    Only includes entries that can be successfully swapped.

    Args:
        swap_result: Output from check_proxy_swap.

    Returns:
        Dict mapping proxy_path -> original_path for swappable entries.
    """
    return {
        e.proxy_path: e.original_path
        for e in swap_result.entries
        if e.status == "swapped" and e.original_exists
    }
