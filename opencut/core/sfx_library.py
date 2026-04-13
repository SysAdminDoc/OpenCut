"""
OpenCut Free SFX Library Search Module (Feature 19.2)

Search Freesound.org API (with local library fallback), download,
cache, and import SFX files.

Functions:
    search_sfx       - Search for sound effects by query
    download_sfx     - Download an SFX by ID to local cache
    list_cached_sfx  - List all locally cached SFX files
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Cache & Config
# ---------------------------------------------------------------------------
SFX_CACHE_DIR = os.path.join(OPENCUT_DIR, "sfx_cache")
SFX_INDEX_FILE = os.path.join(SFX_CACHE_DIR, "index.json")
FREESOUND_API_BASE = "https://freesound.org/apiv2"

# Built-in SFX library (fallback when Freesound API is unavailable)
BUILTIN_SFX = [
    {"id": "builtin_001", "name": "Click", "tags": ["click", "ui", "button"], "duration": 0.1, "category": "ui"},
    {"id": "builtin_002", "name": "Whoosh", "tags": ["whoosh", "transition", "swipe"], "duration": 0.5, "category": "transition"},
    {"id": "builtin_003", "name": "Impact Hit", "tags": ["impact", "hit", "punch"], "duration": 0.3, "category": "impact"},
    {"id": "builtin_004", "name": "Bell Chime", "tags": ["bell", "chime", "notification"], "duration": 1.0, "category": "notification"},
    {"id": "builtin_005", "name": "Pop", "tags": ["pop", "bubble", "ui"], "duration": 0.15, "category": "ui"},
    {"id": "builtin_006", "name": "Glass Break", "tags": ["glass", "break", "shatter"], "duration": 0.8, "category": "impact"},
    {"id": "builtin_007", "name": "Door Close", "tags": ["door", "close", "shut"], "duration": 0.4, "category": "foley"},
    {"id": "builtin_008", "name": "Footstep Wood", "tags": ["footstep", "walk", "wood"], "duration": 0.2, "category": "foley"},
    {"id": "builtin_009", "name": "Water Splash", "tags": ["water", "splash", "liquid"], "duration": 0.6, "category": "nature"},
    {"id": "builtin_010", "name": "Thunder Roll", "tags": ["thunder", "storm", "weather"], "duration": 3.0, "category": "nature"},
    {"id": "builtin_011", "name": "Typing Keys", "tags": ["typing", "keyboard", "keys"], "duration": 0.05, "category": "foley"},
    {"id": "builtin_012", "name": "Wind Gust", "tags": ["wind", "gust", "breeze"], "duration": 2.0, "category": "nature"},
    {"id": "builtin_013", "name": "Explosion Boom", "tags": ["explosion", "boom", "blast"], "duration": 1.5, "category": "impact"},
    {"id": "builtin_014", "name": "Cash Register", "tags": ["cash", "register", "money", "cha-ching"], "duration": 0.5, "category": "foley"},
    {"id": "builtin_015", "name": "Camera Shutter", "tags": ["camera", "shutter", "photo"], "duration": 0.2, "category": "foley"},
    {"id": "builtin_016", "name": "Crowd Cheer", "tags": ["crowd", "cheer", "applause"], "duration": 3.0, "category": "ambient"},
    {"id": "builtin_017", "name": "Siren Alert", "tags": ["siren", "alert", "alarm"], "duration": 2.0, "category": "notification"},
    {"id": "builtin_018", "name": "Drip", "tags": ["drip", "water", "drop"], "duration": 0.3, "category": "nature"},
    {"id": "builtin_019", "name": "Swoosh Riser", "tags": ["riser", "swoosh", "transition"], "duration": 1.5, "category": "transition"},
    {"id": "builtin_020", "name": "Vinyl Scratch", "tags": ["vinyl", "scratch", "record"], "duration": 0.8, "category": "music"},
]


@dataclass
class SFXSearchResult:
    """A single SFX search result."""
    id: str
    name: str
    tags: List[str] = field(default_factory=list)
    duration: float = 0.0
    category: str = ""
    source: str = "builtin"  # "builtin" or "freesound"
    preview_url: str = ""
    license: str = ""


@dataclass
class SFXDownloadResult:
    """Result of downloading an SFX file."""
    id: str
    name: str
    local_path: str
    duration: float
    source: str


def _ensure_cache_dir():
    """Create the SFX cache directory if it doesn't exist."""
    os.makedirs(SFX_CACHE_DIR, exist_ok=True)


def _load_cache_index() -> dict:
    """Load the SFX cache index from disk."""
    if not os.path.isfile(SFX_INDEX_FILE):
        return {}
    try:
        with open(SFX_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache_index(index: dict):
    """Save the SFX cache index to disk atomically."""
    _ensure_cache_dir()
    fd, tmp = tempfile.mkstemp(dir=SFX_CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, SFX_INDEX_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _try_freesound_search(query: str, api_key: str, filters: Optional[Dict] = None,
                          page_size: int = 15) -> Optional[List[SFXSearchResult]]:
    """Attempt to search Freesound.org API. Returns None on failure."""
    try:
        import urllib.parse
        import urllib.request

        params = {
            "query": query,
            "token": api_key,
            "fields": "id,name,tags,duration,license,previews",
            "page_size": str(page_size),
        }
        if filters:
            filter_parts = []
            if "min_duration" in filters:
                filter_parts.append(f"duration:[{filters['min_duration']} TO *]")
            if "max_duration" in filters:
                filter_parts.append(f"duration:[* TO {filters['max_duration']}]")
            if "tag" in filters:
                filter_parts.append(f'tag:"{filters["tag"]}"')
            if filter_parts:
                params["filter"] = " ".join(filter_parts)

        url = f"{FREESOUND_API_BASE}/search/text/?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("results", []):
            previews = item.get("previews", {})
            preview = previews.get("preview-lq-mp3", previews.get("preview-hq-mp3", ""))
            results.append(SFXSearchResult(
                id=str(item["id"]),
                name=item.get("name", ""),
                tags=item.get("tags", []),
                duration=float(item.get("duration", 0)),
                category="",
                source="freesound",
                preview_url=preview,
                license=item.get("license", ""),
            ))
        return results

    except Exception as e:
        logger.debug("Freesound API search failed: %s", e)
        return None


def search_sfx(
    query: str,
    filters: Optional[Dict] = None,
    api_key: str = "",
    use_freesound: bool = True,
    on_progress: Optional[Callable] = None,
) -> List[SFXSearchResult]:
    """Search for sound effects by query string.

    Searches Freesound.org API if an API key is provided, otherwise
    falls back to the built-in SFX library.

    Args:
        query: Search query string.
        filters: Optional dict with 'min_duration', 'max_duration', 'tag', 'category'.
        api_key: Freesound.org API key. Empty string skips API search.
        use_freesound: Whether to try Freesound API (requires api_key).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of SFXSearchResult objects.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    query = query.strip().lower()

    if on_progress:
        on_progress(10, f"Searching SFX: '{query}'...")

    # Try Freesound API first
    if use_freesound and api_key:
        if on_progress:
            on_progress(20, "Querying Freesound.org API...")
        api_results = _try_freesound_search(query, api_key, filters)
        if api_results is not None:
            if on_progress:
                on_progress(90, f"Found {len(api_results)} results from Freesound")
            return api_results

    # Fallback to built-in library
    if on_progress:
        on_progress(50, "Searching built-in SFX library...")

    results = []
    query_words = query.split()

    for sfx in BUILTIN_SFX:
        # Score by how many query words match name or tags
        name_lower = sfx["name"].lower()
        tags_str = " ".join(sfx["tags"]).lower()
        searchable = f"{name_lower} {tags_str} {sfx.get('category', '')}"

        matches = sum(1 for w in query_words if w in searchable)
        if matches > 0:
            results.append(SFXSearchResult(
                id=sfx["id"],
                name=sfx["name"],
                tags=sfx["tags"],
                duration=sfx["duration"],
                category=sfx.get("category", ""),
                source="builtin",
            ))

    # Apply filters
    if filters:
        min_dur = float(filters.get("min_duration", 0))
        max_dur = float(filters.get("max_duration", float("inf")))
        category = filters.get("category", "")

        filtered = []
        for r in results:
            if r.duration < min_dur:
                continue
            if r.duration > max_dur:
                continue
            if category and r.category != category:
                continue
            filtered.append(r)
        results = filtered

    if on_progress:
        on_progress(90, f"Found {len(results)} results")

    return results


def download_sfx(
    sfx_id: str,
    output_dir: Optional[str] = None,
    api_key: str = "",
    on_progress: Optional[Callable] = None,
) -> SFXDownloadResult:
    """Download an SFX file by ID and cache it locally.

    For built-in SFX, generates the audio via FFmpeg synthesis.
    For Freesound SFX, downloads via API (requires api_key).

    Args:
        sfx_id: SFX identifier (e.g. "builtin_003" or a Freesound numeric ID).
        output_dir: Directory to save the file. Uses SFX cache dir if None.
        api_key: Freesound.org API key for downloading Freesound SFX.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SFXDownloadResult with local file path.
    """
    _ensure_cache_dir()
    target_dir = output_dir or SFX_CACHE_DIR

    if on_progress:
        on_progress(10, f"Downloading SFX {sfx_id}...")

    # Check cache first
    index = _load_cache_index()
    if sfx_id in index:
        cached = index[sfx_id]
        if os.path.isfile(cached.get("path", "")):
            if on_progress:
                on_progress(100, "Found in cache")
            return SFXDownloadResult(
                id=sfx_id,
                name=cached.get("name", ""),
                local_path=cached["path"],
                duration=cached.get("duration", 0),
                source=cached.get("source", "cache"),
            )

    # Handle built-in SFX
    if sfx_id.startswith("builtin_"):
        builtin = next((s for s in BUILTIN_SFX if s["id"] == sfx_id), None)
        if builtin is None:
            raise ValueError(f"Unknown built-in SFX ID: {sfx_id}")

        if on_progress:
            on_progress(30, f"Generating {builtin['name']}...")

        # Generate via FFmpeg synthesis
        out_path = os.path.join(target_dir, f"{sfx_id}.wav")
        os.makedirs(target_dir, exist_ok=True)

        # Simple synthesis based on category
        duration = builtin["duration"]
        category = builtin.get("category", "ui")

        # Map category to a synthesis filter
        from opencut.core.ai_sfx import SFX_CATEGORIES, _build_synth_filter
        synth_cat = "click" if category == "ui" else category
        if synth_cat not in SFX_CATEGORIES:
            synth_cat = "click"

        filter_str = _build_synth_filter(synth_cat, duration)
        cmd = (
            FFmpegCmd()
            .filter_complex(f"{filter_str}[out]", maps=["[out]"])
            .audio_codec("pcm_s16le")
            .option("ar", "44100")
            .option("ac", "1")
            .output(out_path)
            .build()
        )
        run_ffmpeg(cmd)

        # Update cache index
        index[sfx_id] = {
            "name": builtin["name"],
            "path": out_path,
            "duration": duration,
            "source": "builtin",
            "cached_at": time.time(),
        }
        _save_cache_index(index)

        if on_progress:
            on_progress(95, "Generated and cached")

        return SFXDownloadResult(
            id=sfx_id,
            name=builtin["name"],
            local_path=out_path,
            duration=duration,
            source="builtin",
        )

    # Handle Freesound SFX
    if not api_key:
        raise ValueError("Freesound API key required to download non-builtin SFX")

    if on_progress:
        on_progress(20, "Fetching from Freesound.org...")

    try:
        import urllib.parse
        import urllib.request

        # Get sound details
        detail_url = f"{FREESOUND_API_BASE}/sounds/{sfx_id}/?token={api_key}&fields=id,name,duration,previews"
        req = urllib.request.Request(detail_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            sound_data = json.loads(resp.read().decode())

        name = sound_data.get("name", f"sfx_{sfx_id}")
        duration = float(sound_data.get("duration", 0))
        previews = sound_data.get("previews", {})
        download_url = previews.get("preview-hq-mp3", previews.get("preview-lq-mp3", ""))

        if not download_url:
            raise ValueError(f"No preview URL found for Freesound sound {sfx_id}")

        if on_progress:
            on_progress(50, f"Downloading {name}...")

        # Download the preview
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:50]
        out_path = os.path.join(target_dir, f"freesound_{sfx_id}_{safe_name}.mp3")
        os.makedirs(target_dir, exist_ok=True)

        urllib.request.urlretrieve(download_url, out_path)

        # Update cache
        index[sfx_id] = {
            "name": name,
            "path": out_path,
            "duration": duration,
            "source": "freesound",
            "cached_at": time.time(),
        }
        _save_cache_index(index)

        if on_progress:
            on_progress(95, "Downloaded and cached")

        return SFXDownloadResult(
            id=sfx_id,
            name=name,
            local_path=out_path,
            duration=duration,
            source="freesound",
        )

    except Exception as e:
        raise RuntimeError(f"Failed to download SFX {sfx_id} from Freesound: {e}") from e


def list_cached_sfx(
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """List all locally cached SFX files.

    Args:
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of dicts with id, name, path, duration, source, cached_at.
    """
    if on_progress:
        on_progress(10, "Loading SFX cache index...")

    index = _load_cache_index()
    results = []

    for sfx_id, info in index.items():
        path = info.get("path", "")
        if os.path.isfile(path):
            results.append({
                "id": sfx_id,
                "name": info.get("name", ""),
                "path": path,
                "duration": info.get("duration", 0),
                "source": info.get("source", "unknown"),
                "cached_at": info.get("cached_at", 0),
                "size_bytes": os.path.getsize(path),
            })

    if on_progress:
        on_progress(90, f"Found {len(results)} cached SFX files")

    return results
