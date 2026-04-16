"""
Stock Media Search.

Search Pexels and Pixabay APIs for stock videos and photos,
preview results, download, and import into projects.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

SUPPORTED_SOURCES = {"pexels", "pixabay"}


@dataclass
class StockMediaResult:
    """A single stock media search result."""
    media_id: str
    source: str               # pexels | pixabay
    media_type: str            # video | photo
    title: str = ""
    url: str = ""
    preview_url: str = ""
    download_url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0     # seconds, for video
    author: str = ""
    author_url: str = ""
    license: str = "free"
    tags: List[str] = field(default_factory=list)


def _get_api_key(source: str) -> str:
    """Retrieve API key from environment variables."""
    key_map = {
        "pexels": "PEXELS_API_KEY",
        "pixabay": "PIXABAY_API_KEY",
    }
    env_var = key_map.get(source, "")
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"API key not set. Set the {env_var} environment variable."
        )
    return key


def _http_get_json(req_or_url, timeout: int = 15) -> dict:
    """Shared GET-and-parse helper that turns network/parse failures into a
    clear ``RuntimeError`` instead of letting raw ``urllib`` /
    ``json.JSONDecodeError`` exceptions propagate up to the worker. Returns
    an empty dict so downstream ``data.get("videos", [])`` calls keep
    returning an empty list rather than crashing on a missing API key /
    rate limit / malformed response.
    """
    try:
        with urllib.request.urlopen(req_or_url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Stock API HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Stock API network error: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"Stock API request failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Stock API returned invalid JSON: {exc}") from exc
    return parsed if isinstance(parsed, dict) else {}


def _pexels_request(endpoint: str, api_key: str) -> dict:
    """Make an authenticated request to the Pexels API."""
    req = urllib.request.Request(
        f"https://api.pexels.com{endpoint}",
        headers={"Authorization": api_key},
    )
    return _http_get_json(req)


def _pixabay_request(endpoint: str, api_key: str) -> dict:
    """Make a request to the Pixabay API."""
    sep = "&" if "?" in endpoint else "?"
    url = f"https://pixabay.com/api{endpoint}{sep}key={api_key}"
    return _http_get_json(url)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def search_stock_video(
    query: str,
    source: str = "pexels",
    page: int = 1,
    per_page: int = 15,
    orientation: str = "",
    min_duration: int = 0,
    max_duration: int = 0,
    on_progress: Optional[Callable] = None,
) -> List[StockMediaResult]:
    """Search for stock videos.

    Args:
        query: Search keywords.
        source: API source (pexels or pixabay).
        page: Page number.
        per_page: Results per page (max 80).
        orientation: landscape, portrait, or square.
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.

    Returns:
        List of StockMediaResult.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    source = source.lower().strip()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}. Use: {SUPPORTED_SOURCES}")
    per_page = min(max(per_page, 1), 80)

    if on_progress:
        on_progress(20, f"Searching {source} for videos")

    api_key = _get_api_key(source)

    if source == "pexels":
        results = _search_pexels_video(api_key, query, page, per_page, orientation)
    else:
        results = _search_pixabay_video(api_key, query, page, per_page)

    # Filter by duration
    if min_duration > 0:
        results = [r for r in results if r.duration >= min_duration]
    if max_duration > 0:
        results = [r for r in results if r.duration <= max_duration]

    if on_progress:
        on_progress(100, f"Found {len(results)} videos")

    return results


def _search_pexels_video(
    api_key: str, query: str, page: int, per_page: int, orientation: str,
) -> List[StockMediaResult]:
    q = urllib.parse.quote(query)
    endpoint = f"/videos/search?query={q}&page={page}&per_page={per_page}"
    if orientation:
        endpoint += f"&orientation={orientation}"
    data = _pexels_request(endpoint, api_key)
    results = []
    for v in data.get("videos", []):
        files = v.get("video_files", [])
        best = max(files, key=lambda f: f.get("width", 0)) if files else {}
        preview = v.get("video_pictures", [{}])
        results.append(StockMediaResult(
            media_id=str(v.get("id", "")),
            source="pexels",
            media_type="video",
            title=v.get("url", "").split("/")[-2] if v.get("url") else "",
            url=v.get("url", ""),
            preview_url=preview[0].get("picture", "") if preview else "",
            download_url=best.get("link", ""),
            width=best.get("width", 0),
            height=best.get("height", 0),
            duration=float(v.get("duration", 0)),
            author=v.get("user", {}).get("name", ""),
            author_url=v.get("user", {}).get("url", ""),
        ))
    return results


def _search_pixabay_video(
    api_key: str, query: str, page: int, per_page: int,
) -> List[StockMediaResult]:
    q = urllib.parse.quote(query)
    endpoint = f"/videos/?q={q}&page={page}&per_page={per_page}"
    data = _pixabay_request(endpoint, api_key)
    results = []
    for v in data.get("hits", []):
        videos = v.get("videos", {})
        large = videos.get("large", {})
        results.append(StockMediaResult(
            media_id=str(v.get("id", "")),
            source="pixabay",
            media_type="video",
            title=v.get("tags", ""),
            url=v.get("pageURL", ""),
            preview_url=v.get("pictureId", ""),
            download_url=large.get("url", ""),
            width=large.get("width", 0),
            height=large.get("height", 0),
            duration=float(v.get("duration", 0)),
            author=v.get("user", ""),
            tags=v.get("tags", "").split(", ") if v.get("tags") else [],
        ))
    return results


def search_stock_photo(
    query: str,
    source: str = "pexels",
    page: int = 1,
    per_page: int = 15,
    orientation: str = "",
    on_progress: Optional[Callable] = None,
) -> List[StockMediaResult]:
    """Search for stock photos.

    Args:
        query: Search keywords.
        source: API source (pexels or pixabay).
        page: Page number.
        per_page: Results per page.
        orientation: landscape, portrait, or square.

    Returns:
        List of StockMediaResult.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    source = source.lower().strip()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source: {source}. Use: {SUPPORTED_SOURCES}")
    per_page = min(max(per_page, 1), 80)

    if on_progress:
        on_progress(20, f"Searching {source} for photos")

    api_key = _get_api_key(source)

    if source == "pexels":
        results = _search_pexels_photo(api_key, query, page, per_page, orientation)
    else:
        results = _search_pixabay_photo(api_key, query, page, per_page)

    if on_progress:
        on_progress(100, f"Found {len(results)} photos")

    return results


def _search_pexels_photo(
    api_key: str, query: str, page: int, per_page: int, orientation: str,
) -> List[StockMediaResult]:
    q = urllib.parse.quote(query)
    endpoint = f"/v1/search?query={q}&page={page}&per_page={per_page}"
    if orientation:
        endpoint += f"&orientation={orientation}"
    data = _pexels_request(endpoint, api_key)
    results = []
    for p in data.get("photos", []):
        src = p.get("src", {})
        results.append(StockMediaResult(
            media_id=str(p.get("id", "")),
            source="pexels",
            media_type="photo",
            title=p.get("alt", ""),
            url=p.get("url", ""),
            preview_url=src.get("medium", ""),
            download_url=src.get("original", ""),
            width=p.get("width", 0),
            height=p.get("height", 0),
            author=p.get("photographer", ""),
            author_url=p.get("photographer_url", ""),
        ))
    return results


def _search_pixabay_photo(
    api_key: str, query: str, page: int, per_page: int,
) -> List[StockMediaResult]:
    q = urllib.parse.quote(query)
    endpoint = f"/?q={q}&page={page}&per_page={per_page}&image_type=photo"
    data = _pixabay_request(endpoint, api_key)
    results = []
    for p in data.get("hits", []):
        results.append(StockMediaResult(
            media_id=str(p.get("id", "")),
            source="pixabay",
            media_type="photo",
            title=p.get("tags", ""),
            url=p.get("pageURL", ""),
            preview_url=p.get("webformatURL", ""),
            download_url=p.get("largeImageURL", ""),
            width=p.get("imageWidth", 0),
            height=p.get("imageHeight", 0),
            author=p.get("user", ""),
            tags=p.get("tags", "").split(", ") if p.get("tags") else [],
        ))
    return results


def download_stock_media(
    media_id: str,
    source: str,
    output_dir: str,
    url: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Download a stock media file.

    Args:
        media_id: Media identifier from search result.
        source: Source service (pexels or pixabay).
        output_dir: Directory to save downloaded file.
        url: Direct download URL (if known).

    Returns:
        Dict with output_path and file_size.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(10, "Preparing download")

    if not url:
        raise ValueError("Download URL is required")

    ext = ".mp4" if "video" in url.lower() else ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    output_path = os.path.join(output_dir, f"{source}_{media_id}{ext}")

    if on_progress:
        on_progress(30, "Downloading")

    urllib.request.urlretrieve(url, output_path)

    file_size = os.path.getsize(output_path)

    if on_progress:
        on_progress(100, "Download complete")

    logger.info("Downloaded %s media %s to %s (%d bytes)",
                source, media_id, output_path, file_size)
    return {
        "output_path": output_path,
        "file_size": file_size,
        "media_id": media_id,
        "source": source,
    }
