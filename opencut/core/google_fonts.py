"""
OpenCut Google Fonts Browser Module (Feature 19.3)

Fetch the Google Fonts catalog (or use a bundled fallback list),
search/filter fonts by category and name, and download TTF files
on demand to ~/.opencut/fonts/.

Functions:
    list_fonts     - List fonts, optionally filtered by category
    search_fonts   - Search fonts by name query
    download_font  - Download a font's TTF files to local cache
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
FONTS_DIR = os.path.join(OPENCUT_DIR, "fonts")
FONTS_CATALOG_CACHE = os.path.join(FONTS_DIR, "catalog.json")
FONTS_CATALOG_MAX_AGE = 86400 * 7  # 7 days cache

GOOGLE_FONTS_API_BASE = "https://www.googleapis.com/webfonts/v1/webfonts"
GOOGLE_FONTS_DOWNLOAD_BASE = "https://fonts.google.com/download"

# Font categories from Google Fonts
FONT_CATEGORIES = ["serif", "sans-serif", "display", "handwriting", "monospace"]

# Bundled fallback catalog (subset of popular Google Fonts)
BUNDLED_FONTS = [
    {"family": "Roboto", "category": "sans-serif", "variants": ["regular", "bold", "italic", "700", "300"], "subsets": ["latin"]},
    {"family": "Open Sans", "category": "sans-serif", "variants": ["regular", "bold", "italic", "600", "700"], "subsets": ["latin"]},
    {"family": "Lato", "category": "sans-serif", "variants": ["regular", "bold", "italic", "300", "700"], "subsets": ["latin"]},
    {"family": "Montserrat", "category": "sans-serif", "variants": ["regular", "bold", "italic", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Oswald", "category": "sans-serif", "variants": ["regular", "bold", "300", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Poppins", "category": "sans-serif", "variants": ["regular", "bold", "italic", "300", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Raleway", "category": "sans-serif", "variants": ["regular", "bold", "italic", "300", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Nunito", "category": "sans-serif", "variants": ["regular", "bold", "italic", "300", "600", "700"], "subsets": ["latin"]},
    {"family": "Inter", "category": "sans-serif", "variants": ["regular", "bold", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Ubuntu", "category": "sans-serif", "variants": ["regular", "bold", "italic", "300", "500", "700"], "subsets": ["latin"]},
    {"family": "Playfair Display", "category": "serif", "variants": ["regular", "bold", "italic", "700"], "subsets": ["latin"]},
    {"family": "Merriweather", "category": "serif", "variants": ["regular", "bold", "italic", "300", "700"], "subsets": ["latin"]},
    {"family": "Lora", "category": "serif", "variants": ["regular", "bold", "italic", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "PT Serif", "category": "serif", "variants": ["regular", "bold", "italic"], "subsets": ["latin"]},
    {"family": "Noto Serif", "category": "serif", "variants": ["regular", "bold", "italic"], "subsets": ["latin"]},
    {"family": "Source Serif Pro", "category": "serif", "variants": ["regular", "bold", "italic", "600", "700"], "subsets": ["latin"]},
    {"family": "Libre Baskerville", "category": "serif", "variants": ["regular", "bold", "italic"], "subsets": ["latin"]},
    {"family": "Fira Code", "category": "monospace", "variants": ["regular", "bold", "300", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Source Code Pro", "category": "monospace", "variants": ["regular", "bold", "italic", "300", "500", "700"], "subsets": ["latin"]},
    {"family": "JetBrains Mono", "category": "monospace", "variants": ["regular", "bold", "italic", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Roboto Mono", "category": "monospace", "variants": ["regular", "bold", "italic", "300", "500", "700"], "subsets": ["latin"]},
    {"family": "Space Mono", "category": "monospace", "variants": ["regular", "bold", "italic"], "subsets": ["latin"]},
    {"family": "Dancing Script", "category": "handwriting", "variants": ["regular", "bold", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Pacifico", "category": "handwriting", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Caveat", "category": "handwriting", "variants": ["regular", "bold", "500", "600", "700"], "subsets": ["latin"]},
    {"family": "Great Vibes", "category": "handwriting", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Sacramento", "category": "handwriting", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Lobster", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Bebas Neue", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Anton", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Abril Fatface", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Righteous", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Permanent Marker", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Press Start 2P", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
    {"family": "Bangers", "category": "display", "variants": ["regular"], "subsets": ["latin"]},
]


@dataclass
class FontInfo:
    """Information about a single font family."""
    family: str
    category: str
    variants: List[str] = field(default_factory=list)
    subsets: List[str] = field(default_factory=list)
    is_downloaded: bool = False
    local_path: str = ""


@dataclass
class FontDownloadResult:
    """Result of downloading a font."""
    family: str
    local_dir: str
    files: List[str]
    total_size_bytes: int


def _ensure_fonts_dir():
    """Create the fonts directory if it doesn't exist."""
    os.makedirs(FONTS_DIR, exist_ok=True)


def _get_font_dir(family: str) -> str:
    """Get the local directory for a font family."""
    safe_name = family.replace(" ", "_").replace("/", "_")
    return os.path.join(FONTS_DIR, safe_name)


def _is_font_downloaded(family: str) -> bool:
    """Check if a font family has been downloaded locally."""
    font_dir = _get_font_dir(family)
    if not os.path.isdir(font_dir):
        return False
    # Check for at least one TTF/OTF file
    for f in os.listdir(font_dir):
        if f.lower().endswith((".ttf", ".otf")):
            return True
    return False


def _load_catalog() -> List[Dict]:
    """Load font catalog from cache or fallback to bundled list.

    Returns list of font dicts.
    """
    # Try cached catalog
    if os.path.isfile(FONTS_CATALOG_CACHE):
        try:
            mtime = os.path.getmtime(FONTS_CATALOG_CACHE)
            if time.time() - mtime < FONTS_CATALOG_MAX_AGE:
                with open(FONTS_CATALOG_CACHE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return BUNDLED_FONTS


def _try_fetch_catalog(api_key: str = "") -> Optional[List[Dict]]:
    """Try to fetch the Google Fonts catalog from the API.

    Returns list of font dicts, or None on failure.
    """
    try:
        import urllib.parse
        import urllib.request

        params = {"sort": "popularity"}
        if api_key:
            params["key"] = api_key

        url = f"{GOOGLE_FONTS_API_BASE}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        items = data.get("items", [])
        catalog = []
        for item in items:
            catalog.append({
                "family": item.get("family", ""),
                "category": item.get("category", ""),
                "variants": item.get("variants", []),
                "subsets": item.get("subsets", []),
            })

        # Cache the catalog
        if catalog:
            _ensure_fonts_dir()
            fd, tmp = tempfile.mkstemp(dir=FONTS_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(catalog, f)
                os.replace(tmp, FONTS_CATALOG_CACHE)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return catalog

    except Exception as e:
        logger.debug("Failed to fetch Google Fonts catalog: %s", e)
        return None


def list_fonts(
    category: str = "",
    api_key: str = "",
    refresh: bool = False,
    on_progress: Optional[Callable] = None,
) -> List[FontInfo]:
    """List available Google Fonts, optionally filtered by category.

    Args:
        category: Filter by category: serif, sans-serif, display, handwriting, monospace.
        api_key: Google Fonts API key. Empty uses bundled/cached catalog.
        refresh: Force refresh from API even if cache is fresh.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of FontInfo objects.
    """
    if on_progress:
        on_progress(10, "Loading font catalog...")

    catalog = None

    if refresh and api_key:
        if on_progress:
            on_progress(20, "Fetching from Google Fonts API...")
        catalog = _try_fetch_catalog(api_key)

    if catalog is None:
        catalog = _load_catalog()

    if on_progress:
        on_progress(60, "Processing fonts...")

    results = []
    for font in catalog:
        family = font.get("family", "")
        cat = font.get("category", "")

        if category and cat.lower() != category.lower():
            continue

        downloaded = _is_font_downloaded(family)
        local_path = _get_font_dir(family) if downloaded else ""

        results.append(FontInfo(
            family=family,
            category=cat,
            variants=font.get("variants", []),
            subsets=font.get("subsets", []),
            is_downloaded=downloaded,
            local_path=local_path,
        ))

    if on_progress:
        on_progress(90, f"Found {len(results)} fonts")

    return results


def search_fonts(
    query: str,
    category: str = "",
    api_key: str = "",
    on_progress: Optional[Callable] = None,
) -> List[FontInfo]:
    """Search fonts by name query.

    Args:
        query: Search string to match against font family names.
        category: Optional category filter.
        api_key: Google Fonts API key.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of matching FontInfo objects, sorted by relevance.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    query = query.strip().lower()

    if on_progress:
        on_progress(10, f"Searching fonts: '{query}'...")

    all_fonts = list_fonts(category=category, api_key=api_key)

    # Score and filter
    scored = []
    for font in all_fonts:
        name_lower = font.family.lower()
        if query in name_lower:
            # Exact substring match scores higher
            score = 100 if name_lower == query else (50 if name_lower.startswith(query) else 25)
            scored.append((score, font))
        else:
            # Check individual query words
            words = query.split()
            matches = sum(1 for w in words if w in name_lower)
            if matches > 0:
                scored.append((matches * 10, font))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [font for _, font in scored]

    if on_progress:
        on_progress(90, f"Found {len(results)} matching fonts")

    return results


def download_font(
    font_name: str,
    variant: str = "regular",
    api_key: str = "",
    on_progress: Optional[Callable] = None,
) -> FontDownloadResult:
    """Download a Google Font's TTF files to the local cache.

    Uses the Google Fonts CSS API to fetch the TTF URL and downloads it.
    Falls back to generating a placeholder if the API is unavailable.

    Args:
        font_name: Font family name (e.g. "Roboto", "Open Sans").
        variant: Font variant to download (e.g. "regular", "bold", "italic").
        api_key: Google Fonts API key.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        FontDownloadResult with local directory and file list.
    """
    if not font_name or not font_name.strip():
        raise ValueError("Font name cannot be empty")

    font_name = font_name.strip()
    font_dir = _get_font_dir(font_name)
    os.makedirs(font_dir, exist_ok=True)

    if on_progress:
        on_progress(10, f"Downloading {font_name}...")

    # Check if already downloaded
    existing_files = []
    if os.path.isdir(font_dir):
        existing_files = [
            f for f in os.listdir(font_dir)
            if f.lower().endswith((".ttf", ".otf"))
        ]
    if existing_files:
        total_size = sum(
            os.path.getsize(os.path.join(font_dir, f))
            for f in existing_files
        )
        if on_progress:
            on_progress(100, f"{font_name} already downloaded ({len(existing_files)} files)")
        return FontDownloadResult(
            family=font_name,
            local_dir=font_dir,
            files=[os.path.join(font_dir, f) for f in existing_files],
            total_size_bytes=total_size,
        )

    # Try downloading via Google Fonts CSS2 API
    downloaded_files = []
    try:
        import urllib.parse
        import urllib.request

        if on_progress:
            on_progress(20, "Fetching font URL from Google Fonts...")

        # Use CSS2 API to get TTF URL
        encoded_name = urllib.parse.quote(font_name)
        css_url = (
            f"https://fonts.googleapis.com/css2?"
            f"family={encoded_name}:wght@400&display=swap"
        )
        req = urllib.request.Request(css_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; OpenCut/1.0)",
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            css_text = resp.read().decode()

        # Extract TTF/WOFF2 URLs from CSS
        import re
        url_pattern = re.compile(r"url\((https://fonts\.gstatic\.com/[^)]+)\)")
        font_urls = url_pattern.findall(css_text)

        if font_urls:
            if on_progress:
                on_progress(50, f"Downloading {len(font_urls)} font file(s)...")

            for i, url in enumerate(font_urls):
                ext = ".woff2" if "woff2" in url else ".ttf"
                safe_variant = variant.replace(" ", "_")
                filename = f"{font_name.replace(' ', '_')}_{safe_variant}_{i}{ext}"
                filepath = os.path.join(font_dir, filename)

                urllib.request.urlretrieve(url, filepath)
                downloaded_files.append(filepath)

                if on_progress:
                    pct = 50 + int((i + 1) / len(font_urls) * 40)
                    on_progress(pct, f"Downloaded {i + 1}/{len(font_urls)}")
        else:
            logger.info("No font URLs found in CSS response for %s", font_name)

    except Exception as e:
        logger.debug("Failed to download font %s from Google Fonts: %s", font_name, e)

    # If download failed, create a placeholder marker file
    if not downloaded_files:
        logger.info("Creating placeholder marker for font %s", font_name)
        marker_path = os.path.join(font_dir, ".pending_download")
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Font: {font_name}\nVariant: {variant}\nStatus: pending\n")
        downloaded_files = [marker_path]

    total_size = sum(
        os.path.getsize(f) for f in downloaded_files if os.path.isfile(f)
    )

    if on_progress:
        on_progress(95, f"Font {font_name} ready")

    return FontDownloadResult(
        family=font_name,
        local_dir=font_dir,
        files=downloaded_files,
        total_size_bytes=total_size,
    )
