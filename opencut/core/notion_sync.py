"""
OpenCut Notion / PM Sync Module

Sync project status and metadata to Notion databases via the
Notion API.  Supports creating new entries, updating existing
pages, and bulk-syncing project data.

Requires a Notion integration token (API key) passed per-call
or stored in ``~/.opencut/notion_config.json``.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_CONFIG_FILE = os.path.join(_OPENCUT_DIR, "notion_config.json")
_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


@dataclass
class NotionSyncResult:
    """Result of a Notion sync operation."""
    success: bool = False
    pages_created: int = 0
    pages_updated: int = 0
    errors: List[str] = field(default_factory=list)
    page_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_notion_config() -> Dict[str, Any]:
    """Load Notion configuration from disk."""
    if not os.path.isfile(_CONFIG_FILE):
        return {}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_notion_config(config: Dict[str, Any]) -> None:
    """Persist Notion configuration to disk."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------

def _notion_request(
    method: str,
    path: str,
    api_key: str,
    body: Optional[Dict] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Make an authenticated request to the Notion API.

    Args:
        method: HTTP method (GET, POST, PATCH).
        path: API path (e.g. ``/pages``).
        api_key: Notion integration token.
        body: Optional JSON body dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: On HTTP errors.
    """
    url = f"{_NOTION_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": _NOTION_VERSION,
    }

    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Notion API error {e.code}: {error_body[:500]}"
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Notion API connection error: {e}") from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_notion_page(
    page_id: str,
    properties: Dict[str, Any],
    api_key: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Update properties on an existing Notion page.

    Args:
        page_id: Notion page UUID.
        properties: Dict of property name -> value in Notion API format.
        api_key: Notion integration token.
        on_progress: Optional progress callback.

    Returns:
        Updated page data from the API.
    """
    if on_progress:
        on_progress(30, f"Updating Notion page {page_id[:8]}...")

    # Build Notion-format properties
    notion_props = _format_properties(properties)

    result = _notion_request(
        "PATCH",
        f"/pages/{page_id}",
        api_key,
        body={"properties": notion_props},
    )

    if on_progress:
        on_progress(100, "Page updated")

    logger.info("Updated Notion page %s", page_id)
    return result


def create_notion_entry(
    database_id: str,
    data: Dict[str, Any],
    api_key: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a new entry in a Notion database.

    Args:
        database_id: Notion database UUID.
        data: Dict of property name -> value pairs.
        api_key: Notion integration token.
        on_progress: Optional progress callback.

    Returns:
        Created page data from the API.
    """
    if on_progress:
        on_progress(30, "Creating Notion database entry...")

    notion_props = _format_properties(data)

    body = {
        "parent": {"database_id": database_id},
        "properties": notion_props,
    }

    result = _notion_request("POST", "/pages", api_key, body=body)

    if on_progress:
        on_progress(100, "Entry created")

    page_id = result.get("id", "")
    logger.info("Created Notion entry %s in database %s", page_id, database_id)
    return result


def sync_to_notion(
    project_data: Dict[str, Any],
    config: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> NotionSyncResult:
    """Sync project data to a Notion database.

    If ``page_id`` is present in config, updates the existing page.
    Otherwise creates a new entry in the configured database.

    Args:
        project_data: Project metadata dict with keys like ``name``,
            ``status``, ``progress``, ``export_path``, ``duration``, etc.
        config: Sync configuration with ``api_key``, ``database_id``,
            and optionally ``page_id`` for updates.
        on_progress: Optional progress callback.

    Returns:
        NotionSyncResult with operation outcome.
    """
    if on_progress:
        on_progress(5, "Preparing Notion sync...")

    api_key = config.get("api_key", "")
    database_id = config.get("database_id", "")
    page_id = config.get("page_id", "")

    if not api_key:
        # Try loading from config file
        saved = load_notion_config()
        api_key = saved.get("api_key", "")
        if not database_id:
            database_id = saved.get("database_id", "")

    if not api_key:
        return NotionSyncResult(
            success=False,
            errors=["No Notion API key configured"],
        )

    result = NotionSyncResult()

    try:
        if page_id:
            # Update existing page
            if on_progress:
                on_progress(30, "Updating existing Notion page...")

            update_notion_page(page_id, project_data, api_key)
            result.pages_updated = 1
            result.page_ids = [page_id]
        elif database_id:
            # Create new entry
            if on_progress:
                on_progress(30, "Creating new Notion entry...")

            resp = create_notion_entry(database_id, project_data, api_key)
            new_id = resp.get("id", "")
            result.pages_created = 1
            result.page_ids = [new_id]
        else:
            result.errors.append("No database_id or page_id configured")
            return result

        result.success = True

    except Exception as e:
        result.errors.append(str(e))
        logger.warning("Notion sync failed: %s", e)

    if on_progress:
        on_progress(100, "Notion sync complete")

    logger.info("Notion sync: created=%d, updated=%d, errors=%d",
                result.pages_created, result.pages_updated, len(result.errors))
    return result


# ---------------------------------------------------------------------------
# Property formatting
# ---------------------------------------------------------------------------

def _format_properties(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert simple key-value pairs to Notion property format.

    Supports string, number, boolean, and list values.
    """
    props = {}
    for key, value in data.items():
        if isinstance(value, bool):
            props[key] = {"checkbox": value}
        elif isinstance(value, str):
            props[key] = {
                "rich_text": [{"text": {"content": value[:2000]}}],
            }
        elif isinstance(value, (int, float)):
            props[key] = {"number": value}
        elif isinstance(value, list):
            # Multi-select
            props[key] = {
                "multi_select": [{"name": str(v)[:100]} for v in value[:25]],
            }
        else:
            props[key] = {
                "rich_text": [{"text": {"content": str(value)[:2000]}}],
            }
    return props
