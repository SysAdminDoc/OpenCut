"""
OpenCut License Tracking Module

Record and query third-party asset usage with license metadata.
Stores records in ``~/.opencut/licenses.db`` and supports exporting
a formatted attribution document grouped by license type.
"""

import logging
import os
import sqlite3
import time
import uuid
from typing import Dict, List

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_LICENSES_DB = os.path.join(_OPENCUT_DIR, "licenses.db")


def _get_db() -> sqlite3.Connection:
    """Open (and initialise if needed) the license database."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    conn = sqlite3.connect(_LICENSES_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS licenses (
            record_id       TEXT PRIMARY KEY,
            source_url      TEXT NOT NULL,
            filename        TEXT NOT NULL,
            license_type    TEXT NOT NULL,
            attribution_text TEXT NOT NULL DEFAULT '',
            project_id      TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_licenses_project
        ON licenses(project_id)
    """)
    conn.commit()
    return conn


def record_asset_usage(
    source_url: str,
    filename: str,
    license_type: str,
    attribution_text: str = "",
    project_id: str = "",
) -> dict:
    """
    Record usage of a third-party asset.

    Args:
        source_url: URL where the asset was obtained.
        filename: Local filename of the asset.
        license_type: License identifier (e.g. ``"CC-BY-4.0"``, ``"MIT"``,
                      ``"Royalty-Free"``, ``"Editorial"``).
        attribution_text: Required attribution/credit text.
        project_id: Associated project identifier.

    Returns:
        dict with the stored record fields including ``record_id``.
    """
    record_id = uuid.uuid4().hex[:12]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO licenses
               (record_id, source_url, filename, license_type, attribution_text, project_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (record_id, source_url, filename, license_type, attribution_text, project_id, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "record_id": record_id,
        "source_url": source_url,
        "filename": filename,
        "license_type": license_type,
        "attribution_text": attribution_text,
        "project_id": project_id,
        "created_at": now,
    }


def get_project_licenses(project_id: str) -> List[dict]:
    """
    Retrieve all license records for a project.

    Returns:
        List of license record dicts ordered by creation date.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM licenses WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def export_attribution(project_id: str, format: str = "text") -> str:
    """
    Export a formatted attribution document for a project.

    Groups assets by license type and includes all required attribution text.

    Args:
        project_id: Project to export attributions for.
        format: ``"text"`` or ``"markdown"``.

    Returns:
        Formatted attribution string.
    """
    records = get_project_licenses(project_id)
    if not records:
        return f"No license records found for project {project_id}."

    # Group by license type
    groups: Dict[str, List[dict]] = {}
    for r in records:
        lt = r["license_type"]
        if lt not in groups:
            groups[lt] = []
        groups[lt].append(r)

    if format == "markdown":
        return _export_markdown(groups, project_id)
    return _export_text(groups, project_id)


def _export_text(groups: Dict[str, List[dict]], project_id: str) -> str:
    lines = [f"Attribution — Project: {project_id}", "=" * 50]

    for license_type, records in sorted(groups.items()):
        lines.append("")
        lines.append(f"--- {license_type} ({len(records)} asset(s)) ---")
        for r in records:
            lines.append(f"  File: {r['filename']}")
            lines.append(f"  Source: {r['source_url']}")
            if r["attribution_text"]:
                lines.append(f"  Credit: {r['attribution_text']}")
            lines.append("")

    return "\n".join(lines)


def _export_markdown(groups: Dict[str, List[dict]], project_id: str) -> str:
    lines = [f"# Attribution — Project: {project_id}", ""]

    for license_type, records in sorted(groups.items()):
        lines.append(f"## {license_type} ({len(records)} asset(s))")
        lines.append("")
        for r in records:
            lines.append(f"- **{r['filename']}**")
            lines.append(f"  - Source: {r['source_url']}")
            if r["attribution_text"]:
                lines.append(f"  - Credit: {r['attribution_text']}")
        lines.append("")

    return "\n".join(lines)
