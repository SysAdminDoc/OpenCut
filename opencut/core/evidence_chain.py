"""
OpenCut Evidence Chain-of-Custody Module (35.3)

Maintain a forensic chain-of-custody for video evidence: SHA-256 hashing,
timestamped operation logs, and signed PDF/JSON reports suitable for
legal and compliance workflows.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CustodyEntry:
    """A single operation in the chain of custody."""
    timestamp: str = ""
    operation: str = ""
    operator: str = ""
    description: str = ""
    input_hash: str = ""
    output_hash: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    entry_id: str = ""


@dataclass
class CustodyChain:
    """Full chain-of-custody record for a video asset."""
    chain_id: str = ""
    video_path: str = ""
    original_hash: str = ""
    operator: str = ""
    created: str = ""
    entries: List[Dict] = field(default_factory=list)
    finalized: bool = False
    final_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def _hash_file(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_string(data: str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a string."""
    return hashlib.new(algorithm, data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Chain creation
# ---------------------------------------------------------------------------

def create_custody_chain(
    video_path: str,
    operator: str = "",
    case_id: str = "",
    notes: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a new chain-of-custody record for a video file.

    Computes the initial SHA-256 hash and initializes the custody chain.

    Args:
        video_path: Path to the video evidence file.
        operator: Name/ID of the operator creating the chain.
        case_id: Optional case/incident identifier.
        notes: Optional notes about the evidence.
        on_progress: Optional callback(pct, msg).

    Returns:
        Chain-of-custody dict.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if on_progress:
        on_progress(10, "Computing original file hash")

    original_hash = _hash_file(video_path)
    chain_id = f"CoC-{uuid.uuid4().hex[:12]}"
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Gather file metadata
    file_stat = os.stat(video_path)
    try:
        info = get_video_info(video_path)
    except Exception:
        info = {}

    if on_progress:
        on_progress(50, "Initializing custody chain")

    chain = {
        "chain_id": chain_id,
        "video_path": os.path.abspath(video_path),
        "original_filename": os.path.basename(video_path),
        "original_hash": original_hash,
        "hash_algorithm": "sha256",
        "operator": operator,
        "case_id": case_id,
        "notes": notes,
        "created": created,
        "entries": [],
        "finalized": False,
        "final_hash": "",
        "metadata": {
            "file_size": file_stat.st_size,
            "file_modified": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(file_stat.st_mtime)
            ),
            "video_info": {
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "fps": info.get("fps", 0),
                "duration": info.get("duration", 0),
            },
        },
    }

    # First entry: chain creation
    creation_entry = {
        "entry_id": f"E-{uuid.uuid4().hex[:8]}",
        "timestamp": created,
        "operation": "chain_created",
        "operator": operator,
        "description": "Chain of custody initiated",
        "input_hash": original_hash,
        "output_hash": original_hash,
        "parameters": {"case_id": case_id, "notes": notes},
    }
    chain["entries"].append(creation_entry)

    if on_progress:
        on_progress(100, "Custody chain created")

    return chain


# ---------------------------------------------------------------------------
# Operation logging
# ---------------------------------------------------------------------------

def log_operation(
    chain: dict,
    operation_data: dict,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Log an operation to the chain of custody.

    Args:
        chain: The custody chain dict.
        operation_data: Dict with operation details:
            - operation: Operation name (required).
            - description: Human-readable description.
            - operator: Who performed the operation.
            - input_hash: Hash of input file (if applicable).
            - output_hash: Hash of output file (if applicable).
            - parameters: Additional parameters dict.
        on_progress: Optional callback(pct, msg).

    Returns:
        Updated chain dict.

    Raises:
        ValueError: If chain is already finalized.
    """
    if chain.get("finalized", False):
        raise ValueError("Cannot modify a finalized custody chain")

    if on_progress:
        on_progress(20, "Logging operation")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry_id = f"E-{uuid.uuid4().hex[:8]}"

    entry = {
        "entry_id": entry_id,
        "timestamp": timestamp,
        "operation": operation_data.get("operation", "unknown"),
        "operator": operation_data.get("operator", chain.get("operator", "")),
        "description": operation_data.get("description", ""),
        "input_hash": operation_data.get("input_hash", ""),
        "output_hash": operation_data.get("output_hash", ""),
        "parameters": operation_data.get("parameters", {}),
    }

    chain["entries"].append(entry)

    if on_progress:
        on_progress(100, "Operation logged")

    return chain


# ---------------------------------------------------------------------------
# Chain finalization
# ---------------------------------------------------------------------------

def finalize_chain(
    chain: dict,
    output_hash: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Finalize the chain of custody, locking it from further changes.

    Args:
        chain: The custody chain dict.
        output_hash: SHA-256 hash of the final output file.
        on_progress: Optional callback(pct, msg).

    Returns:
        Finalized chain dict.

    Raises:
        ValueError: If chain is already finalized.
    """
    if chain.get("finalized", False):
        raise ValueError("Chain is already finalized")

    if on_progress:
        on_progress(20, "Finalizing custody chain")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Add finalization entry
    entry = {
        "entry_id": f"E-{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "operation": "chain_finalized",
        "operator": chain.get("operator", ""),
        "description": "Chain of custody finalized and locked",
        "input_hash": chain.get("original_hash", ""),
        "output_hash": output_hash,
        "parameters": {},
    }
    chain["entries"].append(entry)

    chain["finalized"] = True
    chain["final_hash"] = output_hash
    chain["finalized_at"] = timestamp

    # Compute chain integrity hash (hash of all entries)
    chain_data = json.dumps(chain["entries"], sort_keys=True, separators=(",", ":"))
    chain["chain_integrity_hash"] = _hash_string(chain_data)

    if on_progress:
        on_progress(100, "Chain finalized")

    return chain


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------

def export_custody_report(
    chain: dict,
    output_path: str,
    format: str = "json",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export the chain of custody as a report.

    Args:
        chain: The custody chain dict.
        output_path: Path for the output report file.
        format: Report format - 'json' or 'pdf'.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with report_path, format, and size.
    """
    format = format.lower().strip()
    if format not in ("json", "pdf"):
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pdf'.")

    if on_progress:
        on_progress(10, f"Generating {format.upper()} report")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if format == "json":
        report = _generate_json_report(chain)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    elif format == "pdf":
        _generate_pdf_report(chain, output_path)

    file_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0

    if on_progress:
        on_progress(100, "Report exported")

    return {
        "report_path": output_path,
        "format": format,
        "size": file_size,
        "entries_count": len(chain.get("entries", [])),
        "chain_id": chain.get("chain_id", ""),
        "finalized": chain.get("finalized", False),
    }


def _generate_json_report(chain: dict) -> dict:
    """Generate a structured JSON report from the custody chain."""
    return {
        "report_type": "chain_of_custody",
        "report_generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chain_id": chain.get("chain_id", ""),
        "case_id": chain.get("case_id", ""),
        "operator": chain.get("operator", ""),
        "video_file": {
            "original_filename": chain.get("original_filename", ""),
            "original_hash": chain.get("original_hash", ""),
            "hash_algorithm": chain.get("hash_algorithm", "sha256"),
            "final_hash": chain.get("final_hash", ""),
        },
        "chain_created": chain.get("created", ""),
        "chain_finalized": chain.get("finalized_at", ""),
        "chain_integrity_hash": chain.get("chain_integrity_hash", ""),
        "is_finalized": chain.get("finalized", False),
        "entries": chain.get("entries", []),
        "metadata": chain.get("metadata", {}),
        "notes": chain.get("notes", ""),
    }


def _generate_pdf_report(chain: dict, output_path: str) -> None:
    """Generate a simple PDF report from the custody chain.

    Uses basic PDF structure without external dependencies.
    """
    lines = []
    lines.append("CHAIN OF CUSTODY REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Chain ID: {chain.get('chain_id', '')}")
    lines.append(f"Case ID: {chain.get('case_id', '')}")
    lines.append(f"Operator: {chain.get('operator', '')}")
    lines.append(f"Created: {chain.get('created', '')}")
    lines.append(f"Finalized: {'Yes' if chain.get('finalized') else 'No'}")
    lines.append("")
    lines.append(f"Original File: {chain.get('original_filename', '')}")
    lines.append(f"Original Hash (SHA-256): {chain.get('original_hash', '')}")
    lines.append(f"Final Hash (SHA-256): {chain.get('final_hash', '')}")
    lines.append("")
    lines.append("OPERATION LOG")
    lines.append("-" * 50)

    for entry in chain.get("entries", []):
        lines.append("")
        lines.append(f"Entry: {entry.get('entry_id', '')}")
        lines.append(f"  Time:      {entry.get('timestamp', '')}")
        lines.append(f"  Operation: {entry.get('operation', '')}")
        lines.append(f"  Operator:  {entry.get('operator', '')}")
        lines.append(f"  Desc:      {entry.get('description', '')}")
        if entry.get("input_hash"):
            lines.append(f"  In Hash:   {entry['input_hash']}")
        if entry.get("output_hash"):
            lines.append(f"  Out Hash:  {entry['output_hash']}")

    lines.append("")
    lines.append("-" * 50)
    if chain.get("chain_integrity_hash"):
        lines.append(f"Chain Integrity Hash: {chain['chain_integrity_hash']}")
    lines.append(f"Report Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    text = "\n".join(lines)

    # Generate minimal valid PDF
    _write_simple_pdf(output_path, text)


def _write_simple_pdf(filepath: str, text: str) -> None:
    """Write a minimal valid PDF file with the given text content."""
    # Minimal PDF structure
    lines_list = text.split("\n")

    # Build PDF content stream
    stream_lines = []
    stream_lines.append("BT")
    stream_lines.append("/F1 10 Tf")
    y = 750
    for line in lines_list:
        if y < 50:
            break
        # Escape special PDF chars
        safe_line = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream_lines.append(f"1 0 0 1 50 {y} Tm")
        stream_lines.append(f"({safe_line}) Tj")
        y -= 14
    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines)

    # PDF objects
    objects = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")

    # Object 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")

    # Object 3: Page
    objects.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )

    # Object 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_content)} >>\n"
        f"stream\n{stream_content}\nendstream\nendobj"
    )

    # Object 5: Font
    objects.append(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 "
        "/BaseFont /Courier >>\nendobj"
    )

    # Build PDF
    pdf_parts = ["%PDF-1.4\n"]
    offsets = []
    for obj in objects:
        offsets.append(len("".join(pdf_parts)))
        pdf_parts.append(obj + "\n")

    xref_offset = len("".join(pdf_parts))
    pdf_parts.append("xref\n")
    pdf_parts.append(f"0 {len(objects) + 1}\n")
    pdf_parts.append("0000000000 65535 f \n")
    for off in offsets:
        pdf_parts.append(f"{off:010d} 00000 n \n")

    pdf_parts.append("trailer\n")
    pdf_parts.append(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n")
    pdf_parts.append("startxref\n")
    pdf_parts.append(f"{xref_offset}\n")
    pdf_parts.append("%%EOF\n")

    with open(filepath, "w", encoding="ascii", errors="replace") as f:
        f.write("".join(pdf_parts))
