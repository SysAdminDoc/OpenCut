"""
OpenCut Deliverables Routes

VFX sheet, ADR list, music cue sheet, asset list generation.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import (
    require_csrf,
    validate_path,
)

logger = logging.getLogger("opencut")

deliverables_bp = Blueprint("deliverables", __name__)


def _resolve_deliverable_output_dir(output_dir: str, fallback_dir: str = None) -> str:
    """Resolve and create the output directory for deliverables."""
    if output_dir:
        try:
            resolved = validate_path(output_dir)
        except ValueError as exc:
            raise ValueError(str(exc))
        os.makedirs(resolved, exist_ok=True)
        return resolved
    if fallback_dir:
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir
    import tempfile
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Deliverables: VFX Sheet
# ---------------------------------------------------------------------------
@deliverables_bp.route("/deliverables/vfx-sheet", methods=["POST"])
@require_csrf
def deliverables_vfx_sheet():
    """Generate a VFX shot sheet from sequence data."""
    data = request.get_json(force=True)
    sequence_data = data.get("sequence_data", {})
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if not sequence_data:
        return jsonify({"error": "sequence_data is required"}), 400

    try:
        resolved_dir = _resolve_deliverable_output_dir(output_dir)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    output_path = os.path.join(resolved_dir, "vfx_sheet.csv")

    try:
        from opencut.core import deliverables
        result = deliverables.generate_vfx_sheet(sequence_data, output_path)
        rows = result.get("rows", 0) if isinstance(result, dict) else 0
        out = result.get("output", output_path) if isinstance(result, dict) else output_path
        return jsonify({"output": out, "rows": rows})
    except ImportError:
        return jsonify({"error": "deliverables module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "deliverables")


# ---------------------------------------------------------------------------
# Deliverables: ADR List
# ---------------------------------------------------------------------------
@deliverables_bp.route("/deliverables/adr-list", methods=["POST"])
@require_csrf
def deliverables_adr_list():
    """Generate an ADR (Additional Dialogue Recording) list from sequence data."""
    data = request.get_json(force=True)
    sequence_data = data.get("sequence_data", {})
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if not sequence_data:
        return jsonify({"error": "sequence_data is required"}), 400

    try:
        resolved_dir = _resolve_deliverable_output_dir(output_dir)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    output_path = os.path.join(resolved_dir, "adr_list.csv")

    try:
        from opencut.core import deliverables
        result = deliverables.generate_adr_list(sequence_data, output_path)
        rows = result.get("rows", 0) if isinstance(result, dict) else 0
        out = result.get("output", output_path) if isinstance(result, dict) else output_path
        return jsonify({"output": out, "rows": rows})
    except ImportError:
        return jsonify({"error": "deliverables module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "deliverables")


# ---------------------------------------------------------------------------
# Deliverables: Music Cue Sheet
# ---------------------------------------------------------------------------
@deliverables_bp.route("/deliverables/music-cue-sheet", methods=["POST"])
@require_csrf
def deliverables_music_cue_sheet():
    """Generate a music cue sheet from sequence data."""
    data = request.get_json(force=True)
    sequence_data = data.get("sequence_data", {})
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if not sequence_data:
        return jsonify({"error": "sequence_data is required"}), 400

    try:
        resolved_dir = _resolve_deliverable_output_dir(output_dir)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    output_path = os.path.join(resolved_dir, "music_cue_sheet.csv")

    try:
        from opencut.core import deliverables
        result = deliverables.generate_music_cue_sheet(sequence_data, output_path)
        rows = result.get("rows", 0) if isinstance(result, dict) else 0
        out = result.get("output", output_path) if isinstance(result, dict) else output_path
        return jsonify({"output": out, "rows": rows})
    except ImportError:
        return jsonify({"error": "deliverables module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "deliverables")


# ---------------------------------------------------------------------------
# Deliverables: Asset List
# ---------------------------------------------------------------------------
@deliverables_bp.route("/deliverables/asset-list", methods=["POST"])
@require_csrf
def deliverables_asset_list():
    """Generate an asset list from sequence data."""
    data = request.get_json(force=True)
    sequence_data = data.get("sequence_data", {})
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if not sequence_data:
        return jsonify({"error": "sequence_data is required"}), 400

    try:
        resolved_dir = _resolve_deliverable_output_dir(output_dir)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    output_path = os.path.join(resolved_dir, "asset_list.csv")

    try:
        from opencut.core import deliverables
        result = deliverables.generate_asset_list(sequence_data, output_path)
        rows = result.get("rows", 0) if isinstance(result, dict) else 0
        out = result.get("output", output_path) if isinstance(result, dict) else output_path
        return jsonify({"output": out, "rows": rows})
    except ImportError:
        return jsonify({"error": "deliverables module not available"}), 503
    except Exception as exc:
        return safe_error(exc, "deliverables")
