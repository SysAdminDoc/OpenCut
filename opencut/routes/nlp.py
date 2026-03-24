"""
OpenCut NLP Routes

Natural language command parsing — converts plain-text commands into
route + params for the frontend to execute.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _safe_error
from opencut.security import require_csrf

try:
    from ..core.llm import LLMConfig
except ImportError:
    try:
        from opencut.core.llm import LLMConfig
    except ImportError:
        LLMConfig = None

logger = logging.getLogger("opencut")

nlp_bp = Blueprint("nlp", __name__)


# ---------------------------------------------------------------------------
# NLP: Parse Command
# ---------------------------------------------------------------------------
@nlp_bp.route("/nlp/command", methods=["POST"])
@require_csrf
def nlp_command():
    """Parse a natural-language command and return the matching route + params."""
    data = request.get_json(force=True)
    command = data.get("command", "").strip()
    file_path = data.get("filepath", data.get("file", "")).strip()
    llm_provider = data.get("llm_provider", "ollama").strip()
    if llm_provider not in ("ollama", "openai", "anthropic"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "llama3").strip()
    api_key = data.get("api_key", "").strip()

    if not command:
        return jsonify({"error": "command must not be empty"}), 400

    if len(command) > 2000:
        return jsonify({"error": "command too long (max 2000 characters)"}), 400

    # Validate optional file path
    if file_path:
        from opencut.security import validate_filepath
        try:
            file_path = validate_filepath(file_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    llm_config = None
    if LLMConfig is not None:
        llm_config = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key or "",
        )

    try:
        from opencut.core import nlp_command as _nlp_mod
        parsed = _nlp_mod.parse_command(command, llm_config=llm_config)
        if not parsed:
            return jsonify({"route": "", "params": {}, "confidence": 0.0, "explanation": "No matching command found"})
        return jsonify({
            "route": parsed.get("route", ""),
            "params": parsed.get("params", {}),
            "confidence": float(parsed.get("confidence", 0.0)),
            "explanation": str(parsed.get("explanation", parsed.get("param_source", ""))),
        })
    except ImportError:
        return jsonify({"error": "nlp_command module not available"}), 503
    except Exception as exc:
        return _safe_error(exc)
