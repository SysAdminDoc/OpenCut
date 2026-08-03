"""Shared response helpers for route modules."""

from flask import jsonify

from opencut.errors import error_response
from opencut.security import get_json_dict


def _json_object_or_400():
    """Parse a route body and return a structured 400 response on failure."""
    try:
        return get_json_dict(), None
    except ValueError as exc:
        return None, (
            jsonify({
                "error": str(exc),
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }),
            400,
        )


def _stub_503(
    name: str,
    hint: str = "",
    *,
    code: str = "DEPENDENCY_NOT_INSTALLED",
    message: str = "",
    default_hint: str = "Check the module's INSTALL_HINT.",
) -> tuple:
    """Return the shared 503 response for an unavailable optional backend."""
    return error_response(
        code,
        message or f"{name} dependency is not installed or not configured.",
        status=503,
        suggestion=hint or default_hint,
    )
