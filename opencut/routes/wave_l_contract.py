"""Shared Flask contract for the decomposed Wave L compatibility surface."""

from flask import Blueprint

from opencut.errors import error_response

wave_l_bp = Blueprint("wave_l", "opencut.routes.wave_l_routes")


def _stub_503(name: str, hint: str = "") -> tuple:
    return error_response(
        "DEPENDENCY_NOT_INSTALLED",
        f"{name} dependency is not installed or not configured.",
        status=503,
        suggestion=hint or "Check the module's INSTALL_HINT.",
    )
