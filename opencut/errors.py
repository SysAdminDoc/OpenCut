"""
OpenCut Error Taxonomy

Structured error codes so the frontend can show targeted help
instead of generic "Unknown error" messages.
"""

from flask import jsonify


class OpenCutError(Exception):
    """Application error with a machine-readable code and HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)

    def to_response(self):
        return jsonify({"error": self.message, "code": self.code}), self.status


# --- Predefined error constructors ---

def missing_dependency(name: str) -> OpenCutError:
    """Raised when an optional dependency is not installed."""
    return OpenCutError(
        code="MISSING_DEPENDENCY",
        message=f"{name} is not installed. Install it from the Settings tab.",
        status=400,
    )


def file_not_found(path: str) -> OpenCutError:
    return OpenCutError(
        code="FILE_NOT_FOUND",
        message=f"File not found: {path}",
        status=404,
    )


def gpu_out_of_memory() -> OpenCutError:
    return OpenCutError(
        code="GPU_OUT_OF_MEMORY",
        message="GPU ran out of memory. Try reducing file size or switching to CPU mode.",
        status=503,
    )


def invalid_input(detail: str) -> OpenCutError:
    return OpenCutError(
        code="INVALID_INPUT",
        message=f"Invalid input: {detail}",
        status=400,
    )


def operation_failed(detail: str) -> OpenCutError:
    return OpenCutError(
        code="OPERATION_FAILED",
        message=detail,
        status=500,
    )


def register_error_handlers(app):
    """Register the OpenCutError handler on a Flask app."""

    @app.errorhandler(OpenCutError)
    def handle_opencut_error(e):
        return e.to_response()
