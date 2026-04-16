"""
OpenCut Error Taxonomy

Structured error codes so the frontend can show targeted help
instead of generic "Unknown error" messages.

Every error carries:
  - code:       machine-readable identifier (e.g. "GPU_OUT_OF_MEMORY")
  - message:    user-facing summary
  - suggestion: recovery action the user can take
  - status:     HTTP status code
"""

import logging
import os

from flask import jsonify, request
from werkzeug.exceptions import BadRequest, MethodNotAllowed, NotFound

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class OpenCutError(Exception):
    """Application error with a machine-readable code and HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400,
                 suggestion: str = ""):
        self.code = code
        self.message = message
        self.status = status
        self.suggestion = suggestion
        super().__init__(message)

    def to_response(self):
        body = {
            "error": self.message,
            "code": self.code,
        }
        if self.suggestion:
            body["suggestion"] = self.suggestion
        return jsonify(body), self.status


# ---------------------------------------------------------------------------
# Structured error_response helper — drop-in replacement for ad-hoc dicts
# ---------------------------------------------------------------------------

def error_response(code: str, message: str, status: int = 400,
                   suggestion: str = "", detail: str = ""):
    """Return a JSON error response tuple ready to be returned from a route.

    Usage::

        return error_response("FILE_NOT_FOUND", "File not found",
                              status=404, suggestion="Check the file path.")
    """
    body = {
        "error": message,
        "code": code,
    }
    if suggestion:
        body["suggestion"] = suggestion
    if detail:
        body["detail"] = detail
    return jsonify(body), status


# ---------------------------------------------------------------------------
# safe_error — replaces _safe_error from jobs.py
# ---------------------------------------------------------------------------

def safe_error(exc, context=""):
    """Log the real exception and return a structured error response.

    Classifies common exception types into specific error codes so the
    frontend can show targeted guidance instead of "Unknown error".
    """
    msg = str(exc)
    lower = msg.lower()

    # Classify the exception
    code = "INTERNAL_ERROR"
    user_msg = "An internal error occurred."
    suggestion = "Check the server logs for details."
    status = 500

    if isinstance(exc, OpenCutError):
        return exc.to_response()

    if isinstance(exc, MemoryError) or "out of memory" in lower or "cuda out of memory" in lower or "cuda error: out of memory" in lower:
        code = "GPU_OUT_OF_MEMORY"
        user_msg = "Ran out of memory during processing."
        suggestion = "Try a shorter clip, lower quality setting, or switch to CPU mode in Settings."
        status = 503
    elif isinstance(exc, TimeoutError) or "timed out" in lower or "timeout" in lower:
        code = "OPERATION_TIMEOUT"
        user_msg = "The operation took too long and was stopped."
        suggestion = "Try a shorter clip or simpler settings."
        status = 504
    elif isinstance(exc, PermissionError) or "permission denied" in lower or "errno 13" in lower:
        code = "PERMISSION_DENIED"
        user_msg = "Permission denied when accessing a file."
        suggestion = "Check that the file is not locked by another program and that you have write access."
        status = 403
    elif isinstance(exc, ImportError) or "no module named" in lower or "not installed" in lower:
        code = "MISSING_DEPENDENCY"
        user_msg = "A required package is not installed."
        suggestion = "Install the missing package from the Settings tab."
        status = 503
    elif isinstance(exc, FileNotFoundError) or "no such file" in lower or "file not found" in lower:
        code = "FILE_NOT_FOUND"
        user_msg = "A required file was not found."
        suggestion = "Check that the file has not been moved or deleted."
        status = 404
    elif ("unsupported" in lower and ("format" in lower or "codec" in lower)):
        code = "UNSUPPORTED_FORMAT"
        user_msg = "Unsupported media format or codec."
        suggestion = "Try converting the file to a standard format (MP4/H.264 for video, WAV/MP3 for audio)."
        status = 400
    elif isinstance(exc, RuntimeError) and ("ffmpeg" in lower or "ffprobe" in lower):
        code = "FFMPEG_ERROR"
        user_msg = "FFmpeg encountered an error during processing."
        suggestion = "Check that FFmpeg is installed and the input file is not corrupt."
        status = 500

    log_ctx = f" in {context}" if context else ""
    logger.exception("Error [%s]%s: %s", code, log_ctx, exc)

    return error_response(code, user_msg, status=status,
                          suggestion=suggestion)


# ---------------------------------------------------------------------------
# Predefined error constructors — raise these from routes
# ---------------------------------------------------------------------------

def missing_dependency(name: str) -> OpenCutError:
    """Raised when an optional dependency is not installed."""
    return OpenCutError(
        code="MISSING_DEPENDENCY",
        message=f"{name} is not installed.",
        suggestion="Install it from the Settings tab under Dependencies.",
        status=503,
    )


def file_not_found(path: str) -> OpenCutError:
    return OpenCutError(
        code="FILE_NOT_FOUND",
        message=f"File not found: {os.path.basename(path) if path else 'unknown'}",
        suggestion="Check that the file has not been moved or deleted.",
        status=404,
    )


def gpu_out_of_memory() -> OpenCutError:
    return OpenCutError(
        code="GPU_OUT_OF_MEMORY",
        message="GPU ran out of memory.",
        suggestion="Try a shorter clip, lower quality setting, or switch to CPU mode in Settings.",
        status=503,
    )


def invalid_input(detail: str) -> OpenCutError:
    return OpenCutError(
        code="INVALID_INPUT",
        message=f"Invalid input: {detail}",
        suggestion="Check your settings and try again.",
        status=400,
    )


def invalid_model(model: str, allowed=None) -> OpenCutError:
    msg = f"Unknown model: {model}"
    hint = "Select a valid model from the dropdown."
    if allowed:
        hint = f"Valid models: {', '.join(sorted(allowed))}"
    return OpenCutError(
        code="INVALID_MODEL",
        message=msg,
        suggestion=hint,
        status=400,
    )


def operation_failed(detail: str) -> OpenCutError:
    return OpenCutError(
        code="OPERATION_FAILED",
        message=detail,
        suggestion="Check the server logs for details.",
        status=500,
    )


def rate_limited(operation: str = "") -> OpenCutError:
    ctx = f" ({operation})" if operation else ""
    return OpenCutError(
        code="RATE_LIMITED",
        message=f"Another operation{ctx} is already running.",
        suggestion="Wait for the current operation to finish, or cancel it first.",
        status=429,
    )


def queue_full(max_size: int = 100) -> OpenCutError:
    return OpenCutError(
        code="QUEUE_FULL",
        message=f"Job queue is full (max {max_size}).",
        suggestion="Wait for some jobs to finish before adding more.",
        status=429,
    )


def module_not_available(name: str) -> OpenCutError:
    return OpenCutError(
        code="MODULE_NOT_AVAILABLE",
        message=f"The {name} module is not available.",
        suggestion="Install it from the Settings tab under Dependencies.",
        status=503,
    )


def file_permission_denied(path: str = "") -> OpenCutError:
    ctx = f": {path}" if path else ""
    return OpenCutError(
        code="PERMISSION_DENIED",
        message=f"Permission denied{ctx}.",
        suggestion="Check that the file is not locked by another program and that you have write access.",
        status=403,
    )


def too_many_items(item: str, max_count: int) -> OpenCutError:
    return OpenCutError(
        code="TOO_MANY_ITEMS",
        message=f"Too many {item} (max {max_count}).",
        suggestion=f"Reduce the number of {item} and try again.",
        status=400,
    )


def server_busy() -> OpenCutError:
    return OpenCutError(
        code="SERVER_BUSY",
        message="The server is busy with too many concurrent jobs.",
        suggestion="Wait for a job to finish or cancel one from the processing bar.",
        status=429,
    )


def install_failed(package: str, detail: str = "") -> OpenCutError:
    msg = f"Failed to install {package}."
    if detail:
        msg += f" {detail}"
    return OpenCutError(
        code="INSTALL_FAILED",
        message=msg,
        suggestion="Try running the installer as administrator, or install manually via pip.",
        status=500,
    )


# ---------------------------------------------------------------------------
# Flask integration
# ---------------------------------------------------------------------------

def register_error_handlers(app):
    """Register the OpenCutError handler on a Flask app."""

    from opencut.jobs import TooManyJobsError

    @app.errorhandler(OpenCutError)
    def handle_opencut_error(e):
        return e.to_response()

    @app.errorhandler(TooManyJobsError)
    def handle_too_many_jobs(e):
        return jsonify({
            "error": str(e),
            "code": "TOO_MANY_JOBS",
            "suggestion": "Wait for a job to finish or cancel one from the processing bar.",
        }), 429

    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        description = getattr(e, "description", "") or ""
        if "JSON body must be an object" in description:
            return jsonify({
                "error": "JSON body must be an object.",
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }), 400
        if request.is_json or request.mimetype == "application/json":
            return jsonify({
                "error": "Invalid JSON request body.",
                "code": "INVALID_JSON",
                "suggestion": "Fix malformed JSON or send a top-level JSON object, then retry.",
            }), 400
        return jsonify({
            "error": "Bad request.",
            "code": "BAD_REQUEST",
            "suggestion": getattr(e, "description", "") or "Check the request parameters and try again.",
        }), 400

    @app.errorhandler(NotFound)
    def handle_not_found(e):
        return jsonify({
            "error": "Endpoint not found.",
            "code": "NOT_FOUND",
            "suggestion": "Check the URL or update the client to a supported endpoint.",
        }), 404

    @app.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(e):
        return jsonify({
            "error": "Method not allowed for this endpoint.",
            "code": "METHOD_NOT_ALLOWED",
            "suggestion": "Use the HTTP method documented for this endpoint.",
        }), 405
