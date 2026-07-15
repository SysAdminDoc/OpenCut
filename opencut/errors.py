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

from flask import g, has_request_context, jsonify, request
from werkzeug.exceptions import BadRequest, MethodNotAllowed, NotFound

logger = logging.getLogger("opencut")


def _current_request_id() -> str:
    """Return the active request ID when request-correlation middleware is present."""
    try:
        if has_request_context():
            rid = getattr(g, "request_id", "") or ""
            if rid:
                return str(rid)
    except RuntimeError:
        pass
    try:
        from opencut.core.request_correlation import get_request_id

        return get_request_id()
    except Exception:  # noqa: BLE001 - error rendering must stay best-effort
        return ""


def _attach_request_id(body: dict) -> dict:
    request_id = _current_request_id()
    if not request_id or body.get("request_id"):
        return body
    enriched = dict(body)
    enriched["request_id"] = request_id
    return enriched


def _request_log_fields(context: str = "") -> dict:
    fields = {
        "error_context": context,
        "request_id": _current_request_id(),
        "request_method": "",
        "request_path": "",
    }
    try:
        if has_request_context():
            fields["request_method"] = request.method
            fields["request_path"] = request.path
    except RuntimeError:
        pass
    return fields


def _log_typed_error_response(code: str, message: str, status: int, context: str = "") -> None:
    extra = _request_log_fields(context)
    extra.update(
        {
            "error_code": code,
            "error_status": status,
        }
    )
    level = logging.ERROR if status >= 500 else logging.WARNING
    log_ctx = f" in {context}" if context else ""
    logger.log(
        level,
        "Typed error response [%s]%s status=%s request_id=%s method=%s path=%s: %s",
        code,
        log_ctx,
        status,
        extra["request_id"] or "-",
        extra["request_method"] or "-",
        extra["request_path"] or "-",
        message,
        extra=extra,
    )


def _log_open_cut_error(exc: "OpenCutError", context: str = "") -> None:
    extra = _request_log_fields(context)
    extra.update(
        {
            "error_code": exc.code,
            "error_status": exc.status,
        }
    )
    level = logging.ERROR if exc.status >= 500 else logging.WARNING
    log_ctx = f" in {context}" if context else ""
    logger.log(level, "OpenCutError [%s]%s: %s", exc.code, log_ctx, exc.message, extra=extra)


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

    def to_response(self, *, log: bool = True, context: str = ""):
        body = {
            "error": self.message,
            "code": self.code,
        }
        if self.suggestion:
            body["suggestion"] = self.suggestion
        if log:
            _log_open_cut_error(self, context)
        return jsonify(_attach_request_id(body)), self.status


# ---------------------------------------------------------------------------
# Structured error_response helper — drop-in replacement for ad-hoc dicts
# ---------------------------------------------------------------------------

def error_response(code: str, message: str, status: int = 400,
                   suggestion: str = "", detail: str = "", log: bool = True):
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
    if log:
        _log_typed_error_response(code, message, status)
    return jsonify(_attach_request_id(body)), status


# ---------------------------------------------------------------------------
# safe_error — replaces _safe_error from jobs.py
# ---------------------------------------------------------------------------

def _missing_dependency_suggestion(exc, context: str, fallback: str) -> str:
    try:
        from opencut.core.install_hints import suggestion_for_exception

        return suggestion_for_exception(exc, context=context) or fallback
    except Exception:  # noqa: BLE001 - error handling must never fail closed
        return fallback


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

    if getattr(exc, "code", "") == "LOCAL_ONLY_NETWORK_BLOCKED":
        return error_response(
            "LOCAL_ONLY_NETWORK_BLOCKED",
            str(exc),
            status=getattr(exc, "status", 403),
            suggestion=getattr(exc, "suggestion", ""),
        )

    if getattr(exc, "code", "") in {
        "CREDENTIAL_STORE_UNAVAILABLE",
        "CREDENTIAL_STORE_WRITE_FAILED",
        "REMOTE_AUTH_TOKEN_FILE_INVALID",
        "REMOTE_AUTH_TOKEN_FILE_READ_ONLY",
        "PLUGIN_WORKER_UNAVAILABLE",
        "PLUGIN_WORKER_QUARANTINED",
    }:
        return error_response(
            getattr(exc, "code"),
            str(exc),
            status=getattr(exc, "status", 503),
            suggestion=getattr(exc, "suggestion", ""),
        )

    if isinstance(exc, OpenCutError):
        return exc.to_response(context=context)

    if getattr(exc, "code", "") == "GPU_BUSY":
        response, status_code = error_response(
            "GPU_BUSY",
            str(exc),
            status=getattr(exc, "status_code", 429) or 429,
            suggestion="Wait for the active GPU job to finish, then retry.",
        )
        retry_after = int(getattr(exc, "retry_after", 1) or 1)
        response.headers["Retry-After"] = str(max(1, retry_after))
        return response, status_code

    if isinstance(exc, MemoryError) or "out of memory" in lower or "cuda out of memory" in lower or "cuda error: out of memory" in lower:
        code = "GPU_OUT_OF_MEMORY"
        user_msg = "Ran out of memory during processing."
        suggestion = "Try a shorter clip, lower quality setting, or switch to CPU mode in Settings."
        status = 503
    # TimeoutError instances classify directly; string matching intentionally
    # uses the specific "timed out" / "timeouterror" / subprocess phrases so
    # free-form text like "adjust the timeout setting" doesn't misclassify an
    # unrelated error as a 504.
    elif isinstance(exc, TimeoutError) or "timed out" in lower or "timeouterror" in lower or "operation timed out" in lower:
        code = "OPERATION_TIMEOUT"
        user_msg = "The operation took too long and was stopped."
        suggestion = "Try a shorter clip or simpler settings."
        status = 504
    elif isinstance(exc, PermissionError) or "permission denied" in lower or "errno 13" in lower:
        code = "PERMISSION_DENIED"
        user_msg = "Permission denied when accessing a file."
        suggestion = "Check that the file is not locked by another program and that you have write access."
        status = 403
    elif (
        isinstance(exc, ImportError)
        or "no module named" in lower
        or "not installed" in lower
        or "missing package" in lower
        or "dependencies not installed" in lower
    ):
        code = "MISSING_DEPENDENCY"
        user_msg = "A required package is not installed."
        suggestion = _missing_dependency_suggestion(
            exc,
            context,
            "Install the missing package from the Settings tab.",
        )
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
                          suggestion=suggestion, log=False)


# ---------------------------------------------------------------------------
# Predefined error constructors — raise these from routes
# ---------------------------------------------------------------------------

def missing_dependency(
    name: str,
    *,
    extra: str = "",
    gpu: bool = False,
    vram_mb: int = 0,
) -> OpenCutError:
    """Raised when an optional dependency is not installed."""
    try:
        from opencut.core.install_hints import build_install_suggestion

        suggestion = build_install_suggestion(
            name,
            extra=extra or None,
            gpu=gpu,
            vram_mb=vram_mb,
        )
    except Exception:  # noqa: BLE001 - preserve existing fallback behavior
        suggestion = "Install it from the Settings tab under Dependencies."
    return OpenCutError(
        code="MISSING_DEPENDENCY",
        message=f"{name} is not installed.",
        suggestion=suggestion,
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

    from opencut.credential_store import CredentialStoreError
    from opencut.jobs import TooManyJobsError
    from opencut.network_policy import LocalOnlyNetworkError

    @app.errorhandler(CredentialStoreError)
    def handle_credential_store_error(e):
        return error_response(
            e.code,
            str(e),
            status=e.status,
            suggestion=e.suggestion,
        )

    @app.errorhandler(LocalOnlyNetworkError)
    def handle_local_only_network_error(e):
        return error_response(
            e.code,
            str(e),
            status=e.status,
            suggestion=e.suggestion,
        )

    @app.errorhandler(OpenCutError)
    def handle_opencut_error(e):
        return e.to_response(context="open_cut_error_handler")

    @app.errorhandler(TooManyJobsError)
    def handle_too_many_jobs(e):
        return error_response(
            "TOO_MANY_JOBS",
            str(e),
            status=429,
            suggestion="Wait for a job to finish or cancel one from the processing bar.",
        )

    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        description = getattr(e, "description", "") or ""
        if "JSON body must be an object" in description:
            return error_response(
                "INVALID_INPUT",
                "JSON body must be an object.",
                status=400,
                suggestion="Send a top-level JSON object in the request body.",
            )
        if request.is_json or request.mimetype == "application/json":
            return error_response(
                "INVALID_JSON",
                "Invalid JSON request body.",
                status=400,
                suggestion="Fix malformed JSON or send a top-level JSON object, then retry.",
            )
        return error_response(
            "BAD_REQUEST",
            "Bad request.",
            status=400,
            suggestion=getattr(e, "description", "") or "Check the request parameters and try again.",
        )

    @app.errorhandler(NotFound)
    def handle_not_found(e):
        return error_response(
            "NOT_FOUND",
            "Endpoint not found.",
            status=404,
            suggestion="Check the URL or update the client to a supported endpoint.",
        )

    @app.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(e):
        return error_response(
            "METHOD_NOT_ALLOWED",
            "Method not allowed for this endpoint.",
            status=405,
            suggestion="Use the HTTP method documented for this endpoint.",
        )
