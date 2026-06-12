"""RA-03 guards for structured typed-error logging."""

from __future__ import annotations

import logging

from flask import Blueprint


def _typed_error_record(records, code: str):
    for record in records:
        if getattr(record, "error_code", "") == code:
            return record
    raise AssertionError(f"missing typed error log for {code}")


def test_open_cut_error_handler_logs_structured_context(app, caplog):
    from opencut.errors import invalid_input

    bp = Blueprint("ra03_open_cut_error", __name__)

    @bp.route("/ra03/open-cut-error")
    def ra03_open_cut_error():
        raise invalid_input("demo")

    app.register_blueprint(bp)
    caplog.set_level(logging.WARNING, logger="opencut")
    response = app.test_client().get("/ra03/open-cut-error")
    request_id = response.headers["X-Request-ID"]

    record = _typed_error_record(caplog.records, "INVALID_INPUT")
    assert record.levelno == logging.WARNING
    assert record.error_status == 400
    assert record.error_context == "open_cut_error_handler"
    assert record.request_id == request_id
    assert record.request_method == "GET"
    assert record.request_path == "/ra03/open-cut-error"
    assert "OpenCutError [INVALID_INPUT] in open_cut_error_handler" in record.getMessage()


def test_error_response_logs_structured_context(app, caplog):
    from opencut.errors import error_response

    bp = Blueprint("ra03_error_response", __name__)

    @bp.route("/ra03/error-response")
    def ra03_error_response():
        return error_response("DEMO_TYPED_ERROR", "Demo typed error.", status=409)

    app.register_blueprint(bp)
    caplog.set_level(logging.WARNING, logger="opencut")
    response = app.test_client().get("/ra03/error-response")
    request_id = response.headers["X-Request-ID"]

    record = _typed_error_record(caplog.records, "DEMO_TYPED_ERROR")
    assert record.levelno == logging.WARNING
    assert record.error_status == 409
    assert record.error_context == ""
    assert record.request_id == request_id
    assert record.request_method == "GET"
    assert record.request_path == "/ra03/error-response"
    assert "Typed error response [DEMO_TYPED_ERROR]" in record.getMessage()


def test_safe_open_cut_error_logs_supplied_context(app, caplog):
    from opencut.errors import invalid_input, safe_error
    from opencut.security import get_csrf_token

    bp = Blueprint("ra03_safe_open_cut_error", __name__)

    @bp.route("/ra03/safe-open-cut-error", methods=["POST"])
    def ra03_safe_open_cut_error():
        return safe_error(invalid_input("demo"), context="ra03_safe_open_cut_error")

    app.register_blueprint(bp)
    caplog.set_level(logging.WARNING, logger="opencut")
    csrf_token = get_csrf_token()
    response = app.test_client().post(
        "/ra03/safe-open-cut-error",
        json={"ok": True},
        headers={"X-OpenCut-Token": csrf_token},
    )
    request_id = response.headers["X-Request-ID"]

    record = _typed_error_record(caplog.records, "INVALID_INPUT")
    assert record.levelno == logging.WARNING
    assert record.error_status == 400
    assert record.error_context == "ra03_safe_open_cut_error"
    assert record.request_id == request_id
    assert record.request_method == "POST"
    assert record.request_path == "/ra03/safe-open-cut-error"


def test_safe_error_keeps_single_exception_log_without_duplicate_typed_log(app, caplog):
    from opencut.errors import safe_error

    bp = Blueprint("ra03_safe_error", __name__)

    @bp.route("/ra03/safe-error")
    def ra03_safe_error():
        return safe_error(TimeoutError("timed out"), context="ra03")

    app.register_blueprint(bp)
    caplog.set_level(logging.WARNING, logger="opencut")
    response = app.test_client().get("/ra03/safe-error")

    assert response.status_code == 504
    assert response.get_json()["code"] == "OPERATION_TIMEOUT"
    assert any("Error [OPERATION_TIMEOUT] in ra03" in record.getMessage() for record in caplog.records)
    assert not any(getattr(record, "error_code", "") == "OPERATION_TIMEOUT" for record in caplog.records)
