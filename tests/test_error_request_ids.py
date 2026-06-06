"""RA-04 guards for request IDs in structured error bodies."""

from __future__ import annotations

from flask import Blueprint


def _assert_body_request_id_matches_header(response) -> str:
    payload = response.get_json()
    request_id = payload.get("request_id")

    assert response.status_code >= 400
    assert request_id
    assert request_id == response.headers["X-Request-ID"]
    assert request_id.startswith("r-")
    return request_id


def test_error_response_includes_generated_request_id(app):
    from opencut.errors import error_response

    bp = Blueprint("ra04_error_response", __name__)

    @bp.route("/ra04/error-response")
    def ra04_error_response():
        return error_response("INVALID_INPUT", "Bad input.", status=400)

    app.register_blueprint(bp)
    response = app.test_client().get(
        "/ra04/error-response",
        headers={"X-Request-ID": "client-supplied"},
    )

    request_id = _assert_body_request_id_matches_header(response)
    assert request_id != "client-supplied"
    assert response.get_json()["code"] == "INVALID_INPUT"


def test_open_cut_error_handler_includes_generated_request_id(app):
    from opencut.errors import invalid_input

    bp = Blueprint("ra04_opencut_error", __name__)

    @bp.route("/ra04/open-cut-error")
    def ra04_open_cut_error():
        raise invalid_input("demo")

    app.register_blueprint(bp)
    response = app.test_client().get("/ra04/open-cut-error")

    _assert_body_request_id_matches_header(response)
    assert response.get_json()["code"] == "INVALID_INPUT"


def test_safe_error_includes_generated_request_id(app):
    from opencut.errors import safe_error

    bp = Blueprint("ra04_safe_error", __name__)

    @bp.route("/ra04/safe-error")
    def ra04_safe_error():
        return safe_error(TimeoutError("timed out"), context="ra04")

    app.register_blueprint(bp)
    response = app.test_client().get("/ra04/safe-error")

    _assert_body_request_id_matches_header(response)
    assert response.get_json()["code"] == "OPERATION_TIMEOUT"


def test_builtin_json_error_handlers_include_generated_request_id(client):
    response = client.get("/ra04/missing-route")

    _assert_body_request_id_matches_header(response)
    assert response.get_json()["code"] == "NOT_FOUND"


def test_remote_auth_error_includes_generated_request_id(client, monkeypatch, tmp_path):
    import opencut.auth as auth_module

    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    auth_module.ensure_token()

    response = client.get(
        "/system/feature-state",
        environ_overrides={"REMOTE_ADDR": "192.168.1.5"},
    )

    _assert_body_request_id_matches_header(response)
    assert response.get_json()["code"] == "AUTH_REQUIRED"
