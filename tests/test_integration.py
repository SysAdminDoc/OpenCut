"""
OpenCut Integration Tests

Uses Flask's test client to exercise routes without spawning a real server.
External dependencies (whisper, ffmpeg, LLM) are mocked where needed.
"""

import json
from unittest.mock import patch

from tests.conftest import csrf_headers

# =====================================================================
# /health
# =====================================================================

class TestHealth:
    def test_health_returns_csrf_and_version(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "csrf_token" in data
        assert len(data["csrf_token"]) > 0

    def test_health_contains_capabilities(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "capabilities" in data
        caps = data["capabilities"]
        assert "silence" in caps
        assert "ffmpeg" in caps


# =====================================================================
# /system/update-check
# =====================================================================

class TestUpdateCheck:
    def test_update_check_returns_expected_fields(self, client):
        resp = client.get("/system/update-check")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_version" in data
        assert "latest_version" in data
        assert "update_available" in data
        assert "release_url" in data


# =====================================================================
# /system/dependencies
# =====================================================================

class TestDependencies:
    def test_dependencies_returns_dict(self, client):
        resp = client.get("/system/dependencies")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Should contain at least ffmpeg and numpy checks
        assert "ffmpeg" in data
        assert "numpy" in data
        for key, val in data.items():
            assert "installed" in val


# =====================================================================
# /openapi.json
# =====================================================================

class TestOpenAPISpec:
    def test_openapi_returns_valid_spec(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["openapi"] == "3.0.3"
        assert "info" in data
        assert data["info"]["title"] == "OpenCut API"
        assert "paths" in data
        assert len(data["paths"]) > 0

    def test_openapi_contains_health_path(self, client):
        resp = client.get("/openapi.json")
        data = resp.get_json()
        assert "/health" in data["paths"]

    def test_openapi_version_matches(self, client):
        from opencut import __version__
        resp = client.get("/openapi.json")
        data = resp.get_json()
        assert data["info"]["version"] == __version__


# =====================================================================
# CSRF protection
# =====================================================================

class TestCSRFProtection:
    def test_post_without_csrf_returns_403(self, client):
        resp = client.post(
            "/search/footage",
            data=json.dumps({"query": "hello"}),
            content_type="application/json",
        )
        assert resp.status_code == 403
        data = resp.get_json()
        assert "error" in data

    def test_post_with_valid_csrf_succeeds(self, client, csrf_token):
        """POST with a valid CSRF token should pass CSRF check (may still
        return 400 for missing/invalid body, but NOT 403)."""
        resp = client.post(
            "/search/footage",
            data=json.dumps({"query": "test footage"}),
            headers=csrf_headers(csrf_token),
        )
        # Should not be 403 -- CSRF is valid
        assert resp.status_code != 403

    def test_post_with_bogus_csrf_returns_403(self, client):
        resp = client.post(
            "/search/footage",
            data=json.dumps({"query": "hello"}),
            headers=csrf_headers("totally-invalid-token"),
        )
        assert resp.status_code == 403


# =====================================================================
# /search/footage
# =====================================================================

class TestSearchFootage:
    def test_empty_query_returns_400(self, client, csrf_token):
        resp = client.post(
            "/search/footage",
            data=json.dumps({"query": ""}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_query_returns_results_array(self, client, csrf_token):
        with patch("opencut.core.footage_search.search_footage", return_value=[]):
            resp = client.post(
                "/search/footage",
                data=json.dumps({"query": "sunset establishing shot"}),
                headers=csrf_headers(csrf_token),
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "query" in data
        assert "total_matches" in data

    def test_query_too_long_returns_400(self, client, csrf_token):
        resp = client.post(
            "/search/footage",
            data=json.dumps({"query": "x" * 501}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400


# =====================================================================
# /deliverables/vfx-sheet
# =====================================================================

class TestDeliverablesVFXSheet:
    def test_empty_sequence_data_returns_400(self, client, csrf_token):
        resp = client.post(
            "/deliverables/vfx-sheet",
            data=json.dumps({"sequence_data": {}}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_data_returns_output_and_rows(self, client, csrf_token):
        mock_result = {"output": "/tmp/vfx_sheet.csv", "rows": 3}
        with patch("opencut.core.deliverables.generate_vfx_sheet", return_value=mock_result):
            resp = client.post(
                "/deliverables/vfx-sheet",
                data=json.dumps({
                    "sequence_data": {
                        "name": "Test Sequence",
                        "clips": [
                            {"name": "VFX_001", "start": 0, "end": 5, "type": "vfx"},
                        ],
                    }
                }),
                headers=csrf_headers(csrf_token),
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "output" in data
        assert "rows" in data


# =====================================================================
# /nlp/command
# =====================================================================

class TestNLPCommand:
    def test_empty_command_returns_400(self, client, csrf_token):
        resp = client.post(
            "/nlp/command",
            data=json.dumps({"command": ""}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_command_returns_route_and_confidence(self, client, csrf_token):
        mock_parsed = {
            "route": "/silence",
            "params": {"threshold": -30},
            "confidence": 0.95,
            "explanation": "Detected silence removal command",
        }
        with patch("opencut.core.nlp_command.parse_command", return_value=mock_parsed):
            resp = client.post(
                "/nlp/command",
                data=json.dumps({"command": "remove silences from the video"}),
                headers=csrf_headers(csrf_token),
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "route" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)

    def test_command_too_long_returns_400(self, client, csrf_token):
        resp = client.post(
            "/nlp/command",
            data=json.dumps({"command": "x" * 2001}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400


# =====================================================================
# /settings/llm GET
# =====================================================================

class TestSettingsLLM:
    def test_get_llm_returns_provider_and_masked_key(self, client):
        mock_settings = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-abc123456789",
            "base_url": "",
        }
        with patch("opencut.user_data.load_llm_settings", return_value=mock_settings):
            resp = client.get("/settings/llm")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "provider" in data
        assert "model" in data
        # Key must be masked
        assert data["api_key"].startswith("***")
        assert "sk-abc" not in data["api_key"]


# =====================================================================
# /timeline/batch-rename
# =====================================================================

class TestTimelineBatchRename:
    def test_invalid_renames_returns_400(self, client, csrf_token):
        resp = client.post(
            "/timeline/batch-rename",
            data=json.dumps({"renames": "not-a-list"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_renames_returns_validated(self, client, csrf_token):
        resp = client.post(
            "/timeline/batch-rename",
            data=json.dumps({
                "renames": [
                    {"nodeId": "1", "currentName": "clip_001", "newName": "intro"},
                    {"nodeId": "2", "currentName": "clip_002", "newName": ""},
                ]
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "validated_renames" in data
        assert "invalid" in data
        assert len(data["validated_renames"]) == 1
        assert len(data["invalid"]) == 1

    def test_empty_list_returns_empty_validated(self, client, csrf_token):
        resp = client.post(
            "/timeline/batch-rename",
            data=json.dumps({"renames": []}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["validated_renames"] == []
        assert data["invalid"] == []


# =====================================================================
# /timeline/smart-bins
# =====================================================================

class TestTimelineSmartBins:
    def test_invalid_rules_returns_400(self, client, csrf_token):
        resp = client.post(
            "/timeline/smart-bins",
            data=json.dumps({"rules": "not-a-list"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_valid_rules_returns_validated(self, client, csrf_token):
        resp = client.post(
            "/timeline/smart-bins",
            data=json.dumps({
                "rules": [
                    {"binName": "VFX", "rule": "contains", "field": "name", "value": "vfx"},
                    {"binName": "", "rule": "invalid_rule", "field": "name", "value": "test"},
                ]
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "validated_rules" in data
        assert "invalid" in data
        assert len(data["validated_rules"]) == 1
        assert len(data["invalid"]) == 1

    def test_empty_rules_returns_empty(self, client, csrf_token):
        resp = client.post(
            "/timeline/smart-bins",
            data=json.dumps({"rules": []}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["validated_rules"] == []
        assert data["invalid"] == []
