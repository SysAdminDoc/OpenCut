"""
OpenCut Workflow Engine Tests

Smoke tests for:
  - Workflow validation (invalid endpoints rejected)
  - Empty workflow returns error
  - Preset listing returns all 6 built-ins
  - Save / delete custom workflow
"""

import json
from unittest.mock import patch

import pytest

from tests.conftest import csrf_headers


# =====================================================================
# VALIDATION
# =====================================================================

class TestWorkflowValidation:
    """Tests for workflow step validation."""

    def test_invalid_endpoint_rejected(self, client, csrf_token):
        """Unknown endpoints must be rejected with 400."""
        resp = client.post(
            "/workflow/run",
            data=json.dumps({
                "filepath": __file__,  # use this test file as a stand-in
                "workflow": [
                    {"endpoint": "/nonexistent/route", "params": {}},
                ],
            }),
            headers=csrf_headers(csrf_token),
        )
        # async_job validates filepath first, then our code validates steps.
        # If filepath passes, step validation should fail with an error.
        data = resp.get_json()
        # The job may be created and then error out, or the error may be
        # returned directly.  Either way the response should NOT be a
        # clean success.
        if resp.status_code == 200 and "job_id" in data:
            # Job was created — poll for error
            from opencut.jobs import _get_job_copy
            import time
            job_id = data["job_id"]
            for _ in range(40):
                job = _get_job_copy(job_id)
                if job and job.get("status") != "running":
                    break
                time.sleep(0.1)
            assert job is not None
            assert job["status"] == "error"
            assert "unknown endpoint" in job.get("error", "").lower()
        else:
            # Direct error response
            assert resp.status_code >= 400
            assert "error" in data

    def test_empty_workflow_returns_error(self, client, csrf_token):
        """An empty workflow list should be rejected."""
        resp = client.post(
            "/workflow/run",
            data=json.dumps({
                "filepath": __file__,
                "workflow": [],
            }),
            headers=csrf_headers(csrf_token),
        )
        data = resp.get_json()
        if resp.status_code == 200 and "job_id" in data:
            from opencut.jobs import _get_job_copy
            import time
            job_id = data["job_id"]
            for _ in range(40):
                job = _get_job_copy(job_id)
                if job and job.get("status") != "running":
                    break
                time.sleep(0.1)
            assert job is not None
            assert job["status"] == "error"
        else:
            assert resp.status_code >= 400
            assert "error" in data

    def test_missing_endpoint_field_rejected(self, client, csrf_token):
        """A step without an endpoint key should be rejected."""
        resp = client.post(
            "/workflow/run",
            data=json.dumps({
                "filepath": __file__,
                "workflow": [{"params": {}}],
            }),
            headers=csrf_headers(csrf_token),
        )
        data = resp.get_json()
        if resp.status_code == 200 and "job_id" in data:
            from opencut.jobs import _get_job_copy
            import time
            job_id = data["job_id"]
            for _ in range(40):
                job = _get_job_copy(job_id)
                if job and job.get("status") != "running":
                    break
                time.sleep(0.1)
            assert job is not None
            assert job["status"] == "error"
        else:
            assert resp.status_code >= 400


# =====================================================================
# PRESETS
# =====================================================================

class TestWorkflowPresets:
    """Tests for workflow preset listing."""

    def test_presets_returns_all_builtins(self, client):
        """GET /workflow/presets should return all 6 built-in workflows."""
        resp = client.get("/workflow/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "builtins" in data
        assert "custom" in data
        builtins = data["builtins"]
        assert len(builtins) == 6
        names = {wf["name"] for wf in builtins}
        assert "Clean Interview" in names
        assert "Podcast Polish" in names
        assert "Social Media Clip" in names
        assert "YouTube Upload" in names
        assert "Documentary Rough Cut" in names
        assert "Studio Audio" in names

    def test_builtins_have_required_fields(self, client):
        """Each built-in preset must have name, steps, and builtin flag."""
        resp = client.get("/workflow/presets")
        data = resp.get_json()
        for wf in data["builtins"]:
            assert "name" in wf
            assert "steps" in wf
            assert isinstance(wf["steps"], list)
            assert len(wf["steps"]) > 0
            assert wf.get("builtin") is True


# =====================================================================
# SAVE / DELETE CUSTOM WORKFLOW
# =====================================================================

class TestWorkflowSaveDelete:
    """Tests for saving and deleting custom workflows."""

    def test_save_and_list_custom_workflow(self, client, csrf_token):
        """Saving a custom workflow should make it appear in presets."""
        # Save
        resp = client.post(
            "/workflow/save",
            data=json.dumps({
                "name": "Test Workflow",
                "steps": [
                    {"endpoint": "/silence", "params": {}},
                    {"endpoint": "/audio/normalize", "params": {}},
                ],
                "description": "A test workflow",
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("success") is True

        # Verify it shows up in presets
        resp = client.get("/workflow/presets")
        data = resp.get_json()
        custom_names = {wf["name"] for wf in data["custom"]}
        assert "Test Workflow" in custom_names

    def test_delete_custom_workflow(self, client, csrf_token):
        """Deleting a custom workflow should remove it from presets."""
        # First save one
        client.post(
            "/workflow/save",
            data=json.dumps({
                "name": "To Delete",
                "steps": [{"endpoint": "/silence", "params": {}}],
            }),
            headers=csrf_headers(csrf_token),
        )

        # Delete it
        resp = client.delete(
            "/workflow/delete",
            data=json.dumps({"name": "To Delete"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("success") is True

        # Verify it's gone
        resp = client.get("/workflow/presets")
        data = resp.get_json()
        custom_names = {wf["name"] for wf in data["custom"]}
        assert "To Delete" not in custom_names

    def test_save_requires_name(self, client, csrf_token):
        """Saving without a name should fail."""
        resp = client.post(
            "/workflow/save",
            data=json.dumps({
                "name": "",
                "steps": [{"endpoint": "/silence", "params": {}}],
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_save_rejects_invalid_steps(self, client, csrf_token):
        """Saving with an invalid endpoint should fail."""
        resp = client.post(
            "/workflow/save",
            data=json.dumps({
                "name": "Bad Workflow",
                "steps": [{"endpoint": "/fake/endpoint", "params": {}}],
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_cannot_delete_builtin(self, client, csrf_token):
        """Built-in workflows cannot be deleted."""
        resp = client.delete(
            "/workflow/delete",
            data=json.dumps({"name": "Clean Interview"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "built-in" in data.get("error", "").lower()

    def test_cannot_overwrite_builtin(self, client, csrf_token):
        """Cannot save a custom workflow with a built-in name."""
        resp = client.post(
            "/workflow/save",
            data=json.dumps({
                "name": "Clean Interview",
                "steps": [{"endpoint": "/silence", "params": {}}],
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "built-in" in data.get("error", "").lower()

    def test_delete_nonexistent_returns_404(self, client, csrf_token):
        """Deleting a workflow that doesn't exist should return 404."""
        resp = client.delete(
            "/workflow/delete",
            data=json.dumps({"name": "Nonexistent Workflow"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 404
