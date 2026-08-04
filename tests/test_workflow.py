"""
OpenCut Workflow Engine Tests

Smoke tests for:
  - Workflow validation (invalid endpoints rejected)
  - Empty workflow returns error
  - Preset listing returns all 6 built-ins
  - Save / delete custom workflow
"""

import json
import time

import pytest

from tests.conftest import csrf_headers

# =====================================================================
# VALIDATION
# =====================================================================

class TestWorkflowValidation:
    """Tests for workflow step validation."""

    def test_compile_plan_binds_source_definition_and_preflight(self, tmp_path):
        from opencut.core.workflow import (
            compile_workflow_plan,
            validate_workflow_plan,
            workflow_definition_id,
        )

        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        steps = [{"endpoint": "/audio/denoise", "params": {"strength": 0.5}}]
        plan = compile_workflow_plan(
            str(source),
            steps,
            capabilities={"ffmpeg": True},
            media_info={"duration": 2, "has_audio": True, "has_video": False},
            check_disk=False,
        )

        assert plan["definition_id"] == workflow_definition_id(steps)
        assert plan["preflight"]["status"] == "ready"
        assert validate_workflow_plan(plan, filepath=str(source), steps=steps) == (True, "")

        tampered = dict(plan)
        tampered["steps"] = [dict(plan["steps"][0], params={"strength": 0.9})]
        assert validate_workflow_plan(tampered, filepath=str(source))[0] is False

    def test_compile_plan_rejects_typed_parameters(self, tmp_path):
        from opencut.core.workflow import compile_workflow_plan

        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        with pytest.raises(ValueError, match="target_lufs.*number"):
            compile_workflow_plan(
                str(source),
                [{"endpoint": "/audio/loudness-match", "params": {"target_lufs": "-16"}}],
                media_info={"duration": 2, "has_audio": True},
                capabilities={"ffmpeg": True},
                check_disk=False,
            )

    def test_compile_plan_requires_explicit_approval_for_external_step(self, tmp_path):
        from opencut.core.workflow import compile_workflow_plan, workflow_plan_requires_approval

        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        plan = compile_workflow_plan(
            str(source),
            [{"endpoint": "/audio/tts/generate", "params": {"url": "https://example.test/voice"}}],
            media_info={"duration": 2, "has_audio": True},
            capabilities={"ffmpeg": True},
            check_disk=False,
        )

        assert workflow_plan_requires_approval(plan)
        assert any(item["state"] == "approval_required" for item in plan["preflight"]["checks"])

    def test_validate_plan_rejects_unknown_endpoint_even_with_matching_hash(self, tmp_path):
        from opencut.core.workflow import (
            compile_workflow_plan,
            validate_workflow_plan,
            workflow_definition_id,
            workflow_plan_id,
        )

        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        plan = compile_workflow_plan(
            str(source),
            [{"endpoint": "/audio/denoise", "params": {}}],
            media_info={"duration": 2, "has_audio": True},
            capabilities={"ffmpeg": True},
            check_disk=False,
        )
        plan["steps"][0]["endpoint"] = "/workflow/save"
        plan["definition_id"] = workflow_definition_id(
            [{"endpoint": "/workflow/save", "params": {}}],
        )
        plan["plan_id"] = workflow_plan_id(plan)

        valid, reason = validate_workflow_plan(plan, filepath=str(source))
        assert valid is False
        assert "unknown endpoint" in reason

    def test_resume_accepts_persisted_approval_after_token_redaction(self, app, tmp_path):
        from opencut.core.workflow import compile_workflow_plan
        from opencut.routes.workflow import _workflow_plan_from_request

        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        steps = [{
            "endpoint": "/audio/tts/generate",
            "params": {"url": "https://example.test/voice"},
        }]
        plan = compile_workflow_plan(
            str(source),
            steps,
            media_info={"duration": 2, "has_audio": True},
            capabilities={"ffmpeg": True},
            check_disk=False,
        )
        approved_plan = json.loads(json.dumps(plan))
        approved_plan["approval"].update({
            "approved": True,
            "plan_id": plan["plan_id"],
            "token": "[REDACTED]",
        })
        resume_state = {
            "result": {"plan_id": plan["plan_id"], "steps_completed": 1},
            "payload": {"plan": approved_plan},
        }

        with app.app_context():
            resumed_plan = _workflow_plan_from_request(
                {"filepath": str(source), "workflow": steps, "plan": approved_plan},
                str(source),
                resume_state=resume_state,
            )

        assert resumed_plan["plan_id"] == plan["plan_id"]

    def test_compile_and_approve_routes_return_the_same_plan(self, client, csrf_token, tmp_path):
        source = tmp_path / "source.txt"
        source.write_bytes(b"source")
        payload = {
            "filepath": str(source),
            "workflow": [{"endpoint": "/audio/tts/generate", "params": {"url": "https://example.test/voice"}}],
        }
        compiled = client.post(
            "/workflow/compile",
            data=json.dumps(payload),
            headers=csrf_headers(csrf_token),
        )
        assert compiled.status_code == 200
        plan = compiled.get_json()["plan"]
        assert plan["approval"]["required"] is True

        approved = client.post(
            "/workflow/approve",
            data=json.dumps({"plan": plan}),
            headers=csrf_headers(csrf_token),
        )
        assert approved.status_code == 200
        approved_plan = approved.get_json()["plan"]
        assert approved_plan["plan_id"] == plan["plan_id"]
        assert approved_plan["approval"]["approved"] is True

    def test_compiled_workflow_resumes_from_matching_artifact_checksum(self, tmp_path, monkeypatch):
        import opencut.core.workflow as workflow_core

        source = tmp_path / "source.txt"
        artifact = tmp_path / "source_denoised.txt"
        source.write_bytes(b"source")
        artifact.write_bytes(b"derived")
        steps = [{"endpoint": "/audio/denoise", "params": {}}]
        plan = workflow_core.compile_workflow_plan(
            str(source),
            steps,
            capabilities={"ffmpeg": True},
            media_info={"duration": 2, "has_audio": True},
            check_disk=False,
        )

        class _Response:
            status_code = 200

            @staticmethod
            def get_json():
                return {"job_id": "workflow-step-1"}

        class _Client:
            def __init__(self):
                self.posts = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                self.posts += 1
                return _Response()

        class _App:
            def __init__(self):
                self.client = _Client()

            def test_client(self):
                return self.client

        monkeypatch.setattr(
            workflow_core,
            "_wait_for_job",
            lambda *_args, **_kwargs: {"status": "complete", "result": {"output_path": str(artifact)}},
        )
        first_app = _App()
        first = workflow_core.run_workflow(
            first_app, str(source), steps, "csrf", plan=plan,
        )
        assert first["step_results"][0]["artifact_checksum"]
        assert first_app.client.posts == 1

        resumed_app = _App()
        resumed = workflow_core.run_workflow(
            resumed_app,
            str(source),
            steps,
            "csrf",
            plan=plan,
            resume_state=first,
        )
        assert resumed["success"] is True
        assert resumed["step_results"][0]["resumed"] is True
        assert resumed_app.client.posts == 0

    def test_known_endpoints_come_from_manifest_workflow_metadata(self):
        """Workflow validation should use route-manifest workflow opt-ins."""
        import json

        from opencut.core.workflow import KNOWN_ENDPOINTS, ROUTE_MANIFEST_PATH

        manifest = json.loads(ROUTE_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_workflow = {
            route["rule"]: route["workflow"]["label"]
            for route in manifest["routes"]
            if isinstance(route.get("workflow"), dict)
        }

        assert KNOWN_ENDPOINTS == dict(sorted(manifest_workflow.items()))
        assert KNOWN_ENDPOINTS["/silence"] == "Detecting silence"
        assert "/workflow/run" not in KNOWN_ENDPOINTS

    def test_registered_post_route_without_marker_is_rejected(self):
        """A live POST route is not workflowable unless it opts in."""
        from opencut.core.workflow import validate_workflow_steps

        ok, err = validate_workflow_steps([{"endpoint": "/workflow/save"}])

        assert ok is False
        assert "unknown endpoint" in err.lower()

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
            import time

            from opencut.jobs import _get_job_copy
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
            import time

            from opencut.jobs import _get_job_copy
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
            import time

            from opencut.jobs import _get_job_copy
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

    def test_nested_workflow_steps_payload_runs(self, client, csrf_token, monkeypatch):
        """Nested workflow payloads from the panel should still run."""
        import opencut.core.workflow as workflow_core
        from opencut.jobs import _get_job_copy

        captured = {}

        def fake_run_workflow(app, filepath, steps, csrf_token, on_progress=None, parent_job_id=""):
            captured["steps"] = steps
            return {
                "success": True,
                "steps_completed": len(steps),
                "output": filepath,
                "step_results": [],
            }

        monkeypatch.setattr(workflow_core, "run_workflow", fake_run_workflow)

        resp = client.post(
            "/workflow/run",
            data=json.dumps({
                "filepath": __file__,
                "workflow": {
                    "steps": [
                        {"endpoint": "/silence", "params": {}},
                    ],
                },
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

        job = None
        for _ in range(40):
            job = _get_job_copy(data["job_id"])
            if job and job.get("status") != "running":
                break
            time.sleep(0.1)

        assert job is not None
        assert job["status"] == "complete"
        assert captured["steps"] == [{"endpoint": "/silence", "params": {}}]

    def test_parent_cancellation_cancels_active_step_and_stops_workflow(
        self, app, monkeypatch
    ):
        from unittest.mock import Mock

        import opencut.core.workflow as workflow_core
        import opencut.jobs as jobs

        class _Response:
            status_code = 200

            @staticmethod
            def get_json():
                return {"job_id": "workflow-step-1"}

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return _Response()

        class _App:
            def test_client(self):
                return _Client()

        fake_process = Mock()
        fake_process.wait.return_value = None
        with jobs.job_lock:
            jobs.jobs.clear()
            jobs._job_processes.clear()
            jobs.jobs["workflow-step-1"] = {
                "id": "workflow-step-1",
                "status": "running",
                "progress": 40,
            }
            jobs._job_processes["workflow-step-1"] = fake_process

        parent_cancel_checks = {"count": 0}

        def fake_is_cancelled(job_id):
            if job_id != "parent-job":
                return False
            parent_cancel_checks["count"] += 1
            return parent_cancel_checks["count"] >= 2

        monkeypatch.setattr(jobs, "_is_cancelled", fake_is_cancelled)
        monkeypatch.setattr(jobs, "_persist_job", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(jobs, "_emit_job_webhook", lambda *_args, **_kwargs: None)
        monkeypatch.setattr("opencut.workers.cancel_job", lambda _job_id: False)

        result = workflow_core.run_workflow(
            _App(),
            __file__,
            [
                {"endpoint": "/silence", "params": {}},
                {"endpoint": "/audio/normalize", "params": {}},
            ],
            "csrf-token",
            parent_job_id="parent-job",
        )

        assert result["success"] is False
        assert result["steps_completed"] == 0
        with jobs.job_lock:
            assert jobs.jobs["workflow-step-1"]["status"] == "cancelled"
            assert jobs.jobs["workflow-step-1"]["message"] == (
                "Cancelled because the parent workflow was cancelled"
            )
            assert "workflow-step-1" not in jobs._job_processes
        fake_process.terminate.assert_called_once_with()
        fake_process.wait.assert_called_once_with(timeout=3)

    def test_workflow_step_timeout_cancels_active_subjob(self, monkeypatch):
        import opencut.core.workflow as workflow_core
        import opencut.jobs as jobs

        class _Response:
            status_code = 200

            @staticmethod
            def get_json():
                return {"job_id": "workflow-step-1"}

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return _Response()

        class _App:
            def test_client(self):
                return _Client()

        calls = []

        def fake_cancel(job_id, **kwargs):
            calls.append((job_id, kwargs))
            return None, "not_running"

        monkeypatch.setattr(jobs, "_cancel_job", fake_cancel)
        monkeypatch.setattr(
            workflow_core,
            "_wait_for_job",
            lambda *_args, **_kwargs: None,
        )

        result = workflow_core.run_workflow(
            _App(),
            __file__,
            [{"endpoint": "/silence", "params": {}}],
            "csrf-token",
            parent_job_id="parent-job",
        )

        assert result["success"] is False
        assert calls == [
            (
                "workflow-step-1",
                {"message": "Workflow step timed out", "persist_sync": True},
            )
        ]

    def test_parent_cancellation_between_steps_returns_partial_results(self, monkeypatch):
        import opencut.core.workflow as workflow_core
        import opencut.jobs as jobs

        class _Response:
            status_code = 200

            @staticmethod
            def get_json():
                return {"job_id": "workflow-step-1"}

        class _Client:
            def __init__(self):
                self.posts = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, endpoint, **_kwargs):
                self.posts.append(endpoint)
                return _Response()

        class _App:
            def __init__(self):
                self.client = _Client()

            def test_client(self):
                return self.client

        app = _App()
        cancel_checks = []
        monkeypatch.setattr(
            jobs,
            "_is_cancelled",
            lambda job_id: cancel_checks.append(job_id) or len(cancel_checks) >= 2,
        )
        monkeypatch.setattr(
            workflow_core,
            "_wait_for_job",
            lambda *_args, **_kwargs: {"status": "complete", "result": {}},
        )

        result = workflow_core.run_workflow(
            app,
            __file__,
            [
                {"endpoint": "/silence", "params": {}},
                {"endpoint": "/audio/normalize", "params": {}},
            ],
            "csrf-token",
            parent_job_id="parent-job",
        )

        assert result == {
            "success": False,
            "steps_completed": 1,
            "output": __file__,
            "step_results": [{
                "step": 1,
                "endpoint": "/silence",
                "success": True,
                "output": __file__,
                "job_id": "workflow-step-1",
            }],
            "error": "Workflow cancelled by user",
        }
        assert cancel_checks == ["parent-job", "parent-job"]
        assert app.client.posts == ["/silence"]

    def test_wait_for_job_treats_interrupted_as_terminal(self, monkeypatch):
        import opencut.core.workflow as workflow_core
        import opencut.jobs as jobs

        monkeypatch.setattr(
            jobs,
            "_get_job_copy",
            lambda _job_id: {"id": "interrupted-step", "status": "interrupted"},
        )

        result = workflow_core._wait_for_job(
            object(),
            "interrupted-step",
            "csrf-token",
            1,
            "Interrupted step",
            None,
            1,
            timeout=3600,
        )

        assert result == {"id": "interrupted-step", "status": "interrupted"}


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

        # Preview and delete it
        preview = client.delete(
            "/workflow/delete",
            data=json.dumps({"name": "To Delete", "dry_run": True}),
            headers=csrf_headers(csrf_token),
        )
        assert preview.status_code == 200
        token = preview.get_json()["confirm_token"]

        resp = client.delete(
            "/workflow/delete",
            data=json.dumps({"name": "To Delete", "confirm_token": token}),
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
