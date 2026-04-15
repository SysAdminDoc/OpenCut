"""
OpenCut Collaboration & Review Tests

Comprehensive tests for all 5 modules in Category 73:
  1. ReviewComments — CRUD, filtering, export, import, threading, stats
  2. VersionCompare — modes, SSIM/PSNR, audio, report structure
  3. ApprovalWorkflow — state machine, approvers, deadlines, dashboard
  4. EditHistory — append, query, undo, diff, export, replay, statistics
  5. SharedPresets — CRUD, rating, duplicate detection, import/export, merge

~120 tests covering core logic and route endpoints.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tests.conftest import csrf_headers


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _clean_review_sessions():
    """Reset review session cache between tests."""
    yield
    from opencut.core.review_comments import clear_session_cache
    clear_session_cache()


@pytest.fixture(autouse=True)
def _clean_approval_manager():
    """Reset approval workflow manager between tests."""
    yield
    from opencut.core.approval_workflow import reset_manager
    reset_manager()


@pytest.fixture(autouse=True)
def _clean_history_cache():
    """Reset edit history cache between tests."""
    yield
    from opencut.core.edit_history import clear_history_cache
    clear_history_cache()


@pytest.fixture(autouse=True)
def _clean_preset_library():
    """Reset shared preset library between tests."""
    yield
    from opencut.core.shared_presets import reset_library
    reset_library()


@pytest.fixture
def app(app):
    """Extend the base app fixture to register the collab_review_bp."""
    from opencut.routes.collab_review_routes import collab_review_bp
    # Only register if not already registered
    if "collab_review" not in app.blueprints:
        app.register_blueprint(collab_review_bp)
    return app


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="opencut_collab_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================================
# 1. REVIEW COMMENTS — Core Logic
# ============================================================================

class TestReviewCommentsCore:
    """Unit tests for opencut.core.review_comments."""

    def _session(self, name="default"):
        """Create a fresh ReviewSession with a unique path to avoid cross-test leakage."""
        from opencut.core.review_comments import ReviewSession
        import uuid as _uuid
        return ReviewSession(f"/fake/{name}_{_uuid.uuid4().hex[:8]}.prproj")

    def test_add_comment_basic(self, tmp_dir):
        session = self._session("add_basic")
        comment = session.add_comment(text="Fix color", author="editor1",
                                      timestamp_sec=10.5, frame_number=315)
        assert comment.text == "Fix color"
        assert comment.author == "editor1"
        assert comment.timestamp_sec == 10.5
        assert comment.frame_number == 315
        assert comment.status == "open"
        assert comment.id

    def test_add_comment_empty_text_raises(self):
        session = self._session("empty_text")
        with pytest.raises(ValueError, match="text is required"):
            session.add_comment(text="", author="editor1")

    def test_add_comment_whitespace_only_raises(self):
        session = self._session("ws_text")
        with pytest.raises(ValueError, match="text is required"):
            session.add_comment(text="   ", author="editor1")

    def test_resolve_comment(self):
        session = self._session("resolve")
        comment = session.add_comment(text="Issue here", author="a")
        resolved = session.resolve_comment(comment.id)
        assert resolved.status == "resolved"

    def test_resolve_wontfix(self):
        session = self._session("wontfix")
        comment = session.add_comment(text="Minor", author="a")
        resolved = session.resolve_comment(comment.id, status="wontfix")
        assert resolved.status == "wontfix"

    def test_resolve_invalid_status(self):
        session = self._session("inv_status")
        comment = session.add_comment(text="Test", author="a")
        with pytest.raises(ValueError, match="Invalid resolve status"):
            session.resolve_comment(comment.id, status="invalid")

    def test_delete_comment(self):
        session = self._session("delete")
        comment = session.add_comment(text="Delete me", author="a")
        assert session.delete_comment(comment.id) is True
        assert session.comment_count == 0

    def test_delete_nonexistent(self):
        session = self._session("del_ne")
        assert session.delete_comment("nonexistent") is False

    def test_delete_cascades_replies(self):
        session = self._session("cascade")
        parent = session.add_comment(text="Parent", author="a")
        session.add_comment(text="Reply1", author="b", parent_id=parent.id)
        session.add_comment(text="Reply2", author="c", parent_id=parent.id)
        assert session.comment_count == 3
        session.delete_comment(parent.id)
        assert session.comment_count == 0

    def test_list_comments_sorted(self):
        session = self._session("sorted")
        session.add_comment(text="Second", author="a", timestamp_sec=20.0)
        session.add_comment(text="First", author="a", timestamp_sec=5.0)
        session.add_comment(text="Third", author="a", timestamp_sec=50.0)
        comments = session.list_comments(sort_by="timestamp_sec")
        assert comments[0].text == "First"
        assert comments[2].text == "Third"

    def test_filter_by_status(self):
        session = self._session("filter_status")
        c1 = session.add_comment(text="Open", author="a")
        c2 = session.add_comment(text="Resolved", author="a")
        session.resolve_comment(c2.id)
        open_only = session.filter_by_status("open")
        assert len(open_only) == 1
        assert open_only[0].id == c1.id

    def test_filter_by_time_range(self):
        session = self._session("filter_time")
        session.add_comment(text="A", author="a", timestamp_sec=5.0)
        session.add_comment(text="B", author="a", timestamp_sec=15.0)
        session.add_comment(text="C", author="a", timestamp_sec=25.0)
        filtered = session.filter_by_time_range(10.0, 20.0)
        assert len(filtered) == 1
        assert filtered[0].text == "B"

    def test_filter_by_author(self):
        session = self._session("filter_author")
        session.add_comment(text="A", author="Alice")
        session.add_comment(text="B", author="Bob")
        session.add_comment(text="C", author="alice")
        filtered = session.filter_by_author("alice")
        assert len(filtered) == 2

    def test_get_thread(self):
        session = self._session("thread")
        parent = session.add_comment(text="Parent", author="a")
        session.add_comment(text="Reply", author="b", parent_id=parent.id)
        thread = session.get_thread(parent.id)
        assert len(thread) == 2
        assert thread[0].text == "Parent"

    def test_thread_not_found(self):
        session = self._session("thread_nf")
        with pytest.raises(ValueError, match="not found"):
            session.get_thread("nonexistent")

    def test_invalid_parent_id(self):
        session = self._session("bad_parent")
        with pytest.raises(ValueError, match="Parent comment not found"):
            session.add_comment(text="Reply", author="a", parent_id="bad_id")

    def test_annotation_types(self):
        session = self._session("annot")
        c = session.add_comment(text="Circle here", author="a",
                                annotation_type="drawing_circle",
                                annotation_data={"cx": 100, "cy": 200, "r": 50})
        assert c.annotation_type == "drawing_circle"
        assert c.annotation_data["r"] == 50

    def test_invalid_annotation_type_defaults(self):
        session = self._session("bad_annot")
        c = session.add_comment(text="Bad type", author="a",
                                annotation_type="invalid_type")
        assert c.annotation_type == "text"

    def test_export_json(self):
        session = self._session("export_json")
        session.add_comment(text="A", author="a")
        session.add_comment(text="B", author="b")
        exported = session.export_json()
        data = json.loads(exported)
        assert len(data["comments"]) == 2

    def test_export_csv(self):
        session = self._session("export_csv")
        session.add_comment(text="A", author="a")
        csv_str = session.export_csv()
        assert "id,timestamp_sec" in csv_str
        assert "A" in csv_str

    def test_import_json(self):
        session = self._session("import_json")
        session.add_comment(text="Existing", author="a")
        import_data = json.dumps({
            "comments": [
                {"text": "Imported", "author": "b", "timestamp_sec": 5.0}
            ]
        })
        count = session.import_json(import_data)
        assert count == 1
        assert session.comment_count == 2

    def test_import_json_merge_skips_duplicates(self):
        session = self._session("merge_skip")
        c = session.add_comment(text="Existing", author="a")
        import_data = json.dumps({
            "comments": [{"id": c.id, "text": "Overwrite?", "author": "b"}]
        })
        count = session.import_json(import_data, merge=True)
        assert count == 0
        assert session.get_comment(c.id).text == "Existing"

    def test_import_json_no_merge_overwrites(self):
        session = self._session("merge_overwrite")
        c = session.add_comment(text="Existing", author="a")
        import_data = json.dumps({
            "comments": [{"id": c.id, "text": "Overwritten", "author": "b"}]
        })
        count = session.import_json(import_data, merge=False)
        assert count == 1
        assert session.get_comment(c.id).text == "Overwritten"

    def test_import_frameio(self):
        session = self._session("frameio")
        frameio_data = json.dumps([
            {
                "text": "Frame.io comment",
                "timestamp": 12.5,
                "frame": 375,
                "owner": {"name": "Client"},
                "completed": False,
            }
        ])
        count = session.import_frameio(frameio_data)
        assert count == 1
        comments = session.list_comments()
        assert comments[0].author == "Client"
        assert comments[0].status == "open"

    def test_import_invalid_json(self):
        session = self._session("bad_json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            session.import_json("not json{{{")

    def test_summary_stats(self):
        session = self._session("stats")
        session.add_comment(text="A", author="Alice")
        c2 = session.add_comment(text="B", author="Bob")
        session.resolve_comment(c2.id)
        stats = session.summary_stats()
        assert stats["total"] == 2
        assert stats["open"] == 1
        assert stats["resolved"] == 1
        assert stats["by_author"]["Alice"] == 1

    def test_update_comment(self):
        session = self._session("update")
        c = session.add_comment(text="Original", author="a")
        updated = session.update_comment(c.id, text="Updated", tags=["fixed"])
        assert updated.text == "Updated"
        assert updated.tags == ["fixed"]

    def test_update_nonexistent_comment(self):
        session = self._session("update_nf")
        with pytest.raises(ValueError, match="not found"):
            session.update_comment("nonexistent", text="test")

    def test_from_dict_round_trip(self):
        from opencut.core.review_comments import ReviewComment
        c = ReviewComment(text="Test", author="a", timestamp_sec=5.0)
        d = c.to_dict()
        c2 = ReviewComment.from_dict(d)
        assert c2.text == c.text
        assert c2.id == c.id

    def test_get_session_caching(self):
        from opencut.core.review_comments import get_session
        path = "/fake/cache_test_unique.prproj"
        s1 = get_session(path)
        s2 = get_session(path)
        assert s1 is s2

    def test_tags_on_comment(self):
        session = self._session("tags")
        c = session.add_comment(text="Tagged", author="a", tags=["urgent", "color"])
        assert "urgent" in c.tags


# ============================================================================
# 2. VERSION COMPARE — Core Logic
# ============================================================================

class TestVersionCompareCore:
    """Unit tests for opencut.core.version_compare."""

    def test_compare_report_dataclass(self):
        from opencut.core.version_compare import CompareReport
        report = CompareReport(file_a="a.mp4", file_b="b.mp4", mode="side_by_side")
        d = report.to_dict()
        assert d["file_a"] == "a.mp4"
        assert d["mode"] == "side_by_side"
        assert d["overall_similarity"] == 100.0

    def test_frame_compare_result(self):
        from opencut.core.version_compare import FrameCompareResult
        fr = FrameCompareResult(frame_index=5, ssim=0.92, psnr=35.0, changed=True)
        d = fr.to_dict()
        assert d["frame_index"] == 5
        assert d["changed"] is True

    def test_compare_modes_constant(self):
        from opencut.core.version_compare import COMPARE_MODES
        assert "side_by_side" in COMPARE_MODES
        assert "overlay_diff" in COMPARE_MODES
        assert "flicker" in COMPARE_MODES
        assert "swipe" in COMPARE_MODES

    def test_compare_versions_file_not_found(self):
        from opencut.core.version_compare import compare_versions
        with pytest.raises(FileNotFoundError):
            compare_versions("/nonexistent/a.mp4", "/nonexistent/b.mp4")

    def test_compare_versions_invalid_mode(self, tmp_dir):
        from opencut.core.version_compare import compare_versions
        a = os.path.join(tmp_dir, "a.mp4")
        b = os.path.join(tmp_dir, "b.mp4")
        open(a, "w").close()
        open(b, "w").close()
        with pytest.raises(ValueError, match="Invalid mode"):
            compare_versions(a, b, mode="invalid_mode")

    def test_extract_frames_no_duration(self, monkeypatch, tmp_dir):
        from opencut.core import version_compare
        monkeypatch.setattr(version_compare, "get_video_info",
                            lambda p: {"duration": 0, "width": 1920, "height": 1080})
        with pytest.raises(ValueError, match="Cannot determine duration"):
            version_compare._extract_frames("/fake.mp4", tmp_dir)

    def test_compute_ssim_psnr_mocked(self, monkeypatch):
        import subprocess
        from opencut.core import version_compare

        mock_result = types.SimpleNamespace(
            returncode=0,
            stderr="SSIM All:0.987654 (blah)\n",
            stdout="",
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        result = version_compare._compute_ssim_psnr("a.png", "b.png")
        assert result["ssim"] == pytest.approx(0.987654, abs=0.001)

    def test_get_loudness_mocked(self, monkeypatch):
        import subprocess
        from opencut.core import version_compare

        mock_result = types.SimpleNamespace(
            returncode=0,
            stderr='{\n"input_i" : "-14.5"\n}',
            stdout="",
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        result = version_compare._get_loudness("/fake.mp4")
        assert result == pytest.approx(-14.5)

    def test_compare_audio_mocked(self, monkeypatch):
        from opencut.core import version_compare

        monkeypatch.setattr(version_compare, "_get_loudness",
                            lambda p: -14.0 if "v1" in p else -12.0)
        result = version_compare.compare_audio("v1.mp4", "v2.mp4")
        assert result["loudness_delta"] == pytest.approx(2.0)

    def test_build_comparison_video_invalid_mode(self):
        from opencut.core.version_compare import _build_comparison_video
        with pytest.raises(ValueError, match="Invalid comparison mode"):
            _build_comparison_video("a.mp4", "b.mp4", "bad_mode", "out.mp4")


# ============================================================================
# 3. APPROVAL WORKFLOW — Core Logic
# ============================================================================

class TestApprovalWorkflowCore:
    """Unit tests for opencut.core.approval_workflow."""

    def test_create_workflow(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", project_name="Test")
        assert wf.current_stage == "draft"
        assert wf.project_id == "proj1"
        assert wf.id

    def test_stage_order(self):
        from opencut.core.approval_workflow import STAGES
        assert STAGES == ("draft", "internal_review", "client_review", "approved", "final")

    def test_advance(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1")
        result = wf.advance("editor1")
        assert result["current_stage"] == "internal_review"
        assert result["from_stage"] == "draft"

    def test_advance_through_all_stages(self):
        from opencut.core.approval_workflow import ApprovalWorkflow, STAGES
        wf = ApprovalWorkflow(project_id="proj1")
        for i in range(len(STAGES) - 1):
            wf.advance("admin")
        assert wf.current_stage == "final"

    def test_advance_at_final_raises(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="final")
        with pytest.raises(ValueError, match="final stage"):
            wf.advance("admin")

    def test_approve_individual(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="internal_review")
        result = wf.approve("editor1", notes="LGTM")
        assert result["action"] == "approve"
        assert result["auto_advanced"] is False

    def test_approve_auto_advance(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(
            project_id="proj1",
            current_stage="internal_review",
            required_approvers={"internal_review": ["editor1", "editor2"]},
        )
        wf.approve("editor1")
        result = wf.approve("editor2")
        assert result["auto_advanced"] is True
        assert wf.current_stage == "client_review"

    def test_approve_at_final_raises(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="final")
        with pytest.raises(ValueError, match="final stage"):
            wf.approve("admin")

    def test_reject_sends_to_draft(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="client_review")
        result = wf.reject("client", reason="Too dark")
        assert result["current_stage"] == "draft"
        assert wf.rejection_reason == "Too dark"

    def test_reject_draft_raises(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="draft")
        with pytest.raises(ValueError, match="Cannot reject a draft"):
            wf.reject("admin")

    def test_request_changes(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="internal_review")
        result = wf.request_changes("supervisor", notes="Fix audio levels")
        assert result["action"] == "request_changes"
        assert "Fix audio levels" in wf.change_requests

    def test_request_changes_on_draft_raises(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", current_stage="draft")
        with pytest.raises(ValueError, match="Cannot request changes on a draft"):
            wf.request_changes("admin")

    def test_get_blockers(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(
            project_id="proj1",
            current_stage="internal_review",
            required_approvers={"internal_review": ["editor1", "editor2"]},
        )
        wf.approve("editor1")
        blockers = wf.get_blockers()
        assert blockers == ["editor2"]

    def test_is_overdue(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(
            project_id="proj1",
            deadline=time.time() - 100,  # in the past
        )
        assert wf.is_overdue is True

    def test_not_overdue_no_deadline(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", deadline=0)
        assert wf.is_overdue is False

    def test_not_overdue_when_final(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(
            project_id="proj1",
            current_stage="final",
            deadline=time.time() - 100,
        )
        assert wf.is_overdue is False

    def test_age_hours(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1",
                              created_at=time.time() - 7200)
        assert wf.age_hours >= 1.99

    def test_history_recorded(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1")
        wf.advance("admin", notes="starting")
        assert len(wf.history) == 1
        assert wf.history[0].action == "manual_advance"
        assert wf.history[0].actor == "admin"

    def test_from_dict_round_trip(self):
        from opencut.core.approval_workflow import ApprovalWorkflow
        wf = ApprovalWorkflow(project_id="proj1", project_name="Test")
        wf.advance("admin")
        d = wf.to_dict()
        wf2 = ApprovalWorkflow.from_dict(d.copy())
        assert wf2.current_stage == wf.current_stage
        assert len(wf2.history) == len(wf.history)

    def test_workflow_manager_create_and_get(self, tmp_dir, monkeypatch):
        from opencut.core import approval_workflow
        monkeypatch.setattr(approval_workflow, "_APPROVALS_DIR", tmp_dir)
        from opencut.core.approval_workflow import WorkflowManager
        mgr = WorkflowManager()
        wf = mgr.create("project_x", project_name="Project X")
        assert mgr.get(wf.id) is not None
        assert mgr.get_by_project("project_x") is not None

    def test_workflow_manager_list(self, tmp_dir, monkeypatch):
        from opencut.core import approval_workflow
        monkeypatch.setattr(approval_workflow, "_APPROVALS_DIR", tmp_dir)
        from opencut.core.approval_workflow import WorkflowManager
        mgr = WorkflowManager()
        mgr.create("p1")
        mgr.create("p2")
        wfs = mgr.list_workflows()
        assert len(wfs) == 2

    def test_workflow_manager_dashboard(self, tmp_dir, monkeypatch):
        from opencut.core import approval_workflow
        monkeypatch.setattr(approval_workflow, "_APPROVALS_DIR", tmp_dir)
        from opencut.core.approval_workflow import WorkflowManager
        mgr = WorkflowManager()
        mgr.create("p1")
        dash = mgr.dashboard()
        assert dash["total_active"] == 1

    def test_workflow_manager_delete(self, tmp_dir, monkeypatch):
        from opencut.core import approval_workflow
        monkeypatch.setattr(approval_workflow, "_APPROVALS_DIR", tmp_dir)
        from opencut.core.approval_workflow import WorkflowManager
        mgr = WorkflowManager()
        wf = mgr.create("p1")
        assert mgr.delete(wf.id) is True
        assert mgr.get(wf.id) is None


# ============================================================================
# 4. EDIT HISTORY — Core Logic
# ============================================================================

class TestEditHistoryCore:
    """Unit tests for opencut.core.edit_history."""

    def _history(self, name="default"):
        """Create a fresh EditHistory with a unique ID to avoid disk leakage."""
        from opencut.core.edit_history import EditHistory
        import uuid as _uuid
        return EditHistory(f"{name}_{_uuid.uuid4().hex[:8]}")

    def test_add_entry(self):
        h = self._history("add")
        entry = h.add_entry("trim", parameters={"start": 0, "end": 10},
                            input_file="a.mp4", user="editor1")
        assert entry.operation_type == "trim"
        assert entry.parameters == {"start": 0, "end": 10}

    def test_add_entry_requires_operation_type(self):
        h = self._history("req_op")
        with pytest.raises(ValueError, match="operation_type is required"):
            h.add_entry("")

    def test_get_entries(self):
        h = self._history("get_entries")
        h.add_entry("op1", user="a")
        h.add_entry("op2", user="b")
        entries = h.get_entries()
        assert len(entries) == 2

    def test_get_entry_by_id(self):
        h = self._history("get_id")
        entry = h.add_entry("trim", user="a")
        found = h.get_entry(entry.id)
        assert found is not None
        assert found.operation_type == "trim"

    def test_filter_by_operation(self):
        h = self._history("filter_op")
        h.add_entry("trim")
        h.add_entry("color_grade")
        h.add_entry("trim")
        trimmed = h.filter_by_operation("trim")
        assert len(trimmed) == 2

    def test_filter_by_user(self):
        h = self._history("filter_user")
        h.add_entry("op1", user="Alice")
        h.add_entry("op2", user="Bob")
        h.add_entry("op3", user="alice")
        alice = h.filter_by_user("Alice")
        assert len(alice) == 2

    def test_filter_by_session(self):
        h = self._history("filter_session")
        h.add_entry("op1", session_id="s1")
        h.add_entry("op2", session_id="s2")
        h.add_entry("op3", session_id="s1")
        s1 = h.filter_by_session("s1")
        assert len(s1) == 2

    def test_mark_undone(self):
        h = self._history("undo")
        entry = h.add_entry("trim", user="a")
        undone = h.mark_undone(entry.id)
        assert undone.undone is True
        assert undone.undone_at > 0

    def test_mark_undone_already_undone(self):
        h = self._history("double_undo")
        entry = h.add_entry("trim", user="a")
        h.mark_undone(entry.id)
        with pytest.raises(ValueError, match="already undone"):
            h.mark_undone(entry.id)

    def test_mark_undone_not_found(self):
        h = self._history("undo_nf")
        with pytest.raises(ValueError, match="not found"):
            h.mark_undone("nonexistent")

    def test_undo_last(self):
        h = self._history("undo_last")
        h.add_entry("op1")
        h.add_entry("op2")
        undone = h.undo_last()
        assert undone.operation_type == "op2"

    def test_undo_last_empty(self):
        h = self._history("undo_empty")
        result = h.undo_last()
        assert result is None

    def test_get_entries_excludes_undone(self):
        h = self._history("exclude_undone")
        e1 = h.add_entry("op1")
        h.add_entry("op2")
        h.mark_undone(e1.id)
        entries = h.get_entries(include_undone=False)
        assert len(entries) == 1
        assert entries[0].operation_type == "op2"

    def test_diff(self):
        h = self._history("diff")
        h.add_entry("op1")
        h.add_entry("op2")
        h.add_entry("op3")
        d = h.diff(0, 2)
        assert d["entry_count"] == 2
        assert "op1" in d["operation_counts"]

    def test_diff_invalid_index(self):
        h = self._history("diff_invalid")
        with pytest.raises(ValueError, match="non-negative"):
            h.diff(-1, 0)

    def test_export_json(self):
        h = self._history("export")
        h.add_entry("trim", user="a")
        exported = h.export_json()
        data = json.loads(exported)
        assert data["entry_count"] == 1

    def test_export_timeline(self):
        h = self._history("timeline")
        h.add_entry("trim")
        h.add_entry("color")
        tl = h.export_timeline()
        assert len(tl) == 2
        assert tl[0]["label"] == "trim"

    def test_export_replay(self):
        h = self._history("replay")
        h.add_entry("silence_remove", input_file="a.mp4",
                     parameters={"threshold": -30})
        calls = h.export_replay()
        assert len(calls) == 1
        assert calls[0]["operation"] == "silence_remove"
        assert calls[0]["body"]["threshold"] == -30

    def test_statistics(self):
        h = self._history("stats")
        h.add_entry("trim", duration_sec=2.5, user="a")
        h.add_entry("trim", duration_sec=3.0, user="a")
        h.add_entry("color", duration_sec=5.0, user="b")
        stats = h.statistics()
        assert stats["total_entries"] == 3
        assert stats["most_used_operations"][0]["operation"] == "trim"
        assert stats["avg_duration_by_op"]["trim"] == pytest.approx(2.75, abs=0.01)

    def test_statistics_empty(self):
        h = self._history("stats_empty")
        stats = h.statistics()
        assert stats["total_entries"] == 0

    def test_history_entry_from_dict(self):
        from opencut.core.edit_history import HistoryEntry
        e = HistoryEntry(operation_type="trim", user="a")
        d = e.to_dict()
        e2 = HistoryEntry.from_dict(d)
        assert e2.operation_type == e.operation_type
        assert e2.id == e.id

    def test_entry_count_property(self):
        h = self._history("count")
        assert h.entry_count == 0
        h.add_entry("op1")
        assert h.entry_count == 1


# ============================================================================
# 5. SHARED PRESETS — Core Logic
# ============================================================================

class TestSharedPresetsCore:
    """Unit tests for opencut.core.shared_presets."""

    def test_add_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        preset = SharedPreset(name="Warm LUT", category="color_grades",
                              parameters={"temperature": 6500})
        result = lib.add(preset)
        assert result.name == "Warm LUT"
        assert result.param_hash

    def test_add_preset_no_name_raises(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        with pytest.raises(ValueError, match="name is required"):
            lib.add(SharedPreset(name="", parameters={}))

    def test_get_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Test", parameters={"a": 1}))
        got = lib.get(p.id)
        assert got is not None
        assert got.name == "Test"

    def test_update_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Original", parameters={}))
        updated = lib.update(p.id, name="Renamed")
        assert updated.name == "Renamed"

    def test_delete_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Delete me", parameters={}))
        assert lib.delete(p.id) is True
        assert lib.get(p.id) is None

    def test_delete_nonexistent(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary
        lib = PresetLibrary()
        assert lib.delete("nonexistent") is False

    def test_list_presets_by_category(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        lib.add(SharedPreset(name="Color1", category="color_grades", parameters={}))
        lib.add(SharedPreset(name="Audio1", category="audio_chains", parameters={}))
        colors = lib.list_presets(category="color_grades")
        assert len(colors) == 1
        assert colors[0].name == "Color1"

    def test_list_presets_by_author(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        lib.add(SharedPreset(name="A", author="Alice", parameters={}))
        lib.add(SharedPreset(name="B", author="Bob", parameters={}))
        alice = lib.list_presets(author="Alice")
        assert len(alice) == 1

    def test_list_presets_by_tags(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        lib.add(SharedPreset(name="A", tags=["warm", "film"], parameters={}))
        lib.add(SharedPreset(name="B", tags=["cool"], parameters={}))
        warm = lib.list_presets(tags=["warm"])
        assert len(warm) == 1

    def test_list_presets_search(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        lib.add(SharedPreset(name="Cinematic Color", parameters={}))
        lib.add(SharedPreset(name="Flat Profile", parameters={}))
        found = lib.list_presets(search="cinematic")
        assert len(found) == 1

    def test_rate_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Rated", parameters={}))
        lib.rate_preset(p.id, 5)
        lib.rate_preset(p.id, 3)
        rated = lib.get(p.id)
        assert rated.rating == 4.0
        assert rated.rating_count == 2

    def test_rate_invalid(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Bad", parameters={}))
        with pytest.raises(ValueError, match="between 1 and 5"):
            lib.rate_preset(p.id, 6)

    def test_duplicate_detection(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        params = {"contrast": 1.2, "saturation": 0.9}
        p1 = lib.add(SharedPreset(name="Original", parameters=params))
        p2 = SharedPreset(name="Copy", parameters=params)
        dupes = lib.find_duplicates(p2)
        assert len(dupes) == 1
        assert dupes[0].id == p1.id

    def test_check_duplicate(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        params = {"key": "value"}
        lib.add(SharedPreset(name="Existing", parameters=params))
        dupe = lib.check_duplicate(params)
        assert dupe is not None
        assert dupe.name == "Existing"

    def test_export_import_preset(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="Export Me", category="color_grades",
                                 parameters={"gamma": 2.2}))
        content = lib.export_preset(p.id)
        data = json.loads(content)
        assert data["metadata"]["name"] == "Export Me"
        imported = lib.import_preset(content)
        assert imported.name == "Export Me"
        assert imported.id != p.id  # new ID on import

    def test_export_to_file(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = lib.add(SharedPreset(name="File Export", parameters={"x": 1}))
        out = os.path.join(tmp_dir, "test.opencut-preset")
        lib.export_preset_to_file(p.id, out)
        assert os.path.isfile(out)

    def test_import_from_file(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        p = SharedPreset(name="File Preset", parameters={"y": 2})
        path = os.path.join(tmp_dir, "import.opencut-preset")
        with open(path, "w") as fh:
            fh.write(p.to_preset_file())
        imported = lib.import_from_file(path)
        assert imported.name == "File Preset"

    def test_import_file_not_found(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary
        lib = PresetLibrary()
        with pytest.raises(FileNotFoundError):
            lib.import_from_file("/nonexistent.opencut-preset")

    def test_batch_import(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        batch_dir = os.path.join(tmp_dir, "batch")
        os.makedirs(batch_dir)
        for i in range(3):
            p = SharedPreset(name=f"Batch{i}", parameters={"i": i})
            with open(os.path.join(batch_dir, f"p{i}.opencut-preset"), "w") as fh:
                fh.write(p.to_preset_file())
        result = lib.batch_import(batch_dir)
        assert result["imported"] == 3
        assert result["skipped"] == 0

    def test_merge_keep_local(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        local = lib.add(SharedPreset(name="Local", parameters={"v": 1}))
        remote = SharedPreset(name="Remote", parameters={"v": 2})
        result = lib.merge_preset(local.id, remote, strategy="keep_local")
        assert result.name == "Local"

    def test_merge_keep_remote(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        local = lib.add(SharedPreset(name="Local", parameters={"v": 1}))
        remote = SharedPreset(name="Remote", parameters={"v": 2})
        result = lib.merge_preset(local.id, remote, strategy="keep_remote")
        assert result.name == "Remote"
        assert result.id == local.id

    def test_merge_keep_newest(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        local = lib.add(SharedPreset(name="Older", parameters={"v": 1}))
        remote = SharedPreset(name="Newer", parameters={"v": 2},
                              updated_at=time.time() + 1000)
        result = lib.merge_preset(local.id, remote, strategy="keep_newest")
        assert result.name == "Newer"

    def test_merge_invalid_strategy(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        local = lib.add(SharedPreset(name="A", parameters={}))
        remote = SharedPreset(name="B", parameters={})
        with pytest.raises(ValueError, match="Invalid merge strategy"):
            lib.merge_preset(local.id, remote, strategy="bad")

    def test_preset_file_round_trip(self):
        from opencut.core.shared_presets import SharedPreset
        p = SharedPreset(name="Test", category="audio_chains",
                         author="me", parameters={"eq": [100, 200]})
        content = p.to_preset_file()
        p2 = SharedPreset.from_preset_file(content)
        assert p2.name == "Test"
        assert p2.category == "audio_chains"
        assert p2.parameters == {"eq": [100, 200]}

    def test_invalid_preset_file(self):
        from opencut.core.shared_presets import SharedPreset
        with pytest.raises(ValueError, match="Invalid preset file"):
            SharedPreset.from_preset_file("not json")

    def test_stats(self, tmp_dir, monkeypatch):
        from opencut.core import shared_presets
        monkeypatch.setattr(shared_presets, "_PRESETS_DIR", tmp_dir)
        monkeypatch.setattr(shared_presets, "_INDEX_FILE",
                            os.path.join(tmp_dir, "index.json"))
        from opencut.core.shared_presets import PresetLibrary, SharedPreset
        lib = PresetLibrary()
        lib.add(SharedPreset(name="A", category="color_grades",
                             author="x", parameters={}))
        lib.add(SharedPreset(name="B", category="audio_chains",
                             author="y", parameters={}))
        stats = lib.stats()
        assert stats["total"] == 2
        assert "color_grades" in stats["by_category"]

    def test_invalid_category_defaults(self):
        from opencut.core.shared_presets import SharedPreset
        p = SharedPreset(name="Test", category="invalid_cat", parameters={})
        assert p.category == "export_profiles"

    def test_param_hash_deterministic(self):
        from opencut.core.shared_presets import _param_hash
        h1 = _param_hash({"a": 1, "b": 2})
        h2 = _param_hash({"b": 2, "a": 1})
        assert h1 == h2


# ============================================================================
# 6. ROUTE TESTS — collab_review_bp
# ============================================================================

class TestCollabReviewRoutes:
    """Integration tests for the collab_review_bp routes."""

    def test_add_comment_route(self, client, csrf_token):
        resp = client.post("/api/review/comments",
                           headers=csrf_headers(csrf_token),
                           json={
                               "project_path": "/fake/project.prproj",
                               "text": "Fix this",
                               "author": "editor1",
                               "timestamp_sec": 10.0,
                           })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["comment"]["text"] == "Fix this"

    def test_add_comment_no_text(self, client, csrf_token):
        resp = client.post("/api/review/comments",
                           headers=csrf_headers(csrf_token),
                           json={"project_path": "/fake/p.prproj", "text": ""})
        assert resp.status_code == 400

    def test_add_comment_no_project(self, client, csrf_token):
        resp = client.post("/api/review/comments",
                           headers=csrf_headers(csrf_token),
                           json={"text": "Hello"})
        assert resp.status_code == 400

    def test_list_comments_route(self, client, csrf_token):
        # Add a comment first
        client.post("/api/review/comments",
                    headers=csrf_headers(csrf_token),
                    json={"project_path": "/fake/p.prproj",
                          "text": "Note", "author": "a"})
        resp = client.get("/api/review/comments?project_path=/fake/p.prproj")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1

    def test_list_comments_no_project(self, client):
        resp = client.get("/api/review/comments")
        assert resp.status_code == 400

    def test_resolve_comment_route(self, client, csrf_token):
        resp = client.post("/api/review/comments",
                           headers=csrf_headers(csrf_token),
                           json={"project_path": "/fake/p.prproj",
                                 "text": "Resolve me", "author": "a"})
        comment_id = resp.get_json()["comment"]["id"]
        resp2 = client.put(f"/api/review/comments/{comment_id}/resolve",
                           headers=csrf_headers(csrf_token),
                           json={"project_path": "/fake/p.prproj"})
        assert resp2.status_code == 200
        assert resp2.get_json()["comment"]["status"] == "resolved"

    def test_delete_comment_route(self, client, csrf_token):
        resp = client.post("/api/review/comments",
                           headers=csrf_headers(csrf_token),
                           json={"project_path": "/fake/p.prproj",
                                 "text": "Delete me", "author": "a"})
        comment_id = resp.get_json()["comment"]["id"]
        resp2 = client.delete(f"/api/review/comments/{comment_id}",
                              headers=csrf_headers(csrf_token),
                              json={"project_path": "/fake/p.prproj"})
        assert resp2.status_code == 200
        assert resp2.get_json()["deleted"] == comment_id

    def test_delete_nonexistent_comment(self, client, csrf_token):
        resp = client.delete("/api/review/comments/nonexistent",
                             headers=csrf_headers(csrf_token),
                             json={"project_path": "/fake/p.prproj"})
        assert resp.status_code == 404

    def test_export_comments_route(self, client, csrf_token):
        client.post("/api/review/comments",
                    headers=csrf_headers(csrf_token),
                    json={"project_path": "/fake/p.prproj",
                          "text": "Export me", "author": "a"})
        resp = client.post("/api/review/comments/export",
                           headers=csrf_headers(csrf_token),
                           json={"project_path": "/fake/p.prproj",
                                 "format": "json"})
        assert resp.status_code == 200
        assert resp.get_json()["format"] == "json"

    def test_approval_create_route(self, client, csrf_token):
        resp = client.post("/api/approval/create",
                           headers=csrf_headers(csrf_token),
                           json={"project_id": "test_proj",
                                 "project_name": "Test Project"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["workflow"]["current_stage"] == "draft"

    def test_approval_create_no_project(self, client, csrf_token):
        resp = client.post("/api/approval/create",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_approval_status_dashboard(self, client, csrf_token):
        client.post("/api/approval/create",
                    headers=csrf_headers(csrf_token),
                    json={"project_id": "dash_proj"})
        resp = client.get("/api/approval/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_active" in data

    def test_approval_advance_route(self, client, csrf_token):
        resp = client.post("/api/approval/create",
                           headers=csrf_headers(csrf_token),
                           json={"project_id": "adv_proj"})
        wf_id = resp.get_json()["workflow"]["id"]
        resp2 = client.post("/api/approval/advance",
                            headers=csrf_headers(csrf_token),
                            json={"workflow_id": wf_id,
                                  "action": "approve",
                                  "actor": "editor1",
                                  "notes": "LGTM"})
        assert resp2.status_code == 200

    def test_approval_advance_missing_fields(self, client, csrf_token):
        resp = client.post("/api/approval/advance",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_upload_preset_route(self, client, csrf_token):
        resp = client.post("/api/presets/shared",
                           headers=csrf_headers(csrf_token),
                           json={
                               "name": "Warm Color",
                               "category": "color_grades",
                               "parameters": {"temp": 6500},
                               "author": "colorist",
                           })
        assert resp.status_code == 200
        assert resp.get_json()["preset"]["name"] == "Warm Color"

    def test_upload_preset_no_name(self, client, csrf_token):
        resp = client.post("/api/presets/shared",
                           headers=csrf_headers(csrf_token),
                           json={"parameters": {}})
        assert resp.status_code == 400

    def test_list_presets_route(self, client, csrf_token):
        client.post("/api/presets/shared",
                    headers=csrf_headers(csrf_token),
                    json={"name": "A", "category": "color_grades",
                          "parameters": {"x": 1}})
        resp = client.get("/api/presets/shared")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1

    def test_list_presets_filter_category(self, client, csrf_token):
        client.post("/api/presets/shared",
                    headers=csrf_headers(csrf_token),
                    json={"name": "Color", "category": "color_grades",
                          "parameters": {}})
        client.post("/api/presets/shared",
                    headers=csrf_headers(csrf_token),
                    json={"name": "Audio", "category": "audio_chains",
                          "parameters": {}})
        resp = client.get("/api/presets/shared?category=color_grades")
        data = resp.get_json()
        assert all(p["category"] == "color_grades" for p in data["presets"])

    def test_export_history_route(self, client, csrf_token):
        # Add a history entry first via the core module
        from opencut.core.edit_history import record_operation
        record_operation("hist_proj", "trim", user="a")

        resp = client.post("/api/edit-history/export",
                           headers=csrf_headers(csrf_token),
                           json={"project_id": "hist_proj", "format": "json"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["format"] == "json"

    def test_export_history_no_project(self, client, csrf_token):
        resp = client.post("/api/edit-history/export",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_csrf_required_on_post(self, client):
        """POST without CSRF token should fail with 403."""
        resp = client.post("/api/review/comments",
                           headers={"Content-Type": "application/json"},
                           json={"project_path": "/fake/p.prproj",
                                 "text": "No CSRF"})
        assert resp.status_code == 403
