"""
OpenCut Workflow & Developer Feature Tests

Comprehensive tests for all 8 features:
  1. Undo Stack (undo_stack.py) -- push, undo, history, clear
  2. EDL/AAF (edl_aaf.py) -- EDL export/import, AAF stub export
  3. Project Archive (project_archive.py) -- create, restore, list contents
  4. Scripting Console (scripting_console.py) -- execute, sandbox, modules
  5. Macro Recorder (macro_recorder.py) -- start, stop, record, play, save, load
  6. Edit Snapshots (edit_snapshots.py) -- create, restore, list, compare
  7. Through-Edit (through_edit.py) -- detect, merge
  8. Ripple Edit (ripple_edit.py) -- detect gaps, ripple close

65+ tests covering core logic and route endpoints.
"""

import json
import os
import shutil
import tempfile

import pytest

from tests.conftest import csrf_headers

# ============================================================================
# Helpers
# ============================================================================

@pytest.fixture(autouse=True)
def _clean_undo_sessions():
    """Reset undo stack sessions between tests."""
    yield
    from opencut.core.undo_stack import _lock, _sessions
    with _lock:
        _sessions.clear()


@pytest.fixture(autouse=True)
def _clean_macro_sessions():
    """Reset macro recorder sessions between tests."""
    yield
    from opencut.core.macro_recorder import _lock, _sessions
    with _lock:
        _sessions.clear()


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="opencut_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================================
# 1. UNDO STACK -- Core Logic
# ============================================================================

class TestUndoStackCore:
    """Unit tests for opencut.core.undo_stack."""

    def test_push_operation_basic(self):
        from opencut.core.undo_stack import push_operation
        rec = push_operation({
            "operation": "trim",
            "input_file": "/a.mp4",
            "output_file": "/b.mp4",
            "parameters": {"start": 0, "end": 10},
        }, session_id="test_push")
        assert rec.operation == "trim"
        assert rec.input_file == "/a.mp4"
        assert rec.output_file == "/b.mp4"
        assert rec.parameters == {"start": 0, "end": 10}
        assert rec.undone is False

    def test_push_requires_operation_name(self):
        from opencut.core.undo_stack import push_operation
        with pytest.raises(ValueError, match="required"):
            push_operation({"operation": ""}, session_id="test_req")

    def test_get_history_returns_ordered_list(self):
        from opencut.core.undo_stack import get_history, push_operation
        for i in range(3):
            push_operation({
                "operation": f"op_{i}",
                "input_file": f"/in_{i}.mp4",
                "output_file": f"/out_{i}.mp4",
            }, session_id="test_hist")
        history = get_history(session_id="test_hist")
        assert len(history) == 3
        assert history[0]["operation"] == "op_0"
        assert history[2]["operation"] == "op_2"

    def test_undo_last_marks_undone(self):
        from opencut.core.undo_stack import push_operation, undo_last
        push_operation({
            "operation": "color_grade",
            "input_file": "/a.mp4",
            "output_file": "/b.mp4",
        }, session_id="test_undo")
        result = undo_last(session_id="test_undo")
        assert result is not None
        assert result["reverted_operation"] == "color_grade"
        assert result["revert_to_file"] == "/a.mp4"

    def test_undo_empty_returns_none(self):
        from opencut.core.undo_stack import undo_last
        result = undo_last(session_id="test_empty")
        assert result is None

    def test_clear_history(self):
        from opencut.core.undo_stack import clear_history, get_history, push_operation
        push_operation({"operation": "x", "input_file": "", "output_file": ""},
                       session_id="test_clear")
        count = clear_history(session_id="test_clear")
        assert count == 1
        assert get_history(session_id="test_clear") == []

    def test_max_history_enforced(self):
        from opencut.core.undo_stack import MAX_HISTORY, get_history, push_operation
        for i in range(MAX_HISTORY + 5):
            push_operation({
                "operation": f"op_{i}",
                "input_file": "",
                "output_file": "",
            }, session_id="test_max")
        history = get_history(session_id="test_max")
        assert len(history) == MAX_HISTORY

    def test_undo_skips_already_undone(self):
        from opencut.core.undo_stack import push_operation, undo_last
        push_operation({"operation": "first", "input_file": "/a", "output_file": "/b"},
                       session_id="test_skip")
        push_operation({"operation": "second", "input_file": "/b", "output_file": "/c"},
                       session_id="test_skip")
        r1 = undo_last(session_id="test_skip")
        assert r1["reverted_operation"] == "second"
        r2 = undo_last(session_id="test_skip")
        assert r2["reverted_operation"] == "first"


# ============================================================================
# 2. EDL/AAF -- Core Logic
# ============================================================================

class TestEDLAAFCore:
    """Unit tests for opencut.core.edl_aaf."""

    def test_seconds_to_tc(self):
        from opencut.core.edl_aaf import _seconds_to_tc
        assert _seconds_to_tc(0, 30) == "00:00:00:00"
        assert _seconds_to_tc(1.0, 30) == "00:00:01:00"
        assert _seconds_to_tc(61.5, 30) == "00:01:01:15"

    def test_tc_to_seconds(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        assert _tc_to_seconds("00:00:00:00", 30) == 0.0
        assert _tc_to_seconds("00:00:01:00", 30) == 1.0

    def test_tc_to_seconds_invalid(self):
        from opencut.core.edl_aaf import _tc_to_seconds
        with pytest.raises(ValueError, match="Invalid timecode"):
            _tc_to_seconds("invalid", 30)

    def test_export_edl_creates_file(self, tmp_dir):
        from opencut.core.edl_aaf import export_edl
        out = os.path.join(tmp_dir, "test.edl")
        cuts = [
            {"reel": "AX", "channel": "V", "transition": "C",
             "source_in": 0.0, "source_out": 5.0,
             "record_in": 0.0, "record_out": 5.0,
             "clip_name": "clip1"},
        ]
        result = export_edl(cuts, out, title="Test EDL")
        assert os.path.isfile(result.output_path)
        assert result.event_count == 1
        content = open(out, encoding="utf-8").read()
        assert "TITLE: Test EDL" in content
        assert "clip1" in content

    def test_export_edl_empty_raises(self, tmp_dir):
        from opencut.core.edl_aaf import export_edl
        out = os.path.join(tmp_dir, "empty.edl")
        with pytest.raises(ValueError, match="empty"):
            export_edl([], out)

    def test_import_edl_roundtrip(self, tmp_dir):
        from opencut.core.edl_aaf import export_edl, import_edl
        out = os.path.join(tmp_dir, "roundtrip.edl")
        cuts = [
            {"reel": "AX", "channel": "V", "transition": "C",
             "source_in": "00:00:00:00", "source_out": "00:00:05:00",
             "record_in": "00:00:00:00", "record_out": "00:00:05:00",
             "clip_name": "Intro"},
            {"reel": "BL", "channel": "V", "transition": "C",
             "source_in": "00:00:10:00", "source_out": "00:00:15:00",
             "record_in": "00:00:05:00", "record_out": "00:00:10:00",
             "clip_name": "Main"},
        ]
        export_edl(cuts, out, title="Roundtrip Test")
        result = import_edl(out)
        assert result.title == "Roundtrip Test"
        assert result.event_count == 2
        assert result.cuts[0]["clip_name"] == "Intro"
        assert result.cuts[1]["reel"] == "BL"

    def test_import_edl_missing_file(self):
        from opencut.core.edl_aaf import import_edl
        with pytest.raises(FileNotFoundError):
            import_edl("/nonexistent/path.edl")

    def test_export_aaf_stub(self, tmp_dir):
        from opencut.core.edl_aaf import export_aaf_stub
        out = os.path.join(tmp_dir, "stub.json")
        cuts = [
            {"reel": "AX", "source_in": 0, "source_out": 5, "clip_name": "c1"},
        ]
        result = export_aaf_stub(cuts, out, title="AAF Test")
        assert result["format"] == "aaf_stub_v1"
        assert result["event_count"] == 1
        data = json.loads(open(out).read())
        assert data["title"] == "AAF Test"

    def test_export_aaf_stub_empty_raises(self, tmp_dir):
        from opencut.core.edl_aaf import export_aaf_stub
        out = os.path.join(tmp_dir, "empty.json")
        with pytest.raises(ValueError, match="empty"):
            export_aaf_stub([], out)


# ============================================================================
# 3. PROJECT ARCHIVE -- Core Logic
# ============================================================================

class TestProjectArchiveCore:
    """Unit tests for opencut.core.project_archive."""

    def test_create_archive_with_presets(self, tmp_dir):
        from opencut.core.project_archive import create_archive
        out = os.path.join(tmp_dir, "archive.zip")
        project_data = {
            "name": "Test Project",
            "source_files": [],
            "output_files": [],
            "workflows": [{"name": "wf1", "steps": []}],
            "presets": [{"name": "preset1"}],
        }
        result = create_archive(project_data, out)
        assert os.path.isfile(result.archive_path)
        assert result.manifest_included is True

    def test_create_archive_nothing_raises(self, tmp_dir):
        from opencut.core.project_archive import create_archive
        out = os.path.join(tmp_dir, "empty.zip")
        with pytest.raises(ValueError, match="Nothing to archive"):
            create_archive({"name": "empty"}, out)

    def test_restore_archive_roundtrip(self, tmp_dir):
        from opencut.core.project_archive import create_archive, restore_archive
        # Create a source file
        src = os.path.join(tmp_dir, "source.txt")
        with open(src, "w") as f:
            f.write("test data")
        archive_path = os.path.join(tmp_dir, "archive.zip")
        project_data = {
            "name": "Roundtrip",
            "source_files": [src],
            "output_files": [],
            "workflows": [],
            "presets": [],
        }
        create_archive(project_data, archive_path)
        dest = os.path.join(tmp_dir, "restored")
        result = restore_archive(archive_path, dest)
        assert result.files_restored >= 1
        assert os.path.isdir(dest)

    def test_restore_archive_missing(self):
        from opencut.core.project_archive import restore_archive
        with pytest.raises(FileNotFoundError):
            restore_archive("/nonexistent/archive.zip", "/tmp/dest")

    def test_list_archive_contents(self, tmp_dir):
        from opencut.core.project_archive import create_archive, list_archive_contents
        src = os.path.join(tmp_dir, "data.txt")
        with open(src, "w") as f:
            f.write("hello")
        archive_path = os.path.join(tmp_dir, "list_test.zip")
        create_archive({
            "name": "List Test",
            "source_files": [src],
            "output_files": [],
            "workflows": [],
            "presets": [],
        }, archive_path)
        result = list_archive_contents(archive_path)
        assert result["total_files"] >= 1
        assert "manifest" in result

    def test_list_archive_missing(self):
        from opencut.core.project_archive import list_archive_contents
        with pytest.raises(FileNotFoundError):
            list_archive_contents("/nonexistent.zip")


# ============================================================================
# 4. SCRIPTING CONSOLE -- Core Logic
# ============================================================================

class TestScriptingConsoleCore:
    """Unit tests for opencut.core.scripting_console."""

    def test_execute_simple_print(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["output"]

    def test_execute_with_context(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(x + y)", context={"x": 3, "y": 7})
        assert result["success"] is True
        assert "10" in result["output"]

    def test_execute_empty_code(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("")
        assert result["success"] is True

    def test_execute_syntax_error(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("def bad(")
        assert result["success"] is False
        assert "Syntax error" in result["error"]

    def test_execute_blocked_import(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import os")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_execute_allowed_import(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import math; print(math.pi)")
        assert result["success"] is True
        assert "3.14" in result["output"]

    def test_blocked_builtins(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("open('/etc/passwd')")
        assert result["success"] is False

    def test_blocked_dunder_access(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(''.__class__)")
        assert result["success"] is False
        assert "__class__" in result["error"]

    def test_get_available_modules(self):
        from opencut.core.scripting_console import get_available_modules
        modules = get_available_modules()
        assert isinstance(modules, list)
        assert "math" in modules
        assert "json" in modules
        assert "os" not in modules

    def test_create_sandbox_has_safe_builtins(self):
        from opencut.core.scripting_console import create_sandbox
        sandbox = create_sandbox()
        builtins = sandbox["__builtins__"]
        assert "print" in builtins or hasattr(builtins, "print")
        assert "exec" not in builtins
        assert "eval" not in builtins


# ============================================================================
# 5. MACRO RECORDER -- Core Logic
# ============================================================================

class TestMacroRecorderCore:
    """Unit tests for opencut.core.macro_recorder."""

    def test_start_stop_recording(self):
        from opencut.core.macro_recorder import start_recording, stop_recording
        result = start_recording(session_id="test_ss")
        assert result["recording"] is True
        macro = stop_recording(session_id="test_ss", name="Test Macro")
        assert macro.name == "Test Macro"
        assert len(macro.actions) == 0

    def test_record_action_while_recording(self):
        from opencut.core.macro_recorder import record_action, start_recording, stop_recording
        start_recording(session_id="test_rec")
        ok = record_action("/silence", {"threshold": -30}, session_id="test_rec")
        assert ok is True
        macro = stop_recording(session_id="test_rec")
        assert len(macro.actions) == 1
        assert macro.actions[0].endpoint == "/silence"

    def test_record_action_not_recording(self):
        from opencut.core.macro_recorder import record_action
        ok = record_action("/test", {}, session_id="not_recording")
        assert ok is False

    def test_stop_without_start_raises(self):
        from opencut.core.macro_recorder import stop_recording
        with pytest.raises(ValueError, match="No active recording"):
            stop_recording(session_id="never_started")

    def test_play_macro_dry_run(self):
        from opencut.core.macro_recorder import Macro, MacroAction, play_macro
        macro = Macro(
            name="Test",
            actions=[
                MacroAction(endpoint="/trim", params={"filepath": "/old.mp4", "start": 0}),
                MacroAction(endpoint="/export", params={"filepath": "/old.mp4"}),
            ],
        )
        results = play_macro(macro, target_file="/new.mp4")
        assert len(results) == 2
        assert results[0]["dry_run"] is True
        assert results[0]["params"]["filepath"] == "/new.mp4"

    def test_play_macro_with_executor(self):
        from opencut.core.macro_recorder import Macro, MacroAction, play_macro

        def mock_executor(endpoint, params):
            return {"ok": True, "endpoint": endpoint}

        macro = Macro(
            name="Exec Test",
            actions=[MacroAction(endpoint="/test", params={"filepath": "/x"})],
        )
        results = play_macro(macro, target_file="/y.mp4", executor=mock_executor)
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_save_load_macro(self, tmp_dir):
        from opencut.core.macro_recorder import Macro, MacroAction, load_macro, save_macro
        macro = Macro(
            name="Saved Macro",
            description="A test macro",
            actions=[MacroAction(endpoint="/trim", params={"start": 1})],
        )
        path = os.path.join(tmp_dir, "macro.json")
        save_macro(macro, path)
        assert os.path.isfile(path)
        loaded = load_macro(path)
        assert loaded.name == "Saved Macro"
        assert len(loaded.actions) == 1
        assert loaded.actions[0].endpoint == "/trim"

    def test_load_macro_missing(self):
        from opencut.core.macro_recorder import load_macro
        with pytest.raises(FileNotFoundError):
            load_macro("/nonexistent/macro.json")

    def test_macro_to_from_dict(self):
        from opencut.core.macro_recorder import Macro, MacroAction
        macro = Macro(
            name="Dict Test",
            actions=[MacroAction(endpoint="/a", params={"k": "v"})],
        )
        d = macro.to_dict()
        restored = Macro.from_dict(d)
        assert restored.name == "Dict Test"
        assert restored.actions[0].endpoint == "/a"

    def test_already_recording_returns_status(self):
        from opencut.core.macro_recorder import start_recording
        start_recording(session_id="test_double")
        result = start_recording(session_id="test_double")
        assert result["recording"] is True
        assert "Already recording" in result["message"]


# ============================================================================
# 6. EDIT SNAPSHOTS -- Core Logic
# ============================================================================

class TestEditSnapshotsCore:
    """Unit tests for opencut.core.edit_snapshots."""

    def test_create_snapshot(self, tmp_dir):
        from opencut.core.edit_snapshots import _SNAPSHOTS_DIR, create_snapshot
        # Use a unique project_id so we don't collide with other tests
        pid = f"test_create_{os.getpid()}"
        result = create_snapshot(
            "snap1",
            {"job_history": [{"type": "trim"}], "output_files": ["/a.mp4"],
             "parameters": {"crf": 18}},
            project_id=pid,
        )
        assert result["name"] == "snap1"
        assert result["job_count"] == 1
        assert result["output_count"] == 1
        # Cleanup
        snap_dir = os.path.join(_SNAPSHOTS_DIR, pid)
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_create_snapshot_empty_name_raises(self):
        from opencut.core.edit_snapshots import create_snapshot
        with pytest.raises(ValueError, match="required"):
            create_snapshot("", {}, project_id="test_empty_name")

    def test_restore_snapshot(self, tmp_dir):
        from opencut.core.edit_snapshots import (
            _SNAPSHOTS_DIR,
            create_snapshot,
            restore_snapshot,
        )
        pid = f"test_restore_{os.getpid()}"
        create_snapshot("snap_r", {"parameters": {"x": 42}}, project_id=pid)
        result = restore_snapshot("snap_r", project_id=pid)
        assert result["parameters"]["x"] == 42
        snap_dir = os.path.join(_SNAPSHOTS_DIR, pid)
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_restore_missing_snapshot(self):
        from opencut.core.edit_snapshots import restore_snapshot
        with pytest.raises(FileNotFoundError):
            restore_snapshot("nonexistent", project_id="missing_project")

    def test_list_snapshots(self, tmp_dir):
        from opencut.core.edit_snapshots import (
            _SNAPSHOTS_DIR,
            create_snapshot,
            list_snapshots,
        )
        pid = f"test_list_{os.getpid()}"
        create_snapshot("s1", {"parameters": {}}, project_id=pid)
        create_snapshot("s2", {"parameters": {}}, project_id=pid)
        snaps = list_snapshots(project_id=pid)
        assert len(snaps) >= 2
        snap_dir = os.path.join(_SNAPSHOTS_DIR, pid)
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_compare_snapshots(self, tmp_dir):
        from opencut.core.edit_snapshots import (
            _SNAPSHOTS_DIR,
            compare_snapshots,
            create_snapshot,
        )
        pid = f"test_compare_{os.getpid()}"
        create_snapshot("before", {
            "parameters": {"crf": 18, "preset": "fast"},
            "output_files": ["/a.mp4"],
        }, project_id=pid)
        create_snapshot("after", {
            "parameters": {"crf": 23, "preset": "fast"},
            "output_files": ["/a.mp4", "/b.mp4"],
        }, project_id=pid)
        result = compare_snapshots("before", "after", project_id=pid)
        assert result["snapshot_a"] == "before"
        assert result["snapshot_b"] == "after"
        assert len(result["parameter_diffs"]) >= 1
        assert "/b.mp4" in result["files"]["added"]
        snap_dir = os.path.join(_SNAPSHOTS_DIR, pid)
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_compare_missing_snapshot(self):
        from opencut.core.edit_snapshots import compare_snapshots
        with pytest.raises(FileNotFoundError):
            compare_snapshots("a", "b", project_id="no_project")


# ============================================================================
# 7. THROUGH-EDIT -- Core Logic
# ============================================================================

class TestThroughEditCore:
    """Unit tests for opencut.core.through_edit."""

    def test_detect_continuous_cuts(self):
        from opencut.core.through_edit import detect_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
        ]
        result = detect_through_edits(cuts)
        assert result.mergeable_count == 1
        assert result.through_edits[0].source_file == "a.mp4"

    def test_detect_no_through_edits(self):
        from opencut.core.through_edit import detect_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "b.mp4", "source_in": 0, "source_out": 5},
        ]
        result = detect_through_edits(cuts)
        assert result.mergeable_count == 0

    def test_detect_gap_beyond_tolerance(self):
        from opencut.core.through_edit import detect_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "a.mp4", "source_in": 6, "source_out": 10},
        ]
        result = detect_through_edits(cuts, tolerance=0.05)
        assert result.mergeable_count == 0

    def test_detect_single_cut(self):
        from opencut.core.through_edit import detect_through_edits
        result = detect_through_edits([{"source_file": "a.mp4", "source_in": 0, "source_out": 5}])
        assert result.mergeable_count == 0
        assert result.total_cuts == 1

    def test_detect_different_tracks(self):
        from opencut.core.through_edit import detect_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5, "track": "V1"},
            {"source_file": "a.mp4", "source_in": 5, "source_out": 10, "track": "V2"},
        ]
        result = detect_through_edits(cuts)
        assert result.mergeable_count == 0

    def test_merge_through_edits(self):
        from opencut.core.through_edit import merge_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
            {"source_file": "b.mp4", "source_in": 0, "source_out": 3},
        ]
        merged = merge_through_edits(cuts)
        assert len(merged) == 2
        assert merged[0]["source_out"] == 10

    def test_merge_with_explicit_indices(self):
        from opencut.core.through_edit import merge_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
            {"source_file": "a.mp4", "source_in": 10, "source_out": 15},
        ]
        merged = merge_through_edits(cuts, indices=[(0, 1)])
        assert len(merged) == 2  # only first pair merged

    def test_merge_chain(self):
        from opencut.core.through_edit import merge_through_edits
        cuts = [
            {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
            {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
            {"source_file": "a.mp4", "source_in": 10, "source_out": 15},
        ]
        merged = merge_through_edits(cuts)
        assert len(merged) == 1
        assert merged[0]["source_out"] == 15


# ============================================================================
# 8. RIPPLE EDIT -- Core Logic
# ============================================================================

class TestRippleEditCore:
    """Unit tests for opencut.core.ripple_edit."""

    def test_detect_gaps_basic(self):
        from opencut.core.ripple_edit import detect_gaps
        items = [
            {"start": 0, "end": 5},
            {"start": 7, "end": 12},
        ]
        result = detect_gaps(items)
        assert len(result.gaps) == 1
        assert result.gaps[0].duration == pytest.approx(2.0)
        assert result.total_gap_duration == pytest.approx(2.0)

    def test_detect_gaps_initial_gap(self):
        from opencut.core.ripple_edit import detect_gaps
        items = [{"start": 3, "end": 8}]
        result = detect_gaps(items)
        assert len(result.gaps) == 1
        assert result.gaps[0].start == 0.0
        assert result.gaps[0].duration == pytest.approx(3.0)

    def test_detect_no_gaps(self):
        from opencut.core.ripple_edit import detect_gaps
        items = [
            {"start": 0, "end": 5},
            {"start": 5, "end": 10},
        ]
        result = detect_gaps(items)
        assert len(result.gaps) == 0

    def test_detect_gaps_empty_timeline(self):
        from opencut.core.ripple_edit import detect_gaps
        result = detect_gaps([])
        assert len(result.gaps) == 0
        assert result.item_count == 0

    def test_detect_gaps_with_duration_field(self):
        from opencut.core.ripple_edit import detect_gaps
        items = [
            {"start": 0, "duration": 5},
            {"start": 8, "duration": 4},
        ]
        result = detect_gaps(items)
        assert len(result.gaps) == 1
        assert result.gaps[0].duration == pytest.approx(3.0)

    def test_ripple_close_basic(self):
        from opencut.core.ripple_edit import ripple_close_gaps
        items = [
            {"start": 0, "end": 5},
            {"start": 7, "end": 12},
        ]
        result = ripple_close_gaps(items)
        assert result.gaps_closed == 1
        assert result.total_shift == pytest.approx(2.0)
        assert result.items[1]["start"] == pytest.approx(5.0)
        assert result.items[1]["end"] == pytest.approx(10.0)

    def test_ripple_close_locked_tracks(self):
        from opencut.core.ripple_edit import ripple_close_gaps
        items = [
            {"start": 0, "end": 5, "track": "V1"},
            {"start": 7, "end": 12, "track": "A1"},
            {"start": 7, "end": 12, "track": "V1"},
        ]
        result = ripple_close_gaps(items, locked_tracks=["A1"])
        assert "A1" in result.locked_tracks_skipped
        # A1 track item should not have moved
        a1_item = next(i for i in result.items if i["track"] == "A1")
        assert a1_item["start"] == pytest.approx(7.0)

    def test_ripple_close_empty(self):
        from opencut.core.ripple_edit import ripple_close_gaps
        result = ripple_close_gaps([])
        assert result.gaps_closed == 0
        assert result.items == []

    def test_ripple_close_no_gaps(self):
        from opencut.core.ripple_edit import ripple_close_gaps
        items = [
            {"start": 0, "end": 5},
            {"start": 5, "end": 10},
        ]
        result = ripple_close_gaps(items)
        assert result.gaps_closed == 0

    def test_ripple_preserves_duration_field(self):
        from opencut.core.ripple_edit import ripple_close_gaps
        items = [
            {"start": 0, "end": 5, "duration": 5},
            {"start": 8, "end": 13, "duration": 5},
        ]
        result = ripple_close_gaps(items)
        for item in result.items:
            assert item["duration"] == pytest.approx(5.0)


# ============================================================================
# ROUTE TESTS -- Undo Stack
# ============================================================================

class TestUndoRoutes:
    """Integration tests for undo stack routes."""

    def test_push_route(self, client, csrf_token):
        resp = client.post("/api/undo/push", data=json.dumps({
            "operation": "trim",
            "input_file": "/a.mp4",
            "output_file": "/b.mp4",
            "session_id": "route_test",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["record"]["operation"] == "trim"

    def test_push_missing_operation(self, client, csrf_token):
        resp = client.post("/api/undo/push", data=json.dumps({
            "operation": "",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_push_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/api/undo/push", data=json.dumps(["trim"]),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_history_route(self, client, csrf_token):
        client.post("/api/undo/push", data=json.dumps({
            "operation": "test_op",
            "session_id": "hist_route",
        }), headers=csrf_headers(csrf_token))
        resp = client.get("/api/undo/history?session_id=hist_route")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1

    def test_undo_route(self, client, csrf_token):
        client.post("/api/undo/push", data=json.dumps({
            "operation": "color",
            "input_file": "/in.mp4",
            "output_file": "/out.mp4",
            "session_id": "undo_route",
        }), headers=csrf_headers(csrf_token))
        resp = client.post("/api/undo/undo", data=json.dumps({
            "session_id": "undo_route",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["reverted_operation"] == "color"

    def test_undo_empty_returns_404(self, client, csrf_token):
        resp = client.post("/api/undo/undo", data=json.dumps({
            "session_id": "empty_session",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_clear_route(self, client, csrf_token):
        client.post("/api/undo/push", data=json.dumps({
            "operation": "x",
            "session_id": "clear_route",
        }), headers=csrf_headers(csrf_token))
        resp = client.post("/api/undo/clear", data=json.dumps({
            "session_id": "clear_route",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["cleared"] >= 1


# ============================================================================
# ROUTE TESTS -- EDL / AAF
# ============================================================================

class TestEDLAAFRoutes:
    """Integration tests for EDL/AAF routes."""

    def test_edl_export_route(self, client, csrf_token, tmp_dir):
        out = os.path.join(tmp_dir, "route_test.edl")
        resp = client.post("/api/edl/export", data=json.dumps({
            "cuts": [{"reel": "AX", "channel": "V", "transition": "C",
                       "source_in": 0, "source_out": 5,
                       "record_in": 0, "record_out": 5}],
            "output_path": out,
            "title": "Route Test",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["event_count"] == 1

    def test_edl_export_empty_cuts(self, client, csrf_token):
        resp = client.post("/api/edl/export", data=json.dumps({
            "cuts": [],
            "output_path": "/tmp/test.edl",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_edl_import_route(self, client, csrf_token, tmp_dir):
        # First export an EDL
        from opencut.core.edl_aaf import export_edl
        edl_path = os.path.join(tmp_dir, "import_test.edl")
        export_edl([{
            "reel": "AX", "channel": "V", "transition": "C",
            "source_in": "00:00:00:00", "source_out": "00:00:05:00",
            "record_in": "00:00:00:00", "record_out": "00:00:05:00",
            "clip_name": "Test Clip",
        }], edl_path)
        resp = client.post("/api/edl/import", data=json.dumps({
            "edl_path": edl_path,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["event_count"] == 1

    def test_edl_import_missing_file(self, client, csrf_token):
        resp = client.post("/api/edl/import", data=json.dumps({
            "edl_path": "/nonexistent/file.edl",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_edl_export_invalid_fps_falls_back_instead_of_500(self, client, csrf_token, tmp_dir):
        out = os.path.join(tmp_dir, "bad_fps.edl")
        resp = client.post("/api/edl/export", data=json.dumps({
            "cuts": [{"reel": "AX", "channel": "V", "transition": "C",
                       "source_in": 0, "source_out": 5,
                       "record_in": 0, "record_out": 5}],
            "output_path": out,
            "fps": "bad-value",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

    def test_aaf_export_route(self, client, csrf_token, tmp_dir):
        out = os.path.join(tmp_dir, "aaf_route.json")
        resp = client.post("/api/aaf/export", data=json.dumps({
            "cuts": [{"reel": "AX", "source_in": 0, "source_out": 5}],
            "output_path": out,
            "title": "AAF Route Test",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["event_count"] == 1

    def test_aaf_export_empty_cuts(self, client, csrf_token):
        resp = client.post("/api/aaf/export", data=json.dumps({
            "cuts": [],
            "output_path": "/tmp/test.json",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS -- Scripting Console
# ============================================================================

class TestScriptingRoutes:
    """Integration tests for scripting console routes."""

    def test_execute_route(self, client, csrf_token):
        resp = client.post("/api/scripting/execute", data=json.dumps({
            "code": "print(2 + 2)",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "4" in data["output"]

    def test_execute_no_code(self, client, csrf_token):
        resp = client.post("/api/scripting/execute", data=json.dumps({
            "code": "",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_execute_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/api/scripting/execute", data=json.dumps(["print(1)"]),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_modules_route(self, client):
        resp = client.get("/api/scripting/modules")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "math" in data["modules"]
        assert data["count"] > 0


# ============================================================================
# ROUTE TESTS -- Macro Recording
# ============================================================================

class TestMacroRoutes:
    """Integration tests for macro recording routes."""

    def test_start_stop_route(self, client, csrf_token):
        resp = client.post("/api/macro/start", data=json.dumps({
            "session_id": "route_macro",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["recording"] is True

        resp = client.post("/api/macro/stop", data=json.dumps({
            "session_id": "route_macro",
            "name": "Route Macro",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["macro"]["name"] == "Route Macro"

    def test_stop_without_start_returns_400(self, client, csrf_token):
        resp = client.post("/api/macro/stop", data=json.dumps({
            "session_id": "no_such_session",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_play_route(self, client, csrf_token):
        resp = client.post("/api/macro/play", data=json.dumps({
            "macro": {
                "name": "Test",
                "actions": [{"endpoint": "/trim", "params": {"start": 0}}],
            },
            "target_file": "/test.mp4",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["step_count"] == 1

    def test_play_missing_macro(self, client, csrf_token):
        resp = client.post("/api/macro/play", data=json.dumps({
            "macro": {},
            "target_file": "/test.mp4",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_save_load_route(self, client, csrf_token, tmp_dir):
        path = os.path.join(tmp_dir, "route_macro.json")
        resp = client.post("/api/macro/save", data=json.dumps({
            "macro": {
                "name": "Saved",
                "actions": [{"endpoint": "/test", "params": {}}],
            },
            "path": path,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200

        resp = client.post("/api/macro/load", data=json.dumps({
            "path": path,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["macro"]["name"] == "Saved"

    def test_load_missing_macro(self, client, csrf_token):
        resp = client.post("/api/macro/load", data=json.dumps({
            "path": "/nonexistent/macro.json",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_load_rejects_non_string_path(self, client, csrf_token):
        resp = client.post("/api/macro/load", data=json.dumps({
            "path": 123,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS -- Snapshots
# ============================================================================

class TestSnapshotRoutes:
    """Integration tests for edit snapshot routes."""

    def test_create_snapshot_route(self, client, csrf_token):
        resp = client.post("/api/snapshots/create", data=json.dumps({
            "name": f"route_snap_{os.getpid()}",
            "project_id": f"route_test_{os.getpid()}",
            "project_state": {"parameters": {"crf": 18}},
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        # Cleanup
        from opencut.core.edit_snapshots import _SNAPSHOTS_DIR
        snap_dir = os.path.join(_SNAPSHOTS_DIR, f"route_test_{os.getpid()}")
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_create_snapshot_empty_name(self, client, csrf_token):
        resp = client.post("/api/snapshots/create", data=json.dumps({
            "name": "",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_restore_snapshot_missing(self, client, csrf_token):
        resp = client.post("/api/snapshots/restore", data=json.dumps({
            "name": "nonexistent",
            "project_id": "no_project",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_list_snapshots_route(self, client):
        resp = client.get("/api/snapshots/list?project_id=empty_project")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["snapshots"], list)

    def test_compare_snapshots_route(self, client, csrf_token):
        pid = f"compare_route_{os.getpid()}"
        # Create two snapshots
        client.post("/api/snapshots/create", data=json.dumps({
            "name": "a",
            "project_id": pid,
            "project_state": {"parameters": {"x": 1}},
        }), headers=csrf_headers(csrf_token))
        client.post("/api/snapshots/create", data=json.dumps({
            "name": "b",
            "project_id": pid,
            "project_state": {"parameters": {"x": 2}},
        }), headers=csrf_headers(csrf_token))
        resp = client.post("/api/snapshots/compare", data=json.dumps({
            "name_a": "a",
            "name_b": "b",
            "project_id": pid,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["parameter_diffs"]) >= 1
        # Cleanup
        from opencut.core.edit_snapshots import _SNAPSHOTS_DIR
        snap_dir = os.path.join(_SNAPSHOTS_DIR, pid)
        if os.path.isdir(snap_dir):
            shutil.rmtree(snap_dir)

    def test_compare_missing_names(self, client, csrf_token):
        resp = client.post("/api/snapshots/compare", data=json.dumps({
            "name_a": "",
            "name_b": "b",
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS -- Through-Edit
# ============================================================================

class TestThroughEditRoutes:
    """Integration tests for through-edit routes."""

    def test_detect_through_edits_route(self, client, csrf_token):
        resp = client.post("/api/timeline/through-edits", data=json.dumps({
            "cut_list": [
                {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
                {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
            ],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["mergeable_count"] == 1

    def test_auto_merge_route(self, client, csrf_token):
        resp = client.post("/api/timeline/through-edits", data=json.dumps({
            "cut_list": [
                {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
                {"source_file": "a.mp4", "source_in": 5, "source_out": 10},
            ],
            "auto_merge": True,
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "merged_cut_list" in data
        assert data["merged_count"] == 1

    def test_empty_cut_list(self, client, csrf_token):
        resp = client.post("/api/timeline/through-edits", data=json.dumps({
            "cut_list": [],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS -- Ripple Edit
# ============================================================================

class TestRippleEditRoutes:
    """Integration tests for ripple edit routes."""

    def test_detect_gaps_route(self, client, csrf_token):
        resp = client.post("/api/timeline/detect-gaps", data=json.dumps({
            "timeline_items": [
                {"start": 0, "end": 5},
                {"start": 8, "end": 13},
            ],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["gaps"]) == 1
        assert data["gaps"][0]["duration"] == pytest.approx(3.0)

    def test_detect_gaps_empty(self, client, csrf_token):
        resp = client.post("/api/timeline/detect-gaps", data=json.dumps({
            "timeline_items": [],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_ripple_close_route(self, client, csrf_token):
        resp = client.post("/api/timeline/ripple-close", data=json.dumps({
            "timeline_items": [
                {"start": 0, "end": 5, "track": "V1"},
                {"start": 7, "end": 12, "track": "V1"},
            ],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["gaps_closed"] == 1
        assert data["items"][1]["start"] == pytest.approx(5.0)

    def test_ripple_close_with_locked_tracks(self, client, csrf_token):
        resp = client.post("/api/timeline/ripple-close", data=json.dumps({
            "timeline_items": [
                {"start": 0, "end": 5, "track": "V1"},
                {"start": 7, "end": 12, "track": "A1"},
            ],
            "locked_tracks": ["A1"],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "A1" in data["locked_tracks_skipped"]

    def test_ripple_close_empty(self, client, csrf_token):
        resp = client.post("/api/timeline/ripple-close", data=json.dumps({
            "timeline_items": [],
        }), headers=csrf_headers(csrf_token))
        assert resp.status_code == 400
