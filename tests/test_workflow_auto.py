"""
Tests for OpenCut Workflow Automation features.

Covers: watch folder, render queue, conditional workflow,
best take selection, and their route endpoints.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import csrf_headers


# ========================================================================
# 1. Watch Folder (core)
# ========================================================================
class TestWatchFolderConfig:
    """Tests for WatchFolderConfig dataclass."""

    def test_default_extensions(self):
        from opencut.core.watch_folder import WatchFolderConfig
        config = WatchFolderConfig(folder_path="/tmp/test")
        assert isinstance(config.file_extensions, list)
        assert ".mp4" in config.file_extensions
        assert ".mov" in config.file_extensions

    def test_custom_extensions(self):
        from opencut.core.watch_folder import WatchFolderConfig
        config = WatchFolderConfig(folder_path="/tmp/test", file_extensions=[".mp4", ".mkv"])
        assert config.file_extensions == [".mp4", ".mkv"]

    def test_id_auto_generated(self):
        from opencut.core.watch_folder import WatchFolderConfig
        config = WatchFolderConfig(folder_path="/tmp/test")
        assert len(config.id) == 12

    def test_poll_interval_default(self):
        from opencut.core.watch_folder import WatchFolderConfig
        config = WatchFolderConfig(folder_path="/tmp/test")
        assert config.poll_interval_sec == 5.0


class TestWatchFolderProcessedTracking:
    """Tests for processed file tracking."""

    def test_mark_and_check_processed(self, tmp_path):
        from opencut.core.watch_folder import _is_processed, _mark_processed
        fake_path = str(tmp_path / "test_video.mp4")

        with patch("opencut.core.watch_folder._PROCESSED_PATH", str(tmp_path / "processed.json")):
            # Initially not processed
            assert not _is_processed(fake_path)
            # Mark and verify
            _mark_processed(fake_path)
            assert _is_processed(fake_path)


class TestWatchFolderConfigPersistence:
    """Tests for config save/load."""

    def test_save_and_load_configs(self, tmp_path):
        from opencut.core.watch_folder import WatchFolderConfig, load_watch_configs, save_watch_configs

        with patch("opencut.core.watch_folder._CONFIGS_PATH", str(tmp_path / "configs.json")), \
             patch("opencut.core.watch_folder._ensure_opencut_dir"):
            configs = [
                WatchFolderConfig(folder_path="/tmp/a", workflow_name="Test"),
                WatchFolderConfig(folder_path="/tmp/b", poll_interval_sec=10),
            ]
            save_watch_configs(configs)
            loaded = load_watch_configs()
            assert len(loaded) == 2
            assert loaded[0].folder_path == "/tmp/a"
            assert loaded[0].workflow_name == "Test"
            assert loaded[1].poll_interval_sec == 10

    def test_load_missing_file_returns_empty(self, tmp_path):
        from opencut.core.watch_folder import load_watch_configs
        with patch("opencut.core.watch_folder._CONFIGS_PATH", str(tmp_path / "nonexistent.json")):
            assert load_watch_configs() == []


class TestWatchFolderStartStop:
    """Tests for start/stop watcher lifecycle."""

    def test_start_watch_invalid_folder(self):
        from opencut.core.watch_folder import WatchFolderConfig, start_watch
        config = WatchFolderConfig(folder_path="/nonexistent/path/zzz")
        with pytest.raises(ValueError, match="does not exist"):
            start_watch(config)

    def test_start_and_stop_watch(self, tmp_path):
        from opencut.core.watch_folder import WatchFolderConfig, list_active_watches, start_watch, stop_watch

        config = WatchFolderConfig(
            folder_path=str(tmp_path),
            poll_interval_sec=0.1,
        )
        with patch("opencut.core.watch_folder._validate_media", return_value=True):
            handle = start_watch(config)
            assert handle.active
            watches = list_active_watches()
            assert len(watches) >= 1
            stop_watch(handle)
            time.sleep(0.3)
            assert not handle.active


class TestWatchFolderPolling:
    """Tests for file detection polling."""

    def test_new_file_triggers_callback(self, tmp_path):
        from opencut.core.watch_folder import WatchFolderConfig, start_watch, stop_watch

        callback = MagicMock()
        config = WatchFolderConfig(
            folder_path=str(tmp_path),
            poll_interval_sec=0.1,
            file_extensions=[".mp4"],
        )

        with patch("opencut.core.watch_folder._validate_media", return_value=True), \
             patch("opencut.core.watch_folder._is_processed", return_value=False), \
             patch("opencut.core.watch_folder._mark_processed"):
            handle = start_watch(config, on_new_file=callback)
            # Wait for initial scan
            time.sleep(0.5)
            # Create a new file
            test_file = tmp_path / "new_video.mp4"
            test_file.write_bytes(b"x" * 2048)
            # Wait for detection -- poll loop has a 0.5s stability delay
            time.sleep(1.5)
            stop_watch(handle)

        assert callback.call_count >= 1
        call_args = callback.call_args[0]
        assert "new_video.mp4" in call_args[0]


# ========================================================================
# 2. Render Queue (core)
# ========================================================================
class TestRenderQueueItem:
    """Tests for RenderQueueItem dataclass."""

    def test_default_values(self):
        from opencut.core.render_queue import RenderQueueItem
        item = RenderQueueItem(id="test1", input_path="/tmp/f.mp4", preset_name="youtube_1080p")
        assert item.priority == 3
        assert item.status == "pending"
        assert item.progress == 0
        assert item.error == ""


class TestRenderQueueOperations:
    """Tests for queue add/remove/reorder/get."""

    def test_add_to_queue(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue

        with patch("opencut.core.render_queue._save_queue"):
            item_id = add_to_queue("/tmp/test.mp4", "youtube_1080p", priority=4)
            assert isinstance(item_id, str)
            assert len(item_id) == 12

            # Clean up
            with _queue_lock:
                _queue[:] = [q for q in _queue if q.id != item_id]

    def test_remove_from_queue(self):
        from opencut.core.render_queue import add_to_queue, remove_from_queue

        with patch("opencut.core.render_queue._save_queue"):
            item_id = add_to_queue("/tmp/test.mp4", "youtube_1080p")
            assert remove_from_queue(item_id)
            assert not remove_from_queue(item_id)  # already removed

    def test_remove_rendering_item_fails(self):
        from opencut.core.render_queue import RenderQueueItem, _queue, _queue_lock, remove_from_queue

        with patch("opencut.core.render_queue._save_queue"):
            item = RenderQueueItem(id="rendering1", input_path="/tmp/f.mp4",
                                   preset_name="test", status="rendering")
            with _queue_lock:
                _queue.append(item)
            assert not remove_from_queue("rendering1")
            with _queue_lock:
                _queue[:] = [q for q in _queue if q.id != "rendering1"]

    def test_reorder_queue(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue, reorder_queue

        with patch("opencut.core.render_queue._save_queue"):
            item_id = add_to_queue("/tmp/test.mp4", "youtube_1080p", priority=2)
            reorder_queue(item_id, 5)
            with _queue_lock:
                item = next(q for q in _queue if q.id == item_id)
                assert item.priority == 5
                _queue[:] = [q for q in _queue if q.id != item_id]

    def test_reorder_nonexistent_raises(self):
        from opencut.core.render_queue import reorder_queue
        with patch("opencut.core.render_queue._save_queue"):
            with pytest.raises(ValueError, match="not found"):
                reorder_queue("nonexistent_id", 3)

    def test_get_queue_sorted_by_priority(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue, get_queue

        with patch("opencut.core.render_queue._save_queue"):
            id1 = add_to_queue("/tmp/a.mp4", "test", priority=1)
            id2 = add_to_queue("/tmp/b.mp4", "test", priority=5)
            id3 = add_to_queue("/tmp/c.mp4", "test", priority=3)

            items = get_queue()
            priorities = [i.priority for i in items if i.id in (id1, id2, id3)]
            assert priorities[0] >= priorities[-1]

            # Clean up
            with _queue_lock:
                _queue[:] = [q for q in _queue if q.id not in (id1, id2, id3)]

    def test_priority_clamped(self):
        from opencut.core.render_queue import _queue, _queue_lock, add_to_queue

        with patch("opencut.core.render_queue._save_queue"):
            item_id = add_to_queue("/tmp/test.mp4", "test", priority=99)
            with _queue_lock:
                item = next(q for q in _queue if q.id == item_id)
                assert item.priority == 5
                _queue[:] = [q for q in _queue if q.id != item_id]


class TestRenderQueuePauseResume:
    """Tests for pause/resume."""

    def test_pause_and_resume(self):
        from opencut.core.render_queue import is_queue_paused, pause_queue, resume_queue
        pause_queue()
        assert is_queue_paused()
        resume_queue()
        assert not is_queue_paused()


class TestProcessPriority:
    """Tests for OS process priority setting."""

    def test_set_process_priority_no_crash(self):
        from opencut.core.render_queue import set_process_priority
        # Should not raise on any platform
        set_process_priority(low=True)
        set_process_priority(low=False)


# ========================================================================
# 3. Conditional Workflow (core)
# ========================================================================
class TestEvaluateCondition:
    """Tests for condition evaluation."""

    def test_empty_condition_is_true(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("", {}) is True

    def test_duration_greater_than(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("duration > 60", {"duration": 120}) is True
        assert evaluate_condition("duration > 60", {"duration": 30}) is False

    def test_loudness_less_than(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("loudness_lufs < -20", {"loudness_lufs": -25}) is True
        assert evaluate_condition("loudness_lufs < -20", {"loudness_lufs": -10}) is False

    def test_width_gte(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("width >= 1920", {"width": 1920}) is True
        assert evaluate_condition("width >= 1920", {"width": 1280}) is False

    def test_equality(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("fps == 30", {"fps": 30.0}) is True
        assert evaluate_condition("fps != 30", {"fps": 24.0}) is True

    def test_has_video_flag(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("has_video", {"has_video": True}) is True
        assert evaluate_condition("has_video", {"has_video": False}) is False

    def test_has_audio_flag(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("has_audio", {"has_audio": True}) is True

    def test_negated_flag(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("!has_audio", {"has_audio": False}) is True
        assert evaluate_condition("!has_audio", {"has_audio": True}) is False

    def test_missing_field_is_false(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("bitrate > 5000", {}) is False

    def test_unparseable_condition_is_false(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("nonsense gibberish", {}) is False

    def test_negative_values(self):
        from opencut.core.conditional_workflow import evaluate_condition
        assert evaluate_condition("loudness_lufs < -14", {"loudness_lufs": -23}) is True


class TestBuildClipMetadata:
    """Tests for metadata builder."""

    def test_build_metadata_returns_expected_keys(self):
        from opencut.core.conditional_workflow import build_clip_metadata
        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}

        with patch("opencut.core.conditional_workflow.get_video_info", return_value=mock_info), \
             patch("opencut.helpers.run_ffmpeg", side_effect=RuntimeError("no ffmpeg")), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            meta = build_clip_metadata("/tmp/test.mp4")

        assert meta["width"] == 1920
        assert meta["height"] == 1080
        assert meta["duration"] == 120.0
        assert "has_video" in meta
        assert "has_audio" in meta


class TestRunConditionalWorkflow:
    """Tests for conditional workflow execution."""

    def test_empty_steps_returns_empty(self):
        from opencut.core.conditional_workflow import run_conditional_workflow
        with patch("opencut.core.conditional_workflow.build_clip_metadata", return_value={}):
            result = run_conditional_workflow("/tmp/test.mp4", [])
        assert result["completed"] == 0
        assert result["total"] == 0

    def test_skips_unmet_conditions(self):
        from opencut.core.conditional_workflow import run_conditional_workflow

        meta = {"duration": 30, "width": 1280, "has_video": True}
        steps = [
            {"action": "normalize", "condition": "duration > 60"},
            {"action": "export", "condition": "width >= 1920"},
        ]

        with patch("opencut.core.conditional_workflow.build_clip_metadata", return_value=meta), \
             patch("opencut.core.conditional_workflow._resolve_action", return_value=None):
            result = run_conditional_workflow("/tmp/test.mp4", steps)

        assert result["total"] == 2
        assert result["steps"][0]["skipped"] is True
        assert result["steps"][1]["skipped"] is True

    def test_executes_met_conditions(self):
        from opencut.core.conditional_workflow import run_conditional_workflow

        meta = {"duration": 120, "width": 1920, "has_video": True}
        mock_action = MagicMock(return_value="/tmp/output.mp4")

        steps = [
            {"action": "normalize", "condition": "duration > 60"},
        ]

        with patch("opencut.core.conditional_workflow.build_clip_metadata", return_value=meta), \
             patch("opencut.core.conditional_workflow._resolve_action", return_value=mock_action), \
             patch("os.path.isfile", return_value=True):
            result = run_conditional_workflow("/tmp/test.mp4", steps)

        assert result["completed"] == 1
        assert result["steps"][0]["skipped"] is False
        mock_action.assert_called_once()

    def test_unknown_action_recorded(self):
        from opencut.core.conditional_workflow import run_conditional_workflow

        meta = {"duration": 120}
        steps = [{"action": "nonexistent_action"}]

        with patch("opencut.core.conditional_workflow.build_clip_metadata", return_value=meta), \
             patch("opencut.core.conditional_workflow._resolve_action", return_value=None):
            result = run_conditional_workflow("/tmp/test.mp4", steps)

        assert "Unknown action" in result["steps"][0]["error"]


# ========================================================================
# 4. Best Take Selection (core)
# ========================================================================
class TestTakeScoring:
    """Tests for take scoring logic."""

    def test_score_takes_empty(self):
        from opencut.core.best_take import score_takes
        result = score_takes([])
        assert result.total_scored == 0
        assert result.best_take == ""

    def test_score_takes_single_file(self):
        from opencut.core.best_take import score_takes

        with patch("opencut.core.best_take._get_audio_stats", return_value={
            "rms_level": -20.0, "peak_level": -3.0,
            "noise_floor": -60.0, "dynamic_range": 20.0,
        }), patch("opencut.core.best_take._get_loudness_stats", return_value={
            "integrated_lufs": -16.0, "loudness_range": 8.0, "true_peak": -1.0,
        }), patch("os.path.isfile", return_value=True):
            result = score_takes(["/tmp/take1.mp4"])

        assert result.total_scored == 1
        assert result.takes[0].recommended is True
        assert result.takes[0].overall_score > 0
        assert result.best_take == "/tmp/take1.mp4"

    def test_score_takes_ranks_correctly(self):
        from opencut.core.best_take import score_takes

        good_stats = {
            "rms_level": -18.0, "peak_level": -2.0,
            "noise_floor": -65.0, "dynamic_range": 20.0,
        }
        good_loudness = {
            "integrated_lufs": -14.0, "loudness_range": 5.0, "true_peak": -1.0,
        }
        bad_stats = {
            "rms_level": -40.0, "peak_level": -10.0,
            "noise_floor": -45.0, "dynamic_range": 40.0,
        }
        bad_loudness = {
            "integrated_lufs": -30.0, "loudness_range": 25.0, "true_peak": -6.0,
        }

        def mock_audio_stats(fp):
            return good_stats if "good" in fp else bad_stats

        def mock_loudness(fp):
            return good_loudness if "good" in fp else bad_loudness

        with patch("opencut.core.best_take._get_audio_stats", side_effect=mock_audio_stats), \
             patch("opencut.core.best_take._get_loudness_stats", side_effect=mock_loudness), \
             patch("os.path.isfile", return_value=True):
            result = score_takes(["/tmp/good_take.mp4", "/tmp/bad_take.mp4"])

        assert result.best_take == "/tmp/good_take.mp4"
        good_score = next(t for t in result.takes if "good" in t.file_path)
        bad_score = next(t for t in result.takes if "bad" in t.file_path)
        assert good_score.overall_score > bad_score.overall_score

    def test_score_takes_missing_file(self):
        from opencut.core.best_take import score_takes

        with patch("os.path.isfile", return_value=False):
            result = score_takes(["/tmp/missing.mp4"])
        assert result.total_scored == 1
        assert result.takes[0].overall_score == 0.0

    def test_progress_callback_called(self):
        from opencut.core.best_take import score_takes
        cb = MagicMock()

        with patch("opencut.core.best_take._get_audio_stats", return_value={
            "rms_level": -20, "peak_level": -3, "noise_floor": -60, "dynamic_range": 20,
        }), patch("opencut.core.best_take._get_loudness_stats", return_value={
            "integrated_lufs": -16, "loudness_range": 8, "true_peak": -1,
        }), patch("os.path.isfile", return_value=True):
            score_takes(["/tmp/t1.mp4", "/tmp/t2.mp4"], on_progress=cb)

        assert cb.call_count >= 2


class TestFindRepeatedTakes:
    """Tests for grouping repeated takes."""

    def test_empty_list(self):
        from opencut.core.best_take import find_repeated_takes
        assert find_repeated_takes([]) == []

    def test_groups_similar_names(self):
        from opencut.core.best_take import find_repeated_takes

        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}

        with patch("opencut.core.best_take.get_video_info", return_value=mock_info), \
             patch("os.path.isfile", return_value=True):
            groups = find_repeated_takes([
                "/tmp/interview_take1.mp4",
                "/tmp/interview_take2.mp4",
                "/tmp/broll_01.mp4",
            ], similarity_threshold=0.7)

        # interview_take1 and interview_take2 should group together
        assert len(groups) >= 1
        interview_group = None
        for g in groups:
            if any("interview" in p for p in g):
                interview_group = g
                break
        assert interview_group is not None
        assert len(interview_group) == 2

    def test_no_groups_when_dissimilar(self):
        from opencut.core.best_take import find_repeated_takes

        with patch("opencut.core.best_take.get_video_info", return_value={
            "width": 1920, "height": 1080, "fps": 30, "duration": 60,
        }), patch("os.path.isfile", return_value=True):
            groups = find_repeated_takes([
                "/tmp/alpha.mp4",
                "/tmp/zzzzz.mp4",
            ], similarity_threshold=0.95)

        # Very different names with high threshold -> no groups
        assert len(groups) == 0


class TestNameSimilarity:
    """Tests for _name_similarity helper."""

    def test_identical_names(self):
        from opencut.core.best_take import _name_similarity
        assert _name_similarity("interview", "interview") == 1.0

    def test_take_suffix_stripped(self):
        from opencut.core.best_take import _name_similarity
        sim = _name_similarity("interview_take1", "interview_take2")
        assert sim == 1.0

    def test_version_suffix_stripped(self):
        from opencut.core.best_take import _name_similarity
        sim = _name_similarity("scene_v1", "scene_v2")
        assert sim == 1.0

    def test_completely_different(self):
        from opencut.core.best_take import _name_similarity
        sim = _name_similarity("abc", "xyz")
        assert sim < 0.5


class TestDurationSimilarity:
    """Tests for _duration_similarity helper."""

    def test_identical_durations(self):
        from opencut.core.best_take import _duration_similarity
        assert _duration_similarity(60.0, 60.0) == 1.0

    def test_zero_duration(self):
        from opencut.core.best_take import _duration_similarity
        assert _duration_similarity(0, 60) == 0.0

    def test_similar_durations(self):
        from opencut.core.best_take import _duration_similarity
        sim = _duration_similarity(60.0, 65.0)
        assert sim > 0.9


# ========================================================================
# 5. Route Tests
# ========================================================================
class TestWatchRoutes:
    """Tests for /watch/* endpoints."""

    def test_watch_start_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/watch/start", headers=csrf_headers(csrf_token),
                           data=json.dumps(["bad-body"]))
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_INPUT"

    def test_watch_start_missing_folder(self, client, csrf_token):
        resp = client.post("/watch/start", headers=csrf_headers(csrf_token),
                           data=json.dumps({"folder_path": ""}))
        assert resp.status_code == 400

    def test_watch_start_nonexistent_folder(self, client, csrf_token):
        resp = client.post("/watch/start", headers=csrf_headers(csrf_token),
                           data=json.dumps({"folder_path": "/nonexistent/zzz123"}))
        assert resp.status_code == 400

    def test_watch_start_success(self, client, csrf_token, tmp_path):
        with patch("opencut.core.watch_folder._validate_media", return_value=True):
            resp = client.post("/watch/start", headers=csrf_headers(csrf_token),
                               data=json.dumps({"folder_path": str(tmp_path)}))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "watcher_id" in data

        # Clean up: stop the watcher
        from opencut.core.watch_folder import stop_all_watches
        stop_all_watches()

    def test_watch_stop_not_found(self, client, csrf_token):
        resp = client.post("/watch/stop", headers=csrf_headers(csrf_token),
                           data=json.dumps({"watcher_id": "nonexistent"}))
        assert resp.status_code == 404

    def test_watch_list(self, client):
        resp = client.get("/watch/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "watches" in data
        assert "count" in data

    def test_watch_config_save(self, client, csrf_token):
        with patch("opencut.core.watch_folder.save_watch_configs"):
            resp = client.post("/watch/config", headers=csrf_headers(csrf_token),
                               data=json.dumps({"configs": [
                                   {"folder_path": "/tmp/test", "workflow_name": "Test"}
                               ]}))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_watch_config_save_ignores_invalid_items(self, client, csrf_token):
        with patch("opencut.core.watch_folder.save_watch_configs") as save_configs:
            resp = client.post("/watch/config", headers=csrf_headers(csrf_token),
                               data=json.dumps({"configs": [
                                   {"folder_path": "/tmp/test", "workflow_name": "Test"},
                                   ["bad-item"],
                                   {"folder_path": "/tmp/skip", "file_extensions": "not-a-list"},
                               ]}))
        assert resp.status_code == 200
        saved_configs = save_configs.call_args[0][0]
        assert len(saved_configs) == 1


class TestRenderQueueRoutes:
    """Tests for /render-queue/* endpoints."""

    def test_render_queue_add_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/render-queue/add", headers=csrf_headers(csrf_token),
                           data=json.dumps(["bad-body"]))
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_INPUT"

    def test_render_queue_add_missing_fields(self, client, csrf_token):
        resp = client.post("/render-queue/add", headers=csrf_headers(csrf_token),
                           data=json.dumps({"input_path": ""}))
        assert resp.status_code == 400

    def test_render_queue_add_success(self, client, csrf_token, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake mp4")

        with patch("opencut.core.render_queue._save_queue"):
            resp = client.post("/render-queue/add", headers=csrf_headers(csrf_token),
                               data=json.dumps({
                                   "input_path": str(test_file),
                                   "preset_name": "youtube_1080p",
                                   "priority": 4,
                               }))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "item_id" in data

        # Clean up
        from opencut.core.render_queue import _queue, _queue_lock
        with _queue_lock:
            _queue[:] = [q for q in _queue if q.id != data["item_id"]]

    def test_render_queue_remove_not_found(self, client, csrf_token):
        with patch("opencut.core.render_queue._save_queue"):
            resp = client.delete("/render-queue/remove", headers=csrf_headers(csrf_token),
                                 data=json.dumps({"item_id": "nonexistent"}))
        assert resp.status_code == 404

    def test_render_queue_add_invalid_params_type(self, client, csrf_token, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake mp4")

        resp = client.post("/render-queue/add", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "input_path": str(test_file),
                               "preset_name": "youtube_1080p",
                               "params": ["bad"],
                           }))
        assert resp.status_code == 400

    def test_render_queue_list(self, client):
        resp = client.get("/render-queue/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "items" in data
        assert "count" in data
        assert "running" in data
        assert "paused" in data

    def test_render_queue_start(self, client, csrf_token):
        with patch("opencut.core.render_queue.start_queue_processing"):
            resp = client.post("/render-queue/start", headers=csrf_headers(csrf_token),
                               data=json.dumps({}))
        assert resp.status_code == 200

    def test_render_queue_pause(self, client, csrf_token):
        resp = client.post("/render-queue/pause", headers=csrf_headers(csrf_token),
                           data=json.dumps({}))
        assert resp.status_code == 200
        # Restore
        from opencut.core.render_queue import resume_queue
        resume_queue()

    def test_render_queue_resume(self, client, csrf_token):
        resp = client.post("/render-queue/resume", headers=csrf_headers(csrf_token),
                           data=json.dumps({}))
        assert resp.status_code == 200


class TestConditionalWorkflowRoutes:
    """Tests for /workflow/conditional endpoint."""

    def test_conditional_workflow_no_steps(self, client, csrf_token, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake mp4")

        resp = client.post("/workflow/conditional", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "filepath": str(test_file),
                               "steps": [],
                           }))
        # async_job returns job_id even for errors in the handler body
        # Empty steps raises ValueError -> job error
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_conditional_workflow_returns_job_id(self, client, csrf_token, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake mp4")

        resp = client.post("/workflow/conditional", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "filepath": str(test_file),
                               "steps": [
                                   {"action": "normalize", "condition": "duration > 60"}
                               ],
                           }))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_conditional_workflow_non_list_steps_still_returns_job(self, client, csrf_token, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake mp4")

        resp = client.post("/workflow/conditional", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "filepath": str(test_file),
                               "steps": "bad-steps",
                           }))
        assert resp.status_code == 200
        assert "job_id" in resp.get_json()


class TestTakeRoutes:
    """Tests for /takes/* endpoints."""

    def test_takes_score_no_takes(self, client, csrf_token):
        resp = client.post("/takes/score", headers=csrf_headers(csrf_token),
                           data=json.dumps({"takes": []}))
        assert resp.status_code == 200  # job created, error in job body
        data = resp.get_json()
        assert "job_id" in data

    def test_takes_score_returns_job_id(self, client, csrf_token, tmp_path):
        f1 = tmp_path / "take1.mp4"
        f1.write_bytes(b"fake")
        f2 = tmp_path / "take2.mp4"
        f2.write_bytes(b"fake")

        resp = client.post("/takes/score", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "takes": [str(f1), str(f2)],
                           }))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_takes_find_repeats_no_files(self, client, csrf_token):
        resp = client.post("/takes/find-repeats", headers=csrf_headers(csrf_token),
                           data=json.dumps({"file_paths": []}))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_takes_find_repeats_returns_job_id(self, client, csrf_token, tmp_path):
        f1 = tmp_path / "scene_take1.mp4"
        f1.write_bytes(b"fake")

        resp = client.post("/takes/find-repeats", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "file_paths": [str(f1)],
                               "similarity_threshold": 0.8,
                           }))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_takes_find_repeats_invalid_threshold_does_not_500(self, client, csrf_token, tmp_path):
        f1 = tmp_path / "scene_take1.mp4"
        f1.write_bytes(b"fake")

        resp = client.post("/takes/find-repeats", headers=csrf_headers(csrf_token),
                           data=json.dumps({
                               "file_paths": [str(f1)],
                               "similarity_threshold": "not-a-number",
                           }))
        assert resp.status_code == 200
        assert "job_id" in resp.get_json()


# ========================================================================
# 6. Scoring helper unit tests
# ========================================================================
class TestNormalizeScore:
    """Tests for _normalize_score helper."""

    def test_min_value(self):
        from opencut.core.best_take import _normalize_score
        assert _normalize_score(0.0, 0.0, 100.0) == 0.0

    def test_max_value(self):
        from opencut.core.best_take import _normalize_score
        assert _normalize_score(100.0, 0.0, 100.0) == 1.0

    def test_mid_value(self):
        from opencut.core.best_take import _normalize_score
        assert abs(_normalize_score(50.0, 0.0, 100.0) - 0.5) < 0.01

    def test_clamped_below(self):
        from opencut.core.best_take import _normalize_score
        assert _normalize_score(-10.0, 0.0, 100.0) == 0.0

    def test_clamped_above(self):
        from opencut.core.best_take import _normalize_score
        assert _normalize_score(200.0, 0.0, 100.0) == 1.0

    def test_equal_bounds(self):
        from opencut.core.best_take import _normalize_score
        assert _normalize_score(5.0, 5.0, 5.0) == 0.5
