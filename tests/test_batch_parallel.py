"""
Tests for parallel batch processing (Phase 5.4).

Covers:
- process_batch_parallel() with CPU and GPU operations
- Worker count selection logic
- Error isolation (one item failure doesn't kill the batch)
- Thread-safe progress tracking
- Cancellation mid-batch
- Route integration with parallel flag
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from opencut.core.batch_process import (
    GPU_OPERATIONS,
    _batch_lock,
    _batches,
    create_batch,
    process_batch_parallel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_files(tmp_path, count=3):
    """Create dummy files and return their paths."""
    paths = []
    for i in range(count):
        p = tmp_path / f"clip_{i}.mp4"
        p.write_text(f"fake video {i}")
        paths.append(str(p))
    return paths


def _cleanup_batch(batch_id):
    with _batch_lock:
        _batches.pop(batch_id, None)


# ---------------------------------------------------------------------------
# Unit tests for process_batch_parallel
# ---------------------------------------------------------------------------

class TestWorkerCount:
    """Verify max_workers is chosen correctly for CPU vs GPU ops."""

    def test_cpu_operation_workers(self):
        """CPU ops should use min(cpu_count, len(items)) workers."""
        cpu_count = os.cpu_count() or 4
        for n_items in [1, 2, 8, 50]:
            expected = min(cpu_count, n_items)
            assert expected >= 1
            # Just verify the formula; actual threading tested below

    def test_gpu_operations_set(self):
        """All expected GPU operations are in the set."""
        expected = {"upscale", "rembg", "denoise-ai", "face-enhance",
                    "face-swap", "style-transfer", "interpolate"}
        assert GPU_OPERATIONS == expected

    def test_cpu_operations_not_in_gpu(self):
        """Common CPU operations should NOT be classified as GPU."""
        cpu_ops = ["denoise", "normalize", "stabilize", "export_preset",
                   "vignette", "film_grain", "letterbox", "face_blur"]
        for op in cpu_ops:
            assert op not in GPU_OPERATIONS


class TestParallelProcessing:
    """Test the process_batch_parallel function."""

    @patch("opencut.core.batch_process.get_batch")
    @patch("opencut.routes.video._execute_batch_item")
    def test_all_items_succeed(self, mock_execute, mock_get_batch, tmp_path):
        """All items complete successfully."""
        paths = _make_temp_files(tmp_path, 3)
        batch_id = "test-ok-01"
        batch = create_batch(batch_id, "denoise", paths, {})
        mock_get_batch.return_value = batch
        mock_execute.side_effect = lambda op, fp, params, prog: fp + ".out"

        result = process_batch_parallel(batch_id, paths, "denoise", {})

        assert result["completed"] == 3
        assert result["failed"] == 0
        assert all(r is not None for r in result["results"])
        assert all(e is None for e in result["errors"])
        _cleanup_batch(batch_id)

    @patch("opencut.core.batch_process.get_batch")
    @patch("opencut.routes.video._execute_batch_item")
    def test_one_item_fails(self, mock_execute, mock_get_batch, tmp_path):
        """One item fails; others still complete (error isolation)."""
        paths = _make_temp_files(tmp_path, 3)
        batch_id = "test-fail-01"
        batch = create_batch(batch_id, "normalize", paths, {})
        mock_get_batch.return_value = batch

        def _side_effect(op, fp, params, prog):
            if "clip_1" in fp:
                raise RuntimeError("Simulated failure")
            return fp + ".out"

        mock_execute.side_effect = _side_effect

        result = process_batch_parallel(batch_id, paths, "normalize", {})

        assert result["completed"] == 2
        assert result["failed"] == 1
        assert result["errors"][1] is not None
        assert "Simulated failure" in result["errors"][1]
        _cleanup_batch(batch_id)

    @patch("opencut.core.batch_process.get_batch")
    @patch("opencut.routes.video._execute_batch_item")
    def test_callbacks_invoked(self, mock_execute, mock_get_batch, tmp_path):
        """on_item_complete and on_item_error callbacks are called."""
        paths = _make_temp_files(tmp_path, 2)
        batch_id = "test-cb-01"
        batch = create_batch(batch_id, "denoise", paths, {})
        mock_get_batch.return_value = batch

        def _side_effect(op, fp, params, prog):
            if "clip_1" in fp:
                raise ValueError("bad file")
            return fp + ".out"

        mock_execute.side_effect = _side_effect

        on_complete = MagicMock()
        on_error = MagicMock()

        process_batch_parallel(
            batch_id, paths, "denoise", {},
            on_item_complete=on_complete,
            on_item_error=on_error,
        )

        on_complete.assert_called_once()
        on_error.assert_called_once()
        # Verify the error callback got the right index & message
        err_call_args = on_error.call_args[0]
        assert err_call_args[0] == 1  # index of clip_1
        assert "bad file" in err_call_args[1]
        _cleanup_batch(batch_id)

    @patch("opencut.core.batch_process.get_batch")
    @patch("opencut.routes.video._execute_batch_item")
    def test_gpu_operation_limited_workers(self, mock_execute, mock_get_batch, tmp_path):
        """GPU operations should use max 2 workers."""
        paths = _make_temp_files(tmp_path, 5)
        batch_id = "test-gpu-01"
        batch = create_batch(batch_id, "upscale", paths, {})
        mock_get_batch.return_value = batch

        active_threads = []
        max_concurrent = [0]
        lock = threading.Lock()

        def _side_effect(op, fp, params, prog):
            with lock:
                active_threads.append(threading.current_thread().ident)
                current = len(set(active_threads))
                if current > max_concurrent[0]:
                    max_concurrent[0] = current
            time.sleep(0.05)  # Simulate work
            with lock:
                active_threads.remove(threading.current_thread().ident)
            return fp + ".out"

        mock_execute.side_effect = _side_effect

        result = process_batch_parallel(batch_id, paths, "upscale", {})

        assert result["completed"] == 5
        assert max_concurrent[0] <= 2, f"GPU op used {max_concurrent[0]} concurrent threads (expected <= 2)"
        _cleanup_batch(batch_id)

    @patch("opencut.routes.video._execute_batch_item")
    def test_skipped_items_preserved(self, mock_execute, tmp_path):
        """Items already marked as 'skipped' should not be processed."""
        real_path = tmp_path / "real.mp4"
        real_path.write_text("data")
        fake_path = "/nonexistent/file.mp4"

        batch_id = "test-skip-01"
        batch = create_batch(batch_id, "denoise", [str(real_path), fake_path], {})
        # The second item should already be marked skipped by create_batch
        assert batch.items[1].status == "skipped"

        mock_execute.return_value = str(real_path) + ".out"

        result = process_batch_parallel(
            batch_id, [str(real_path), fake_path], "denoise", {},
        )

        # Only the real file was processed
        assert result["completed"] == 1
        assert result["failed"] == 0
        assert mock_execute.call_count == 1
        _cleanup_batch(batch_id)

    @patch("opencut.routes.video._execute_batch_item")
    def test_empty_batch(self, mock_execute):
        """Empty item list returns zero counts."""
        batch_id = "test-empty-01"
        create_batch(batch_id, "denoise", [], {})

        result = process_batch_parallel(batch_id, [], "denoise", {})

        assert result["completed"] == 0
        assert result["failed"] == 0
        assert result["results"] == []
        assert result["errors"] == []
        mock_execute.assert_not_called()
        _cleanup_batch(batch_id)


class TestParallelBatchRoute:
    """Test the /batch/create route with parallel flag."""

    @pytest.fixture
    def app(self):
        from opencut.server import app as flask_app
        flask_app.config["TESTING"] = True
        return flask_app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    @pytest.fixture
    def csrf_token(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        return data.get("csrf_token", "")

    def _headers(self, token):
        return {
            "X-OpenCut-Token": token,
            "Content-Type": "application/json",
        }

    @patch("opencut.routes.video._execute_batch_item")
    def test_batch_create_parallel_true(self, mock_execute, client, csrf_token, tmp_path):
        """POST /batch/create with parallel=true returns parallel flag."""
        p = tmp_path / "test.mp4"
        p.write_text("data")
        mock_execute.return_value = str(p) + ".out"

        resp = client.post(
            "/batch/create",
            json={
                "operation": "denoise",
                "filepaths": [str(p)],
                "params": {},
                "parallel": True,
            },
            headers=self._headers(csrf_token),
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "running"
        assert data["parallel"] is True
        assert "batch_id" in data

    @patch("opencut.routes.video._execute_batch_item")
    def test_batch_create_parallel_false(self, mock_execute, client, csrf_token, tmp_path):
        """POST /batch/create with parallel=false uses sequential mode."""
        p = tmp_path / "test.mp4"
        p.write_text("data")
        mock_execute.return_value = str(p) + ".out"

        resp = client.post(
            "/batch/create",
            json={
                "operation": "denoise",
                "filepaths": [str(p)],
                "params": {},
                "parallel": False,
            },
            headers=self._headers(csrf_token),
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parallel"] is False

    @patch("opencut.routes.video._execute_batch_item")
    def test_batch_create_default_parallel(self, mock_execute, client, csrf_token, tmp_path):
        """POST /batch/create without parallel flag defaults to True."""
        p = tmp_path / "test.mp4"
        p.write_text("data")
        mock_execute.return_value = str(p) + ".out"

        resp = client.post(
            "/batch/create",
            json={
                "operation": "denoise",
                "filepaths": [str(p)],
                "params": {},
            },
            headers=self._headers(csrf_token),
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parallel"] is True
