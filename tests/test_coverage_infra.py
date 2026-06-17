"""Coverage expansion: infrastructure modules."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ========================================================================
# 1. opencut/errors.py — OpenCutError, error_response, safe_error, factories
# ========================================================================

class TestOpenCutErrorConstruction:
    """OpenCutError stores code, message, status, suggestion."""

    def test_default_status_is_400(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "msg")
        assert e.status == 400

    def test_custom_status(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "msg", status=503)
        assert e.status == 503

    def test_code_and_message_stored(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("MY_CODE", "my message")
        assert e.code == "MY_CODE"
        assert e.message == "my message"

    def test_suggestion_default_empty(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "msg")
        assert e.suggestion == ""

    def test_suggestion_stored(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "msg", suggestion="try again")
        assert e.suggestion == "try again"

    def test_exception_str_is_message(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "something broke")
        assert str(e) == "something broke"

    def test_is_exception_subclass(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST", "msg")
        assert isinstance(e, Exception)


class TestErrorResponseWithFlask:
    """error_response returns (json_response, status_code) inside Flask context."""

    def _make_app(self):
        from flask import Flask
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_error_response_status_code(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            resp, status = error_response("NOT_FOUND", "gone", status=404, log=False)
            assert status == 404

    def test_error_response_body_has_code(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            resp, status = error_response("BAD", "bad input", status=400, log=False)
            data = resp.get_json()
            assert data["code"] == "BAD"
            assert data["error"] == "bad input"

    def test_error_response_includes_suggestion(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            resp, _ = error_response("X", "y", suggestion="do Z", log=False)
            data = resp.get_json()
            assert data["suggestion"] == "do Z"

    def test_error_response_includes_detail(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            resp, _ = error_response("X", "y", detail="extra info", log=False)
            data = resp.get_json()
            assert data["detail"] == "extra info"

    def test_error_response_omits_empty_suggestion(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            resp, _ = error_response("X", "y", log=False)
            data = resp.get_json()
            assert "suggestion" not in data

    def test_error_response_default_status_400(self):
        from opencut.errors import error_response
        app = self._make_app()
        with app.test_request_context():
            _, status = error_response("X", "y", log=False)
            assert status == 400


class TestSafeError:
    """safe_error classifies exception types into error codes."""

    def _make_app(self):
        from flask import Flask
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_memory_error_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(MemoryError("out of memory"))
            data = resp.get_json()
            assert data["code"] == "GPU_OUT_OF_MEMORY"
            assert status == 503

    def test_timeout_error_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(TimeoutError("timed out"))
            data = resp.get_json()
            assert data["code"] == "OPERATION_TIMEOUT"
            assert status == 504

    def test_import_error_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(ImportError("No module named 'torch'"))
            data = resp.get_json()
            assert data["code"] == "MISSING_DEPENDENCY"
            assert status == 503

    def test_permission_error_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(PermissionError("permission denied"))
            data = resp.get_json()
            assert data["code"] == "PERMISSION_DENIED"
            assert status == 403

    def test_file_not_found_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(FileNotFoundError("no such file"))
            data = resp.get_json()
            assert data["code"] == "FILE_NOT_FOUND"
            assert status == 404

    def test_unsupported_format_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(ValueError("unsupported format or codec"))
            data = resp.get_json()
            assert data["code"] == "UNSUPPORTED_FORMAT"
            assert status == 400

    def test_ffmpeg_runtime_error_classification(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(RuntimeError("ffmpeg failed to process"))
            data = resp.get_json()
            assert data["code"] == "FFMPEG_ERROR"
            assert status == 500

    def test_generic_error_defaults_to_internal(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(ValueError("something weird"))
            data = resp.get_json()
            assert data["code"] == "INTERNAL_ERROR"
            assert status == 500

    def test_opencut_error_passthrough(self):
        from opencut.errors import OpenCutError, safe_error
        app = self._make_app()
        with app.test_request_context():
            exc = OpenCutError("CUSTOM", "custom msg", status=418, suggestion="brew tea")
            resp, status = safe_error(exc)
            data = resp.get_json()
            assert data["code"] == "CUSTOM"
            assert status == 418

    def test_cuda_out_of_memory_string_match(self):
        from opencut.errors import safe_error
        app = self._make_app()
        with app.test_request_context():
            resp, status = safe_error(RuntimeError("CUDA out of memory"))
            data = resp.get_json()
            assert data["code"] == "GPU_OUT_OF_MEMORY"


class TestErrorFactories:
    """Factory functions return OpenCutError with correct codes and statuses."""

    def test_missing_dependency(self):
        from opencut.errors import missing_dependency
        e = missing_dependency("torch")
        assert e.code == "MISSING_DEPENDENCY"
        assert e.status == 503
        assert "torch" in e.message

    def test_file_not_found(self):
        from opencut.errors import file_not_found
        e = file_not_found("/path/to/video.mp4")
        assert e.code == "FILE_NOT_FOUND"
        assert e.status == 404
        assert "video.mp4" in e.message

    def test_file_not_found_empty_path(self):
        from opencut.errors import file_not_found
        e = file_not_found("")
        assert e.code == "FILE_NOT_FOUND"
        assert "unknown" in e.message

    def test_gpu_out_of_memory(self):
        from opencut.errors import gpu_out_of_memory
        e = gpu_out_of_memory()
        assert e.code == "GPU_OUT_OF_MEMORY"
        assert e.status == 503

    def test_invalid_input(self):
        from opencut.errors import invalid_input
        e = invalid_input("bad data")
        assert e.code == "INVALID_INPUT"
        assert e.status == 400
        assert "bad data" in e.message

    def test_invalid_model(self):
        from opencut.errors import invalid_model
        e = invalid_model("gpt-7", allowed=["whisper-base", "whisper-large"])
        assert e.code == "INVALID_MODEL"
        assert e.status == 400
        assert "gpt-7" in e.message
        assert "whisper-base" in e.suggestion

    def test_invalid_model_no_allowed(self):
        from opencut.errors import invalid_model
        e = invalid_model("gpt-7")
        assert e.code == "INVALID_MODEL"
        assert "valid model" in e.suggestion.lower()

    def test_operation_failed(self):
        from opencut.errors import operation_failed
        e = operation_failed("encode crashed")
        assert e.code == "OPERATION_FAILED"
        assert e.status == 500
        assert e.message == "encode crashed"

    def test_rate_limited(self):
        from opencut.errors import rate_limited
        e = rate_limited("transcription")
        assert e.code == "RATE_LIMITED"
        assert e.status == 429
        assert "transcription" in e.message

    def test_rate_limited_no_operation(self):
        from opencut.errors import rate_limited
        e = rate_limited()
        assert e.code == "RATE_LIMITED"
        assert "()" not in e.message

    def test_queue_full(self):
        from opencut.errors import queue_full
        e = queue_full(50)
        assert e.code == "QUEUE_FULL"
        assert e.status == 429
        assert "50" in e.message

    def test_module_not_available(self):
        from opencut.errors import module_not_available
        e = module_not_available("demucs")
        assert e.code == "MODULE_NOT_AVAILABLE"
        assert e.status == 503
        assert "demucs" in e.message

    def test_file_permission_denied(self):
        from opencut.errors import file_permission_denied
        e = file_permission_denied("/locked/file.txt")
        assert e.code == "PERMISSION_DENIED"
        assert e.status == 403
        assert "/locked/file.txt" in e.message

    def test_file_permission_denied_empty(self):
        from opencut.errors import file_permission_denied
        e = file_permission_denied()
        assert e.code == "PERMISSION_DENIED"
        assert ":" not in e.message

    def test_too_many_items(self):
        from opencut.errors import too_many_items
        e = too_many_items("clips", 100)
        assert e.code == "TOO_MANY_ITEMS"
        assert e.status == 400
        assert "100" in e.message

    def test_server_busy(self):
        from opencut.errors import server_busy
        e = server_busy()
        assert e.code == "SERVER_BUSY"
        assert e.status == 429

    def test_install_failed(self):
        from opencut.errors import install_failed
        e = install_failed("torch", "network error")
        assert e.code == "INSTALL_FAILED"
        assert e.status == 500
        assert "torch" in e.message
        assert "network error" in e.message

    def test_install_failed_no_detail(self):
        from opencut.errors import install_failed
        e = install_failed("torch")
        assert e.code == "INSTALL_FAILED"
        assert "torch" in e.message


# ========================================================================
# 2. opencut/schemas.py — Dataclass construction, to_dict, _strip_none
# ========================================================================

class TestStripNone:
    """_strip_none removes None-valued keys."""

    def test_removes_none(self):
        from opencut.schemas import _strip_none
        assert _strip_none({"a": 1, "b": None}) == {"a": 1}

    def test_keeps_zero_and_empty(self):
        from opencut.schemas import _strip_none
        result = _strip_none({"a": 0, "b": "", "c": [], "d": False})
        assert result == {"a": 0, "b": "", "c": [], "d": False}

    def test_empty_dict(self):
        from opencut.schemas import _strip_none
        assert _strip_none({}) == {}

    def test_all_none(self):
        from opencut.schemas import _strip_none
        assert _strip_none({"a": None, "b": None}) == {}


class TestJobResponse:
    """JobResponse schema."""

    def test_defaults(self):
        from opencut.schemas import JobResponse
        r = JobResponse(job_id="abc-123")
        assert r.job_id == "abc-123"
        assert r.status == "running"

    def test_to_dict(self):
        from opencut.schemas import JobResponse
        d = JobResponse(job_id="x").to_dict()
        assert d == {"job_id": "x", "status": "running"}


class TestJobResult:
    """JobResult schema."""

    def test_defaults(self):
        from opencut.schemas import JobResult
        r = JobResult()
        assert r.status == "complete"
        assert r.progress == 100
        assert r.message == "Done"
        assert r.result is None
        assert r.error is None

    def test_to_dict_strips_none(self):
        from opencut.schemas import JobResult
        d = JobResult().to_dict()
        assert "result" not in d
        assert "error" not in d
        assert d["status"] == "complete"

    def test_to_dict_includes_result(self):
        from opencut.schemas import JobResult
        d = JobResult(result={"output": "/tmp/out.mp4"}).to_dict()
        assert d["result"] == {"output": "/tmp/out.mp4"}


class TestActionResult:
    """ActionResult schema."""

    def test_defaults(self):
        from opencut.schemas import ActionResult
        r = ActionResult()
        assert r.success is True
        assert r.ok is True

    def test_to_dict_strips_none_optionals(self):
        from opencut.schemas import ActionResult
        d = ActionResult().to_dict()
        assert "output" not in d
        assert "output_path" not in d
        assert "job_id" not in d


class TestHealthResult:
    """HealthResult schema."""

    def test_defaults(self):
        from opencut.schemas import HealthResult
        r = HealthResult()
        assert r.status == "ok"

    def test_to_dict_with_capabilities(self):
        from opencut.schemas import HealthResult
        d = HealthResult(version="1.0.0", capabilities={"gpu": True}).to_dict()
        assert d["version"] == "1.0.0"
        assert d["capabilities"]["gpu"] is True


class TestJobStatusResult:
    """JobStatusResult schema with many fields."""

    def test_defaults_are_zero(self):
        from opencut.schemas import JobStatusResult
        r = JobStatusResult()
        assert r.progress == 0
        assert r.peak_vram_mb == 0
        assert r.peak_cpu_pct == 0

    def test_to_dict_strips_none_but_keeps_zero(self):
        from opencut.schemas import JobStatusResult
        d = JobStatusResult(job_id="j1", status="running", progress=0).to_dict()
        assert d["progress"] == 0
        assert "result" not in d


class TestBeatMarkersResult:
    """BeatMarkersResult schema."""

    def test_to_dict(self):
        from opencut.schemas import BeatMarkersResult
        d = BeatMarkersResult(beats=[0.5, 1.0], bpm=120.0, total_beats=2).to_dict()
        assert d["bpm"] == 120.0
        assert d["total_beats"] == 2
        assert len(d["beats"]) == 2


class TestBatchResult:
    """BatchResult schema."""

    def test_empty_defaults(self):
        from opencut.schemas import BatchResult
        d = BatchResult().to_dict()
        assert d["total"] == 0
        assert d["completed"] == 0
        assert d["failed"] == 0


class TestWorkflowResult:
    """WorkflowResult schema."""

    def test_to_dict_strips_none_output(self):
        from opencut.schemas import WorkflowResult
        d = WorkflowResult(workflow="transcode", steps_completed=3, total_steps=5).to_dict()
        assert d["workflow"] == "transcode"
        assert "output_path" not in d


# ========================================================================
# 3. opencut/workers.py — JobPriority enum, WorkerPool basics
# ========================================================================

class TestJobPriority:
    """JobPriority enum values and ordering."""

    def test_critical_is_zero(self):
        from opencut.workers import JobPriority
        assert JobPriority.CRITICAL == 0

    def test_high_is_10(self):
        from opencut.workers import JobPriority
        assert JobPriority.HIGH == 10

    def test_normal_is_50(self):
        from opencut.workers import JobPriority
        assert JobPriority.NORMAL == 50

    def test_low_is_100(self):
        from opencut.workers import JobPriority
        assert JobPriority.LOW == 100

    def test_background_is_200(self):
        from opencut.workers import JobPriority
        assert JobPriority.BACKGROUND == 200

    def test_critical_less_than_background(self):
        from opencut.workers import JobPriority
        assert JobPriority.CRITICAL < JobPriority.BACKGROUND

    def test_priority_ordering(self):
        from opencut.workers import JobPriority
        order = [JobPriority.CRITICAL, JobPriority.HIGH, JobPriority.NORMAL,
                 JobPriority.LOW, JobPriority.BACKGROUND]
        assert order == sorted(order)

    def test_is_int_enum(self):
        from opencut.workers import JobPriority
        assert isinstance(JobPriority.NORMAL, int)
        assert int(JobPriority.NORMAL) == 50


class TestWorkerPoolBasics:
    """WorkerPool construction and get_pool singleton."""

    def test_get_pool_returns_worker_pool(self):
        import opencut.workers as w
        # Reset singleton for isolated test
        with w._pool_lock:
            old = w._pool
            w._pool = None
        try:
            pool = w.get_pool(max_workers=2)
            assert isinstance(pool, w.WorkerPool)
            assert pool._max_workers == 2
        finally:
            pool.shutdown(wait=True)
            with w._pool_lock:
                w._pool = old

    def test_get_pool_returns_same_instance(self):
        import opencut.workers as w
        with w._pool_lock:
            old = w._pool
            w._pool = None
        try:
            p1 = w.get_pool(max_workers=2)
            p2 = w.get_pool(max_workers=2)
            assert p1 is p2
        finally:
            p1.shutdown(wait=True)
            with w._pool_lock:
                w._pool = old

    def test_pool_active_count_starts_zero(self):
        import opencut.workers as w
        pool = w.WorkerPool(max_workers=1)
        try:
            assert pool.active_count() == 0
        finally:
            pool.shutdown(wait=True)


# ========================================================================
# 4. opencut/config.py — OpenCutConfig, env helpers, is_local_only
# ========================================================================

class TestEnvBool:
    """_env_bool parses boolean env vars."""

    def test_true_values(self):
        from opencut.config import _env_bool
        for val in ("1", "true", "yes", "on", "TRUE", "  Yes  "):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL") is True, f"Failed for {val!r}"

    def test_false_values(self):
        from opencut.config import _env_bool
        for val in ("0", "false", "no", "off", "FALSE", "  No  "):
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL") is False, f"Failed for {val!r}"

    def test_missing_returns_default(self):
        from opencut.config import _env_bool
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNSET_VAR_XYZ", None)
            assert _env_bool("UNSET_VAR_XYZ", default=True) is True
            assert _env_bool("UNSET_VAR_XYZ", default=False) is False

    def test_invalid_returns_default(self):
        from opencut.config import _env_bool
        with patch.dict(os.environ, {"TEST_BOOL": "maybe"}):
            assert _env_bool("TEST_BOOL", default=True) is True


class TestEnvInt:
    """_env_int parses integer env vars with bounds."""

    def test_valid_int(self):
        from opencut.config import _env_int
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert _env_int("TEST_INT", 0) == 42

    def test_missing_returns_default(self):
        from opencut.config import _env_int
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISSING_INT_XYZ", None)
            assert _env_int("MISSING_INT_XYZ", 99) == 99

    def test_invalid_returns_default(self):
        from opencut.config import _env_int
        with patch.dict(os.environ, {"TEST_INT": "abc"}):
            assert _env_int("TEST_INT", 5) == 5

    def test_min_val_clamps(self):
        from opencut.config import _env_int
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            assert _env_int("TEST_INT", 10, min_val=5) == 5

    def test_max_val_clamps(self):
        from opencut.config import _env_int
        with patch.dict(os.environ, {"TEST_INT": "9999"}):
            assert _env_int("TEST_INT", 10, max_val=100) == 100


class TestEnvCsv:
    """_env_csv parses comma-separated env vars."""

    def test_basic_csv(self):
        from opencut.config import _env_csv
        with patch.dict(os.environ, {"TEST_CSV": "a, b, c"}):
            assert _env_csv("TEST_CSV", []) == ["a", "b", "c"]

    def test_empty_string_returns_empty_list(self):
        from opencut.config import _env_csv
        with patch.dict(os.environ, {"TEST_CSV": ""}):
            assert _env_csv("TEST_CSV", ["default"]) == []

    def test_missing_returns_default(self):
        from opencut.config import _env_csv
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISSING_CSV_XYZ", None)
            assert _env_csv("MISSING_CSV_XYZ", ["x"]) == ["x"]

    def test_strips_blanks(self):
        from opencut.config import _env_csv
        with patch.dict(os.environ, {"TEST_CSV": "a,,b, ,c"}):
            assert _env_csv("TEST_CSV", []) == ["a", "b", "c"]


class TestOpenCutConfigDefaults:
    """OpenCutConfig dataclass defaults."""

    def test_defaults(self):
        from opencut.config import OpenCutConfig
        c = OpenCutConfig()
        assert c.local_only is False
        assert c.bundled_mode is False
        assert c.whisper_models_dir is None
        assert c.max_content_length == 100 * 1024 * 1024
        assert c.max_concurrent_jobs == 10
        assert c.max_batch_files == 100
        assert c.job_max_age == 3600
        assert c.job_stuck_timeout == 7200
        assert c.cors_origins == []


class TestOpenCutConfigFromEnv:
    """OpenCutConfig.from_env reads env vars."""

    def test_local_only_from_env(self):
        from opencut.config import OpenCutConfig
        with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "true"}, clear=False):
            c = OpenCutConfig.from_env()
            assert c.local_only is True

    def test_bundled_mode_from_whisper_dir(self):
        from opencut.config import OpenCutConfig
        with patch.dict(os.environ, {"WHISPER_MODELS_DIR": "/models"}, clear=False):
            c = OpenCutConfig.from_env()
            assert c.bundled_mode is True
            assert c.whisper_models_dir == "/models"

    def test_max_content_length_from_env(self):
        from opencut.config import OpenCutConfig
        with patch.dict(os.environ, {"OPENCUT_MAX_CONTENT_LENGTH": "5242880"}, clear=False):
            c = OpenCutConfig.from_env()
            assert c.max_content_length == 5242880

    def test_cors_origins_from_env(self):
        from opencut.config import OpenCutConfig
        with patch.dict(os.environ, {"OPENCUT_CORS_ORIGINS": "http://localhost:3000,http://localhost:5173"}, clear=False):
            c = OpenCutConfig.from_env()
            assert "http://localhost:3000" in c.cors_origins
            assert "http://localhost:5173" in c.cors_origins


class TestIsLocalOnly:
    """is_local_only checks env var and user settings."""

    def test_env_var_true(self):
        from opencut.config import is_local_only
        with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}, clear=False):
            assert is_local_only() is True

    def test_env_var_false_falls_through(self):
        from opencut.config import is_local_only
        env_clean = {k: v for k, v in os.environ.items() if k != "OPENCUT_LOCAL_ONLY"}
        with patch.dict(os.environ, env_clean, clear=True):
            # With env var unset, is_local_only falls through to user_data
            # file check. In the test environment this returns False because
            # there is no local_only.json user setting file.
            assert is_local_only() is False


class TestRequireNetworkAllowed:
    """require_network_allowed raises in local-only mode."""

    def test_raises_when_local_only(self):
        from opencut.config import require_network_allowed
        with patch("opencut.config.is_local_only", return_value=True):
            with pytest.raises(RuntimeError, match="Local-only mode"):
                require_network_allowed("Cloud TTS")

    def test_no_raise_when_not_local(self):
        from opencut.config import require_network_allowed
        with patch("opencut.config.is_local_only", return_value=False):
            require_network_allowed("Cloud TTS")  # should not raise

    def test_includes_alternative_in_message(self):
        from opencut.config import require_network_allowed
        with patch("opencut.config.is_local_only", return_value=True):
            with pytest.raises(RuntimeError, match="Use offline-tts instead"):
                require_network_allowed("Cloud TTS", local_alternative="offline-tts")


# ========================================================================
# 5. opencut/helpers.py — _try_import, output_path, compute_estimate,
#    FFmpegCmd, escape helpers, _unique_output_path, _make_sequence_name
# ========================================================================

class TestTryImport:
    """_try_import returns module or None."""

    def test_valid_module(self):
        from opencut.helpers import _try_import
        mod = _try_import("os")
        assert mod is not None
        assert hasattr(mod, "path")

    def test_invalid_module(self):
        from opencut.helpers import _try_import
        mod = _try_import("nonexistent_module_xyz_12345")
        assert mod is None

    def test_stdlib_json(self):
        from opencut.helpers import _try_import
        mod = _try_import("json")
        assert mod is not None
        assert hasattr(mod, "dumps")


class TestTryImportFrom:
    """_try_import_from gets attributes from modules."""

    def test_valid_import_from(self):
        from opencut.helpers import _try_import_from
        result = _try_import_from("os.path", "join")
        assert result is os.path.join

    def test_missing_module(self):
        from opencut.helpers import _try_import_from
        result = _try_import_from("nonexistent_pkg_xyz", "thing")
        assert result is None

    def test_missing_attribute(self):
        from opencut.helpers import _try_import_from
        result = _try_import_from("os", "nonexistent_attr_xyz_999")
        assert result is None


class TestOutputPath:
    """output_path generates suffixed paths."""

    def test_basic_suffix(self):
        from opencut.helpers import output_path
        result = output_path("/videos/clip.mp4", "denoised")
        assert result.endswith("clip_denoised.mp4")
        assert "/videos/" in result.replace("\\", "/")

    def test_preserves_extension(self):
        from opencut.helpers import output_path
        result = output_path("/music/song.wav", "normalized")
        assert result.endswith("song_normalized.wav")

    def test_no_extension_defaults_to_mp4(self):
        from opencut.helpers import output_path
        result = output_path("/videos/clip", "out")
        assert result.endswith("clip_out.mp4")

    def test_custom_output_dir(self):
        from opencut.helpers import output_path
        result = output_path("/videos/clip.mp4", "out", output_dir="/output")
        assert result.replace("\\", "/").startswith("/output/")


class TestComputeEstimate:
    """compute_estimate returns timing estimates from historical data."""

    def test_no_history_returns_none_estimate(self):
        from opencut.helpers import compute_estimate
        with patch("opencut.helpers._load_job_times", return_value={}):
            result = compute_estimate("transcribe", 60.0)
            assert result["estimate_seconds"] is None
            assert result["confidence"] == "none"

    def test_with_ratios(self):
        from opencut.helpers import compute_estimate
        history = {
            "transcribe": [
                {"job_secs": 30, "file_secs": 60, "ratio": 0.5, "ts": 1},
                {"job_secs": 35, "file_secs": 70, "ratio": 0.5, "ts": 2},
            ]
        }
        with patch("opencut.helpers._load_job_times", return_value=history):
            result = compute_estimate("transcribe", 120.0)
            assert result["estimate_seconds"] == 60.0
            assert result["confidence"] == "medium"
            assert result["based_on"] == 2

    def test_high_confidence_with_5_entries(self):
        from opencut.helpers import compute_estimate
        entries = [
            {"job_secs": 10, "file_secs": 100, "ratio": 0.1, "ts": i}
            for i in range(5)
        ]
        history = {"denoise": entries}
        with patch("opencut.helpers._load_job_times", return_value=history):
            result = compute_estimate("denoise", 200.0)
            assert result["confidence"] == "high"
            assert result["estimate_seconds"] == pytest.approx(20.0, abs=0.1)

    def test_low_confidence_with_one_entry(self):
        from opencut.helpers import compute_estimate
        history = {
            "upscale": [
                {"job_secs": 120, "file_secs": 60, "ratio": 2.0, "ts": 1},
            ]
        }
        with patch("opencut.helpers._load_job_times", return_value=history):
            result = compute_estimate("upscale", 30.0)
            assert result["confidence"] == "low"

    def test_fallback_to_avg_times_when_no_ratio(self):
        from opencut.helpers import compute_estimate
        history = {
            "export": [
                {"job_secs": 10, "file_secs": 0, "ratio": 0, "ts": 1},
                {"job_secs": 20, "file_secs": 0, "ratio": 0, "ts": 2},
            ]
        }
        with patch("opencut.helpers._load_job_times", return_value=history):
            result = compute_estimate("export", 0)
            assert result["estimate_seconds"] == 15.0
            assert result["confidence"] == "low"


class TestEscapeFilterPath:
    """escape_filter_path handles colons and apostrophes."""

    def test_windows_drive_colon(self):
        from opencut.helpers import escape_filter_path
        result = escape_filter_path("C:/videos/clip.mp4")
        assert "C\\:/" in result

    def test_backslash_to_forward(self):
        from opencut.helpers import escape_filter_path
        result = escape_filter_path("C:\\videos\\clip.mp4")
        assert "\\" not in result or "\\:" in result  # only escaped colons

    def test_apostrophe_escaped(self):
        from opencut.helpers import escape_filter_path
        result = escape_filter_path("/path/it's here.mp4")
        assert "'" not in result.replace("\\'\\''", "")
        assert "\\'" in result


class TestEscapeDrawtext:
    """escape_drawtext handles backslash, colon, apostrophe."""

    def test_colon_escaped(self):
        from opencut.helpers import escape_drawtext
        result = escape_drawtext("Time: 12:00")
        assert "\\:" in result

    def test_backslash_escaped(self):
        from opencut.helpers import escape_drawtext
        result = escape_drawtext("path\\to\\file")
        assert "\\\\" in result


class TestConcatQuote:
    """_concat_quote escapes paths for concat demuxer."""

    def test_apostrophe_escaped(self):
        from opencut.helpers import _concat_quote
        result = _concat_quote("/path/it's.mp4")
        assert "'\\''" in result

    def test_cr_lf_stripped(self):
        from opencut.helpers import _concat_quote
        result = _concat_quote("/path/file\r\n.mp4")
        assert "\r" not in result
        assert "\n" not in result


class TestConcatFileLine:
    """_concat_file_line formats a file entry for concat list."""

    def test_format(self):
        from opencut.helpers import _concat_file_line
        result = _concat_file_line("/path/video.mp4")
        assert result.startswith("file '")
        assert result.endswith("'\n")


class TestUniqueOutputPath:
    """_unique_output_path appends counter when path exists."""

    def test_returns_original_when_no_collision(self, tmp_path):
        from opencut.helpers import _unique_output_path
        path = str(tmp_path / "new_file.mp4")
        assert _unique_output_path(path) == path

    def test_appends_counter_on_collision(self, tmp_path):
        from opencut.helpers import _unique_output_path
        existing = tmp_path / "file.mp4"
        existing.write_text("x")
        result = _unique_output_path(str(existing))
        assert result.endswith("file_2.mp4")

    def test_skips_existing_counters(self, tmp_path):
        from opencut.helpers import _unique_output_path
        (tmp_path / "file.mp4").write_text("x")
        (tmp_path / "file_2.mp4").write_text("x")
        result = _unique_output_path(str(tmp_path / "file.mp4"))
        assert result.endswith("file_3.mp4")


class TestMakeSequenceName:
    """_make_sequence_name generates display names."""

    def test_basic_name(self):
        from opencut.helpers import _make_sequence_name
        name = _make_sequence_name("/path/to/my_video.mp4")
        assert name == "OpenCut - my_video"

    def test_with_suffix(self):
        from opencut.helpers import _make_sequence_name
        name = _make_sequence_name("/video.mp4", suffix="denoised")
        assert name == "OpenCut - video (denoised)"

    def test_long_name_truncated(self):
        from opencut.helpers import _make_sequence_name
        long_name = "a" * 100 + ".mp4"
        name = _make_sequence_name(f"/path/{long_name}")
        assert len(name) < 80
        assert "..." in name


class TestFFmpegCmdBuilder:
    """FFmpegCmd fluent builder produces correct command lists."""

    def test_basic_build(self):
        from opencut.helpers import FFmpegCmd
        cmd = (FFmpegCmd()
               .input("/in.mp4")
               .video_codec("libx264", crf=18)
               .output("/out.mp4")
               .build())
        assert "-i" in cmd
        assert "/in.mp4" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-crf" in cmd
        assert "18" in cmd
        assert cmd[-1] == "/out.mp4"

    def test_hide_banner_and_overwrite(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").output("/out.mp4").build()
        assert "-hide_banner" in cmd
        assert "-y" in cmd

    def test_copy_streams(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").copy_streams().output("/out.mp4").build()
        assert "-c" in cmd
        assert "copy" in cmd

    def test_filter_complex_overrides_vf(self):
        from opencut.helpers import FFmpegCmd
        cmd = (FFmpegCmd()
               .input("/in.mp4")
               .video_filter("scale=1920:1080")
               .filter_complex("[0:v]scale=1920:1080[out]", maps=["[out]"])
               .output("/out.mp4")
               .build())
        assert "-filter_complex" in cmd
        assert "-vf" not in cmd

    def test_faststart(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").faststart().output("/out.mp4").build()
        assert "-movflags" in cmd
        assert "+faststart" in cmd

    def test_no_video(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").no_video().output("/out.wav").build()
        assert "-vn" in cmd

    def test_audio_codec(self):
        from opencut.helpers import FFmpegCmd
        cmd = (FFmpegCmd()
               .input("/in.mp4")
               .audio_codec("aac", bitrate="192k")
               .output("/out.mp4")
               .build())
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-b:a" in cmd
        assert "192k" in cmd

    def test_seek(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").seek(start=10, end=20).output("/out.mp4").build()
        assert "-ss" in cmd
        assert "10" in cmd
        assert "-to" in cmd
        assert "20" in cmd

    def test_frames(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").frames(1).output("/out.png").build()
        assert "-vframes" in cmd
        assert "1" in cmd

    def test_format_option(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().input("/in.mp4").format("null").output("/dev/null").build()
        assert "-f" in cmd
        assert "null" in cmd

    def test_pre_input(self):
        from opencut.helpers import FFmpegCmd
        cmd = FFmpegCmd().pre_input("ss", "5").input("/in.mp4").output("/out.mp4").build()
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx


# ========================================================================
# 6. opencut/checks.py — check_X_available functions with mocked _try_import
# ========================================================================

class TestCheckFailureRegistry:
    """Structured failure registry for check diagnostics."""

    def test_record_and_get(self):
        from opencut.checks import clear_check_failures, get_check_failures, record_check_failure
        clear_check_failures()
        record_check_failure("test_check", ImportError("no module"))
        failures = get_check_failures()
        assert "test_check" in failures
        assert failures["test_check"]["exception"] == "ImportError"
        assert "no module" in failures["test_check"]["message"]
        assert "ts" in failures["test_check"]
        clear_check_failures()

    def test_clear(self):
        from opencut.checks import clear_check_failures, get_check_failures, record_check_failure
        record_check_failure("x", ValueError("v"))
        clear_check_failures()
        assert get_check_failures() == {}

    def test_message_capped_at_500(self):
        from opencut.checks import clear_check_failures, get_check_failures, record_check_failure
        clear_check_failures()
        record_check_failure("big", RuntimeError("x" * 1000))
        failures = get_check_failures()
        assert len(failures["big"]["message"]) <= 500
        clear_check_failures()

    def test_thread_safe_copy(self):
        from opencut.checks import clear_check_failures, get_check_failures, record_check_failure
        clear_check_failures()
        record_check_failure("a", ValueError("v"))
        f1 = get_check_failures()
        f1["a"]["message"] = "mutated"
        f2 = get_check_failures()
        assert f2["a"]["message"] != "mutated"
        clear_check_failures()


class TestChecksSimpleImport:
    """check_X_available functions that use _try_import only."""

    def _patch_try_import(self, return_val):
        return patch("opencut.checks._try_import", return_value=return_val)

    def test_demucs_available(self):
        from opencut.checks import check_demucs_available
        with self._patch_try_import(MagicMock()):
            assert check_demucs_available() is True
        with self._patch_try_import(None):
            assert check_demucs_available() is False

    def test_pedalboard_available(self):
        from opencut.checks import check_pedalboard_available
        with self._patch_try_import(MagicMock()):
            assert check_pedalboard_available() is True
        with self._patch_try_import(None):
            assert check_pedalboard_available() is False

    def test_audiocraft_available(self):
        from opencut.checks import check_audiocraft_available
        with self._patch_try_import(MagicMock()):
            assert check_audiocraft_available() is True
        with self._patch_try_import(None):
            assert check_audiocraft_available() is False

    def test_edge_tts_available(self):
        from opencut.checks import check_edge_tts_available
        with self._patch_try_import(MagicMock()):
            assert check_edge_tts_available() is True
        with self._patch_try_import(None):
            assert check_edge_tts_available() is False

    def test_rembg_available(self):
        from opencut.checks import check_rembg_available
        with self._patch_try_import(MagicMock()):
            assert check_rembg_available() is True
        with self._patch_try_import(None):
            assert check_rembg_available() is False

    def test_upscale_available(self):
        from opencut.checks import check_upscale_available
        with self._patch_try_import(MagicMock()):
            assert check_upscale_available() is True
        with self._patch_try_import(None):
            assert check_upscale_available() is False

    def test_scenedetect_available(self):
        from opencut.checks import check_scenedetect_available
        with self._patch_try_import(MagicMock()):
            assert check_scenedetect_available() is True
        with self._patch_try_import(None):
            assert check_scenedetect_available() is False

    def test_ocr_available(self):
        from opencut.checks import check_ocr_available
        with self._patch_try_import(MagicMock()):
            assert check_ocr_available() is True
        with self._patch_try_import(None):
            assert check_ocr_available() is False

    def test_mediapipe_available(self):
        from opencut.checks import check_mediapipe_available
        with self._patch_try_import(MagicMock()):
            assert check_mediapipe_available() is True
        with self._patch_try_import(None):
            assert check_mediapipe_available() is False

    def test_otio_available(self):
        from opencut.checks import check_otio_available
        with self._patch_try_import(MagicMock()):
            assert check_otio_available() is True
        with self._patch_try_import(None):
            assert check_otio_available() is False

    def test_websocket_available(self):
        from opencut.checks import check_websocket_available
        with self._patch_try_import(MagicMock()):
            assert check_websocket_available() is True
        with self._patch_try_import(None):
            assert check_websocket_available() is False

    def test_deepface_available(self):
        from opencut.checks import check_deepface_available
        with self._patch_try_import(MagicMock()):
            assert check_deepface_available() is True
        with self._patch_try_import(None):
            assert check_deepface_available() is False

    def test_pyonfx_available(self):
        from opencut.checks import check_pyonfx_available
        with self._patch_try_import(MagicMock()):
            assert check_pyonfx_available() is True
        with self._patch_try_import(None):
            assert check_pyonfx_available() is False

    def test_sentry_available(self):
        from opencut.checks import check_sentry_available
        with self._patch_try_import(MagicMock()):
            assert check_sentry_available() is True
        with self._patch_try_import(None):
            assert check_sentry_available() is False

    def test_colour_science_available(self):
        from opencut.checks import check_colour_science_available
        with self._patch_try_import(MagicMock()):
            assert check_colour_science_available() is True
        with self._patch_try_import(None):
            assert check_colour_science_available() is False


class TestChecksAlwaysTrue:
    """Checks that always return True (stdlib-only)."""

    def test_footage_search_always_true(self):
        from opencut.checks import check_footage_search_available
        assert check_footage_search_available() is True

    def test_runpod_always_true(self):
        from opencut.checks import check_runpod_available
        assert check_runpod_available() is True

    def test_openapi_always_true(self):
        from opencut.checks import check_openapi_available
        assert check_openapi_available() is True

    def test_gpu_semaphore_always_true(self):
        from opencut.checks import check_gpu_semaphore_available
        assert check_gpu_semaphore_available() is True

    def test_rate_limit_categories_always_true(self):
        from opencut.checks import check_rate_limit_categories_available
        assert check_rate_limit_categories_available() is True

    def test_changelog_feed_always_true(self):
        from opencut.checks import check_changelog_feed_available
        assert check_changelog_feed_available() is True

    def test_issue_report_always_true(self):
        from opencut.checks import check_issue_report_available
        assert check_issue_report_available() is True

    def test_gist_sync_always_true(self):
        from opencut.checks import check_gist_sync_available
        assert check_gist_sync_available() is True

    def test_onboarding_always_true(self):
        from opencut.checks import check_onboarding_available
        assert check_onboarding_available() is True

    def test_upscale_hub_always_true(self):
        from opencut.checks import check_upscale_hub_available
        assert check_upscale_hub_available() is True


class TestChecksMultiModule:
    """Checks that need multiple modules (cv2 + numpy, etc.)."""

    def test_color_match_needs_cv2_and_numpy(self):
        from opencut.checks import check_color_match_available
        with patch("opencut.checks._try_import", return_value=MagicMock()):
            assert check_color_match_available() is True

    def test_color_match_missing_one(self):
        from opencut.checks import check_color_match_available
        def selective_import(name):
            if name == "cv2":
                return MagicMock()
            return None
        with patch("opencut.checks._try_import", side_effect=selective_import):
            assert check_color_match_available() is False

    def test_auto_zoom_needs_cv2(self):
        from opencut.checks import check_auto_zoom_available
        with patch("opencut.checks._try_import", return_value=MagicMock()):
            assert check_auto_zoom_available() is True
        with patch("opencut.checks._try_import", return_value=None):
            assert check_auto_zoom_available() is False


class TestCheckRunpodApiKey:
    """check_runpod_api_key_set reads env var."""

    def test_set(self):
        from opencut.checks import check_runpod_api_key_set
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "sk-123"}):
            assert check_runpod_api_key_set() is True

    def test_empty(self):
        from opencut.checks import check_runpod_api_key_set
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "  "}):
            assert check_runpod_api_key_set() is False

    def test_missing(self):
        from opencut.checks import check_runpod_api_key_set
        env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert check_runpod_api_key_set() is False


class TestCheckSocialPost:
    """check_social_post_available returns bool."""

    def test_returns_bool(self):
        from opencut.checks import check_social_post_available
        assert isinstance(check_social_post_available(), bool)
