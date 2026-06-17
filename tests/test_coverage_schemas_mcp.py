"""Coverage expansion: schemas + MCP tool validation."""
from __future__ import annotations

# ---- Schema dataclass tests ----

class TestJobResponse:
    def test_defaults(self):
        from opencut.schemas import JobResponse
        r = JobResponse(job_id="abc123")
        assert r.job_id == "abc123"
        assert r.status == "running"

    def test_to_dict_strips_none(self):
        from opencut.schemas import JobResponse
        d = JobResponse(job_id="x").to_dict()
        assert "job_id" in d
        assert d["status"] == "running"
        assert None not in d.values()


class TestJobResult:
    def test_defaults(self):
        from opencut.schemas import JobResult
        r = JobResult()
        assert r.status == "complete"
        assert r.progress == 100

    def test_to_dict_strips_none_fields(self):
        from opencut.schemas import JobResult
        d = JobResult().to_dict()
        assert "result" not in d
        assert "error" not in d
        assert d["status"] == "complete"

    def test_with_error(self):
        from opencut.schemas import JobResult
        r = JobResult(status="error", error="boom", progress=50)
        d = r.to_dict()
        assert d["error"] == "boom"
        assert d["status"] == "error"


class TestHealthResult:
    def test_defaults(self):
        from opencut.schemas import HealthResult
        r = HealthResult()
        assert r.status == "ok"

    def test_to_dict(self):
        from opencut.schemas import HealthResult
        d = HealthResult(version="1.33.1", csrf_token="tok").to_dict()
        assert d["version"] == "1.33.1"
        assert d["csrf_token"] == "tok"


class TestActionResult:
    def test_defaults(self):
        from opencut.schemas import ActionResult
        r = ActionResult()
        assert r.success is True
        assert r.ok is True

    def test_to_dict_strips_none(self):
        from opencut.schemas import ActionResult
        d = ActionResult(message="done").to_dict()
        assert "output" not in d
        assert d["message"] == "done"


class TestSettingsResult:
    def test_defaults(self):
        from opencut.schemas import SettingsResult
        r = SettingsResult()
        assert r.ok is True
        assert r.settings == {}


class TestListEnvelopeResult:
    def test_defaults(self):
        from opencut.schemas import ListEnvelopeResult
        r = ListEnvelopeResult()
        assert r.items == []
        assert r.total == 0

    def test_with_items(self):
        from opencut.schemas import ListEnvelopeResult
        r = ListEnvelopeResult(items=[{"id": 1}], total=1)
        d = r.to_dict()
        assert d["total"] == 1
        assert len(d["items"]) == 1


class TestCapabilityResult:
    def test_defaults(self):
        from opencut.schemas import CapabilityResult
        r = CapabilityResult()
        assert r.available is False

    def test_available(self):
        from opencut.schemas import CapabilityResult
        d = CapabilityResult(available=True, status="ready").to_dict()
        assert d["available"] is True


class TestFileOutputResult:
    def test_defaults(self):
        from opencut.schemas import FileOutputResult
        r = FileOutputResult()
        assert r.output == ""

    def test_with_output(self):
        from opencut.schemas import FileOutputResult
        d = FileOutputResult(output="/tmp/out.mp4", count=1).to_dict()
        assert d["output"] == "/tmp/out.mp4"


class TestJobStatusResult:
    def test_defaults(self):
        from opencut.schemas import JobStatusResult
        r = JobStatusResult()
        assert r.progress == 0

    def test_to_dict(self):
        from opencut.schemas import JobStatusResult
        d = JobStatusResult(job_id="j1", status="running", progress=50).to_dict()
        assert d["job_id"] == "j1"


class TestJobListResult:
    def test_defaults(self):
        from opencut.schemas import JobListResult
        r = JobListResult()
        assert r.jobs == []
        assert r.total == 0


class TestJobStatsResult:
    def test_defaults(self):
        from opencut.schemas import JobStatsResult
        r = JobStatsResult()
        assert r.total == 0
        assert r.running == 0


class TestModelListResult:
    def test_defaults(self):
        from opencut.schemas import ModelListResult
        r = ModelListResult()
        assert r.models == []
        assert r.total == 0


class TestGpuStatusResult:
    def test_defaults(self):
        from opencut.schemas import GpuStatusResult
        r = GpuStatusResult()
        assert r.available is False
        assert r.vram_total_mb == 0


class TestCaptionPreviewResult:
    def test_defaults(self):
        from opencut.schemas import CaptionPreviewResult
        r = CaptionPreviewResult()
        assert r.css == ""
        assert r.warnings == []


class TestStripNone:
    def test_strips_none_values(self):
        from opencut.schemas import _strip_none
        assert _strip_none({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}

    def test_preserves_falsy_non_none(self):
        from opencut.schemas import _strip_none
        d = _strip_none({"a": 0, "b": "", "c": False, "d": [], "e": None})
        assert d == {"a": 0, "b": "", "c": False, "d": []}


# ---- MCP tool validation tests ----

class TestMcpToolCatalog:
    def test_tool_count_at_least_80(self):
        from opencut.mcp_server import MCP_TOOLS
        assert len(MCP_TOOLS) >= 80

    def test_all_tools_have_required_fields(self):
        from opencut.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "inputSchema" in tool, f"Tool {tool.get('name')} missing 'inputSchema'"

    def test_all_tool_names_are_prefixed(self):
        from opencut.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert tool["name"].startswith("opencut_"), f"Tool {tool['name']} missing opencut_ prefix"

    def test_no_duplicate_tool_names(self):
        from opencut.mcp_server import MCP_TOOLS
        names = [t["name"] for t in MCP_TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tools: {[n for n in names if names.count(n) > 1]}"

    def test_all_tools_have_matching_routes(self):
        from opencut.mcp_server import _TOOL_ROUTES, MCP_TOOLS
        for tool in MCP_TOOLS:
            assert tool["name"] in _TOOL_ROUTES, f"Tool {tool['name']} has no route mapping"

    def test_input_schemas_are_valid_objects(self):
        from opencut.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            schema = tool["inputSchema"]
            assert schema.get("type") == "object", f"Tool {tool['name']} schema type != object"
            assert "properties" in schema, f"Tool {tool['name']} schema missing properties"

    def test_route_methods_are_valid(self):
        from opencut.mcp_server import _TOOL_ROUTES
        valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
        for name, (method, path) in _TOOL_ROUTES.items():
            assert method in valid_methods, f"Tool {name} has invalid method {method}"
            assert path.startswith("/"), f"Tool {name} path {path} doesn't start with /"

    def test_descriptions_are_nonempty(self):
        from opencut.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert len(tool["description"]) > 10, f"Tool {tool['name']} has short description"

    def test_route_count_matches_tool_count(self):
        from opencut.mcp_server import _TOOL_ROUTES, MCP_TOOLS
        assert len(MCP_TOOLS) == len(_TOOL_ROUTES)


class TestMcpToolSchemaQuality:
    def test_filepath_tools_require_filepath(self):
        from opencut.mcp_server import MCP_TOOLS
        filepath_tools = [t for t in MCP_TOOLS if "filepath" in t["inputSchema"].get("properties", {})]
        for tool in filepath_tools:
            if "filepath" in tool["inputSchema"]["properties"]:
                props = tool["inputSchema"]["properties"]["filepath"]
                assert props.get("type") == "string", f"Tool {tool['name']} filepath not string"

    def test_no_empty_properties(self):
        from opencut.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            props = tool["inputSchema"].get("properties", {})
            for key, prop in props.items():
                assert "type" in prop or "enum" in prop, (
                    f"Tool {tool['name']} property {key} has no type"
                )


# ---- Error module tests ----

class TestOpenCutError:
    def test_construction(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("TEST_CODE", "test message", status=418, suggestion="try again")
        assert e.code == "TEST_CODE"
        assert e.message == "test message"
        assert e.status == 418
        assert e.suggestion == "try again"
        assert str(e) == "test message"

    def test_default_status(self):
        from opencut.errors import OpenCutError
        e = OpenCutError("X", "y")
        assert e.status == 400
        assert e.suggestion == ""


class TestErrorFactories:
    def test_missing_dependency(self):
        from opencut.errors import missing_dependency
        e = missing_dependency("torch")
        assert e.code == "MISSING_DEPENDENCY"
        assert "torch" in e.message
        assert e.status == 503

    def test_file_not_found(self):
        from opencut.errors import file_not_found
        e = file_not_found("/path/to/video.mp4")
        assert e.code == "FILE_NOT_FOUND"
        assert "video.mp4" in e.message
        assert e.status == 404

    def test_file_not_found_empty(self):
        from opencut.errors import file_not_found
        e = file_not_found("")
        assert e.code == "FILE_NOT_FOUND"
        assert e.status == 404

    def test_gpu_out_of_memory(self):
        from opencut.errors import gpu_out_of_memory
        e = gpu_out_of_memory()
        assert e.code == "GPU_OUT_OF_MEMORY"
        assert e.status == 503

    def test_invalid_input(self):
        from opencut.errors import invalid_input
        e = invalid_input("bad parameter")
        assert e.code == "INVALID_INPUT"
        assert "bad parameter" in e.message
        assert e.status == 400

    def test_invalid_model(self):
        from opencut.errors import invalid_model
        e = invalid_model("gpt-99", allowed=["base", "large"])
        assert e.code == "INVALID_MODEL"
        assert "gpt-99" in e.message
        assert "base" in e.suggestion

    def test_operation_failed(self):
        from opencut.errors import operation_failed
        e = operation_failed("FFmpeg crashed")
        assert e.code == "OPERATION_FAILED"
        assert e.status == 500

    def test_rate_limited(self):
        from opencut.errors import rate_limited
        e = rate_limited("model install")
        assert e.code == "RATE_LIMITED"
        assert "model install" in e.message
        assert e.status == 429

    def test_queue_full(self):
        from opencut.errors import queue_full
        e = queue_full(50)
        assert e.code == "QUEUE_FULL"
        assert "50" in e.message
        assert e.status == 429

    def test_module_not_available(self):
        from opencut.errors import module_not_available
        e = module_not_available("demucs")
        assert e.code == "MODULE_NOT_AVAILABLE"
        assert "demucs" in e.message

    def test_file_permission_denied(self):
        from opencut.errors import file_permission_denied
        e = file_permission_denied("/tmp/locked.mp4")
        assert e.code == "PERMISSION_DENIED"
        assert "/tmp/locked.mp4" in e.message

    def test_too_many_items(self):
        from opencut.errors import too_many_items
        e = too_many_items("files", 100)
        assert e.code == "TOO_MANY_ITEMS"
        assert e.status == 400


# ---- Config module tests ----

class TestOpenCutConfig:
    def test_defaults(self):
        from opencut.config import OpenCutConfig
        cfg = OpenCutConfig()
        assert cfg.local_only is False
        assert cfg.bundled_mode is False
        assert cfg.max_content_length == 100 * 1024 * 1024

    def test_custom_values(self):
        from opencut.config import OpenCutConfig
        cfg = OpenCutConfig(local_only=True, bundled_mode=True)
        assert cfg.local_only is True
        assert cfg.bundled_mode is True


class TestIsLocalOnly:
    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_LOCAL_ONLY", raising=False)
        from opencut.config import is_local_only
        assert is_local_only() is False

    def test_env_var_enables(self, monkeypatch):
        monkeypatch.setenv("OPENCUT_LOCAL_ONLY", "1")
        import importlib

        from opencut import config
        importlib.reload(config)
        try:
            assert config.is_local_only() is True
        finally:
            monkeypatch.delenv("OPENCUT_LOCAL_ONLY", raising=False)
            importlib.reload(config)


# ---- Helpers module tests ----

class TestTryImport:
    def test_valid_module(self):
        from opencut.helpers import _try_import
        result = _try_import("json")
        assert result is not None

    def test_invalid_module(self):
        from opencut.helpers import _try_import
        result = _try_import("nonexistent_module_xyz_12345")
        assert result is None

    def test_empty_string(self):
        from opencut.helpers import _try_import
        result = _try_import("_____no_such_module_ever_____")
        assert result is None


class TestComputeEstimate:
    def test_returns_dict(self):
        from opencut.helpers import compute_estimate
        result = compute_estimate(job_type="silence", file_duration=60.0)
        assert isinstance(result, dict)


# ---- Checks module tests ----

class TestChecksAvailability:
    def test_check_loudness_match_available(self):
        from opencut.checks import check_loudness_match_available
        result = check_loudness_match_available()
        assert isinstance(result, bool)

    def test_check_footage_search_available(self):
        from opencut.checks import check_footage_search_available
        result = check_footage_search_available()
        assert result is True

    def test_check_autoshot_available(self):
        from opencut.checks import check_autoshot_available
        result = check_autoshot_available()
        assert isinstance(result, bool)

    def test_check_ocr_available(self):
        from opencut.checks import check_ocr_available
        result = check_ocr_available()
        assert isinstance(result, bool)

    def test_check_transnetv2_available(self):
        from opencut.checks import check_transnetv2_available
        result = check_transnetv2_available()
        assert isinstance(result, bool)

    def test_check_color_match_available(self):
        from opencut.checks import check_color_match_available
        result = check_color_match_available()
        assert isinstance(result, bool)

    def test_check_auto_zoom_available(self):
        from opencut.checks import check_auto_zoom_available
        result = check_auto_zoom_available()
        assert isinstance(result, bool)

    def test_check_neural_interp_available(self):
        from opencut.checks import check_neural_interp_available
        result = check_neural_interp_available()
        assert isinstance(result, bool)

    def test_check_declarative_compose_available(self):
        from opencut.checks import check_declarative_compose_available
        result = check_declarative_compose_available()
        assert isinstance(result, bool)

    def test_check_ollama_available(self):
        from opencut.checks import check_ollama_available
        result = check_ollama_available()
        assert isinstance(result, bool)


# ---- Workers module tests ----

class TestJobPriority:
    def test_priority_values(self):
        from opencut.workers import JobPriority
        assert JobPriority.CRITICAL.value < JobPriority.HIGH.value
        assert JobPriority.HIGH.value < JobPriority.NORMAL.value
        assert JobPriority.NORMAL.value < JobPriority.LOW.value
        assert JobPriority.LOW.value < JobPriority.BACKGROUND.value

    def test_get_pool_returns_pool(self):
        from opencut.workers import get_pool
        pool = get_pool()
        assert pool is not None


# ---- user_data config schema tests ----

class TestConfigSchemaRegistry:
    def test_register_and_read(self):

        from opencut.user_data import (
            CONFIG_SCHEMAS,
            register_config_schema,
        )

        register_config_schema(
            "_test_schema_check.json",
            version=2,
            migrations={
                1: lambda d: {**d, "_migrated_v1": True},
                2: lambda d: {**d, "_migrated_v2": True},
            },
        )
        assert "_test_schema_check.json" in CONFIG_SCHEMAS

    def test_versioned_read_returns_none_for_nonexistent(self):
        from opencut.user_data import read_user_file_versioned
        result = read_user_file_versioned("_nonexistent_test_file_xyzzy.json", default=None)
        assert result is None
