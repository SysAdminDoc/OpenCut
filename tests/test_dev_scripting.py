"""
OpenCut Developer & Scripting Platform Tests

Comprehensive tests for Category 81 features:
  1. Scripting Console — sandbox safety, timeout, stdout capture, history
  2. Macro Recorder — record/play, variable substitution, CRUD
  3. Filter Chain Builder — filter types, validation, cycle detection, FFmpeg strings
  4. Webhook System — registration CRUD, delivery, retry, test event
  5. Batch Scripting — glob expansion, operation chaining, dry-run, error handling

~130 tests covering core logic and route endpoints.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tests.conftest import csrf_headers


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def app():
    """Create a Flask app with dev_scripting_bp registered."""
    from opencut.config import OpenCutConfig
    from opencut.routes.dev_scripting_routes import dev_scripting_bp
    from opencut.server import create_app
    test_config = OpenCutConfig()
    flask_app = create_app(config=test_config)
    flask_app.config["TESTING"] = True
    # Register the dev_scripting_bp if not already registered
    if "dev_scripting" not in flask_app.blueprints:
        flask_app.register_blueprint(dev_scripting_bp)
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client with dev_scripting_bp available."""
    return app.test_client()


@pytest.fixture
def csrf_token(client):
    """Fetch a valid CSRF token from the /health endpoint."""
    resp = client.get("/health")
    data = resp.get_json()
    return data.get("csrf_token", "")


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
    d = tempfile.mkdtemp(prefix="opencut_devscript_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_opencut_dir(tmp_dir, monkeypatch):
    """Override OpenCut user dir to a temp directory."""
    monkeypatch.setattr(
        "opencut.core.scripting_console._OPENCUT_DIR", tmp_dir)
    monkeypatch.setattr(
        "opencut.core.scripting_console._HISTORY_FILE",
        os.path.join(tmp_dir, "console_history.json"))
    monkeypatch.setattr(
        "opencut.core.macro_recorder._OPENCUT_DIR", tmp_dir)
    monkeypatch.setattr(
        "opencut.core.macro_recorder._MACROS_DIR",
        os.path.join(tmp_dir, "macros"))
    monkeypatch.setattr(
        "opencut.core.webhook_system._OPENCUT_DIR", tmp_dir)
    monkeypatch.setattr(
        "opencut.core.webhook_system._WEBHOOKS_FILE",
        os.path.join(tmp_dir, "webhooks.json"))
    monkeypatch.setattr(
        "opencut.core.webhook_system._DELIVERY_LOG_FILE",
        os.path.join(tmp_dir, "webhook_deliveries.json"))
    monkeypatch.setattr(
        "opencut.core.batch_script._OPENCUT_DIR", tmp_dir)
    monkeypatch.setattr(
        "opencut.core.batch_script._BATCH_LOG_DIR",
        os.path.join(tmp_dir, "batch_logs"))
    return tmp_dir


@pytest.fixture
def sample_files(tmp_dir):
    """Create sample files for batch testing."""
    files = []
    for i in range(3):
        path = os.path.join(tmp_dir, f"video_{i}.mp4")
        with open(path, "w") as f:
            f.write(f"fake video {i}")
        files.append(path)
    return files


# ============================================================================
# 1. SCRIPTING CONSOLE — Core Logic
# ============================================================================

class TestScriptingConsoleCore:
    """Unit tests for opencut.core.scripting_console."""

    def test_execute_basic_print(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print('hello world')")
        assert result.success is True
        assert "hello world" in result.output
        assert result.execution_time_ms > 0

    def test_execute_math_expression(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(2 + 3)")
        assert result.success is True
        assert "5" in result.output

    def test_execute_math_module(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import math\nprint(math.pi)")
        assert result.success is True
        assert "3.14" in result.output

    def test_execute_json_module(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import json\nprint(json.dumps({'a': 1}))")
        assert result.success is True
        assert '"a"' in result.output

    def test_execute_top_level_math_helpers(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(sqrt(16))")
        assert result.success is True
        assert "4" in result.output

    def test_execute_json_helpers(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(dumps({'x': 42}))")
        assert result.success is True
        assert '"x"' in result.output

    def test_execute_empty_code(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("")
        assert result.success is True
        assert result.output == ""

    def test_execute_whitespace_only(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("   \n  \n  ")
        assert result.success is True

    def test_execute_syntax_error(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("def foo(")
        assert result.success is False
        assert "Syntax error" in result.error or "SyntaxError" in result.error

    def test_execute_runtime_error(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("x = 1 / 0")
        assert result.success is False
        assert "ZeroDivision" in result.error

    def test_execute_context_variables(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(my_var)", context={"my_var": 42})
        assert result.success is True
        assert "42" in result.output

    def test_execute_context_rejects_private_keys(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(_secret)", context={"_secret": "bad"})
        assert result.success is False

    # --- Sandbox safety tests ---

    def test_sandbox_blocks_os_import(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import os")
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_sandbox_blocks_sys_import(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import sys")
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_sandbox_blocks_subprocess(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import subprocess")
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_sandbox_blocks_open(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("f = open('/etc/passwd')")
        assert result.success is False

    def test_sandbox_blocks_eval(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("eval('1+1')")
        assert result.success is False

    def test_sandbox_blocks_exec(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("exec('x=1')")
        assert result.success is False

    def test_sandbox_blocks___import__(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("__import__('os')")
        assert result.success is False

    def test_sandbox_blocks_dunder_class(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print(''.__class__)")
        assert result.success is False
        assert "__class__" in result.error

    def test_sandbox_blocks_dunder_subclasses(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("x = object.__subclasses__()")
        assert result.success is False

    def test_sandbox_blocks_dunder_globals(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print.__globals__")
        assert result.success is False

    def test_sandbox_blocks_dunder_builtins(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("x = __builtins__")
        assert result.success is False

    def test_sandbox_blocks_dunder_code(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("print.__code__")
        assert result.success is False

    def test_sandbox_blocks_socket(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import socket")
        assert result.success is False

    def test_sandbox_blocks_shutil(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import shutil")
        assert result.success is False

    def test_sandbox_blocks_pickle(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import pickle")
        assert result.success is False

    def test_sandbox_blocks_io(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import io")
        assert result.success is False

    def test_sandbox_allows_re(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import re\nprint(re.match(r'\\d+', '123').group())")
        assert result.success is True
        assert "123" in result.output

    def test_sandbox_allows_datetime(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import datetime\nprint(datetime.date.today())")
        assert result.success is True

    def test_sandbox_allows_collections(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script(
            "from collections import Counter\nc = Counter('aabb')\nprint(c)")
        assert result.success is True

    def test_timeout_enforcement(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("import time\ntime.sleep(10)", timeout=2)
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_timeout_infinite_loop(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script("while True: pass", timeout=2)
        assert result.success is False
        assert "timed out" in result.error.lower()

    # --- History tests ---

    def test_history_records_execution(self, tmp_opencut_dir):
        from opencut.core.scripting_console import execute_script, get_history
        execute_script("print('test')")
        history = get_history()
        assert len(history) >= 1
        assert history[-1]["success"] is True
        assert "print" in history[-1]["code"]

    def test_history_records_failures(self, tmp_opencut_dir):
        from opencut.core.scripting_console import execute_script, get_history
        execute_script("1/0")
        history = get_history()
        assert len(history) >= 1
        assert history[-1]["success"] is False

    def test_history_limit(self, tmp_opencut_dir):
        from opencut.core.scripting_console import execute_script, get_history
        for i in range(5):
            execute_script(f"print({i})")
        history = get_history(limit=3)
        assert len(history) <= 3

    def test_clear_history(self, tmp_opencut_dir):
        from opencut.core.scripting_console import (
            clear_history,
            execute_script,
            get_history,
        )
        execute_script("print(1)")
        clear_history()
        history = get_history()
        assert len(history) == 0

    # --- Namespace tests ---

    def test_get_available_modules(self):
        from opencut.core.scripting_console import get_available_modules
        modules = get_available_modules()
        assert "math" in modules
        assert "json" in modules
        assert "os" not in modules
        assert "sys" not in modules

    def test_get_namespace_info(self):
        from opencut.core.scripting_console import get_namespace_info
        info = get_namespace_info()
        assert "modules" in info
        assert "functions" in info
        assert "blocked_builtins" in info
        assert "open" in info["blocked_builtins"]

    def test_get_available_functions(self):
        from opencut.core.scripting_console import get_available_functions
        funcs = get_available_functions()
        names = [f["name"] for f in funcs]
        assert "opencut.get_video_info" in names

    def test_opencut_namespace_helpers_are_callable(self):
        from opencut.core.scripting_console import execute_script
        result = execute_script(
            "info = opencut.get_video_info('missing.mp4')\n"
            "print(isinstance(info, dict))"
        )
        assert result.success is True
        assert "True" in result.output


# ============================================================================
# 2. MACRO RECORDER — Core Logic
# ============================================================================

class TestMacroRecorderCore:
    """Unit tests for opencut.core.macro_recorder."""

    def test_start_recording(self):
        from opencut.core.macro_recorder import start_recording
        result = start_recording(session_id="test_start")
        assert result["recording"] is True
        assert "started" in result["message"].lower()

    def test_double_start_returns_already_recording(self):
        from opencut.core.macro_recorder import start_recording
        start_recording(session_id="test_double")
        result = start_recording(session_id="test_double")
        assert "already" in result["message"].lower()

    def test_add_step(self):
        from opencut.core.macro_recorder import add_step, start_recording
        start_recording(session_id="test_add")
        recorded = add_step("/api/silence", payload={"threshold": -30},
                            session_id="test_add")
        assert recorded is True

    def test_add_step_not_recording(self):
        from opencut.core.macro_recorder import add_step
        recorded = add_step("/api/silence", session_id="not_recording")
        assert recorded is False

    def test_stop_recording(self):
        from opencut.core.macro_recorder import (
            add_step,
            start_recording,
            stop_recording,
        )
        start_recording(session_id="test_stop")
        add_step("/api/silence", payload={"threshold": -30},
                 session_id="test_stop")
        add_step("/api/denoise", payload={"level": 0.5},
                 session_id="test_stop")
        macro = stop_recording(session_id="test_stop", name="Test Macro")
        assert macro.name == "Test Macro"
        assert len(macro.steps) == 2
        assert macro.steps[0].endpoint == "/api/silence"

    def test_stop_recording_no_session_raises(self):
        from opencut.core.macro_recorder import stop_recording
        with pytest.raises(ValueError, match="No active recording"):
            stop_recording(session_id="nonexistent")

    def test_macro_to_dict_and_from_dict(self):
        from opencut.core.macro_recorder import MacroRecording, MacroStep
        macro = MacroRecording(
            name="Test",
            steps=[
                MacroStep(endpoint="/api/silence", payload={"t": -30}),
                MacroStep(endpoint="/api/denoise", method="POST"),
            ],
        )
        d = macro.to_dict()
        loaded = MacroRecording.from_dict(d)
        assert loaded.name == "Test"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].endpoint == "/api/silence"

    def test_variable_substitution(self):
        from opencut.core.macro_recorder import _substitute_vars
        payload = {
            "filepath": "${input_file}",
            "output": "${output_dir}/result.mp4",
            "tag": "ts_${timestamp}",
        }
        variables = {
            "input_file": "/videos/clip.mp4",
            "output_dir": "/output",
            "timestamp": "20260414",
        }
        result = _substitute_vars(payload, variables)
        assert result["filepath"] == "/videos/clip.mp4"
        assert result["output"] == "/output/result.mp4"
        assert result["tag"] == "ts_20260414"

    def test_variable_substitution_nested(self):
        from opencut.core.macro_recorder import _substitute_vars
        payload = {"nested": {"path": "${input_file}"}}
        result = _substitute_vars(payload, {"input_file": "/a.mp4"})
        assert result["nested"]["path"] == "/a.mp4"

    def test_variable_substitution_list(self):
        from opencut.core.macro_recorder import _substitute_vars
        payload = {"files": ["${input_file}", "other.mp4"]}
        result = _substitute_vars(payload, {"input_file": "/a.mp4"})
        assert result["files"][0] == "/a.mp4"
        assert result["files"][1] == "other.mp4"

    def test_variable_substitution_unknown_var_unchanged(self):
        from opencut.core.macro_recorder import _substitute_vars
        payload = {"x": "${unknown_var}"}
        result = _substitute_vars(payload, {})
        assert result["x"] == "${unknown_var}"

    def test_play_macro_dry_run(self):
        from opencut.core.macro_recorder import MacroRecording, MacroStep, play_macro
        macro = MacroRecording(
            name="Play Test",
            steps=[MacroStep(endpoint="/api/silence", payload={"t": -30})],
        )
        results = play_macro(macro, target_file="/test.mp4")
        assert len(results) == 1
        assert results[0]["dry_run"] is True
        assert results[0]["endpoint"] == "/api/silence"

    def test_play_macro_with_executor(self):
        from opencut.core.macro_recorder import MacroRecording, MacroStep, play_macro

        def fake_executor(endpoint, method, payload):
            return {"status": "ok"}

        macro = MacroRecording(
            name="Exec Test",
            steps=[MacroStep(endpoint="/api/test", payload={})],
        )
        results = play_macro(macro, executor=fake_executor)
        assert results[0]["success"] is True
        assert results[0]["result"]["status"] == "ok"

    def test_play_macro_executor_error(self):
        from opencut.core.macro_recorder import MacroRecording, MacroStep, play_macro

        def bad_executor(endpoint, method, payload):
            raise RuntimeError("boom")

        macro = MacroRecording(
            name="Err Test",
            steps=[MacroStep(endpoint="/api/test", payload={})],
        )
        results = play_macro(macro, executor=bad_executor)
        assert results[0]["success"] is False
        assert "boom" in results[0]["error"]

    # --- CRUD tests ---

    def test_save_and_load_macro(self, tmp_opencut_dir):
        from opencut.core.macro_recorder import (
            MacroRecording,
            MacroStep,
            load_macro,
            save_macro,
        )
        macro = MacroRecording(
            name="SaveTest",
            steps=[MacroStep(endpoint="/api/test", payload={"a": 1})],
        )
        path = save_macro(macro)
        assert os.path.isfile(path)

        loaded = load_macro(path)
        assert loaded.name == "SaveTest"
        assert len(loaded.steps) == 1

    def test_list_macros(self, tmp_opencut_dir):
        from opencut.core.macro_recorder import (
            MacroRecording,
            MacroStep,
            list_macros,
            save_macro,
        )
        save_macro(MacroRecording(name="A", steps=[
            MacroStep(endpoint="/api/a")]))
        save_macro(MacroRecording(name="B", steps=[
            MacroStep(endpoint="/api/b")]))
        macros = list_macros()
        assert len(macros) >= 2
        names = [m["name"] for m in macros]
        assert "A" in names
        assert "B" in names

    def test_delete_macro(self, tmp_opencut_dir):
        from opencut.core.macro_recorder import (
            MacroRecording,
            MacroStep,
            delete_macro,
            list_macros,
            save_macro,
        )
        save_macro(MacroRecording(name="ToDelete", steps=[
            MacroStep(endpoint="/api/x")]))
        assert delete_macro("ToDelete") is True
        macros = list_macros()
        names = [m["name"] for m in macros]
        assert "ToDelete" not in names

    def test_delete_nonexistent_macro(self, tmp_opencut_dir):
        from opencut.core.macro_recorder import delete_macro
        assert delete_macro("does_not_exist") is False

    def test_export_import_macro(self, tmp_opencut_dir, tmp_dir):
        from opencut.core.macro_recorder import (
            MacroRecording,
            MacroStep,
            export_macro,
            import_macro,
            save_macro,
        )
        save_macro(MacroRecording(name="ExportMe", steps=[
            MacroStep(endpoint="/api/x")]))
        export_path = os.path.join(tmp_dir, "exported.opencut-macro")
        export_macro("ExportMe", export_path)
        assert os.path.isfile(export_path)

        imported = import_macro(export_path)
        assert imported.name == "ExportMe"


# ============================================================================
# 3. FILTER CHAIN BUILDER — Core Logic
# ============================================================================

class TestFilterChainBuilderCore:
    """Unit tests for opencut.core.filter_chain_builder."""

    def test_single_scale_node(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, build_filter_string
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name="scale",
                       params={"w": 1280, "h": 720}),
        ])
        result = build_filter_string(chain)
        assert "scale" in result
        assert "1280" in result
        assert "720" in result

    def test_chain_two_nodes(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, build_filter_string
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name="scale",
                       params={"w": 1280, "h": 720}),
            FilterNode(node_id="n1", filter_name="eq",
                       params={"brightness": 0.1}),
        ])
        result = build_filter_string(chain)
        assert "scale" in result
        assert "eq" in result
        assert ";" in result

    def test_connected_nodes(self):
        from opencut.core.filter_chain_builder import (
            FilterChain,
            FilterConnection,
            FilterNode,
            build_filter_string,
        )
        chain = FilterChain(
            nodes=[
                FilterNode(node_id="n0", filter_name="scale",
                           params={"w": 1280, "h": 720}),
                FilterNode(node_id="n1", filter_name="eq",
                           params={"brightness": 0.1}),
            ],
            connections=[
                FilterConnection(from_node="n0", from_pad="default",
                                 to_node="n1", to_pad="default"),
            ],
        )
        result = build_filter_string(chain)
        assert "[n0_default]" in result

    def test_validate_empty_chain(self):
        from opencut.core.filter_chain_builder import FilterChain, validate_chain
        chain = FilterChain()
        result = validate_chain(chain)
        assert result["valid"] is False
        assert any("at least one" in e.lower() for e in result["errors"])

    def test_validate_missing_filter_name(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, validate_chain
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name=""),
        ])
        result = validate_chain(chain)
        assert result["valid"] is False

    def test_validate_duplicate_node_ids(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, validate_chain
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name="scale"),
            FilterNode(node_id="n0", filter_name="crop"),
        ])
        result = validate_chain(chain)
        assert result["valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_validate_bad_connection_ref(self):
        from opencut.core.filter_chain_builder import (
            FilterChain,
            FilterConnection,
            FilterNode,
            validate_chain,
        )
        chain = FilterChain(
            nodes=[FilterNode(node_id="n0", filter_name="scale")],
            connections=[
                FilterConnection(from_node="n0", from_pad="default",
                                 to_node="n999", to_pad="default"),
            ],
        )
        result = validate_chain(chain)
        assert result["valid"] is False
        assert any("unknown" in e.lower() for e in result["errors"])

    def test_validate_unknown_filter_warning(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, validate_chain
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name="totally_fake_filter"),
        ])
        result = validate_chain(chain)
        # Unknown filter is a warning, not an error
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_cycle_detection(self):
        from opencut.core.filter_chain_builder import (
            FilterChain,
            FilterConnection,
            FilterNode,
            validate_chain,
        )
        chain = FilterChain(
            nodes=[
                FilterNode(node_id="a", filter_name="scale"),
                FilterNode(node_id="b", filter_name="crop"),
                FilterNode(node_id="c", filter_name="eq"),
            ],
            connections=[
                FilterConnection(from_node="a", from_pad="d", to_node="b", to_pad="d"),
                FilterConnection(from_node="b", from_pad="d", to_node="c", to_pad="d"),
                FilterConnection(from_node="c", from_pad="d", to_node="a", to_pad="d"),
            ],
        )
        result = validate_chain(chain)
        assert result["valid"] is False
        assert any("cycle" in e.lower() for e in result["errors"])

    def test_no_cycle_linear(self):
        from opencut.core.filter_chain_builder import (
            FilterChain,
            FilterConnection,
            FilterNode,
            validate_chain,
        )
        chain = FilterChain(
            nodes=[
                FilterNode(node_id="a", filter_name="scale"),
                FilterNode(node_id="b", filter_name="crop"),
                FilterNode(node_id="c", filter_name="eq"),
            ],
            connections=[
                FilterConnection(from_node="a", from_pad="d", to_node="b", to_pad="d"),
                FilterConnection(from_node="b", from_pad="d", to_node="c", to_pad="d"),
            ],
        )
        result = validate_chain(chain)
        assert result["valid"] is True

    # --- All 20 filter types ---

    @pytest.mark.parametrize("filter_name", [
        "scale", "crop", "overlay", "drawtext", "colorbalance",
        "eq", "hue", "unsharp", "noise", "vignette",
        "fade", "concat", "split", "hstack", "vstack",
        "pad", "transpose", "setpts", "volume", "amerge",
    ])
    def test_known_filter_types(self, filter_name):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, build_filter_string
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name=filter_name),
        ])
        result = build_filter_string(chain)
        assert filter_name in result

    def test_drawtext_colon_escape(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode, build_filter_string
        chain = FilterChain(nodes=[
            FilterNode(node_id="n0", filter_name="drawtext",
                       params={"text": "Time: 12:00", "fontsize": 24}),
        ])
        result = build_filter_string(chain)
        assert "Time\\: 12\\:00" in result

    def test_from_dict_round_trip(self):
        from opencut.core.filter_chain_builder import FilterChain, FilterNode
        chain = FilterChain(
            name="test",
            nodes=[FilterNode(node_id="n0", filter_name="scale",
                              params={"w": 640})],
        )
        d = chain.to_dict()
        loaded = FilterChain.from_dict(d)
        assert loaded.name == "test"
        assert loaded.nodes[0].filter_name == "scale"
        assert loaded.nodes[0].params["w"] == 640

    def test_build_filter_chain_convenience(self):
        from opencut.core.filter_chain_builder import build_filter_chain
        result = build_filter_chain([
            {"node_id": "n0", "filter_name": "scale",
             "params": {"w": 1920, "h": 1080}},
        ])
        assert "scale" in result
        assert "1920" in result

    def test_validate_filter_graph_convenience(self):
        from opencut.core.filter_chain_builder import validate_filter_graph
        result = validate_filter_graph({
            "nodes": [
                {"node_id": "n0", "filter_name": "scale"},
            ],
        })
        assert result["valid"] is True

    def test_build_empty_chain_raises(self):
        from opencut.core.filter_chain_builder import FilterChain, build_filter_string
        with pytest.raises(ValueError, match="(?i)at least one"):
            build_filter_string(FilterChain())


# ============================================================================
# 4. WEBHOOK SYSTEM — Core Logic
# ============================================================================

class TestWebhookSystemCore:
    """Unit tests for opencut.core.webhook_system."""

    def test_register_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        wh = register_webhook("https://example.com/hook",
                              events=["job_complete"])
        assert wh.url == "https://example.com/hook"
        assert wh.events == ["job_complete"]
        assert wh.id

    def test_register_webhook_no_url_raises(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        with pytest.raises(ValueError, match="URL is required"):
            register_webhook("")

    def test_register_webhook_invalid_event_raises(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        with pytest.raises(ValueError, match="Invalid event"):
            register_webhook("https://example.com", events=["bogus_event"])

    def test_register_webhook_invalid_scheme_raises(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        with pytest.raises(ValueError, match="http:// or https://"):
            register_webhook("file:///tmp/out")

    def test_register_all_events_empty_list(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        wh = register_webhook("https://example.com/hook", events=[])
        assert wh.events == []

    def test_update_existing_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import register_webhook
        wh = register_webhook("https://old.com", events=["job_complete"])
        updated = register_webhook("https://new.com", webhook_id=wh.id)
        assert updated.id == wh.id
        assert updated.url == "https://new.com"

    def test_list_webhooks(self, tmp_opencut_dir):
        from opencut.core.webhook_system import list_webhooks, register_webhook
        register_webhook("https://a.com", events=["job_complete"])
        register_webhook("https://b.com", events=["error"])
        webhooks = list_webhooks()
        assert len(webhooks) >= 2

    def test_unregister_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import (
            list_webhooks,
            register_webhook,
            unregister_webhook,
        )
        wh = register_webhook("https://remove.me")
        assert unregister_webhook(wh.id) is True
        webhooks = list_webhooks()
        ids = [w["id"] for w in webhooks]
        assert wh.id not in ids

    def test_unregister_nonexistent(self, tmp_opencut_dir):
        from opencut.core.webhook_system import unregister_webhook
        assert unregister_webhook("fake_id") is False

    def test_get_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import get_webhook, register_webhook
        wh = register_webhook("https://get.me")
        found = get_webhook(wh.id)
        assert found is not None
        assert found.url == "https://get.me"

    def test_get_webhook_not_found(self, tmp_opencut_dir):
        from opencut.core.webhook_system import get_webhook
        assert get_webhook("nonexistent") is None

    def test_send_payload_bad_url(self):
        from opencut.core.webhook_system import _send_payload
        status, ok, error = _send_payload(
            "http://127.0.0.1:1/nope", "test", {}, timeout=1)
        assert ok is False

    def test_fire_event_no_targets(self, tmp_opencut_dir):
        from opencut.core.webhook_system import fire_event
        deliveries = fire_event("job_complete", {"status": "ok"})
        assert len(deliveries) == 0

    def test_test_webhook_not_found(self, tmp_opencut_dir):
        from opencut.core.webhook_system import test_webhook
        with pytest.raises(ValueError, match="not found"):
            test_webhook("fake_id")

    def test_delivery_log(self, tmp_opencut_dir):
        from opencut.core.webhook_system import (
            WebhookDelivery,
            _append_delivery,
            get_delivery_log,
        )
        d = WebhookDelivery(
            webhook_id="wh1", url="https://x.com",
            event_type="test", success=True, status_code=200)
        _append_delivery(d)
        log = get_delivery_log()
        assert len(log) >= 1
        assert log[-1]["webhook_id"] == "wh1"

    def test_clear_delivery_log(self, tmp_opencut_dir):
        from opencut.core.webhook_system import (
            WebhookDelivery,
            _append_delivery,
            clear_delivery_log,
            get_delivery_log,
        )
        _append_delivery(WebhookDelivery(webhook_id="wh1"))
        clear_delivery_log()
        log = get_delivery_log()
        assert len(log) == 0

    def test_delivery_log_filter_by_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import (
            WebhookDelivery,
            _append_delivery,
            get_delivery_log,
        )
        _append_delivery(WebhookDelivery(webhook_id="wh1"))
        _append_delivery(WebhookDelivery(webhook_id="wh2"))
        log = get_delivery_log(webhook_id="wh1")
        assert all(d["webhook_id"] == "wh1" for d in log)

    def test_webhook_config_dataclass(self):
        from opencut.core.webhook_system import WebhookConfig
        cfg = WebhookConfig(url="https://x.com", events=["error"])
        d = cfg.to_dict()
        assert d["url"] == "https://x.com"
        loaded = WebhookConfig.from_dict(d)
        assert loaded.url == "https://x.com"
        assert loaded.events == ["error"]

    def test_webhook_delivery_dataclass(self):
        from opencut.core.webhook_system import WebhookDelivery
        d = WebhookDelivery(
            webhook_id="wh1", url="https://x.com",
            event_type="test", success=True, status_code=200)
        result = d.to_dict()
        assert result["success"] is True
        assert result["status_code"] == 200

    # --- Legacy compatibility ---

    def test_legacy_send_webhook(self, tmp_opencut_dir):
        from opencut.core.webhook_system import send_webhook
        # Bad URL — should return False (not crash)
        result = send_webhook("http://127.0.0.1:1/nope", "test", {},
                              timeout=1)
        assert result is False

    def test_legacy_load_save_config(self, tmp_opencut_dir):
        from opencut.core.webhook_system import (
            load_webhook_config,
            save_webhook_config,
        )
        save_webhook_config([{"url": "https://x.com", "events": ["error"]}])
        configs = load_webhook_config()
        assert len(configs) >= 1
        assert configs[0]["url"] == "https://x.com"

    def test_legacy_save_config_invalid_scheme_raises(self, tmp_opencut_dir):
        from opencut.core.webhook_system import save_webhook_config
        with pytest.raises(ValueError, match="http:// or https://"):
            save_webhook_config([{"url": "file:///tmp/out"}])


# ============================================================================
# 5. BATCH SCRIPTING — Core Logic
# ============================================================================

class TestBatchScriptCore:
    """Unit tests for opencut.core.batch_script."""

    def test_batch_script_from_dict(self):
        from opencut.core.batch_script import BatchScript
        script = BatchScript.from_dict({
            "name": "Test",
            "operations": [
                {"operation": "silence", "file_pattern": "*.mp4",
                 "parameters": {"threshold": -30}},
            ],
        })
        assert script.name == "Test"
        assert len(script.operations) == 1
        assert script.operations[0].operation == "silence"

    def test_batch_script_to_dict(self):
        from opencut.core.batch_script import BatchOperation, BatchScript
        script = BatchScript(
            name="Test",
            operations=[BatchOperation(operation="silence")],
        )
        d = script.to_dict()
        assert d["name"] == "Test"
        assert len(d["operations"]) == 1

    def test_expand_output_pattern_default(self):
        from opencut.core.batch_script import _expand_output_pattern
        result = _expand_output_pattern("", "/videos/clip.mp4")
        assert result.endswith("_processed.mp4")
        assert "clip" in result

    def test_expand_output_pattern_template(self):
        from opencut.core.batch_script import _expand_output_pattern
        result = _expand_output_pattern(
            "${dir}/${basename}_clean${ext}",
            "/videos/clip.mp4",
        )
        assert result == "/videos/clip_clean.mp4"

    def test_expand_output_pattern_index(self):
        from opencut.core.batch_script import _expand_output_pattern
        result = _expand_output_pattern("out_${index}${ext}", "/a.mp4", 5)
        assert "0005" in result

    def test_expand_file_pattern(self, sample_files, tmp_dir):
        from opencut.core.batch_script import _expand_file_pattern
        pattern = os.path.join(tmp_dir, "*.mp4")
        matches = _expand_file_pattern(pattern)
        assert len(matches) == 3

    def test_expand_file_pattern_no_matches(self):
        from opencut.core.batch_script import _expand_file_pattern
        matches = _expand_file_pattern("/nonexistent_dir_12345/*.xyz")
        assert matches == []

    def test_expand_file_pattern_empty(self):
        from opencut.core.batch_script import _expand_file_pattern
        assert _expand_file_pattern("") == []

    def test_validate_script_dry_run(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, validate_script
        pattern = os.path.join(tmp_dir, "*.mp4")
        script = BatchScript.from_dict({
            "name": "DryRun",
            "operations": [
                {"operation": "silence", "file_pattern": pattern},
            ],
        })
        result = validate_script(script)
        assert result.dry_run is True
        assert result.total_files == 3
        assert len(result.file_results) == 3

    def test_validate_script_no_operations(self, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, validate_script
        script = BatchScript(name="Empty")
        result = validate_script(script)
        assert "no operations" in result.errors[0].lower()

    def test_validate_script_no_matches(self, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, validate_script
        script = BatchScript.from_dict({
            "name": "NoMatch",
            "operations": [
                {"operation": "test", "file_pattern": "/fake_12345/*.xyz"},
            ],
        })
        result = validate_script(script)
        assert any("matches no files" in e for e in result.errors)

    def test_validate_skip_existing(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, validate_script
        pattern = os.path.join(tmp_dir, "*.mp4")
        # Create a "processed" version of the first file
        processed = sample_files[0].replace(".mp4", "_processed.mp4")
        with open(processed, "w") as f:
            f.write("already done")

        script = BatchScript.from_dict({
            "name": "SkipTest",
            "operations": [
                {"operation": "test", "file_pattern": pattern,
                 "skip_existing": True},
            ],
        })
        result = validate_script(script)
        assert result.skipped >= 1

    def test_execute_script_no_executor(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, execute_script
        pattern = os.path.join(tmp_dir, "*.mp4")
        script = BatchScript.from_dict({
            "name": "NoExec",
            "operations": [
                {"operation": "test", "file_pattern": pattern},
            ],
        })
        result = execute_script(script)
        assert result.successful == 3
        assert result.failed == 0

    def test_execute_script_with_executor(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, execute_script
        pattern = os.path.join(tmp_dir, "*.mp4")
        results_collected = []

        def fake_executor(operation, filepath, params, output_path):
            results_collected.append(filepath)
            return {"status": "ok"}

        script = BatchScript.from_dict({
            "name": "WithExec",
            "operations": [
                {"operation": "test", "file_pattern": pattern},
            ],
        })
        result = execute_script(script, executor=fake_executor)
        assert result.successful == 3
        assert len(results_collected) == 3

    def test_execute_script_executor_error_continue(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, execute_script
        pattern = os.path.join(tmp_dir, "*.mp4")
        call_count = [0]

        def failing_executor(operation, filepath, params, output_path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("boom")
            return {"status": "ok"}

        script = BatchScript.from_dict({
            "name": "ErrContinue",
            "operations": [
                {"operation": "test", "file_pattern": pattern,
                 "continue_on_error": True},
            ],
        })
        result = execute_script(script, executor=failing_executor)
        assert result.failed == 1
        assert result.successful == 2

    def test_execute_script_executor_error_abort(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, execute_script
        pattern = os.path.join(tmp_dir, "*.mp4")

        def always_fail(operation, filepath, params, output_path):
            raise RuntimeError("fail")

        script = BatchScript.from_dict({
            "name": "ErrAbort",
            "operations": [
                {"operation": "test", "file_pattern": pattern,
                 "continue_on_error": False},
            ],
        })
        result = execute_script(script, executor=always_fail)
        # Should abort after first failure
        assert result.failed >= 1
        assert result.successful + result.failed < 3

    def test_batch_log_created(self, sample_files, tmp_dir, tmp_opencut_dir):
        from opencut.core.batch_script import BatchScript, execute_script, list_batch_logs
        pattern = os.path.join(tmp_dir, "*.mp4")
        script = BatchScript.from_dict({
            "name": "LogTest",
            "operations": [
                {"operation": "test", "file_pattern": pattern},
            ],
        })
        execute_script(script)
        logs = list_batch_logs()
        assert len(logs) >= 1
        assert logs[0]["script_name"] == "LogTest"

    def test_load_save_script(self, tmp_dir):
        from opencut.core.batch_script import BatchScript, load_script, save_script
        script = BatchScript(name="SaveLoad", operations=[])
        path = os.path.join(tmp_dir, "test_script.json")
        save_script(script, path)
        assert os.path.isfile(path)

        loaded = load_script(path)
        assert loaded.name == "SaveLoad"

    def test_load_script_not_found(self):
        from opencut.core.batch_script import load_script
        with pytest.raises(FileNotFoundError):
            load_script("/nonexistent_12345.json")


# ============================================================================
# ROUTE TESTS — Scripting Console
# ============================================================================

class TestScriptingRoutes:
    """Route integration tests for scripting console endpoints."""

    def test_execute_route(self, client, csrf_token):
        resp = client.post("/api/scripting/execute",
                           headers=csrf_headers(csrf_token),
                           json={"code": "print(1+2)"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "3" in data["output"]

    def test_execute_route_no_code(self, client, csrf_token):
        resp = client.post("/api/scripting/execute",
                           headers=csrf_headers(csrf_token),
                           json={"code": ""})
        assert resp.status_code == 400

    def test_execute_route_sandbox_block(self, client, csrf_token):
        resp = client.post("/api/scripting/execute",
                           headers=csrf_headers(csrf_token),
                           json={"code": "import os"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is False

    def test_execute_route_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/api/scripting/execute",
                           headers=csrf_headers(csrf_token),
                           json=["print(1+2)"])
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["code"] == "INVALID_INPUT"

    def test_execute_route_alias_works(self, client, csrf_token):
        resp = client.post("/api/dev/scripting/execute",
                           headers=csrf_headers(csrf_token),
                           json={"code": "print(7)"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "7" in data["output"]

    def test_history_route(self, client):
        resp = client.get("/api/scripting/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "history" in data

    def test_namespace_route(self, client):
        resp = client.get("/api/scripting/namespace")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "modules" in data
        assert "functions" in data


# ============================================================================
# ROUTE TESTS — Macro Recorder
# ============================================================================

class TestMacroRoutes:
    """Route integration tests for macro recorder endpoints."""

    def test_record_start_route(self, client, csrf_token):
        resp = client.post("/api/macro/record/start",
                           headers=csrf_headers(csrf_token),
                           json={"session_id": "route_test"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["recording"] is True

    def test_record_stop_no_session(self, client, csrf_token):
        resp = client.post("/api/macro/record/stop",
                           headers=csrf_headers(csrf_token),
                           json={"session_id": "nonexistent",
                                 "name": "Test"})
        assert resp.status_code == 400

    def test_macro_list_route(self, client):
        resp = client.get("/api/macro/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "macros" in data

    def test_macro_delete_not_found(self, client, csrf_token):
        resp = client.delete("/api/macro/does_not_exist",
                             headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_macro_play_alias_route_returns_job_id(self, client, csrf_token, tmp_opencut_dir):
        from opencut.core.macro_recorder import MacroRecording, MacroStep, save_macro

        save_macro(MacroRecording(
            name="AliasRoute",
            steps=[MacroStep(endpoint="/api/test", payload={"x": 1})],
        ))

        resp = client.post("/api/dev/macro/play",
                           headers=csrf_headers(csrf_token),
                           json={"name": "AliasRoute", "target_file": "/test.mp4"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data


# ============================================================================
# ROUTE TESTS — Filter Chain Builder
# ============================================================================

class TestFilterChainRoutes:
    """Route integration tests for filter chain builder endpoints."""

    def test_build_route(self, client, csrf_token):
        resp = client.post("/api/filter-chain/build",
                           headers=csrf_headers(csrf_token),
                           json={
                               "nodes": [
                                   {"node_id": "n0", "filter_name": "scale",
                                    "params": {"w": 1280, "h": 720}},
                               ],
                           })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is True
        assert "filter_complex" in data

    def test_build_route_empty_nodes(self, client, csrf_token):
        resp = client.post("/api/filter-chain/build",
                           headers=csrf_headers(csrf_token),
                           json={"nodes": []})
        assert resp.status_code == 400

    def test_build_route_invalid_chain(self, client, csrf_token):
        resp = client.post("/api/filter-chain/build",
                           headers=csrf_headers(csrf_token),
                           json={
                               "nodes": [
                                   {"node_id": "n0", "filter_name": ""},
                               ],
                           })
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS — Webhooks
# ============================================================================

class TestWebhookRoutes:
    """Route integration tests for webhook endpoints."""

    def test_register_route(self, client, csrf_token, tmp_opencut_dir):
        resp = client.post("/api/webhooks",
                           headers=csrf_headers(csrf_token),
                           json={"url": "https://test.com/hook",
                                 "events": ["job_complete"]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["webhook"]["url"] == "https://test.com/hook"

    def test_register_route_no_url(self, client, csrf_token):
        resp = client.post("/api/webhooks",
                           headers=csrf_headers(csrf_token),
                           json={"url": ""})
        assert resp.status_code == 400

    def test_list_route(self, client, tmp_opencut_dir):
        resp = client.get("/api/webhooks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "webhooks" in data

    def test_delete_route_not_found(self, client, csrf_token, tmp_opencut_dir):
        resp = client.delete("/api/webhooks/fake_id",
                             headers=csrf_headers(csrf_token))
        assert resp.status_code == 404

    def test_test_route_no_id(self, client, csrf_token):
        resp = client.post("/api/webhooks/test",
                           headers=csrf_headers(csrf_token),
                           json={"id": ""})
        assert resp.status_code == 400

    def test_register_route_rejects_non_object_json(self, client, csrf_token):
        resp = client.post("/api/webhooks",
                           headers=csrf_headers(csrf_token),
                           json=["https://test.com/hook"])
        assert resp.status_code == 400


# ============================================================================
# ROUTE TESTS — Batch Scripting
# ============================================================================

class TestBatchRoutes:
    """Route integration tests for batch scripting endpoints."""

    def test_validate_route(self, client, csrf_token, tmp_opencut_dir):
        resp = client.post("/api/batch/validate",
                           headers=csrf_headers(csrf_token),
                           json={
                               "name": "test",
                               "operations": [
                                   {"operation": "silence",
                                    "file_pattern": "/nonexistent/*.mp4"},
                               ],
                           })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["dry_run"] is True

    def test_validate_route_rejects_invalid_operations_type(self, client, csrf_token):
        resp = client.post("/api/batch/validate",
                           headers=csrf_headers(csrf_token),
                           json={"name": "test", "operations": "not-a-list"})
        assert resp.status_code == 400
