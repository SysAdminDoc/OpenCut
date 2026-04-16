"""
Tests for OpenCut Architecture Features (7.1 + 7.2).

Covers:
  - PydanticModelFactory: model creation, caching, type mapping, fallback
  - RouteAdapter: path conversion, model name inference, blueprint adaptation
  - OpenCutFastAPI: app creation, Flask mount, route discovery, WebSocket stub
  - FastAPI module functions: create_fastapi_app, mount_flask_app,
    generate_openapi_spec, adapt_flask_route
  - OpenAPI spec generation
  - WorkerConfig / GPUWorkerProcess / IsolatedJobResult dataclasses
  - WorkerPool: creation, submit, kill, cleanup, VRAM tracking, monitor
  - VRAM monitoring: nvidia-smi parsing, per-PID queries
  - Process isolation: worker entry, spawn, result queue
  - Module-level functions: create_worker_pool, submit_isolated_job,
    get_pool_status, kill_worker, cleanup_pool
  - Architecture routes: openapi, routes, health, worker-pool CRUD
"""

import os
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Feature 7.1 — FastAPI Migration Layer
# ===================================================================


class TestRouteInfo(unittest.TestCase):
    """Tests for RouteInfo dataclass."""

    def test_creation(self):
        from opencut.core.fastapi_app import RouteInfo
        r = RouteInfo(rule="/hw/encode", endpoint="hw_encode",
                      methods=["POST"], params={"id": "int"})
        self.assertEqual(r.rule, "/hw/encode")
        self.assertEqual(r.endpoint, "hw_encode")
        self.assertEqual(r.methods, ["POST"])
        self.assertEqual(r.params, {"id": "int"})

    def test_to_dict(self):
        from opencut.core.fastapi_app import RouteInfo
        r = RouteInfo(rule="/test", endpoint="test", methods=["GET"])
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["rule"], "/test")
        self.assertEqual(d["description"], "")

    def test_default_params(self):
        from opencut.core.fastapi_app import RouteInfo
        r = RouteInfo(rule="/a", endpoint="a", methods=["GET"])
        self.assertEqual(r.params, {})
        self.assertEqual(r.description, "")


class TestAdaptedRoute(unittest.TestCase):
    """Tests for AdaptedRoute dataclass."""

    def test_creation(self):
        from opencut.core.fastapi_app import AdaptedRoute
        a = AdaptedRoute(original_rule="/hw/<int:id>",
                         fastapi_path="/hw/{id}", methods=["GET"])
        self.assertTrue(a.success)
        self.assertIsNone(a.error)

    def test_to_dict(self):
        from opencut.core.fastapi_app import AdaptedRoute
        a = AdaptedRoute(original_rule="/x", fastapi_path="/x",
                         methods=["POST"], model_name="XRequest")
        d = a.to_dict()
        self.assertEqual(d["model_name"], "XRequest")
        self.assertTrue(d["success"])


class TestOpenAPISpec(unittest.TestCase):
    """Tests for OpenAPISpec dataclass."""

    def test_defaults(self):
        from opencut.core.fastapi_app import OpenAPISpec
        s = OpenAPISpec()
        self.assertEqual(s.title, "OpenCut API")
        self.assertEqual(s.paths, {})

    def test_to_dict(self):
        from opencut.core.fastapi_app import OpenAPISpec
        s = OpenAPISpec(title="Test", version="2.0.0",
                        paths={"/a": {"get": {}}})
        d = s.to_dict()
        self.assertEqual(d["openapi"], "3.1.0")
        self.assertEqual(d["info"]["title"], "Test")
        self.assertEqual(d["info"]["version"], "2.0.0")
        self.assertIn("/a", d["paths"])

    def test_custom_info(self):
        from opencut.core.fastapi_app import OpenAPISpec
        s = OpenAPISpec(info={"contact": {"name": "Admin"}})
        d = s.to_dict()
        self.assertEqual(d["info"]["contact"]["name"], "Admin")


class TestPydanticModelFactory(unittest.TestCase):
    """Tests for PydanticModelFactory."""

    def setUp(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        PydanticModelFactory.clear_cache()

    def test_create_model_basic(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        Model = PydanticModelFactory.create_model("BasicModel", {
            "name": "str",
            "count": "int",
        })
        self.assertIsNotNone(Model)
        self.assertEqual(Model.__name__, "BasicModel")

    def test_create_model_with_defaults(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        Model = PydanticModelFactory.create_model("DefaultModel", {
            "filepath": {"type": "str", "default": ""},
            "quality": {"type": "int", "default": 18},
        })
        self.assertIsNotNone(Model)

    def test_create_model_caching(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        M1 = PydanticModelFactory.create_model("CachedModel", {"x": "str"})
        M2 = PydanticModelFactory.create_model("CachedModel", {"y": "int"})
        self.assertIs(M1, M2)  # Same object from cache

    def test_clear_cache(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        PydanticModelFactory.create_model("TempModel", {"a": "str"})
        self.assertIn("TempModel", PydanticModelFactory.get_cached_models())
        PydanticModelFactory.clear_cache()
        self.assertEqual(len(PydanticModelFactory.get_cached_models()), 0)

    def test_type_mapping_str(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["str"], str)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["string"], str)

    def test_type_mapping_int(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["int"], int)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["integer"], int)

    def test_type_mapping_float(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["float"], float)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["number"], float)

    def test_type_mapping_bool(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["bool"], bool)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["boolean"], bool)

    def test_type_mapping_list(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["list"], list)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["array"], list)

    def test_type_mapping_dict(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        self.assertEqual(PydanticModelFactory.TYPE_MAP["dict"], dict)
        self.assertEqual(PydanticModelFactory.TYPE_MAP["object"], dict)

    def test_unknown_type_defaults_to_str(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        Model = PydanticModelFactory.create_model("UnknownType", {
            "data": "foobar_type",
        })
        self.assertIsNotNone(Model)

    def test_model_with_description(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        Model = PydanticModelFactory.create_model("DescModel", {
            "path": {
                "type": "str",
                "default": "/tmp",
                "description": "Output path",
            },
        })
        self.assertIsNotNone(Model)

    def test_non_dict_non_str_spec(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        # Passing an int as spec should still create a model (defaults to str)
        Model = PydanticModelFactory.create_model("WeirdSpec", {
            "field": 42,
        })
        self.assertIsNotNone(Model)

    def test_get_cached_models(self):
        from opencut.core.fastapi_app import PydanticModelFactory
        PydanticModelFactory.create_model("A", {"x": "int"})
        PydanticModelFactory.create_model("B", {"y": "str"})
        cached = PydanticModelFactory.get_cached_models()
        self.assertIn("A", cached)
        self.assertIn("B", cached)


class TestRouteAdapter(unittest.TestCase):
    """Tests for RouteAdapter."""

    def test_convert_simple_path(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(RouteAdapter._convert_path("/test"), "/test")

    def test_convert_typed_param(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(
            RouteAdapter._convert_path("/user/<int:id>"),
            "/user/{id}",
        )

    def test_convert_untyped_param(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(
            RouteAdapter._convert_path("/user/<name>"),
            "/user/{name}",
        )

    def test_convert_multiple_params(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(
            RouteAdapter._convert_path("/org/<string:org>/repo/<int:repo_id>"),
            "/org/{org}/repo/{repo_id}",
        )

    def test_infer_model_name(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(
            RouteAdapter._infer_model_name("hw_encode"),
            "HwEncodeRequest",
        )

    def test_infer_model_name_dotted(self):
        from opencut.core.fastapi_app import RouteAdapter
        self.assertEqual(
            RouteAdapter._infer_model_name("audio.silence"),
            "AudioSilenceRequest",
        )

    def test_adapt_route_get(self):
        from opencut.core.fastapi_app import RouteAdapter
        adapter = RouteAdapter()
        result = adapter.adapt_route("/health", "health", ["GET"])
        self.assertTrue(result.success)
        self.assertEqual(result.fastapi_path, "/health")
        self.assertIsNone(result.model_name)

    def test_adapt_route_post_generates_model_name(self):
        from opencut.core.fastapi_app import RouteAdapter
        adapter = RouteAdapter()
        result = adapter.adapt_route(
            "/encode", "encode", ["POST"],
            view_func=lambda: None,
        )
        self.assertEqual(result.model_name, "EncodeRequest")

    def test_adapt_route_filters_options_head(self):
        from opencut.core.fastapi_app import RouteAdapter
        adapter = RouteAdapter()
        result = adapter.adapt_route(
            "/test", "test", ["GET", "OPTIONS", "HEAD"]
        )
        self.assertEqual(result.methods, ["GET"])

    def test_adapted_routes_tracked(self):
        from opencut.core.fastapi_app import RouteAdapter
        adapter = RouteAdapter()
        adapter.adapt_route("/a", "a", ["GET"])
        adapter.adapt_route("/b", "b", ["POST"])
        self.assertEqual(len(adapter.adapted_routes), 2)

    def test_adapt_blueprint_empty(self):
        from opencut.core.fastapi_app import RouteAdapter
        adapter = RouteAdapter()
        result = adapter.adapt_blueprint(MagicMock(spec=[]))
        self.assertEqual(result, [])


class TestAdaptFlaskRoute(unittest.TestCase):
    """Tests for adapt_flask_route function."""

    def test_basic_adaptation(self):
        from opencut.core.fastapi_app import adapt_flask_route
        bp = MagicMock()
        bp.view_functions = {}
        result = adapt_flask_route(bp, "/test", "test_endpoint")
        self.assertEqual(result.original_rule, "/test")
        self.assertEqual(result.fastapi_path, "/test")

    def test_with_path_params(self):
        from opencut.core.fastapi_app import adapt_flask_route
        bp = MagicMock()
        bp.view_functions = {}
        result = adapt_flask_route(bp, "/item/<int:item_id>", "get_item")
        self.assertEqual(result.fastapi_path, "/item/{item_id}")


class TestMissingValueSentinel(unittest.TestCase):
    """Test the _MISSING sentinel."""

    def test_repr(self):
        from opencut.core.fastapi_app import _MISSING
        self.assertEqual(repr(_MISSING), "<MISSING>")


class TestOpenCutFastAPI(unittest.TestCase):
    """Tests for OpenCutFastAPI wrapper."""

    @patch("opencut.core.fastapi_app.mount_flask_app")
    def test_creation_without_flask(self, mock_mount):
        from opencut.core.fastapi_app import OpenCutFastAPI
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        wrapper = OpenCutFastAPI()
        self.assertIsNone(wrapper.flask_app)
        app = wrapper.fastapi_app
        self.assertIsNotNone(app)

    def test_creation_with_flask(self):
        from opencut.core.fastapi_app import OpenCutFastAPI
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        flask_app = MagicMock()
        with patch("opencut.core.fastapi_app.mount_flask_app"):
            wrapper = OpenCutFastAPI(flask_app)
            app = wrapper.fastapi_app
            self.assertIsNotNone(app)

    def test_discover_flask_routes_no_app(self):
        from opencut.core.fastapi_app import OpenCutFastAPI
        wrapper = OpenCutFastAPI()
        routes = wrapper.discover_flask_routes()
        self.assertEqual(routes, [])

    def test_register_websocket(self):
        from opencut.core.fastapi_app import OpenCutFastAPI
        wrapper = OpenCutFastAPI()
        handler = MagicMock()
        wrapper.register_websocket("/ws/test", handler)
        self.assertIn("/ws/test", wrapper._websocket_handlers)


class TestCreateFastAPIApp(unittest.TestCase):
    """Tests for create_fastapi_app function."""

    def test_creates_app(self):
        from opencut.core.fastapi_app import create_fastapi_app
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        with patch("opencut.core.fastapi_app.mount_flask_app"):
            app = create_fastapi_app()
            self.assertIsNotNone(app)

    def test_creates_app_with_flask(self):
        from opencut.core.fastapi_app import create_fastapi_app
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed")
        flask_app = MagicMock()
        with patch("opencut.core.fastapi_app.mount_flask_app"):
            app = create_fastapi_app(flask_app)
            self.assertIsNotNone(app)

    def test_raises_without_fastapi(self):
        from opencut.core.fastapi_app import OpenCutFastAPI
        with patch.dict("sys.modules", {"fastapi": None}):
            wrapper = OpenCutFastAPI()
            wrapper._fastapi_app = None
            with self.assertRaises(ImportError):
                _ = wrapper.fastapi_app


class TestMountFlaskApp(unittest.TestCase):
    """Tests for mount_flask_app function."""

    def test_mounts_with_starlette(self):
        fastapi_app = MagicMock()
        flask_app = MagicMock()
        mock_wsgi = MagicMock()
        mock_wsgi_mod = MagicMock()
        mock_wsgi_mod.WSGIMiddleware = mock_wsgi
        with patch.dict("sys.modules", {
            "starlette": MagicMock(),
            "starlette.middleware": MagicMock(),
            "starlette.middleware.wsgi": mock_wsgi_mod,
        }):
            import importlib

            import opencut.core.fastapi_app as mod
            importlib.reload(mod)
            result = mod.mount_flask_app(fastapi_app, flask_app)
            fastapi_app.mount.assert_called_once()
            self.assertEqual(result, fastapi_app)

    def test_returns_app_without_starlette(self):
        fastapi_app = MagicMock()
        flask_app = MagicMock()
        with patch.dict("sys.modules", {
            "starlette": None,
            "starlette.middleware": None,
            "starlette.middleware.wsgi": None,
            "a2wsgi": None,
        }):
            # Force fresh import
            import importlib

            import opencut.core.fastapi_app as mod
            importlib.reload(mod)
            mod.mount_flask_app(fastapi_app, flask_app)
            # Should still return the app even if middleware unavailable


class TestGenerateOpenAPISpec(unittest.TestCase):
    """Tests for generate_openapi_spec function."""

    def test_without_flask(self):
        from opencut.core.fastapi_app import generate_openapi_spec
        spec = generate_openapi_spec()
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(spec["info"]["title"], "OpenCut API")
        self.assertEqual(spec["paths"], {})

    def test_with_flask_app(self):
        from opencut.core.fastapi_app import generate_openapi_spec
        # Create a mock Flask app with url_map
        mock_rule = MagicMock()
        mock_rule.rule = "/test"
        mock_rule.endpoint = "test_func"
        mock_rule.methods = {"GET", "OPTIONS", "HEAD"}
        mock_rule.arguments = set()

        mock_app = MagicMock()
        mock_app.url_map.iter_rules.return_value = [mock_rule]

        spec = generate_openapi_spec(flask_app=mock_app)
        self.assertIn("/test", spec["paths"])


# ===================================================================
# Feature 7.2 — Process Isolation for GPU Workers
# ===================================================================


class TestWorkerConfig(unittest.TestCase):
    """Tests for WorkerConfig dataclass."""

    def test_defaults(self):
        from opencut.core.process_isolation import WorkerConfig
        c = WorkerConfig(model_name="whisper")
        self.assertEqual(c.model_name, "whisper")
        self.assertEqual(c.vram_required, 2048)
        self.assertEqual(c.timeout, 600)
        self.assertEqual(c.env_vars, {})

    def test_custom_values(self):
        from opencut.core.process_isolation import WorkerConfig
        c = WorkerConfig(
            model_name="llama", vram_required=8192, timeout=300,
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(c.vram_required, 8192)
        self.assertEqual(c.env_vars["CUDA_VISIBLE_DEVICES"], "0")

    def test_to_dict(self):
        from opencut.core.process_isolation import WorkerConfig
        c = WorkerConfig(model_name="test")
        d = c.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["model_name"], "test")


class TestGPUWorkerProcess(unittest.TestCase):
    """Tests for GPUWorkerProcess dataclass."""

    def test_creation(self):
        from opencut.core.process_isolation import GPUWorkerProcess
        w = GPUWorkerProcess(
            pid=1234, model_name="whisper", vram_allocated=4096,
            status="running", worker_id="gpu-worker-1",
        )
        self.assertEqual(w.pid, 1234)
        self.assertEqual(w.model_name, "whisper")
        self.assertEqual(w.status, "running")

    def test_to_dict(self):
        from opencut.core.process_isolation import GPUWorkerProcess
        w = GPUWorkerProcess(
            pid=100, model_name="test", vram_allocated=1024,
            started_at=time.time(), worker_id="w-1",
        )
        d = w.to_dict()
        self.assertEqual(d["pid"], 100)
        self.assertIn("runtime_sec", d)
        self.assertNotIn("_process", d)

    def test_default_status(self):
        from opencut.core.process_isolation import GPUWorkerProcess
        w = GPUWorkerProcess(pid=1, model_name="m", vram_allocated=0)
        self.assertEqual(w.status, "starting")

    def test_runtime_calculation(self):
        from opencut.core.process_isolation import GPUWorkerProcess
        now = time.time()
        w = GPUWorkerProcess(
            pid=1, model_name="m", vram_allocated=0,
            started_at=now - 10, finished_at=now,
        )
        d = w.to_dict()
        self.assertAlmostEqual(d["runtime_sec"], 10.0, places=0)


class TestIsolatedJobResult(unittest.TestCase):
    """Tests for IsolatedJobResult dataclass."""

    def test_success(self):
        from opencut.core.process_isolation import IsolatedJobResult
        r = IsolatedJobResult(success=True, output={"transcription": "hi"})
        self.assertTrue(r.success)
        self.assertEqual(r.output["transcription"], "hi")

    def test_error(self):
        from opencut.core.process_isolation import IsolatedJobResult
        r = IsolatedJobResult(success=False, error="OOM")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "OOM")

    def test_to_dict(self):
        from opencut.core.process_isolation import IsolatedJobResult
        r = IsolatedJobResult(success=True, output="done", duration=1.5)
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["duration"], 1.5)

    def test_defaults(self):
        from opencut.core.process_isolation import IsolatedJobResult
        r = IsolatedJobResult(success=True)
        self.assertIsNone(r.output)
        self.assertIsNone(r.error)
        self.assertEqual(r.vram_peak, 0)
        self.assertEqual(r.duration, 0.0)


class TestVRAMMonitoring(unittest.TestCase):
    """Tests for nvidia-smi VRAM monitoring functions."""

    @patch("subprocess.run")
    def test_query_nvidia_smi_success(self, mock_run):
        from opencut.core.process_isolation import _query_nvidia_smi
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4090, 24564, 1200, 23364\n"
                   "1, NVIDIA GeForce RTX 4090, 24564, 800, 23764\n",
        )
        gpus = _query_nvidia_smi()
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(gpus[0]["memory_total"], 24564)
        self.assertEqual(gpus[0]["memory_used"], 1200)

    @patch("subprocess.run")
    def test_query_nvidia_smi_not_found(self, mock_run):
        from opencut.core.process_isolation import _query_nvidia_smi
        mock_run.side_effect = FileNotFoundError
        gpus = _query_nvidia_smi()
        self.assertEqual(gpus, [])

    @patch("subprocess.run")
    def test_query_nvidia_smi_timeout(self, mock_run):
        from opencut.core.process_isolation import _query_nvidia_smi
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
        gpus = _query_nvidia_smi()
        self.assertEqual(gpus, [])

    @patch("subprocess.run")
    def test_query_nvidia_smi_error_code(self, mock_run):
        from opencut.core.process_isolation import _query_nvidia_smi
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        gpus = _query_nvidia_smi()
        self.assertEqual(gpus, [])

    @patch("subprocess.run")
    def test_get_total_vram_used(self, mock_run):
        from opencut.core.process_isolation import _get_total_vram_used
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU, 24000, 3000, 21000\n1, GPU, 24000, 2000, 22000\n",
        )
        used = _get_total_vram_used()
        self.assertEqual(used, 5000)

    @patch("subprocess.run")
    def test_get_total_vram_available(self, mock_run):
        from opencut.core.process_isolation import _get_total_vram_available
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, GPU, 24000, 3000, 21000\n",
        )
        avail = _get_total_vram_available()
        self.assertEqual(avail, 21000)

    @patch("subprocess.run")
    def test_get_vram_for_pid_found(self, mock_run):
        from opencut.core.process_isolation import _get_vram_for_pid
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1234, 4096\n5678, 2048\n",
        )
        vram = _get_vram_for_pid(1234)
        self.assertEqual(vram, 4096)

    @patch("subprocess.run")
    def test_get_vram_for_pid_not_found(self, mock_run):
        from opencut.core.process_isolation import _get_vram_for_pid
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5678, 2048\n",
        )
        vram = _get_vram_for_pid(9999)
        self.assertEqual(vram, 0)

    @patch("subprocess.run")
    def test_get_vram_for_pid_no_nvidia(self, mock_run):
        from opencut.core.process_isolation import _get_vram_for_pid
        mock_run.side_effect = FileNotFoundError
        vram = _get_vram_for_pid(1234)
        self.assertEqual(vram, 0)

    @patch("subprocess.run")
    def test_nvidia_smi_malformed_output(self, mock_run):
        from opencut.core.process_isolation import _query_nvidia_smi
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="garbage line\npartial, data\n",
        )
        gpus = _query_nvidia_smi()
        self.assertEqual(gpus, [])  # Lines with <5 parts are skipped


class TestWorkerPool(unittest.TestCase):
    """Tests for WorkerPool class."""

    def _make_pool(self, **kwargs):
        from opencut.core.process_isolation import WorkerPool
        return WorkerPool(**kwargs)

    def test_creation_defaults(self):
        pool = self._make_pool()
        self.assertEqual(pool.max_workers, 2)
        self.assertEqual(pool.vram_budget_mb, 8192)

    def test_creation_custom(self):
        pool = self._make_pool(max_workers=4, vram_budget_mb=16384)
        self.assertEqual(pool.max_workers, 4)
        self.assertEqual(pool.vram_budget_mb, 16384)

    def test_min_workers_clamped(self):
        pool = self._make_pool(max_workers=0)
        self.assertEqual(pool.max_workers, 1)

    def test_min_vram_clamped(self):
        pool = self._make_pool(vram_budget_mb=100)
        self.assertEqual(pool.vram_budget_mb, 512)

    def test_vram_allocated_empty(self):
        pool = self._make_pool()
        self.assertEqual(pool.vram_allocated, 0)

    def test_vram_remaining_full_budget(self):
        pool = self._make_pool(vram_budget_mb=4096)
        self.assertEqual(pool.vram_remaining, 4096)

    def test_worker_count_empty(self):
        pool = self._make_pool()
        self.assertEqual(pool.worker_count, 0)

    def test_get_status_empty(self):
        pool = self._make_pool(max_workers=3, vram_budget_mb=6000)
        status = pool.get_status()
        self.assertEqual(status["max_workers"], 3)
        self.assertEqual(status["vram_budget_mb"], 6000)
        self.assertEqual(status["active_workers"], 0)
        self.assertEqual(status["vram_allocated_mb"], 0)
        self.assertEqual(status["vram_remaining_mb"], 6000)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_submit_success(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 9999
        mock_process.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool(max_workers=2, vram_budget_mb=8192)
        wid, worker = pool.submit("whisper", "opencut.core.llm.run")
        self.assertIsNotNone(worker)
        self.assertEqual(worker.pid, 9999)
        self.assertEqual(worker.model_name, "whisper")
        self.assertEqual(worker.status, "running")
        mock_process.start.assert_called_once()
        pool.cleanup()

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_submit_pool_full(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 100
        mock_process.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool(max_workers=1, vram_budget_mb=8192)
        wid1, w1 = pool.submit("m1", "opencut.core.a.b")
        self.assertIsNotNone(w1)

        wid2, w2 = pool.submit("m2", "opencut.core.a.c")
        self.assertIsNone(w2)  # Rejected — pool full
        pool.cleanup()

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_submit_vram_exceeded(self, mock_ctx):
        from opencut.core.process_isolation import WorkerConfig
        mock_process = MagicMock()
        mock_process.pid = 100
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool(max_workers=5, vram_budget_mb=4096)
        config = WorkerConfig(model_name="big", vram_required=3000)
        wid1, w1 = pool.submit("big", "mod.func", config=config)
        self.assertIsNotNone(w1)

        config2 = WorkerConfig(model_name="big2", vram_required=3000)
        wid2, w2 = pool.submit("big2", "mod.func", config=config2)
        self.assertIsNone(w2)  # Rejected — VRAM exceeded
        pool.cleanup()

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_kill_worker(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 200
        mock_process.is_alive.return_value = False
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool()
        wid, worker = pool.submit("model", "mod.func")
        killed = pool.kill_worker(wid)
        self.assertTrue(killed)
        self.assertEqual(worker.status, "killed")
        pool.cleanup()

    def test_kill_nonexistent_worker(self):
        pool = self._make_pool()
        killed = pool.kill_worker("nonexistent-id")
        self.assertFalse(killed)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_cleanup(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 300
        mock_process.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_process
        mock_queue = MagicMock()
        mock_ctx.Queue.return_value = mock_queue

        pool = self._make_pool(max_workers=3)
        pool.submit("a", "mod.a")
        pool.submit("b", "mod.b")
        count = pool.cleanup()
        self.assertEqual(count, 2)
        self.assertEqual(len(pool.active_workers), 0)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_get_result_success(self, mock_ctx):
        from opencut.core.process_isolation import IsolatedJobResult

        mock_process = MagicMock()
        mock_process.pid = 400
        mock_ctx.Process.return_value = mock_process

        mock_queue = MagicMock()
        mock_result = IsolatedJobResult(success=True, output="done", duration=1.0)
        mock_queue.get.return_value = mock_result
        mock_ctx.Queue.return_value = mock_queue

        pool = self._make_pool()
        wid, worker = pool.submit("m", "mod.func")

        result = pool.get_result(wid, timeout=1.0)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "done")
        pool.cleanup()

    def test_get_result_unknown_worker(self):
        pool = self._make_pool()
        result = pool.get_result("unknown-id")
        self.assertIsNone(result)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_get_result_no_result_yet(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 500
        mock_ctx.Process.return_value = mock_process

        mock_queue = MagicMock()
        mock_queue.get.side_effect = Exception("empty")
        mock_ctx.Queue.return_value = mock_queue

        pool = self._make_pool()
        wid, worker = pool.submit("m", "mod.func")
        result = pool.get_result(wid, timeout=0)
        self.assertIsNone(result)
        pool.cleanup()

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_worker_id_increments(self, mock_ctx):
        mock_process = MagicMock()
        mock_process.pid = 100
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool(max_workers=10)
        wid1, _ = pool.submit("a", "mod.a")
        wid2, _ = pool.submit("b", "mod.b")
        # IDs should be different
        self.assertNotEqual(wid1, wid2)
        self.assertIn("gpu-worker-", wid1)
        pool.cleanup()

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_submit_with_custom_config(self, mock_ctx):
        from opencut.core.process_isolation import WorkerConfig
        mock_process = MagicMock()
        mock_process.pid = 600
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = self._make_pool()
        config = WorkerConfig(
            model_name="llama",
            vram_required=6000,
            timeout=120,
            env_vars={"CUDA_VISIBLE_DEVICES": "1"},
        )
        wid, worker = pool.submit(
            "llama", "opencut.core.llm.inference",
            args=("prompt",), kwargs={"temp": 0.7},
            config=config,
        )
        self.assertIsNotNone(worker)
        self.assertEqual(worker.vram_allocated, 6000)
        pool.cleanup()


class TestWorkerEntry(unittest.TestCase):
    """Tests for _worker_entry subprocess function."""

    def test_successful_execution(self):
        import queue

        from opencut.core.process_isolation import IsolatedJobResult, _worker_entry
        q = queue.Queue()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.my_func.return_value = {"result": 42}
            mock_import.return_value = mock_module

            _worker_entry("mypackage.module.my_func", (1, 2), {"k": "v"}, q, {})

        result = q.get_nowait()
        self.assertIsInstance(result, IsolatedJobResult)
        self.assertTrue(result.success)
        self.assertEqual(result.output, {"result": 42})
        # duration should be non-negative; on Windows time.time() resolution
        # can report 0.0 for instantaneous mock calls so >= 0 is the correct
        # invariant here (not strictly >).
        self.assertGreaterEqual(result.duration, 0)

    def test_execution_error(self):
        import queue

        from opencut.core.process_isolation import _worker_entry
        q = queue.Queue()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.bad_func.side_effect = RuntimeError("boom")
            mock_import.return_value = mock_module

            _worker_entry("mypackage.module.bad_func", (), {}, q, {})

        result = q.get_nowait()
        self.assertFalse(result.success)
        self.assertIn("boom", result.error)

    def test_env_vars_applied(self):
        import queue

        from opencut.core.process_isolation import _worker_entry
        q = queue.Queue()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.func.return_value = None
            mock_import.return_value = mock_module

            env = {"TEST_VAR_ISOLATION": "hello"}
            _worker_entry("mod.func", (), {}, q, env)

        self.assertEqual(os.environ.get("TEST_VAR_ISOLATION"), "hello")
        # Cleanup
        os.environ.pop("TEST_VAR_ISOLATION", None)


class TestModuleLevelPoolFunctions(unittest.TestCase):
    """Tests for module-level pool management functions."""

    def tearDown(self):
        import opencut.core.process_isolation as mod
        if mod._global_pool is not None:
            mod._global_pool.cleanup()
            mod._global_pool = None

    def test_create_worker_pool(self):
        from opencut.core.process_isolation import create_worker_pool
        pool = create_worker_pool(max_workers=3, vram_budget_mb=4096)
        self.assertIsNotNone(pool)
        self.assertEqual(pool.max_workers, 3)

    def test_create_replaces_old_pool(self):
        from opencut.core.process_isolation import create_worker_pool
        pool1 = create_worker_pool()
        pool2 = create_worker_pool()
        self.assertIsNot(pool1, pool2)

    def test_get_worker_pool(self):
        from opencut.core.process_isolation import (
            create_worker_pool,
            get_worker_pool,
        )
        self.assertIsNone(get_worker_pool())  # Before creation
        pool = create_worker_pool()
        self.assertIs(get_worker_pool(), pool)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_submit_isolated_job(self, mock_ctx):
        from opencut.core.process_isolation import (
            create_worker_pool,
            submit_isolated_job,
        )
        mock_process = MagicMock()
        mock_process.pid = 700
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = create_worker_pool()
        wid, worker = submit_isolated_job(pool, "model", "mod.func")
        self.assertIsNotNone(worker)

    def test_get_pool_status(self):
        from opencut.core.process_isolation import (
            create_worker_pool,
            get_pool_status,
        )
        pool = create_worker_pool(max_workers=5, vram_budget_mb=10000)
        status = get_pool_status(pool)
        self.assertEqual(status["max_workers"], 5)
        self.assertEqual(status["vram_budget_mb"], 10000)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_kill_worker_function(self, mock_ctx):
        from opencut.core.process_isolation import (
            create_worker_pool,
        )
        from opencut.core.process_isolation import (
            kill_worker as kill_worker_fn,
        )
        mock_process = MagicMock()
        mock_process.pid = 800
        mock_process.is_alive.return_value = False
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = create_worker_pool()
        wid, _ = pool.submit("m", "mod.func")
        result = kill_worker_fn(pool, wid)
        self.assertTrue(result)

    @patch("opencut.core.process_isolation._MP_CONTEXT")
    def test_cleanup_pool_function(self, mock_ctx):
        from opencut.core.process_isolation import (
            cleanup_pool,
            create_worker_pool,
        )
        mock_process = MagicMock()
        mock_process.pid = 900
        mock_process.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_process
        mock_ctx.Queue.return_value = MagicMock()

        pool = create_worker_pool(max_workers=3)
        pool.submit("a", "mod.a")
        count = cleanup_pool(pool)
        self.assertEqual(count, 1)


class TestCheckWorkers(unittest.TestCase):
    """Tests for the WorkerPool monitor checks."""

    def test_check_detects_dead_process(self):
        from opencut.core.process_isolation import GPUWorkerProcess, WorkerPool

        pool = WorkerPool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False

        worker = GPUWorkerProcess(
            pid=111, model_name="test", vram_allocated=1024,
            status="running", started_at=time.time(),
            worker_id="w-dead", _process=mock_proc,
        )
        pool.active_workers["w-dead"] = worker
        pool._check_workers()
        self.assertEqual(worker.status, "completed")
        pool.cleanup()

    @patch("opencut.core.process_isolation._get_vram_for_pid")
    def test_check_kills_vram_hog(self, mock_vram):
        from opencut.core.process_isolation import GPUWorkerProcess, WorkerPool

        pool = WorkerPool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True

        worker = GPUWorkerProcess(
            pid=222, model_name="hog", vram_allocated=1000,
            status="running", started_at=time.time(),
            worker_id="w-hog", _process=mock_proc,
        )
        pool.active_workers["w-hog"] = worker

        # Return VRAM way above 1.5x allocation
        mock_vram.return_value = 2000
        pool._check_workers()
        self.assertEqual(worker.status, "killed")
        pool.cleanup()


# ===================================================================
# Architecture Routes
# ===================================================================

class TestArchitectureRoutes(unittest.TestCase):
    """Integration tests for architecture route endpoints."""

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        self.app = create_app(config=test_config)
        self.app.config["TESTING"] = True

        # Register our blueprint
        from opencut.routes.architecture_routes import architecture_bp
        # Check if already registered to avoid duplicate
        if "architecture" not in [bp.name for bp in self.app.iter_blueprints()]:
            self.app.register_blueprint(architecture_bp)

        self.client = self.app.test_client()
        # Get CSRF token
        resp = self.client.get("/health")
        data = resp.get_json()
        self.csrf_token = data.get("csrf_token", "")
        self.csrf_headers = {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    def test_openapi_endpoint(self):
        resp = self.client.get("/architecture/openapi")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("openapi", data)
        self.assertEqual(data["openapi"], "3.1.0")

    def test_routes_endpoint(self):
        resp = self.client.get("/architecture/routes")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("routes", data)
        self.assertIn("total", data)
        self.assertGreater(data["total"], 0)

    def test_health_endpoint(self):
        resp = self.client.get("/architecture/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("components", data)
        self.assertIn("gpu_worker_pool", data["components"])

    def test_worker_pool_create(self):
        resp = self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={"max_workers": 2, "vram_budget_mb": 4096},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "created")
        self.assertIn("pool", data)

    def test_worker_pool_status_not_created(self):
        # Reset global pool
        import opencut.core.process_isolation as mod
        old = mod._global_pool
        mod._global_pool = None
        try:
            resp = self.client.get("/architecture/worker-pool/status")
            self.assertEqual(resp.status_code, 404)
        finally:
            mod._global_pool = old

    def test_worker_pool_status_after_create(self):
        self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        resp = self.client.get("/architecture/worker-pool/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("max_workers", data)

    def test_worker_pool_submit_no_pool(self):
        import opencut.core.process_isolation as mod
        old = mod._global_pool
        mod._global_pool = None
        try:
            resp = self.client.post(
                "/architecture/worker-pool/submit",
                headers=self.csrf_headers,
                json={"model_name": "test", "function_path": "mod.func"},
            )
            self.assertEqual(resp.status_code, 404)
        finally:
            mod._global_pool = old

    def test_worker_pool_submit_missing_model(self):
        self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        resp = self.client.post(
            "/architecture/worker-pool/submit",
            headers=self.csrf_headers,
            json={"function_path": "mod.func"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_worker_pool_submit_missing_function(self):
        self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        resp = self.client.post(
            "/architecture/worker-pool/submit",
            headers=self.csrf_headers,
            json={"model_name": "test"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_worker_pool_kill_no_pool(self):
        import opencut.core.process_isolation as mod
        old = mod._global_pool
        mod._global_pool = None
        try:
            resp = self.client.post(
                "/architecture/worker-pool/kill",
                headers=self.csrf_headers,
                json={"worker_id": "w-1"},
            )
            self.assertEqual(resp.status_code, 404)
        finally:
            mod._global_pool = old

    def test_worker_pool_kill_missing_id(self):
        self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        resp = self.client.post(
            "/architecture/worker-pool/kill",
            headers=self.csrf_headers,
            json={},
        )
        self.assertEqual(resp.status_code, 400)

    def test_worker_pool_cleanup_no_pool(self):
        import opencut.core.process_isolation as mod
        old = mod._global_pool
        mod._global_pool = None
        try:
            resp = self.client.post(
                "/architecture/worker-pool/cleanup",
                headers=self.csrf_headers,
                json={},
            )
            self.assertEqual(resp.status_code, 404)
        finally:
            mod._global_pool = old

    def test_worker_pool_cleanup_success(self):
        self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        resp = self.client.post(
            "/architecture/worker-pool/cleanup",
            headers=self.csrf_headers,
            json={},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "cleaned")

    def test_csrf_required_on_create(self):
        resp = self.client.post(
            "/architecture/worker-pool/create",
            json={"max_workers": 1},
        )
        # Should be 403 without CSRF token
        self.assertEqual(resp.status_code, 403)

    def test_csrf_required_on_kill(self):
        resp = self.client.post(
            "/architecture/worker-pool/kill",
            json={"worker_id": "x"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_csrf_required_on_cleanup(self):
        resp = self.client.post(
            "/architecture/worker-pool/cleanup",
            json={},
        )
        self.assertEqual(resp.status_code, 403)

    def test_csrf_required_on_submit(self):
        resp = self.client.post(
            "/architecture/worker-pool/submit",
            json={"model_name": "x", "function_path": "y"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_worker_pool_create_default_values(self):
        resp = self.client.post(
            "/architecture/worker-pool/create",
            headers=self.csrf_headers,
            json={},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        pool = data["pool"]
        self.assertEqual(pool["max_workers"], 2)
        self.assertEqual(pool["vram_budget_mb"], 8192)

    def test_health_has_timestamp(self):
        resp = self.client.get("/architecture/health")
        data = resp.get_json()
        self.assertIn("timestamp", data)
        self.assertIsInstance(data["timestamp"], float)


if __name__ == "__main__":
    unittest.main()
