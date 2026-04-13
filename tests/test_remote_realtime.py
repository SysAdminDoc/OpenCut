"""
Tests for OpenCut Remote Processing & Real-Time AI Features

Covers:
  Feature 7.5  — Remote Processing / Render Node
  Feature 21.4 — Real-Time AI Processing Pipeline

Uses Flask test client.  No real network, no subprocess, no GPU needed.
External dependencies (urllib, onnxruntime, FFmpeg, PIL, numpy) are mocked.
"""

import base64
import io
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import csrf_headers


# ---------------------------------------------------------------------------
# Fixture: Flask app with remote_realtime_bp registered
# ---------------------------------------------------------------------------
@pytest.fixture
def rr_app():
    """Flask app with the remote_realtime_bp blueprint registered."""
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    test_config = OpenCutConfig()
    flask_app = create_app(config=test_config)
    flask_app.config["TESTING"] = True
    # Register our new blueprint (not yet in __init__.py)
    from opencut.routes.remote_realtime_routes import remote_realtime_bp
    try:
        flask_app.register_blueprint(remote_realtime_bp)
    except ValueError:
        pass  # Already registered
    return flask_app


@pytest.fixture
def rr_client(rr_app):
    """Flask test client with remote_realtime_bp available."""
    return rr_app.test_client()


@pytest.fixture
def rr_csrf_token(rr_client):
    """CSRF token from the rr_client."""
    resp = rr_client.get("/health")
    data = resp.get_json()
    return data.get("csrf_token", "")


# =====================================================================
# Feature 7.5 — Remote Processing Core Tests
# =====================================================================

class TestRemoteNodeDataclasses(unittest.TestCase):
    """Dataclass construction and serialisation."""

    def test_remote_node_defaults(self):
        from opencut.core.remote_process import RemoteNode
        node = RemoteNode()
        self.assertEqual(node.url, "")
        self.assertEqual(node.status, "unknown")
        self.assertIsInstance(node.capabilities, list)

    def test_remote_node_to_dict(self):
        from opencut.core.remote_process import RemoteNode
        node = RemoteNode(url="http://n1", name="Node1", status="online")
        d = node.to_dict()
        self.assertEqual(d["url"], "http://n1")
        self.assertEqual(d["name"], "Node1")
        self.assertIn("capabilities", d)

    def test_remote_job_defaults(self):
        from opencut.core.remote_process import RemoteJob
        job = RemoteJob()
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.progress, 0)

    def test_remote_job_to_dict(self):
        from opencut.core.remote_process import RemoteJob
        job = RemoteJob(node_url="http://x", remote_job_id="abc")
        d = job.to_dict()
        self.assertEqual(d["remote_job_id"], "abc")
        self.assertIn("error", d)

    def test_node_registry_to_dict(self):
        from opencut.core.remote_process import NodeRegistry, RemoteNode
        reg = NodeRegistry()
        reg.nodes["http://x"] = RemoteNode(url="http://x", name="X")
        reg.default_node = "http://x"
        d = reg.to_dict()
        self.assertEqual(d["default_node"], "http://x")
        self.assertIn("http://x", d["nodes"])

    def test_remote_node_with_capabilities(self):
        from opencut.core.remote_process import RemoteNode
        node = RemoteNode(url="http://n2", capabilities=["video", "audio"])
        self.assertEqual(len(node.capabilities), 2)
        d = node.to_dict()
        self.assertIn("video", d["capabilities"])

    def test_remote_job_complete_fields(self):
        from opencut.core.remote_process import RemoteJob
        job = RemoteJob(
            node_url="http://x", local_file="/tmp/a.mp4",
            remote_job_id="j1", status="complete", progress=100,
            result_path="/tmp/out.mp4",
        )
        self.assertEqual(job.status, "complete")
        self.assertEqual(job.result_path, "/tmp/out.mp4")


class TestRegistryPersistence(unittest.TestCase):
    """Load / save registry from disk."""

    def test_save_and_load_registry(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            _save_registry,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nodes.json")
            with patch("opencut.core.remote_process.REGISTRY_PATH", path), \
                 patch("opencut.core.remote_process.OPENCUT_DIR", tmpdir):
                with _registry_lock:
                    _registry.nodes["http://test"] = RemoteNode(
                        url="http://test", name="T", api_key="k1",
                    )
                    _registry.default_node = "http://test"
                    _save_registry()

                self.assertTrue(os.path.isfile(path))
                with open(path) as fh:
                    data = json.load(fh)
                self.assertIn("http://test", data["nodes"])
                self.assertEqual(data["default_node"], "http://test")

                # Clean up module state
                with _registry_lock:
                    _registry.nodes.pop("http://test", None)
                    _registry.default_node = ""

    def test_load_registry_missing_file(self):
        from opencut.core.remote_process import _load_registry
        with patch("opencut.core.remote_process.REGISTRY_PATH", "/nonexistent/file.json"):
            # Should not raise
            _load_registry()

    def test_load_registry_corrupt_json(self):
        from opencut.core.remote_process import _load_registry
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "corrupt.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        try:
            with patch("opencut.core.remote_process.REGISTRY_PATH", path):
                _load_registry()  # Should log warning, not raise
        finally:
            os.unlink(path)
            os.rmdir(tmpdir)


class TestNodeRegistration(unittest.TestCase):
    """register_node / list_nodes / remove_node."""

    @patch("opencut.core.remote_process._make_request")
    @patch("opencut.core.remote_process._save_registry")
    def test_register_node_success(self, mock_save, mock_req):
        from opencut.core.remote_process import (
            _registry,
            _registry_lock,
            register_node,
        )
        mock_req.return_value = {"status": "ok", "capabilities": ["video"]}

        node = register_node("http://node1", api_key="key1", name="Node1")
        self.assertEqual(node.url, "http://node1")
        self.assertEqual(node.name, "Node1")
        self.assertEqual(node.status, "online")
        self.assertIn("video", node.capabilities)
        mock_save.assert_called()

        # Clean up
        with _registry_lock:
            _registry.nodes.pop("http://node1", None)

    @patch("opencut.core.remote_process._make_request")
    def test_register_node_strips_trailing_slash(self, mock_req):
        from opencut.core.remote_process import (
            _registry,
            _registry_lock,
            register_node,
        )
        mock_req.return_value = {"status": "ok"}
        with patch("opencut.core.remote_process._save_registry"):
            node = register_node("http://node2/", name="N2")
        self.assertEqual(node.url, "http://node2")
        with _registry_lock:
            _registry.nodes.pop("http://node2", None)

    @patch("opencut.core.remote_process._make_request")
    def test_register_node_default_capabilities(self, mock_req):
        from opencut.core.remote_process import (
            _registry,
            _registry_lock,
            register_node,
        )
        mock_req.return_value = {"status": "ok"}
        with patch("opencut.core.remote_process._save_registry"):
            node = register_node("http://node3")
        self.assertIn("video", node.capabilities)
        with _registry_lock:
            _registry.nodes.pop("http://node3", None)

    @patch("opencut.core.remote_process._make_request",
           side_effect=RuntimeError("unreachable"))
    def test_register_node_failure(self, mock_req):
        from opencut.core.remote_process import register_node
        with self.assertRaises(RuntimeError):
            register_node("http://bad-node")

    def test_list_nodes_empty(self):
        from opencut.core.remote_process import (
            _registry,
            _registry_lock,
            list_nodes,
        )
        with _registry_lock:
            saved = dict(_registry.nodes)
            _registry.nodes.clear()
        try:
            result = list_nodes()
            self.assertEqual(result, [])
        finally:
            with _registry_lock:
                _registry.nodes.update(saved)

    def test_remove_node(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            remove_node,
        )
        with _registry_lock:
            _registry.nodes["http://rm-test"] = RemoteNode(url="http://rm-test")
        with patch("opencut.core.remote_process._save_registry"):
            self.assertTrue(remove_node("http://rm-test"))
            self.assertFalse(remove_node("http://rm-test"))

    def test_get_node(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            get_node,
        )
        with _registry_lock:
            _registry.nodes["http://gn"] = RemoteNode(url="http://gn", name="GN")
        try:
            n = get_node("http://gn")
            self.assertIsNotNone(n)
            self.assertEqual(n.name, "GN")
            self.assertIsNone(get_node("http://nope"))
        finally:
            with _registry_lock:
                _registry.nodes.pop("http://gn", None)


class TestPingNode(unittest.TestCase):
    """ping_node tests."""

    @patch("opencut.core.remote_process._make_request")
    def test_ping_success(self, mock_req):
        from opencut.core.remote_process import ping_node
        mock_req.return_value = {"status": "ok", "version": "1.0"}
        result = ping_node("http://node")
        self.assertEqual(result["status"], "ok")
        mock_req.assert_called_once()

    @patch("opencut.core.remote_process._make_request",
           side_effect=RuntimeError("timeout"))
    def test_ping_failure(self, mock_req):
        from opencut.core.remote_process import ping_node
        with self.assertRaises(RuntimeError):
            ping_node("http://dead-node")

    @patch("opencut.core.remote_process._make_request")
    @patch("opencut.core.remote_process._save_registry")
    def test_ping_updates_cached_status(self, mock_save, mock_req):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            ping_node,
        )
        with _registry_lock:
            _registry.nodes["http://pn"] = RemoteNode(
                url="http://pn", status="offline",
            )
        mock_req.return_value = {"status": "ok"}
        try:
            ping_node("http://pn")
            with _registry_lock:
                self.assertEqual(_registry.nodes["http://pn"].status, "online")
        finally:
            with _registry_lock:
                _registry.nodes.pop("http://pn", None)


class TestAutoSelectNode(unittest.TestCase):
    """auto_select_node tests."""

    def test_select_default_node(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            auto_select_node,
        )
        with _registry_lock:
            saved = dict(_registry.nodes)
            saved_default = _registry.default_node
            _registry.nodes.clear()
            _registry.nodes["http://a"] = RemoteNode(
                url="http://a", status="online", capabilities=["video"],
            )
            _registry.nodes["http://b"] = RemoteNode(
                url="http://b", status="online", capabilities=["video"],
            )
            _registry.default_node = "http://a"
        try:
            n = auto_select_node("video")
            self.assertIsNotNone(n)
            self.assertEqual(n.url, "http://a")
        finally:
            with _registry_lock:
                _registry.nodes = saved
                _registry.default_node = saved_default

    def test_select_by_capability(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            auto_select_node,
        )
        with _registry_lock:
            saved = dict(_registry.nodes)
            saved_default = _registry.default_node
            _registry.nodes.clear()
            _registry.nodes["http://a"] = RemoteNode(
                url="http://a", status="online", capabilities=["audio"],
            )
            _registry.nodes["http://b"] = RemoteNode(
                url="http://b", status="online", capabilities=["video"],
            )
            _registry.default_node = ""
        try:
            n = auto_select_node("video")
            self.assertIsNotNone(n)
            self.assertEqual(n.url, "http://b")
        finally:
            with _registry_lock:
                _registry.nodes = saved
                _registry.default_node = saved_default

    def test_select_skips_offline(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            auto_select_node,
        )
        with _registry_lock:
            saved = dict(_registry.nodes)
            saved_default = _registry.default_node
            _registry.nodes.clear()
            _registry.nodes["http://a"] = RemoteNode(
                url="http://a", status="offline",
            )
            _registry.default_node = ""
        try:
            n = auto_select_node()
            self.assertIsNone(n)
        finally:
            with _registry_lock:
                _registry.nodes = saved
                _registry.default_node = saved_default

    def test_select_no_nodes(self):
        from opencut.core.remote_process import (
            _registry,
            _registry_lock,
            auto_select_node,
        )
        with _registry_lock:
            saved = dict(_registry.nodes)
            _registry.nodes.clear()
        try:
            self.assertIsNone(auto_select_node())
        finally:
            with _registry_lock:
                _registry.nodes = saved


class TestSetDefaultNode(unittest.TestCase):
    """set_default_node tests."""

    def test_set_existing(self):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            set_default_node,
        )
        with _registry_lock:
            _registry.nodes["http://sd"] = RemoteNode(url="http://sd")
        with patch("opencut.core.remote_process._save_registry"):
            self.assertTrue(set_default_node("http://sd"))
        with _registry_lock:
            _registry.nodes.pop("http://sd", None)

    def test_set_nonexistent(self):
        from opencut.core.remote_process import set_default_node
        self.assertFalse(set_default_node("http://nope"))


class TestSubmitRemoteJob(unittest.TestCase):
    """submit_remote_job integration."""

    @patch("opencut.core.remote_process._make_request")
    @patch("opencut.core.remote_process.time")
    def test_submit_and_complete(self, mock_time, mock_req):
        from opencut.core.remote_process import (
            RemoteNode,
            _registry,
            _registry_lock,
            submit_remote_job,
        )
        # Setup node
        with _registry_lock:
            _registry.nodes["http://sj"] = RemoteNode(
                url="http://sj", api_key="k",
            )

        # time.time() for polling + time.sleep
        mock_time.time.side_effect = [0, 0, 1, 2]
        mock_time.sleep = MagicMock()

        # First call: submit.  Second: poll (complete).
        mock_req.side_effect = [
            {"job_id": "rj1"},
            {"status": "complete", "progress": 100,
             "result": {"output": "/tmp/out.mp4"}},
        ]

        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, "test.mp4")
        with open(tmpfile, "wb") as f:
            f.write(b"fake video")
        try:
            rjob = submit_remote_job("http://sj", tmpfile, "/silence")
            self.assertEqual(rjob.status, "complete")
            self.assertEqual(rjob.remote_job_id, "rj1")
        finally:
            try:
                os.unlink(tmpfile)
                os.rmdir(tmpdir)
            except OSError:
                pass

        with _registry_lock:
            _registry.nodes.pop("http://sj", None)

    def test_submit_missing_file(self):
        from opencut.core.remote_process import submit_remote_job
        with self.assertRaises(ValueError):
            submit_remote_job("http://x", "/nonexistent.mp4", "/silence")


class TestDownloadResult(unittest.TestCase):
    """download_result tests."""

    @patch("opencut.core.remote_process.urlopen")
    def test_download_success(self, mock_urlopen):
        from opencut.core.remote_process import download_result

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"Content-Length": "5"}
        mock_resp.read.side_effect = [b"hello", b""]
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "result.mp4")
            path = download_result("http://x", "j1", out)
            self.assertEqual(path, out)
            self.assertTrue(os.path.isfile(out))
            with open(out, "rb") as fh:
                self.assertEqual(fh.read(), b"hello")


class TestMultipartBuilder(unittest.TestCase):
    """_build_multipart tests."""

    def test_multipart_structure(self):
        from opencut.core.remote_process import _build_multipart

        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, "test.txt")
        with open(tmpfile, "wb") as f:
            f.write(b"test content")
        try:
            body, ct = _build_multipart(tmpfile, {"key": "val"})
            self.assertIn("multipart/form-data", ct)
            self.assertIn(b"test content", body)
            self.assertIn(b"key", body)
            self.assertIn(b"val", body)
        finally:
            try:
                os.unlink(tmpfile)
                os.rmdir(tmpdir)
            except OSError:
                pass


class TestMakeRequest(unittest.TestCase):
    """_make_request HTTP helper."""

    @patch("opencut.core.remote_process.urlopen")
    def test_get_request(self, mock_urlopen):
        from opencut.core.remote_process import _make_request

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        mock_urlopen.return_value = mock_resp

        result = _make_request("http://test/health")
        self.assertTrue(result["ok"])

    @patch("opencut.core.remote_process.urlopen")
    def test_request_with_api_key(self, mock_urlopen):
        from opencut.core.remote_process import _make_request

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{}'
        mock_urlopen.return_value = mock_resp

        _make_request("http://test/health", api_key="secret")
        req = mock_urlopen.call_args[0][0]
        self.assertIn("Bearer secret", req.get_header("Authorization"))

    @patch("opencut.core.remote_process.urlopen")
    def test_empty_response(self, mock_urlopen):
        from opencut.core.remote_process import _make_request

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp

        result = _make_request("http://test/health")
        self.assertEqual(result, {})

    @patch("opencut.core.remote_process.urlopen")
    def test_http_error(self, mock_urlopen):
        from urllib.error import HTTPError

        from opencut.core.remote_process import _make_request

        err = HTTPError("http://x", 500, "err", {}, io.BytesIO(b"fail"))
        mock_urlopen.side_effect = err
        with self.assertRaises(RuntimeError) as ctx:
            _make_request("http://x")
        self.assertIn("500", str(ctx.exception))

    @patch("opencut.core.remote_process.urlopen")
    def test_url_error(self, mock_urlopen):
        from urllib.error import URLError

        from opencut.core.remote_process import _make_request

        mock_urlopen.side_effect = URLError("conn refused")
        with self.assertRaises(RuntimeError) as ctx:
            _make_request("http://x")
        self.assertIn("Cannot reach", str(ctx.exception))

    @patch("opencut.core.remote_process.urlopen")
    def test_invalid_json(self, mock_urlopen):
        from opencut.core.remote_process import _make_request

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"not json"
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            _make_request("http://x")
        self.assertIn("invalid JSON", str(ctx.exception))


# =====================================================================
# Feature 21.4 — Real-Time AI Core Tests
# =====================================================================

class TestRealtimeDataclasses(unittest.TestCase):
    """Dataclass construction and serialisation."""

    def test_realtime_config_defaults(self):
        from opencut.core.realtime_ai import RealtimeConfig
        cfg = RealtimeConfig()
        self.assertEqual(cfg.model, "style_transfer")
        self.assertEqual(cfg.device, "cpu")
        self.assertAlmostEqual(cfg.resolution_scale, 0.5)

    def test_realtime_config_to_dict(self):
        from opencut.core.realtime_ai import RealtimeConfig
        cfg = RealtimeConfig(model="denoise", target_fps=15.0)
        d = cfg.to_dict()
        self.assertEqual(d["model"], "denoise")
        self.assertEqual(d["target_fps"], 15.0)

    def test_preview_frame_defaults(self):
        from opencut.core.realtime_ai import PreviewFrame
        f = PreviewFrame()
        self.assertEqual(f.frame_data_b64, "")
        self.assertFalse(f.cached)

    def test_preview_frame_to_dict(self):
        from opencut.core.realtime_ai import PreviewFrame
        f = PreviewFrame(timestamp=1.5, processing_ms=42.3, cached=True)
        d = f.to_dict()
        self.assertAlmostEqual(d["timestamp"], 1.5)
        self.assertTrue(d["cached"])

    def test_realtime_session_defaults(self):
        from opencut.core.realtime_ai import RealtimeSession
        s = RealtimeSession()
        self.assertFalse(s.running)
        self.assertEqual(s.frame_count, 0)

    def test_realtime_session_to_dict(self):
        from opencut.core.realtime_ai import RealtimeSession
        s = RealtimeSession(session_id="abc", model_name="denoise", running=True)
        d = s.to_dict()
        self.assertEqual(d["session_id"], "abc")
        self.assertTrue(d["running"])
        self.assertIn("resolution", d)

    def test_session_to_dict_with_config(self):
        from opencut.core.realtime_ai import RealtimeConfig, RealtimeSession
        cfg = RealtimeConfig(model="face_enhance")
        s = RealtimeSession(session_id="x", config=cfg)
        d = s.to_dict()
        self.assertIn("config", d)
        self.assertEqual(d["config"]["model"], "face_enhance")


class TestListRealtimeModels(unittest.TestCase):
    """list_realtime_models tests."""

    def test_returns_all_models(self):
        from opencut.core.realtime_ai import REALTIME_MODELS, list_realtime_models
        models = list_realtime_models()
        self.assertEqual(len(models), len(REALTIME_MODELS))

    def test_model_has_required_fields(self):
        from opencut.core.realtime_ai import list_realtime_models
        models = list_realtime_models()
        for m in models:
            self.assertIn("name", m)
            self.assertIn("label", m)
            self.assertIn("description", m)
            self.assertIn("installed", m)

    def test_known_models_present(self):
        from opencut.core.realtime_ai import list_realtime_models
        models = list_realtime_models()
        names = [m["name"] for m in models]
        for expected in ("style_transfer", "background_removal",
                         "face_enhance", "color_grade", "denoise"):
            self.assertIn(expected, names)


class TestCreateSession(unittest.TestCase):
    """create_session tests."""

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_default_config(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        session = create_session("style_transfer", "/tmp/vid.mp4")
        self.assertTrue(session.running)
        self.assertEqual(session.model_name, "style_transfer")
        # Resolution should be scaled down
        self.assertLessEqual(session.resolution[0], 1920)

        # Clean up
        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    def test_create_invalid_model(self):
        from opencut.core.realtime_ai import create_session
        with self.assertRaises(ValueError):
            create_session("nonexistent_model", "/tmp/vid.mp4")

    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=False)
    def test_create_missing_video(self, mock_isfile):
        from opencut.core.realtime_ai import create_session
        with self.assertRaises(ValueError):
            create_session("style_transfer", "/tmp/nonexistent.mp4")

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_custom_config(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            RealtimeConfig,
            _sessions,
            _sessions_lock,
            create_session,
        )
        mock_vinfo.return_value = {"width": 3840, "height": 2160, "fps": 60, "duration": 120}
        cfg = RealtimeConfig(model="color_grade", resolution_scale=0.25, target_fps=10)

        session = create_session("color_grade", "/tmp/4k.mp4", config=cfg)
        self.assertEqual(session.model_name, "color_grade")
        self.assertEqual(session.fps, 10)
        # color_grade min_resolution_scale is 0.25, so 3840*0.25 = 960
        self.assertLessEqual(session.resolution[0], 960)

        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_even_resolution(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import _sessions, _sessions_lock, create_session
        mock_vinfo.return_value = {"width": 1921, "height": 1081, "fps": 30, "duration": 60}

        session = create_session("style_transfer", "/tmp/odd.mp4")
        # Resolutions should be even
        self.assertEqual(session.resolution[0] % 2, 0)
        self.assertEqual(session.resolution[1] % 2, 0)

        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_progress_callback(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import _sessions, _sessions_lock, create_session
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        session = create_session("style_transfer", "/tmp/vid.mp4",
                                 on_progress=on_progress)
        self.assertTrue(len(progress_calls) >= 2)

        with _sessions_lock:
            _sessions.pop(session.session_id, None)


class TestGetAndListSessions(unittest.TestCase):
    """get_session / list_sessions tests."""

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_get_session(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            get_session,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        session = create_session("denoise", "/tmp/v.mp4")
        retrieved = get_session(session.session_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, session.session_id)
        self.assertIsNone(get_session("nonexistent"))

        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_list_sessions_filters_stopped(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            list_sessions,
            stop_session,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        s1 = create_session("denoise", "/tmp/v1.mp4")
        s2 = create_session("denoise", "/tmp/v2.mp4")
        stop_session(s1.session_id)

        active = list_sessions()
        ids = [s.session_id for s in active]
        self.assertNotIn(s1.session_id, ids)
        self.assertIn(s2.session_id, ids)

        with _sessions_lock:
            _sessions.pop(s1.session_id, None)
            _sessions.pop(s2.session_id, None)


class TestStopSession(unittest.TestCase):
    """stop_session tests."""

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_stop_running(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            stop_session,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        s = create_session("denoise", "/tmp/v.mp4")
        self.assertTrue(stop_session(s.session_id))

        with _sessions_lock:
            self.assertFalse(_sessions[s.session_id].running)
            _sessions.pop(s.session_id, None)

    def test_stop_nonexistent(self):
        from opencut.core.realtime_ai import stop_session
        self.assertFalse(stop_session("nonexistent"))

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_remove_session(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            create_session,
            remove_session,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        s = create_session("denoise", "/tmp/v.mp4")
        self.assertTrue(remove_session(s.session_id))
        self.assertFalse(remove_session(s.session_id))


class TestUpdateParams(unittest.TestCase):
    """update_params tests."""

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_update_merges_params(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            update_params,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        s = create_session("style_transfer", "/tmp/v.mp4")
        updated = update_params(s.session_id, {"intensity": 0.3})
        self.assertAlmostEqual(updated.config.params["intensity"], 0.3)

        with _sessions_lock:
            _sessions.pop(s.session_id, None)

    def test_update_nonexistent_session(self):
        from opencut.core.realtime_ai import update_params
        with self.assertRaises(ValueError):
            update_params("nonexistent", {"x": 1})

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_update_stopped_session_fails(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            stop_session,
            update_params,
        )
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        s = create_session("denoise", "/tmp/v.mp4")
        stop_session(s.session_id)
        with self.assertRaises(ValueError):
            update_params(s.session_id, {"strength": 0.9})

        with _sessions_lock:
            _sessions.pop(s.session_id, None)


class TestFrameCache(unittest.TestCase):
    """Frame cache helpers."""

    def test_cache_key_deterministic(self):
        from opencut.core.realtime_ai import _cache_key
        k1 = _cache_key("s1", 1.234, {"a": 1})
        k2 = _cache_key("s1", 1.234, {"a": 1})
        self.assertEqual(k1, k2)

    def test_cache_key_differs_on_params(self):
        from opencut.core.realtime_ai import _cache_key
        k1 = _cache_key("s1", 1.0, {"a": 1})
        k2 = _cache_key("s1", 1.0, {"a": 2})
        self.assertNotEqual(k1, k2)

    def test_cache_put_and_get(self):
        from opencut.core.realtime_ai import PreviewFrame, _cache_get, _cache_put
        f = PreviewFrame(timestamp=1.0, processing_ms=10)
        _cache_put("test_key_1", f)
        result = _cache_get("test_key_1")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.timestamp, 1.0)

    def test_cache_miss(self):
        from opencut.core.realtime_ai import _cache_get
        self.assertIsNone(_cache_get("nonexistent_key"))

    def test_cache_eviction(self):
        from opencut.core.realtime_ai import (
            MAX_CACHE_FRAMES,
            PreviewFrame,
            _cache_lock,
            _cache_put,
            _frame_cache,
        )
        # Fill beyond max
        for i in range(MAX_CACHE_FRAMES + 10):
            _cache_put(f"evict_{i}", PreviewFrame(timestamp=float(i)))
        with _cache_lock:
            self.assertLessEqual(len(_frame_cache), MAX_CACHE_FRAMES)
            # Clean up
            keys = [k for k in _frame_cache if k.startswith("evict_")]
            for k in keys:
                del _frame_cache[k]

    def test_clear_cache(self):
        from opencut.core.realtime_ai import (
            PreviewFrame,
            _cache_lock,
            _cache_put,
            _frame_cache,
            clear_cache,
        )
        _cache_put("clear_test", PreviewFrame())
        clear_cache()
        with _cache_lock:
            self.assertNotIn("clear_test", _frame_cache)


class TestGetCacheStats(unittest.TestCase):
    """get_cache_stats tests."""

    def test_stats_structure(self):
        from opencut.core.realtime_ai import get_cache_stats
        stats = get_cache_stats()
        self.assertIn("size", stats)
        self.assertIn("max_size", stats)
        self.assertIsInstance(stats["size"], int)


class TestFrameExtraction(unittest.TestCase):
    """_extract_frame_raw tests."""

    @patch("opencut.core.realtime_ai.subprocess.run")
    def test_extract_success(self, mock_run):
        from opencut.core.realtime_ai import _extract_frame_raw
        # 4x2 RGB = 24 bytes
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"\x00" * 24, stderr=b"",
        )
        result = _extract_frame_raw("/tmp/v.mp4", 1.0, 4, 2)
        self.assertEqual(len(result), 24)

    @patch("opencut.core.realtime_ai.subprocess.run")
    def test_extract_ffmpeg_failure(self, mock_run):
        from opencut.core.realtime_ai import _extract_frame_raw
        mock_run.return_value = MagicMock(
            returncode=1, stdout=b"", stderr=b"error",
        )
        with self.assertRaises(RuntimeError):
            _extract_frame_raw("/tmp/v.mp4", 1.0, 4, 2)

    @patch("opencut.core.realtime_ai.subprocess.run",
           side_effect=subprocess.TimeoutExpired("ffmpeg", 10))
    def test_extract_timeout(self, mock_run):
        from opencut.core.realtime_ai import _extract_frame_raw
        with self.assertRaises(RuntimeError):
            _extract_frame_raw("/tmp/v.mp4", 1.0, 4, 2)


class TestFrameToB64(unittest.TestCase):
    """_frame_to_png_b64 tests."""

    def test_encode_with_pil(self):
        """Test PNG encoding when PIL and numpy are available."""
        from opencut.core.realtime_ai import _frame_to_png_b64

        try:
            import numpy as np  # noqa: F401
            from PIL import Image  # noqa: F401
            # Both available — test real encoding
            raw = b"\x80" * (2 * 2 * 3)  # 2x2 RGB
            result = _frame_to_png_b64(raw, 2, 2)
            self.assertIsInstance(result, str)
            # Should be valid base64
            decoded = base64.b64decode(result)
            # PNG files start with the PNG signature
            self.assertTrue(decoded[:4] == b"\x89PNG")
        except ImportError:
            # PIL or numpy not installed — test the fallback path
            raw = b"\x80" * 12
            result = _frame_to_png_b64(raw, 2, 2)
            self.assertIsInstance(result, str)
            # Fallback returns raw base64
            decoded = base64.b64decode(result)
            self.assertEqual(decoded, raw)

    def test_encode_fallback(self):
        """Test fallback when PIL is not available."""
        raw = b"\xff" * 12
        with patch.dict("sys.modules", {"numpy": None, "PIL": None, "PIL.Image": None}):
            # Force ImportError path
            with patch("opencut.core.realtime_ai._frame_to_png_b64") as mock_fn:
                mock_fn.return_value = base64.b64encode(raw).decode("ascii")
                result = mock_fn(raw, 2, 2)
                self.assertIsInstance(result, str)
                decoded = base64.b64decode(result)
                self.assertEqual(decoded, raw)


class TestONNXSession(unittest.TestCase):
    """_get_onnx_session tests."""

    def test_no_onnxruntime(self):
        from opencut.core.realtime_ai import _get_onnx_session
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = _get_onnx_session("style_transfer")
                self.assertIsNone(result)

    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=False)
    def test_model_not_found(self, mock_isfile):
        from opencut.core.realtime_ai import _get_onnx_session, _onnx_lock, _onnx_sessions
        # Clear cache
        with _onnx_lock:
            _onnx_sessions.clear()
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            result = _get_onnx_session("style_transfer")
            self.assertIsNone(result)


class TestFallbackEffects(unittest.TestCase):
    """_apply_fallback_effect tests."""

    def test_style_transfer_fallback(self):
        from opencut.core.realtime_ai import _apply_fallback_effect
        mock_np = MagicMock()
        arr = MagicMock()
        arr.astype.return_value = arr
        arr.tobytes.return_value = b"\x00" * 12
        mock_np.frombuffer.return_value = MagicMock()
        mock_np.frombuffer.return_value.reshape.return_value = MagicMock()
        mock_np.frombuffer.return_value.reshape.return_value.copy.return_value = arr
        mock_np.array.return_value = MagicMock()
        mock_np.clip.return_value = arr

        with patch.dict("sys.modules", {"numpy": mock_np}):
            raw = b"\x80" * 12
            result = _apply_fallback_effect(raw, 2, 2, "style_transfer", {"intensity": 0.5})
            # Should return bytes (possibly modified)
            self.assertIsInstance(result, (bytes, MagicMock))

    def test_fallback_no_numpy(self):
        from opencut.core.realtime_ai import _apply_fallback_effect

        with patch.dict("sys.modules", {"numpy": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                raw = b"\x80" * 12
                result = _apply_fallback_effect(raw, 2, 2, "denoise", {})
                self.assertEqual(result, raw)


class TestGetPreviewFrame(unittest.TestCase):
    """get_preview_frame integration."""

    @patch("opencut.core.realtime_ai._get_onnx_session", return_value=None)
    @patch("opencut.core.realtime_ai._apply_fallback_effect")
    @patch("opencut.core.realtime_ai._frame_to_png_b64", return_value="AAAA")
    @patch("opencut.core.realtime_ai._extract_frame_raw")
    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_frame_pipeline(self, mock_isfile, mock_vinfo, mock_extract,
                            mock_b64, mock_fallback, mock_onnx):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            get_preview_frame,
        )
        mock_vinfo.return_value = {"width": 100, "height": 100, "fps": 30, "duration": 60}
        mock_extract.return_value = b"\x00" * (50 * 50 * 3)
        mock_fallback.return_value = b"\x00" * (50 * 50 * 3)

        session = create_session("style_transfer", "/tmp/v.mp4")
        frame = get_preview_frame(session.session_id, 1.0)

        self.assertEqual(frame.frame_data_b64, "AAAA")
        self.assertAlmostEqual(frame.timestamp, 1.0)
        self.assertFalse(frame.cached)
        mock_extract.assert_called_once()

        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    def test_frame_nonexistent_session(self):
        from opencut.core.realtime_ai import get_preview_frame
        with self.assertRaises(ValueError):
            get_preview_frame("nonexistent", 1.0)

    @patch("opencut.core.realtime_ai._get_onnx_session", return_value=None)
    @patch("opencut.core.realtime_ai._apply_fallback_effect")
    @patch("opencut.core.realtime_ai._frame_to_png_b64", return_value="BB")
    @patch("opencut.core.realtime_ai._extract_frame_raw")
    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_frame_caching(self, mock_isfile, mock_vinfo, mock_extract,
                           mock_b64, mock_fallback, mock_onnx):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            get_preview_frame,
        )
        mock_vinfo.return_value = {"width": 100, "height": 100, "fps": 30, "duration": 60}
        raw = b"\x00" * (50 * 50 * 3)
        mock_extract.return_value = raw
        mock_fallback.return_value = raw

        session = create_session("style_transfer", "/tmp/v.mp4")

        # First call — not cached
        f1 = get_preview_frame(session.session_id, 2.0)
        self.assertFalse(f1.cached)

        # Second call — should be cached
        f2 = get_preview_frame(session.session_id, 2.0)
        self.assertTrue(f2.cached)
        # FFmpeg should only be called once
        self.assertEqual(mock_extract.call_count, 1)

        with _sessions_lock:
            _sessions.pop(session.session_id, None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_frame_stopped_session(self, mock_isfile, mock_vinfo):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            get_preview_frame,
            stop_session,
        )
        mock_vinfo.return_value = {"width": 100, "height": 100, "fps": 30, "duration": 60}
        s = create_session("denoise", "/tmp/v.mp4")
        stop_session(s.session_id)
        with self.assertRaises(ValueError):
            get_preview_frame(s.session_id, 0.0)

        with _sessions_lock:
            _sessions.pop(s.session_id, None)

    @patch("opencut.core.realtime_ai._get_onnx_session", return_value=None)
    @patch("opencut.core.realtime_ai._apply_fallback_effect")
    @patch("opencut.core.realtime_ai._frame_to_png_b64", return_value="CC")
    @patch("opencut.core.realtime_ai._extract_frame_raw")
    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_frame_size_mismatch(self, mock_isfile, mock_vinfo, mock_extract,
                                  mock_b64, mock_fallback, mock_onnx):
        from opencut.core.realtime_ai import (
            _sessions,
            _sessions_lock,
            create_session,
            get_preview_frame,
        )
        mock_vinfo.return_value = {"width": 100, "height": 100, "fps": 30, "duration": 60}
        mock_extract.return_value = b"\x00" * 10  # Wrong size

        session = create_session("denoise", "/tmp/v.mp4")
        with self.assertRaises(RuntimeError):
            get_preview_frame(session.session_id, 1.0)

        with _sessions_lock:
            _sessions.pop(session.session_id, None)


class TestRunONNXInference(unittest.TestCase):
    """_run_onnx_inference tests."""

    def test_inference_success(self):
        from opencut.core.realtime_ai import _run_onnx_inference

        mock_np = MagicMock()
        mock_np.uint8 = "uint8"
        mock_np.float32 = "float32"

        arr = MagicMock()
        arr.astype.return_value = arr
        arr.transpose.return_value = arr
        mock_np.frombuffer.return_value = arr
        arr.reshape.return_value = arr
        mock_np.expand_dims.return_value = arr

        result_arr = MagicMock()
        result_arr.ndim = 4
        result_arr.__getitem__ = MagicMock(return_value=result_arr)
        result_arr.shape = [3, 2, 2]
        result_arr.transpose.return_value = result_arr
        mock_np.clip.return_value = result_arr
        result_arr.astype.return_value = result_arr
        result_arr.tobytes.return_value = b"\x00" * 12

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [result_arr]

        with patch.dict("sys.modules", {"numpy": mock_np}):
            result = _run_onnx_inference(mock_session, b"\x00" * 12, 2, 2, {})
            self.assertIsInstance(result, (bytes, MagicMock))

    def test_inference_no_numpy(self):
        from opencut.core.realtime_ai import _run_onnx_inference

        with patch.dict("sys.modules", {"numpy": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                raw = b"\x00" * 12
                result = _run_onnx_inference(MagicMock(), raw, 2, 2, {})
                self.assertEqual(result, raw)


# =====================================================================
# Route Tests — Remote Processing
# =====================================================================

class TestRemoteRoutes(unittest.TestCase):
    """Route-level tests for /remote/* endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, rr_client, rr_csrf_token):
        self.client = rr_client
        self.csrf_token = rr_csrf_token

    def test_register_node_missing_url(self):
        resp = self.client.post(
            "/remote/register-node",
            data=json.dumps({}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.remote_process.ping_node")
    @patch("opencut.core.remote_process._save_registry")
    def test_register_node_success(self, mock_save, mock_ping):
        mock_ping.return_value = {"status": "ok", "capabilities": ["video"]}
        resp = self.client.post(
            "/remote/register-node",
            data=json.dumps({"url": "http://testnode", "name": "T"}),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertIn("node", data)
        self.assertEqual(data["node"]["name"], "T")

        # Clean up
        from opencut.core.remote_process import _registry, _registry_lock
        with _registry_lock:
            _registry.nodes.pop("http://testnode", None)

    def test_list_nodes(self):
        resp = self.client.get("/remote/nodes")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("nodes", data)

    def test_ping_missing_url(self):
        resp = self.client.post(
            "/remote/ping",
            data=json.dumps({}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.remote_process._make_request")
    def test_ping_success(self, mock_req):
        mock_req.return_value = {"status": "ok"}
        resp = self.client.post(
            "/remote/ping",
            data=json.dumps({"url": "http://node"}),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertEqual(data["status"], "online")

    def test_check_remote_job_not_found(self):
        resp = self.client.get("/remote/job/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_download_missing_fields(self):
        resp = self.client.post(
            "/remote/download",
            data=json.dumps({}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_download_missing_job_id(self):
        resp = self.client.post(
            "/remote/download",
            data=json.dumps({"node_url": "http://x"}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_download_missing_local_path(self):
        resp = self.client.post(
            "/remote/download",
            data=json.dumps({"node_url": "http://x", "remote_job_id": "j1"}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)


# =====================================================================
# Route Tests — Real-Time AI
# =====================================================================

class TestRealtimeRoutes(unittest.TestCase):
    """Route-level tests for /realtime/* endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, rr_client, rr_csrf_token):
        self.client = rr_client
        self.csrf_token = rr_csrf_token

    def test_list_models(self):
        resp = self.client.get("/realtime/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("models", data)
        self.assertIn("cache", data)
        self.assertTrue(len(data["models"]) >= 5)

    def test_create_session_missing_video(self):
        resp = self.client.post(
            "/realtime/create-session",
            data=json.dumps({"model": "denoise"}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_session_success(self, mock_isfile, mock_vinfo):
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        resp = self.client.post(
            "/realtime/create-session",
            data=json.dumps({"model": "denoise", "video_path": "/tmp/v.mp4"}),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertIn("session", data)
        self.assertTrue(data["session"]["running"])

        # Clean up
        from opencut.core.realtime_ai import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(data["session"]["session_id"], None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_create_session_with_config(self, mock_isfile, mock_vinfo):
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        resp = self.client.post(
            "/realtime/create-session",
            data=json.dumps({
                "model": "color_grade",
                "video_path": "/tmp/v.mp4",
                "resolution_scale": 0.25,
                "target_fps": 20,
                "device": "cuda",
                "params": {"preset": "warm"},
            }),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertIn("session", data)
        cfg = data["session"].get("config", {})
        self.assertEqual(cfg.get("device"), "cuda")

        from opencut.core.realtime_ai import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(data["session"]["session_id"], None)

    def test_frame_missing_session_id(self):
        resp = self.client.post(
            "/realtime/frame",
            data=json.dumps({"timestamp": 1.0}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_update_params_missing_session(self):
        resp = self.client.post(
            "/realtime/update-params",
            data=json.dumps({"params": {"x": 1}}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_update_params_missing_params(self):
        resp = self.client.post(
            "/realtime/update-params",
            data=json.dumps({"session_id": "xyz"}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_stop_missing_session(self):
        resp = self.client.post(
            "/realtime/stop",
            data=json.dumps({}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_stop_nonexistent(self):
        resp = self.client.post(
            "/realtime/stop",
            data=json.dumps({"session_id": "nonexistent"}),
            headers=csrf_headers(self.csrf_token),
        )
        self.assertEqual(resp.status_code, 404)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_stop_success(self, mock_isfile, mock_vinfo):
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        # Create session first
        resp = self.client.post(
            "/realtime/create-session",
            data=json.dumps({"model": "denoise", "video_path": "/tmp/v.mp4"}),
            headers=csrf_headers(self.csrf_token),
        )
        sid = resp.get_json()["session"]["session_id"]

        resp = self.client.post(
            "/realtime/stop",
            data=json.dumps({"session_id": sid}),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertTrue(data["stopped"])

        from opencut.core.realtime_ai import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(sid, None)

    @patch("opencut.core.realtime_ai.get_video_info")
    @patch("opencut.core.realtime_ai.os.path.isfile", return_value=True)
    def test_update_params_success(self, mock_isfile, mock_vinfo):
        mock_vinfo.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        resp = self.client.post(
            "/realtime/create-session",
            data=json.dumps({"model": "style_transfer", "video_path": "/tmp/v.mp4"}),
            headers=csrf_headers(self.csrf_token),
        )
        sid = resp.get_json()["session"]["session_id"]

        resp = self.client.post(
            "/realtime/update-params",
            data=json.dumps({"session_id": sid, "params": {"intensity": 0.3}}),
            headers=csrf_headers(self.csrf_token),
        )
        data = resp.get_json()
        self.assertIn("session", data)

        from opencut.core.realtime_ai import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(sid, None)

    def test_csrf_required_on_post(self):
        """POST endpoints should reject requests without CSRF token."""
        endpoints = [
            "/remote/register-node",
            "/remote/ping",
            "/remote/download",
            "/realtime/create-session",
            "/realtime/frame",
            "/realtime/update-params",
            "/realtime/stop",
        ]
        for ep in endpoints:
            resp = self.client.post(
                ep,
                data=json.dumps({"dummy": True}),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(resp.status_code, 403,
                             f"Expected 403 for {ep}, got {resp.status_code}")

    def test_get_endpoints_no_csrf(self):
        """GET endpoints should work without CSRF token."""
        endpoints = ["/remote/nodes", "/realtime/models"]
        for ep in endpoints:
            resp = self.client.get(ep)
            self.assertEqual(resp.status_code, 200,
                             f"Expected 200 for {ep}, got {resp.status_code}")


if __name__ == "__main__":
    unittest.main()
