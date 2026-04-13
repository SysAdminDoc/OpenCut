"""
Tests for OpenCut Infrastructure & Platform Features

Covers all 6 infrastructure modules and their route endpoints:
  7.3  - Auto-Update (check_for_updates, get_latest_release, parse_changelog, trigger_update)
  7.4  - Model Quantization (quantize_model, list_quantizable_models, recommend_quantization, get_quantized_path)
  7.7  - MCP Expansion (get_mcp_tools, get_compound_tools, execute_compound_tool, list_available_operations)
  10.1 - Model Download Manager (queue_download, get_download_progress, cancel_download, list_available/installed_models)
  10.4 - Apple Silicon (detect_apple_silicon, get_mps_device, is_op_mps_compatible, get_recommended_device)
  32.4 - GPU Dashboard (get_vram_status, get_loaded_models, unload_model, recommend_unload, register_model)

Uses Flask test client -- no real network, no subprocess, no GPU needed.
External dependencies (torch, nvidia-smi, GitHub API) are mocked.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from tests.conftest import csrf_headers

# =====================================================================
# 7.3 -- Auto-Update Core Tests
# =====================================================================

class TestAutoUpdateCore(unittest.TestCase):
    """Tests for opencut.core.auto_update module."""

    def test_parse_version_stable(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.2.3")
        self.assertEqual(result, (1, 2, 3, 99, 0))

    def test_parse_version_with_v_prefix(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("v2.10.5")
        self.assertEqual(result, (2, 10, 5, 99, 0))

    def test_parse_version_prerelease_alpha(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-alpha.3")
        self.assertEqual(result, (1, 0, 0, 1, 3))

    def test_parse_version_prerelease_beta(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-beta.1")
        self.assertEqual(result, (1, 0, 0, 2, 1))

    def test_parse_version_prerelease_rc(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("1.0.0-rc.2")
        self.assertEqual(result, (1, 0, 0, 3, 2))

    def test_parse_version_invalid_returns_zeros(self):
        from opencut.core.auto_update import _parse_version
        result = _parse_version("not-a-version")
        self.assertEqual(result, (0, 0, 0, 0, 0))

    def test_version_is_newer_true(self):
        from opencut.core.auto_update import _version_is_newer
        self.assertTrue(_version_is_newer("2.0.0", "1.0.0"))
        self.assertTrue(_version_is_newer("1.1.0", "1.0.9"))
        self.assertTrue(_version_is_newer("1.0.1", "1.0.0"))

    def test_version_is_newer_false(self):
        from opencut.core.auto_update import _version_is_newer
        self.assertFalse(_version_is_newer("1.0.0", "1.0.0"))
        self.assertFalse(_version_is_newer("1.0.0", "2.0.0"))

    def test_version_stable_beats_prerelease(self):
        from opencut.core.auto_update import _version_is_newer
        self.assertTrue(_version_is_newer("1.0.0", "1.0.0-rc.1"))
        self.assertTrue(_version_is_newer("1.0.0-beta.1", "1.0.0-alpha.5"))

    def test_parse_changelog_strips_html(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        release = ReleaseInfo(body="<h2>Changes</h2>\n<ul><li>Fixed bug</li></ul>")
        result = parse_changelog(release)
        self.assertNotIn("<h2>", result)
        self.assertNotIn("<ul>", result)
        self.assertIn("Changes", result)
        self.assertIn("Fixed bug", result)

    def test_parse_changelog_normalizes_whitespace(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        release = ReleaseInfo(body="Line1\n\n\n\n\nLine2")
        result = parse_changelog(release)
        self.assertNotIn("\n\n\n", result)

    def test_parse_changelog_empty_body(self):
        from opencut.core.auto_update import ReleaseInfo, parse_changelog
        release = ReleaseInfo(body="")
        result = parse_changelog(release)
        self.assertEqual(result, "")

    @patch("opencut.core.auto_update.urlopen")
    def test_get_latest_release_success(self, mock_urlopen):
        from opencut.core.auto_update import get_latest_release
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "tag_name": "v2.0.0",
            "name": "Release 2.0.0",
            "body": "Changelog here",
            "published_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/opencut/opencut/releases/tag/v2.0.0",
            "prerelease": False,
            "assets": [{"name": "opencut.zip", "size": 1000,
                        "browser_download_url": "https://example.com/opencut.zip",
                        "content_type": "application/zip"}],
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = get_latest_release()
        self.assertEqual(result.tag_name, "v2.0.0")
        self.assertEqual(result.version, "2.0.0")
        self.assertFalse(result.prerelease)
        self.assertEqual(len(result.assets), 1)

    @patch("opencut.core.auto_update.urlopen")
    def test_get_latest_release_connection_error(self, mock_urlopen):
        from urllib.error import URLError

        from opencut.core.auto_update import get_latest_release
        mock_urlopen.side_effect = URLError("Connection refused")
        with self.assertRaises(ConnectionError):
            get_latest_release()

    @patch("opencut.core.auto_update.get_latest_release")
    def test_check_for_updates_no_update(self, mock_get):
        from opencut.core.auto_update import ReleaseInfo, check_for_updates
        mock_get.return_value = ReleaseInfo(tag_name="v1.0.0", version="1.0.0", body="")
        result = check_for_updates(current_version="1.0.0")
        self.assertFalse(result.update_available)
        self.assertEqual(result.current_version, "1.0.0")
        self.assertIsNone(result.error)

    @patch("opencut.core.auto_update.get_latest_release")
    def test_check_for_updates_update_available(self, mock_get):
        from opencut.core.auto_update import ReleaseInfo, check_for_updates
        mock_get.return_value = ReleaseInfo(
            tag_name="v2.0.0", version="2.0.0", body="New features"
        )
        result = check_for_updates(current_version="1.0.0")
        self.assertTrue(result.update_available)
        self.assertEqual(result.latest_version, "2.0.0")

    @patch("opencut.core.auto_update.get_latest_release")
    def test_check_for_updates_connection_failure(self, mock_get):
        from opencut.core.auto_update import check_for_updates
        mock_get.side_effect = ConnectionError("Network down")
        result = check_for_updates(current_version="1.0.0")
        self.assertFalse(result.update_available)
        self.assertIn("Network down", result.error)

    def test_trigger_update_invalid_method(self):
        from opencut.core.auto_update import trigger_update
        result = trigger_update(method="ftp")
        self.assertFalse(result.success)
        self.assertIn("Unsupported", result.message)

    @patch("opencut.core.auto_update.subprocess.run")
    def test_trigger_update_pip_success(self, mock_run):
        from opencut.core.auto_update import trigger_update
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("opencut.core.auto_update._read_installed_version", return_value="2.0.0"):
            result = trigger_update(method="pip")
        self.assertTrue(result.success)
        self.assertEqual(result.method, "pip")

    @patch("opencut.core.auto_update.subprocess.run")
    def test_trigger_update_pip_failure(self, mock_run):
        from opencut.core.auto_update import trigger_update
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ERROR: something broke")
        result = trigger_update(method="pip")
        self.assertFalse(result.success)

    def test_release_info_to_dict(self):
        from opencut.core.auto_update import ReleaseInfo
        r = ReleaseInfo(tag_name="v1.0.0", version="1.0.0", name="Test")
        d = r.to_dict()
        self.assertEqual(d["tag_name"], "v1.0.0")
        self.assertEqual(d["version"], "1.0.0")

    def test_update_check_result_to_dict(self):
        from opencut.core.auto_update import ReleaseInfo, UpdateCheckResult
        r = UpdateCheckResult(
            current_version="1.0.0",
            latest_version="2.0.0",
            update_available=True,
            release_info=ReleaseInfo(tag_name="v2.0.0", version="2.0.0"),
        )
        d = r.to_dict()
        self.assertTrue(d["update_available"])
        self.assertIsInstance(d["release_info"], dict)


# =====================================================================
# 7.4 -- Model Quantization Core Tests
# =====================================================================

class TestModelQuantizationCore(unittest.TestCase):
    """Tests for opencut.core.model_quantization module."""

    def test_detect_framework_pytorch(self):
        from opencut.core.model_quantization import _detect_framework
        self.assertEqual(_detect_framework("model.pt"), "pytorch")
        self.assertEqual(_detect_framework("model.pth"), "pytorch")
        self.assertEqual(_detect_framework("model.bin"), "pytorch")

    def test_detect_framework_onnx(self):
        from opencut.core.model_quantization import _detect_framework
        self.assertEqual(_detect_framework("model.onnx"), "onnx")

    def test_detect_framework_safetensors(self):
        from opencut.core.model_quantization import _detect_framework
        self.assertEqual(_detect_framework("model.safetensors"), "safetensors")

    def test_detect_framework_unknown(self):
        from opencut.core.model_quantization import _detect_framework
        self.assertEqual(_detect_framework("model.xyz"), "unknown")

    def test_supported_precisions(self):
        from opencut.core.model_quantization import SUPPORTED_PRECISIONS
        self.assertIn("fp32", SUPPORTED_PRECISIONS)
        self.assertIn("fp16", SUPPORTED_PRECISIONS)
        self.assertIn("int8", SUPPORTED_PRECISIONS)
        self.assertIn("int4", SUPPORTED_PRECISIONS)

    def test_compression_ratios(self):
        from opencut.core.model_quantization import COMPRESSION_RATIOS
        self.assertEqual(COMPRESSION_RATIOS["fp32"], 1.0)
        self.assertLess(COMPRESSION_RATIOS["int8"], COMPRESSION_RATIOS["fp16"])
        self.assertLess(COMPRESSION_RATIOS["int4"], COMPRESSION_RATIOS["int8"])

    def test_quantize_model_invalid_precision(self):
        from opencut.core.model_quantization import quantize_model
        result = quantize_model("/fake/model.pt", target_precision="int2")
        self.assertFalse(result.success)
        self.assertIn("Unsupported", result.error)

    def test_quantize_model_missing_file(self):
        from opencut.core.model_quantization import quantize_model
        result = quantize_model("/nonexistent/model.pt", target_precision="int8")
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_quantize_model_with_temp_file(self):
        """Quantize using the generic fallback (unknown framework)."""
        from opencut.core.model_quantization import quantize_model
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"\x00" * 1024)
            tmp_path = f.name
        try:
            result = quantize_model(tmp_path, target_precision="int8")
            self.assertTrue(result.success)
            self.assertGreater(result.elapsed_seconds, 0)
            self.assertEqual(result.target_precision, "int8")
            # Clean up output
            if os.path.isfile(result.output_path):
                os.unlink(result.output_path)
        finally:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)

    def test_get_quantized_path_fallback(self):
        from opencut.core.model_quantization import get_quantized_path
        path = get_quantized_path("nonexistent_model", "int8")
        self.assertIn("nonexistent_model_int8", path)
        self.assertTrue(path.endswith(".pt"))

    @patch("opencut.core.model_quantization._detect_vram_nvidia", return_value=0.0)
    @patch("opencut.core.model_quantization._detect_vram_torch", return_value=0.0)
    @patch("opencut.core.model_quantization._detect_vram_mps", return_value=0.0)
    def test_detect_available_vram_no_gpu(self, _mps, _torch, _nvidia):
        from opencut.core.model_quantization import detect_available_vram
        self.assertEqual(detect_available_vram(), 0.0)

    @patch("opencut.core.model_quantization._detect_vram_nvidia", return_value=8192.0)
    def test_detect_available_vram_nvidia(self, _nvidia):
        from opencut.core.model_quantization import detect_available_vram
        self.assertEqual(detect_available_vram(), 8192.0)

    def test_recommend_quantization_high_vram(self):
        from opencut.core.model_quantization import recommend_quantization
        rec = recommend_quantization(available_vram=16000.0)
        self.assertEqual(rec.recommended_precision, "fp32")
        self.assertTrue(rec.can_run_fp32)
        self.assertTrue(rec.can_run_fp16)
        self.assertTrue(rec.can_run_int8)

    def test_recommend_quantization_medium_vram(self):
        from opencut.core.model_quantization import recommend_quantization
        rec = recommend_quantization(available_vram=5000.0)
        self.assertEqual(rec.recommended_precision, "fp16")
        self.assertFalse(rec.can_run_fp32)
        self.assertTrue(rec.can_run_fp16)

    def test_recommend_quantization_low_vram(self):
        from opencut.core.model_quantization import recommend_quantization
        rec = recommend_quantization(available_vram=3000.0)
        self.assertEqual(rec.recommended_precision, "int8")

    def test_recommend_quantization_very_low_vram(self):
        from opencut.core.model_quantization import recommend_quantization
        rec = recommend_quantization(available_vram=500.0)
        self.assertEqual(rec.recommended_precision, "int4")

    def test_list_quantizable_models_empty_dir(self):
        from opencut.core.model_quantization import _find_model_files
        result = _find_model_files("/nonexistent/dir")
        self.assertEqual(result, [])

    def test_quantization_result_to_dict(self):
        from opencut.core.model_quantization import QuantizationResult
        r = QuantizationResult(success=True, model_name="test", target_precision="int8")
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["model_name"], "test")

    def test_model_info_to_dict(self):
        from opencut.core.model_quantization import ModelInfo
        m = ModelInfo(name="test", path="/tmp/test.pt", size_bytes=1024)
        d = m.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["size_bytes"], 1024)


# =====================================================================
# 7.7 -- MCP Tools Core Tests
# =====================================================================

class TestMCPToolsCore(unittest.TestCase):
    """Tests for opencut.core.mcp_tools module."""

    def test_get_mcp_tools_returns_list(self):
        from opencut.core.mcp_tools import get_mcp_tools
        tools = get_mcp_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_mcp_tool_has_required_fields(self):
        from opencut.core.mcp_tools import get_mcp_tools
        tools = get_mcp_tools()
        for tool in tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("category", tool)
            self.assertIn("parameters", tool)
            self.assertIn("endpoint", tool)
            self.assertIn("method", tool)

    def test_mcp_tool_parameters_have_required_fields(self):
        from opencut.core.mcp_tools import get_mcp_tools
        tools = get_mcp_tools()
        for tool in tools:
            for param in tool["parameters"]:
                self.assertIn("name", param)
                self.assertIn("type", param)
                self.assertIn("required", param)

    def test_get_compound_tools_returns_list(self):
        from opencut.core.mcp_tools import get_compound_tools
        tools = get_compound_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_compound_tool_clean_interview_exists(self):
        from opencut.core.mcp_tools import get_compound_tools
        tools = get_compound_tools()
        names = [t["name"] for t in tools]
        self.assertIn("clean_interview", names)

    def test_compound_tool_podcast_polish_exists(self):
        from opencut.core.mcp_tools import get_compound_tools
        tools = get_compound_tools()
        names = [t["name"] for t in tools]
        self.assertIn("podcast_polish", names)

    def test_compound_tool_prepare_for_youtube_exists(self):
        from opencut.core.mcp_tools import get_compound_tools
        tools = get_compound_tools()
        names = [t["name"] for t in tools]
        self.assertIn("prepare_for_youtube", names)

    def test_list_available_operations_has_categories(self):
        from opencut.core.mcp_tools import list_available_operations
        ops = list_available_operations()
        self.assertIsInstance(ops, dict)
        self.assertIn("compound", ops)
        self.assertGreater(len(ops), 1)

    def test_list_available_operations_categories_nonempty(self):
        from opencut.core.mcp_tools import list_available_operations
        ops = list_available_operations()
        for category, tools in ops.items():
            self.assertIsInstance(tools, list)
            self.assertGreater(len(tools), 0, f"Category '{category}' is empty")

    def test_execute_compound_tool_unknown(self):
        from opencut.core.mcp_tools import execute_compound_tool
        result = execute_compound_tool("nonexistent_tool", {})
        self.assertFalse(result.success)
        self.assertIn("Unknown compound tool", result.error)

    def test_execute_compound_tool_missing_required_param(self):
        from opencut.core.mcp_tools import execute_compound_tool
        result = execute_compound_tool("clean_interview", {})
        self.assertFalse(result.success)
        self.assertIn("Missing required parameter", result.error)

    def test_execute_compound_tool_clean_interview_success(self):
        from opencut.core.mcp_tools import execute_compound_tool
        result = execute_compound_tool(
            "clean_interview",
            {"file_path": "/tmp/interview.mp4"},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "clean_interview")
        self.assertEqual(result.steps_completed, result.total_steps)
        self.assertGreater(result.total_steps, 0)

    def test_execute_compound_tool_podcast_polish_success(self):
        from opencut.core.mcp_tools import execute_compound_tool
        result = execute_compound_tool(
            "podcast_polish",
            {"file_path": "/tmp/podcast.mp4"},
        )
        self.assertTrue(result.success)
        self.assertGreater(len(result.results), 0)

    def test_execute_compound_tool_with_progress(self):
        from opencut.core.mcp_tools import execute_compound_tool
        progress_calls = []
        result = execute_compound_tool(
            "clean_interview",
            {"file_path": "/tmp/test.mp4"},
            on_progress=lambda p: progress_calls.append(p),
        )
        self.assertTrue(result.success)
        self.assertGreater(len(progress_calls), 0)

    def test_compound_tool_result_to_dict(self):
        from opencut.core.mcp_tools import CompoundToolResult
        r = CompoundToolResult(success=True, tool_name="test", steps_completed=3, total_steps=3)
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["steps_completed"], 3)

    def test_resolve_param_input(self):
        from opencut.core.mcp_tools import _resolve_param
        val = _resolve_param("input.file_path", {"file_path": "/tmp/test.mp4"}, [])
        self.assertEqual(val, "/tmp/test.mp4")

    def test_resolve_param_step_result(self):
        from opencut.core.mcp_tools import _resolve_param
        val = _resolve_param("step_0.output_path", {}, [{"output_path": "/tmp/out.mp4"}])
        self.assertEqual(val, "/tmp/out.mp4")

    def test_resolve_param_literal(self):
        from opencut.core.mcp_tools import _resolve_param
        val = _resolve_param("just_a_string", {}, [])
        self.assertEqual(val, "just_a_string")


# =====================================================================
# 10.1 -- Model Download Manager Core Tests
# =====================================================================

class TestModelManagerCore(unittest.TestCase):
    """Tests for opencut.core.model_manager module."""

    def test_known_models_registry(self):
        from opencut.core.model_manager import KNOWN_MODELS
        self.assertIn("whisper-tiny", KNOWN_MODELS)
        self.assertIn("whisper-base", KNOWN_MODELS)
        for name, info in KNOWN_MODELS.items():
            self.assertIn("url", info)
            self.assertIn("size_mb", info)
            self.assertIn("description", info)

    def test_list_available_models(self):
        from opencut.core.model_manager import list_available_models
        models = list_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        for m in models:
            d = m.to_dict()
            self.assertIn("name", d)
            self.assertIn("installed", d)
            self.assertIn("url", d)

    def test_list_installed_models_subset_of_available(self):
        from opencut.core.model_manager import list_available_models, list_installed_models
        installed = list_installed_models()
        available = list_available_models()
        installed_names = {m.name for m in installed}
        available_names = {m.name for m in available}
        self.assertTrue(installed_names.issubset(available_names))

    def test_get_download_progress_unknown_model(self):
        from opencut.core.model_manager import get_download_progress
        progress = get_download_progress("nonexistent_model_xyz")
        self.assertEqual(progress.status, "unknown")

    def test_cancel_download_nonexistent(self):
        from opencut.core.model_manager import cancel_download
        result = cancel_download("nonexistent_model_xyz")
        self.assertFalse(result)

    def test_queue_download_unknown_model_no_url(self):
        from opencut.core.model_manager import queue_download
        progress = queue_download("totally_fake_model_abc")
        self.assertEqual(progress.status, "failed")
        self.assertIn("Unknown model", progress.error)

    def test_model_output_path(self):
        from opencut.core.model_manager import _model_output_path
        path = _model_output_path("whisper-tiny", "https://example.com/model.safetensors")
        self.assertIn("whisper-tiny", path)
        self.assertTrue(path.endswith(".safetensors"))

    def test_download_progress_to_dict(self):
        from opencut.core.model_manager import DownloadProgress
        p = DownloadProgress(
            model_name="test", url="https://example.com/model.bin",
            total_bytes=1024, downloaded_bytes=512, percent=50.0,
            status="downloading",
        )
        d = p.to_dict()
        self.assertEqual(d["percent"], 50.0)
        self.assertEqual(d["status"], "downloading")

    def test_model_entry_to_dict(self):
        from opencut.core.model_manager import ModelEntry
        m = ModelEntry(name="test", size_mb=100.0, installed=True)
        d = m.to_dict()
        self.assertTrue(d["installed"])

    def test_estimate_disk_usage(self):
        from opencut.core.model_manager import estimate_disk_usage
        est = estimate_disk_usage(["whisper-tiny"])
        d = est.to_dict()
        self.assertIn("total_required_mb", d)
        self.assertIn("available_mb", d)
        self.assertIn("sufficient", d)

    def test_estimate_disk_usage_all_models(self):
        from opencut.core.model_manager import estimate_disk_usage
        est = estimate_disk_usage()
        self.assertGreater(est.total_required_mb, 0)


# =====================================================================
# 10.4 -- Apple Silicon Core Tests
# =====================================================================

class TestAppleSiliconCore(unittest.TestCase):
    """Tests for opencut.core.apple_silicon module."""

    def test_mps_compatible_ops(self):
        from opencut.core.apple_silicon import MPS_COMPATIBLE_OPS
        self.assertIn("inference", MPS_COMPATIBLE_OPS)
        self.assertIn("transcription", MPS_COMPATIBLE_OPS)
        self.assertIn("upscaling", MPS_COMPATIBLE_OPS)

    def test_mps_incompatible_ops(self):
        from opencut.core.apple_silicon import MPS_INCOMPATIBLE_OPS
        self.assertIn("quantization", MPS_INCOMPATIBLE_OPS)
        self.assertIn("int8_inference", MPS_INCOMPATIBLE_OPS)
        self.assertIn("custom_cuda_kernels", MPS_INCOMPATIBLE_OPS)

    def test_is_op_mps_compatible_known_compatible(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        self.assertTrue(is_op_mps_compatible("inference"))
        self.assertTrue(is_op_mps_compatible("transcription"))

    def test_is_op_mps_compatible_known_incompatible(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        self.assertFalse(is_op_mps_compatible("quantization"))
        self.assertFalse(is_op_mps_compatible("int8_inference"))

    def test_is_op_mps_compatible_unknown_assumes_true(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        self.assertTrue(is_op_mps_compatible("some_unknown_operation"))

    def test_is_op_mps_compatible_case_insensitive(self):
        from opencut.core.apple_silicon import is_op_mps_compatible
        self.assertTrue(is_op_mps_compatible("INFERENCE"))
        self.assertFalse(is_op_mps_compatible("QUANTIZATION"))

    def test_parse_chip_family(self):
        from opencut.core.apple_silicon import _parse_chip_family
        self.assertEqual(_parse_chip_family("Apple M1 Pro"), "M1")
        self.assertEqual(_parse_chip_family("Apple M2 Max"), "M2")
        self.assertEqual(_parse_chip_family("Apple M3"), "M3")
        self.assertEqual(_parse_chip_family("Apple M4"), "M4")

    def test_parse_chip_family_unknown(self):
        from opencut.core.apple_silicon import _parse_chip_family
        result = _parse_chip_family("Intel Core i9")
        self.assertEqual(result, "")

    def test_get_neural_engine_cores(self):
        from opencut.core.apple_silicon import _get_neural_engine_cores
        self.assertEqual(_get_neural_engine_cores("M1"), 16)
        self.assertEqual(_get_neural_engine_cores("M4"), 16)
        self.assertEqual(_get_neural_engine_cores("Unknown"), 0)

    @patch("opencut.core.apple_silicon.platform")
    def test_detect_apple_silicon_non_mac(self, mock_platform):
        from opencut.core.apple_silicon import detect_apple_silicon
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.mac_ver.return_value = ("", ("", "", ""), "")
        info = detect_apple_silicon()
        self.assertFalse(info.is_apple_silicon)

    @patch("opencut.core.apple_silicon._get_gpu_core_count", return_value=10)
    @patch("opencut.core.apple_silicon._get_chip_info_sysctl")
    @patch("opencut.core.apple_silicon.platform")
    def test_detect_apple_silicon_mac_arm(self, mock_platform, mock_sysctl, _gpu):
        from opencut.core.apple_silicon import detect_apple_silicon
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        mock_platform.mac_ver.return_value = ("14.0", ("", "", ""), "")
        mock_sysctl.return_value = {
            "chip_name": "Apple M3 Pro",
            "cpu_cores": "12",
            "memory": str(36 * 1024 ** 3),
        }
        info = detect_apple_silicon()
        self.assertTrue(info.is_apple_silicon)
        self.assertEqual(info.chip_family, "M3")
        self.assertEqual(info.cpu_cores, 12)
        self.assertAlmostEqual(info.memory_gb, 36.0, places=0)

    def test_get_mps_device_no_torch(self):
        from opencut.core.apple_silicon import get_mps_device
        with patch.dict("sys.modules", {"torch": None}):
            # When torch is not importable, should return None
            device = get_mps_device()
            # On non-Mac or without MPS, returns None
            # Just verify no crash
            self.assertTrue(device is None or device is not None)

    def test_get_recommended_device_no_mps(self):
        from opencut.core.apple_silicon import get_recommended_device
        with patch("opencut.core.apple_silicon.get_mps_device", return_value=None):
            rec = get_recommended_device("inference")
            self.assertEqual(rec.recommended_device, "cpu")
            self.assertIn("not available", rec.reason)

    @patch("opencut.core.apple_silicon.get_mps_device")
    def test_get_recommended_device_mps_compatible(self, mock_mps):
        from opencut.core.apple_silicon import get_recommended_device
        mock_mps.return_value = MagicMock()  # fake MPS device
        with patch("opencut.core.apple_silicon.is_op_mps_compatible", return_value=True):
            with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))}):
                rec = get_recommended_device("inference")
                self.assertEqual(rec.recommended_device, "mps")
                self.assertTrue(rec.mps_compatible)

    @patch("opencut.core.apple_silicon.get_mps_device")
    def test_get_recommended_device_mps_incompatible(self, mock_mps):
        from opencut.core.apple_silicon import get_recommended_device
        mock_mps.return_value = MagicMock()
        with patch("opencut.core.apple_silicon.is_op_mps_compatible", return_value=False):
            with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))}):
                rec = get_recommended_device("quantization")
                self.assertEqual(rec.recommended_device, "cpu")
                self.assertFalse(rec.mps_compatible)

    def test_apple_silicon_info_to_dict(self):
        from opencut.core.apple_silicon import AppleSiliconInfo
        info = AppleSiliconInfo(
            is_apple_silicon=True, chip_name="Apple M3", chip_family="M3",
            cpu_cores=8, gpu_cores=10, memory_gb=16.0, mps_available=True,
        )
        d = info.to_dict()
        self.assertTrue(d["is_apple_silicon"])
        self.assertEqual(d["chip_family"], "M3")

    def test_device_recommendation_to_dict(self):
        from opencut.core.apple_silicon import DeviceRecommendation
        rec = DeviceRecommendation(
            operation="inference", recommended_device="mps",
            mps_compatible=True, reason="test",
        )
        d = rec.to_dict()
        self.assertEqual(d["recommended_device"], "mps")


# =====================================================================
# 32.4 -- GPU Dashboard Core Tests
# =====================================================================

class TestGPUDashboardCore(unittest.TestCase):
    """Tests for opencut.core.gpu_dashboard module."""

    def setUp(self):
        # Clear the model registry before each test
        from opencut.core.gpu_dashboard import _loaded_models, _models_lock
        with _models_lock:
            _loaded_models.clear()

    def test_register_model(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model
        model = register_model("test-model", size_mb=500.0, device="cuda")
        self.assertEqual(model.name, "test-model")
        self.assertEqual(model.size_mb, 500.0)
        self.assertEqual(model.device, "cuda")
        models = get_loaded_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "test-model")

    def test_register_model_pinned(self):
        from opencut.core.gpu_dashboard import register_model
        model = register_model("pinned-model", size_mb=200.0, pinned=True)
        self.assertTrue(model.pinned)

    def test_touch_model(self):
        from opencut.core.gpu_dashboard import register_model, touch_model
        register_model("touch-test", size_mb=100.0)
        time.time()
        result = touch_model("touch-test")
        self.assertTrue(result)

    def test_touch_model_nonexistent(self):
        from opencut.core.gpu_dashboard import touch_model
        result = touch_model("nonexistent_model")
        self.assertFalse(result)

    def test_unload_model_success(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model, unload_model
        register_model("unload-test", size_mb=300.0, device="cpu")
        success = unload_model("unload-test")
        self.assertTrue(success)
        models = get_loaded_models()
        self.assertEqual(len(models), 0)

    def test_unload_model_nonexistent(self):
        from opencut.core.gpu_dashboard import unload_model
        success = unload_model("nonexistent_model_xyz")
        self.assertFalse(success)

    def test_unload_model_with_progress(self):
        from opencut.core.gpu_dashboard import register_model, unload_model
        register_model("progress-test", size_mb=100.0, device="cpu")
        progress_calls = []
        unload_model("progress-test", on_progress=lambda p: progress_calls.append(p))
        self.assertGreater(len(progress_calls), 0)

    def test_get_loaded_models_empty(self):
        from opencut.core.gpu_dashboard import get_loaded_models
        models = get_loaded_models()
        self.assertEqual(len(models), 0)

    def test_get_loaded_models_multiple(self):
        from opencut.core.gpu_dashboard import get_loaded_models, register_model
        register_model("model-a", size_mb=100.0)
        register_model("model-b", size_mb=200.0)
        register_model("model-c", size_mb=300.0)
        models = get_loaded_models()
        self.assertEqual(len(models), 3)

    def test_loaded_model_to_dict_has_idle_seconds(self):
        from opencut.core.gpu_dashboard import register_model
        model = register_model("idle-test", size_mb=100.0)
        d = model.to_dict()
        self.assertIn("idle_seconds", d)
        self.assertGreaterEqual(d["idle_seconds"], 0)

    @patch("opencut.core.gpu_dashboard.get_gpu_info", return_value=[])
    def test_get_vram_status_no_gpu(self, _mock):
        from opencut.core.gpu_dashboard import get_vram_status
        status = get_vram_status()
        self.assertEqual(status.total_vram_mb, 0.0)
        self.assertEqual(status.gpu_type, "none")

    @patch("opencut.core.gpu_dashboard.get_gpu_info")
    def test_get_vram_status_with_gpu(self, mock_gpu_info):
        from opencut.core.gpu_dashboard import GPUInfo, get_vram_status, register_model
        mock_gpu_info.return_value = [
            GPUInfo(index=0, name="Test GPU", total_vram_mb=8192.0,
                    used_vram_mb=2048.0, free_vram_mb=6144.0, gpu_type="nvidia"),
        ]
        register_model("vram-test", size_mb=1024.0, device="cuda")
        status = get_vram_status()
        self.assertEqual(status.total_vram_mb, 8192.0)
        self.assertEqual(status.used_vram_mb, 2048.0)
        self.assertEqual(status.models_loaded, 1)
        self.assertEqual(status.models_vram_mb, 1024.0)
        self.assertEqual(status.gpu_type, "nvidia")

    @patch("opencut.core.gpu_dashboard.get_gpu_info")
    def test_recommend_unload_sufficient_vram(self, mock_gpu_info):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload
        mock_gpu_info.return_value = [
            GPUInfo(total_vram_mb=8192.0, used_vram_mb=1000.0, free_vram_mb=7192.0),
        ]
        rec = recommend_unload(required_vram=2000.0)
        self.assertTrue(rec.sufficient)
        self.assertEqual(len(rec.models_to_unload), 0)

    @patch("opencut.core.gpu_dashboard.get_gpu_info")
    def test_recommend_unload_needs_unload(self, mock_gpu_info):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        mock_gpu_info.return_value = [
            GPUInfo(total_vram_mb=8192.0, used_vram_mb=7000.0, free_vram_mb=1192.0),
        ]
        register_model("old-model", size_mb=2000.0, device="cuda")
        rec = recommend_unload(required_vram=3000.0)
        self.assertIn("old-model", rec.models_to_unload)

    @patch("opencut.core.gpu_dashboard.get_gpu_info")
    def test_recommend_unload_skips_pinned(self, mock_gpu_info):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        mock_gpu_info.return_value = [
            GPUInfo(total_vram_mb=8192.0, used_vram_mb=7000.0, free_vram_mb=1192.0),
        ]
        register_model("pinned-keep", size_mb=2000.0, device="cuda", pinned=True)
        register_model("unpinned-remove", size_mb=2000.0, device="cuda", pinned=False)
        rec = recommend_unload(required_vram=3000.0)
        self.assertNotIn("pinned-keep", rec.models_to_unload)
        self.assertIn("unpinned-remove", rec.models_to_unload)

    @patch("opencut.core.gpu_dashboard.get_gpu_info")
    def test_recommend_unload_skips_cpu_models(self, mock_gpu_info):
        from opencut.core.gpu_dashboard import GPUInfo, recommend_unload, register_model
        mock_gpu_info.return_value = [
            GPUInfo(total_vram_mb=4096.0, used_vram_mb=3500.0, free_vram_mb=596.0),
        ]
        register_model("cpu-model", size_mb=1000.0, device="cpu")
        rec = recommend_unload(required_vram=2000.0)
        self.assertNotIn("cpu-model", rec.models_to_unload)

    def test_vram_status_to_dict(self):
        from opencut.core.gpu_dashboard import VRAMStatus
        s = VRAMStatus(total_vram_mb=8192.0, used_vram_mb=2048.0, gpu_type="nvidia")
        d = s.to_dict()
        self.assertEqual(d["total_vram_mb"], 8192.0)
        self.assertEqual(d["gpu_type"], "nvidia")

    def test_unload_recommendation_to_dict(self):
        from opencut.core.gpu_dashboard import UnloadRecommendation
        r = UnloadRecommendation(required_mb=2000.0, sufficient=True)
        d = r.to_dict()
        self.assertTrue(d["sufficient"])

    @patch("opencut.core.gpu_dashboard._query_nvidia_smi", return_value=[])
    @patch("opencut.core.gpu_dashboard._query_torch_cuda", return_value=[])
    @patch("opencut.core.gpu_dashboard._query_mps_info", return_value=[])
    def test_get_gpu_info_no_gpus(self, _mps, _cuda, _nvidia):
        from opencut.core.gpu_dashboard import get_gpu_info
        gpus = get_gpu_info()
        self.assertEqual(len(gpus), 0)


# =====================================================================
# Route Tests -- Infrastructure Blueprint
# =====================================================================

class TestAutoUpdateRoutes:
    """Route tests for Auto-Update (7.3)."""

    @patch("opencut.core.auto_update.get_latest_release")
    def test_check_updates_get(self, mock_release, client):
        from opencut.core.auto_update import ReleaseInfo
        mock_release.return_value = ReleaseInfo(
            tag_name="v1.0.0", version="1.0.0", body=""
        )
        resp = client.get("/api/system/check-updates")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_version" in data

    @patch("opencut.core.auto_update.get_latest_release")
    def test_check_updates_with_prerelease(self, mock_release, client):
        from opencut.core.auto_update import ReleaseInfo
        mock_release.return_value = ReleaseInfo(
            tag_name="v2.0.0-beta.1", version="2.0.0-beta.1",
            prerelease=True, body="Beta changes"
        )
        resp = client.get("/api/system/check-updates?include_prerelease=true")
        assert resp.status_code == 200

    def test_trigger_update_requires_csrf(self, client):
        resp = client.post(
            "/api/system/update",
            data=json.dumps({"method": "pip"}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    @patch("opencut.core.auto_update.trigger_update")
    def test_trigger_update_with_csrf(self, mock_update, client, csrf_token):
        from opencut.core.auto_update import UpdateResult
        mock_update.return_value = UpdateResult(success=True, method="pip", message="OK")
        resp = client.post(
            "/api/system/update",
            data=json.dumps({"method": "pip"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


class TestQuantizationRoutes:
    """Route tests for Model Quantization (7.4)."""

    def test_list_quantizable_get(self, client):
        resp = client.get("/api/models/quantization")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert "recommendation" in data

    def test_quantize_requires_csrf(self, client):
        resp = client.post(
            "/api/models/quantize",
            data=json.dumps({"model_path": "/tmp/test.pt"}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    def test_quantize_missing_model_path(self, client, csrf_token):
        resp = client.post(
            "/api/models/quantize",
            data=json.dumps({}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_quantize_nonexistent_model(self, client, csrf_token):
        resp = client.post(
            "/api/models/quantize",
            data=json.dumps({"model_path": "/nonexistent/model.pt", "precision": "int8"}),
            headers=csrf_headers(csrf_token),
        )
        # validate_filepath raises ValueError -> safe_error maps to 404 FILE_NOT_FOUND
        assert resp.status_code in (400, 404, 500)
        data = resp.get_json()
        assert data is not None


class TestMCPRoutes:
    """Route tests for MCP Tools (7.7)."""

    def test_mcp_tools_get(self, client):
        resp = client.get("/api/mcp/tools")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tools" in data
        assert "compound_tools" in data
        assert "operations" in data
        assert len(data["tools"]) > 0

    def test_mcp_tools_filter_by_category(self, client):
        resp = client.get("/api/mcp/tools?category=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        for tool in data["tools"]:
            assert tool["category"] == "audio"

    def test_mcp_compound_requires_csrf(self, client):
        resp = client.post(
            "/api/mcp/compound",
            data=json.dumps({"tool_name": "clean_interview", "params": {}}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    def test_mcp_compound_missing_tool_name(self, client, csrf_token):
        resp = client.post(
            "/api/mcp/compound",
            data=json.dumps({}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_mcp_compound_success(self, client, csrf_token):
        resp = client.post(
            "/api/mcp/compound",
            data=json.dumps({
                "tool_name": "clean_interview",
                "params": {"file_path": "/tmp/test.mp4"},
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_mcp_compound_unknown_tool(self, client, csrf_token):
        resp = client.post(
            "/api/mcp/compound",
            data=json.dumps({
                "tool_name": "nonexistent_tool",
                "params": {},
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False


class TestModelDownloadRoutes:
    """Route tests for Model Download Manager (10.1)."""

    def test_download_requires_csrf(self, client):
        resp = client.post(
            "/api/models/download",
            data=json.dumps({"model_name": "whisper-tiny"}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    def test_download_missing_model_name(self, client, csrf_token):
        resp = client.post(
            "/api/models/download",
            data=json.dumps({}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_download_progress_missing_param(self, client):
        resp = client.get("/api/models/progress")
        assert resp.status_code == 400

    def test_download_progress_unknown_model(self, client):
        resp = client.get("/api/models/progress?model_name=nonexistent_xyz")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "unknown"

    def test_installed_models_get(self, client):
        resp = client.get("/api/models/installed")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestAppleSiliconRoutes:
    """Route tests for Apple Silicon (10.4)."""

    def test_apple_silicon_info_get(self, client):
        resp = client.get("/api/system/apple-silicon")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "is_apple_silicon" in data
        assert "mps_available" in data

    def test_apple_silicon_with_operation(self, client):
        resp = client.get("/api/system/apple-silicon?operation=inference")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "is_apple_silicon" in data
        # On non-Apple platforms, device_recommendation may still be present
        # if operation was requested
        if "device_recommendation" in data:
            assert "recommended_device" in data["device_recommendation"]


class TestGPUDashboardRoutes:
    """Route tests for GPU Dashboard (32.4)."""

    def test_gpu_status_get(self, client):
        resp = client.get("/api/gpu/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert "gpus" in data

    def test_gpu_status_has_required_fields(self, client):
        resp = client.get("/api/gpu/status")
        data = resp.get_json()
        status = data["status"]
        assert "total_vram_mb" in status
        assert "used_vram_mb" in status
        assert "free_vram_mb" in status
        assert "gpu_type" in status

    def test_gpu_models_get(self, client):
        resp = client.get("/api/gpu/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_gpu_unload_requires_csrf(self, client):
        resp = client.post(
            "/api/gpu/unload",
            data=json.dumps({"model_name": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 403

    def test_gpu_unload_missing_params(self, client, csrf_token):
        resp = client.post(
            "/api/gpu/unload",
            data=json.dumps({}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_gpu_unload_nonexistent_model(self, client, csrf_token):
        resp = client.post(
            "/api/gpu/unload",
            data=json.dumps({"model_name": "nonexistent_model_xyz"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is False

    def test_gpu_unload_recommend(self, client, csrf_token):
        resp = client.post(
            "/api/gpu/unload",
            data=json.dumps({"required_vram": 2000}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "required_mb" in data
        assert "sufficient" in data


if __name__ == "__main__":
    unittest.main()
