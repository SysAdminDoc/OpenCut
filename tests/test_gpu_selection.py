import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import csrf_headers


DEVICES = [
    {"index": 0, "name": "RTX A", "memory_total_mb": 8192.0},
    {"index": 1, "name": "RTX B", "memory_total_mb": 16384.0},
]


def test_gpu_selection_status_marks_manual_adapter(monkeypatch):
    import opencut.gpu as gpu

    monkeypatch.setattr(gpu, "_configured_gpu_index", lambda: 1)
    status = gpu.gpu_selection_status(DEVICES)

    assert status["configured_index"] == 1
    assert status["selected_index"] == 1
    assert status["selection_mode"] == "manual"
    assert status["device"] == "cuda:1"
    assert status["selection_error"] is None


def test_blackwell_gpu_gets_safe_faster_whisper_compute_type():
    import opencut.gpu as gpu

    recommendation = gpu.faster_whisper_compute_recommendation({
        "index": 0,
        "name": "NVIDIA GeForce RTX 5090",
    })

    assert recommendation["architecture"] == "blackwell"
    assert recommendation["affected"] is True
    assert recommendation["compute_type"] == "float16"
    assert recommendation["changed"] is True
    assert "cuBLAS" in recommendation["warning"]


def test_gpu_selection_status_reports_faster_whisper_substitution(monkeypatch):
    import opencut.gpu as gpu

    monkeypatch.setattr(gpu, "_configured_gpu_index", lambda: 0)
    status = gpu.gpu_selection_status([{
        "index": 0,
        "name": "NVIDIA RTX 5090 Laptop GPU",
        "memory_total_mb": 16384.0,
    }])

    assert status["faster_whisper"]["architecture"] == "blackwell"
    assert status["faster_whisper"]["compute_type"] == "float16"
    assert status["devices"][0]["faster_whisper"]["changed"] is True


def test_invalid_gpu_index_lists_available_adapters(monkeypatch):
    import opencut.gpu as gpu

    monkeypatch.setattr(gpu, "list_gpu_devices", lambda: DEVICES)
    with pytest.raises(gpu.GPUSelectionError) as exc_info:
        gpu.validate_gpu_index(4)

    payload = exc_info.value.to_dict()
    assert payload["code"] == "INVALID_GPU_INDEX"
    assert [item["index"] for item in payload["available_devices"]] == [0, 1]


def test_activate_selected_gpu_sets_worker_thread_device(monkeypatch):
    import opencut.gpu as gpu

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            set_device=MagicMock(),
        )
    )
    monkeypatch.setattr(gpu, "_configured_gpu_index", lambda: 1)
    index = gpu.activate_selected_gpu(torch_module=fake_torch, devices=DEVICES)

    assert index == 1
    fake_torch.cuda.set_device.assert_called_once_with(1)


def test_activate_invalid_gpu_fails_before_model_work(monkeypatch):
    import opencut.gpu as gpu

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    monkeypatch.setattr(gpu, "_configured_gpu_index", lambda: 3)
    with pytest.raises(gpu.GPUSelectionError) as exc_info:
        gpu.activate_selected_gpu(torch_module=fake_torch, devices=DEVICES)
    assert exc_info.value.code == "INVALID_GPU_INDEX"


def test_system_gpu_reports_all_devices_and_selection(client, monkeypatch):
    import opencut.routes.system_runtime_routes as runtime

    monkeypatch.setattr(runtime, "_detect_gpu", lambda: {
        "available": True,
        "name": "RTX B",
        "vram_mb": 16384,
        "selected_index": 1,
        "configured_index": 1,
        "devices": DEVICES,
    })
    response = client.get("/system/gpu")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["selected_index"] == 1
    assert [device["index"] for device in payload["devices"]] == [0, 1]


def test_system_status_reports_selected_adapter(client, monkeypatch):
    import opencut.routes.system_runtime_routes as runtime

    monkeypatch.setattr(runtime, "_detect_gpu", lambda: {
        "available": True,
        "name": "RTX B",
        "vram_mb": 16384,
        "selected_index": 1,
        "configured_index": 1,
        "selection_mode": "manual",
        "device": "cuda:1",
        "index": 1,
        "devices": DEVICES,
        "selection_error": None,
    })
    monkeypatch.setattr(runtime, "_get_vram_used", lambda selected_index: 2048)

    response = client.get("/system/status")
    gpu = response.get_json()["gpu"]

    assert response.status_code == 200
    assert gpu["index"] == 1
    assert gpu["configured_index"] == 1
    assert gpu["selected_index"] == 1
    assert gpu["device"] == "cuda:1"
    assert gpu["vram_used_mb"] == 2048


def test_vram_cache_is_scoped_to_selected_adapter(client, monkeypatch):
    import opencut.routes.system_runtime_routes as runtime

    monkeypatch.setattr(runtime, "_vram_cache", {"used_mb": 0, "index": None, "ts": 0})
    monkeypatch.setattr(runtime.time, "time", lambda: 100)
    monkeypatch.setattr(
        runtime._sp,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0, 100\n1, 200\n",
        ),
    )

    assert runtime._get_vram_used(1) == 200
    assert runtime._get_vram_used(0) == 100


def test_onnx_inference_pins_selected_provider_without_duplicate_cpu(tmp_path, monkeypatch):
    import opencut.core.onnx_runtime as runtime
    import opencut.gpu as gpu

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    captured = {}

    class FakeSession:
        def __init__(self, path, providers):
            captured["path"] = path
            captured["providers"] = providers

        def run(self, _outputs, _inputs):
            return [[1.0]]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    monkeypatch.setattr(runtime, "get_optimal_provider", lambda: "CUDAExecutionProvider")
    monkeypatch.setattr(
        gpu,
        "selected_onnx_providers",
        lambda: [("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
    )

    result = runtime.run_onnx_inference(str(model_path), [1.0])

    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]
    assert result["provider"] == "CUDAExecutionProvider"


def test_system_gpu_rejects_invalid_selection_with_structured_error(client, csrf_token, monkeypatch):
    import opencut.routes.system_runtime_routes as runtime
    from opencut.gpu import GPUSelectionError

    monkeypatch.setattr(runtime, "save_gpu_selection", MagicMock(side_effect=GPUSelectionError(9, DEVICES)))
    response = client.post(
        "/system/gpu",
        data=json.dumps({"gpu_index": 9}),
        headers=csrf_headers(csrf_token),
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["code"] == "INVALID_GPU_INDEX"
    assert payload["available_devices"] == DEVICES
