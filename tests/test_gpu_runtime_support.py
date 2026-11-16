"""An adapter being present is not the same as this build being able to run it.

Issue #7: an RTX 5070 (compute capability 12.0) was listed by nvidia-smi, shown
as healthy in Settings, and then failed every job with

    GPU index 0 is not available. Available CUDA devices: 0: NVIDIA GeForce RTX 5070.

The message denied the availability of the index it was listing. The real cause
was a PyTorch build with no sm_120 kernel image, which is a different problem
with a different fix (install a cu128+ wheel, not pick another device).
"""

from __future__ import annotations

import types

import pytest

from opencut import gpu_lanes
from opencut.gpu import (
    GPU_SUPPORT_UNKNOWN,
    GPU_SUPPORT_UNSUPPORTED,
    GPU_SUPPORT_USABLE,
    GPUSelectionError,
    GPUUnsupportedBuildError,
    activate_selected_gpu,
    gpu_runtime_support,
    torch_supported_capabilities,
)

BLACKWELL = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "compute_capability": "12.0"}
ADA = {"index": 0, "name": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": "8.9"}


def _torch_stub(arch_list, *, set_device=None, device_count=1):
    """A torch double exposing only what the GPU helpers touch."""
    calls = []

    def _set_device(index):
        calls.append(index)
        if set_device is not None:
            set_device(index)

    cuda = types.SimpleNamespace(
        get_arch_list=lambda: list(arch_list),
        is_available=lambda: True,
        device_count=lambda: device_count,
        set_device=_set_device,
    )
    module = types.SimpleNamespace(cuda=cuda)
    module.set_device_calls = calls
    return module


# ---------------------------------------------------------------------------
# Capability parsing
# ---------------------------------------------------------------------------

def test_arch_list_counts_only_compiled_kernels_not_ptx():
    """compute_90 is PTX that JITs; it is exactly what fails on Blackwell."""
    module = _torch_stub(["sm_80", "sm_86", "sm_90", "compute_90"])
    assert torch_supported_capabilities(module) == {(8, 0), (8, 6), (9, 0)}


def test_three_digit_arch_parses_as_major_minor():
    module = _torch_stub(["sm_120"])
    assert torch_supported_capabilities(module) == {(12, 0)}


# ---------------------------------------------------------------------------
# Support resolution
# ---------------------------------------------------------------------------

def test_blackwell_on_a_cu121_build_is_unsupported_not_missing():
    module = _torch_stub(["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
    support = gpu_runtime_support(BLACKWELL, torch_module=module)
    assert support["state"] == GPU_SUPPORT_UNSUPPORTED
    assert support["compute_capability"] == "12.0"
    assert "no kernel image" in support["reason"]
    # The advice has to name the wheel, not a different device.
    assert "cu128" in support["required_build"]


def test_supported_adapter_is_usable():
    module = _torch_stub(["sm_86", "sm_89", "sm_90"])
    support = gpu_runtime_support(ADA, torch_module=module)
    assert support["state"] == GPU_SUPPORT_USABLE
    assert support["reason"] == ""


def test_unknown_when_the_runtime_reports_no_arch_list():
    module = _torch_stub([])
    assert gpu_runtime_support(BLACKWELL, torch_module=module)["state"] == GPU_SUPPORT_UNKNOWN


def test_unknown_when_the_device_carries_no_capability():
    module = _torch_stub(["sm_90"])
    bare = {"index": 0, "name": "NVIDIA GeForce RTX 5070"}
    assert gpu_runtime_support(bare, torch_module=module)["state"] == GPU_SUPPORT_UNKNOWN


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------

def test_activate_refuses_an_unsupported_adapter_without_denying_it_exists(monkeypatch):
    monkeypatch.setattr("opencut.gpu._configured_gpu_index", lambda: 0)
    module = _torch_stub(["sm_80", "sm_86", "sm_90"])

    with pytest.raises(GPUUnsupportedBuildError) as excinfo:
        activate_selected_gpu(torch_module=module, devices=[dict(BLACKWELL)])

    message = str(excinfo.value)
    assert "is not available" not in message, (
        "the issue #7 message is back: it denies an index it also lists"
    )
    assert "RTX 5070" in message and "12.0" in message
    assert "cu128" in message
    # It must never have touched CUDA to find this out.
    assert module.set_device_calls == []


def test_set_device_failure_reports_the_underlying_error(monkeypatch):
    monkeypatch.setattr("opencut.gpu._configured_gpu_index", lambda: 0)

    def _boom(index):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    # Arch list claims support, so only set_device can reveal the problem.
    module = _torch_stub(["sm_120"], set_device=_boom)

    with pytest.raises(GPUUnsupportedBuildError) as excinfo:
        activate_selected_gpu(torch_module=module, devices=[dict(BLACKWELL)])

    message = str(excinfo.value)
    assert "no kernel image is available" in message, "the real cause was discarded"
    assert "is not available" not in message


def test_a_genuinely_absent_index_still_reports_unavailable(monkeypatch):
    """The original error is still correct when the index really is missing."""
    monkeypatch.setattr("opencut.gpu._configured_gpu_index", lambda: 3)
    module = _torch_stub(["sm_89"], device_count=1)

    with pytest.raises(GPUSelectionError) as excinfo:
        activate_selected_gpu(torch_module=module, devices=[dict(ADA)])

    assert not isinstance(excinfo.value, GPUUnsupportedBuildError)
    assert "is not available" in str(excinfo.value)


def test_unsupported_error_serializes_the_fix_for_the_panel():
    error = GPUUnsupportedBuildError(
        0,
        [dict(BLACKWELL)],
        reason="the installed PyTorch build has no kernel image for compute capability 12.0",
        required_build="install cu128",
    )
    body = error.to_dict()
    assert body["code"] == "GPU_BUILD_UNSUPPORTED"
    assert body["requested_index"] == 0
    assert body["required_build"] == "install cu128"
    assert body["suggestion"] == "install cu128"
    assert body["available_devices"][0]["name"] == "NVIDIA GeForce RTX 5070"


# ---------------------------------------------------------------------------
# Discovery carries capability on both paths
# ---------------------------------------------------------------------------

def test_nvidia_smi_devices_carry_a_compute_capability(monkeypatch):
    """The nvidia-smi path set no capability at all, so nothing could grade it."""
    import subprocess

    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_lanes.shutil, "which", lambda name: "nvidia-smi")

    captured = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA GeForce RTX 5070, 12282, 11000, 580.00, 12.0\n",
            stderr="",
        )

    monkeypatch.setattr(gpu_lanes.subprocess, "run", _fake_run)

    devices = gpu_module.list_gpu_devices()
    assert len(devices) == 1
    assert devices[0]["compute_capability"] == "12.0"
    assert devices[0]["name"] == "NVIDIA GeForce RTX 5070"
    assert devices[0]["memory_total_mb"] == pytest.approx(12282.0)
    assert devices[0]["driver_version"] == "580.00"
    assert "compute_cap" in " ".join(captured["command"])
    assert "runtime_support" in devices[0]


def test_nvidia_smi_falls_back_when_the_driver_has_no_compute_cap(monkeypatch):
    """Drivers before 510 reject the field; the listing must survive."""
    import subprocess

    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_lanes.shutil, "which", lambda name: "nvidia-smi")
    attempts = []

    def _fake_run(command, **kwargs):
        query = next(part for part in command if part.startswith("--query-gpu="))
        attempts.append(query)
        if "compute_cap" in query:
            return subprocess.CompletedProcess(command, 6, stdout="", stderr="Field not supported")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA GeForce GTX 1080, 8192, 7000, 470.00\n",
            stderr="",
        )

    monkeypatch.setattr(gpu_lanes.subprocess, "run", _fake_run)

    devices = gpu_module.list_gpu_devices()
    assert len(attempts) == 2
    assert len(devices) == 1
    assert devices[0]["name"] == "NVIDIA GeForce GTX 1080"
    assert devices[0]["driver_version"] == "470.00"
    assert "compute_capability" not in devices[0]


def test_device_names_containing_commas_still_parse(monkeypatch):
    import subprocess

    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_lanes.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        gpu_lanes.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA RTX A4000, Laptop GPU, 8192, 7000, 555.00, 8.6\n",
            stderr="",
        ),
    )
    devices = gpu_module.list_gpu_devices()
    # Each CSV field is stripped before rejoining, so the rebuilt name loses the
    # separator's trailing space. That is pre-existing behaviour, asserted here
    # so the comma-splitting stays covered while the field count changes.
    assert devices[0]["name"] == "NVIDIA RTX A4000,Laptop GPU"
    assert devices[0]["compute_capability"] == "8.6"


# ---------------------------------------------------------------------------
# Regressions found by adversarial review of the first fix
# ---------------------------------------------------------------------------

def test_architecture_conditional_entries_are_understood():
    """PyTorch emits sm_90a / sm_120a for Hopper and Blackwell.

    Dropping them left a half-parsed set that still looked authoritative, so a
    working H100 or RTX 5090 was graded unsupported -- worse than the original
    bug.
    """
    module = _torch_stub(["sm_75", "sm_80", "sm_86", "sm_90a", "sm_100a", "sm_120a"])
    supported = torch_supported_capabilities(module)
    assert (9, 0) in supported
    assert (12, 0) in supported
    assert (10, 0) in supported


def test_a_supported_blackwell_build_grades_the_card_usable():
    module = _torch_stub(["sm_90a", "sm_120a"])
    assert gpu_runtime_support(BLACKWELL, torch_module=module)["state"] == GPU_SUPPORT_USABLE


def test_an_unrecognised_arch_entry_refuses_to_grade():
    """Half-understanding the list must not produce a confident verdict."""
    module = _torch_stub(["sm_90", "totally_new_format"])
    assert gpu_runtime_support(ADA, torch_module=module)["state"] == GPU_SUPPORT_UNKNOWN


def test_auto_selection_skips_an_unsupported_adapter(monkeypatch):
    """Index 0 unsupported, index 1 usable: pick 1, not 0."""
    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "_configured_gpu_index", lambda: None)
    monkeypatch.setattr(gpu_module, "torch_supported_capabilities", lambda *a, **k: {(8, 9)})

    devices = [dict(BLACKWELL), {"index": 1, "name": "RTX 4070 SUPER", "compute_capability": "8.9"}]
    status = gpu_module.gpu_selection_status(devices)
    assert status["selected_index"] == 1
    assert status["device"] == "cuda:1"
    assert status["selection_error"] is None


def test_an_unsupported_only_machine_selects_no_device(monkeypatch):
    """A device the build cannot run is not a selection."""
    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_module, "_configured_gpu_index", lambda: None)
    monkeypatch.setattr(gpu_module, "torch_supported_capabilities", lambda *a, **k: {(8, 9)})

    status = gpu_module.gpu_selection_status([dict(BLACKWELL)])
    assert status["selection_error"]["code"] == "GPU_BUILD_UNSUPPORTED"
    assert status["selected_index"] is None, (
        "get_device_index() would hand cuda:0 to callers that never read selection_error"
    )
    assert status["device"] == "cpu"


# ---------------------------------------------------------------------------
# An adapter the PyTorch build cannot run must not fail work that never asks
# PyTorch for it. The first round of fixes made the issue #7 message accurate,
# but every job still failed: the job boundary activated the GPU strictly for
# preview-frame (FFmpeg only) as well, faster-whisper (CTranslate2 kernels)
# was vetoed by torch's arch list, and the CTranslate2/ONNX helpers rebuilt the
# verdict as the original "GPU index 0 is not available" text.
# ---------------------------------------------------------------------------

CU126_ARCHES = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def _blackwell_machine(monkeypatch, *, configured=None, extra_devices=()):
    """nvidia-smi lists an RTX 5070; the installed torch tops out at sm_90."""
    from opencut import gpu as gpu_module

    module = _torch_stub(CU126_ARCHES, device_count=1 + len(extra_devices))
    monkeypatch.setattr(gpu_module, "torch", module)
    monkeypatch.setattr(gpu_module, "_HAS_TORCH", True)
    monkeypatch.setattr(gpu_module, "_configured_gpu_index", lambda: configured)
    devices = [dict(BLACKWELL), *[dict(device) for device in extra_devices]]
    monkeypatch.setattr(
        gpu_module,
        "list_gpu_devices",
        lambda: [gpu_module._annotate_gpu_device(device) for device in devices],
    )
    return gpu_module, module


def test_auto_mode_runs_torch_on_cpu_instead_of_failing(monkeypatch):
    gpu_module, module = _blackwell_machine(monkeypatch)

    assert gpu_module.activate_selected_gpu() is None
    assert module.set_device_calls == []
    assert gpu_module.get_device() == "cpu"


def test_auto_mode_activates_the_adapter_status_selects(monkeypatch):
    """Status skipped the unrunnable index 0; activation used to hardcode it."""
    gpu_module, module = _blackwell_machine(
        monkeypatch,
        extra_devices=[{"index": 1, "name": "NVIDIA GeForce RTX 3080", "compute_capability": "8.6"}],
    )

    assert gpu_module.gpu_selection_status()["selected_index"] == 1
    assert gpu_module.activate_selected_gpu() == 1
    assert module.set_device_calls == [1]


def test_auto_mode_set_device_failure_falls_back_to_cpu(monkeypatch):
    from opencut import gpu as gpu_module

    def _boom(index):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    module = _torch_stub(["sm_120"], set_device=_boom)
    monkeypatch.setattr(gpu_module, "_configured_gpu_index", lambda: None)

    assert gpu_module.activate_selected_gpu(torch_module=module, devices=[dict(BLACKWELL)]) is None


def test_manual_mode_still_names_the_wheel_for_torch_work(monkeypatch):
    gpu_module, _module = _blackwell_machine(monkeypatch, configured=0)

    with pytest.raises(GPUUnsupportedBuildError):
        gpu_module.get_device()


@pytest.mark.parametrize("configured, expected", [
    (None, {"device": "cuda", "device_index": 0}),
    (0, {"device": "cuda", "device_index": 0}),
])
def test_ctranslate2_ignores_a_torch_only_verdict(monkeypatch, configured, expected):
    gpu_module, _module = _blackwell_machine(monkeypatch, configured=configured)

    assert gpu_module.selected_ct2_device_kwargs() == expected


@pytest.mark.parametrize("configured, expected", [
    (None, [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]),
    (0, [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]),
])
def test_onnx_ignores_a_torch_only_verdict(monkeypatch, configured, expected):
    gpu_module, _module = _blackwell_machine(monkeypatch, configured=configured)
    monkeypatch.setattr(gpu_module, "onnx_cuda_provider_available", lambda: True)

    assert gpu_module.selected_onnx_providers() == expected


def test_a_missing_index_still_fails_ctranslate2_and_onnx(monkeypatch):
    gpu_module, _module = _blackwell_machine(monkeypatch, configured=3)

    with pytest.raises(GPUSelectionError) as excinfo:
        gpu_module.selected_ct2_device_kwargs()
    assert excinfo.value.code == "INVALID_GPU_INDEX"
    with pytest.raises(GPUSelectionError):
        gpu_module.selected_onnx_providers()


@pytest.mark.parametrize("configured", [None, 0])
def test_cpu_only_job_completes_on_an_unrunnable_adapter(monkeypatch, configured):
    """The reporter's log shows preview-frame, an FFmpeg job, dying on this."""
    from concurrent.futures import Future

    from flask import Flask

    from opencut.jobs import _get_job_copy, async_job

    _blackwell_machine(monkeypatch, configured=configured)

    class InlinePool:
        def submit(self, job_id, fn):
            future = Future()
            fn()
            future.set_result(None)
            return future

    monkeypatch.setattr("opencut.workers.get_pool", lambda: InlinePool())
    monkeypatch.setattr("opencut.jobs._persist_job", lambda *args, **kwargs: None)

    app = Flask(__name__)

    @app.route("/cpu-only-job", methods=["POST"])
    @async_job("cpu-only-job", filepath_required=False)
    def cpu_only_job(job_id, filepath, data):
        return {"frames": 1}

    response = app.test_client().post("/cpu-only-job", json={})
    assert response.status_code == 200
    job = _get_job_copy(response.get_json()["job_id"])
    assert job["status"] == "complete", job.get("error")


@pytest.mark.parametrize("configured, expected_index", [(None, None), (0, 0)])
def test_faster_whisper_keeps_the_gpu_when_only_torch_lacks_kernels(
    monkeypatch, configured, expected_index,
):
    import sys

    import opencut.core.captions as captions
    import opencut.user_data as user_data
    from opencut.utils.config import CaptionConfig

    gpu_module, module = _blackwell_machine(monkeypatch, configured=configured)
    module.cuda.empty_cache = lambda: None
    monkeypatch.setitem(sys.modules, "torch", module)
    # CPU mode on would skip the GPU before the code under test runs.
    monkeypatch.setattr(user_data, "load_whisper_settings", lambda: {"cpu_mode": False})
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            calls.append(kwargs)

        def transcribe(self, *_args, **_kwargs):
            segment = types.SimpleNamespace(
                text="hello", start=0.0, end=1.0, words=[],
                avg_logprob=0.0, no_speech_prob=0.0,
            )
            return [segment], types.SimpleNamespace(language="en", language_probability=1.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    captions._transcribe_faster_whisper("audio.wav", CaptionConfig(model="base", word_timestamps=False))

    assert calls[0]["device"] == "cuda"
    assert calls[0]["compute_type"] == "float16"
    assert calls[0].get("device_index") == expected_index


def _host_gpus(monkeypatch, *gpus):
    """Pin the adapters the host lane is chosen from, read fresh."""
    monkeypatch.setattr(gpu_lanes, "query_nvidia_gpus", lambda timeout=5: [dict(g) for g in gpus])
    monkeypatch.setattr(gpu_lanes, "_HOST_PLAN", None)


def test_a_raw_cuda_arch_error_gets_the_install_command(monkeypatch):
    """Modules that call torch.cuda directly fail inside the model instead."""
    from opencut.core.install_hints import suggestion_for_exception

    _host_gpus(monkeypatch, {**BLACKWELL, "driver_version": "572.16"})
    error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
    suggestion = suggestion_for_exception(error, context="upscale")
    # Driver 572 can't run CUDA 13, so the host lane is cu128, not the default.
    assert "whl/cu128" in suggestion
    assert suggestion_for_exception(RuntimeError("disk full"), context="upscale") == ""


def test_the_install_command_follows_the_host_lane(monkeypatch):
    from opencut import gpu as gpu_module

    _host_gpus(monkeypatch, {**BLACKWELL, "driver_version": "581.57"})
    assert gpu_module.torch_gpu_install_command().endswith("whl/cu132")

    _host_gpus(monkeypatch, {**ADA, "driver_version": "546.33"})
    assert gpu_module.torch_gpu_install_command().endswith("whl/cu126")

    # No NVIDIA adapter at all still gets the documented lane, not "cpu".
    _host_gpus(monkeypatch)
    assert gpu_module.torch_gpu_install_command().endswith(f"whl/{gpu_module.CUDA_WHEEL_INDEX}")


def test_an_unrunnable_adapter_on_a_current_driver_is_sent_to_cu132():
    module = _torch_stub(["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
    support = gpu_runtime_support({**BLACKWELL, "driver_version": "581.57"}, torch_module=module)
    assert support["required_build"].endswith("whl/cu132")


def test_packaged_builds_get_an_install_command_that_reaches_them(monkeypatch):
    """The frozen server ignores system site-packages, so plain pip is a no-op."""
    import sys

    from opencut import gpu as gpu_module

    _host_gpus(monkeypatch, {**BLACKWELL, "driver_version": "572.16"})
    assert gpu_module.torch_gpu_install_command().startswith("pip install torch")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_module.os, "name", "nt")
    command = gpu_module.torch_gpu_install_command()
    major, minor = sys.version_info[:2]
    assert command.startswith(f"py -{major}.{minor} -m pip install")
    assert r"%USERPROFILE%\.opencut\packages" in command
    assert "cu128" in command
    hint = gpu_module.gpu_runtime_support(
        BLACKWELL, torch_module=_torch_stub(CU126_ARCHES)
    )["required_build"]
    assert "--target" in hint
