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

    monkeypatch.setattr(gpu_module.shutil, "which", lambda name: "nvidia-smi")

    captured = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="0, NVIDIA GeForce RTX 5070, 12282, 11000, 580.00, 12.0\n",
            stderr="",
        )

    monkeypatch.setattr(gpu_module.subprocess, "run", _fake_run)

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

    monkeypatch.setattr(gpu_module.shutil, "which", lambda name: "nvidia-smi")
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

    monkeypatch.setattr(gpu_module.subprocess, "run", _fake_run)

    devices = gpu_module.list_gpu_devices()
    assert len(attempts) == 2
    assert len(devices) == 1
    assert devices[0]["name"] == "NVIDIA GeForce GTX 1080"
    assert devices[0]["driver_version"] == "470.00"
    assert "compute_capability" not in devices[0]


def test_device_names_containing_commas_still_parse(monkeypatch):
    import subprocess

    from opencut import gpu as gpu_module

    monkeypatch.setattr(gpu_module.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        gpu_module.subprocess,
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
