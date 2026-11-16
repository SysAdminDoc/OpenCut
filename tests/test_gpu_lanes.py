"""The per-backend CUDA matrix and the PyTorch lane each host gets (F465, issue #7).

nvidia-smi is faked, so these cover the decision table, not real hardware.
Facts the expectations come from: CUDA 13 needs driver 580 or newer and
compute capability 7.5 or newer, CUDA 12 needs driver 525 or newer, the
pinned CTranslate2 4.8.1 is a CUDA 12.8 build and onnxruntime-gpu 1.27 and
newer are CUDA 13 builds.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest

from opencut import gpu_lanes

RTX_5090_581 = "0, NVIDIA GeForce RTX 5090, 32607, 31000, 581.57, 12.0\n"
RTX_5090_572 = "0, NVIDIA GeForce RTX 5090, 32607, 31000, 572.16, 12.0\n"
RTX_3080_546 = "0, NVIDIA GeForce RTX 3080, 10240, 9000, 546.33, 8.6\n"
GTX_1080_581 = "0, NVIDIA GeForce GTX 1080, 8192, 7000, 581.57, 6.1\n"
GTX_1080_470 = "0, NVIDIA GeForce GTX 1080, 8192, 7000, 470.00, 6.1\n"


def _fake_smi(monkeypatch, stdout, *, present=True, returncode=0):
    calls = []

    def _run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(gpu_lanes.shutil, "which", lambda name: "nvidia-smi" if present else None)
    monkeypatch.setattr(gpu_lanes.subprocess, "run", _run)
    return calls


def _host_lane(monkeypatch, stdout, **kwargs):
    calls = _fake_smi(monkeypatch, stdout, **kwargs)
    monkeypatch.setattr(gpu_lanes, "_HOST_PLAN", None)
    return gpu_lanes.host_torch_lane(), calls


def test_rtx_5090_on_a_580_driver_gets_the_cuda_13_lane(monkeypatch):
    lane, calls = _host_lane(monkeypatch, RTX_5090_581)

    assert lane.index == "cu132"
    assert lane.cuda_major == 13
    assert lane.index_url == "https://download.pytorch.org/whl/cu132"
    assert lane.driver_version == "581.57" and lane.capability == "12.0"
    assert "581.57" in lane.reason
    # The query must not flash a console window from a windowed build.
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert calls[0]["creationflags"] == subprocess.CREATE_NO_WINDOW


def test_rtx_3080_on_an_older_driver_stays_on_cuda_12_and_says_why(monkeypatch):
    lane, _calls = _host_lane(monkeypatch, RTX_3080_546)

    assert lane.index == "cu126"
    assert lane.cuda_major == 12
    assert "546.33 is older than 580" in lane.reason
    assert "Update the driver to 580" in lane.reason


def test_no_nvidia_gpu_stays_on_cpu(monkeypatch):
    lane, calls = _host_lane(monkeypatch, "", present=False)

    assert lane.index == "cpu"
    assert lane.cuda_major is None
    assert "No NVIDIA GPU" in lane.reason
    assert calls == []


def test_a_failing_nvidia_smi_counts_as_no_gpu(monkeypatch):
    lane, calls = _host_lane(monkeypatch, "", returncode=9)

    assert lane.index == "cpu"
    assert len(calls) == 2  # with and without compute_cap


def test_rtx_5090_below_driver_580_gets_cu128_not_cu126(monkeypatch):
    lane, _calls = _host_lane(monkeypatch, RTX_5090_572)

    # cu126 has no sm_120 kernels, so Blackwell on CUDA 12 needs cu128.
    assert lane.index == "cu128"
    assert "572.16 is older than 580" in lane.reason
    assert "2.11.0" in lane.reason


def test_pascal_on_a_new_driver_stays_on_cuda_12(monkeypatch):
    lane, _calls = _host_lane(monkeypatch, GTX_1080_581)

    assert lane.index == "cu126"
    assert "older than Turing" in lane.reason
    assert "Update the driver" not in lane.reason


def test_a_driver_too_old_for_cuda_12_runs_on_cpu(monkeypatch):
    lane, _calls = _host_lane(monkeypatch, GTX_1080_470)

    assert lane.index == "cpu"
    assert "470.00 is older than 525" in lane.reason


def test_mixed_adapters_take_the_lane_every_one_can_run():
    lane = gpu_lanes.choose_torch_lane([
        {"name": "NVIDIA GeForce RTX 5090", "driver_version": "581.57", "compute_capability": "12.0"},
        {"name": "Quadro GV100", "driver_version": "581.57", "compute_capability": "7.0"},
    ])
    assert lane.index == "cu128"


def test_a_blackwell_name_stands_in_for_a_missing_capability():
    lane = gpu_lanes.choose_torch_lane([
        {"name": "NVIDIA GeForce RTX 5070", "driver_version": "572.16", "compute_capability": ""},
    ])
    assert lane.index == "cu128"


def test_the_backend_table_records_each_cuda_major():
    table = gpu_lanes.BACKEND_CUDA
    assert table["ctranslate2"].cuda_major == 12
    assert table["onnxruntime-gpu"].cuda_major == 13
    assert table["torch"].cuda_major == gpu_lanes.TORCH_LANES[gpu_lanes.DEFAULT_TORCH_INDEX].cuda_major == 13
    assert gpu_lanes.CUDA_DRIVER_FLOOR == {12: 525, 13: 580}


def test_a_cuda_13_lane_isolates_ctranslate2_and_takes_the_cuda_13_onnxruntime(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCUT_HOME", str(tmp_path))
    lane = gpu_lanes.choose_torch_lane([
        {"name": "RTX 5090", "driver_version": "581.57", "compute_capability": "12.0"},
    ])
    plan = gpu_lanes.plan_backends(lane)

    assert plan["torch"]["index"] == "cu132"
    assert plan["ctranslate2"]["separate_process"] is True
    command = plan["ctranslate2"]["runtime_command"]
    assert "--target" in command and "cuda12" in command
    assert '"nvidia-cudnn-cu12==9.10.2.21"' in command
    assert plan["onnxruntime"]["requirement"] == "onnxruntime-gpu>=1.27,<2"


def test_a_cuda_12_lane_keeps_ctranslate2_in_process_and_pins_a_cuda_12_onnxruntime():
    lane = gpu_lanes.choose_torch_lane([
        {"name": "RTX 3080", "driver_version": "546.33", "compute_capability": "8.6"},
    ])
    plan = gpu_lanes.plan_backends(lane)

    assert plan["ctranslate2"]["separate_process"] is False
    assert plan["ctranslate2"]["runtime_command"] == ""
    assert plan["onnxruntime"]["requirement"] == "onnxruntime-gpu>=1.21,<1.27"


def test_frozen_install_commands_target_the_packages_folder(monkeypatch):
    monkeypatch.setattr(gpu_lanes.os, "name", "nt")
    command = gpu_lanes.torch_install_command("cu126", frozen=True)
    assert r'--target "%USERPROFILE%\.opencut\packages" torch' in command
    assert command.endswith("whl/cu126")
    assert gpu_lanes.torch_install_command("cu126", frozen=False) == (
        "pip install torch --index-url https://download.pytorch.org/whl/cu126"
    )


@pytest.mark.parametrize("stdout, expected", [
    (RTX_5090_581, "cu132"),
    (RTX_3080_546, "cu126"),
    ("", "cpu"),
])
def test_install_py_prints_the_lane_and_its_reason(monkeypatch, capsys, tmp_path, stdout, expected):
    monkeypatch.setenv("OPENCUT_HOME", str(tmp_path))
    _fake_smi(monkeypatch, stdout, present=bool(stdout))
    installer = importlib.import_module("install")

    installer.print_gpu_lane()
    out = capsys.readouterr().out

    assert f"PyTorch lane: {expected}." in out
    if expected == "cpu":
        assert "For GPU" not in out
    else:
        assert f"--index-url https://download.pytorch.org/whl/{expected}" in out
    assert ("nvidia-cudnn-cu12==9.10.2.21" in out) is (expected == "cu132")
