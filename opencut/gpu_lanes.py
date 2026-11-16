"""Which CUDA build each GPU backend needs, and which PyTorch index fits this host (F465).

OpenCut drives three GPU backends and they don't agree on CUDA:

* PyTorch publishes one wheel index per CUDA release. ``cu132`` (CUDA 13.2)
  carries current releases (2.12.0 through 2.15.0 for cp313 on Windows).
  ``cu128`` stopped at 2.11.0, and ``cu126`` is the CUDA 12 lane that still
  gets new releases and still has Maxwell, Pascal and Volta kernels.
* CTranslate2 (faster-whisper) 4.8.1 is built against CUDA 12.8 and cuDNN
  9.10. It loads ``cublas64_12.dll`` by name and bundles only the
  ``cudnn64_9.dll`` shim, so the rest of its runtime has to come from
  somewhere else in the process.
* onnxruntime-gpu 1.27 and newer are CUDA 13 builds. 1.21 through 1.26 are
  CUDA 12 builds.

CUDA 13 needs NVIDIA driver 580 or newer and a Turing (compute capability
7.5) or newer GPU. CUDA 12 needs driver 525 or newer.

The PyTorch lane decides the process's CUDA major, so CTranslate2 runs in a
child process with its own CUDA 12 runtime whenever torch is a CUDA 13 build
(:mod:`opencut.core.ct2_isolation`). cuDNN's DLLs carry the same file names
in both CUDA majors, so one process can't hold both.

Stdlib only: ``install.py`` and the hint paths use this before torch exists.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger("opencut")

#: Oldest NVIDIA driver branch that runs each CUDA major (minor version
#: compatibility, CUDA Toolkit release notes).
CUDA_DRIVER_FLOOR = {12: 525, 13: 580}

PYTORCH_INDEX_ROOT = "https://download.pytorch.org/whl/"


@dataclass(frozen=True)
class TorchLane:
    """One PyTorch wheel index OpenCut can install from."""

    index: str
    cuda_major: Optional[int]
    min_capability: Optional[tuple]
    max_capability: Optional[tuple]
    newest_torch: str


#: The PyTorch indexes OpenCut picks between, newest CUDA first. Capability
#: ranges come from each index's ``torch.cuda.get_arch_list()``. cu128 also
#: dropped sm_50 through sm_60 in torch 2.8, which is why cu126 stays the
#: lane for older cards.
TORCH_LANES = {
    "cu132": TorchLane("cu132", 13, (7, 5), (12, 0), "2.15.0"),
    "cu128": TorchLane("cu128", 12, (7, 0), (12, 0), "2.11.0"),
    "cu126": TorchLane("cu126", 12, (5, 0), (9, 0), "2.14.1"),
    "cpu": TorchLane("cpu", None, None, None, ""),
}

#: The lane a host with a current driver and a Turing or newer GPU gets.
DEFAULT_TORCH_INDEX = "cu132"


@dataclass(frozen=True)
class BackendCuda:
    """The CUDA build one GPU backend's pinned release links against."""

    backend: str
    requirement: str
    cuda_major: int
    runtime_dlls: tuple
    note: str


#: One row per GPU backend: the CUDA major its pinned build needs.
BACKEND_CUDA = {
    "torch": BackendCuda(
        "torch",
        "torch>=2.10",
        13,
        ("cublas64_13.dll", "cudnn64_9.dll"),
        "cu132 by default. choose_torch_lane moves a host to cu128 or cu126 (CUDA 12) "
        "when its driver is older than 580 or its GPU is older than Turing.",
    ),
    "ctranslate2": BackendCuda(
        "ctranslate2",
        "ctranslate2==4.8.1",
        12,
        ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"),
        "Windows wheels are built with CUDA 12.8 and cuDNN 9.10.2. Runs in a child "
        "process beside a CUDA 13 torch.",
    ),
    "onnxruntime-gpu": BackendCuda(
        "onnxruntime-gpu",
        "onnxruntime-gpu>=1.27,<2",
        13,
        ("cublas64_13.dll", "cudnn64_9.dll"),
        "1.21 through 1.26 are the CUDA 12 builds a CUDA 12 lane needs.",
    ),
}

#: onnxruntime-gpu requirement per CUDA major.
ONNXRUNTIME_GPU_REQUIREMENT = {13: "onnxruntime-gpu>=1.27,<2", 12: "onnxruntime-gpu>=1.21,<1.27"}

#: CTranslate2's CUDA 12 runtime, installed into its own folder so it never
#: shares ``nvidia/cudnn/bin`` with a CUDA 13 cuDNN. cuDNN is pinned to the
#: shim CTranslate2 4.8.1 bundles because cuDNN 9 refuses mismatched
#: sub-libraries. nvidia-cublas-cu12 12.9.2.10 is the newest with a Windows wheel.
CT2_CUDA12_RUNTIME = ("nvidia-cublas-cu12>=12.8.3,<12.10", "nvidia-cudnn-cu12==9.10.2.21")

_BLACKWELL_NAME_RE = re.compile(r"\brtx\s*50\d{2}\b", re.IGNORECASE)


def parse_capability(value) -> Optional[tuple]:
    """Return ``(major, minor)`` from a capability in any of its shapes."""
    if isinstance(value, (tuple, list)) and value:
        value = ".".join(str(part) for part in value[:2])
    match = re.search(r"(\d+)(?:\.(\d+))?", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def parse_driver_version(value) -> Optional[tuple]:
    """``"581.57"`` becomes ``(581, 57)``. Anything unreadable is None."""
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?(?:\.\d+)*\s*", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _label(capability) -> str:
    return f"{capability[0]}.{capability[1]}" if capability else ""


def torch_index_url(index: str) -> str:
    return f"{PYTORCH_INDEX_ROOT}{index}"


def _no_window() -> dict:
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flag} if flag else {}


def query_nvidia_gpus(timeout: float = 5) -> list:
    """Every NVIDIA adapter ``nvidia-smi`` lists, unannotated.

    Each record has ``index``, ``name``, ``memory_total_mb``,
    ``memory_free_mb``, ``driver_version`` and, on drivers that know the
    field (510 and newer), ``compute_capability``. Empty when nvidia-smi is
    missing or fails. The window flag keeps a console from flashing up when a
    windowed build polls the GPU.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    for fields, has_capability in (
        ("index,name,memory.total,memory.free,driver_version,compute_cap", True),
        ("index,name,memory.total,memory.free,driver_version", False),
    ):
        try:
            result = subprocess.run(
                [nvidia_smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                **_no_window(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("nvidia-smi GPU query failed: %s", exc)
            return []
        if result.returncode != 0:
            continue
        trailing = 4 if has_capability else 3
        devices = []
        stdout = result.stdout if isinstance(result.stdout, str) else ""
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < trailing + 2:
                continue
            try:
                index = int(parts[0])
            except (TypeError, ValueError):
                continue
            # GPU names can contain commas. The trailing fields are fixed.
            record = {
                "index": index,
                "name": ",".join(parts[1:-trailing]).strip(),
                "memory_total_mb": _float(parts[-trailing]),
                "memory_free_mb": _float(parts[-trailing + 1]),
                "driver_version": parts[-trailing + 2],
            }
            if has_capability:
                record["compute_capability"] = parts[-1]
            devices.append(record)
        if devices:
            return sorted(devices, key=lambda device: device["index"])
    return []


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class TorchLanePlan:
    """The PyTorch index chosen for a host, and why."""

    index: str
    reason: str
    driver_version: str = ""
    capability: str = ""
    gpu_name: str = ""

    @property
    def cuda_major(self) -> Optional[int]:
        return TORCH_LANES[self.index].cuda_major

    @property
    def index_url(self) -> str:
        return torch_index_url(self.index)

    def to_dict(self) -> dict:
        body = asdict(self)
        body.update({"cuda_major": self.cuda_major, "index_url": self.index_url})
        return body


def choose_torch_lane(gpus) -> TorchLanePlan:
    """Pick the PyTorch index for these NVIDIA adapters.

    ``gpus`` are records shaped like :func:`query_nvidia_gpus` output. The
    newest CUDA lane wins when the driver and every adapter allow it. Each
    step down says what held it back.
    """
    gpus = [gpu for gpu in (gpus or []) if isinstance(gpu, dict)]
    if not gpus:
        return TorchLanePlan("cpu", "No NVIDIA GPU was found, so PyTorch stays on the CPU build.")

    names = ", ".join(dict.fromkeys(str(gpu.get("name") or "NVIDIA GPU") for gpu in gpus))
    driver_text = ""
    driver = None
    for gpu in gpus:
        driver = parse_driver_version(gpu.get("driver_version"))
        if driver:
            driver_text = str(gpu.get("driver_version")).strip()
            break
    capabilities = []
    for gpu in gpus:
        capability = parse_capability(gpu.get("compute_capability", gpu.get("compute_cap")))
        if capability is None and _BLACKWELL_NAME_RE.search(str(gpu.get("name") or "")):
            capability = (12, 0)
        if capability is not None:
            capabilities.append(capability)
    lowest = min(capabilities) if capabilities else None
    highest = max(capabilities) if capabilities else None

    def plan(index: str, reason: str) -> TorchLanePlan:
        return TorchLanePlan(index, reason, driver_text, _label(lowest), names)

    floor12, floor13 = CUDA_DRIVER_FLOOR[12], CUDA_DRIVER_FLOOR[13]
    if driver is not None and driver[0] < floor12:
        return plan(
            "cpu",
            f"NVIDIA driver {driver_text} is older than {floor12}, the oldest that runs a "
            f"CUDA 12 PyTorch build. Update the driver to {floor13} or newer to use the GPU.",
        )
    oldest_supported = TORCH_LANES["cu126"].min_capability
    if highest is not None and highest < oldest_supported:
        return plan(
            "cpu",
            f"{names} (compute capability {_label(highest)}) is older than any current "
            f"PyTorch CUDA build supports ({_label(oldest_supported)} or newer).",
        )
    if lowest is None:
        return plan(
            "cu126",
            "The GPU's compute capability couldn't be read, so OpenCut picked cu126, "
            "the CUDA 12 build that covers the most GPUs.",
        )

    cuda13 = TORCH_LANES[DEFAULT_TORCH_INDEX]
    driver_allows_13 = driver is not None and driver[0] >= floor13
    if driver_allows_13 and lowest >= cuda13.min_capability:
        return plan(
            DEFAULT_TORCH_INDEX,
            f"Driver {driver_text} and compute capability {_label(lowest)} both meet "
            f"CUDA 13's needs (driver {floor13} or newer, Turing or newer GPU).",
        )

    if driver is None:
        held = (
            "The NVIDIA driver version couldn't be read, so OpenCut can't confirm the "
            f"driver {floor13} that CUDA 13 builds need."
        )
    elif not driver_allows_13:
        held = f"Driver {driver_text} is older than {floor13}, which CUDA 13 builds need."
    else:
        held = (
            f"Compute capability {_label(lowest)} is older than Turing (7.5), "
            "and CUDA 13 dropped those GPUs."
        )
    upgrade = (
        f" Update the driver to {floor13} or newer to move to {DEFAULT_TORCH_INDEX}."
        if not driver_allows_13 and lowest >= cuda13.min_capability
        else ""
    )
    if highest > TORCH_LANES["cu126"].max_capability:
        return plan(
            "cu128",
            f"{held} cu128 is the newest CUDA 12 build with kernels for compute capability "
            f"{_label(highest)}, though PyTorch stopped at "
            f"{TORCH_LANES['cu128'].newest_torch} there.{upgrade}",
        )
    return plan(
        "cu126",
        f"{held} cu126 covers compute capability 5.0 through 9.0 and still gets "
        f"current PyTorch releases.{upgrade}",
    )


_HOST_PLAN: Optional[TorchLanePlan] = None


def host_torch_lane(*, refresh: bool = False) -> TorchLanePlan:
    """The lane for this machine's adapters, read once per process."""
    global _HOST_PLAN
    if _HOST_PLAN is None or refresh:
        _HOST_PLAN = choose_torch_lane(query_nvidia_gpus())
    return _HOST_PLAN


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _frozen_pip() -> str:
    major, minor = sys.version_info[:2]
    return f"py -{major}.{minor} -m pip" if os.name == "nt" else f"python{major}.{minor} -m pip"


def torch_install_command(index: str, *, frozen: Optional[bool] = None) -> str:
    """The torch install command that reaches this server, for one index.

    A packaged server ignores the system interpreter's site-packages and loads
    optional packages from ``~/.opencut/packages`` with its own Python
    version, so a plain ``pip install`` changes nothing for installer users.
    """
    frozen = _is_frozen() if frozen is None else frozen
    url = torch_index_url(index)
    if not frozen:
        return f"pip install torch --index-url {url}"
    target = r'"%USERPROFILE%\.opencut\packages"' if os.name == "nt" else "~/.opencut/packages"
    return f"{_frozen_pip()} install --upgrade --target {target} torch --index-url {url}"


def host_torch_install_command() -> str:
    """:func:`torch_install_command` for this host's lane.

    A host with no usable NVIDIA GPU gets the default CUDA lane, since the
    command only comes up when someone asks how to put torch on a GPU.
    """
    index = host_torch_lane().index
    return torch_install_command(DEFAULT_TORCH_INDEX if index == "cpu" else index)


def cuda12_runtime_dir() -> str:
    """Where CTranslate2's own CUDA 12 runtime lives: ``<data folder>/cuda12``."""
    from opencut.server_log import opencut_data_dir

    return os.path.join(opencut_data_dir(), "cuda12")


def ct2_cuda12_runtime_command(*, frozen: Optional[bool] = None) -> str:
    """The command that installs CTranslate2's CUDA 12 runtime into its folder.

    A packaged server does it itself (:mod:`opencut.core.optional_tools`),
    checking the download before it replaces anything.
    """
    frozen = _is_frozen() if frozen is None else frozen
    if frozen:
        server = "OpenCut-Server.exe" if os.name == "nt" else "OpenCut-Server"
        return f"{server} --install-tools cuda12"
    packages = " ".join(f'"{requirement}"' for requirement in CT2_CUDA12_RUNTIME)
    return f'pip install --upgrade --target "{cuda12_runtime_dir()}" {packages}'


def onnxruntime_gpu_requirement() -> Optional[str]:
    """The onnxruntime-gpu build this machine needs, or None when no lane names one.

    ONNX Runtime loads into the server beside torch, so an installed CUDA
    torch decides the major. Without one, the host's lane does. None on a CPU
    lane, where the caller keeps its own bound.
    """
    from opencut.ct2_isolation import torch_cuda_major

    cuda_major = torch_cuda_major()
    if cuda_major is None:
        cuda_major = host_torch_lane().cuda_major
    return None if cuda_major is None else ONNXRUNTIME_GPU_REQUIREMENT.get(cuda_major)


#: The projects PyTorch builds once per wheel index. PyPI carries one build of
#: each, and on Windows that's the CPU build.
TORCH_FAMILY = frozenset({"torch", "torchvision", "torchaudio"})

_LOCAL_INDEX_RE = re.compile(r"\+(cu\d{2,3}|cpu)\b")


PYPI_INDEX = "pypi"


def installed_torch_index() -> Optional[str]:
    """The index the installed torch came from (``2.15.0+cu132`` is cu132), or None.

    A version with no local tag at all is a PyPI build, :data:`PYPI_INDEX`.
    None means no torch, or a tag that names no index (a source build).
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("torch")
    except PackageNotFoundError:
        return None
    if installed and "+" not in installed:
        return PYPI_INDEX
    match = _LOCAL_INDEX_RE.search(installed or "")
    return match.group(1) if match else None


def torch_family_index(projects) -> Optional[str]:
    """The PyTorch index a pip install of ``projects`` should use, or None.

    None unless every project is torch, torchvision or torchaudio, and on
    macOS, where PyPI carries PyTorch's own builds. torchvision or torchaudio
    without torch follows the installed torch's index, since each release
    pins one torch build and a different one would replace it. Over a PyPI
    torch that's PyPI (None), which keeps a CUDA torchvision off a CPU torch
    on Windows. Otherwise the host's lane decides.
    """
    names = {re.sub(r"[-_.]+", "-", str(project)).lower() for project in projects}
    if not names or not names <= TORCH_FAMILY or sys.platform == "darwin":
        return None
    if "torch" not in names:
        installed = installed_torch_index()
        if installed == PYPI_INDEX:
            return None
        if installed:
            return installed
    return host_torch_lane().index


def torch_family_pip_args(projects) -> list:
    """``--index-url`` for :func:`torch_family_index`, or nothing."""
    index = torch_family_index(projects)
    return [] if index is None else ["--index-url", torch_index_url(index)]


def plan_backends(lane: TorchLanePlan) -> dict:
    """What every GPU backend installs and how it runs under ``lane``."""
    cuda_major = lane.cuda_major
    separate = cuda_major is not None and cuda_major != BACKEND_CUDA["ctranslate2"].cuda_major
    return {
        "torch": lane.to_dict(),
        "ctranslate2": {
            "cuda_major": BACKEND_CUDA["ctranslate2"].cuda_major,
            "separate_process": separate,
            "runtime_command": ct2_cuda12_runtime_command() if separate else "",
        },
        "onnxruntime": {
            "cuda_major": cuda_major,
            "requirement": ONNXRUNTIME_GPU_REQUIREMENT.get(cuda_major, "onnxruntime>=1.26,<2"),
        },
    }
