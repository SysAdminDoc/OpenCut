"""Keep CTranslate2's CUDA 12 runtime away from a CUDA 13 PyTorch (F465).

CTranslate2 4.8.1 (faster-whisper) is a CUDA 12 build. It loads
``cublas64_12.dll`` by name and bundles only the ``cudnn64_9.dll`` shim. A
cu132 PyTorch brings ``cublas64_13.dll`` and a CUDA 13 cuDNN whose DLLs
carry the same ``cudnn*64_9.dll`` names. Windows hands a by-name load
whichever module of that name loaded first, so one process can't hold both
cuDNN builds.

Two measures follow from that:

* faster-whisper transcription runs in a child process with torch blocked
  and CTranslate2's own CUDA 12 runtime (``<data folder>/cuda12``, see
  :func:`opencut.gpu_lanes.ct2_cuda12_runtime_command`) on its DLL path.
* In the server process, the first ``import ctranslate2`` imports torch
  first, so torch's CUDA 13 DLLs claim their names before CTranslate2's
  bundled shim can.

Nothing here applies while torch is a CUDA 12 build, a CPU build or absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional

from opencut import gpu_lanes

logger = logging.getLogger("opencut")

#: Hidden server flag that turns a server process into the worker.
WORKER_FLAG = "--ct2-worker"

#: Set in the worker's environment so it never spawns a worker of its own.
CHILD_ENV = "OPENCUT_CT2_CHILD"

#: Loaded by full path in this order, so each one's dependencies are already
#: in memory when it loads.
PRELOAD_ORDER = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")

_LOCAL_CUDA_RE = re.compile(r"\+cu(\d{2,3})\b")
_DLL_DIRECTORY_HANDLES: list = []


def torch_cuda_major() -> Optional[int]:
    """The CUDA major the installed torch was built for, without importing it.

    A loaded torch answers through ``torch.version.cuda``. Otherwise the
    wheel's local version tag (``2.15.0+cu132``) does. None for CPU builds,
    untagged builds and no torch at all.
    """
    module = sys.modules.get("torch")
    if module is not None:
        version = getattr(getattr(module, "version", None), "cuda", None)
        match = re.match(r"\s*(\d+)", str(version or ""))
        return int(match.group(1)) if match else None
    try:
        from importlib.metadata import version

        installed = version("torch")
    except Exception:
        return None
    match = _LOCAL_CUDA_RE.search(installed or "")
    return int(match.group(1)[:-1]) if match else None


def should_isolate() -> bool:
    """True when CTranslate2 must run outside this process."""
    if os.environ.get(CHILD_ENV) == "1":
        return False
    major = torch_cuda_major()
    return major is not None and major != gpu_lanes.BACKEND_CUDA["ctranslate2"].cuda_major


def cuda12_library_dirs(root: Optional[str] = None) -> list:
    """The DLL (or shared library) folders of the CUDA 12 runtime install."""
    root = root or gpu_lanes.cuda12_runtime_dir()
    base = os.path.join(root, "nvidia")
    if not os.path.isdir(base):
        return []
    leaf = "bin" if os.name == "nt" else "lib"
    return [
        os.path.join(base, name, leaf)
        for name in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, name, leaf))
    ]


def preload_cuda12_runtime(
    dirs: Optional[list] = None,
    *,
    loader: Optional[Callable] = None,
    add_dll_directory: Optional[Callable] = None,
) -> list:
    """Load CTranslate2's CUDA 12 cuBLAS and cuDNN by full path. Windows only.

    Runs in the worker before ``ctranslate2`` is imported, so the by-name
    loads CTranslate2 makes later find these modules already in memory.
    Returns the paths it loaded.
    """
    if os.name != "nt":
        return []
    dirs = cuda12_library_dirs() if dirs is None else list(dirs)
    add = add_dll_directory or getattr(os, "add_dll_directory", None)
    if loader is None:
        import ctypes

        loader = ctypes.WinDLL
    for folder in dirs:
        if add is not None:
            try:
                _DLL_DIRECTORY_HANDLES.append(add(folder))
            except OSError as exc:
                logger.debug("Could not add %s to the DLL search path: %s", folder, exc)
    loaded = []
    for name in PRELOAD_ORDER:
        path = next((os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))), None)
        if path is None:
            continue
        try:
            loader(path)
        except OSError as exc:
            logger.warning("Could not load %s for CTranslate2: %s", path, exc)
            continue
        loaded.append(path)
    return loaded


def worker_command(request_path: str, result_path: str) -> list:
    """This server, started as the faster-whisper worker."""
    if getattr(sys, "frozen", False):
        base = [sys.executable]
    else:
        base = [sys.executable, "-m", "opencut.server"]
    return [*base, WORKER_FLAG, request_path, result_path]


def worker_env(dirs: list) -> dict:
    """The worker's environment: marked as the child, CUDA 12 runtime first on the path."""
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join([*dirs, env.get(var, "")]).rstrip(os.pathsep)
    return env


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def transcribe_in_child(wav_path: str, config):
    """Run ``_transcribe_faster_whisper`` in the worker and return its result.

    The worker is registered with the running job, so cancelling the job
    kills it. ``hf_token`` stays out of the request file because
    faster-whisper never reads it.
    """
    from dataclasses import asdict

    from opencut.core.captions import transcription_result_from_dict

    dirs = cuda12_library_dirs()
    if not dirs:
        logger.warning(
            "faster-whisper's worker process has no CUDA 12 runtime of its own, so it "
            "will likely fall back to CPU. Install the runtime with: %s",
            gpu_lanes.ct2_cuda12_runtime_command(),
        )
    payload = asdict(config)
    payload["hf_token"] = None
    workdir = tempfile.mkdtemp(prefix="opencut-ct2-")
    request_path = os.path.join(workdir, "request.json")
    result_path = os.path.join(workdir, "result.json")
    try:
        _write_json(request_path, {"wav_path": wav_path, "config": payload})
        proc = subprocess.Popen(
            worker_command(request_path, result_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=worker_env(dirs),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        from opencut.jobs import _register_job_process, _unregister_job_process, get_current_job_id

        job_id = get_current_job_id()
        if job_id:
            _register_job_process(job_id, proc)
        try:
            _stdout, stderr = proc.communicate()
        finally:
            if job_id:
                _unregister_job_process(job_id)
        reply = _read_json(result_path)
        if proc.returncode == 0 and isinstance(reply.get("result"), dict):
            return transcription_result_from_dict(reply["result"])
        detail = reply.get("error") or (stderr or "").strip()[-600:] or f"exit code {proc.returncode}"
        raise RuntimeError(f"faster-whisper failed in its CUDA 12 worker process: {detail}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_worker(request_path: str, result_path: str, *, transcribe=None, preload=None) -> int:
    """``--ct2-worker``: transcribe one request with torch blocked.

    Exit code 0 writes ``{"result": ...}``, anything else ``{"error": ...}``.
    """
    loaded = sys.modules.get("torch")
    if loaded is not None:
        _write_json(result_path, {
            "error": "PyTorch was already loaded in the faster-whisper worker, so its CUDA "
                     "DLLs would collide with CTranslate2's.",
        })
        return 3
    sys.modules["torch"] = None  # every later "import torch" raises ImportError
    os.environ[CHILD_ENV] = "1"
    try:
        (preload or preload_cuda12_runtime)()
        from dataclasses import fields

        from opencut.core.captions import _transcribe_faster_whisper, transcription_result_to_dict
        from opencut.utils.config import CaptionConfig

        request = _read_json(request_path)
        known = {f.name for f in fields(CaptionConfig)}
        config = CaptionConfig(**{k: v for k, v in (request.get("config") or {}).items() if k in known})
        result = (transcribe or _transcribe_faster_whisper)(request["wav_path"], config)
        _write_json(result_path, {"result": transcription_result_to_dict(result)})
        return 0
    except Exception as exc:
        logger.exception("faster-whisper worker failed")
        _write_json(result_path, {"error": f"{type(exc).__name__}: {exc}"})
        return 1


class _TorchFirstFinder:
    """Imports torch just before the first ``ctranslate2`` import, then retires."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "ctranslate2":
            return None
        _remove_torch_first_hook()
        if "torch" not in sys.modules and should_isolate():
            try:
                import torch  # noqa: F401
            except Exception as exc:
                logger.debug("Could not import torch ahead of ctranslate2: %s", exc)
        return None


def _remove_torch_first_hook() -> None:
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _TorchFirstFinder)]


def prepare_process(argv) -> None:
    """Called first thing by the server.

    The worker blocks torch before anything can import it. The server
    process makes torch load ahead of CTranslate2.
    """
    if WORKER_FLAG in list(argv or []):
        sys.modules.setdefault("torch", None)
        return
    if "ctranslate2" not in sys.modules and not any(
        isinstance(f, _TorchFirstFinder) for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _TorchFirstFinder())
