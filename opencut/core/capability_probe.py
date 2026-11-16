"""Optional capability discovery in a child process.

``/health`` used to import Torch, ONNX Runtime and the other optional native
stacks inside the server to fill in its capability flags. A library that
aborts the process on import (a DLL built for another runtime, an access
violation in a CUDA probe) took the server down with it, so the panel never
got an answer at all (discussions #9 and #10).

Discovery now runs in a child copy of the server, the way ``--import-check``
vets an optional tool before it's installed:

* ``/health`` answers from cheap facts right away and starts discovery in the
  background. A capability's keys stay out of the payload until its probe has
  finished, which every client already reads as "not known yet".
* The child runs one probe at a time and reports each result on stdout. When
  it dies in the middle of a probe, that probe is marked ``crashed`` (so the
  feature reads as unavailable) with the exit code, a record goes to
  crash.log, and a fresh child picks up the probes that are left. A probe
  that runs past its time limit is killed and marked ``timeout``.
* Results are saved to ``capability_probes.json`` under a fingerprint of the
  runtime and the installed packages. A restart reuses them, so a probe that
  crashed isn't re-run on every start, and any package change runs them all
  again.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("opencut")

#: Hidden server flag that runs probes and exits (see ``opencut.server.main``).
PROBE_FLAG = "--capability-probe"
#: Bump when a probe's keys or meaning change, so saved results are dropped.
SCHEMA_VERSION = 1
CACHE_FILENAME = "capability_probes.json"
#: The parent's ``sys.path``, as JSON, so the child imports what it would.
SYS_PATH_ENV = "OPENCUT_PROBE_SYS_PATH"
START_MARK = "OPENCUT-PROBE-START "
RESULT_MARK = "OPENCUT-PROBE-RESULT "

#: Child start until its first probe begins.
STARTUP_TIMEOUT = 90.0
#: One probe, from its start line to its result line.
PROBE_TIMEOUT = 120.0
#: How long a saved result is trusted before it's checked again.
RESULT_MAX_AGE = 24 * 3600.0
#: How soon a probe the child never reached is tried again.
UNKNOWN_RETRY = 300.0
#: How often ``ensure_started`` re-reads the package fingerprint.
FINGERPRINT_RECHECK = 15.0
#: How long after the child exits its pipe may stay open (held by a helper
#: process it started) before the parent stops waiting for EOF.
EXIT_GRACE = 2.0

STATE_OK = "ok"
#: The probe raised a Python exception; reported by the child.
STATE_ERROR = "error"
#: The child died while the probe ran (access violation, abort, exit).
STATE_CRASHED = "crashed"
#: The probe ran past its time limit and the child was killed.
STATE_TIMEOUT = "timeout"
#: The child never reached the probe; its keys stay out of the payload.
STATE_UNKNOWN = "unknown"

_FAILURE_EXCEPTION = {
    STATE_CRASHED: "ProbeCrashed",
    STATE_TIMEOUT: "ProbeTimeout",
    STATE_UNKNOWN: "ProbeNotRun",
}

# Windows NTSTATUS codes a native crash leaves as the exit code.
_NTSTATUS_NAMES = {
    0xC0000005: "access violation",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC0000135: "DLL not found",
    0xC0000139: "DLL entry point not found",
    0xC0000374: "heap corruption",
    0xC0000409: "fast fail (stack buffer overrun)",
}


# ---------------------------------------------------------------------------
# Probes. Each runs only inside the child.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    name: str
    #: Keys this probe owns, with the value each takes when it's unavailable.
    defaults: dict
    run: Callable[[], dict] = field(compare=False)


def _flag(key: str, module: str, func: str) -> Probe:
    def run() -> dict:
        return {key: bool(getattr(importlib.import_module(module), func)())}

    return Probe(key, {key: False}, run)


def _captions() -> dict:
    from opencut.core.captions import check_whisper_available

    available, backend = check_whisper_available()
    return {"captions": bool(available), "whisper_backend": backend or "none"}


def _watermark_removal() -> dict:
    from opencut.helpers import _try_import

    if _try_import("transformers") is None:
        return {"watermark_removal": False}
    sli = _try_import("simple_lama_inpainting")
    return {"watermark_removal": sli is not None and hasattr(sli, "SimpleLama")}


def _video_ai() -> dict:
    from opencut.core.video_ai import get_ai_capabilities

    return {"video_ai": dict(get_ai_capabilities())}


def _face_tools() -> dict:
    from opencut.core.face_tools import check_face_tools_available

    return {"face_tools": dict(check_face_tools_available())}


_VIDEO_AI_UNAVAILABLE = {
    "upscale": False,
    "rembg": False,
    "rvm": False,
    "frame_interp": True,  # FFmpeg minterpolate, no optional import
    "rife_ncnn": False,
    "video_denoise": True,  # FFmpeg, no optional import
    "gpu_available": False,
}
_FACE_TOOLS_UNAVAILABLE = {"mediapipe": False, "insightface": False, "opencv": False, "haar": False}

PROBES: tuple[Probe, ...] = (
    Probe("captions", {"captions": False, "whisper_backend": "none"}, _captions),
    _flag("silero_vad", "opencut.checks", "check_silero_vad_available"),
    _flag("crisper_whisper", "opencut.checks", "check_crisper_whisper_available"),
    _flag("otio", "opencut.checks", "check_otio_available"),
    _flag("separation", "opencut.checks", "check_stem_separation_available"),
    Probe("watermark_removal", {"watermark_removal": False}, _watermark_removal),
    Probe("video_ai", {"video_ai": _VIDEO_AI_UNAVAILABLE}, _video_ai),
    _flag("pedalboard", "opencut.core.audio_pro", "check_pedalboard_available"),
    _flag("deepfilter", "opencut.core.audio_pro", "check_deepfilter_available"),
    Probe("face_tools", {"face_tools": _FACE_TOOLS_UNAVAILABLE}, _face_tools),
    _flag("whisperx", "opencut.core.captions_enhanced", "check_whisperx_available"),
    _flag("nllb", "opencut.core.captions_enhanced", "check_nllb_available"),
    _flag("edge_tts", "opencut.core.voice_gen", "check_edge_tts_available"),
    _flag("kokoro", "opencut.core.voice_gen", "check_kokoro_available"),
    _flag("depth_effects", "opencut.checks", "check_depth_available"),
    _flag("resolve", "opencut.checks", "check_resolve_available"),
    _flag("deepface", "opencut.checks", "check_deepface_available"),
    _flag("multimodal_diarize", "opencut.checks", "check_multimodal_diarize_available"),
    _flag("broll_generate", "opencut.checks", "check_broll_generate_available"),
    _flag("websocket", "opencut.checks", "check_websocket_available"),
)
PROBES_BY_NAME = {probe.name: probe for probe in PROBES}
PROBE_NAMES = tuple(probe.name for probe in PROBES)


def describe_exit(code: Optional[int]) -> str:
    """Readable form of a child's exit code, naming Windows crash codes."""
    if code is None:
        return "no exit code"
    if code < 0 and os.name != "nt":
        try:
            import signal

            return f"signal {signal.Signals(-code).name}"
        except (ValueError, AttributeError):
            return f"signal {-code}"
    unsigned = code & 0xFFFFFFFF
    if unsigned >= 0xC0000000:
        name = _NTSTATUS_NAMES.get(unsigned)
        return f"exit code 0x{unsigned:08X}" + (f" ({name})" if name else "")
    if code == 3 and os.name == "nt":
        return "exit code 3 (abort)"
    return f"exit code {code}"


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------
def _quiet_error_dialogs() -> None:
    """Keep a crash in this child from raising a Windows error dialog."""
    if os.name != "nt":
        return
    try:
        import ctypes

        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:  # noqa: BLE001 - dialogs are a nuisance, not a reason to stop
        pass


def _apply_forwarded_sys_path() -> None:
    raw = os.environ.pop(SYS_PATH_ENV, "")
    if not raw:
        return
    try:
        forwarded = [entry for entry in json.loads(raw) if isinstance(entry, str)]
    except ValueError:
        return
    sys.path[:] = forwarded + [entry for entry in sys.path if entry not in forwarded]


def _failure_stamps() -> dict:
    from opencut import checks

    return {name: entry.get("ts") for name, entry in checks.get_check_failures().items()}


def _run_one(name: str) -> dict:
    probe = PROBES_BY_NAME.get(name)
    if probe is None:
        return {
            "name": name, "state": STATE_ERROR, "values": {}, "failures": {},
            "exception": "KeyError", "reason": f"unknown probe {name!r}",
        }
    from opencut import checks

    before = _failure_stamps()
    state, exception, reason = STATE_OK, "", ""
    try:
        values = probe.run()
    except Exception as exc:  # noqa: BLE001 - a probe that raises is reported, not fatal
        values = {}
        state, exception = STATE_ERROR, type(exc).__name__
        reason = f"{exception}: {str(exc)[:300]}"
    shaped = {key: values.get(key, default) for key, default in probe.defaults.items()}
    failures = {
        key: entry for key, entry in checks.get_check_failures().items()
        if before.get(key) != entry.get("ts")
    }
    return {
        "name": name, "state": state, "values": shaped, "failures": failures,
        "exception": exception, "reason": reason,
    }


def _emit(stream, line: str) -> None:
    # A leading newline ends whatever a probe printed without one, so the
    # mark always starts its own line.
    stream.write("\n" + line + "\n")
    stream.flush()


def run_child(names_arg: str) -> int:
    """``--capability-probe NAMES``: run each probe and report it on stdout."""
    _quiet_error_dialogs()
    _apply_forwarded_sys_path()
    # What the probes print goes to stderr, which the parent keeps as
    # evidence, so stdout carries only the protocol.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    try:
        for name in [n.strip() for n in (names_arg or "").split(",") if n.strip()]:
            _emit(protocol, START_MARK + name)
            result = _run_one(name)
            _emit(protocol, RESULT_MARK + json.dumps(result, ensure_ascii=True, sort_keys=True, default=str))
    finally:
        sys.stdout = protocol
    return 0


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------
def probe_command(names) -> list[str]:
    """This server, run as a probe child for ``names``.

    The same entry the ``--import-check`` child uses: the packaged exe, or
    ``python -m opencut.server`` from a source checkout.
    """
    if getattr(sys, "frozen", False):
        base = [sys.executable]
    else:
        base = [sys.executable, "-m", "opencut.server"]
    return [*base, PROBE_FLAG, ",".join(names)]


_FINGERPRINT_SUFFIXES = (".dist-info", ".egg-info", ".egg-link", ".pth", ".py", ".pyd", ".so")


def runtime_fingerprint() -> str:
    """Hash of the runtime and of what each ``sys.path`` entry provides.

    Only names go in, never times, so the server writing its own files can't
    make the fingerprint drift. Installing, upgrading or removing a package
    changes a ``dist-info`` name or a top-level entry, and that's enough.
    """
    from opencut import __version__

    digest = hashlib.sha256()

    def add(text: str) -> None:
        digest.update(text.encode("utf-8", "replace") + b"\0")

    for part in (
        str(SCHEMA_VERSION), sys.executable, sys.version, __version__,
        str(bool(getattr(sys, "frozen", False))), ",".join(PROBE_NAMES),
    ):
        add(part)
    for entry in sys.path:
        add("path:" + entry)
        names = []
        try:
            with os.scandir(entry or ".") as listing:
                for item in listing:
                    if item.name.startswith((".", "__pycache__")):
                        continue
                    if item.name.endswith(_FINGERPRINT_SUFFIXES) or item.is_dir():
                        names.append(item.name)
        except OSError:
            pass
        for name in sorted(names):
            add(name)
    return digest.hexdigest()


def _default_cache_path() -> str:
    from opencut.helpers import OPENCUT_DIR

    return os.path.join(OPENCUT_DIR, CACHE_FILENAME)


# ---------------------------------------------------------------------------
# Process trees
#
# A probe's import can start helper processes of its own. Killing only the
# child leaves them running and holding its stdout, so the pipe never reaches
# EOF and closing it blocks behind the reader thread. On Windows each child
# goes into a job object that kills everything in it when it's terminated or
# when the server's handle closes, which also covers ``os._exit``. Elsewhere
# the child leads its own session and ``killpg`` reaches its helpers.
# ---------------------------------------------------------------------------
_JOB_ATTR = "_opencut_probe_job"
_job_lock = threading.Lock()
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


@functools.lru_cache(maxsize=1)
def _job_api():
    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.POINTER(ExtendedLimit), wintypes.DWORD,
    )
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    k32.TerminateJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    k32.CloseHandle.restype = wintypes.BOOL
    return ctypes, k32, ExtendedLimit


def _popen_tree_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def _attach_job(proc: subprocess.Popen) -> None:
    """Put a Windows child in a kill-on-close job. Best effort."""
    if os.name != "nt":
        return
    try:
        ctypes, k32, ExtendedLimit = _job_api()
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not (
            k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                        ctypes.byref(info), ctypes.sizeof(info))
            and k32.AssignProcessToJobObject(job, int(getattr(proc, "_handle")))
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            k32.CloseHandle(job)
            raise error
    except Exception as exc:  # noqa: BLE001 - without a job, killing the child alone still works
        logger.debug("Could not put probe process %s in a job object: %s", proc.pid, exc)
        return
    setattr(proc, _JOB_ATTR, job)


def _close_job(proc: subprocess.Popen) -> None:
    """Release the child's job, killing anything still in it."""
    with _job_lock:
        job = proc.__dict__.pop(_JOB_ATTR, None)
        if job:
            _job_api()[1].CloseHandle(job)


def _kill(proc: subprocess.Popen) -> None:
    """Stop the probe child and every process it started."""
    with _job_lock:
        job = getattr(proc, _JOB_ATTR, None)
        if job:
            _job_api()[1].TerminateJobObject(job, 1)
    if not job and os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - a process that won't die
        pass


class CapabilityDiscovery:
    """Runs the probes in child processes, one batch at a time, and keeps
    the results.

    Every public method is safe to call from any request thread. Only one
    discovery thread runs at once, so concurrent ``/health`` calls share it.
    """

    def __init__(
        self,
        *,
        cache_path: Optional[str] = None,
        probes: Optional[tuple] = None,
        startup_timeout: float = STARTUP_TIMEOUT,
        probe_timeout: float = PROBE_TIMEOUT,
        exit_grace: float = EXIT_GRACE,
        env: Optional[dict] = None,
    ) -> None:
        self.cache_path = cache_path if cache_path is not None else _default_cache_path()
        self.probe_names = tuple(probes) if probes is not None else PROBE_NAMES
        self.startup_timeout = float(startup_timeout)
        self.probe_timeout = float(probe_timeout)
        self.exit_grace = float(exit_grace)
        self._env = dict(env or {})
        self._lock = threading.Lock()
        self._results: dict = {}
        self._fingerprint: Optional[str] = None
        self._fingerprint_checked = 0.0
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._active: Optional[subprocess.Popen] = None
        self._closed = False
        #: Children started over this object's life (tests read it).
        self.spawn_count = 0

    # -- reading -----------------------------------------------------------
    def capabilities(self) -> dict:
        """Capability values for every probe that has a usable answer."""
        merged: dict = {}
        with self._lock:
            for name in self.probe_names:
                result = self._results.get(name)
                if result and result["state"] != STATE_UNKNOWN:
                    merged.update(json.loads(json.dumps(result["values"])))
        return merged

    def value(self, key: str, default: Any = None) -> Any:
        return self.capabilities().get(key, default)

    def snapshot(self) -> dict:
        """Discovery progress for ``/health``: what's pending and what failed."""
        with self._lock:
            running = self._thread is not None
            pending = [n for n in self.probe_names if n not in self._results]
            failed = {}
            checked = [r["checked_at"] for r in self._results.values() if r.get("checked_at")]
            for name in self.probe_names:
                result = self._results.get(name)
                if result and result["state"] != STATE_OK:
                    failed[name] = {
                        "state": result["state"],
                        "reason": result.get("reason", ""),
                        "exit_code": result.get("exit_code"),
                        "checked_at": result.get("checked_at"),
                    }
        if running:
            state = "running"
        elif pending:
            state = "idle"
        else:
            state = "ready"
        return {
            "state": state,
            "pending": pending,
            "failed": failed,
            "checked_at": max(checked) if checked else None,
        }

    def failures(self) -> dict:
        """Check-failure entries for ``/system/check-failures``.

        Each probe that didn't finish cleanly is ``probe:<name>``, and the
        import failures a child reported keep their ``import:<module>`` names.
        """
        entries: dict = {}
        with self._lock:
            results = [(n, self._results[n]) for n in self.probe_names if n in self._results]
        for name, result in results:
            for key, entry in (result.get("failures") or {}).items():
                entries.setdefault(key, dict(entry))
            if result["state"] == STATE_OK:
                continue
            entries[f"probe:{name}"] = {
                "exception": result.get("exception") or _FAILURE_EXCEPTION.get(result["state"], "ProbeError"),
                "message": result.get("reason", ""),
                "ts": result.get("checked_at"),
                "state": result["state"],
                "exit_code": result.get("exit_code"),
            }
        return entries

    # -- running -----------------------------------------------------------
    def ensure_started(self) -> bool:
        """Start a discovery run when any probe lacks a fresh answer.

        Returns True when a run is in progress afterwards.
        """
        now = time.time()
        with self._lock:
            if self._closed:
                return False
            if self._thread is not None:
                return True
            recheck = self._fingerprint is None or now - self._fingerprint_checked >= FINGERPRINT_RECHECK
            if recheck:
                self._fingerprint_checked = now
        # Listing sys.path takes a few milliseconds; readers shouldn't wait on it.
        fingerprint = runtime_fingerprint() if recheck else None
        with self._lock:
            if fingerprint is not None and fingerprint != self._fingerprint:
                self._adopt_fingerprint_locked(fingerprint)
            if self._closed or self._thread is not None:
                return self._thread is not None
            if not self._missing_locked(now):
                return False
            thread = threading.Thread(target=self._worker, name="opencut-capability-probe", daemon=True)
            self._thread = thread
        thread.start()
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the current run ends. True when nothing is running."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        with self._lock:
            return self._thread is None

    def invalidate(self) -> None:
        """Forget every result, after an install changed what can import."""
        with self._lock:
            self._generation += 1
            self._results.clear()
            self._fingerprint_checked = 0.0
            proc = self._active
        if proc is not None:
            _kill(proc)
        try:
            os.remove(self.cache_path)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
            proc = self._active
            thread = self._thread
        if proc is not None:
            _kill(proc)
        if thread is not None:
            thread.join(30)

    # -- internals ---------------------------------------------------------
    def _adopt_fingerprint_locked(self, fingerprint: str) -> None:
        first = self._fingerprint is None
        self._fingerprint = fingerprint
        self._generation += 1
        self._results.clear()
        if first:
            self._results.update(self._load_locked(fingerprint))
        else:
            logger.info("Installed packages changed; checking optional capabilities again")

    def _missing_locked(self, now: float) -> list:
        missing = []
        for name in self.probe_names:
            result = self._results.get(name)
            if result is None:
                missing.append(name)
                continue
            limit = UNKNOWN_RETRY if result["state"] == STATE_UNKNOWN else RESULT_MAX_AGE
            if now - float(result.get("checked_at") or 0) > limit:
                missing.append(name)
        return missing

    def _load_locked(self, fingerprint: str) -> dict:
        try:
            with open(self.cache_path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return {}
        if not isinstance(saved, dict) or saved.get("schema") != SCHEMA_VERSION:
            return {}
        if saved.get("fingerprint") != fingerprint:
            return {}
        results = saved.get("results")
        if not isinstance(results, dict):
            return {}
        return {
            name: result for name, result in results.items()
            if name in self.probe_names and isinstance(result, dict) and isinstance(result.get("values"), dict)
            and result.get("state") in (STATE_OK, STATE_ERROR, STATE_CRASHED, STATE_TIMEOUT, STATE_UNKNOWN)
        }

    def _save(self) -> None:
        with self._lock:
            if self._fingerprint is None:
                return
            payload = {
                "schema": SCHEMA_VERSION,
                "fingerprint": self._fingerprint,
                "results": json.loads(json.dumps(self._results)),
            }
        directory = os.path.dirname(self.cache_path) or "."
        tmp = os.path.join(directory, f".{CACHE_FILENAME}.{os.getpid()}.tmp")
        try:
            os.makedirs(directory, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            logger.debug("Could not save capability probe results to %s: %s", self.cache_path, exc)

    def _record(self, generation: int, name: str, result: dict) -> bool:
        result.setdefault("checked_at", time.time())
        with self._lock:
            if generation != self._generation or self._closed:
                return False
            self._results[name] = result
        return True

    def _worker(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._closed:
                        return
                    missing = self._missing_locked(time.time())
                    generation = self._generation
                if not missing:
                    return
                self._run_batch(missing, generation)
                with self._lock:
                    if generation == self._generation:
                        # Whatever is still missing had its chance this round.
                        return
        except Exception:  # noqa: BLE001 - discovery is advisory, the server goes on
            logger.exception("Capability discovery failed")
        finally:
            with self._lock:
                self._thread = None

    def _run_batch(self, names: list, generation: int) -> None:
        remaining = list(names)
        for _ in range(len(names) + 1):
            if not remaining:
                break
            done, fatal = self._run_child(remaining, generation)
            remaining = [n for n in remaining if n not in done]
            self._save()
            if fatal is not None:
                for name in remaining:
                    self._record(generation, name, {
                        "state": STATE_UNKNOWN, "values": {}, "failures": {},
                        "reason": fatal, "exit_code": None,
                    })
                self._save()
                return
            with self._lock:
                if generation != self._generation:
                    return

    def _child_env(self) -> dict:
        env = dict(os.environ)
        env.update(self._env)
        env[SYS_PATH_ENV] = json.dumps(list(sys.path))
        env["PYTHONIOENCODING"] = "utf-8"
        if not getattr(sys, "frozen", False):
            # The child imports the same modules from the start, not only
            # after it reads the forwarded path.
            import opencut

            root = os.path.dirname(os.path.dirname(os.path.abspath(opencut.__file__)))
            entries = [entry for entry in sys.path if entry]
            if root not in entries:
                entries.insert(0, root)
            env["PYTHONPATH"] = os.pathsep.join(entries)
        return env

    def _run_child(self, names: list, generation: int):
        """Run one child over ``names``. Returns (names settled, fatal reason)."""
        try:
            proc = subprocess.Popen(
                probe_command(names),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._child_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                **_popen_tree_kwargs(),
            )
        except OSError as exc:
            return set(), f"couldn't start the probe process: {exc}"
        _attach_job(proc)
        with self._lock:
            self._active = proc
            self.spawn_count += 1
            if generation != self._generation or self._closed:
                stale = True
            else:
                stale = False
        if stale:
            _kill(proc)
            _close_job(proc)
            with self._lock:
                self._active = None
            return set(), None

        lines: queue.Queue = queue.Queue()
        output = deque(maxlen=60)

        def pump_stdout() -> None:
            try:
                for line in proc.stdout:
                    lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)

        def pump_stderr() -> None:
            try:
                for line in proc.stderr:
                    output.append(line.rstrip("\r\n"))
            except (OSError, ValueError):
                pass

        readers = [
            threading.Thread(target=pump_stdout, name="opencut-probe-stdout", daemon=True),
            threading.Thread(target=pump_stderr, name="opencut-probe-stderr", daemon=True),
        ]
        for reader in readers:
            reader.start()

        done: set = set()
        fatal = None
        current = None
        exited_at = None
        deadline = time.monotonic() + self.startup_timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    _kill(proc)
                    if current is not None:
                        limit = self.probe_timeout
                        self._fail(generation, current, STATE_TIMEOUT,
                                   f"no answer within {limit:g} seconds", None, output)
                        done.add(current)
                    else:
                        fatal = f"the probe process didn't start within {self.startup_timeout:g} seconds"
                    break
                try:
                    line = lines.get(timeout=min(left, 1.0))
                except queue.Empty:
                    if proc.poll() is None:
                        continue
                    # The child is gone, but a process it started still holds
                    # stdout, so EOF won't come. Once what was buffered has
                    # been read, settle it as if the pipe had closed.
                    now = time.monotonic()
                    if exited_at is None:
                        exited_at = now
                    if now - exited_at < self.exit_grace:
                        continue
                    line = None
                if line is None:
                    # stdout closes as the child exits; let it finish so the
                    # exit code is the crash's, not the one a kill would set.
                    try:
                        code = proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        _kill(proc)
                        code = proc.returncode
                    if current is not None:
                        self._fail(generation, current, STATE_CRASHED,
                                   f"the probe process stopped with {describe_exit(code)}", code, output)
                        done.add(current)
                    elif not done and names:
                        fatal = f"the probe process exited with {describe_exit(code)} before running a probe"
                    break
                line = line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith(START_MARK):
                    current = line[len(START_MARK):].strip()
                    deadline = time.monotonic() + self.probe_timeout
                elif line.startswith(RESULT_MARK):
                    settled = self._accept(generation, names, line[len(RESULT_MARK):])
                    if settled:
                        done.add(settled)
                    current = None
                    deadline = time.monotonic() + self.probe_timeout
                else:
                    output.append(line)
        finally:
            # Always, even after a clean exit: helpers the probes started are
            # still in the job and may hold the pipes.
            _kill(proc)
            for reader in readers:
                reader.join(5)
            for reader, stream in zip(readers, (proc.stdout, proc.stderr)):
                if reader.is_alive():
                    # A process outside the job still holds this pipe. Closing
                    # it would block behind the reader, so leave both be.
                    continue
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            _close_job(proc)
            with self._lock:
                if self._active is proc:
                    self._active = None
        return done, fatal

    def _accept(self, generation: int, names: list, raw: str) -> Optional[str]:
        try:
            payload = json.loads(raw)
        except ValueError:
            return None
        name = payload.get("name") if isinstance(payload, dict) else None
        if name not in names:
            return None
        probe = PROBES_BY_NAME.get(name)
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        if probe is not None:
            values = {key: values.get(key, default) for key, default in probe.defaults.items()}
        state = payload.get("state") if payload.get("state") in (STATE_OK, STATE_ERROR) else STATE_ERROR
        failures = payload.get("failures") if isinstance(payload.get("failures"), dict) else {}
        result = {
            "state": state,
            "values": values,
            "failures": failures,
            "exception": str(payload.get("exception") or ""),
            "reason": str(payload.get("reason") or "")[:500],
            "exit_code": None,
        }
        if state == STATE_ERROR:
            logger.warning("Capability probe %s raised %s; treating it as unavailable", name, result["reason"])
        self._record(generation, name, result)
        return name

    def _fail(self, generation: int, name: str, state: str, reason: str, code, output) -> None:
        probe = PROBES_BY_NAME.get(name)
        recorded = self._record(generation, name, {
            "state": state,
            "values": json.loads(json.dumps(probe.defaults)) if probe is not None else {},
            "failures": {},
            "reason": reason,
            "exit_code": code,
        })
        if not recorded:
            return
        tail = "\n".join(list(output)[-40:])
        record_path = ""
        try:
            from opencut.crash_reporter import write_probe_record

            record_path = write_probe_record(name, state, reason, tail)
        except Exception as exc:  # noqa: BLE001 - the record is evidence, not a requirement
            logger.debug("Could not write a crash record for probe %s: %s", name, exc)
        logger.warning(
            "Capability probe %s %s (%s); treating it as unavailable. Details: %s",
            name, state, reason, record_path or "crash.log",
        )


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------
_discovery: Optional[CapabilityDiscovery] = None
_discovery_lock = threading.Lock()


def get_discovery() -> CapabilityDiscovery:
    global _discovery
    with _discovery_lock:
        if _discovery is None:
            _discovery = CapabilityDiscovery()
        return _discovery


def set_discovery(discovery: Optional[CapabilityDiscovery]) -> Optional[CapabilityDiscovery]:
    """Swap the process-wide instance (tests). Returns the previous one."""
    global _discovery
    with _discovery_lock:
        previous, _discovery = _discovery, discovery
    return previous


def invalidate() -> None:
    with _discovery_lock:
        discovery = _discovery
    if discovery is not None:
        discovery.invalidate()


def reported_failures() -> dict:
    with _discovery_lock:
        discovery = _discovery
    return discovery.failures() if discovery is not None else {}


def stop_active() -> None:
    """Stop discovery and kill the running probe child with its helpers.

    Registered with ``atexit``. ``/shutdown`` ends the server with
    ``os._exit``, which skips ``atexit``, so it calls this first.
    """
    with _discovery_lock:
        discovery = _discovery
    if discovery is not None:
        with discovery._lock:
            discovery._closed = True
            proc = discovery._active
        if proc is not None:
            _kill(proc)


atexit.register(stop_active)
