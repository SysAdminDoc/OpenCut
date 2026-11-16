"""CTranslate2's CUDA 12 runtime stays apart from a CUDA 13 torch (F465, issue #7).

CTranslate2 4.8.1 loads cublas64_12.dll and cudnn64_9.dll by name. A cu132
torch ships cublas64_13.dll and a CUDA 13 cuDNN under the same cudnn*64_9
names, so faster-whisper runs in a worker process beside it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

from opencut import ct2_isolation
from opencut.core.captions import CaptionSegment, TranscriptionResult
from opencut.utils.config import CaptionConfig


def _fake_torch(cuda):
    return types.SimpleNamespace(version=types.SimpleNamespace(cuda=cuda))


@pytest.mark.parametrize("cuda, expected", [("13.2", 13), ("12.8", 12), (None, None)])
def test_a_loaded_torch_reports_its_cuda_major(monkeypatch, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
    assert ct2_isolation.torch_cuda_major() == expected


@pytest.mark.parametrize("installed, expected", [
    ("2.15.0+cu132", 13),
    ("2.11.0+cu128", 12),
    ("2.7.1+cu118", 11),
    ("2.15.0+cpu", None),
    ("2.15.0", None),
])
def test_an_unloaded_torch_reports_its_cuda_major_from_the_wheel_tag(monkeypatch, installed, expected):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("importlib.metadata.version", lambda name: installed)
    assert ct2_isolation.torch_cuda_major() == expected


@pytest.mark.parametrize("cuda, isolate", [("13.0", True), ("12.8", False), (None, False)])
def test_only_a_non_cuda_12_torch_isolates_ctranslate2(monkeypatch, cuda, isolate):
    monkeypatch.delenv(ct2_isolation.CHILD_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
    assert ct2_isolation.should_isolate() is isolate


def test_the_worker_never_spawns_another_worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("13.2"))
    monkeypatch.setenv(ct2_isolation.CHILD_ENV, "1")
    assert ct2_isolation.should_isolate() is False


def _cuda12_runtime(root):
    files = {
        "cublas": ["cublas64_12.dll", "cublasLt64_12.dll"],
        "cudnn": ["cudnn64_9.dll", "cudnn_ops64_9.dll"],
    }
    for package, names in files.items():
        folder = root / "nvidia" / package / "bin"
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(b"")
    return [str(root / "nvidia" / "cublas" / "bin"), str(root / "nvidia" / "cudnn" / "bin")]


def test_the_cuda12_runtime_loads_by_full_path_in_dependency_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ct2_isolation.os, "name", "nt")
    dirs = _cuda12_runtime(tmp_path)
    assert ct2_isolation.cuda12_library_dirs(str(tmp_path)) == dirs

    loaded, added = [], []
    result = ct2_isolation.preload_cuda12_runtime(
        dirs, loader=loaded.append, add_dll_directory=added.append,
    )

    cublas, cudnn = dirs
    assert loaded == [
        os.path.join(cublas, "cublasLt64_12.dll"),
        os.path.join(cublas, "cublas64_12.dll"),
        os.path.join(cudnn, "cudnn64_9.dll"),
    ]
    assert result == loaded
    assert added == dirs


def test_the_worker_environment_puts_the_cuda12_runtime_first(monkeypatch):
    monkeypatch.setattr(ct2_isolation.os, "name", "nt")
    monkeypatch.setenv("PATH", "C:\\Windows")
    env = ct2_isolation.worker_env(["C:\\rt\\cublas\\bin", "C:\\rt\\cudnn\\bin"])
    assert env[ct2_isolation.CHILD_ENV] == "1"
    assert env["PATH"].split(ct2_isolation.os.pathsep)[:2] == ["C:\\rt\\cublas\\bin", "C:\\rt\\cudnn\\bin"]


def test_the_worker_command_is_this_server(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert ct2_isolation.worker_command("req", "res") == [
        sys.executable, "-m", "opencut.server", "--ct2-worker", "req", "res",
    ]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert ct2_isolation.worker_command("req", "res") == [sys.executable, "--ct2-worker", "req", "res"]


def _result():
    return TranscriptionResult(
        segments=[CaptionSegment(text="hello there", start=0.0, end=1.5)],
        language="en",
        duration=1.5,
    )


@pytest.fixture
def no_torch_loaded(monkeypatch):
    """The worker blocks torch in sys.modules. Put the module table back after."""
    saved = sys.modules.pop("torch", None)
    monkeypatch.setenv(ct2_isolation.CHILD_ENV, "")
    yield
    sys.modules.pop("torch", None)
    if saved is not None:
        sys.modules["torch"] = saved


def test_the_worker_blocks_torch_and_writes_the_result(tmp_path, no_torch_loaded):
    request = tmp_path / "request.json"
    reply = tmp_path / "result.json"
    request.write_text(json.dumps({
        "wav_path": "clip.wav",
        "config": {"model": "tiny", "language": "en", "not_a_field": 1},
    }), encoding="utf-8")
    seen = {}

    def _transcribe(wav_path, config):
        seen["wav"] = wav_path
        seen["config"] = config
        try:
            import torch  # noqa: F401
        except ImportError:
            seen["torch_blocked"] = True
        return _result()

    preloaded = []
    code = ct2_isolation.run_worker(
        str(request), str(reply), transcribe=_transcribe, preload=lambda: preloaded.append(True),
    )

    assert code == 0
    assert seen["torch_blocked"] is True
    assert preloaded == [True]
    assert seen["wav"] == "clip.wav"
    assert isinstance(seen["config"], CaptionConfig) and seen["config"].model == "tiny"
    body = json.loads(reply.read_text(encoding="utf-8"))
    assert body["result"]["segments"][0]["text"] == "hello there"


def test_the_worker_reports_a_failure(tmp_path, no_torch_loaded):
    request = tmp_path / "request.json"
    reply = tmp_path / "result.json"
    request.write_text(json.dumps({"wav_path": "clip.wav", "config": {}}), encoding="utf-8")

    def _transcribe(wav_path, config):
        raise RuntimeError("cublas64_12.dll is not found")

    code = ct2_isolation.run_worker(str(request), str(reply), transcribe=_transcribe, preload=lambda: None)

    assert code == 1
    assert "cublas64_12.dll is not found" in json.loads(reply.read_text(encoding="utf-8"))["error"]


def test_the_worker_refuses_when_torch_is_already_loaded(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("13.2"))
    reply = tmp_path / "result.json"
    assert ct2_isolation.run_worker(str(tmp_path / "missing.json"), str(reply)) == 3
    assert "already loaded" in json.loads(reply.read_text(encoding="utf-8"))["error"]


class _FakeWorker:
    """Plays the worker: reads the request, writes a reply, exits."""

    launches = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        request_path, result_path = command[-2], command[-1]
        with open(request_path, encoding="utf-8") as fh:
            self.request = json.load(fh)
        _FakeWorker.launches.append(self)
        self.returncode = self.reply(result_path)

    def reply(self, result_path):
        from opencut.core.captions import transcription_result_to_dict

        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump({"result": transcription_result_to_dict(_result())}, fh)
        return 0

    def communicate(self):
        return "", ""


def test_the_server_side_round_trips_a_transcription(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCUT_HOME", str(tmp_path))
    monkeypatch.setattr(ct2_isolation.subprocess, "Popen", _FakeWorker)
    _FakeWorker.launches = []
    config = CaptionConfig(model="tiny", hf_token="hf_secret_value")

    result = ct2_isolation.transcribe_in_child("clip.wav", config)

    assert result.segments[0].text == "hello there"
    launch = _FakeWorker.launches[0]
    assert launch.command[-3] == "--ct2-worker"
    assert launch.request["wav_path"] == "clip.wav"
    assert launch.request["config"]["model"] == "tiny"
    assert launch.request["config"]["hf_token"] is None
    assert launch.kwargs["env"][ct2_isolation.CHILD_ENV] == "1"
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        assert launch.kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW


def test_a_failed_worker_raises_its_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCUT_HOME", str(tmp_path))

    class _Failing(_FakeWorker):
        def reply(self, result_path):
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump({"error": "OSError: no CUDA device"}, fh)
            return 1

    monkeypatch.setattr(ct2_isolation.subprocess, "Popen", _Failing)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        ct2_isolation.transcribe_in_child("clip.wav", CaptionConfig(model="tiny"))


def test_captions_hand_faster_whisper_to_the_worker_before_importing_it(monkeypatch, tmp_path):
    from opencut.core import captions

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # importing it would raise
    monkeypatch.setattr(ct2_isolation, "should_isolate", lambda: True)
    calls = []
    monkeypatch.setattr(
        ct2_isolation, "transcribe_in_child", lambda wav, cfg: calls.append((wav, cfg)) or _result(),
    )

    result = captions._transcribe_faster_whisper("clip.wav", CaptionConfig(model="tiny"))

    assert result.segments[0].text == "hello there"
    assert calls and calls[0][0] == "clip.wav"


def test_the_server_flag_runs_the_worker(monkeypatch):
    from opencut import server

    seen = []
    monkeypatch.setattr(server._ct2_isolation, "run_worker", lambda req, res: seen.append((req, res)) or 7)
    monkeypatch.setattr(server._ct2_isolation, "prepare_process", lambda argv: None)
    monkeypatch.setattr(sys, "argv", ["opencut-server", "--ct2-worker", "req.json", "res.json"])

    assert server.main() == 7
    assert seen == [("req.json", "res.json")]


def test_the_worker_process_blocks_torch_before_anything_imports_it(no_torch_loaded):
    ct2_isolation.prepare_process(["--ct2-worker", "req", "res"])
    assert "torch" in sys.modules and sys.modules["torch"] is None


def test_the_server_process_imports_torch_ahead_of_ctranslate2(monkeypatch, tmp_path, no_torch_loaded):
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text("LOADED_FIRST = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(ct2_isolation, "should_isolate", lambda: True)

    ct2_isolation.prepare_process([])
    hooks = [f for f in sys.meta_path if isinstance(f, ct2_isolation._TorchFirstFinder)]
    assert len(hooks) == 1

    assert hooks[0].find_spec("ctranslate2") is None
    assert getattr(sys.modules.get("torch"), "LOADED_FIRST", False) is True
    assert not any(isinstance(f, ct2_isolation._TorchFirstFinder) for f in sys.meta_path)
