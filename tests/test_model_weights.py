"""Install and readiness cover the model weights an add-on loads (F466).

Discussion #10: Depth Effects said "installed" with no Depth Anything weights
on disk, and the first job downloaded them (or failed in local-only mode).
These tests hold readiness, the install route and the loaders to the pinned
weights in ``opencut.core.model_weights``.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import types
import zipfile
from concurrent.futures import Future

import pytest

from opencut.core import model_weights
from opencut.network_policy import LocalOnlyNetworkError
from tests.conftest import csrf_headers

DEPTH_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Empty Hub cache, deepface home and InsightFace folder under tmp_path."""
    hub = tmp_path / "hub"
    deepface = tmp_path / "deepface" / ".deepface" / "weights"
    insightface = tmp_path / "insightface" / "buffalo_l"
    monkeypatch.setattr(model_weights, "_hf_cache_dir", lambda: str(hub))
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path / "deepface"))
    monkeypatch.setattr(model_weights, "insightface_model_dir", lambda name="buffalo_l": str(insightface))
    monkeypatch.delenv("DEEPFACE_BACKEND_ENGINE", raising=False)
    monkeypatch.setattr(model_weights, "_deepface_has_onnx_engine", lambda: True)
    monkeypatch.delenv("OPENCUT_LOCAL_ONLY", raising=False)
    return types.SimpleNamespace(hub=hub, deepface=deepface, insightface=insightface)


@pytest.fixture
def no_network(monkeypatch):
    """Any download attempt fails the test."""
    import urllib.request

    from opencut.core import model_safety

    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(model_safety, "safe_snapshot_download", refuse)


def _place_hf_snapshot(stores, repo_id):
    """Files of the pinned sizes where the Hub cache keeps the pinned commit."""
    model = model_weights.HF_MODELS[repo_id]
    snapshot = (
        stores.hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / model.revision
    )
    snapshot.mkdir(parents=True)
    for name, size, _sha in model.files:
        with open(snapshot / name, "wb") as handle:
            handle.truncate(size)
    return snapshot


def _local_only(monkeypatch):
    monkeypatch.setenv("OPENCUT_LOCAL_ONLY", "1")


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

@pytest.fixture
def addon_packages(monkeypatch, tmp_path):
    """Every add-on's Python packages report installed.

    /health takes these flags from the capability probe, which runs the
    checks in a child process, so the probe's answers are stubbed here.
    """
    from opencut.core import capability_probe

    class ProbedDiscovery(capability_probe.CapabilityDiscovery):
        def capabilities(self):
            return {
                "depth_effects": True,
                "crisper_whisper": True,
                "deepface": True,
                "multimodal_diarize": True,
            }

        def ensure_started(self):
            return False

    previous = capability_probe.set_discovery(
        ProbedDiscovery(cache_path=str(tmp_path / "probe-cache.json"), probes=())
    )
    real_find_spec = model_weights.importlib.util.find_spec
    monkeypatch.setattr(
        model_weights.importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "insightface" else real_find_spec(name, *a),
    )
    from opencut.routes.system import invalidate_caps_cache

    invalidate_caps_cache()
    yield
    capability_probe.set_discovery(previous)
    invalidate_caps_cache()


def _capabilities(client):
    from opencut.routes.system import invalidate_caps_cache

    invalidate_caps_cache()
    data = client.get("/health").get_json()
    return data["capabilities"]


def test_packages_without_weights_report_not_ready(stores, addon_packages, client):
    caps = _capabilities(client)

    assert caps["depth_effects"] is False
    assert caps["crisper_whisper"] is False
    assert caps["deepface"] is False
    # Multimodal diarization still runs on its other face detectors.
    assert caps["multimodal_diarize"] is True
    components = caps["model_weights"]["components"]
    assert components["depth_effects"] == {
        "label": "Depth Anything V2 Small",
        "packages": True,
        "state": "missing",
        "download_mb": 99,
    }
    assert components["crisper_whisper"]["state"] == "missing"
    # Sum of the pinned CrisperWhisper file sizes: 3,221,603,435 bytes.
    assert components["crisper_whisper"]["download_mb"] == 3222
    assert components["emotion_highlights"]["download_mb"] == 6
    assert components["multimodal_diarize"] == {
        "label": "InsightFace buffalo_l",
        "packages": True,
        "state": "missing",
        "download_mb": 289,
    }
    assert caps["model_weights"]["local_only"] is False


def test_weights_on_disk_report_ready(stores, addon_packages, client):
    _place_hf_snapshot(stores, DEPTH_SMALL)
    stores.deepface.mkdir(parents=True)
    with open(stores.deepface / "facial_expression_model_weights.onnx", "wb") as handle:
        handle.truncate(5_947_004)

    caps = _capabilities(client)

    assert caps["depth_effects"] is True
    assert caps["deepface"] is True
    components = caps["model_weights"]["components"]
    assert components["depth_effects"]["state"] == "ready"
    assert components["emotion_highlights"]["state"] == "ready"
    assert caps["crisper_whisper"] is False


def test_a_truncated_weights_file_is_not_ready(stores):
    snapshot = _place_hf_snapshot(stores, DEPTH_SMALL)
    with open(snapshot / "model.safetensors", "r+b") as handle:
        handle.truncate(1024)

    assert model_weights.hf_model_ready(DEPTH_SMALL) is False


def test_packages_missing_leaves_the_capability_to_the_package_check(stores):
    caps = {"depth_effects": False, "crisper_whisper": False, "deepface": False}
    model_weights.apply_to_capabilities(caps)

    assert caps["model_weights"]["components"]["depth_effects"]["packages"] is False
    assert caps["depth_effects"] is False


# ---------------------------------------------------------------------------
# Install route
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, cmd):
        self.args = cmd
        self.stdout = iter(["Successfully installed thing-1.0\n"])
        self.returncode = 0
        self.pid = 4242

    def communicate(self, timeout=None):
        return "Successfully installed thing-1.0\n", ""

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def install_run(monkeypatch, client):
    """Run the install job inline with pip faked; record pip and weight fetches in order."""
    class InlinePool:
        def submit(self, job_id, fn):
            future = Future()
            fn()
            future.set_result(None)
            return future

    monkeypatch.setattr("opencut.workers.get_pool", lambda: InlinePool())
    monkeypatch.setattr("opencut.jobs._persist_job", lambda *args, **kwargs: None)
    monkeypatch.setitem(client.application.config, "OPENCUT_RUN_INSTALLS_IN_TESTS", True)
    events = []

    def fake_popen(cmd, *args, **kwargs):
        if "pip" in cmd:
            events.append(("pip", cmd))
        return _FakeProc(cmd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "ok\n", ""),
    )

    def fake_ensure(name, *, on_progress=None):
        events.append(("weights", name))
        if on_progress:
            on_progress(50, f"Downloading {name}")
            on_progress(100, f"{name} is ready.")
        return "C:/weights"

    monkeypatch.setattr(model_weights, "ensure", fake_ensure)
    return events


def _install(client, csrf_token, route, payload):
    from opencut.jobs import _get_job_copy

    response = client.post(route, json=payload, headers=csrf_headers(csrf_token))
    assert response.status_code == 200, response.get_json()
    job = _get_job_copy(response.get_json()["job_id"])
    assert job is not None
    return job


@pytest.mark.parametrize(
    ("route", "component"),
    [
        ("/video/depth/install", "depth_effects"),
        ("/video/emotion/install", "emotion_highlights"),
        ("/audio/crisper-whisper/install", "crisper_whisper"),
        ("/video/multimodal-diarize/install", "multimodal_diarize"),
    ],
)
def test_install_fetches_the_weights_after_the_packages(
    install_run, client, csrf_token, route, component,
):
    job = _install(client, csrf_token, route, {})

    assert job["status"] == "complete", job.get("error")
    kinds = [kind for kind, _detail in install_run]
    assert "pip" in kinds
    assert kinds[-1] == "weights" and kinds.count("weights") == 1
    assert install_run[-1] == ("weights", component)


def test_weights_only_install_skips_pip(install_run, client, csrf_token):
    job = _install(client, csrf_token, "/video/depth/install", {"weights_only": True})

    assert job["status"] == "complete", job.get("error")
    assert install_run == [("weights", "depth_effects")]


def test_an_install_without_weights_is_unchanged(install_run, client, csrf_token):
    job = _install(client, csrf_token, "/video/broll-generate/install", {"weights_only": True})

    assert job["status"] == "complete", job.get("error")
    assert install_run and all(kind == "pip" for kind, _detail in install_run)


# ---------------------------------------------------------------------------
# Local-only mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", sorted(model_weights.COMPONENTS))
def test_local_only_refuses_a_missing_download_before_any_network(
    monkeypatch, stores, no_network, component,
):
    _local_only(monkeypatch)

    with pytest.raises(LocalOnlyNetworkError) as caught:
        model_weights.ensure(component)

    assert caught.value.code == "LOCAL_ONLY_NETWORK_BLOCKED"
    assert "Turn off local-only mode in Settings" in caught.value.suggestion


def test_local_only_loads_weights_already_on_disk(monkeypatch, stores, no_network):
    snapshot = _place_hf_snapshot(stores, DEPTH_SMALL)
    _local_only(monkeypatch)

    assert model_weights.hf_model_dir(DEPTH_SMALL) == str(snapshot)


# ---------------------------------------------------------------------------
# Downloads are checked before they land
# ---------------------------------------------------------------------------

class _Response(io.BytesIO):
    def __init__(self, body, url):
        super().__init__(body)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serve(monkeypatch, body, final_url=None):
    import urllib.request

    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request.full_url)
        return _Response(body, final_url or request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def _spec(body, *, url="https://example.invalid/w.onnx", size=None, sha=None, members=()):
    return model_weights.UrlFile(
        url,
        sha if sha is not None else hashlib.sha256(body).hexdigest(),
        len(body) if size is None else size,
        members=members,
    )


def test_a_matching_download_lands(monkeypatch, tmp_path):
    body = b"weights" * 100
    _serve(monkeypatch, body)
    dest = tmp_path / "w.onnx"

    model_weights._download(_spec(body), str(dest), "w", None)

    assert dest.read_bytes() == body
    assert not (tmp_path / "w.onnx.part").exists()


def test_a_sha256_mismatch_never_lands(monkeypatch, tmp_path):
    body = b"weights" * 100
    _serve(monkeypatch, body)
    dest = tmp_path / "w.onnx"

    with pytest.raises(model_weights.WeightsError, match="sha256"):
        model_weights._download(_spec(body, sha="0" * 64), str(dest), "w", None)

    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("declared", [699, 701])
def test_a_size_mismatch_never_lands(monkeypatch, tmp_path, declared):
    body = b"weights" * 100
    _serve(monkeypatch, body)
    dest = tmp_path / "w.onnx"

    with pytest.raises(model_weights.WeightsError):
        model_weights._download(_spec(body, size=declared), str(dest), "w", None)

    assert os.listdir(tmp_path) == []


def test_plain_http_is_refused(monkeypatch, tmp_path):
    seen = _serve(monkeypatch, b"x")

    with pytest.raises(model_weights.WeightsError, match="HTTPS"):
        model_weights._download(
            _spec(b"x", url="http://example.invalid/w.onnx"), str(tmp_path / "w"), "w", None,
        )
    assert seen == []


def test_a_redirect_to_plain_http_is_refused(monkeypatch, tmp_path):
    _serve(monkeypatch, b"x", final_url="http://mirror.invalid/w.onnx")

    with pytest.raises(model_weights.WeightsError, match="HTTPS"):
        model_weights._download(_spec(b"x"), str(tmp_path / "w"), "w", None)
    assert os.listdir(tmp_path) == []


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def test_a_model_pack_unpacks_and_is_checked(monkeypatch, tmp_path):
    body = _zip([("det.onnx", b"d" * 10), ("rec.onnx", b"r" * 20)])
    _serve(monkeypatch, body)
    dest = tmp_path / "models" / "pack"

    model_weights._fetch_zip(
        _spec(body, members=(("det.onnx", 10), ("rec.onnx", 20))), str(dest), "pack", None,
    )

    assert sorted(os.listdir(dest)) == ["det.onnx", "rec.onnx"]
    assert sorted(os.listdir(tmp_path / "models")) == ["pack"]


def test_a_model_pack_that_escapes_its_folder_is_refused(monkeypatch, tmp_path):
    body = _zip([("../escaped.onnx", b"x" * 10)])
    _serve(monkeypatch, body)
    dest = tmp_path / "models" / "pack"

    with pytest.raises(Exception):
        model_weights._fetch_zip(
            _spec(body, members=(("escaped.onnx", 10),)), str(dest), "pack", None,
        )

    assert not (tmp_path / "models" / "escaped.onnx").exists()
    assert not dest.exists()


def test_a_model_pack_missing_a_file_is_refused(monkeypatch, tmp_path):
    body = _zip([("det.onnx", b"d" * 10)])
    _serve(monkeypatch, body)

    with pytest.raises(model_weights.WeightsError, match="without the files"):
        model_weights._fetch_zip(
            _spec(body, members=(("det.onnx", 10), ("rec.onnx", 20))),
            str(tmp_path / "pack"), "pack", None,
        )


def test_hub_download_is_pinned_and_verified(monkeypatch, stores):
    from opencut.core import model_safety

    calls = []

    def fake_snapshot_download(repo_id, *, revision, **kwargs):
        calls.append((repo_id, revision, kwargs))
        return str(_place_hf_snapshot(stores, repo_id))

    monkeypatch.setattr(model_safety, "safe_snapshot_download", fake_snapshot_download)

    # Zero-filled files have the pinned sizes but not the pinned digest.
    with pytest.raises(model_weights.WeightsError, match="sha256"):
        model_weights.hf_model_dir(DEPTH_SMALL)

    model = model_weights.HF_MODELS[DEPTH_SMALL]
    assert calls == [(
        DEPTH_SMALL,
        model.revision,
        {"allow_patterns": ["config.json", "preprocessor_config.json", "model.safetensors"]},
    )]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _fake_ml_modules(monkeypatch, loaded):
    class _Loaded:
        def to(self, device):
            raise RuntimeError("stop after load")

    class _Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.append(path)
            return _Loaded()

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.float16 = "float16"
    torch.float32 = "float32"
    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = _Auto
    transformers.AutoModelForDepthEstimation = _Auto
    transformers.AutoModelForSpeechSeq2Seq = _Auto
    transformers.AutoProcessor = _Auto
    transformers.pipeline = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_the_depth_loader_reads_the_pinned_local_snapshot(monkeypatch, stores, no_network, tmp_path):
    from opencut.core import depth_anything_v2

    snapshot = _place_hf_snapshot(stores, DEPTH_SMALL)
    loaded = []
    _fake_ml_modules(monkeypatch, loaded)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0")

    with pytest.raises(RuntimeError, match="stop after load"):
        depth_anything_v2.estimate_depth(str(video), model="small", output_path=str(tmp_path / "d.mp4"))

    # The image processor, then the model.
    assert loaded == [str(snapshot), str(snapshot)]


@pytest.mark.parametrize("entry", ["estimate_depth", "generate_parallax"])
def test_the_depth_tools_in_local_only_mode_keep_the_settings_advice(
    monkeypatch, stores, no_network, tmp_path, entry,
):
    from opencut.core import depth_anything_v2

    _fake_ml_modules(monkeypatch, [])
    _local_only(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0")
    output = tmp_path / "out.mp4"
    output.write_bytes(b"\0")

    # Not wrapped in a generic RuntimeError: the job record keeps the code.
    with pytest.raises(LocalOnlyNetworkError) as caught:
        getattr(depth_anything_v2, entry)(str(video), output_path=str(output))

    assert caught.value.code == "LOCAL_ONLY_NETWORK_BLOCKED"
    assert not output.exists()


def test_crisper_whisper_in_local_only_mode_reports_the_missing_model(
    monkeypatch, stores, no_network, tmp_path,
):
    from opencut.core import crisper_whisper

    _fake_ml_modules(monkeypatch, [])
    _local_only(monkeypatch)
    fallback = []
    monkeypatch.setattr(
        crisper_whisper, "_fallback_filler_detection", lambda *a, **k: fallback.append(a) or {},
    )

    with pytest.raises(LocalOnlyNetworkError):
        crisper_whisper.detect_fillers_crisper(str(tmp_path / "talk.wav"))

    assert fallback == []


# ---------------------------------------------------------------------------
# deepface engine
# ---------------------------------------------------------------------------

def test_deepface_runs_on_the_onnx_engine_when_it_has_one(monkeypatch):
    monkeypatch.delenv("DEEPFACE_BACKEND_ENGINE", raising=False)
    monkeypatch.setattr(model_weights, "_deepface_has_onnx_engine", lambda: True)

    assert model_weights.prepare_deepface() == "onnx"
    assert os.environ["DEEPFACE_BACKEND_ENGINE"] == "onnx"
    assert model_weights._emotion_target()[0].url.endswith("facial_expression_model_weights.onnx")


def test_an_older_deepface_fetches_the_tensorflow_weights(monkeypatch):
    monkeypatch.delenv("DEEPFACE_BACKEND_ENGINE", raising=False)
    monkeypatch.setattr(model_weights, "_deepface_has_onnx_engine", lambda: False)

    assert model_weights.prepare_deepface() == "tensorflow"
    assert "DEEPFACE_BACKEND_ENGINE" not in os.environ
    assert model_weights._emotion_target()[0].url.endswith(".h5")


def test_a_chosen_deepface_engine_is_respected(monkeypatch):
    monkeypatch.setenv("DEEPFACE_BACKEND_ENGINE", "pytorch")
    monkeypatch.setattr(model_weights, "_deepface_has_onnx_engine", lambda: True)

    assert model_weights.prepare_deepface() == "pytorch"
    assert model_weights._emotion_target()[0].url.endswith(".pth")
