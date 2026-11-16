"""Model weights the add-on features load, pinned and fetched up front.

Installing an add-on used to bring only its Python packages. The weights came
later, on the first job, from whatever the library downloaded: a mutable Hub
branch or an unverified URL, and in local-only mode a failure halfway through
the job. Readiness said "ready" the whole time.

This module pins each feature's weights (a Hub commit with the expected file
sizes and sha256, or an HTTPS URL with its sha256), reports whether they're on
disk, and fetches them through those checks. The install routes fetch them
after the packages; the loaders call :func:`hf_model_dir` or :func:`ensure`
first and load from the returned local path, so a load never reaches the
network on its own. In local-only mode a missing download raises
:class:`~opencut.network_policy.LocalOnlyNetworkError` before any I/O.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger("opencut")

Progress = Optional[Callable[[int, str], None]]

_CHUNK = 1024 * 1024


class WeightsError(RuntimeError):
    """A download didn't match its pinned size or digest."""


@dataclass(frozen=True)
class HfModel:
    """A Hub repository pinned to one commit and the files a loader reads."""

    repo_id: str
    revision: str
    # (filename, size in bytes, sha256 or "" for small non-LFS files)
    files: Tuple[Tuple[str, int, str], ...]

    @property
    def size_bytes(self) -> int:
        return sum(size for _name, size, _sha in self.files)

    @property
    def filenames(self) -> Tuple[str, ...]:
        return tuple(name for name, _size, _sha in self.files)


@dataclass(frozen=True)
class UrlFile:
    """One HTTPS download with a pinned digest, optionally a zip to unpack."""

    url: str
    sha256: str
    size_bytes: int
    # For a zip: the (member, size) pairs expected after extraction.
    members: Tuple[Tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Component:
    """The weights one install route's feature needs."""

    name: str
    label: str
    # Capability key that turns false while the weights are missing. None
    # when the feature runs without them (multimodal diarization falls back
    # to other face detectors).
    capability: Optional[str]
    # Import names whose presence means the add-on packages are installed,
    # used when there is no capability key to read.
    packages: Tuple[str, ...] = ()


# Commit hashes and file digests read from the Hub API on 2026-11-15.
_DEPTH_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
HF_MODELS: Dict[str, HfModel] = {
    _DEPTH_SMALL: HfModel(
        _DEPTH_SMALL,
        "5426e4f0f36572d16453bbda7a8389317b1bef99",
        (
            ("config.json", 950, ""),
            ("preprocessor_config.json", 775, ""),
            ("model.safetensors", 99_173_660,
             "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"),
        ),
    ),
    "depth-anything/Depth-Anything-V2-Base-hf": HfModel(
        "depth-anything/Depth-Anything-V2-Base-hf",
        "b1958afc87fb45a9e3746cb387596094de553ed8",
        (
            ("config.json", 953, ""),
            ("preprocessor_config.json", 775, ""),
            ("model.safetensors", 389_916_980,
             "9996d15c334d264cfcc29f4318d7d3789a006ecc444782c2fb82f7eaa1acb8bf"),
        ),
    ),
    "depth-anything/Depth-Anything-V2-Large-hf": HfModel(
        "depth-anything/Depth-Anything-V2-Large-hf",
        "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
        (
            ("config.json", 1_431, ""),
            ("preprocessor_config.json", 775, ""),
            ("model.safetensors", 1_341_322_868,
             "4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9"),
        ),
    ),
    # training_args.bin (a pickle), the README and the training logs stay out.
    "nyrahealth/CrisperWhisper": HfModel(
        "nyrahealth/CrisperWhisper",
        "5a513ffabb3429bb61da3521c4e24fbd33ec3e7c",
        (
            ("added_tokens.json", 34_648, ""),
            ("config.json", 1_297, ""),
            ("generation_config.json", 76_117, ""),
            ("merges.txt", 365_730, ""),
            ("model.safetensors", 3_219_908_024,
             "ab370a321fa94ce08d419959813540a4f9b0239b88b2f4207ae28ef6607083c0"),
            ("normalizer.json", 52_666, ""),
            ("preprocessor_config.json", 340, ""),
            ("special_tokens_map.json", 2_186, ""),
            ("tokenizer_config.json", 283_982, ""),
            ("vocab.json", 878_445, ""),
        ),
    ),
}


def depth_model_id(size: str) -> str:
    """Repository for a Depth Anything V2 size, defaulting to Small."""
    size = str(size or "").strip().lower()
    if size not in ("small", "base", "large"):
        size = "small"
    return f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"


# deepface picks TensorFlow, then PyTorch, then ONNX Runtime by what is
# installed, and each engine loads a different emotion file. OpenCut pins the
# ONNX engine when the installed deepface has one, so the file is known.
_DEEPFACE_RELEASE = "https://github.com/serengil/deepface_models/releases/download/v1.0/"
DEEPFACE_EMOTION: Dict[str, Tuple[str, UrlFile]] = {
    "onnx": ("facial_expression_model_weights.onnx", UrlFile(
        _DEEPFACE_RELEASE + "facial_expression_model_weights.onnx",
        "2131942f16003c515f6b1fb09827f9eef111048f033bcc477fbb71f0d446f2a9",
        5_947_004,
    )),
    "pytorch": ("facial_expression_model_weights.pth", UrlFile(
        _DEEPFACE_RELEASE + "facial_expression_model_weights.pth",
        "145a56e2e1f74bc3096bf5ef0091a60c92fafefce21cbe227e0562bd716f0045",
        5_949_382,
    )),
    "tensorflow": ("facial_expression_model_weights.h5", UrlFile(
        _DEEPFACE_RELEASE + "facial_expression_model_weights.h5",
        "e8e8851d3fa05c001b1c27fd8841dfe08d7f82bb786a53ad8776725b7a1e824c",
        5_977_392,
    )),
}
_DEEPFACE_ENGINE_ENV = "DEEPFACE_BACKEND_ENGINE"

# InsightFace's default model pack, unpacked where FaceAnalysis looks for it.
INSIGHTFACE_BUFFALO_L = UrlFile(
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    288_621_354,
    members=(
        ("1k3d68.onnx", 143_607_619),
        ("2d106det.onnx", 5_030_888),
        ("det_10g.onnx", 16_923_827),
        ("genderage.onnx", 1_322_532),
        ("w600k_r50.onnx", 174_383_860),
    ),
)

COMPONENTS: Dict[str, Component] = {
    "depth_effects": Component("depth_effects", "Depth Anything V2 Small", "depth_effects"),
    "crisper_whisper": Component("crisper_whisper", "CrisperWhisper", "crisper_whisper"),
    "emotion_highlights": Component("emotion_highlights", "deepface emotion model", "deepface"),
    "multimodal_diarize": Component(
        "multimodal_diarize", "InsightFace buffalo_l", None, packages=("insightface",),
    ),
}
_COMPONENT_HF = {
    "depth_effects": _DEPTH_SMALL,
    "crisper_whisper": "nyrahealth/CrisperWhisper",
}

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def _expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def hf_cache_dir() -> str:
    """``huggingface_hub.constants.HF_HUB_CACHE``, worked out without importing it.

    ``/health`` reads weight readiness on every call, and importing
    huggingface_hub there pulls in its HTTP stack (and would take the server
    down with a broken install). The precedence is huggingface_hub 1.28's:
    ``HF_HUB_CACHE``, the legacy ``HUGGINGFACE_HUB_CACHE``, ``HF_HOME/hub``,
    ``$XDG_CACHE_HOME/huggingface/hub``, then ``~/.cache/huggingface/hub``.
    Every module that reads the Hub cache asks here, so none of them looks
    somewhere the loaders don't.
    """
    cache_root = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    hf_home = _expand_path(os.getenv("HF_HOME", os.path.join(cache_root, "huggingface")))
    legacy = os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    return _expand_path(os.getenv("HF_HUB_CACHE", legacy))


def _hf_repo_dir(repo_id: str) -> str:
    return os.path.join(hf_cache_dir(), "models--" + repo_id.replace("/", "--"))


def _hf_snapshot_dir(model: HfModel) -> str:
    return os.path.join(_hf_repo_dir(model.repo_id), "snapshots", model.revision)


def _deepface_weights_dir() -> str:
    # deepface.commons.folder_utils.get_deepface_home()
    home = os.getenv("DEEPFACE_HOME", default=os.path.expanduser("~"))
    return os.path.join(str(home), ".deepface", "weights")


def insightface_model_dir(name: str = "buffalo_l") -> str:
    """Where ``insightface.app.FaceAnalysis`` looks for a model pack."""
    return os.path.join(os.path.expanduser("~"), ".insightface", "models", name)


def _deepface_has_onnx_engine() -> Optional[bool]:
    """True when the installed deepface ships the ONNX emotion model, None if absent."""
    try:
        files = importlib.metadata.files("deepface")
    except importlib.metadata.PackageNotFoundError:
        return None
    return any(
        str(path).replace("\\", "/").endswith("deepface/models/demography/onnx/Emotion.py")
        for path in (files or ())
    )


def deepface_engine() -> str:
    """The engine deepface will use for emotion analysis inside OpenCut."""
    forced = os.environ.get(_DEEPFACE_ENGINE_ENV, "").strip().lower()
    if forced in DEEPFACE_EMOTION:
        return forced
    # An older deepface without the ONNX engine only runs on TensorFlow.
    return "tensorflow" if _deepface_has_onnx_engine() is False else "onnx"


def prepare_deepface() -> str:
    """Pin deepface to the engine whose weights :func:`ensure` fetched.

    Call before ``from deepface import DeepFace``; the engine is fixed when
    that module is first imported.
    """
    engine = deepface_engine()
    if engine == "onnx" and not os.environ.get(_DEEPFACE_ENGINE_ENV):
        os.environ[_DEEPFACE_ENGINE_ENV] = "onnx"
    return engine


def _emotion_target() -> Tuple[UrlFile, str]:
    filename, spec = DEEPFACE_EMOTION[deepface_engine()]
    return spec, os.path.join(_deepface_weights_dir(), filename)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def _file_has_size(path: str, size: int) -> bool:
    try:
        return os.path.getsize(path) == size
    except OSError:
        return False


def hf_model_ready(repo_id: str) -> bool:
    model = HF_MODELS[repo_id]
    snapshot = _hf_snapshot_dir(model)
    return all(
        _file_has_size(os.path.join(snapshot, name), size)
        for name, size, _sha in model.files
    )


def _zip_ready(spec: UrlFile, dest_dir: str) -> bool:
    return all(
        _file_has_size(os.path.join(dest_dir, name), size)
        for name, size in spec.members
    )


def component_ready(name: str) -> bool:
    """True when every weight file the component's feature loads is on disk."""
    if name in _COMPONENT_HF:
        return hf_model_ready(_COMPONENT_HF[name])
    if name == "emotion_highlights":
        spec, path = _emotion_target()
        return _file_has_size(path, spec.size_bytes)
    if name == "multimodal_diarize":
        return _zip_ready(INSIGHTFACE_BUFFALO_L, insightface_model_dir())
    raise KeyError(name)


def download_bytes(name: str) -> int:
    if name in _COMPONENT_HF:
        return HF_MODELS[_COMPONENT_HF[name]].size_bytes
    if name == "emotion_highlights":
        return _emotion_target()[0].size_bytes
    if name == "multimodal_diarize":
        return INSIGHTFACE_BUFFALO_L.size_bytes
    raise KeyError(name)


def _packages_present(component: Component, caps: dict) -> bool:
    if component.capability:
        return bool(caps.get(component.capability))
    try:
        return all(importlib.util.find_spec(mod) is not None for mod in component.packages)
    except (ImportError, ValueError):
        return False


def apply_to_capabilities(caps: dict) -> dict:
    """Add ``model_weights`` to *caps* and mark features without weights unready.

    ``caps["model_weights"]["components"][name]`` is ``{"label", "packages",
    "state", "download_mb"}``: ``packages`` says the add-on's Python packages
    are installed, ``state`` is ``"ready"`` or ``"missing"`` for the weights.
    A feature whose packages are in place but whose weights aren't reports
    false, so nothing offers to run it until the download is done.
    """
    from opencut.config import is_local_only

    entries = {}
    for component in COMPONENTS.values():
        packages = _packages_present(component, caps)
        try:
            ready = component_ready(component.name)
        except Exception as exc:  # noqa: BLE001 - readiness must answer
            logger.debug("weights readiness for %s failed: %s", component.name, exc)
            ready = False
        entries[component.name] = {
            "label": component.label,
            "packages": packages,
            "state": "ready" if ready else "missing",
            "download_mb": max(1, round(download_bytes(component.name) / 1_000_000)),
        }
        if component.capability and packages and not ready:
            caps[component.capability] = False
    caps["model_weights"] = {"local_only": bool(is_local_only()), "components": entries}
    return caps


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _refuse_in_local_only(target: str) -> None:
    from opencut.config import is_local_only

    if is_local_only():
        from opencut.network_policy import LocalOnlyNetworkError

        raise LocalOnlyNetworkError(
            target,
            feature="Model download",
            local_alternative=(
                "Turn off local-only mode in Settings, download the model with the "
                "feature's Install button, then turn local-only mode back on."
            ),
        )


def scaled_progress(on_progress: Progress, start: int, end: int) -> Progress:
    """Map a 0-100 download onto *start*..*end* of a job's own progress."""
    if on_progress is None:
        return None
    span = max(0, end - start)
    return lambda pct, message: on_progress(start + pct * span // 100, message)


def _report(on_progress: Progress, pct: int, message: str) -> None:
    if on_progress is None:
        return
    try:
        on_progress(max(0, min(100, int(pct))), message)
    except Exception:  # noqa: BLE001 - progress must never break a download
        logger.debug("weights progress callback failed", exc_info=True)


def _mb(num: int) -> str:
    return f"{num / 1_000_000:,.0f} MB"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_bytes(path: str) -> int:
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _watch_hf_download(
    blobs_dir: str, total: int, label: str, on_progress: Progress, stop: threading.Event,
) -> None:
    """Turn the growth of the Hub cache's blob folder into progress reports."""
    baseline = _dir_bytes(blobs_dir)
    while not stop.wait(1.0):
        done = max(0, _dir_bytes(blobs_dir) - baseline)
        pct = min(99, int(done * 100 / total)) if total else 0
        _report(on_progress, pct, f"Downloading {label}: {_mb(done)} of {_mb(total)}")


def _verify_hf(model: HfModel, snapshot: str) -> None:
    for name, size, sha in model.files:
        path = os.path.join(snapshot, name)
        if not _file_has_size(path, size):
            raise WeightsError(f"{model.repo_id}/{name} is not the pinned size ({size} bytes)")
        if sha and _sha256_file(path) != sha:
            raise WeightsError(f"{model.repo_id}/{name} does not match its pinned sha256")


def hf_model_dir(repo_id: str, *, on_progress: Progress = None, label: str = "") -> str:
    """Local snapshot folder for a pinned Hub model, downloading it if needed.

    Loaders pass the returned folder to ``from_pretrained`` or ``pipeline``,
    so loading never reaches the network itself.
    """
    model = HF_MODELS[repo_id]
    snapshot = _hf_snapshot_dir(model)
    if hf_model_ready(repo_id):
        return snapshot
    _refuse_in_local_only(f"https://huggingface.co/{repo_id}")
    label = label or repo_id
    with _lock_for(repo_id):
        if hf_model_ready(repo_id):
            return snapshot
        from opencut.core.model_safety import safe_snapshot_download

        _report(on_progress, 0, f"Downloading {label} ({_mb(model.size_bytes)})...")
        blobs = os.path.join(_hf_repo_dir(repo_id), "blobs")
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_hf_download,
            args=(blobs, model.size_bytes, label, on_progress, stop),
            name=f"weights-{repo_id}",
            daemon=True,
        )
        watcher.start()
        try:
            path = safe_snapshot_download(
                repo_id,
                revision=model.revision,
                allow_patterns=list(model.filenames),
            )
        finally:
            stop.set()
            watcher.join(timeout=5)
        _report(on_progress, 99, f"Checking {label}...")
        _verify_hf(model, path)
        _report(on_progress, 100, f"{label} is ready.")
        return path


def _require_https(url: str) -> None:
    if urlsplit(url).scheme.lower() != "https":
        raise WeightsError(f"Refusing a model download that isn't HTTPS: {url}")


def _download(spec: UrlFile, dest: str, label: str, on_progress: Progress) -> None:
    """Stream *spec* to *dest*, checking size and sha256 before it lands."""
    from urllib.request import Request, urlopen

    from opencut import __version__

    _require_https(spec.url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    digest = hashlib.sha256()
    done = 0
    _report(on_progress, 0, f"Downloading {label} ({_mb(spec.size_bytes)})...")
    try:
        request = Request(spec.url, headers={"User-Agent": f"OpenCut/{__version__}"})
        with urlopen(request, timeout=60) as response, open(part, "wb") as out:
            _require_https(response.geturl())
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                done += len(chunk)
                if done > spec.size_bytes:
                    raise WeightsError(f"{label} is larger than its pinned size")
                digest.update(chunk)
                out.write(chunk)
                _report(
                    on_progress,
                    done * 99 // spec.size_bytes,
                    f"Downloading {label}: {_mb(done)} of {_mb(spec.size_bytes)}",
                )
        if done != spec.size_bytes:
            raise WeightsError(f"{label} download stopped at {done} of {spec.size_bytes} bytes")
        if digest.hexdigest() != spec.sha256:
            raise WeightsError(f"{label} does not match its pinned sha256")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                logger.debug("could not remove %s", part, exc_info=True)


def _fetch_zip(spec: UrlFile, dest_dir: str, label: str, on_progress: Progress) -> None:
    from opencut.core.archive_safety import safe_extract_all

    archive = dest_dir + ".zip.download"
    try:
        _download(spec, archive, label, on_progress)
        _report(on_progress, 99, f"Unpacking {label}...")
        safe_extract_all(archive, dest_dir)
    finally:
        if os.path.exists(archive):
            os.remove(archive)
    if not _zip_ready(spec, dest_dir):
        raise WeightsError(f"{label} unpacked without the files it should contain")


def ensure(name: str, *, on_progress: Progress = None) -> str:
    """Make sure a component's weights are on disk; return their local path.

    A no-op when they already are. In local-only mode a missing download
    raises :class:`~opencut.network_policy.LocalOnlyNetworkError` before any
    network access.
    """
    component = COMPONENTS[name]
    if name in _COMPONENT_HF:
        return hf_model_dir(_COMPONENT_HF[name], on_progress=on_progress, label=component.label)
    if name == "emotion_highlights":
        spec, path = _emotion_target()
        if _file_has_size(path, spec.size_bytes):
            return path
        _refuse_in_local_only(spec.url)
        with _lock_for(path):
            if not _file_has_size(path, spec.size_bytes):
                _download(spec, path, component.label, on_progress)
        _report(on_progress, 100, f"{component.label} is ready.")
        return path
    if name == "multimodal_diarize":
        dest = insightface_model_dir()
        if _zip_ready(INSIGHTFACE_BUFFALO_L, dest):
            return dest
        _refuse_in_local_only(INSIGHTFACE_BUFFALO_L.url)
        with _lock_for(dest):
            if not _zip_ready(INSIGHTFACE_BUFFALO_L, dest):
                _fetch_zip(INSIGHTFACE_BUFFALO_L, dest, component.label, on_progress)
        _report(on_progress, 100, f"{component.label} is ready.")
        return dest
    raise KeyError(name)
