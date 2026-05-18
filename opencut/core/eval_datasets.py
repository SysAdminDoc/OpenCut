"""F176 — public eval dataset catalogue.

Wave H+ model surfaces (video generation, restoration, VSR, dubbing,
lip-sync, ASR, etc.) need standardised eval benchmarks. F176 captures
the canonical public datasets OpenCut **could** download for
reproducible eval runs, without actually committing the weights.

This module is a registry only — no download is performed unless the
operator explicitly opts in via ``OPENCUT_DOWNLOAD_EVAL=1`` and runs
``opencut.tools.download_eval_dataset`` (which lands in a future
pass). Today the registry serves three purposes:

1. The Wave-H "ai-eval" harness (F120 + F178) records which dataset
   each evaluation ran against; the dataset ID is resolved to a
   canonical entry here so the reported result is reproducible.
2. ``docs/MODELS.md`` and the panel can list the supported eval
   benchmarks alongside each model card.
3. The F176 release-smoke gate fails closed if a card lists an
   unknown dataset ID (catches typos), and ensures every entry has a
   license + an upstream URL.

Each entry carries the dataset's licence so the operator can decide
whether to opt-in based on commercial / research / NC posture. We
**never** auto-download non-commercial corpora.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

from opencut.openapi_registry import openapi_response_schema

# Modality tags — used by the panel to filter the list.
MODALITIES: Tuple[str, ...] = (
    "video",
    "image",
    "audio",
    "speech",
    "music",
    "captions",
    "interchange",
    "provenance",
)


@dataclass(frozen=True)
@openapi_response_schema("/system/eval-datasets/<dataset_id>")
class EvalDataset:
    """One public eval dataset OpenCut benchmarks against."""

    dataset_id: str               # snake_case, unique
    label: str                    # human-readable
    modality: str                 # one of MODALITIES
    benchmark_targets: Tuple[str, ...]  # capability tags: 't2v', 'vsr', 'denoise', etc.
    upstream: str                 # canonical project URL (paper / GitHub / HF)
    download_url: str = ""        # optional direct asset URL; empty if behind a manual flow
    license: str = "see upstream"
    license_notes: str = ""
    size_gb: float = 0.0          # rough on-disk footprint (0 = unknown / metadata only)
    sha256: str = ""              # optional integrity hash for downloaded asset
    citation: str = ""            # short bibliographic reference
    commercial_use_ok: bool = False  # the operator's responsibility to confirm
    # Per the F176 contract, "auto" datasets can be fetched by the
    # opt-in `OPENCUT_DOWNLOAD_EVAL=1` runner. "manual" datasets
    # require a human signing the licence/EULA before download.
    acquisition: str = "manual"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Sorted by (modality, dataset_id) for deterministic dumps. Add new
# entries alphabetically within the right block.


DATASETS: Tuple[EvalDataset, ...] = (
    # ---- video ---------------------------------------------------
    EvalDataset(
        dataset_id="davis_2017",
        label="DAVIS 2017",
        modality="video",
        benchmark_targets=("matte", "video_object_segmentation"),
        upstream="https://davischallenge.org/davis2017/code.html",
        download_url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
        license="CC-BY-4.0",
        size_gb=0.8,
        commercial_use_ok=True,
        acquisition="auto",
        citation="Pont-Tuset et al., 'The 2017 DAVIS Challenge on Video Object Segmentation', arXiv:1704.00675",
    ),
    EvalDataset(
        dataset_id="reds_120",
        label="REDS (Realistic and Dynamic Scenes)",
        modality="video",
        benchmark_targets=("vsr", "denoise", "deblur"),
        upstream="https://seungjunnah.github.io/Datasets/reds",
        license="CC-BY-4.0",
        license_notes="Research + commercial OK per dataset README.",
        size_gb=39.0,
        commercial_use_ok=True,
        acquisition="manual",
        citation="Nah et al., 'NTIRE 2019 Challenge on Video Deblurring and Super-Resolution', CVPRW 2019",
    ),
    EvalDataset(
        dataset_id="spring_2024",
        label="Spring optical-flow / scene-flow benchmark",
        modality="video",
        benchmark_targets=("optical_flow", "scene_flow", "matte"),
        upstream="https://spring-benchmark.org/",
        license="CC-BY-NC-SA-4.0",
        license_notes="Non-commercial; research eval only.",
        size_gb=37.0,
        commercial_use_ok=False,
        acquisition="manual",
        citation="Mehl et al., 'Spring: A High-Resolution High-Detail Dataset', CVPR 2023",
    ),
    EvalDataset(
        dataset_id="vbench",
        label="VBench (video generation quality)",
        modality="video",
        benchmark_targets=("t2v", "i2v", "video_quality"),
        upstream="https://github.com/Vchitect/VBench",
        license="Apache-2.0",
        size_gb=0.5,
        commercial_use_ok=True,
        acquisition="auto",
        citation="Huang et al., 'VBench: Comprehensive Benchmark Suite for Video Generative Models', CVPR 2024",
    ),
    EvalDataset(
        dataset_id="vfi_2024",
        label="VFI-2024 (video frame interpolation)",
        modality="video",
        benchmark_targets=("frame_interpolation",),
        upstream="https://github.com/JihyongOh/VFI-Bench",
        license="MIT",
        size_gb=12.0,
        commercial_use_ok=True,
        acquisition="manual",
        citation="Oh et al., 'VFI-Bench: Benchmark for Video Frame Interpolation', 2024",
    ),

    # ---- speech / dubbing ----------------------------------------
    EvalDataset(
        dataset_id="libri_tts",
        label="LibriTTS",
        modality="speech",
        benchmark_targets=("tts", "voice_clone", "asr"),
        upstream="https://www.openslr.org/60/",
        download_url="https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        license="CC-BY-4.0",
        size_gb=24.0,
        commercial_use_ok=True,
        acquisition="auto",
        citation="Zen et al., 'LibriTTS: A Corpus Derived from LibriSpeech', Interspeech 2019",
    ),
    EvalDataset(
        dataset_id="lrs3",
        label="LRS3 (Lip Reading Sentences 3)",
        modality="speech",
        benchmark_targets=("lip_sync", "lip_reading", "av_speech"),
        upstream="https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html",
        license="Custom (Oxford VGG)",
        license_notes="Research only; requires signed agreement.",
        size_gb=51.0,
        commercial_use_ok=False,
        acquisition="manual",
        citation="Afouras et al., 'LRS3-TED: a large-scale dataset for visual speech recognition', arXiv:1809.00496",
    ),
    EvalDataset(
        dataset_id="voxceleb2",
        label="VoxCeleb 2",
        modality="speech",
        benchmark_targets=("speaker_id", "diarisation", "voice_clone"),
        upstream="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
        license="CC-BY-4.0",
        license_notes="Audio under CC-BY; video YouTube terms apply.",
        size_gb=300.0,
        commercial_use_ok=False,
        acquisition="manual",
        citation="Chung et al., 'VoxCeleb2: Deep Speaker Recognition', Interspeech 2018",
    ),

    # ---- music ---------------------------------------------------
    EvalDataset(
        dataset_id="musdb18_hq",
        label="MUSDB18-HQ",
        modality="music",
        benchmark_targets=("stem_separation", "music_demix"),
        upstream="https://sigsep.github.io/datasets/musdb.html",
        license="CC-BY-NC-SA-4.0",
        license_notes="Non-commercial; demixing research benchmark.",
        size_gb=30.0,
        commercial_use_ok=False,
        acquisition="manual",
        citation="Rafii et al., 'MUSDB18-HQ', Zenodo 2019",
    ),
    EvalDataset(
        dataset_id="ebu_sqam",
        label="EBU SQAM (Sound Quality Assessment Material)",
        modality="audio",
        benchmark_targets=("loudness", "audio_quality", "codec_eval"),
        upstream="https://tech.ebu.ch/publications/sqamcd",
        license="EBU TC.045 (free for research + reference)",
        license_notes="Free reproduction for technical research.",
        size_gb=0.7,
        commercial_use_ok=True,
        acquisition="manual",
        citation="EBU Tech 3253 — Sound Quality Assessment Material recordings",
    ),

    # ---- captions / interchange / provenance ---------------------
    EvalDataset(
        dataset_id="netflix_open_content",
        label="Netflix Open Content (Cosmos Laundromat, El Fuente, Meridian)",
        modality="video",
        benchmark_targets=("encoding", "hdr", "codec_eval", "imsc_captions"),
        upstream="https://opencontent.netflix.com/",
        license="CC-BY-NC-ND-4.0",
        license_notes="Non-commercial; encoder research only.",
        size_gb=12.0,
        commercial_use_ok=False,
        acquisition="manual",
        citation="Netflix Open Content programme",
    ),
    EvalDataset(
        dataset_id="imsc_reference",
        label="W3C IMSC 1.x reference test suite",
        modality="captions",
        benchmark_targets=("imsc", "caption_compliance", "ttml"),
        upstream="https://www.w3.org/wiki/TimedText/IMSC",
        license="W3C Document License",
        size_gb=0.05,
        commercial_use_ok=True,
        acquisition="auto",
        citation="W3C Timed Text Working Group IMSC reference materials",
    ),
    EvalDataset(
        dataset_id="c2pa_test_vectors",
        label="C2PA test vectors (provenance)",
        modality="provenance",
        benchmark_targets=("c2pa_verify", "provenance"),
        upstream="https://github.com/contentauth/c2pa-test-files",
        license="Apache-2.0",
        size_gb=0.2,
        commercial_use_ok=True,
        acquisition="auto",
        citation="Coalition for Content Provenance and Authenticity test vectors",
    ),
)


def list_datasets() -> List[dict]:
    """Return every dataset as a JSON-friendly dict."""
    return [d.as_dict() for d in DATASETS]


def get_dataset(dataset_id: str) -> Optional[EvalDataset]:
    """Return one dataset entry by ID, or ``None`` if unknown."""
    for d in DATASETS:
        if d.dataset_id == dataset_id:
            return d
    return None


def datasets_for_target(target: str) -> List[EvalDataset]:
    """Return every dataset whose ``benchmark_targets`` includes ``target``."""
    return [d for d in DATASETS if target in d.benchmark_targets]


def datasets_for_modality(modality: str) -> List[EvalDataset]:
    return [d for d in DATASETS if d.modality == modality]


def commercial_safe_datasets() -> List[EvalDataset]:
    """Return only datasets whose licence allows commercial eval."""
    return [d for d in DATASETS if d.commercial_use_ok]


def download_opt_in() -> bool:
    """Return True when the operator has explicitly opted into downloads."""
    return os.environ.get("OPENCUT_DOWNLOAD_EVAL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def manifest(*, include_metadata: bool = True) -> dict:
    """Return the registry as a structured manifest dict.

    The ``include_metadata`` switch keeps the public ``GET /system/eval-datasets``
    response compact for the panel while letting documentation tools
    grab the full citation block.
    """
    payload: dict = {
        "version": 1,
        "modalities": list(MODALITIES),
        "count": len(DATASETS),
        "auto_download_count": sum(1 for d in DATASETS if d.acquisition == "auto"),
        "commercial_safe_count": sum(1 for d in DATASETS if d.commercial_use_ok),
        "datasets": [d.as_dict() for d in DATASETS],
    }
    if not include_metadata:
        # Trim verbose fields for compact responses.
        compact: List[dict] = []
        for d in payload["datasets"]:
            compact.append({
                "dataset_id": d["dataset_id"],
                "label": d["label"],
                "modality": d["modality"],
                "benchmark_targets": list(d["benchmark_targets"]),
                "license": d["license"],
                "acquisition": d["acquisition"],
                "commercial_use_ok": d["commercial_use_ok"],
            })
        payload["datasets"] = compact
    return payload


def assert_registry_invariants(datasets: Optional[Iterable[EvalDataset]] = None) -> None:
    """Validate the registry shape. Raises ``ValueError`` on the first failure."""
    rows = list(datasets) if datasets is not None else list(DATASETS)
    seen_ids: dict = {}
    for d in rows:
        if not d.dataset_id or not d.dataset_id.replace("_", "").isalnum():
            raise ValueError(f"invalid dataset_id: {d.dataset_id!r}")
        if d.dataset_id in seen_ids:
            raise ValueError(f"duplicate dataset_id: {d.dataset_id!r}")
        seen_ids[d.dataset_id] = d.label
        if d.modality not in MODALITIES:
            raise ValueError(f"{d.dataset_id}: unknown modality {d.modality!r}")
        if not d.upstream.startswith(("http://", "https://")):
            raise ValueError(f"{d.dataset_id}: upstream must be an http(s) URL")
        if d.acquisition not in {"auto", "manual"}:
            raise ValueError(f"{d.dataset_id}: acquisition must be 'auto' or 'manual'")
        if not d.license:
            raise ValueError(f"{d.dataset_id}: license is required")
        if not d.benchmark_targets:
            raise ValueError(f"{d.dataset_id}: at least one benchmark_target is required")


__all__ = [
    "DATASETS",
    "MODALITIES",
    "EvalDataset",
    "assert_registry_invariants",
    "commercial_safe_datasets",
    "datasets_for_modality",
    "datasets_for_target",
    "download_opt_in",
    "get_dataset",
    "list_datasets",
    "manifest",
]
