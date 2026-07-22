"""Pinned NVIDIA NeMo ASR model identities used by OpenCut.

The Hugging Face repositories publish mutable ``main`` branches.  OpenCut
loads the exact revisions and checkpoint hashes below so a cache key, support
bundle, and transcript sidecar all identify the bytes that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_NEMO_VERSION = "2.7.3"
NEMO_REQUIREMENT = "nemo_toolkit[asr]>=2.7.3,<2.8"
PINNED_ON = "2026-07-22"


@dataclass(frozen=True)
class NemoModelSpec:
    """Immutable download and runtime identity for one NeMo checkpoint."""

    key: str
    engine: str
    model_id: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    loader: str

    @property
    def download_url(self) -> str:
        return (
            f"https://huggingface.co/{self.model_id}/resolve/"
            f"{self.revision}/{self.filename}"
        )

    @property
    def size_mb(self) -> int:
        return round(self.size_bytes / (1024 * 1024))


PARAKEET_SPEC = NemoModelSpec(
    key="parakeet-tdt-0.6b-v3",
    engine="parakeet-tdt",
    model_id="nvidia/parakeet-tdt-0.6b-v3",
    revision="7c35754d166cca382ad1e53e68b01e7c575f3a1d",
    filename="parakeet-tdt-0.6b-v3.nemo",
    sha256="3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d",
    size_bytes=2_509_332_480,
    loader="asr",
)

CANARY_SPEC = NemoModelSpec(
    key="canary-1b-flash",
    engine="canary-1b-flash",
    model_id="nvidia/canary-1b-flash",
    revision="2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e",
    filename="canary-1b-flash.nemo",
    sha256="3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324",
    size_bytes=3_540_715_520,
    loader="multitask",
)

NEMO_MODEL_SPECS = {
    PARAKEET_SPEC.key: PARAKEET_SPEC,
    CANARY_SPEC.key: CANARY_SPEC,
}


__all__ = [
    "CANARY_SPEC",
    "MIN_NEMO_VERSION",
    "NEMO_MODEL_SPECS",
    "NEMO_REQUIREMENT",
    "NemoModelSpec",
    "PARAKEET_SPEC",
    "PINNED_ON",
]
