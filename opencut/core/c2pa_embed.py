"""
OpenCut C2PA Content Credentials Module (27.1)

Create and embed C2PA (Coalition for Content Provenance and Authenticity)
manifests that record AI operations applied to media, providing
tamper-evident content provenance.
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from opencut import __version__
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

# C2PA manifest version
C2PA_VERSION = "2.4"
C2PA_CLAIM_GENERATOR = "OpenCut Video Editor"
C2PA_ACTIONS_LABEL = "c2pa.actions.v2"
C2PA_AI_DISCLOSURE_LABEL = "c2pa.ai-disclosure"
C2PA_DEFAULT_MODEL_TYPE = "c2pa.types.model"

# IPTC digital-source-type vocabulary (C2PA 2.4 AI transparency). Fully
# synthesized output is trainedAlgorithmicMedia; AI-assisted edits of real
# footage are a composite with trained-algorithmic media.
_DST_TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
_DST_COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"

# Action keywords that indicate an AI/ML step.
_AI_ACTION_KEYWORDS = ("ai", "auto", "generate", "enhance", "ml", "upscale", "relight", "synth")
# Keywords that indicate the step fully generated new media (vs. editing real).
_GENERATIVE_KEYWORDS = ("generate", "synth", "t2i", "t2v", "text_to")
_HUMAN_OVERSIGHT_VALUES = {"fully_autonomous", "prompt_guided", "human_validated"}
_ENTITY_ACTION_RE = re.compile(r"^(?:[A-Za-z0-9_-]+\.){2,}[A-Za-z0-9_-]+$")
_C2PA_24_ACTIONS = frozenset({
    "c2pa.addedText",
    "c2pa.adjustedColor",
    "c2pa.changedSpeed",
    "c2pa.converted",
    "c2pa.created",
    "c2pa.cropped",
    "c2pa.deleted",
    "c2pa.drawing",
    "c2pa.dubbed",
    "c2pa.edited",
    "c2pa.edited.metadata",
    "c2pa.enhanced",
    "c2pa.filtered",
    "c2pa.mastered",
    "c2pa.mixed",
    "c2pa.opened",
    "c2pa.orientation",
    "c2pa.placed",
    "c2pa.published",
    "c2pa.redacted",
    "c2pa.remixed",
    "c2pa.removed",
    "c2pa.repackaged",
    "c2pa.resized",
    "c2pa.resized.proportional",
    "c2pa.transcoded",
    "c2pa.translated",
    "c2pa.trimmed",
    "c2pa.unknown",
    "c2pa.watermarked.bound",
    "c2pa.watermarked.unbound",
})
_ACTION_ALIASES = {
    "caption": "c2pa.addedText",
    "c2pa.captioned": "c2pa.addedText",
    "c2pa.color_adjustments": "c2pa.adjustedColor",
    "c2pa.transcribed": "c2pa.addedText",
    "export": "c2pa.published",
    "trim": "c2pa.trimmed",
}


def _digital_source_type(action: str) -> str:
    a = (action or "").lower()
    if any(kw in a for kw in _GENERATIVE_KEYWORDS):
        return _DST_TRAINED
    return _DST_COMPOSITE


def _is_ai_action(action: str) -> bool:
    return any(kw in (action or "").lower() for kw in _AI_ACTION_KEYWORDS)


def _standard_action_name(action: str) -> str:
    """Return a C2PA 2.4 action name, preserving the original in parameters."""
    raw = (action or "").strip()
    lowered = raw.lower()
    if lowered in _ACTION_ALIASES:
        return _ACTION_ALIASES[lowered]
    if raw in _C2PA_24_ACTIONS or (
        not raw.startswith("c2pa.") and _ENTITY_ACTION_RE.fullmatch(raw)
    ):
        return raw
    if raw.startswith("c2pa."):
        return "c2pa.unknown"
    if any(keyword in lowered for keyword in _GENERATIVE_KEYWORDS):
        return "c2pa.created"
    if any(keyword in lowered for keyword in ("upscale", "enhance", "denoise", "relight")):
        return "c2pa.enhanced"
    return "c2pa.edited"


def _software_agent(value: Any) -> dict:
    if isinstance(value, dict) and value.get("name"):
        return deepcopy(value)
    text = str(value or C2PA_CLAIM_GENERATOR).strip()
    match = re.match(r"^(?P<name>[^/]+)/(?P<version>\d+\.\d+\.\d+)", text)
    if match:
        return {"name": match.group("name"), "version": match.group("version")}
    if text == C2PA_CLAIM_GENERATOR:
        return {"name": "OpenCut", "version": __version__}
    return {"name": text}


def _operation_value(operation: dict, *keys: str, default: Any = "") -> Any:
    parameters = operation.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    for key in keys:
        if operation.get(key) not in (None, ""):
            return operation[key]
        if parameters.get(key) not in (None, ""):
            return parameters[key]
    return default


def build_c2pa_assertions(
    operations: Optional[List[dict]],
    *,
    claim_generator: str = C2PA_CLAIM_GENERATOR,
    created: str = "",
    source_hash: str = "",
) -> tuple[List[dict], List[dict]]:
    """Build C2PA 2.4 actions/AI assertions plus OpenCut source evidence.

    The source digest is deliberately namespaced metadata, not a C2PA hard
    binding. ``c2patool`` creates and validates the authoritative hard binding
    against the output asset when a manifest is signed.
    """
    action_items: List[dict] = []
    disclosures: List[dict] = []
    seen_disclosures = set()

    for operation in operations or []:
        if not isinstance(operation, dict):
            continue
        raw_action = str(operation.get("action") or "c2pa.unknown").strip()
        action = _standard_action_name(raw_action)
        parameters = operation.get("parameters")
        parameters = deepcopy(parameters) if isinstance(parameters, dict) else {}
        if action != raw_action:
            parameters.setdefault("org.opencut.operation", raw_action)

        source_type = str(
            _operation_value(
                operation,
                "digitalSourceType",
                "digital_source_type",
                default="",
            )
        ).strip()
        model_type = str(
            _operation_value(operation, "modelType", "model_type", default="")
        ).strip()
        is_ai = bool(source_type or model_type or _is_ai_action(raw_action))
        if is_ai and not source_type:
            source_type = _digital_source_type(raw_action)

        item: Dict[str, Any] = {
            "action": action,
            "softwareAgent": _software_agent(
                operation.get("softwareAgent")
                or operation.get("software_agent")
                or claim_generator
            ),
        }
        when = str(
            operation.get("when")
            or operation.get("timestamp")
            or created
            or ""
        ).strip()
        if when:
            item["when"] = when
        description = str(operation.get("description") or "").strip()
        if description:
            item["description"] = description
        if parameters:
            item["parameters"] = parameters
        if source_type:
            item["digitalSourceType"] = source_type
        action_items.append(item)

        if not is_ai:
            continue
        model_type = model_type or C2PA_DEFAULT_MODEL_TYPE
        model_name = str(
            _operation_value(
                operation,
                "modelName",
                "model_name",
                default="OpenCut AI Pipeline",
            )
        ).strip()
        model_identifier = str(
            _operation_value(
                operation,
                "modelIdentifier",
                "model_identifier",
                default="",
            )
        ).strip()
        oversight = str(
            _operation_value(
                operation,
                "humanOversightLevel",
                "human_oversight_level",
                default=("prompt_guided" if source_type == _DST_TRAINED else "human_validated"),
            )
        ).strip()
        if oversight not in _HUMAN_OVERSIGHT_VALUES:
            oversight = "human_validated"
        scientific_domain = _operation_value(
            operation, "scientificDomain", "scientific_domain", default=""
        )
        if isinstance(scientific_domain, str):
            scientific_domains = [scientific_domain] if scientific_domain else []
        elif isinstance(scientific_domain, list):
            scientific_domains = [str(value) for value in scientific_domain if value]
        else:
            scientific_domains = []

        disclosure: Dict[str, Any] = {
            "modelType": model_type,
            "contentProfile": {"humanOversightLevel": oversight},
        }
        if model_name:
            disclosure["modelName"] = model_name
        if model_identifier:
            disclosure["modelIdentifier"] = model_identifier
        if scientific_domains:
            disclosure["scientificDomain"] = scientific_domains
        disclosure_key = json.dumps(disclosure, sort_keys=True, separators=(",", ":"))
        if disclosure_key not in seen_disclosures:
            seen_disclosures.add(disclosure_key)
            disclosures.append(disclosure)

    assertions: List[dict] = []
    if action_items:
        assertions.append({
            "label": C2PA_ACTIONS_LABEL,
            "data": {"actions": action_items, "allActionsIncluded": False},
        })
    assertions.extend(
        {"label": C2PA_AI_DISCLOSURE_LABEL, "data": disclosure}
        for disclosure in disclosures
    )
    if source_hash:
        assertions.append({
            "label": "org.opencut.source",
            "data": {"alg": "sha256", "hash": source_hash},
        })
    return action_items, assertions


def c2patool_manifest_definition(manifest: dict) -> dict:
    """Return the strict declarative subset accepted by ``c2patool``."""
    claim_generator = str(manifest.get("claim_generator") or C2PA_CLAIM_GENERATOR)
    return {
        "claim_generator": claim_generator,
        "claim_generator_info": [_software_agent(claim_generator)],
        "title": str(manifest.get("title") or "OpenCut export"),
        "format": str(manifest.get("format") or "application/octet-stream"),
        "assertions": deepcopy(manifest.get("assertions") or []),
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class C2PAOperation:
    """A single operation recorded in the C2PA manifest."""
    action: str = ""
    software_agent: str = C2PA_CLAIM_GENERATOR
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    description: str = ""


@dataclass
class C2PAManifest:
    """C2PA content credentials manifest."""
    claim_generator: str = C2PA_CLAIM_GENERATOR
    title: str = ""
    instance_id: str = ""
    format: str = "video/mp4"
    operations: List[Dict] = field(default_factory=list)
    source_hash: str = ""
    hash_algorithm: str = "sha256"
    created: str = ""
    assertions: List[Dict] = field(default_factory=list)
    signature: str = ""


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------

def create_c2pa_manifest(
    operations: Optional[List[dict]] = None,
    source_hash: str = "",
    title: str = "",
    format: str = "video/mp4",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a C2PA manifest recording AI/edit operations.

    Args:
        operations: List of operation dicts, each with at least 'action'.
            Supported fields: action, parameters, description.
        source_hash: SHA-256 hash of the source file.
        title: Content title / filename.
        format: MIME type of the content.
        on_progress: Optional callback(pct, msg).

    Returns:
        C2PA manifest dict ready for embedding.
    """
    if on_progress:
        on_progress(10, "Creating C2PA manifest")

    instance_id = f"urn:uuid:{uuid.uuid4()}"
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest_ops, assertions = build_c2pa_assertions(
        operations,
        claim_generator=C2PA_CLAIM_GENERATOR,
        created=created,
        source_hash=source_hash,
    )

    manifest = {
        "c2pa_version": C2PA_VERSION,
        "claim_generator": C2PA_CLAIM_GENERATOR,
        "title": title,
        "instance_id": instance_id,
        "format": format,
        "operations": manifest_ops,
        "source_hash": source_hash,
        "hash_algorithm": "sha256",
        "created": created,
        "assertions": assertions,
    }
    manifest["manifest_definition"] = c2patool_manifest_definition(manifest)

    if on_progress:
        on_progress(50, "C2PA manifest created")

    return manifest


# ---------------------------------------------------------------------------
# Hash a file
# ---------------------------------------------------------------------------

def _hash_file(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Embed C2PA manifest into video
# ---------------------------------------------------------------------------

def embed_c2pa(
    video_path: str,
    manifest: dict,
    output: Optional[str] = None,
    signed_credential: bool = True,
    credential_output: Optional[str] = None,
    c2patool_path: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Sign and embed a real C2PA manifest with ``c2patool``.

    The former FFmpeg-comment surrogate was not a Content Credential and is
    no longer emitted. ``signed_credential`` remains for API compatibility;
    embedding always requires a real signing key/certificate and c2patool.

    Args:
        video_path: Path to input video.
        manifest: C2PA manifest dict (from create_c2pa_manifest).
        output: Output path (auto-generated if None).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, manifest_size, source_hash, and optionally a
        signed C2PA credential sidecar / embedded output summary.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if on_progress:
        on_progress(10, "Computing source hash")

    if not isinstance(manifest, dict):
        raise ValueError("C2PA manifest must be an object")

    source_hash = manifest.get("source_hash", "")
    if not source_hash:
        source_hash = _hash_file(video_path)

    operations = manifest.get("operations")
    if not isinstance(operations, list):
        operations = []
    if not operations:
        for assertion in manifest.get("assertions", []):
            if assertion.get("label") in {"c2pa.actions", C2PA_ACTIONS_LABEL}:
                candidate = assertion.get("data", {}).get("actions", [])
                if isinstance(candidate, list):
                    operations = candidate
                    break
    existing_disclosures = [
        deepcopy(assertion)
        for assertion in manifest.get("assertions", [])
        if isinstance(assertion, dict)
        and assertion.get("label") == C2PA_AI_DISCLOSURE_LABEL
        and isinstance(assertion.get("data"), dict)
        and assertion["data"].get("modelType")
    ]
    created = str(
        manifest.get("created")
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    claim_generator = str(
        manifest.get("claim_generator") or C2PA_CLAIM_GENERATOR
    )
    manifest_ops, assertions = build_c2pa_assertions(
        operations,
        claim_generator=claim_generator,
        created=created,
        source_hash=source_hash,
    )
    if existing_disclosures:
        assertions = [
            assertion
            for assertion in assertions
            if assertion.get("label") != C2PA_AI_DISCLOSURE_LABEL
        ]
        source_index = next(
            (
                index
                for index, assertion in enumerate(assertions)
                if assertion.get("label") == "org.opencut.source"
            ),
            len(assertions),
        )
        assertions[source_index:source_index] = existing_disclosures
    manifest["c2pa_version"] = C2PA_VERSION
    manifest["claim_generator"] = claim_generator
    manifest["operations"] = manifest_ops
    manifest["source_hash"] = source_hash
    manifest["hash_algorithm"] = "sha256"
    manifest["created"] = created
    manifest["assertions"] = assertions
    manifest.pop("signature", None)
    manifest["manifest_definition"] = c2patool_manifest_definition(manifest)

    out = credential_output or output or _output_path(video_path, "_c2pa", "")

    if on_progress:
        on_progress(40, "Signing and embedding C2PA 2.4 manifest")

    from opencut.core.c2pa_sidecar import (
        C2paAction,
        _run_c2patool_embed,
        build_sidecar,
    )

    embedded, method, embedded_path, warnings = _run_c2patool_embed(
        asset=Path(video_path).resolve(),
        manifest=manifest,
        output_path=out,
        c2patool_path=c2patool_path,
    )
    if not embedded:
        detail = "; ".join(warnings) or "unknown c2patool failure"
        raise RuntimeError(f"C2PA embedding failed: {detail}")

    disclosure_data = next(
        (
            assertion.get("data", {})
            for assertion in assertions
            if assertion.get("label") == C2PA_AI_DISCLOSURE_LABEL
        ),
        {},
    )
    content_profile = disclosure_data.get("contentProfile") or {}
    actions = [
        C2paAction(
            action=str(op.get("action") or "c2pa.unknown"),
            when=str(op.get("when") or created),
            parameters=op.get("parameters") or {},
            software_agent=op.get("softwareAgent") or claim_generator,
            digital_source_type=str(op.get("digitalSourceType") or ""),
            model_type=(
                str(disclosure_data.get("modelType") or "")
                if op.get("digitalSourceType") else ""
            ),
            model_name=(
                str(disclosure_data.get("modelName") or "")
                if op.get("digitalSourceType") else ""
            ),
            model_identifier=(
                str(disclosure_data.get("modelIdentifier") or "")
                if op.get("digitalSourceType") else ""
            ),
            human_oversight_level=str(
                content_profile.get("humanOversightLevel") or ""
            ) if op.get("digitalSourceType") else "",
            scientific_domain=(
                disclosure_data.get("scientificDomain") or []
                if op.get("digitalSourceType") else []
            ),
        )
        for op in manifest_ops
    ]
    sidecar_result = build_sidecar(
        asset_path=embedded_path,
        actions=actions,
        title=str(manifest.get("title") or os.path.basename(embedded_path)),
    )
    credential = sidecar_result.as_dict()
    credential.update({
        "embedded": True,
        "embedding_method": method,
        "embedded_path": embedded_path,
        "warnings": warnings,
    })
    sidecar_path = sidecar_result.sidecar_path
    manifest_str = json.dumps(manifest, indent=2)

    if on_progress:
        on_progress(85, "C2PA manifest embedded and verified")

    if on_progress:
        on_progress(100, "C2PA manifest embedded")

    return {
        "output_path": out,
        "sidecar_path": sidecar_path,
        "manifest_size": len(manifest_str),
        "source_hash": source_hash,
        "instance_id": manifest.get("instance_id", ""),
        "operations_count": len(manifest.get("operations", [])),
        "embedded": True,
        "c2pa_spec_version": C2PA_VERSION,
        "credential": credential,
    }


# ---------------------------------------------------------------------------
# Read C2PA manifest from video
# ---------------------------------------------------------------------------

def read_c2pa(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Read a C2PA manifest from a video file's metadata.

    Tries a real embedded credential through c2patool, then the legacy FFmpeg
    comment format, then a JSON sidecar.

    Args:
        video_path: Path to video file.
        on_progress: Optional callback(pct, msg).

    Returns:
        C2PA manifest dict, or empty dict if none found.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if on_progress:
        on_progress(10, "Reading C2PA manifest")

    try:
        import subprocess

        from opencut.core.c2pa_sidecar import _find_c2patool

        c2patool = _find_c2patool()
        if c2patool:
            result = subprocess.run(
                [c2patool, video_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                report = json.loads(result.stdout)
                active_label = str(report.get("active_manifest") or "")
                active = (report.get("manifests") or {}).get(active_label)
                if isinstance(active, dict):
                    manifest = deepcopy(active)
                    manifest["embedded"] = True
                    validation_state = report.get("validation_state")
                    manifest["validation_state"] = validation_state
                    manifest["validation_status"] = report.get("validation_status", [])
                    # c2patool exits 0 even when the credential fails
                    # validation (e.g. "Invalid"). ``embedded`` means "a
                    # credential was found"; ``valid`` means it actually
                    # passed C2PA validation.
                    manifest["valid"] = validation_state == "Valid"
                    if on_progress:
                        on_progress(
                            100,
                            "C2PA credential read and validated"
                            if manifest["valid"]
                            else "C2PA credential read (validation failed)",
                        )
                    return manifest
    except Exception as exc:
        logger.debug("Could not read embedded C2PA credential: %s", exc)

    # Legacy compatibility: releases before 1.33.1 used an FFmpeg comment.
    try:
        from opencut.helpers import get_ffprobe_path
        ffprobe = get_ffprobe_path()
        cmd = [
            ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            comment = probe_data.get("format", {}).get("tags", {}).get("comment", "")
            if comment.startswith("C2PA:"):
                manifest_json = comment[5:]
                manifest = json.loads(manifest_json)
                if on_progress:
                    on_progress(100, "C2PA manifest read from metadata")
                return manifest
    except Exception as e:
        logger.debug("Could not read C2PA from metadata: %s", e)

    # Try sidecar file
    sidecar_path = video_path + ".c2pa.json"
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if on_progress:
                on_progress(100, "C2PA manifest read from sidecar")
            return manifest
        except Exception as e:
            logger.debug("Could not read C2PA sidecar: %s", e)

    if on_progress:
        on_progress(100, "No C2PA manifest found")

    return {}
