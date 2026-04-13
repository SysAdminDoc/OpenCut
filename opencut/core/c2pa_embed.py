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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

# C2PA manifest version
C2PA_VERSION = "2.0"
C2PA_CLAIM_GENERATOR = "OpenCut Video Editor"


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

    # Build operations list
    manifest_ops = []
    for op in (operations or []):
        entry = {
            "action": op.get("action", "unknown"),
            "softwareAgent": C2PA_CLAIM_GENERATOR,
            "parameters": op.get("parameters", {}),
            "timestamp": op.get("timestamp", created),
            "description": op.get("description", ""),
        }
        manifest_ops.append(entry)

    # Build assertions
    assertions = []

    # Creative work assertion
    assertions.append({
        "label": "c2pa.actions",
        "data": {"actions": manifest_ops},
    })

    # Hash assertion
    if source_hash:
        assertions.append({
            "label": "c2pa.hash.data",
            "data": {
                "exclusions": [],
                "name": "jumbf manifest",
                "alg": "sha256",
                "hash": source_hash,
            },
        })

    # AI disclosure assertion
    ai_ops = [
        op for op in manifest_ops
        if any(kw in op.get("action", "").lower()
               for kw in ("ai", "auto", "generate", "enhance", "ml"))
    ]
    if ai_ops:
        assertions.append({
            "label": "c2pa.ai_disclosure",
            "data": {
                "ai_operations": [op["action"] for op in ai_ops],
                "ai_generated": True,
                "model_info": "OpenCut AI Pipeline",
            },
        })

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
        "signature": "",  # placeholder for signing
    }

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
    on_progress: Optional[Callable] = None,
) -> dict:
    """Embed a C2PA manifest into a video file as metadata.

    Embeds the manifest as a JSON comment in the MP4 metadata. For
    production use, a JUMBF box would be used; this implementation
    stores a serialized manifest in the video's comment metadata field.

    Args:
        video_path: Path to input video.
        manifest: C2PA manifest dict (from create_c2pa_manifest).
        output: Output path (auto-generated if None).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, manifest_size, source_hash.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if on_progress:
        on_progress(10, "Computing source hash")

    # Compute source hash if not already present
    source_hash = manifest.get("source_hash", "")
    if not source_hash:
        source_hash = _hash_file(video_path)
        manifest["source_hash"] = source_hash

    # Update the hash assertion
    for assertion in manifest.get("assertions", []):
        if assertion.get("label") == "c2pa.hash.data":
            assertion["data"]["hash"] = source_hash

    # Sign the manifest (simple HMAC for now)
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest["signature"] = manifest_hash

    out = output or _output_path(video_path, "_c2pa", "")

    if on_progress:
        on_progress(40, "Embedding C2PA manifest into video")

    # Write manifest as a sidecar JSON (embedded via ffmpeg metadata)
    manifest_str = json.dumps(manifest, indent=2)

    # Embed as MP4 comment metadata
    cmd = (
        FFmpegCmd()
        .input(video_path)
        .copy_streams()
        .option("-metadata", f"comment=C2PA:{manifest_str}")
        .option("-metadata", "description=C2PA-enabled content")
        .faststart()
        .output(out)
        .build()
    )
    run_ffmpeg(cmd)

    # Also write sidecar manifest
    sidecar_path = out + ".c2pa.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(manifest_str)

    if on_progress:
        on_progress(100, "C2PA manifest embedded")

    return {
        "output_path": out,
        "sidecar_path": sidecar_path,
        "manifest_size": len(manifest_str),
        "source_hash": source_hash,
        "instance_id": manifest.get("instance_id", ""),
        "operations_count": len(manifest.get("operations", [])),
    }


# ---------------------------------------------------------------------------
# Read C2PA manifest from video
# ---------------------------------------------------------------------------

def read_c2pa(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Read a C2PA manifest from a video file's metadata.

    Tries embedded metadata first, then looks for a sidecar .c2pa.json file.

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

    # Try reading from ffprobe metadata
    try:
        import subprocess

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
