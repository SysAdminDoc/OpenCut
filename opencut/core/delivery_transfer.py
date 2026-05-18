"""Delivery transfer bundle planner for croc and rclone (F234)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from opencut.security import validate_output_path, validate_path

TRANSFER_METHODS: Dict[str, Dict[str, str]] = {
    "croc": {
        "label": "croc one-shot P2P",
        "description": "Send a prepared delivery bundle directly to one recipient with a short code.",
        "binary": "croc",
    },
    "rclone": {
        "label": "rclone cloud bucket",
        "description": "Copy a prepared delivery bundle to a configured rclone remote.",
        "binary": "rclone",
    },
}


@dataclass
class TransferCommand:
    method: str
    label: str
    available: bool
    argv: List[str] = field(default_factory=list)
    shell_command: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransferBundleResult:
    bundle_path: str
    manifest_path: str
    source_paths: List[str]
    source_count: int
    total_source_bytes: int
    bundle_bytes: int
    methods: List[str]
    commands: List[TransferCommand]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["commands"] = [command.to_dict() for command in self.commands]
        return data


def list_transfer_methods() -> List[Dict[str, Any]]:
    """Return delivery-transfer menu options and local tool availability."""
    methods = []
    for method, spec in TRANSFER_METHODS.items():
        binary = spec["binary"]
        executable = shutil.which(binary)
        methods.append(
            {
                "method": method,
                "label": spec["label"],
                "description": spec["description"],
                "binary": binary,
                "available": bool(executable),
                "executable": executable or "",
            }
        )
    return methods


def _normalise_methods(method: str | Iterable[str] | None) -> List[str]:
    if method is None:
        return ["croc"]
    if isinstance(method, str):
        raw = [part.strip().lower() for part in method.split(",") if part.strip()]
    else:
        raw = [str(part).strip().lower() for part in method if str(part).strip()]
    if not raw:
        raw = ["croc"]
    if "both" in raw:
        raw = ["croc", "rclone"]
    unknown = [item for item in raw if item not in TRANSFER_METHODS]
    if unknown:
        raise ValueError(f"Unsupported transfer method(s): {', '.join(unknown)}")
    return list(dict.fromkeys(raw))


def _resolve_sources(paths: Iterable[str]) -> List[str]:
    resolved = []
    for raw in paths:
        path = validate_path(str(raw or ""))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Transfer source not found: {raw}")
        resolved.append(path)
    if not resolved:
        raise ValueError("At least one transfer source path is required")
    return resolved


def _iter_files(path: str) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            yield os.path.join(root, name)


def _archive_name(source_root: str, file_path: str, index: int) -> str:
    if os.path.isfile(source_root):
        name = os.path.basename(source_root)
    else:
        name = os.path.join(os.path.basename(source_root), os.path.relpath(file_path, source_root))
    if index:
        name = os.path.join(f"source_{index + 1}", name)
    return name.replace("\\", "/")


def _default_bundle_path(sources: List[str], output_dir: str = "", bundle_name: str = "") -> str:
    base_dir = validate_path(output_dir) if output_dir else os.path.dirname(sources[0])
    name = bundle_name.strip() if bundle_name else f"{os.path.splitext(os.path.basename(sources[0]))[0]}_delivery_transfer.zip"
    if not name.lower().endswith(".zip"):
        name += ".zip"
    return validate_output_path(os.path.join(base_dir, name))


def _command(method: str, bundle_path: str, *, croc_code: str = "", croc_relay: str = "", rclone_remote: str = "", rclone_path: str = "") -> TransferCommand:
    spec = TRANSFER_METHODS[method]
    executable = shutil.which(spec["binary"])
    available = bool(executable)
    argv: List[str] = []
    note = ""

    if method == "croc":
        argv = [spec["binary"], "send"]
        if croc_code:
            argv.extend(["--code", croc_code])
        if croc_relay:
            argv.extend(["--relay", croc_relay])
        argv.append(bundle_path)
        note = "Run this command on the sender; give the recipient the displayed croc code."
    elif method == "rclone":
        remote = str(rclone_remote or "").strip()
        if not remote:
            raise ValueError("rclone_remote is required when method includes rclone")
        target_dir = remote.rstrip("/")
        if rclone_path:
            cleaned_path = str(rclone_path).strip("/\\")
            target_dir = f"{target_dir}/{cleaned_path}"
        argv = [spec["binary"], "copy", bundle_path, target_dir]
        note = "Requires an existing rclone remote configured outside OpenCut."

    return TransferCommand(
        method=method,
        label=spec["label"],
        available=available,
        argv=argv,
        shell_command=subprocess.list2cmdline(argv),
        note=note if available else f"{spec['binary']} is not installed or not on PATH.",
    )


def prepare_transfer_bundle(
    *,
    paths: Iterable[str],
    output_path: str = "",
    output_dir: str = "",
    bundle_name: str = "",
    method: str | Iterable[str] | None = "croc",
    croc_code: str = "",
    croc_relay: str = "",
    rclone_remote: str = "",
    rclone_path: str = "",
) -> TransferBundleResult:
    """Create a zip bundle and return croc/rclone command plans."""
    sources = _resolve_sources(paths)
    methods = _normalise_methods(method)
    bundle_path = validate_output_path(output_path) if output_path else _default_bundle_path(sources, output_dir, bundle_name)
    manifest_path = os.path.splitext(bundle_path)[0] + ".transfer.json"
    manifest_path = validate_output_path(manifest_path)

    files = []
    total_source_bytes = 0
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, source in enumerate(sources):
            for file_path in _iter_files(source):
                arcname = _archive_name(source, file_path, index if len(sources) > 1 else 0)
                size = os.path.getsize(file_path)
                total_source_bytes += size
                files.append({"source": file_path, "archive_name": arcname, "bytes": size})
                zf.write(file_path, arcname)
        manifest = {
            "schema": "opencut.delivery-transfer.v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bundle_name": os.path.basename(bundle_path),
            "source_count": len(sources),
            "files": files,
            "methods": methods,
        }
        zf.writestr("delivery_transfer_manifest.json", json.dumps(manifest, indent=2))

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    commands = [
        _command(
            item,
            bundle_path,
            croc_code=croc_code,
            croc_relay=croc_relay,
            rclone_remote=rclone_remote,
            rclone_path=rclone_path,
        )
        for item in methods
    ]
    warnings = [command.note for command in commands if not command.available]
    return TransferBundleResult(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        source_paths=sources,
        source_count=len(sources),
        total_source_bytes=total_source_bytes,
        bundle_bytes=os.path.getsize(bundle_path),
        methods=methods,
        commands=commands,
        warnings=warnings,
    )
