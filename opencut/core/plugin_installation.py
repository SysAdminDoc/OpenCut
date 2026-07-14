"""Authenticated, staged installation primitives for third-party plugins."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from opencut.core.plugin_manifest import (
    SIGNATURE_FILENAME,
    compute_plugin_lock,
    validate_plugin_manifest,
)
from opencut.helpers import OPENCUT_DIR
from opencut.security import is_path_within

TRUST_STORE_PATH = os.path.join(OPENCUT_DIR, "trusted-plugin-publishers.json")
STAGING_ROOT = os.path.join(OPENCUT_DIR, "plugin-staging")
TRUST_STORE_VERSION = 1
SIGNATURE_VERSION = 1
MAX_PLUGIN_FILES = 5000
MAX_PLUGIN_BYTES = 512 * 1024 * 1024


class PluginInstallError(ValueError):
    """An install failed before activation."""


class PluginApprovalRequired(PluginInstallError):
    """Publisher or capability approval is missing or stale."""

    def __init__(self, preview: dict):
        self.preview = preview
        super().__init__(
            "Plugin install requires approval of the publisher fingerprint "
            "and exact declared capabilities"
        )


@dataclass(frozen=True)
class PublisherIdentity:
    publisher_id: str
    public_key: str
    fingerprint: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StagedPlugin:
    name: str
    version: str
    capabilities: tuple[str, ...]
    publisher: PublisherIdentity
    path: str


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | os.PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _decode_public_key(public_key: str) -> bytes:
    if not isinstance(public_key, str) or not public_key.strip():
        raise PluginInstallError("publisher public_key is required")
    try:
        raw = base64.b64decode(public_key, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PluginInstallError("publisher public_key must be valid base64") from exc
    if len(raw) != 32:
        raise PluginInstallError("publisher public_key must contain a 32-byte Ed25519 key")
    return raw


def publisher_fingerprint(public_key: str) -> str:
    """Return the stable SHA-256 fingerprint for a base64 Ed25519 key."""
    return hashlib.sha256(_decode_public_key(public_key)).hexdigest()


def _publisher_identity(publisher_id: str, public_key: str) -> PublisherIdentity:
    if not isinstance(publisher_id, str) or not publisher_id.strip():
        raise PluginInstallError("publisher id is required")
    cleaned_id = publisher_id.strip()
    if len(cleaned_id) > 100 or not all(c.isalnum() or c in "-_." for c in cleaned_id):
        raise PluginInstallError("publisher id contains invalid characters")
    cleaned_key = public_key.strip()
    return PublisherIdentity(cleaned_id, cleaned_key, publisher_fingerprint(cleaned_key))


def _decode_signature(signature: str) -> bytes:
    if not isinstance(signature, str) or not signature.strip():
        raise PluginInstallError("publisher signature is required")
    try:
        raw = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PluginInstallError("publisher signature must be valid base64") from exc
    if len(raw) != 64:
        raise PluginInstallError("publisher signature must contain a 64-byte Ed25519 signature")
    return raw


def _verify_signature(identity: PublisherIdentity, signature: str, message: bytes) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(_decode_public_key(identity.public_key)).verify(
            _decode_signature(signature),
            message,
        )
    except InvalidSignature as exc:
        raise PluginInstallError("publisher signature verification failed") from exc


def artifact_signature_message(
    plugin_id: str,
    version: str,
    artifact_sha256: str,
) -> bytes:
    return (
        "opencut-plugin-artifact-v1\n"
        f"{plugin_id}\n{version}\n{artifact_sha256.lower()}\n"
    ).encode("utf-8")


def directory_signature_message(name: str, version: str, lock_sha256: str) -> bytes:
    return (
        "opencut-plugin-directory-v1\n"
        f"{name}\n{version}\n{lock_sha256.lower()}\n"
    ).encode("utf-8")


def verify_artifact_publisher(
    *,
    plugin_id: str,
    version: str,
    archive_path: str | os.PathLike,
    artifact_sha256: str,
    publisher_id: str,
    public_key: str,
    signature: str,
) -> PublisherIdentity:
    """Verify a registry-pinned artifact digest and Ed25519 publisher signature."""
    expected = str(artifact_sha256 or "").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise PluginInstallError("registry artifact_sha256 must be a 64-character hex digest")
    actual = sha256_file(archive_path)
    if actual != expected:
        raise PluginInstallError(
            f"plugin artifact SHA-256 mismatch: expected {expected}, received {actual}"
        )
    identity = _publisher_identity(publisher_id, public_key)
    _verify_signature(
        identity,
        signature,
        artifact_signature_message(plugin_id, version, expected),
    )
    return identity


def _read_json_object(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginInstallError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise PluginInstallError(f"{label} must be a JSON object")
    return payload


def verify_directory_publisher(
    plugin_dir: str | os.PathLike,
    manifest: dict,
) -> PublisherIdentity:
    """Verify ``plugin.signature.json`` against the locked directory payload."""
    base = Path(plugin_dir)
    signature_path = base / SIGNATURE_FILENAME
    if not signature_path.is_file():
        raise PluginInstallError(
            f"{SIGNATURE_FILENAME} is required for direct plugin installation"
        )
    payload = _read_json_object(signature_path, label=SIGNATURE_FILENAME)
    if payload.get("version") != SIGNATURE_VERSION:
        raise PluginInstallError(
            f"{SIGNATURE_FILENAME} version must be {SIGNATURE_VERSION}"
        )
    if payload.get("algorithm") != "ed25519":
        raise PluginInstallError(f"{SIGNATURE_FILENAME} algorithm must be ed25519")
    identity = _publisher_identity(payload.get("publisher_id", ""), payload.get("public_key", ""))
    lock_sha256 = hashlib.sha256(_canonical_json(compute_plugin_lock(base))).hexdigest()
    _verify_signature(
        identity,
        payload.get("signature", ""),
        directory_signature_message(
            str(manifest.get("name") or ""),
            str(manifest.get("version") or ""),
            lock_sha256,
        ),
    )
    return identity


def _load_trust_store() -> dict:
    path = Path(TRUST_STORE_PATH)
    if not path.exists():
        return {"version": TRUST_STORE_VERSION, "publishers": {}}
    if path.is_symlink() or not path.is_file():
        raise PluginInstallError("publisher trust store must be a regular non-symlink file")
    payload = _read_json_object(path, label="publisher trust store")
    publishers = payload.get("publishers")
    if payload.get("version") != TRUST_STORE_VERSION or not isinstance(publishers, dict):
        raise PluginInstallError("publisher trust store has an unsupported schema")
    return payload


def _save_trust_store(payload: dict) -> None:
    path = Path(TRUST_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _trusted_publisher(identity: PublisherIdentity) -> bool:
    entry = _load_trust_store()["publishers"].get(identity.publisher_id)
    if not isinstance(entry, dict):
        return False
    stored_key = entry.get("public_key")
    if stored_key != identity.public_key:
        raise PluginInstallError(
            f"trusted publisher key changed for {identity.publisher_id}; "
            "remove the old trust entry before approving a replacement"
        )
    return entry.get("fingerprint") == identity.fingerprint


def _trust_publisher(identity: PublisherIdentity) -> None:
    payload = _load_trust_store()
    payload["publishers"][identity.publisher_id] = {
        "public_key": identity.public_key,
        "fingerprint": identity.fingerprint,
        "trusted_at": int(time.time()),
    }
    _save_trust_store(payload)


def _approval_preview(
    manifest: dict,
    identity: PublisherIdentity,
    *,
    publisher_trusted: bool,
) -> dict:
    return {
        "name": manifest.get("name", ""),
        "version": manifest.get("version", ""),
        "description": manifest.get("description", ""),
        "capabilities": sorted(manifest.get("capabilities") or []),
        "publisher": {
            **identity.as_dict(),
            "trusted": publisher_trusted,
        },
    }


def require_install_approval(
    manifest: dict,
    identity: PublisherIdentity,
    *,
    approved_capabilities: Iterable[str] | None,
    approve_publisher_fingerprint: str | None,
) -> dict:
    """Require exact capability consent and a trusted publisher identity."""
    declared = sorted(manifest.get("capabilities") or [])
    approved = approved_capabilities
    if approved is None:
        approved_list: list[str] = []
    elif isinstance(approved, (str, bytes)):
        raise PluginInstallError("approved_capabilities must be a list of strings")
    else:
        approved_list = list(approved)
    if any(not isinstance(value, str) for value in approved_list):
        raise PluginInstallError("approved_capabilities must be a list of strings")
    normalized_approved = sorted(set(approved_list))
    publisher_trusted = _trusted_publisher(identity)
    publisher_approved = str(approve_publisher_fingerprint or "").strip().lower()
    preview = _approval_preview(manifest, identity, publisher_trusted=publisher_trusted)
    if normalized_approved != declared or (
        not publisher_trusted and publisher_approved != identity.fingerprint
    ):
        raise PluginApprovalRequired(preview)
    if not publisher_trusted:
        _trust_publisher(identity)
        preview["publisher"]["trusted"] = True
    return preview


def _iter_tree_entries(root: Path):
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            yield current_path / name
        for name in files:
            yield current_path / name


def _is_reparse_point(path_stat: os.stat_result) -> bool:
    attributes = int(getattr(path_stat, "st_file_attributes", 0) or 0)
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & flag)


def validate_directory_tree(root: str | os.PathLike) -> None:
    """Reject links, special files, and unbounded local plugin trees."""
    base = Path(root)
    if base.is_symlink() or not base.is_dir() or _is_reparse_point(base.lstat()):
        raise PluginInstallError("plugin source must be a regular non-symlink directory")
    count = 0
    total = 0
    for path in _iter_tree_entries(base):
        count += 1
        if count > MAX_PLUGIN_FILES:
            raise PluginInstallError(f"plugin contains more than {MAX_PLUGIN_FILES} entries")
        path_stat = path.lstat()
        mode = path_stat.st_mode
        if stat.S_ISLNK(mode) or _is_reparse_point(path_stat):
            raise PluginInstallError(f"plugin contains a link or reparse point: {path.name}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PluginInstallError(f"plugin contains a non-regular file: {path.name}")
        total += path.stat().st_size
        if total > MAX_PLUGIN_BYTES:
            raise PluginInstallError(
                f"plugin expands beyond the {MAX_PLUGIN_BYTES}-byte safety limit"
            )


def create_staging_dir(plugin_name: str) -> str:
    root = Path(STAGING_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{plugin_name}-", dir=root)


def stage_local_directory(source: str | os.PathLike, plugin_name: str = "local") -> str:
    validate_directory_tree(source)
    stage = create_staging_dir(plugin_name)
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=True)
        validate_directory_tree(stage)
        return stage
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_staged_plugin(
    stage: str | os.PathLike,
    *,
    publisher: PublisherIdentity | None = None,
    approved_capabilities: Iterable[str] | None = None,
    approve_publisher_fingerprint: str | None = None,
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> StagedPlugin:
    """Run schema, lock, signature, identity, and capability checks on a stage."""
    validate_directory_tree(stage)
    validation = validate_plugin_manifest(stage)
    if not validation.valid:
        raise PluginInstallError("plugin validation failed: " + "; ".join(validation.errors))
    manifest = _read_json_object(Path(stage) / "plugin.json", label="plugin.json")
    name = str(manifest.get("name") or "")
    version = str(manifest.get("version") or "")
    if expected_name is not None and name != expected_name:
        raise PluginInstallError(
            f"plugin manifest name {name!r} does not match registry id {expected_name!r}"
        )
    if expected_version is not None and version != expected_version:
        raise PluginInstallError(
            f"plugin manifest version {version!r} does not match registry version {expected_version!r}"
        )
    identity = publisher or verify_directory_publisher(stage, manifest)
    require_install_approval(
        manifest,
        identity,
        approved_capabilities=approved_capabilities,
        approve_publisher_fingerprint=approve_publisher_fingerprint,
    )
    return StagedPlugin(
        name=name,
        version=version,
        capabilities=tuple(sorted(manifest.get("capabilities") or [])),
        publisher=identity,
        path=str(stage),
    )


def activate_staged_plugin(
    staged: StagedPlugin,
    plugins_dir: str | os.PathLike,
    *,
    replace_existing: bool = False,
    commit_metadata: Callable[[str], None] | None = None,
) -> str:
    """Atomically activate a verified stage, restoring the prior version on error."""
    root = Path(plugins_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / staged.name
    if not is_path_within(destination, root):
        raise PluginInstallError("plugin destination escapes the managed plugin directory")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Plugin already exists: {staged.name}")

    stage = Path(staged.path)
    if not stage.is_dir() or not is_path_within(stage, STAGING_ROOT):
        raise PluginInstallError("plugin stage is outside the managed staging directory")

    backup: Path | None = None
    if destination.exists():
        backup = Path(STAGING_ROOT) / f"backup-{staged.name}-{uuid.uuid4().hex}"
        os.replace(destination, backup)

    try:
        os.replace(stage, destination)
        if commit_metadata is not None:
            commit_metadata(str(destination))
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        if backup is not None and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    return str(destination)


def install_local_plugin(
    source: str | os.PathLike,
    plugins_dir: str | os.PathLike,
    *,
    approved_capabilities: Iterable[str] | None = None,
    approve_publisher_fingerprint: str | None = None,
) -> StagedPlugin:
    """Stage, authenticate, approve, and atomically activate a local directory."""
    stage = stage_local_directory(source)
    try:
        staged = validate_staged_plugin(
            stage,
            approved_capabilities=approved_capabilities,
            approve_publisher_fingerprint=approve_publisher_fingerprint,
        )
        activate_staged_plugin(staged, plugins_dir)
        return staged
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)
