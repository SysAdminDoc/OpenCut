"""Authenticated plugin installation and rollback regression tests."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from opencut.core.plugin_installation import (
    PluginApprovalRequired,
    PluginInstallError,
    activate_staged_plugin,
    artifact_signature_message,
    directory_signature_message,
    install_local_plugin,
    publisher_fingerprint,
    stage_local_directory,
    validate_staged_plugin,
)
from opencut.core.plugin_manifest import compute_plugin_lock, write_plugin_lock


def _public_key_b64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _signature_b64(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


def _lock_digest(plugin_dir: Path) -> str:
    payload = json.dumps(
        compute_plugin_lock(plugin_dir),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_plugin(
    plugin_dir: Path,
    private_key: Ed25519PrivateKey,
    *,
    name: str = "demo-plugin",
    version: str = "1.0.0",
    capabilities: list[str] | None = None,
    include_directory_signature: bool = True,
) -> dict:
    plugin_dir.mkdir(parents=True)
    manifest = {
        "name": name,
        "version": version,
        "description": "Signed test plugin",
        "api_version": 1,
        "capabilities": capabilities or [],
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (plugin_dir / "payload.txt").write_text(f"payload-{version}", encoding="utf-8")
    write_plugin_lock(plugin_dir)
    if include_directory_signature:
        public_key = _public_key_b64(private_key)
        signature = {
            "version": 1,
            "algorithm": "ed25519",
            "publisher_id": "publisher.example",
            "public_key": public_key,
            "signature": _signature_b64(
                private_key,
                directory_signature_message(name, version, _lock_digest(plugin_dir)),
            ),
        }
        (plugin_dir / "plugin.signature.json").write_text(
            json.dumps(signature),
            encoding="utf-8",
        )
    return manifest


@pytest.fixture
def isolated_install_paths(tmp_path, monkeypatch):
    from opencut.core import plugin_installation

    trust_store = tmp_path / "trusted-plugin-publishers.json"
    staging = tmp_path / "plugin-staging"
    plugins = tmp_path / "plugins"
    monkeypatch.setattr(plugin_installation, "TRUST_STORE_PATH", str(trust_store))
    monkeypatch.setattr(plugin_installation, "STAGING_ROOT", str(staging))
    return plugins, staging, trust_store


def test_direct_install_requires_exact_capabilities_and_publisher_approval(
    tmp_path,
    isolated_install_paths,
):
    plugins, staging, trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "source"
    _write_plugin(source, key, capabilities=["http.routes", "host.network"])
    fingerprint = publisher_fingerprint(_public_key_b64(key))

    with pytest.raises(PluginApprovalRequired) as exc_info:
        install_local_plugin(source, plugins)

    assert exc_info.value.preview["capabilities"] == ["host.network", "http.routes"]
    assert exc_info.value.preview["publisher"]["fingerprint"] == fingerprint
    assert not (plugins / "demo-plugin").exists()
    assert not trust_store.exists()

    installed = install_local_plugin(
        source,
        plugins,
        approved_capabilities=["host.network", "http.routes"],
        approve_publisher_fingerprint=fingerprint,
    )

    assert installed.publisher.publisher_id == "publisher.example"
    assert (plugins / "demo-plugin" / "payload.txt").read_text() == "payload-1.0.0"
    trusted = json.loads(trust_store.read_text(encoding="utf-8"))
    assert trusted["publishers"]["publisher.example"]["fingerprint"] == fingerprint
    assert not any(staging.iterdir())


def test_direct_install_rejects_unsigned_and_tampered_plugins(
    tmp_path,
    isolated_install_paths,
):
    plugins, _staging, _trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    unsigned = tmp_path / "unsigned"
    _write_plugin(unsigned, key, include_directory_signature=False)

    with pytest.raises(PluginInstallError, match="plugin.signature.json is required"):
        install_local_plugin(unsigned, plugins, approved_capabilities=[])

    tampered = tmp_path / "tampered"
    _write_plugin(tampered, key)
    (tampered / "payload.txt").write_text("changed after signing", encoding="utf-8")
    with pytest.raises(PluginInstallError, match="sha-256 mismatch"):
        install_local_plugin(tampered, plugins, approved_capabilities=[])

    assert not plugins.exists() or not any(plugins.iterdir())


def test_direct_install_rejects_linked_content(tmp_path, isolated_install_paths):
    plugins, _staging, _trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "linked"
    _write_plugin(source, key)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = source / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable in this Windows environment")

    with pytest.raises(PluginInstallError, match="link or reparse point"):
        install_local_plugin(source, plugins, approved_capabilities=[])

    assert not (plugins / "demo-plugin").exists()


def test_direct_install_route_returns_structured_approval_then_activates(
    client,
    csrf_token,
    monkeypatch,
    tmp_path,
    isolated_install_paths,
):
    import opencut.routes.plugins as plugin_routes

    plugins, _staging, _trust_store = isolated_install_paths
    monkeypatch.setattr(plugin_routes, "PLUGINS_DIR", str(plugins))
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "route-source"
    _write_plugin(source, key, capabilities=["ui.panel"])
    headers = {"X-OpenCut-Token": csrf_token}

    preview_response = client.post(
        "/plugins/install",
        json={"source": str(source)},
        headers=headers,
    )

    assert preview_response.status_code == 409
    preview = preview_response.get_json()
    assert preview["code"] == "PLUGIN_APPROVAL_REQUIRED"
    assert preview["approval"]["capabilities"] == ["ui.panel"]

    approved_response = client.post(
        "/plugins/install",
        json={
            "source": str(source),
            "approved_capabilities": preview["approval"]["capabilities"],
            "approve_publisher_fingerprint": preview["approval"]["publisher"]["fingerprint"],
        },
        headers=headers,
    )

    assert approved_response.status_code == 200
    body = approved_response.get_json()
    assert body["success"] is True
    assert body["publisher"]["publisher_id"] == "publisher.example"
    assert (plugins / "demo-plugin" / "plugin.json").is_file()


def test_activation_restores_prior_version_when_metadata_commit_fails(
    tmp_path,
    isolated_install_paths,
):
    plugins, _staging, _trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    old_plugin = plugins / "demo-plugin"
    _write_plugin(old_plugin, key, version="1.0.0")
    source = tmp_path / "new-source"
    _write_plugin(source, key, version="2.0.0")
    fingerprint = publisher_fingerprint(_public_key_b64(key))

    stage = stage_local_directory(source, "demo-plugin")
    staged = validate_staged_plugin(
        stage,
        approved_capabilities=[],
        approve_publisher_fingerprint=fingerprint,
    )

    with pytest.raises(OSError, match="metadata write interrupted"):
        activate_staged_plugin(
            staged,
            plugins,
            replace_existing=True,
            commit_metadata=lambda _path: (_ for _ in ()).throw(
                OSError("metadata write interrupted")
            ),
        )

    assert (old_plugin / "payload.txt").read_text() == "payload-1.0.0"
    assert not Path(stage).exists()


def _zip_plugin(plugin_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(plugin_dir.rglob("*")):
            if path.is_file():
                archive.write(path, f"plugin-main/{path.relative_to(plugin_dir).as_posix()}")


def test_marketplace_install_verifies_digest_signature_and_manifest(
    tmp_path,
    isolated_install_paths,
    monkeypatch,
):
    from opencut.core import plugin_marketplace

    plugins, _staging, _trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "market-source"
    _write_plugin(
        source,
        key,
        name="market-plugin",
        capabilities=["ui.panel"],
        include_directory_signature=False,
    )
    archive = tmp_path / "market-plugin.zip"
    _zip_plugin(source, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    public_key = _public_key_b64(key)
    fingerprint = publisher_fingerprint(public_key)
    target = plugin_marketplace.PluginInfo(
        plugin_id="market-plugin",
        name="Market Plugin",
        version="1.0.0",
        author="Publisher",
        description="Authenticated marketplace plugin",
        repo_url="https://example.com/market-plugin",
        download_url="https://example.com/market-plugin.zip",
        artifact_sha256=digest,
        publisher_id="publisher.example",
        publisher_public_key=public_key,
        publisher_signature=_signature_b64(
            key,
            artifact_signature_message("market-plugin", "1.0.0", digest),
        ),
        capabilities=["ui.panel"],
    )
    monkeypatch.setattr(plugin_marketplace, "PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(plugin_marketplace, "fetch_plugin_registry", lambda: [target])
    monkeypatch.setattr(
        plugin_marketplace,
        "transactional_download",
        lambda _url, destination, **_kwargs: shutil.copyfile(archive, destination),
    )

    with pytest.raises(PluginApprovalRequired):
        plugin_marketplace.install_plugin("market-plugin")
    assert not (plugins / "market-plugin").exists()

    result = plugin_marketplace.install_plugin(
        "market-plugin",
        approved_capabilities=["ui.panel"],
        approve_publisher_fingerprint=fingerprint,
    )

    assert result.installed is True
    assert (plugins / "market-plugin" / "plugin.json").is_file()
    installed = json.loads((plugins / "manifest.json").read_text(encoding="utf-8"))
    assert installed["market-plugin"]["artifact_sha256"] == digest
    assert installed["market-plugin"]["publisher_fingerprint"] == fingerprint


def test_marketplace_tamper_and_interruption_never_activate(
    tmp_path,
    isolated_install_paths,
    monkeypatch,
):
    from opencut.core import plugin_marketplace

    plugins, _staging, _trust_store = isolated_install_paths
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "market-source"
    _write_plugin(
        source,
        key,
        name="market-plugin",
        include_directory_signature=False,
    )
    archive = tmp_path / "market-plugin.zip"
    _zip_plugin(source, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    public_key = _public_key_b64(key)
    target = plugin_marketplace.PluginInfo(
        plugin_id="market-plugin",
        name="Market Plugin",
        version="1.0.0",
        author="Publisher",
        description="Authenticated marketplace plugin",
        repo_url="https://example.com/market-plugin",
        download_url="https://example.com/market-plugin.zip",
        artifact_sha256="0" * 64,
        publisher_id="publisher.example",
        publisher_public_key=public_key,
        publisher_signature=_signature_b64(
            key,
            artifact_signature_message("market-plugin", "1.0.0", digest),
        ),
    )
    monkeypatch.setattr(plugin_marketplace, "PLUGINS_DIR", str(plugins))
    monkeypatch.setattr(plugin_marketplace, "fetch_plugin_registry", lambda: [target])
    monkeypatch.setattr(
        plugin_marketplace,
        "transactional_download",
        lambda _url, destination, **_kwargs: shutil.copyfile(archive, destination),
    )

    with pytest.raises(PluginInstallError, match="SHA-256 mismatch"):
        plugin_marketplace.install_plugin("market-plugin")
    assert not (plugins / "market-plugin").exists()
    assert not (plugins / "manifest.json").exists()

    def _interrupted(_url, destination, **_kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError("download interrupted")

    monkeypatch.setattr(plugin_marketplace, "transactional_download", _interrupted)
    with pytest.raises(OSError, match="download interrupted"):
        plugin_marketplace.install_plugin("market-plugin")
    assert not (plugins / "market-plugin").exists()
