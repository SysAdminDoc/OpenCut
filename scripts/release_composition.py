#!/usr/bin/env python3
"""Generate and validate OpenCut's resolved release composition.

The release lock is the authorization boundary: every resolved Python
distribution must be exactly pinned and backed by one or more SHA-256 download
hashes.  The generated composition also binds the packaged application tree
and bundled media tools to deterministic digests, license evidence, and source
provenance.  Missing evidence is a build failure, not a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "requirements-release-lock.txt"
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements.txt"
DEFAULT_BUILD_LOCK = REPO_ROOT / "requirements-build-lock.txt"
DEFAULT_LICENSE_EVIDENCE = REPO_ROOT / "release-license-evidence.json"
SCHEMA = "https://opencut.dev/schemas/release-composition/v1"
_REQ_RE = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s*;\s*(.*?))?\s+--hash=sha256:"
)
_NAME_RE = re.compile(r"^([A-Za-z0-9_.-]+)")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
_LICENSE_BASENAMES = ("license", "licence", "copying", "notice")


class CompositionError(RuntimeError):
    """Raised when release evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class LockEntry:
    name: str
    version: str
    hashes: tuple[str, ...]
    marker: str = ""


def normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower()).strip("-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_record(path: Path) -> dict:
    """Return a reproducible Merkle-style digest for a file or directory."""
    resolved = path.resolve()
    if not resolved.exists():
        raise CompositionError(f"release artifact does not exist: {resolved}")
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "file_count": 1,
        }

    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise CompositionError(f"release artifact directory is empty: {resolved}")
    digest = hashlib.sha256()
    total_size = 0
    for item in files:
        rel = item.relative_to(resolved).as_posix()
        size = item.stat().st_size
        item_hash = sha256_file(item)
        digest.update(f"{rel}\0{size}\0{item_hash}\n".encode("utf-8"))
        total_size += size
    return {
        "path": str(resolved),
        "kind": "directory",
        "size": total_size,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _logical_requirement_lines(text: str) -> list[str]:
    logical: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--") and not current:
            continue
        continued = stripped.endswith("\\")
        part = stripped[:-1].strip() if continued else stripped
        current = f"{current} {part}".strip()
        if not continued:
            logical.append(current)
            current = ""
    if current:
        logical.append(current)
    return logical


def parse_hashed_lock(path: Path) -> dict[str, LockEntry]:
    if not path.is_file():
        raise CompositionError(f"release lock is missing: {path}")
    entries: dict[str, LockEntry] = {}
    problems: list[str] = []
    for line in _logical_requirement_lines(path.read_text(encoding="utf-8")):
        match = _REQ_RE.match(line)
        if not match:
            problems.append(f"not exactly pinned: {line[:120]}")
            continue
        name, version, marker = match.groups()
        key = normalise_name(name)
        hashes = tuple(sorted(set(value.lower() for value in _HASH_RE.findall(line))))
        if not hashes:
            problems.append(f"{name}=={version} has no SHA-256 hashes")
            continue
        storage_key = key
        suffix = 2
        while storage_key in entries:
            storage_key = f"{key}#{suffix}"
            suffix += 1
        entries[storage_key] = LockEntry(
            name=name,
            version=version,
            hashes=hashes,
            marker=(marker or "").strip(),
        )
    if problems:
        raise CompositionError("release lock is not fail-closed:\n  - " + "\n  - ".join(problems))
    if not entries:
        raise CompositionError("release lock contains no packages")
    return entries


def active_lock_entries(
    entries: dict[str, LockEntry],
    *,
    environment: Optional[dict[str, str]] = None,
) -> dict[str, LockEntry]:
    """Return entries whose PEP 508 markers apply to the release runtime."""
    try:
        from packaging.markers import InvalidMarker, Marker, default_environment
    except ImportError as exc:  # pragma: no cover - packaging is itself locked
        raise CompositionError("packaging is required to evaluate release-lock markers") from exc
    marker_environment = default_environment()
    if environment:
        marker_environment.update(environment)
    active: dict[str, LockEntry] = {}
    for entry in entries.values():
        if not entry.marker:
            applies = True
        else:
            try:
                applies = Marker(entry.marker).evaluate(marker_environment)
            except InvalidMarker as exc:
                raise CompositionError(f"invalid marker for {entry.name}=={entry.version}: {entry.marker}") from exc
        if applies:
            key = normalise_name(entry.name)
            if key in active:
                other = active[key]
                raise CompositionError(
                    f"overlapping lock markers select both {other.name}=={other.version} "
                    f"and {entry.name}=={entry.version}"
                )
            active[key] = entry
    return active


def direct_requirement_names(path: Path) -> set[str]:
    if not path.is_file():
        raise CompositionError(f"release requirements are missing: {path}")
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _NAME_RE.match(line)
        if match:
            names.add(normalise_name(match.group(1)))
    if not names:
        raise CompositionError("release requirements contain no packages")
    return names


def _metadata_urls(dist: metadata.Distribution) -> dict[str, str]:
    urls: dict[str, str] = {}
    for value in dist.metadata.get_all("Project-URL") or []:
        label, separator, url = value.partition(",")
        if separator and url.strip().startswith(("https://", "http://")):
            urls[label.strip().lower()] = url.strip()
    homepage = (dist.metadata.get("Home-page") or "").strip()
    if homepage.startswith(("https://", "http://")):
        urls.setdefault("homepage", homepage)
    return urls


def _load_license_evidence(path: Path = DEFAULT_LICENSE_EVIDENCE) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"release license evidence is unreadable: {path}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("packages"), dict):
        raise CompositionError("release license evidence has an unsupported schema")
    return {normalise_name(name): value for name, value in payload["packages"].items()}


def _catalog_license_document(package: str, version: str, evidence: dict) -> dict[str, str] | None:
    if not evidence:
        return None
    if evidence.get("version") != version:
        raise CompositionError(
            f"license evidence for {package} is pinned to {evidence.get('version')}; resolved {version}"
        )
    relative = str(evidence.get("license_file") or "")
    expected_hash = str(evidence.get("license_sha256") or "").lower()
    if not relative:
        return None
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise CompositionError(f"license evidence escapes the repository: {relative}") from exc
    if not path.is_file():
        raise CompositionError(f"license evidence file is missing: {relative}")
    actual_hash = sha256_file(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or actual_hash != expected_hash:
        raise CompositionError(f"license evidence hash mismatch: {relative}")
    return {
        "path": relative,
        "sha256": actual_hash,
        "source_url": str(evidence.get("license_url") or ""),
        "text": path.read_text(encoding="utf-8").strip(),
    }


def _license_expression(dist: metadata.Distribution) -> str:
    expression = (dist.metadata.get("License-Expression") or "").strip()
    if expression and expression.upper() != "UNKNOWN":
        return expression
    classifiers = [
        value.removeprefix("License ::").strip(" :")
        for value in (dist.metadata.get_all("Classifier") or [])
        if value.startswith("License ::")
    ]
    classifier_aliases = {
        "OSI Approved :: Apache Software License": "Apache-2.0",
        "OSI Approved :: BSD License": "BSD",
        "OSI Approved :: MIT License": "MIT",
        "OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
        "OSI Approved :: Python Software Foundation License": "PSF-2.0",
        "OSI Approved :: Zope Public License": "ZPL-2.1",
    }
    normalised_classifiers = [classifier_aliases.get(value, value) for value in classifiers]
    if normalised_classifiers:
        return "; ".join(sorted(set(normalised_classifiers)))
    license_text = (dist.metadata.get("License") or "").strip()
    if license_text and license_text.upper() != "UNKNOWN":
        first_line = next((line.strip() for line in license_text.splitlines() if line.strip()), "")
        if first_line and len(first_line) <= 160:
            aliases = {
                "Apache 2.0": "Apache-2.0",
                "Apache License 2.0": "Apache-2.0",
                "BSD 3-Clause License": "BSD-3-Clause",
                "3-Clause BSD License": "BSD-3-Clause",
                "ZPL 2.1": "ZPL-2.1",
            }
            return aliases.get(first_line, first_line)
    return ""


def _license_documents(dist: metadata.Distribution) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for relative in dist.files or []:
        base = Path(str(relative)).name.lower()
        if not base.startswith(_LICENSE_BASENAMES):
            continue
        located = Path(dist.locate_file(relative))
        try:
            if not located.is_file() or located.stat().st_size > 512 * 1024:
                continue
            text = located.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            documents.append({"path": str(relative), "sha256": sha256_file(located), "text": text})
    unique: dict[str, dict[str, str]] = {}
    for document in documents:
        unique.setdefault(document["sha256"], document)
    return [unique[key] for key in sorted(unique)]


def _distribution_tree_digest(dist: metadata.Distribution) -> tuple[str, int]:
    rows: list[tuple[str, int, str]] = []
    for relative in dist.files or []:
        located = Path(dist.locate_file(relative))
        try:
            if located.is_file():
                rows.append((str(relative).replace("\\", "/"), located.stat().st_size, sha256_file(located)))
        except OSError:
            continue
    if not rows:
        raise CompositionError(f"installed distribution has no hashable files: {dist.metadata.get('Name')}")
    digest = hashlib.sha256()
    for relative, size, item_hash in sorted(rows):
        digest.update(f"{relative}\0{size}\0{item_hash}\n".encode("utf-8"))
    return digest.hexdigest(), len(rows)


def _resolved_dependencies(dist: metadata.Distribution, admitted: set[str]) -> list[str]:
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:  # pragma: no cover - release lock always includes packaging
        raise CompositionError("packaging is required to evaluate resolved dependency markers") from exc
    dependencies: set[str] = set()
    for raw in dist.requires or []:
        try:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
        except Exception as exc:
            raise CompositionError(f"cannot parse dependency metadata for {dist.metadata.get('Name')}: {raw}") from exc
        name = normalise_name(requirement.name)
        if name in admitted:
            dependencies.add(name)
    return sorted(dependencies)


def resolved_python_components(
    entries: dict[str, LockEntry],
    *,
    direct_names: set[str],
) -> list[dict]:
    components: list[dict] = []
    problems: list[str] = []
    admitted = set(entries)
    evidence_catalog = _load_license_evidence()
    for key, entry in sorted(entries.items()):
        try:
            dist = metadata.distribution(entry.name)
        except metadata.PackageNotFoundError:
            problems.append(f"locked distribution is not installed: {entry.name}=={entry.version}")
            continue
        if dist.version != entry.version:
            problems.append(f"resolved {entry.name}=={dist.version}; lock requires {entry.version}")
            continue
        license_expression = _license_expression(dist)
        documents = _license_documents(dist)
        urls = _metadata_urls(dist)
        evidence = evidence_catalog.get(key) or {}
        if evidence and evidence.get("version") != entry.version:
            problems.append(
                f"license evidence for {entry.name} is pinned to {evidence.get('version')}; "
                f"resolved {entry.version}"
            )
            continue
        if evidence.get("license") and evidence["license"] != license_expression:
            problems.append(
                f"license evidence for {entry.name} says {evidence['license']!r}; "
                f"resolved metadata says {license_expression!r}"
            )
            continue
        if not documents:
            try:
                catalog_document = _catalog_license_document(entry.name, entry.version, evidence)
            except CompositionError as exc:
                problems.append(str(exc))
                continue
            if catalog_document:
                documents.append(catalog_document)
        source_url = next(
            (
                urls[label]
                for label in ("source", "repository", "source code", "code", "homepage")
                if label in urls
            ),
            str(evidence.get("source_url") or ""),
        )
        if not license_expression:
            problems.append(f"{entry.name}=={entry.version} has no usable license metadata")
        if not documents:
            problems.append(f"{entry.name}=={entry.version} contains no LICENSE/COPYING/NOTICE document")
        if not source_url:
            problems.append(f"{entry.name}=={entry.version} has no source or homepage URL")
        try:
            installed_digest, installed_file_count = _distribution_tree_digest(dist)
        except CompositionError as exc:
            problems.append(str(exc))
            continue
        components.append(
            {
                "name": normalise_name(dist.metadata.get("Name") or entry.name),
                "version": dist.version,
                "direct": key in direct_names,
                "purl": f"pkg:pypi/{normalise_name(entry.name)}@{entry.version}",
                "license": license_expression,
                "source_url": source_url,
                "download_sha256": list(entry.hashes),
                "installed_tree_sha256": installed_digest,
                "installed_file_count": installed_file_count,
                "dependencies": _resolved_dependencies(dist, admitted),
                "license_documents": documents,
            }
        )
    if problems:
        raise CompositionError("resolved release evidence is incomplete:\n  - " + "\n  - ".join(problems))
    return components


def validate_ffmpeg_provenance(path: Path) -> dict:
    if not path.is_file():
        raise CompositionError(f"FFmpeg release provenance is missing: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"FFmpeg provenance is not valid JSON: {path}") from exc
    source = record.get("source") or {}
    build = record.get("build") or {}
    redistribution = record.get("redistribution") or {}
    missing: list[str] = []
    if not record.get("bundled", {}).get("ok"):
        missing.append("bundled.ok")
    if not source.get("url"):
        missing.append("source.url")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")), re.IGNORECASE):
        missing.append("source.sha256")
    if not build.get("origin"):
        missing.append("build.origin")
    if not build.get("configuration"):
        missing.append("build.configuration")
    if not redistribution.get("license"):
        missing.append("redistribution.license")
    if not redistribution.get("corresponding_source"):
        missing.append("redistribution.corresponding_source")
    artifacts = record.get("artifacts") or []
    if not artifacts:
        missing.append("artifacts")
    for artifact in artifacts:
        artifact_path = Path(str(artifact.get("path", "")))
        expected = str(artifact.get("sha256", ""))
        if not artifact_path.is_file():
            missing.append(f"artifact file {artifact_path}")
        elif expected != sha256_file(artifact_path):
            missing.append(f"artifact hash {artifact_path}")
    if missing:
        raise CompositionError("FFmpeg release provenance is incomplete: " + ", ".join(missing))
    return record


def build_composition(
    *,
    lock_path: Path,
    requirements_path: Path,
    artifact_paths: Sequence[Path],
    ffmpeg_provenance_path: Optional[Path],
    lane: str,
    build_lock_path: Optional[Path] = None,
) -> dict:
    locked_entries = parse_hashed_lock(lock_path)
    entries = active_lock_entries(locked_entries)
    direct_names = direct_requirement_names(requirements_path)
    missing_direct = sorted(direct_names - set(entries))
    if missing_direct:
        raise CompositionError("release lock omits direct requirements: " + ", ".join(missing_direct))
    components = resolved_python_components(entries, direct_names=direct_names)
    bundled_components = []
    if ffmpeg_provenance_path is not None:
        bundled_components.append({"name": "ffmpeg", **validate_ffmpeg_provenance(ffmpeg_provenance_path)})
    version = "0.0.0"
    init_path = REPO_ROOT / "opencut" / "__init__.py"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', init_path.read_text(encoding="utf-8"), re.MULTILINE)
    if match:
        version = match.group(1)
    inputs = {
        "requirements": tree_record(requirements_path),
        "release_lock": tree_record(lock_path),
        "hash_locked": True,
    }
    if build_lock_path is not None:
        parse_hashed_lock(build_lock_path)
        inputs["build_lock"] = tree_record(build_lock_path)
    return {
        "$schema": SCHEMA,
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "application": {"name": "opencut-ppro", "version": version},
        "lane": lane,
        "inputs": inputs,
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "direct_count": sum(1 for component in components if component["direct"]),
            "transitive_count": sum(1 for component in components if not component["direct"]),
            "components": components,
        },
        "bundled_components": bundled_components,
        "artifacts": [tree_record(path) for path in artifact_paths],
    }


def render_notices(composition: dict) -> str:
    app = composition["application"]
    lines = [
        f"OpenCut {app['version']} third-party notices",
        "=" * 72,
        "",
        "This file is generated from the exact resolved release environment.",
        "Download and installed-tree hashes are recorded in release-composition.json.",
        "",
    ]
    for component in composition["python"]["components"]:
        lines.extend(
            [
                f"{component['name']} {component['version']}",
                f"License: {component['license']}",
                f"Source: {component['source_url']}",
                "",
            ]
        )
        for document in component["license_documents"]:
            if document.get("source_url"):
                lines.append(f"License source: {document['source_url']} (SHA-256 {document['sha256']})")
            lines.extend([document["text"].rstrip(), "", "-" * 72, ""])

    for bundled in composition["bundled_components"]:
        source = bundled["source"]
        build = bundled["build"]
        redistribution = bundled["redistribution"]
        lines.extend(
            [
                bundled.get("name", "Bundled component").upper(),
                f"License: {redistribution['license']}",
                f"Exact corresponding source: {source['url']}",
                f"Source SHA-256: {source['sha256']}",
                f"Build origin: {build['origin']}",
                "Build configuration:",
                build["configuration"],
                "",
                "Corresponding-source instructions:",
                redistribution["corresponding_source"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_release_outputs(composition: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    composition_path = output_dir / "release-composition.json"
    notices_path = output_dir / "THIRD-PARTY-NOTICES.txt"
    sbom_path = output_dir / "opencut-artifact-sbom.cyclonedx.json"
    composition_path.write_text(json.dumps(composition, indent=2), encoding="utf-8")
    notices_path.write_text(render_notices(composition), encoding="utf-8")

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import sbom

    resolved_sbom = sbom.build_resolved_sbom(composition)
    sbom_path.write_text(json.dumps(resolved_sbom, indent=2), encoding="utf-8")
    return {"composition": composition_path, "notices": notices_path, "sbom": sbom_path}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--build-lock", type=Path)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--ffmpeg-provenance", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lane", choices=("windows", "linux", "macos"))
    parser.add_argument("--check-lock-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.check_lock_only:
            direct = direct_requirement_names(args.requirements)
            lock_path = args.lock or DEFAULT_LOCK
            locked_entries = parse_hashed_lock(lock_path)
            entries = active_lock_entries(locked_entries)
            missing = sorted(direct - set(entries))
            if missing:
                raise CompositionError(f"{lock_path.name} omits direct requirements: " + ", ".join(missing))
            build_entries = parse_hashed_lock(args.build_lock or DEFAULT_BUILD_LOCK)
            print(
                f"release locks complete: {len(locked_entries)} universal / {len(entries)} active runtime packages; "
                f"{len(build_entries)} build packages; {len(direct)} direct requirements"
            )
            return 0
        if not args.artifact or args.output_dir is None or args.lane is None:
            parser.error("--artifact, --output-dir, and --lane are required unless --check-lock-only is used")
        composition = build_composition(
            lock_path=args.lock or DEFAULT_LOCK,
            requirements_path=args.requirements,
            artifact_paths=args.artifact,
            ffmpeg_provenance_path=args.ffmpeg_provenance,
            lane=args.lane,
            build_lock_path=args.build_lock,
        )
        outputs = write_release_outputs(composition, args.output_dir)
    except CompositionError as exc:
        print(f"release composition failed: {exc}", file=sys.stderr)
        return 1
    print(
        "release composition complete: "
        f"{composition['python']['direct_count']} direct, "
        f"{composition['python']['transitive_count']} transitive Python distributions"
    )
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
