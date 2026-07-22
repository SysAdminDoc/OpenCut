"""
Generate CycloneDX 1.5 SBOMs for OpenCut source or resolved artifacts.

Reads dependency declarations from ``pyproject.toml`` +
``requirements.txt`` plus OpenCut model-card metadata and emits a
declared-dependency CycloneDX JSON document at
``dist/opencut-declared-sbom.cyclonedx.json`` (or XML with ``--format xml``).

The default source view records declared dependencies.  Release builds pass a
validated ``release-composition.json`` to ``--composition``; that view records
the exact direct/transitive distributions, their installed-tree and locked
download hashes, bundled FFmpeg provenance, and packaged artifact digests.

Usage::

    python scripts/sbom.py                # JSON to dist/
    python scripts/sbom.py --format xml
    python scripts/sbom.py --output path/to/file.json
    python scripts/sbom.py --composition dist/release/release-composition.json

Zero-dependency: stdlib only. No ``cyclonedx-bom`` install required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SBOM_BASENAME = "opencut-declared-sbom.cyclonedx"
RESOLVED_SBOM_BASENAME = "opencut-artifact-sbom.cyclonedx"
SBOM_FIDELITY = "declared-only"
SBOM_SOURCE_LABEL = "pyproject.toml, requirements.txt, opencut.model_cards"
SBOM_EXCLUDES_LABEL = "installed transitive packages and requirements-lock.txt pins"
SBOM_AUDIT_LABEL = "requirements.txt, requirements-lock.txt, pyproject[all] via opencut.tools.pip_audit_extras"


def _read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _opencut_version() -> str:
    init_py = _read_file(os.path.join(REPO_ROOT, "opencut", "__init__.py"))
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _parse_requirements_txt(path: str) -> List[Tuple[str, Optional[str]]]:
    """Return ``[(package, pinned_version)]`` from a ``requirements.txt``."""
    out: List[Tuple[str, Optional[str]]] = []
    for raw in _read_file(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip inline comments
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # Handle `pkg==x.y.z`, `pkg>=x.y`, `pkg[extra]==...`, `pkg @ url`
        if " @ " in line:
            pkg_part, _ = line.split(" @ ", 1)
            name = re.split(r"\[|==|>=|<=|~=|>|<", pkg_part, maxsplit=1)[0].strip()
            out.append((name, None))
            continue
        m = re.match(
            r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*"
            r"(==|>=|<=|~=|>|<)\s*"
            r"([A-Za-z0-9_.\-]+)",
            line,
        )
        if m:
            name = m.group(1)
            op = m.group(2)
            ver = m.group(3)
            out.append((name, ver if op == "==" else None))
            continue
        # Plain `pkg` with no pin
        name = re.split(r"\[", line, maxsplit=1)[0].strip()
        if name:
            out.append((name, None))
    return out


def _parse_pyproject_dependencies(path: str) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Return {'runtime': [...], 'extras.ai': [...], ...} from pyproject.toml.

    Python 3.11+ has :mod:`tomllib` in stdlib; on earlier Pythons we
    degrade to a regex pass that only catches the common shapes.
    """
    raw = _read_file(path)
    if not raw:
        return {}
    try:
        import tomllib  # Python 3.11+
        parsed = tomllib.loads(raw)
    except ImportError:
        # Fallback — regex-extract the relevant arrays
        return _regex_pyproject(raw)

    sections: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    project = parsed.get("project") or {}
    runtime = project.get("dependencies") or []
    sections["runtime"] = [_split_requirement(r) for r in runtime]
    extras = project.get("optional-dependencies") or {}
    for name, reqs in extras.items():
        sections[f"extras.{name}"] = [_split_requirement(r) for r in reqs]
    return sections


def _regex_pyproject(raw: str) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Pre-3.11 fallback for pyproject parsing — best-effort."""
    out: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    deps_block = re.search(r"dependencies\s*=\s*\[(.*?)\]", raw, re.DOTALL)
    if deps_block:
        items = re.findall(r'"([^"]+)"', deps_block.group(1))
        out["runtime"] = [_split_requirement(i) for i in items]
    return out


def _split_requirement(req: str) -> Tuple[str, Optional[str]]:
    """Split a PEP 508 requirement into (name, pinned_version_or_None)."""
    req = (req or "").strip()
    if not req or req.startswith("#"):
        return ("", None)
    m = re.match(
        r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*"
        r"(==|>=|<=|~=|>|<)\s*"
        r"([A-Za-z0-9_.\-]+)",
        req,
    )
    if m:
        return (m.group(1), m.group(3) if m.group(2) == "==" else None)
    return (re.split(r"\[|\s|;", req, maxsplit=1)[0], None)


def _normalise_pypi_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower()).strip("-")


def _purl(name: str, version: Optional[str]) -> str:
    name = _normalise_pypi_name(name)
    if version:
        return f"pkg:pypi/{name}@{version}"
    return f"pkg:pypi/{name}"


# ---------------------------------------------------------------------------
# CycloneDX builder
# ---------------------------------------------------------------------------

def _make_component(
    name: str,
    version: Optional[str],
    *,
    scope: str = "required",
    component_type: str = "library",
    bom_ref: str = "",
    purl: str = "",
    properties: Optional[List[Dict[str, str]]] = None,
    external_references: Optional[List[Dict[str, str]]] = None,
    licenses: Optional[List[Dict[str, Dict[str, str]]]] = None,
) -> Dict:
    component_ref = bom_ref or _purl(name, version)
    component: Dict = {
        "type": component_type,
        "bom-ref": component_ref,
        "name": name,
        "scope": scope,
    }
    if purl or (component_type == "library" and not bom_ref):
        component["purl"] = purl or _purl(name, version)
    if version:
        component["version"] = version
    if properties:
        component["properties"] = properties
    if external_references:
        component["externalReferences"] = external_references
    if licenses:
        component["licenses"] = licenses
    return component


def _record_dependency(
    records: Dict[str, Dict],
    name: str,
    version: Optional[str],
    *,
    scope: str,
    source: str,
) -> None:
    key = _normalise_pypi_name(name)
    if not key:
        return
    record = records.setdefault(
        key,
        {
            "name": key,
            "version": None,
            "scope": "optional",
            "sources": set(),
        },
    )
    if version and not record["version"]:
        record["version"] = version
    if scope == "required":
        record["scope"] = "required"
    record["sources"].add(source)


def _dependency_components(records: Dict[str, Dict]) -> List[Dict]:
    components: List[Dict] = []
    for key in sorted(records):
        record = records[key]
        sources = ", ".join(sorted(record["sources"]))
        components.append(
            _make_component(
                record["name"],
                record["version"],
                scope=record["scope"],
                properties=[{"name": "opencut:dependency-sources", "value": sources}],
            )
        )
    return components


def _model_card_bom_ref(check_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", check_name.lower()).strip("-")
    return f"opencut:model-card:{slug}"


def _model_card_components() -> List[Dict]:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from opencut.model_cards import CARDS

    components: List[Dict] = []
    for card in sorted(CARDS, key=lambda c: c.check_name):
        properties = [
            {"name": "opencut:surface", "value": "model-card"},
            {"name": "opencut:model-card:check-name", "value": card.check_name},
            {"name": "opencut:model-card:feature-id", "value": card.feature_id},
            {"name": "opencut:model-card:category", "value": card.category},
            {"name": "opencut:model-card:hardware", "value": card.hardware},
            {"name": "opencut:model-card:install-hint", "value": card.install_hint},
            {"name": "opencut:model-card:privacy", "value": card.privacy},
        ]
        if card.requires_checkpoint_env:
            properties.append(
                {
                    "name": "opencut:model-card:requires-checkpoint-env",
                    "value": card.requires_checkpoint_env,
                }
            )

        components.append(
            _make_component(
                card.label,
                None,
                scope="optional",
                bom_ref=_model_card_bom_ref(card.check_name),
                properties=properties,
                external_references=[{"type": "website", "url": card.upstream}],
                licenses=[{"license": {"name": card.license}}],
            )
        )
    return components


def _dedup_components(components: Iterable[Dict]) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for component in components:
        ref = component.get("bom-ref", "")
        if not ref:
            continue
        if ref not in seen:
            seen[ref] = component
            continue
        existing = seen[ref]
        if component.get("scope") == "required":
            existing["scope"] = "required"
        if component.get("version") and not existing.get("version"):
            existing["version"] = component["version"]
        for key in ("properties", "externalReferences", "licenses"):
            existing_items = existing.setdefault(key, [])
            for item in component.get(key, []):
                if item not in existing_items:
                    existing_items.append(item)
    return [seen[ref] for ref in sorted(seen)]


def _dedup(
    pairs: Iterable[Tuple[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    seen: Dict[str, Optional[str]] = {}
    for name, ver in pairs:
        if not name:
            continue
        key = _normalise_pypi_name(name)
        if key not in seen or (ver and not seen[key]):
            seen[key] = ver
    return [(n, v) for n, v in seen.items()]


def _dependency_graph(root_ref: str, components: List[Dict]) -> List[Dict]:
    component_refs = sorted(c["bom-ref"] for c in components)
    return [{"ref": root_ref, "dependsOn": component_refs}] + [
        {"ref": ref, "dependsOn": []}
        for ref in component_refs
    ]


def build_sbom() -> Dict:
    opencut_ver = _opencut_version()
    req_txt = _parse_requirements_txt(os.path.join(REPO_ROOT, "requirements.txt"))
    pyproject = _parse_pyproject_dependencies(os.path.join(REPO_ROOT, "pyproject.toml"))

    dependency_records: Dict[str, Dict] = {}
    for n, v in pyproject.get("runtime", []):
        _record_dependency(dependency_records, n, v, scope="required", source="pyproject.runtime")
    for n, v in req_txt:
        _record_dependency(dependency_records, n, v, scope="required", source="requirements.txt")

    for extra_key, reqs in sorted(pyproject.items()):
        if not extra_key.startswith("extras."):
            continue
        for n, v in reqs:
            _record_dependency(dependency_records, n, v, scope="optional", source=extra_key)

    components = _dedup_components(
        [
            *_dependency_components(dependency_records),
            *_model_card_components(),
        ]
    )

    root_component = {
        "type": "application",
        "bom-ref": f"pkg:pypi/opencut-ppro@{opencut_ver}",
        "name": "opencut-ppro",
        "version": opencut_ver,
        "purl": f"pkg:pypi/opencut-ppro@{opencut_ver}",
        "description": "Video editing automation for Adobe Premiere Pro",
    }

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tools": [
                {
                    "vendor": "opencut",
                    "name": "scripts/sbom.py",
                    "version": opencut_ver,
                },
            ],
            "component": root_component,
            "properties": [
                {"name": "opencut:sbom:fidelity", "value": SBOM_FIDELITY},
                {"name": "opencut:sbom:sources", "value": SBOM_SOURCE_LABEL},
                {"name": "opencut:sbom:excludes", "value": SBOM_EXCLUDES_LABEL},
                {"name": "opencut:sbom:vulnerability-audit-targets", "value": SBOM_AUDIT_LABEL},
            ],
        },
        "components": components,
        "dependencies": _dependency_graph(root_component["bom-ref"], components),
    }


def build_resolved_sbom(composition: Dict) -> Dict:
    """Build an artifact SBOM from a validated release composition."""
    if composition.get("schema_version") != 1:
        raise ValueError("unsupported or missing release composition schema_version")
    application = composition.get("application") or {}
    app_name = application.get("name") or "opencut-ppro"
    app_version = application.get("version") or "0.0.0"
    root_ref = f"pkg:pypi/{app_name}@{app_version}"
    components: List[Dict] = []
    dependency_rows: List[Dict] = []

    python_components = composition.get("python", {}).get("components") or []
    python_refs: Dict[str, str] = {}
    for resolved in python_components:
        name = _normalise_pypi_name(str(resolved["name"]))
        version = str(resolved["version"])
        ref = str(resolved.get("purl") or _purl(name, version))
        python_refs[name] = ref
        component = _make_component(
            name,
            version,
            scope="required",
            bom_ref=ref,
            purl=ref,
            properties=[
                {
                    "name": "opencut:dependency-kind",
                    "value": "direct" if resolved.get("direct") else "transitive",
                },
                {
                    "name": "opencut:installed-tree-sha256",
                    "value": str(resolved["installed_tree_sha256"]),
                },
                {
                    "name": "opencut:locked-download-sha256",
                    "value": ",".join(resolved.get("download_sha256") or []),
                },
            ],
            external_references=[{"type": "website", "url": str(resolved["source_url"])}],
            licenses=[{"license": {"name": str(resolved["license"])}}],
        )
        component["hashes"] = [
            {"alg": "SHA-256", "content": str(resolved["installed_tree_sha256"])}
        ]
        components.append(component)

    for resolved in python_components:
        name = _normalise_pypi_name(str(resolved["name"]))
        ref = python_refs[name]
        dependency_rows.append(
            {
                "ref": ref,
                "dependsOn": sorted(
                    python_refs[dependency]
                    for dependency in resolved.get("dependencies") or []
                    if dependency in python_refs
                ),
            }
        )

    bundled_refs: List[str] = []
    for bundled in composition.get("bundled_components") or []:
        name = str(bundled.get("name") or "bundled-component")
        version = str((bundled.get("bundled") or {}).get("version") or "unknown")
        ref = f"opencut:bundled:{name}@{version}"
        bundled_refs.append(ref)
        source = bundled.get("source") or {}
        build = bundled.get("build") or {}
        redistribution = bundled.get("redistribution") or {}
        artifact_hashes = [
            {"alg": "SHA-256", "content": str(item["sha256"])}
            for item in bundled.get("artifacts") or []
        ]
        component = _make_component(
            name,
            version,
            scope="required",
            component_type="application",
            bom_ref=ref,
            properties=[
                {"name": "opencut:source-sha256", "value": str(source.get("sha256", ""))},
                {"name": "opencut:build-origin", "value": str(build.get("origin", ""))},
                {"name": "opencut:build-configuration", "value": str(build.get("configuration", ""))},
                {
                    "name": "opencut:corresponding-source",
                    "value": str(redistribution.get("corresponding_source", "")),
                },
            ],
            external_references=[{"type": "distribution", "url": str(source.get("url", ""))}],
            licenses=[{"license": {"name": str(redistribution.get("license", ""))}}],
        )
        component["hashes"] = artifact_hashes
        components.append(component)
        dependency_rows.append({"ref": ref, "dependsOn": []})

    artifact_refs: List[str] = []
    for index, artifact in enumerate(composition.get("artifacts") or []):
        name = os.path.basename(str(artifact["path"]).rstrip("/\\")) or f"artifact-{index}"
        ref = f"opencut:artifact:{index}:{artifact['sha256']}"
        artifact_refs.append(ref)
        component = _make_component(
            name,
            None,
            scope="required",
            component_type="file",
            bom_ref=ref,
            properties=[
                {"name": "opencut:artifact-kind", "value": str(artifact["kind"])},
                {"name": "opencut:artifact-size", "value": str(artifact["size"])},
                {"name": "opencut:artifact-file-count", "value": str(artifact["file_count"])},
            ],
        )
        component["hashes"] = [{"alg": "SHA-256", "content": str(artifact["sha256"])}]
        components.append(component)
        dependency_rows.append({"ref": ref, "dependsOn": []})

    direct_refs = sorted(
        python_refs[_normalise_pypi_name(str(item["name"]))]
        for item in python_components
        if item.get("direct")
    )
    root_component = {
        "type": "application",
        "bom-ref": root_ref,
        "name": app_name,
        "version": app_version,
        "purl": root_ref,
        "description": "Video editing automation for Adobe Premiere Pro",
    }
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": composition.get("generated_at")
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tools": [{"vendor": "opencut", "name": "scripts/sbom.py", "version": app_version}],
            "component": root_component,
            "properties": [
                {"name": "opencut:sbom:fidelity", "value": "resolved-artifact"},
                {"name": "opencut:sbom:lane", "value": str(composition.get("lane", "unknown"))},
                {
                    "name": "opencut:sbom:composition-schema",
                    "value": str(composition.get("$schema", "")),
                },
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": root_ref, "dependsOn": sorted(set(direct_refs + bundled_refs + artifact_refs))},
            *dependency_rows,
        ],
    }

# ---------------------------------------------------------------------------
# XML serialiser (pure stdlib, CycloneDX 1.5 namespace)
# ---------------------------------------------------------------------------

def _to_xml(bom: Dict) -> str:
    from xml.etree.ElementTree import Element, SubElement, tostring

    ns = "http://cyclonedx.org/schema/bom/1.5"
    root = Element(
        "bom",
        {
            "xmlns": ns,
            "serialNumber": bom.get("serialNumber", ""),
            "version": str(bom.get("version", 1)),
        },
    )

    meta = SubElement(root, "metadata")
    ts = SubElement(meta, "timestamp")
    ts.text = bom["metadata"]["timestamp"]
    if bom["metadata"].get("properties"):
        props_el = SubElement(meta, "properties")
        for prop in bom["metadata"]["properties"]:
            prop_el = SubElement(props_el, "property", {"name": prop.get("name", "")})
            prop_el.text = str(prop.get("value", ""))
    rc = bom["metadata"]["component"]
    comp = SubElement(
        meta,
        "component",
        {"type": "application", "bom-ref": rc.get("bom-ref", "")},
    )
    for key in ("name", "version", "purl", "description"):
        if rc.get(key):
            el = SubElement(comp, key)
            el.text = str(rc[key])

    components_el = SubElement(root, "components")
    for c in bom["components"]:
        ce = SubElement(
            components_el, "component",
            {"type": c.get("type", "library"), "bom-ref": c.get("bom-ref", "")},
        )
        for key in ("name", "version", "purl", "scope", "description"):
            if c.get(key):
                el = SubElement(ce, key)
                el.text = str(c[key])
        if c.get("licenses"):
            licenses_el = SubElement(ce, "licenses")
            for license_entry in c["licenses"]:
                license_el = SubElement(licenses_el, "license")
                name = license_entry.get("license", {}).get("name")
                if name:
                    name_el = SubElement(license_el, "name")
                    name_el.text = str(name)
        if c.get("hashes"):
            hashes_el = SubElement(ce, "hashes")
            for item_hash in c["hashes"]:
                hash_el = SubElement(hashes_el, "hash", {"alg": item_hash.get("alg", "SHA-256")})
                hash_el.text = str(item_hash.get("content", ""))
        if c.get("externalReferences"):
            refs_el = SubElement(ce, "externalReferences")
            for ref in c["externalReferences"]:
                attrs = {"type": ref.get("type", "")}
                ref_el = SubElement(refs_el, "reference", attrs)
                url = ref.get("url")
                if url:
                    url_el = SubElement(ref_el, "url")
                    url_el.text = str(url)
        if c.get("properties"):
            props_el = SubElement(ce, "properties")
            for prop in c["properties"]:
                prop_el = SubElement(props_el, "property", {"name": prop.get("name", "")})
                prop_el.text = str(prop.get("value", ""))

    if bom.get("dependencies"):
        dependencies_el = SubElement(root, "dependencies")
        for dependency in bom["dependencies"]:
            dep_el = SubElement(dependencies_el, "dependency", {"ref": dependency.get("ref", "")})
            for ref in dependency.get("dependsOn", []):
                SubElement(dep_el, "dependency", {"ref": ref})

    return tostring(root, encoding="utf-8", xml_declaration=True).decode()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--format", choices=("json", "xml"), default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output", default="",
        help=f"Output path. Default: dist/{SBOM_BASENAME}.{{json,xml}}",
    )
    parser.add_argument(
        "--composition", default="",
        help="Validated release-composition.json; emits an exact artifact SBOM instead of the source view.",
    )
    args = parser.parse_args(argv)

    if args.composition:
        with open(args.composition, encoding="utf-8") as handle:
            bom = build_resolved_sbom(json.load(handle))
    else:
        bom = build_sbom()

    if args.output:
        out_path = args.output
    else:
        dist_dir = os.path.join(REPO_ROOT, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        suffix = "json" if args.format == "json" else "xml"
        basename = RESOLVED_SBOM_BASENAME if args.composition else SBOM_BASENAME
        out_path = os.path.join(dist_dir, f"{basename}.{suffix}")

    if args.format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bom, f, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(_to_xml(bom))

    n_runtime = sum(1 for c in bom["components"] if c.get("scope") == "required")
    n_optional = sum(1 for c in bom["components"] if c.get("scope") == "optional")
    n_model_cards = sum(
        1
        for c in bom["components"]
        for prop in c.get("properties", [])
        if prop == {"name": "opencut:surface", "value": "model-card"}
    )
    print(
        f"SBOM written: {out_path}\n"
        f"  runtime:  {n_runtime}\n"
        f"  optional: {n_optional}\n"
        f"  model cards: {n_model_cards}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
