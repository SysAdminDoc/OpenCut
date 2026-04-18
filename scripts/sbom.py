"""
Generate a CycloneDX 1.5 SBOM for the OpenCut repository.

Reads dependency declarations from ``pyproject.toml`` +
``requirements.txt`` and emits a CycloneDX JSON document at
``dist/opencut-sbom.cyclonedx.json`` (or XML with ``--format xml``).

Does **not** walk installed site-packages — the output reflects the
repository's *declared* dependency surface, which is what downstream
security scanners and compliance reviewers care about.  For an
installed-packages SBOM, use ``cyclonedx-py`` or ``syft`` on a
populated virtualenv.

Usage::

    python scripts/sbom.py                # JSON to dist/
    python scripts/sbom.py --format xml
    python scripts/sbom.py --output path/to/file.json

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


def _purl(name: str, version: Optional[str]) -> str:
    name = (name or "").lower().replace("_", "-")
    if version:
        return f"pkg:pypi/{name}@{version}"
    return f"pkg:pypi/{name}"


# ---------------------------------------------------------------------------
# CycloneDX builder
# ---------------------------------------------------------------------------

def _make_component(
    name: str, version: Optional[str], *, scope: str = "required",
) -> Dict:
    component: Dict = {
        "type": "library",
        "bom-ref": _purl(name, version),
        "name": name,
        "purl": _purl(name, version),
        "scope": scope,
    }
    if version:
        component["version"] = version
    return component


def _dedup(
    pairs: Iterable[Tuple[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    seen: Dict[str, Optional[str]] = {}
    for name, ver in pairs:
        if not name:
            continue
        key = name.lower().replace("_", "-")
        if key not in seen or (ver and not seen[key]):
            seen[key] = ver
    return [(n, v) for n, v in seen.items()]


def build_sbom() -> Dict:
    opencut_ver = _opencut_version()
    req_txt = _parse_requirements_txt(os.path.join(REPO_ROOT, "requirements.txt"))
    pyproject = _parse_pyproject_dependencies(os.path.join(REPO_ROOT, "pyproject.toml"))

    runtime: List[Tuple[str, Optional[str]]] = []
    runtime.extend(pyproject.get("runtime", []))
    runtime.extend(req_txt)
    runtime = _dedup(runtime)

    components = [
        _make_component(n, v, scope="required")
        for n, v in sorted(runtime)
    ]

    for extra_key, reqs in sorted(pyproject.items()):
        if not extra_key.startswith("extras."):
            continue
        for n, v in _dedup(reqs):
            components.append(
                _make_component(n, v, scope="optional")
            )

    root_component = {
        "type": "application",
        "bom-ref": f"pkg:pypi/opencut@{opencut_ver}",
        "name": "opencut",
        "version": opencut_ver,
        "purl": f"pkg:pypi/opencut@{opencut_ver}",
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
        },
        "components": components,
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
    comp = SubElement(meta, "component", {"type": "application"})
    rc = bom["metadata"]["component"]
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
        for key in ("name", "version", "purl", "scope"):
            if c.get(key):
                el = SubElement(ce, key)
                el.text = str(c[key])

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
        help="Output path. Default: dist/opencut-sbom.cyclonedx.{json,xml}",
    )
    args = parser.parse_args(argv)

    bom = build_sbom()

    if args.output:
        out_path = args.output
    else:
        dist_dir = os.path.join(REPO_ROOT, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        suffix = "json" if args.format == "json" else "xml"
        out_path = os.path.join(dist_dir, f"opencut-sbom.cyclonedx.{suffix}")

    if args.format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bom, f, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(_to_xml(bom))

    n_runtime = sum(1 for c in bom["components"] if c.get("scope") == "required")
    n_optional = sum(1 for c in bom["components"] if c.get("scope") == "optional")
    print(
        f"SBOM written: {out_path}\n"
        f"  runtime:  {n_runtime}\n"
        f"  optional: {n_optional}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
