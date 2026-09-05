"""Generated artefacts (route_manifest.json, model_cards.json, ...).

Do not hand-edit. Regenerate via the tools in ``opencut/tools/``.

``REQUIRED_MANIFESTS`` is declared rather than globbed on purpose. Globbing the
package directory would make a packaging bug undetectable: if the files are
missing from a frozen artifact the glob simply returns nothing and every check
passes. The declared set is the authority at runtime, and
``tests/test_generated_manifest_packaging.py`` keeps it equal to the source tree
so a newly generated manifest cannot be forgotten here.
"""

from __future__ import annotations

from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parent

#: Every manifest that must ship inside the package, source or frozen build.
REQUIRED_MANIFESTS = frozenset({
    "adobe_premierepro_versions.json",
    "adobe_uxp_compatibility.json",
    "api_aliases.json",
    "cep_uxp_parity.json",
    "feature_readiness.json",
    "mcp_agent_skill.json",
    "mcp_extended_tools.json",
    "mcp_server_registry.json",
    "model_cards.json",
    "openapi_contract.json",
    "panel_feature_parity.json",
    "project_facts.json",
    "route_manifest.json",
    "surface_ratchet.json",
    "uxp_migration_dashboard.json",
    "uxp_udt_harness.json",
})


class GeneratedManifestMissing(FileNotFoundError):
    """A generated manifest that must ship with the package is absent.

    Raised instead of falling back silently, because the fallbacks hide a
    packaging defect: a frozen build that omits ``opencut/_generated`` still
    starts, still serves, and only misbehaves in ways that look like unrelated
    feature bugs.
    """

    def __init__(self, names: str | list[str], directory: Path | None = None):
        self.names = [names] if isinstance(names, str) else sorted(names)
        self.directory = Path(directory) if directory is not None else GENERATED_DIR
        listed = ", ".join(self.names)
        super().__init__(
            f"Generated manifest(s) missing from {self.directory}: {listed}. "
            "The package was built without opencut/_generated; rebuild with "
            "collect_data_files('opencut._generated') or reinstall OpenCut."
        )


def manifest_path(name: str) -> Path:
    """Return the on-disk path for a generated manifest without reading it."""
    return GENERATED_DIR / name


def missing_manifests(directory: Path | None = None) -> list[str]:
    """Return the sorted names of required manifests absent from ``directory``."""
    base = Path(directory) if directory is not None else GENERATED_DIR
    return sorted(name for name in REQUIRED_MANIFESTS if not (base / name).is_file())


def require_manifest(name: str) -> Path:
    """Return the path to ``name``, raising when it did not ship.

    Callers that can genuinely operate without the manifest should catch
    :class:`GeneratedManifestMissing` explicitly, so the degraded path is a
    stated decision rather than an incidental ``except OSError``.
    """
    path = manifest_path(name)
    if not path.is_file():
        raise GeneratedManifestMissing(name)
    return path
