import importlib.util
import json
import re
from pathlib import Path

from opencut import model_cards

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD_JSON = REPO_ROOT / "opencut" / "_generated" / "model_cards.json"


def _load_sbom_module():
    spec = importlib.util.spec_from_file_location("opencut_sbom_script", REPO_ROOT / "scripts" / "sbom.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _normalise_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower()).strip("-")


def _requirement_name(requirement: str) -> str:
    requirement = requirement.split(";", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)", requirement)
    return _normalise_package_name(match.group(1)) if match else ""


def _declared_pyproject_dependency_names() -> set[str]:
    raw = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib

        parsed = tomllib.loads(raw)
        project = parsed["project"]
        requirements = list(project.get("dependencies") or [])
        for optional_requirements in (project.get("optional-dependencies") or {}).values():
            requirements.extend(optional_requirements)
    except ModuleNotFoundError:
        sbom = _load_sbom_module()
        sections = sbom._parse_pyproject_dependencies(str(REPO_ROOT / "pyproject.toml"))
        requirements = [
            name
            for section in sections.values()
            for name, _version in section
        ]
    return {name for requirement in requirements if (name := _requirement_name(requirement))}


def _declared_requirements_txt_names() -> set[str]:
    declared: set[str] = set()
    for raw in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            declared.add(_requirement_name(line))
    return declared


def _component_properties(component: dict) -> dict[str, str]:
    return {prop["name"]: prop["value"] for prop in component.get("properties", [])}


def _metadata_properties(bom: dict) -> dict[str, str]:
    return {prop["name"]: prop["value"] for prop in bom["metadata"].get("properties", [])}


def _build_bom() -> dict:
    return _load_sbom_module().build_sbom()


def test_sbom_declares_inventory_fidelity_boundary():
    bom = _build_bom()
    properties = _metadata_properties(bom)

    assert properties["opencut:sbom:fidelity"] == "declared-only"
    assert "pyproject.toml" in properties["opencut:sbom:sources"]
    assert "requirements.txt" in properties["opencut:sbom:sources"]
    assert "requirements-lock.txt" in properties["opencut:sbom:excludes"]
    assert "requirements-lock.txt" in properties["opencut:sbom:vulnerability-audit-targets"]


def test_sbom_contains_every_declared_python_dependency():
    bom = _build_bom()
    declared = _declared_pyproject_dependency_names() | _declared_requirements_txt_names()
    component_names = {
        _normalise_package_name(component["name"])
        for component in bom["components"]
        if component.get("purl", "").startswith("pkg:pypi/")
    }

    assert declared
    assert declared <= component_names


def test_sbom_contains_all_committed_model_cards():
    bom = _build_bom()
    generated_manifest = json.loads(MODEL_CARD_JSON.read_text(encoding="utf-8"))
    card_components = {
        _component_properties(component).get("opencut:model-card:check-name"): component
        for component in bom["components"]
        if _component_properties(component).get("opencut:surface") == "model-card"
    }

    expected_check_names = {card.check_name for card in model_cards.CARDS}
    assert len(expected_check_names) == generated_manifest["total"] == 55
    assert set(card_components) == expected_check_names

    for card in model_cards.CARDS:
        component = card_components[card.check_name]
        properties = _component_properties(component)
        assert component["bom-ref"] == f"opencut:model-card:{card.check_name}"
        assert component["name"] == card.label
        assert properties["opencut:model-card:feature-id"] == card.feature_id
        assert properties["opencut:model-card:privacy"] == card.privacy
        assert component["licenses"] == [{"license": {"name": card.license}}]


def test_sbom_dependencies_graph_references_every_component_once():
    bom = _build_bom()
    root_ref = bom["metadata"]["component"]["bom-ref"]
    component_refs = [component["bom-ref"] for component in bom["components"]]
    dependencies_by_ref = {dependency["ref"]: dependency["dependsOn"] for dependency in bom["dependencies"]}

    assert bom["dependencies"]
    assert len(component_refs) == len(set(component_refs))
    assert set(dependencies_by_ref[root_ref]) == set(component_refs)
    assert set(component_refs) <= set(dependencies_by_ref)
    for ref in component_refs:
        assert dependencies_by_ref[ref] == []
