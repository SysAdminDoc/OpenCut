"""Contract tests for decomposed route facade exports."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

ROUTES_ROOT = Path(__file__).resolve().parents[1] / "opencut" / "routes"


FACADE_MODULES = {
    "opencut.routes.wave_l_routes": (
        "opencut.routes.advanced_video_generation_routes",
        "opencut.routes.avatar_generation_routes",
        "opencut.routes.multimodal_image_routes",
        "opencut.routes.music_generation_routes",
        "opencut.routes.speech_generation_routes",
        "opencut.routes.video_enhancement_routes",
        "opencut.routes.video_generation_routes",
    ),
    "opencut.routes.captions": (
        "opencut.routes.caption_analysis_routes",
        "opencut.routes.caption_enhancement_routes",
        "opencut.routes.caption_generation_routes",
        "opencut.routes.caption_interview_routes",
        "opencut.routes.caption_pipeline_routes",
        "opencut.routes.caption_render_routes",
        "opencut.routes.caption_transcript_routes",
    ),
    "opencut.routes.system": (
        "opencut.routes.system_runtime_routes",
        "opencut.routes.system_workspace_routes",
        "opencut.routes.system_whisper_routes",
        "opencut.routes.system_model_routes",
        "opencut.routes.system_integration_routes",
        "opencut.routes.system_social_routes",
        "opencut.routes.system_realtime_routes",
        "opencut.routes.system_diagnostics_routes",
    ),
}


@pytest.mark.parametrize(("facade_name", "submodule_names"), FACADE_MODULES.items())
def test_route_facade_exports_are_explicit_and_unique(facade_name, submodule_names):
    facade = import_module(facade_name)
    exported_by: dict[str, str] = {}

    for submodule_name in submodule_names:
        submodule = import_module(submodule_name)
        exports = getattr(submodule, "__all__", None)

        assert isinstance(exports, list), f"{submodule_name} must define a list __all__"
        assert exports, f"{submodule_name} must export at least one route"
        assert len(exports) == len(set(exports)), f"{submodule_name} has duplicate exports"

        for name in exports:
            assert hasattr(submodule, name), f"{submodule_name} does not define {name}"
            assert name not in exported_by, f"{name} is exported by both {exported_by[name]} and {submodule_name}"
            exported_by[name] = submodule_name
            assert getattr(facade, name) is getattr(submodule, name)


def test_shared_route_helpers_have_single_definitions():
    definitions = {"_json_object_or_400": [], "_stub_503": []}

    for source_path in ROUTES_ROOT.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in definitions:
                definitions[node.name].append(source_path.name)

    assert definitions == {
        "_json_object_or_400": ["_common.py"],
        "_stub_503": ["_common.py"],
    }
