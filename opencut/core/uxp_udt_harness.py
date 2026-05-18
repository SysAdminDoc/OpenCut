"""UXP Developer Tool smoke-harness manifest.

F267 turns the code-owned CEP/UXP parity catalogue into a concrete UDT smoke
plan. The manifest is intentionally data-only: the UXP panel runner consumes it
inside Premiere, while release smoke can verify that every direct UXP host action
has a documented payload and safety boundary.
"""

from __future__ import annotations

from copy import deepcopy

from opencut.core.cep_uxp_parity import UXP_TYPINGS, build_manifest

HARNESS_VERSION = 1
SOURCE_PARITY_ARTIFACT = "opencut/_generated/cep_uxp_parity.json"
PANEL_MANIFEST = "extension/com.opencut.uxp/uxp-udt-harness.json"

_PAYLOADS: dict[str, dict] = {
    "ocPing": {},
    "ocGetSequenceInfo": {},
    "ocAddSequenceMarkers": {
        "markers": [
            {
                "time": 0,
                "label": "OpenCut UDT Marker",
                "color": "green",
                "comment": "F267 smoke marker",
            }
        ]
    },
    "ocGetSequenceMarkers": {},
    "ocApplyClipKeyframes": {
        "trackIndex": 0,
        "clipStartTime": 0,
        "keyframes": [
            {"time": 0, "scale": 100, "x": 0, "y": 0},
            {"time": 0.5, "scale": 104, "x": 0, "y": 0},
        ],
    },
    "ocBatchRenameProjectItems": {
        "renames": [
            {
                "oldName": "OpenCut UDT Fixture",
                "newName": "OpenCut UDT Fixture Renamed",
            }
        ]
    },
    "ocCreateSmartBins": {"bins": [{"name": "OpenCut UDT Smoke"}]},
    "ocGetProjectBins": {},
    "ocExportSequenceRange": {
        "startSeconds": 0,
        "endSeconds": 1,
        "outputPath": "__OPENCUT_UDT_TEMP__/opencut-udt-export.mp4",
        "exportType": "immediate",
        "queueToAme": False,
        "startBatch": False,
    },
    "ocRemoveSequenceMarkers": {
        "fingerprints": [
            {
                "time": 0,
                "name": "OpenCut UDT Marker",
                "comment": "F267 smoke marker",
            }
        ]
    },
    "ocUnrenameItems": {
        "renames": [
            {
                "oldName": "OpenCut UDT Fixture Renamed",
                "newName": "OpenCut UDT Fixture",
            }
        ]
    },
    "ocRemoveImportedSequence": {"name": "OpenCut UDT Imported Sequence"},
    "ocSetSequencePlayhead": {"seconds": 0},
    "ocRemoveImportedItem": {"name": "OpenCut UDT Imported Item"},
}

_SCENARIO_SAFETY: dict[str, dict] = {
    "ocPing": {"fixture": "none", "mutates_project": False, "writes_files": False, "safe_by_default": True},
    "ocGetSequenceInfo": {"fixture": "active_sequence", "mutates_project": False, "writes_files": False, "safe_by_default": True},
    "ocAddSequenceMarkers": {"fixture": "active_sequence", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocGetSequenceMarkers": {"fixture": "active_sequence", "mutates_project": False, "writes_files": False, "safe_by_default": True},
    "ocApplyClipKeyframes": {"fixture": "selected_video_clip", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocBatchRenameProjectItems": {"fixture": "project_item_named_open_cut_udt_fixture", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocCreateSmartBins": {"fixture": "open_project", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocGetProjectBins": {"fixture": "open_project", "mutates_project": False, "writes_files": False, "safe_by_default": True},
    "ocExportSequenceRange": {"fixture": "active_sequence_and_writable_temp_path", "mutates_project": False, "writes_files": True, "safe_by_default": False},
    "ocRemoveSequenceMarkers": {"fixture": "marker_created_by_ocAddSequenceMarkers", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocUnrenameItems": {"fixture": "renamed_project_item", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocRemoveImportedSequence": {"fixture": "imported_test_sequence", "mutates_project": True, "writes_files": False, "safe_by_default": False},
    "ocSetSequencePlayhead": {"fixture": "active_sequence", "mutates_project": False, "writes_files": False, "safe_by_default": True},
    "ocRemoveImportedItem": {"fixture": "imported_test_media_item", "mutates_project": True, "writes_files": False, "safe_by_default": False},
}

_RESULT_KEYS: dict[str, tuple[str, ...]] = {
    "ocPing": ("ok", "result", "source"),
    "ocGetSequenceInfo": ("ok", "data"),
    "ocAddSequenceMarkers": ("ok", "added"),
    "ocGetSequenceMarkers": ("ok", "markers", "count"),
    "ocApplyClipKeyframes": ("ok", "applied"),
    "ocBatchRenameProjectItems": ("ok", "renamed"),
    "ocCreateSmartBins": ("ok", "created"),
    "ocGetProjectBins": ("ok", "bins", "count"),
    "ocExportSequenceRange": ("ok", "outputPath"),
    "ocRemoveSequenceMarkers": ("ok", "removed"),
    "ocUnrenameItems": ("ok", "renamed"),
    "ocRemoveImportedSequence": ("ok", "removed"),
    "ocSetSequencePlayhead": ("ok", "seconds"),
    "ocRemoveImportedItem": ("ok", "removed"),
}


def _scenario_for_entry(entry: dict, index: int) -> dict:
    name = entry["name"]
    if name not in _PAYLOADS:
        raise KeyError(f"missing F267 UDT payload for {name}")
    safety = _SCENARIO_SAFETY[name]
    return {
        "id": f"f267-{index:02d}-{name}",
        "action": name,
        "role": entry["role"],
        "risk": entry["risk"],
        "status": entry["status"],
        "uxp_path": entry["uxp_path"],
        "fixture": safety["fixture"],
        "mutates_project": safety["mutates_project"],
        "writes_files": safety["writes_files"],
        "safe_by_default": safety["safe_by_default"],
        "payload": deepcopy(_PAYLOADS[name]),
        "expected_result_keys": list(_RESULT_KEYS[name]),
        "acceptable_blockers": [
            "No open project",
            "No active sequence",
            "UXP API unavailable",
            "Premiere EncoderManager is unavailable",
            "Project item not found",
        ],
    }


def build_udt_harness_manifest() -> dict:
    """Return a JSON-safe UDT smoke harness for the 14 direct UXP actions."""

    source = build_manifest()
    direct_entries = [
        entry for entry in source["functions"] if entry["status"] == "direct_uxp"
    ]
    scenarios = [
        _scenario_for_entry(entry, index)
        for index, entry in enumerate(direct_entries, start=1)
    ]
    return {
        "harness_version": HARNESS_VERSION,
        "f_number": "F267",
        "source_catalogue_version": source["catalogue_version"],
        "source": SOURCE_PARITY_ARTIFACT,
        "panel_manifest": PANEL_MANIFEST,
        "uxp_typings": UXP_TYPINGS,
        "scenario_count": len(scenarios),
        "safe_default_count": sum(1 for scenario in scenarios if scenario["safe_by_default"]),
        "mutating_count": sum(1 for scenario in scenarios if scenario["mutates_project"]),
        "file_write_count": sum(1 for scenario in scenarios if scenario["writes_files"]),
        "actions": [scenario["action"] for scenario in scenarios],
        "run_command": "await window.OpenCutUXPUdtHarness.run({ includeMutating: true })",
        "read_only_command": "await window.OpenCutUXPUdtHarness.run()",
        "scenarios": scenarios,
    }
