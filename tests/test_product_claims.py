"""Guardrails for product positioning against Premiere's current baseline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _locale(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_readme_does_not_claim_premiere_lacks_native_cleanup_or_caption_workflows():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    comparison = readme.split("## What OpenCut adds beyond Premiere 26", 1)[1].split(
        "## Feature Overview", 1
    )[0]

    assert "Pause-mute only (no ripple delete)" not in comparison
    assert "| Filler word detection | CrisperWhisper verbatim markers + custom words | Not available |" not in comparison
    assert "| Animated captions | 55 styles, word-by-word pop/fade/bounce/glow | Basic captions, cloud translation |" not in comparison
    assert "Not available" not in comparison
    comparison_lower = comparison.lower()
    for differentiator in (
        "reviewable",
        "cross-project",
        "exportable",
        "headless",
    ):
        assert differentiator in comparison_lower
    assert "single-word captions" in comparison_lower


def test_panel_copy_leads_with_scope_review_and_artifacts():
    cep = _locale(ROOT / "extension/com.opencut.panel/client/locales/en.json")
    uxp = _locale(ROOT / "extension/com.opencut.uxp/locales/en.json")

    assert "review" in cep["cut.silence_desc"].lower()
    assert "review" in cep["cut.filler_desc"].lower()
    assert "exportable" in cep["palette.description_captions"].lower()
    assert "explicit" in cep["audio.normalize_desc"].lower()
    assert uxp["uxp.agent.sequence_index_q7_f273"] == "Sequence Report"
    assert "cross-project" in uxp["uxp.agent.sequence_index_hint"]
    assert "sequence index" not in uxp["uxp.guide.agent_text"].lower()


def test_mcp_positioning_names_the_differentiated_automation_surface():
    docs = (ROOT / "docs/MCP_SERVER.md").read_text(encoding="utf-8").lower()
    for term in ("cross-project", "reviewable", "exportable", "headless"):
        assert term in docs
    assert "not marketed as replacements" in docs


def test_native_overlap_commands_keep_differentiated_commands_ahead():
    from opencut.core.command_palette import build_feature_index, fuzzy_search

    index = {entry["id"]: entry for entry in build_feature_index()}
    for feature_id in (
        "normalize_audio",
        "remove_silence",
        "profanity_bleep",
        "loudness_match",
        "add_captions",
        "styled_captions",
        "multilang_subtitle",
        "paper_edit",
        "dead_time",
        "footage_search",
    ):
        assert index[feature_id]["native_overlap"] is True
        assert index[feature_id]["palette_priority"] < 1.0

    assert index["stem_split"]["native_overlap"] is False
    custom = [
        {
            "id": "native",
            "name": "Loudness",
            "description": "Native overlap",
            "category": "audio",
            "aliases": [],
            "route": "/native",
            "tags": [],
            "native_overlap": True,
            "palette_priority": 0.75,
        },
        {
            "id": "differentiated",
            "name": "Loudness",
            "description": "Cross-project report",
            "category": "audio",
            "aliases": [],
            "route": "/differentiated",
            "tags": [],
            "native_overlap": False,
            "palette_priority": 1.0,
        },
    ]
    results = fuzzy_search("loudness", index=custom)
    assert [result["id"] for result in results] == ["differentiated", "native"]
