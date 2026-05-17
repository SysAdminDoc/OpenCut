"""Pin the generated /api alias manifest (F199)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ALIAS_MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "api_aliases.json"


@pytest.fixture(scope="module")
def committed_alias_manifest() -> dict:
    assert ALIAS_MANIFEST_PATH.exists(), (
        "opencut/_generated/api_aliases.json missing; run "
        "`python -m opencut.tools.dump_api_aliases` to regenerate"
    )
    return json.loads(ALIAS_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_alias_manifest() -> dict:
    from opencut.tools.dump_api_aliases import build_alias_manifest

    return build_alias_manifest()


def test_committed_alias_manifest_matches_live_app(committed_alias_manifest, live_alias_manifest):
    from opencut.tools.dump_api_aliases import diff_alias_manifests

    diffs = diff_alias_manifests(committed_alias_manifest, live_alias_manifest)

    assert not diffs, (
        "api_aliases.json is out of sync with the live Flask app.\n"
        "Run `python -m opencut.tools.dump_api_aliases` and commit the result.\n"
        "First diffs:\n  - " + "\n  - ".join(diffs[:10])
    )


def test_alias_manifest_counts_are_consistent(committed_alias_manifest):
    assert committed_alias_manifest["version"] >= 1
    assert committed_alias_manifest["total_api_routes"] == (
        committed_alias_manifest["alias_count"] + committed_alias_manifest["api_only_count"]
    )
    assert committed_alias_manifest["alias_count"] >= 1
    assert committed_alias_manifest["api_only_count"] >= 1
    assert committed_alias_manifest["total_api_routes"] >= 200


def test_alias_entries_point_to_bare_canonical_routes(committed_alias_manifest):
    for entry in committed_alias_manifest["aliases"]:
        assert entry["alias_rule"].startswith("/api/")
        assert not entry["canonical_rule"].startswith("/api/")
        assert entry["alias_rule"][4:] == entry["canonical_rule"]
        assert entry["methods"], f"{entry['alias_rule']} must list methods"


def test_api_only_entries_are_not_mislabeled_aliases(committed_alias_manifest):
    alias_keys = {
        (entry["alias_rule"], tuple(entry["methods"]))
        for entry in committed_alias_manifest["aliases"]
    }
    for entry in committed_alias_manifest["api_only"]:
        assert entry["rule"].startswith("/api/")
        assert (entry["rule"], tuple(entry["methods"])) not in alias_keys
        assert entry["canonical_candidate"] == entry["rule"][4:]
