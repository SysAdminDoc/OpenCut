"""Tests for the source-linked GitHub issue seeder (F097)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed_github_issues.py"


def load_seeder():
    spec = importlib.util.spec_from_file_location("seed_github_issues_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_yaml_loader_handles_seed_manifest():
    seeder = load_seeder()
    text = (REPO_ROOT / ".github" / "issue-seeds.yml").read_text(encoding="utf-8")

    parsed = seeder._load_yaml_minimal(text)

    assert isinstance(parsed, dict)
    assert "seeds" in parsed
    assert isinstance(parsed["seeds"], list)
    assert all(isinstance(entry, dict) for entry in parsed["seeds"])


def test_minimal_yaml_loader_handles_labels_manifest():
    seeder = load_seeder()
    text = (REPO_ROOT / ".github" / "labels.yml").read_text(encoding="utf-8")

    parsed = seeder._load_yaml_minimal(text)

    # Top-level is a list, but the embedded parser wraps it in a dict.
    # Allow either shape so PyYAML and our shim produce equivalent output.
    if isinstance(parsed, dict):
        entries = parsed.get("labels") or list(parsed.values())
    else:
        entries = parsed
    if isinstance(entries, list) and entries and isinstance(entries[0], list):
        # Our shim represents an unkeyed top-level list as a nested list.
        entries = entries[0]
    assert isinstance(entries, list)
    assert any(item.get("name") == "type:bug" for item in entries)


def test_load_seeds_returns_parsed_entries():
    seeder = load_seeder()
    seeds = seeder.load_seeds()

    assert len(seeds) >= 14, "should preserve every historical seed row"
    ids = {seed.roadmap_id for seed in seeds}
    assert {"F097", "F098", "F099", "F100", "F111", "F112", "F115", "F117", "F118"}.issubset(ids)
    status_by_id = {seed.roadmap_id: seed.status for seed in seeds}
    assert status_by_id["F098"] == "shipped"
    assert status_by_id["F101"] == "archived"

    for seed in seeds:
        assert seed.title.startswith(f"{seed.roadmap_id}"), seed.title
        assert seed.body.strip(), f"empty body for {seed.roadmap_id}"
        assert seed.labels, f"no labels for {seed.roadmap_id}"
        assert seed.status in {"active", "archived", "shipped"}


def test_seed_titles_dedup_on_roadmap_id():
    seeder = load_seeder()
    seeds = seeder.load_seeds()
    sample = seeds[0]

    assert seeder.issue_already_seeded(sample, [f"{sample.roadmap_id}: existing"])
    assert not seeder.issue_already_seeded(sample, ["unrelated bug report"])


def test_load_labels_validates_hex_color():
    seeder = load_seeder()
    labels = seeder.load_labels()

    names = {label.name for label in labels}
    assert {"type:bug", "type:feature", "good first issue", "roadmap:now"}.issubset(names)
    for label in labels:
        assert len(label.color) == 6, label


def test_apply_seeds_dry_run_does_not_invoke_gh(monkeypatch):
    seeder = load_seeder()
    seeds = seeder.load_seeds()

    def _fail(*_args, **_kwargs):
        raise AssertionError("gh CLI must not be invoked in dry-run mode")

    monkeypatch.setattr(seeder, "_run_gh", _fail)
    monkeypatch.setattr(seeder, "_gh_available", lambda: False)

    actions = seeder.apply_seeds(
        "owner/repo",
        seeds,
        dry_run=True,
        good_first_only=False,
        once=False,
    )

    assert len(actions) == len(seeds)
    assert all(action.startswith("SKIP ") for action in actions)
    assert any("shipped seed" in action for action in actions)
    assert any("archived seed" in action for action in actions)


def test_active_seed_must_match_unchecked_roadmap_item(tmp_path):
    seeder = load_seeder()
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Roadmap\n\n- [ ] P2 — F200 active tracker seed\n",
        encoding="utf-8",
    )
    seed = seeder.Seed(
        roadmap_id="F200",
        title="F200: active tracker seed",
        body="Create the matching issue.",
        labels=["type:chore"],
    )

    actions = seeder.apply_seeds(
        "owner/repo",
        [seed],
        dry_run=True,
        good_first_only=False,
        once=False,
        roadmap_path=roadmap,
    )

    assert len(actions) == 1
    assert actions[0].startswith("DRY F200: gh issue create ")


def test_stale_active_seed_fails_before_issue_creation(tmp_path, monkeypatch):
    seeder = load_seeder()
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("# Roadmap\n\n- [ ] P2 — unrelated active item\n", encoding="utf-8")
    seed = seeder.Seed(
        roadmap_id="F201",
        title="F201: stale tracker seed",
        body="This should not create an issue.",
        labels=["type:chore"],
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("gh CLI must not be invoked for stale seeds")

    monkeypatch.setattr(seeder, "_run_gh", _fail)

    with pytest.raises(RuntimeError, match="stale active issue seed"):
        seeder.apply_seeds(
            "owner/repo",
            [seed],
            dry_run=False,
            good_first_only=False,
            once=False,
            roadmap_path=roadmap,
        )


def test_shipped_seed_is_skipped_even_if_roadmap_mentions_it(tmp_path):
    seeder = load_seeder()
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text("# Roadmap\n\n- [ ] P2 — F202 historical cleanup\n", encoding="utf-8")
    seed = seeder.Seed(
        roadmap_id="F202",
        title="F202: historical cleanup",
        body="Already shipped.",
        labels=["type:chore"],
        status="shipped",
    )

    actions = seeder.apply_seeds(
        "owner/repo",
        [seed],
        dry_run=True,
        good_first_only=False,
        once=False,
        roadmap_path=roadmap,
    )

    assert actions == ["SKIP F202: shipped seed is not active in ROADMAP.md"]


def test_apply_labels_dry_run_does_not_require_gh(monkeypatch):
    seeder = load_seeder()
    labels = seeder.load_labels()

    def _fail(*_args, **_kwargs):
        raise AssertionError("gh CLI must not be invoked in dry-run mode")

    monkeypatch.setattr(seeder, "_run_gh", _fail)
    monkeypatch.setattr(seeder, "_gh_available", lambda: False)

    actions = seeder.apply_labels("owner/repo", labels, dry_run=True)

    assert len(actions) == len(labels)
    assert all(action.startswith("DRY: gh label create ") for action in actions)


def test_good_first_filter_picks_labeled_active_seeds(tmp_path):
    seeder = load_seeder()
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Roadmap\n\n- [ ] P2 — F300 starter task\n- [ ] P2 — F301 regular task\n",
        encoding="utf-8",
    )
    seeds = [
        seeder.Seed(
            roadmap_id="F300",
            title="F300: starter task",
            body="Starter task.",
            labels=["type:chore", "good first issue"],
            good_first=True,
        ),
        seeder.Seed(
            roadmap_id="F301",
            title="F301: regular task",
            body="Regular task.",
            labels=["type:chore"],
        ),
        seeder.Seed(
            roadmap_id="F302",
            title="F302: archived starter",
            body="Archived task.",
            labels=["type:chore", "good first issue"],
            good_first=True,
            status="archived",
        ),
    ]

    actions = seeder.apply_seeds(
        "owner/repo",
        seeds,
        dry_run=True,
        good_first_only=True,
        once=False,
        roadmap_path=roadmap,
    )

    assert actions == [
        "SKIP F302: archived seed is not active in ROADMAP.md",
        "DRY F300: gh issue create --repo owner/repo --title F300: starter task --body Starter task. --label type:chore --label good first issue",
    ]


def test_cli_dry_run_emits_json(monkeypatch, capsys):
    seeder = load_seeder()

    rc = seeder.main(["--dry-run", "--json"])

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert rc == 0
    assert payload["dry_run"] is True
    assert payload["repo"]
    assert len(payload["seeds"]) >= 14
    assert all(item.startswith("SKIP ") for item in payload["seeds"])
