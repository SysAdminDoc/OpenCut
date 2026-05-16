"""Tests for the source-linked GitHub issue seeder (F097)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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

    assert len(seeds) >= 14, "should seed every Now/Next tier roadmap item"
    ids = {seed.roadmap_id for seed in seeds}
    assert {"F097", "F098", "F099", "F100", "F111", "F112", "F115", "F117", "F118"}.issubset(ids)

    for seed in seeds:
        assert seed.title.startswith(f"{seed.roadmap_id}"), seed.title
        assert seed.body.strip(), f"empty body for {seed.roadmap_id}"
        assert seed.labels, f"no labels for {seed.roadmap_id}"


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
    assert all(action.startswith("DRY ") for action in actions)


def test_good_first_filter_picks_labeled_seeds():
    seeder = load_seeder()
    seeds = seeder.load_seeds()

    actions = seeder.apply_seeds(
        "owner/repo",
        seeds,
        dry_run=True,
        good_first_only=True,
        once=False,
    )

    assert actions, "F097 manifest should expose at least one good-first seed (F117)"
    for line in actions:
        assert "good first issue" in line.lower() or "F117" in line


def test_cli_dry_run_emits_json(monkeypatch, capsys):
    seeder = load_seeder()

    rc = seeder.main(["--dry-run", "--json"])

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert rc == 0
    assert payload["dry_run"] is True
    assert payload["repo"]
    assert len(payload["seeds"]) >= 14
