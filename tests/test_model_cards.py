"""Model card invariants (F115).

* Every ``check_*_available`` function in ``opencut.checks`` either has
  a model card or is on ``NON_AI_CHECKS``. New optional dependencies
  must be triaged before they land.
* The committed JSON/Markdown artefacts stay in sync with the cards
  list — the ``--check`` mode of ``opencut.tools.dump_model_cards`` is
  the canonical enforcement.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from opencut import checks
from opencut import model_cards

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "opencut" / "_generated" / "model_cards.json"
DOCS_PATH = REPO_ROOT / "docs" / "MODELS.md"


def _all_check_names() -> list:
    """Return every ``check_*_available`` callable defined in opencut.checks."""
    out = []
    for name, value in inspect.getmembers(checks):
        if not name.startswith("check_") or not name.endswith("_available"):
            continue
        if not callable(value):
            continue
        out.append(name)
    return sorted(out)


def test_every_check_has_card_or_is_excluded():
    """Stops new opt-in deps from sneaking in without metadata triage."""
    cards = model_cards.cards_by_check_name()
    excluded = set(model_cards.NON_AI_CHECKS)
    all_checks = set(_all_check_names())

    missing = [name for name in sorted(all_checks) if name not in cards and name not in excluded]

    assert not missing, (
        "These check_*_available functions are missing both a model card "
        "AND an entry on opencut.model_cards.NON_AI_CHECKS:\n  - "
        + "\n  - ".join(missing)
        + "\nAdd one or the other before merging."
    )


def test_no_card_has_a_stale_check_name():
    all_checks = set(_all_check_names())
    stale = [
        card.check_name
        for card in model_cards.CARDS
        if card.check_name not in all_checks
    ]
    assert not stale, f"model card references missing check function(s): {stale}"


def test_card_invariants_hold():
    model_cards.assert_card_invariants()


def test_manifest_matches_committed_json():
    payload = model_cards.manifest()
    committed = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert committed == payload, (
        "opencut/_generated/model_cards.json is out of sync with "
        "opencut/model_cards.py. Run `python -m opencut.tools.dump_model_cards`."
    )


def test_markdown_in_sync_with_cards():
    """Cheap structural assertions — exact text drift is caught by --check."""
    md = DOCS_PATH.read_text(encoding="utf-8")
    for card in model_cards.CARDS:
        assert card.label in md, f"docs/MODELS.md missing card {card.label!r}"
    # Every license string must appear at least once.
    licenses = {card.license for card in model_cards.CARDS}
    for lic in licenses:
        assert lic in md, f"license token {lic!r} missing from generated markdown"


def test_non_ai_checks_are_real_checks():
    """Keep the allowlist honest — only list checks that actually exist."""
    all_checks = set(_all_check_names())
    bogus = [name for name in model_cards.NON_AI_CHECKS if name not in all_checks]
    assert not bogus, (
        "NON_AI_CHECKS lists names that don't exist in opencut.checks: "
        f"{bogus}"
    )


def test_dumper_check_mode_passes_against_committed_artefacts():
    """`python -m opencut.tools.dump_model_cards --check` must return 0."""
    from opencut.tools import dump_model_cards

    rc = dump_model_cards.cli(["--check"])
    assert rc == 0
