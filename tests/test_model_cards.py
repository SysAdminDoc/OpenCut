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
from pathlib import Path

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


def test_caption_translation_engines_are_license_classified():
    """NLLB and SeamlessM4T must stay explicit opt-in restricted engines."""
    cards = {card.feature_id: card for card in model_cards.CARDS}
    required = {
        "captions.translate.nllb": "NLLB",
        "captions.translate.seamless-m4t": "SeamlessM4T",
    }
    missing = set(required) - set(cards)
    assert not missing, f"missing caption translation model cards: {sorted(missing)}"

    for feature_id, label in required.items():
        card = cards[feature_id]
        assert card.license.startswith("CC-BY-NC"), f"{label} license must stay restricted"
        assert card.license_waiver, f"{label} needs an explicit opt-in waiver"
        joined_notes = " ".join(card.advisory_notes).lower()
        assert "non-commercial" in joined_notes
        assert "never auto-selected" in joined_notes
        assert "accept_restricted_license" in joined_notes


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


def test_model_cards_do_not_reference_dead_roadmap_anchors():
    encoded = json.dumps(model_cards.manifest(), ensure_ascii=False)
    assert "roadmap H" not in encoded
    assert "Roadmap stub" not in encoded


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


# ---------------------------------------------------------------------------
# F177 — 2026-Q2 sweep gates
# ---------------------------------------------------------------------------


def test_each_category_has_at_least_one_card():
    """A category accidentally drained to zero rows is a regression signal —
    it means a refactor wiped a whole feature surface without filing an
    F-number. The current expected categories: audio, captions, editing,
    generation, lipsync, llm, video.
    """
    cats = {card.category for card in model_cards.CARDS}
    expected = {"audio", "captions", "editing", "generation", "lipsync", "llm", "video"}
    missing = expected - cats
    assert not missing, (
        f"F177 — these documented categories have no model cards left: "
        f"{sorted(missing)}. If this is intentional, update the expected "
        f"set in tests/test_model_cards.py with the F-number authorising "
        f"the removal."
    )


def test_each_license_starts_with_known_prefix():
    """License strings must start with a documented prefix so casual
    skim-reads catch new license families immediately. Free-text after
    the prefix is allowed for nuance (e.g. ``MIT (client only)``)."""
    # SPDX-friendly tokens + the in-house markers used for hosted
    # services and multi-license aggregators.
    known_prefixes = (
        "Apache-2.0", "Apache 2.0",
        "MIT",
        "BSD-3-Clause", "BSD-2-Clause", "BSD",
        "GPL-3.0", "AGPL-3.0",
        "LGPL-3.0",
        "Microsoft Edge TTS terms",
        "Commercial API",
        "CC-BY",                         # covers CC-BY-NC, CC-BY-4.0, etc.
        "OpenRAIL-M", "OpenRAIL",
        "Custom",
        "Tencent",                       # HunyuanVideo-family carve-outs
        "Public Domain",
        "NTU S-Lab License",             # ProPainter, MatAnyone
        "Unlicense",                     # auto-editor (public domain)
        "proprietary",                   # ElevenLabs, Hailuo, Seedance, Edge TTS
        "varies per backend",            # multi-backend LLM aggregators
        "varies per model",              # HF model hub passthroughs
        "MIT (orchestration",            # OpenCut wrappers over commercial APIs
    )
    offenders: list = []
    for card in model_cards.CARDS:
        if not card.license.startswith(known_prefixes):
            offenders.append(f"{card.label!r}: license={card.license!r}")
    assert not offenders, (
        "F177 — model cards carry unknown license prefixes. Add the "
        "prefix to known_prefixes in tests/test_model_cards.py after "
        "verifying the licence is OK to ship under:\n  - "
        + "\n  - ".join(offenders)
    )


def test_each_privacy_token_starts_with_known_prefix():
    """Privacy posture must begin with one of: 'local-only', 'local +
    optional cloud', 'cloud'. Free-text qualifier after the prefix is
    allowed for nuance (e.g. ``cloud — text is sent to ElevenLabs``)."""
    prefixes = ("local-only", "local + optional cloud", "cloud")
    offenders = [
        f"{card.label!r}: privacy={card.privacy!r}"
        for card in model_cards.CARDS
        if not card.privacy.startswith(prefixes)
    ]
    assert not offenders, (
        "F177 — model cards carry unknown privacy prefixes. Privacy must "
        "begin with 'local-only', 'local + optional cloud', or 'cloud':\n  - "
        + "\n  - ".join(offenders)
    )


def test_each_hardware_field_starts_with_known_shape():
    """Hardware must start with one of the documented shapes so the
    panel can render a consistent chip. Free-text qualifier in
    parentheses is allowed for nuance (e.g. ``gpu (recommended)``,
    ``cpu (client)``, ``gpu (>= 16 GB VRAM for VRT)``)."""
    prefixes = ("cpu", "gpu", "cpu/gpu", "cloud")
    offenders = [
        f"{card.label!r}: hardware={card.hardware!r}"
        for card in model_cards.CARDS
        if not card.hardware.startswith(prefixes)
    ]
    assert not offenders, (
        "F177 — model cards carry off-prefix hardware strings. Use "
        "'cpu', 'gpu', 'cpu/gpu', or 'cloud' as the leading token:\n  - "
        + "\n  - ".join(offenders)
    )


def test_card_count_meets_2026_q2_floor():
    """F177 sweep baseline — fewer than 40 cards means a regression dropped
    a major feature surface. Bump this number deliberately when a card is
    intentionally retired."""
    assert len(model_cards.CARDS) >= 40, (
        f"F177 — model card count dropped below the 2026-Q2 floor "
        f"(got {len(model_cards.CARDS)}, expected >= 40). If a feature "
        f"was intentionally retired, update the floor in "
        f"tests/test_model_cards.py."
    )


def test_no_two_feature_ids_are_duplicated():
    """Multiple cards can point at the same check_name (different
    backends), but `feature_id` should be unique per card so the
    registry can dedupe."""
    seen: dict = {}
    duplicates: list = []
    for card in model_cards.CARDS:
        if card.feature_id in seen:
            duplicates.append(
                f"{card.label!r} reuses feature_id={card.feature_id!r} (also: {seen[card.feature_id]!r})"
            )
        else:
            seen[card.feature_id] = card.label
    assert not duplicates, (
        "F177 — feature_id collisions across model cards:\n  - "
        + "\n  - ".join(duplicates)
    )
