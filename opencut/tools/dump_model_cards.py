"""Render model cards to JSON + Markdown (F115).

Reads :mod:`opencut.model_cards.CARDS` and writes two deterministic
artefacts:

* ``opencut/_generated/model_cards.json``
* ``docs/MODELS.md`` (Markdown table grouped by category)

Both files are checked into git so docs site previews and security
review tooling can consume them without booting the Python package.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "opencut" / "_generated" / "model_cards.json"
DOCS_PATH = REPO_ROOT / "docs" / "MODELS.md"


def _render_markdown(manifest: dict) -> str:
    cards = manifest["cards"]
    cards_by_cat = defaultdict(list)
    for card in cards:
        cards_by_cat[card["category"]].append(card)

    lines = []
    lines.append("# OpenCut model + license cards")
    lines.append("")
    lines.append(
        "Generated from `opencut/model_cards.py`. **Do not hand-edit** — "
        "regenerate with `python -m opencut.tools.dump_model_cards`."
    )
    lines.append("")
    lines.append(
        f"Total optional AI/model surfaces: **{manifest['total']}**. "
        "Each row carries license, hardware, install hint, privacy posture, "
        "and (where relevant) an advisory note. Backends not listed here "
        "are infrastructure guards on the explicit `NON_AI_CHECKS` "
        "allowlist."
    )
    lines.append("")

    for category in sorted(cards_by_cat):
        lines.append(f"## {category.title()}")
        lines.append("")
        lines.append("| Backend | License | Hardware | Privacy | Install |")
        lines.append("|---|---|---|---|---|")
        for card in sorted(cards_by_cat[category], key=lambda c: c["label"].lower()):
            label = f"[{card['label']}]({card['upstream']})"
            lines.append(
                f"| {label} | {card['license']} | {card['hardware']} | "
                f"{card['privacy']} | `{card['install_hint']}` |"
            )
        # Advisory notes appear as a sub-list per-category.
        advisories = [
            (card["label"], card["advisory_notes"])
            for card in cards_by_cat[category]
            if card["advisory_notes"]
        ]
        if advisories:
            lines.append("")
            lines.append("**Advisory notes**:")
            for label, notes in advisories:
                for note in notes:
                    lines.append(f"- *{label}* — {note}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Excluded infrastructure checks")
    lines.append("")
    lines.append(
        "These `opencut.checks` functions do not gate a model/AI "
        "dependency and are deliberately excluded from this surface:"
    )
    lines.append("")
    for name in manifest["non_ai_checks"]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_artifacts(manifest: dict) -> List[Path]:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    DOCS_PATH.write_text(_render_markdown(manifest), encoding="utf-8")
    return [JSON_PATH, DOCS_PATH]


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed artefacts differ from the current cards",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from opencut.model_cards import assert_card_invariants, manifest

    assert_card_invariants()
    payload = manifest()

    if args.check:
        existing_json = JSON_PATH.read_text(encoding="utf-8") if JSON_PATH.exists() else ""
        existing_md = DOCS_PATH.read_text(encoding="utf-8") if DOCS_PATH.exists() else ""
        live_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        live_md = _render_markdown(payload)
        if existing_json != live_json or existing_md != live_md:
            print(
                "[model-cards] FAIL — committed artefacts out of sync with "
                "opencut/model_cards.py. Re-run without --check to regenerate."
            )
            return 1
        print(f"[model-cards] OK — {payload['total']} cards across {len(set(c['category'] for c in payload['cards']))} categories")
        return 0

    paths = write_artifacts(payload)
    rels = ", ".join(str(p.relative_to(REPO_ROOT)) for p in paths)
    print(f"[model-cards] wrote {rels} — {payload['total']} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
