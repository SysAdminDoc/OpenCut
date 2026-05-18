"""F180 - Wave N-T planning ledger coverage.

The model-surface wave plan and the F-number governance backlog intentionally
coexist. This test keeps the F180 bridge document in sync with every Wave N-T
row in ROADMAP.md so new wave rows cannot bypass the governance disposition
rules.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "ROADMAP.md"
LEDGER = REPO_ROOT / ".ai" / "research" / "2026-05-18" / "WAVE_N_T_F_NUMBER_LEDGER.md"

WAVE_ROW_RE = re.compile(r"^\| ([N-T]\d\.\d) \|", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_wave_n_t_f_number_ledger_covers_every_roadmap_wave_item():
    roadmap_ids = set(WAVE_ROW_RE.findall(_read(ROADMAP)))
    ledger_ids = set(WAVE_ROW_RE.findall(_read(LEDGER)))

    assert {"N1.1", "R1.1", "T3.6"}.issubset(roadmap_ids)
    assert roadmap_ids - ledger_ids == set()
    assert ledger_ids - roadmap_ids == set()


def test_roadmap_marks_f180_closed_and_links_the_ledger():
    text = _read(ROADMAP)
    assert "[x] F180" in text
    assert "WAVE_N_T_F_NUMBER_LEDGER.md" in text

