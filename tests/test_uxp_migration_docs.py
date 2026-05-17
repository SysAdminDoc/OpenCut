"""Guardrails for the CEP to UXP migration plan."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_MIGRATION = REPO_ROOT / "docs" / "UXP_MIGRATION.md"


def test_f266_documents_cep_residual_and_drop_qe_plan():
    text = UXP_MIGRATION.read_text(encoding="utf-8")

    assert "F266" in text
    assert "ocAddNativeCaptionTrack" in text
    assert "ocQeReflect" in text
    assert "No new user-facing feature should depend on QE reflection." in text
    assert "Hybrid Plugin scope should prioritize native caption-track creation" in text
