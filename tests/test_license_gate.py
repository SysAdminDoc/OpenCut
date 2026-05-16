"""Tests for the model-card / dependency license gate (F006)."""

from __future__ import annotations

import json

import pytest

from opencut.tools import license_gate as lg


# ----- classify_license ---------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MIT", "allowed"),
        ("Apache-2.0", "allowed"),
        ("BSD-3-Clause", "allowed"),
        ("ISC", "allowed"),
        ("Unlicense (public domain)", "allowed"),
        ("LGPL-3.0", "warning"),
        ("MPL-2.0", "allowed"),
        ("GPL-3.0", "denied"),
        ("CC-BY-NC", "denied"),
        ("research-only", "denied"),
        ("NTU S-Lab License 1.0", "denied"),
        ("proprietary cloud service", "warning"),
        ("varies per backend", "warning"),
        ("totally made up", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_license_buckets(text, expected):
    assert lg.classify_license(text) == expected


def test_classify_license_word_boundaries_do_not_collide():
    # The substring 'GPL-3.0' MUST NOT match LGPL-3.0.
    assert lg.classify_license("LGPL-3.0") == "warning"


# ----- lint() -------------------------------------------------------------


def test_lint_does_not_fail_on_current_repo():
    report = lg.lint()
    assert not report.has_errors(), (
        "license gate must pass on the committed model cards (every denied "
        "license has a waiver). Findings:\n  - "
        + "\n  - ".join(f"{f.surface}/{f.name}: {f.license_text} ({f.severity})" for f in report.findings if f.severity != "info")
    )


def test_lint_emits_findings_per_card():
    report = lg.lint()
    model_card_findings = [f for f in report.findings if f.surface == "model_card"]
    assert len(model_card_findings) > 30  # >40 cards today


# ----- CLI ----------------------------------------------------------------


def test_cli_returns_zero_with_warnings_only():
    rc = lg.cli([])
    assert rc == 0


def test_cli_returns_zero_with_strict_warnings_only_when_no_warnings(monkeypatch):
    monkeypatch.setattr(lg, "lint", lambda root=lg.REPO_ROOT: lg.LicenseReport())
    rc = lg.cli(["--strict-warnings"])
    assert rc == 0


def test_cli_returns_nonzero_with_strict_warnings_and_findings():
    rc = lg.cli(["--strict-warnings"])
    assert rc == 1


def test_cli_emits_json_payload(capsys):
    rc = lg.cli(["--json"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert rc == 0
    assert "findings" in payload
    assert "surfaces" in payload


# ----- waiver path --------------------------------------------------------


def test_denied_license_with_waiver_becomes_warning():
    """The committed cards already exercise the waiver path — assert
    that no waived rows surface as errors."""
    report = lg.lint()
    waived = [f for f in report.findings if "waived" in f.message]
    assert waived, "expected at least one waived license finding in the committed cards"
    assert all(f.severity == "warning" for f in waived)


def test_denied_license_without_waiver_is_error(monkeypatch):
    from opencut.model_cards import ModelCard

    fake_card = ModelCard(
        check_name="check_x_available",
        feature_id="demo",
        label="Demo",
        category="video",
        license="CC-BY-NC",
        upstream="https://example.com",
        hardware="cpu",
        install_hint="pip install demo",
        privacy="local-only",
    )

    monkeypatch.setattr("opencut.model_cards.CARDS", [fake_card])
    report = lg.lint()
    errors = [f for f in report.findings if f.severity == "error"]
    assert errors and "denied list" in errors[0].message
