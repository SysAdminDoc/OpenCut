"""F327 — say when a dependency's upstream has stopped.

A package can install and import perfectly while being abandoned. OpenCut
pip-installed DeepFilterNet on demand years after its last release, and nothing
in the product could tell a user that. Availability and maintenance are separate
questions and are now reported separately.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from opencut.core.engine_registry import get_registry  # noqa: E402
from opencut.dependency_support import (  # noqa: E402
    ABANDONED_DEPENDENCIES,
    maintenance_status,
)


class TestMaintenanceRecord:
    def test_known_abandoned_package_is_reported(self):
        status = maintenance_status("deepfilternet")
        assert status["abandoned"] is True
        assert "unmaintained" in status["warning"]
        assert "2023-08-31" in status["warning"]

    def test_the_warning_names_a_maintained_alternative(self):
        """Telling someone their tool is dead without a next step is not help."""
        for name in ABANDONED_DEPENDENCIES:
            status = maintenance_status(name)
            assert status.get("alternative"), f"{name} records no alternative"
            assert status["alternative"] in status["warning"]

    def test_a_version_specifier_still_resolves(self):
        assert maintenance_status("deepfilternet>=0.5.6")["abandoned"] is True
        assert maintenance_status("demucs[dev]")["abandoned"] is True

    def test_an_unrecorded_package_is_unknown_not_healthy(self):
        status = maintenance_status("faster-whisper")
        assert status["abandoned"] is False
        assert status["warning"] == ""

    def test_every_record_is_re_verifiable(self):
        """Each claim must be checkable at a URL, not asserted."""
        for name, record in ABANDONED_DEPENDENCIES.items():
            assert record["upstream"].startswith("http"), name
            assert record["state"] in ("unmaintained", "archived"), name
            for field in ("last_release", "last_activity", "checked"):
                assert len(record[field]) == 10 and record[field][4] == "-", (
                    f"{name}.{field} is not an ISO date: {record[field]!r}"
                )


class TestRegistryMaintenanceFields:
    def test_an_archived_engine_is_marked(self):
        demucs = get_registry().get_engine("stem_separation", "demucs")
        assert demucs is not None
        assert demucs.is_unmaintained is True
        assert "archived" in demucs.maintenance_note

    def test_an_engine_without_a_record_is_not_claimed_healthy(self):
        engine = get_registry().get_engine("stem_separation", "mel_band_roformer")
        assert engine.is_unmaintained is False
        assert engine.maintenance_note == ""

    def test_no_unmaintained_engine_is_the_top_choice_in_its_domain(self):
        """The whole point: dead software must never be what runs by default."""
        registry = get_registry()
        offenders = []
        for domain in registry.get_all_domains():
            engines = list(registry.get_engines(domain))
            if not engines:
                continue
            top = max(engines, key=lambda e: e.priority)
            if top.is_unmaintained:
                offenders.append(f"{domain} -> {top.name}")
        assert not offenders, (
            f"unmaintained engines rank highest in their domain: {offenders}"
        )

    def test_a_marked_engine_records_when_it_was_checked(self):
        for domain in get_registry().get_all_domains():
            for engine in get_registry().get_engines(domain):
                if engine.is_unmaintained:
                    assert engine.maintenance_checked, (
                        f"{engine.name} is marked unmaintained with no check date"
                    )
                    assert engine.upstream.startswith("http"), engine.name


class TestInstallPathWarns:
    def test_installing_an_abandoned_package_reports_it(self, monkeypatch):
        """Installing dead software is allowed, but never silent."""
        import opencut.helpers as helpers

        messages = []
        monkeypatch.setattr(
            "opencut.security.safe_pip_install",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("blocked in test")),
        )

        ok = helpers.ensure_package(
            "a_module_that_does_not_exist_xyz",
            "deepfilternet",
            lambda pct, msg="": messages.append(msg),
        )

        assert ok is False
        assert any("unmaintained" in m for m in messages), messages

    def test_a_maintained_package_gets_no_maintenance_warning(self, monkeypatch):
        import opencut.helpers as helpers

        messages = []
        monkeypatch.setattr(
            "opencut.security.safe_pip_install",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("blocked in test")),
        )

        helpers.ensure_package(
            "another_missing_module_xyz",
            "librosa",
            lambda pct, msg="": messages.append(msg),
        )

        assert not any("unmaintained" in m for m in messages), messages


@pytest.mark.parametrize("field", ["maintenance", "maintenance_note", "maintenance_checked"])
def test_dependency_dashboard_exposes_maintenance(field):
    source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "opencut", "routes", "system_runtime_routes.py",
    )
    with open(source, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert field in text, f"/system/dependencies never reports {field}"
