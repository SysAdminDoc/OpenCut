"""Plugins declare a host-API range; the host says what it implements.

`api_version == 1` was a hard equality check, so the first host API bump would
have rejected every installed plugin with "unsupported value 1" and given the
author no way to say "I work on 1 and 2". These tests pin the versioned
contract, the actionable refusal, the doctor command, and — importantly — that
an existing v1 manifest still loads untouched.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from opencut.core import plugin_manifest as pm

V1_MANIFEST = {
    "name": "timecode-watermark",
    "version": "1.0.0",
    "description": "Legacy plugin from before the compatibility contract",
    "api_version": 1,
}


def _write_plugin(root: Path, name: str, manifest: dict) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / pm.MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (directory / "routes.py").write_text(
        "from flask import Blueprint\n\nplugin_bp = Blueprint('p', __name__)\n",
        encoding="utf-8",
    )
    return directory


class TestHostContract(unittest.TestCase):
    def test_host_publishes_an_api_range(self):
        low, high = pm.host_api_range()
        self.assertLessEqual(low, high)
        self.assertEqual(high, pm.PLUGIN_API_VERSION)
        self.assertEqual(low, pm.MIN_SUPPORTED_PLUGIN_API)

    def test_manifest_schema_version_is_separate_from_the_api_version(self):
        # They move independently: a manifest can gain fields without the
        # runtime contract changing.
        self.assertIsInstance(pm.MANIFEST_SCHEMA_VERSION, int)
        self.assertIsInstance(pm.PLUGIN_API_VERSION, int)


class TestCompatibilityChecks(unittest.TestCase):
    def test_v1_manifest_is_read_as_a_single_point_range(self):
        report = pm.check_api_compatibility(V1_MANIFEST)
        self.assertTrue(report.compatible, report.reason)
        self.assertEqual(report.plugin_min_api_version, 1)
        self.assertEqual(report.plugin_max_api_version, 1)

    def test_a_declared_range_spanning_the_host_is_accepted(self):
        report = pm.check_api_compatibility(
            {**V1_MANIFEST, "min_api_version": 1, "max_api_version": 3}
        )
        self.assertTrue(report.compatible, report.reason)

    def test_a_plugin_that_needs_a_newer_host_is_refused_with_a_fix(self):
        report = pm.check_api_compatibility(
            {**V1_MANIFEST, "api_version": pm.PLUGIN_API_VERSION + 5}
        )
        self.assertFalse(report.compatible)
        self.assertIn("implements up to", report.reason)
        self.assertIn("Upgrade OpenCut", report.remediation)

    def test_a_plugin_too_old_for_the_host_is_refused_with_a_fix(self):
        low = pm.MIN_SUPPORTED_PLUGIN_API
        report = pm.check_api_compatibility(
            {**V1_MANIFEST, "api_version": low - 1, "max_api_version": low - 1}
        )
        self.assertFalse(report.compatible)
        self.assertTrue(report.remediation)

    def test_a_newer_manifest_schema_is_refused_rather_than_guessed(self):
        report = pm.check_api_compatibility(
            {**V1_MANIFEST, "schema_version": pm.MANIFEST_SCHEMA_VERSION + 1}
        )
        self.assertFalse(report.compatible)
        self.assertIn("newer than this host", report.reason)

    def test_an_inverted_range_is_rejected(self):
        report = pm.check_api_compatibility(
            {**V1_MANIFEST, "min_api_version": 3, "max_api_version": 1}
        )
        self.assertFalse(report.compatible)

    def test_booleans_are_not_accepted_as_versions(self):
        # `True == 1` in Python; a manifest must not sneak through on that.
        report = pm.check_api_compatibility({**V1_MANIFEST, "api_version": True})
        self.assertFalse(report.compatible)

    def test_report_serialises(self):
        payload = pm.check_api_compatibility(V1_MANIFEST).as_dict()
        self.assertIn("compatible", payload)
        json.dumps(payload)


class TestSchemaValidationCarriesRemediation(unittest.TestCase):
    def test_incompatible_manifest_fails_schema_validation_with_a_fix(self):
        result = pm.validate_manifest_schema(
            {**V1_MANIFEST, "api_version": pm.PLUGIN_API_VERSION + 5}
        )
        self.assertFalse(result.valid)
        joined = " ".join(result.errors)
        self.assertIn("api_version", joined)
        self.assertIn("Upgrade OpenCut", joined)

    def test_missing_api_version_still_names_the_host_generation(self):
        manifest = {k: v for k, v in V1_MANIFEST.items() if k != "api_version"}
        result = pm.validate_manifest_schema(manifest)
        self.assertFalse(result.valid)
        self.assertIn(str(pm.PLUGIN_API_VERSION), " ".join(result.errors))

    def test_existing_v1_manifest_still_validates(self):
        self.assertTrue(pm.validate_manifest_schema(V1_MANIFEST).valid)


class TestDoctor(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_directory_is_not_an_error(self):
        report = pm.doctor(self.root / "nope")
        self.assertEqual(report["total"], 0)
        self.assertEqual(report["plugins"], [])

    def test_reports_the_host_contract(self):
        report = pm.doctor(self.root)
        self.assertEqual(report["host_api_version"], pm.PLUGIN_API_VERSION)
        self.assertEqual(
            report["manifest_schema_version"], pm.MANIFEST_SCHEMA_VERSION
        )

    def test_incompatible_plugin_is_counted_and_explained(self):
        _write_plugin(
            self.root, "future-plugin",
            {**V1_MANIFEST, "name": "future-plugin",
             "api_version": pm.PLUGIN_API_VERSION + 4},
        )
        report = pm.doctor(self.root)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["incompatible"], 1)
        entry = report["plugins"][0]
        self.assertFalse(entry["compatible"])
        self.assertTrue(entry["reason"])
        self.assertTrue(entry["remediation"])

    def test_unreadable_manifest_is_reported_not_raised(self):
        directory = self.root / "broken"
        directory.mkdir()
        (directory / pm.MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        report = pm.doctor(self.root)
        self.assertEqual(report["invalid"], 1)
        self.assertTrue(report["plugins"][0]["errors"])

    def test_report_is_json_serialisable(self):
        _write_plugin(self.root, "plain", V1_MANIFEST)
        json.dumps(pm.doctor(self.root))


class TestDoctorSurfaces(unittest.TestCase):
    def test_route_is_registered_and_returns_the_report(self):
        from opencut.server import create_app

        client = create_app().test_client()
        body = client.get("/plugins/doctor").get_json()
        self.assertEqual(body["host_api_version"], pm.PLUGIN_API_VERSION)
        self.assertIn("plugins", body)

    def test_cli_exposes_a_doctor_command(self):
        from click.testing import CliRunner

        from opencut.cli import cli

        result = CliRunner().invoke(cli, ["plugins", "doctor", "--json"])
        self.assertIn(result.exit_code, (0, 1), result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["host_api_version"], pm.PLUGIN_API_VERSION)


class TestDocumentedMigrationPath(unittest.TestCase):
    """The upgrade story has to be written down, not inferred."""

    @classmethod
    def setUpClass(cls):
        cls.source = Path(pm.__file__).read_text(encoding="utf-8")

    def test_module_documents_why_the_equality_check_was_replaced(self):
        self.assertIn("hard equality check", self.source)

    def test_v1_default_is_documented_as_a_single_point_range(self):
        self.assertIn("single-point", self.source)


if __name__ == "__main__":
    unittest.main()
