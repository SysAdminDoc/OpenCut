"""Standards labels must be earned from an independent validator.

OpenCut calls its IMSC 1.3 output "validated" and reports loudness against
ITU-R BS.1770. Until now the only thing checking either was OpenCut, and that
self-assessment hid a real defect: the default caption style emitted
`rgba(0,0,0,0.8)` — a CSS float alpha where TTML requires an integer 0-255 —
so the W3C reference implementation discarded the property on every document
OpenCut labelled conformant.

These tests run the reference implementations, pin that regression, and
require that an *absent* validator reports "not checked" rather than "passed".
"""
from __future__ import annotations

import logging
import unittest

import pytest

from opencut.core import standards_validators as sv
from opencut.core.caption_interchange import (
    document_from_items,
    serialize_caption_document,
)

imsc_required = pytest.mark.skipif(
    not sv.check_imsc_validator_available(),
    reason="ttconv/imschrm not installed (pip install 'opencut[standards]')",
)
ffmpeg_required = pytest.mark.skipif(
    not sv.check_loudness_validator_available(), reason="FFmpeg not installed"
)


def _imsc(items, language="en") -> bytes:
    return serialize_caption_document(
        document_from_items(items, language=language), "imsc1.3"
    )


SIMPLE = [
    {"start": 0.5, "end": 2.0, "text": "Hello world"},
    {"start": 2.5, "end": 4.0, "text": "Second cue\nwith a break"},
]


# ---------------------------------------------------------------------------
# Honesty of the status report
# ---------------------------------------------------------------------------
class TestValidatorStatus(unittest.TestCase):
    def test_status_names_every_validator_and_its_install_hint(self):
        status = sv.validator_status()
        self.assertEqual(set(status), {"imsc", "imf", "loudness"})
        for name, entry in status.items():
            with self.subTest(validator=name):
                self.assertIn("available", entry)
                self.assertTrue(entry["install_hint"])

    def test_an_absent_validator_reports_not_checked_not_passed(self):
        """`passed=None` is the whole point: absence must not read as success."""
        report = sv.validate_imf_package("/definitely/not/a/package")
        if not sv.check_imf_validator_available():
            self.assertFalse(report.available)
            self.assertIsNone(report.passed)
            self.assertTrue(report.notes)

    def test_report_is_jsonify_friendly(self):
        report = sv.ValidationReport(validator="imsc", available=True, passed=True)
        self.assertEqual(report["validator"], "imsc")
        self.assertIn("passed", report)
        self.assertEqual(report.to_dict()["passed"], True)


# ---------------------------------------------------------------------------
# IMSC 1.3 against the W3C reference implementation
# ---------------------------------------------------------------------------
@imsc_required
class TestImscConformance(unittest.TestCase):
    def test_log_capture_ignores_unrelated_records_and_deduplicates(self):
        from unittest import mock

        import ttconv.imsc.reader as imsc_reader

        original_to_model = imsc_reader.to_model

        def noisy_to_model(tree):
            logging.getLogger("opencut").error("unrelated validator failure")
            ttconv_logger = logging.getLogger("ttconv")
            ttconv_logger.warning("synthetic ttconv warning")
            ttconv_logger.warning("synthetic ttconv warning")
            return original_to_model(tree)

        with mock.patch.object(imsc_reader, "to_model", side_effect=noisy_to_model):
            report = sv.validate_imsc(_imsc(SIMPLE))

        self.assertTrue(report.passed, report.errors)
        self.assertNotIn("unrelated validator failure", report.errors)
        self.assertNotIn("unrelated validator failure", report.warnings)
        self.assertEqual(report.warnings.count("synthetic ttconv warning"), 1)

    def test_generated_imsc13_passes_the_reference_implementation(self):
        report = sv.validate_imsc(_imsc(SIMPLE), target="imsc1.3")
        self.assertTrue(report.available)
        self.assertTrue(report.passed, f"{report.errors}\n{report.warnings}")

    def test_generated_imsc13_passes_the_hrm(self):
        report = sv.validate_imsc(_imsc(SIMPLE))
        hrm_findings = [e for e in report.errors if e.startswith("HRM:")]
        self.assertEqual(hrm_findings, [])
        self.assertGreater(report.measurements["significant_times"], 0)

    def test_css_float_alpha_regression_is_caught(self):
        """The exact defect this validator was added to find."""
        broken = _imsc(SIMPLE).replace(b"rgba(0,0,0,204)", b"rgba(0,0,0,0.8)")
        self.assertIn(b"0.8", broken, "the default style no longer sets a background")
        report = sv.validate_imsc(broken, target="regression")
        self.assertFalse(report.passed)
        self.assertTrue(
            any("BackgroundColor" in error for error in report.errors), report.errors
        )

    def test_default_style_uses_an_integer_alpha(self):
        from opencut.core.caption_interchange import default_styles

        background = default_styles()["default"].properties["backgroundColor"]
        self.assertNotIn(".", background.split("(")[-1])

    def test_rtl_and_cjk_documents_pass(self):
        for language, text in (
            ("ar", "مرحبا بالعالم"),
            ("ja", "テスト字幕"),
            ("he", "שלום עולם"),
        ):
            with self.subTest(language=language):
                report = sv.validate_imsc(
                    _imsc([{"start": 0.0, "end": 2.0, "text": text}], language=language)
                )
                self.assertTrue(report.passed, report.errors)

    def test_malformed_xml_fails_closed(self):
        report = sv.validate_imsc(b"<tt><not-closed>")
        self.assertFalse(report.passed)
        self.assertTrue(report.errors)

    def test_entity_declarations_are_refused_before_parsing(self):
        hostile = (
            b'<?xml version="1.0"?><!DOCTYPE tt [<!ENTITY x "boom">]>'
            b'<tt xmlns="http://www.w3.org/ns/ttml"/>'
        )
        report = sv.validate_imsc(hostile)
        self.assertFalse(report.passed)
        self.assertTrue(
            any("entity" in error.lower() for error in report.errors), report.errors
        )

    def test_report_names_the_validator_versions(self):
        report = sv.validate_imsc(_imsc(SIMPLE))
        self.assertIn("ttconv", report.version)
        self.assertIn("imschrm", report.version)


# ---------------------------------------------------------------------------
# Loudness against the standard's own filter definition
# ---------------------------------------------------------------------------
class TestLoudnessConformance(unittest.TestCase):
    def test_k_weighting_gain_comes_from_the_bs1770_coefficients(self):
        # BS.1770-4's K-weighting is +0.6977 dB at 1 kHz / 48 kHz. A value
        # fitted to OpenCut's own measurement would not match this.
        self.assertAlmostEqual(sv.k_weighting_gain_db(1000.0, 48000), 0.6977, places=3)

    def test_expected_loudness_tracks_the_signal_level(self):
        self.assertAlmostEqual(
            sv.expected_tone_lufs(-23.0) - sv.expected_tone_lufs(-18.0), -5.0, places=3
        )

    @ffmpeg_required
    def test_measurement_matches_a_known_loudness_signal(self):
        for rms in (-23.0, -18.0):
            with self.subTest(rms_dbfs=rms):
                report = sv.validate_loudness_measurement(rms_dbfs=rms)
                self.assertTrue(report.available)
                self.assertTrue(report.passed, report.errors)
                self.assertLessEqual(
                    abs(report.measurements["delta_lu"]), sv.LOUDNESS_TOLERANCE_LU
                )

    @ffmpeg_required
    def test_a_wrong_target_is_detected(self):
        """The check must be capable of failing, not just of passing."""
        report = sv.validate_loudness_measurement(rms_dbfs=-23.0, tolerance_lu=0.0001)
        self.assertFalse(report.passed)
        self.assertTrue(report.errors)


# ---------------------------------------------------------------------------
# IMF / Photon
# ---------------------------------------------------------------------------
class TestImfValidatorAdapter(unittest.TestCase):
    def test_missing_jar_is_reported_not_guessed(self, ):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {sv.PHOTON_JAR_ENV: ""}, clear=False):
            self.assertFalse(sv.check_imf_validator_available())
            report = sv.validate_imf_package(".")
            self.assertFalse(report.available)
            self.assertIsNone(report.passed)
            self.assertTrue(any(sv.PHOTON_JAR_ENV in note for note in report.notes))

    def test_install_hint_names_the_real_artifact(self):
        self.assertIn("Photon", sv.INSTALL_HINT_IMF)
        self.assertIn(sv.PHOTON_JAR_ENV, sv.INSTALL_HINT_IMF)

    def test_clean_photon_summary_is_not_classified_as_an_error(self):
        from unittest import mock

        completed = mock.Mock(returncode=0, stdout=(
            "INFO IMPAnalyzer completed\n"
            "CPL_123.xml has no errors or warnings\n"
            "INFO reports/error-path.txt inspected\n"
        ), stderr="")
        with mock.patch.object(sv, "check_imf_validator_available", return_value=True), \
                mock.patch.object(sv, "photon_jar_path", return_value="photon-test.jar"), \
                mock.patch.object(sv.subprocess, "run", return_value=completed):
            report = sv.validate_imf_package(".")

        self.assertTrue(report.passed, report.errors)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_photon_severity_lines_are_reported(self):
        from unittest import mock

        completed = mock.Mock(returncode=1, stdout=(
            "ERROR Failed to read CPL\n"
            "FATAL IMPAnalyzer aborted\n"
            "WARNING Optional sidecar missing\n"
            "INFO error text in a diagnostic path\n"
        ), stderr="")
        with mock.patch.object(sv, "check_imf_validator_available", return_value=True), \
                mock.patch.object(sv, "photon_jar_path", return_value="photon-test.jar"), \
                mock.patch.object(sv.subprocess, "run", return_value=completed):
            report = sv.validate_imf_package(".")

        self.assertFalse(report.passed)
        self.assertEqual(report.errors, ["ERROR Failed to read CPL", "FATAL IMPAnalyzer aborted"])
        self.assertEqual(report.warnings, ["WARNING Optional sidecar missing"])


# ---------------------------------------------------------------------------
# Capability wording
# ---------------------------------------------------------------------------
class TestClaimWording(unittest.TestCase):
    """A "validated" label in the README must name who validated it."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8"
        )

    def test_imsc_claim_names_its_validator(self):
        claim_lines = [
            line for line in self.readme.splitlines()
            if "IMSC 1.3" in line and ("validat" in line or "conforman" in line)
        ]
        self.assertTrue(claim_lines, "README no longer makes an IMSC 1.3 claim")
        for line in claim_lines:
            with self.subTest(line=line[:70]):
                self.assertTrue(
                    "ttconv" in line or "reference implementation" in line,
                    "an IMSC conformance claim must name what validated it",
                )

    def test_readme_says_an_absent_validator_is_not_a_pass(self):
        self.assertIn("passed", self.readme)
        self.assertIn('"not checked" never reads as "passed"', self.readme)

    def test_readme_documents_the_standards_extra(self):
        self.assertIn("opencut[standards]", self.readme)


if __name__ == "__main__":
    unittest.main()
