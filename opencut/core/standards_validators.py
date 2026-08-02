"""Independent reference validators for standards-labelled output.

OpenCut labels IMF packages and IMSC 1.3 captions as conformant and reports
loudness against ITU-R BS.1770 targets. Those labels were self-assessed: the
same code that wrote the file decided the file was valid. That is not
evidence, and it hid a real defect — the default IMSC style emitted a CSS
float alpha (`rgba(0,0,0,0.8)`) where TTML requires an integer, so every
"validated IMSC 1.3" document was rejected by the W3C reference
implementation.

Each adapter here shells out to (or imports) an *independent* implementation
and returns a machine-readable :class:`ValidationReport`. When the validator
is absent the report says so — `available=False`, `passed=None` — so callers
downgrade the claim instead of silently treating "not checked" as "passed".

Validators
----------
``imsc``      ``ttconv`` (W3C reference implementation) parses the document and
              ``imschrm`` runs the IMSC Hypothetical Render Model. Install with
              ``pip install "opencut[standards]"``.
``imf``       Netflix Photon ``IMPAnalyzer``. Requires a JRE plus the Photon
              all-in-one jar; point ``OPENCUT_PHOTON_JAR`` at it.
``loudness``  Measures a signal whose true loudness is known by construction
              and compares against OpenCut's own BS.1770 measurement.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("opencut")

PHOTON_JAR_ENV = "OPENCUT_PHOTON_JAR"
INSTALL_HINT_IMSC = 'pip install "opencut[standards]"  (ttconv + imschrm)'
INSTALL_HINT_IMF = (
    "Install a JRE and set OPENCUT_PHOTON_JAR to the Netflix Photon "
    "all-in-one jar (photon-<version>-all.jar)."
)

#: EBU R 128 / ITU-R BS.1770 tolerance for a programme-loudness measurement.
LOUDNESS_TOLERANCE_LU = 0.5


@dataclass
class ValidationReport:
    """Machine-readable verdict from one reference validator."""

    validator: str
    available: bool = False
    passed: Optional[bool] = None      # None = not checked
    version: str = ""
    target: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    # Flask jsonify protocol, matching the rest of the codebase.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.keys()}


class _CollectingHandler(logging.Handler):
    """Capture a third-party validator's own log output as findings.

    ``ttconv`` reports malformed style properties through its logger rather
    than by raising, so a parse that "succeeded" can still have discarded
    invalid attributes. Without this, the IMSC adapter would report a pass on
    a document the reference implementation refused to read correctly.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            self.errors.append(message)
        else:
            self.warnings.append(message)


# ---------------------------------------------------------------------------
# IMSC 1.3
# ---------------------------------------------------------------------------
def check_imsc_validator_available() -> bool:
    try:
        import imschrm.hrm  # noqa: F401
        import ttconv.imsc.reader  # noqa: F401
    except ImportError:
        return False
    return True


def _imsc_validator_version() -> str:
    try:
        import importlib.metadata as metadata
        return (
            f"ttconv {metadata.version('ttconv')}, "
            f"imschrm {metadata.version('imschrm')}"
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def validate_imsc(source: str | bytes, *, target: str = "") -> ValidationReport:
    """Validate an IMSC/TTML document with the W3C reference implementation.

    Args:
        source: XML bytes, an XML string, or a path to a document.
        target: Label recorded in the report (defaults to the path).
    """
    report = ValidationReport(validator="imsc", target=target)
    if not check_imsc_validator_available():
        report.notes.append(INSTALL_HINT_IMSC)
        return report

    import xml.etree.ElementTree as ET

    import ttconv.imsc.reader as imsc_reader
    from imschrm.hrm import EventHandler, validate
    from ttconv.isd import ISD

    from opencut.core.caption_interchange import (
        CaptionInterchangeError,
        _read_xml_source,
    )

    report.available = True
    report.version = _imsc_validator_version()

    if isinstance(source, str) and os.path.isfile(source):
        report.target = report.target or source
    try:
        # Reuse the caption reader's screening: it rejects DTD and entity
        # declarations before anything is parsed, so a hostile document
        # cannot expand entities or reach the filesystem.
        payload = _read_xml_source(source)
    except CaptionInterchangeError as exc:
        report.errors.append(str(exc))
        report.passed = False
        return report

    collector = _CollectingHandler()
    ttconv_logger = logging.getLogger("ttconv")
    root_logger = logging.getLogger()
    ttconv_logger.addHandler(collector)
    root_logger.addHandler(collector)
    try:
        try:
            tree = ET.ElementTree(ET.fromstring(payload))
        except ET.ParseError as exc:
            report.errors.append(f"XML is not well-formed: {exc}")
            report.passed = False
            return report

        try:
            model = imsc_reader.to_model(tree)
        except Exception as exc:  # noqa: BLE001 - any reader failure is a finding
            report.errors.append(f"ttconv could not read the document: {exc}")
            report.passed = False
            return report

        if model is None:
            report.errors.append("ttconv produced no document model")
            report.passed = False
            return report

        class _Handler(EventHandler):
            def error(self, msg):  # noqa: D102
                report.errors.append(f"HRM: {msg}")

            def warn(self, msg):  # noqa: D102
                report.warnings.append(f"HRM: {msg}")

        try:
            times = sorted(ISD.significant_times(model))
            validate(((t, ISD.from_model(model, t)) for t in times), _Handler())
            report.measurements["significant_times"] = len(times)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"HRM validation failed: {exc}")
    finally:
        ttconv_logger.removeHandler(collector)
        root_logger.removeHandler(collector)

    # A reader diagnostic means an attribute was discarded — the document did
    # not round-trip, even though parsing "succeeded".
    report.errors.extend(collector.errors)
    report.warnings.extend(collector.warnings)
    report.passed = not report.errors
    return report


# ---------------------------------------------------------------------------
# IMF (Netflix Photon)
# ---------------------------------------------------------------------------
def photon_jar_path() -> str:
    return (os.environ.get(PHOTON_JAR_ENV) or "").strip()


def check_imf_validator_available() -> bool:
    jar = photon_jar_path()
    return bool(jar) and os.path.isfile(jar) and bool(shutil.which("java"))


def validate_imf_package(package_dir: str, *, timeout: int = 900) -> ValidationReport:
    """Run Netflix Photon's ``IMPAnalyzer`` over an IMP directory."""
    report = ValidationReport(validator="imf", target=package_dir)
    if not check_imf_validator_available():
        report.notes.append(INSTALL_HINT_IMF)
        if not shutil.which("java"):
            report.notes.append("java executable not found on PATH")
        elif not photon_jar_path():
            report.notes.append(f"{PHOTON_JAR_ENV} is not set")
        else:
            report.notes.append(f"{PHOTON_JAR_ENV} does not point at a file")
        return report

    if not os.path.isdir(package_dir):
        report.available = True
        report.errors.append(f"not a directory: {package_dir}")
        report.passed = False
        return report

    report.available = True
    report.version = os.path.basename(photon_jar_path())
    try:
        completed = subprocess.run(
            [
                "java", "-cp", photon_jar_path(),
                "com.netflix.imflibrary.app.IMPAnalyzer", package_dir,
            ],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.errors.append(f"Photon could not be run: {exc}")
        report.passed = False
        return report

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if "error" in lowered or "fatal" in lowered:
            report.errors.append(stripped)
        elif "warning" in lowered:
            report.warnings.append(stripped)
    report.measurements["exit_code"] = completed.returncode
    report.passed = completed.returncode == 0 and not report.errors
    return report


# ---------------------------------------------------------------------------
# Loudness (ITU-R BS.1770 / EBU R 128)
# ---------------------------------------------------------------------------
def check_loudness_validator_available() -> bool:
    from opencut.helpers import get_ffmpeg_path
    return bool(get_ffmpeg_path())


#: The fixed offset in the BS.1770 loudness equation.
BS1770_OFFSET_DB = -0.691

#: BS.1770-4 K-weighting, specified at 48 kHz: a high-shelf stage followed by
#: an RLB high-pass. These are the standard's own coefficients, so the expected
#: loudness below is derived from the specification rather than fitted to
#: whatever OpenCut happens to measure.
_K_STAGE1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_K_STAGE1_A = (1.0, -1.69065929318241, 0.73248077421585)
_K_STAGE2_B = (1.0, -2.0, 1.0)
_K_STAGE2_A = (1.0, -1.99004745483398, 0.99007225036621)


def _biquad_gain(b, a, frequency: float, sample_rate: int) -> float:
    import cmath
    import math

    z = cmath.exp(-2j * math.pi * frequency / sample_rate)
    numerator = sum(coefficient * z ** index for index, coefficient in enumerate(b))
    denominator = sum(coefficient * z ** index for index, coefficient in enumerate(a))
    return abs(numerator / denominator)


def k_weighting_gain_db(frequency: float = 1000.0, sample_rate: int = 48000) -> float:
    """K-weighting gain the standard applies at *frequency*."""
    import math

    gain = (
        _biquad_gain(_K_STAGE1_B, _K_STAGE1_A, frequency, sample_rate)
        * _biquad_gain(_K_STAGE2_B, _K_STAGE2_A, frequency, sample_rate)
    )
    return 20 * math.log10(gain)


def expected_tone_lufs(
    rms_dbfs: float, frequency: float = 1000.0, sample_rate: int = 48000
) -> float:
    """Loudness the standard specifies for a steady mono sine.

    A constant tone has nothing for the gate to remove, so programme loudness
    is just the K-weighted mean square plus the standard's fixed offset.
    """
    return round(
        rms_dbfs + k_weighting_gain_db(frequency, sample_rate) + BS1770_OFFSET_DB, 3
    )


def build_reference_tone(
    path: str,
    *,
    frequency: int = 1000,
    duration: float = 20.0,
    rms_dbfs: float = -23.0,
) -> str:
    """Write a mono sine whose BS.1770 loudness is known by construction.

    The amplitude is written into the sample expression rather than applied as
    a gain on top of a source whose own level is a convention (``lavfi``'s
    ``sine`` is not full scale), so the signal level is specified by this
    function rather than inherited from FFmpeg.
    """
    from opencut.helpers import get_ffmpeg_path

    # dBFS RMS -> peak amplitude for a sine (RMS = peak / sqrt(2)).
    amplitude = (10 ** (rms_dbfs / 20.0)) * (2 ** 0.5)
    expression = f"{amplitude:.9f}*sin(2*PI*{frequency}*t)"
    subprocess.run(
        [
            get_ffmpeg_path(), "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi",
            "-i", f"aevalsrc={expression}:s=48000:d={duration}",
            "-c:a", "pcm_s24le", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return path


def validate_loudness_measurement(
    *,
    rms_dbfs: float = -23.0,
    tolerance_lu: float = LOUDNESS_TOLERANCE_LU,
) -> ValidationReport:
    """Check OpenCut's loudness measurement against a known-loudness signal."""
    target_lufs = expected_tone_lufs(rms_dbfs)
    report = ValidationReport(validator="loudness", target=f"{target_lufs} LUFS")
    if not check_loudness_validator_available():
        report.notes.append("FFmpeg is required for loudness validation")
        return report

    report.available = True
    report.version = "ITU-R BS.1770 via FFmpeg loudnorm"

    from opencut.core.loudness_match import measure_loudness

    handle, path = tempfile.mkstemp(suffix=".wav", prefix="opencut_r128_")
    os.close(handle)
    try:
        build_reference_tone(path, rms_dbfs=rms_dbfs)
        measured = measure_loudness(path)
        lufs = float(measured.get("lufs", 0.0))
        report.measurements = {
            "rms_dbfs": rms_dbfs,
            "target_lufs": target_lufs,
            "measured_lufs": lufs,
            "delta_lu": round(lufs - target_lufs, 3),
            "tolerance_lu": tolerance_lu,
        }
        report.passed = abs(lufs - target_lufs) <= tolerance_lu
        if not report.passed:
            report.errors.append(
                f"measured {lufs:.2f} LUFS for a {target_lufs:.2f} LUFS "
                f"reference tone (tolerance +/-{tolerance_lu} LU)"
            )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"loudness validation failed: {exc}")
        report.passed = False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return report


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
def validator_status() -> dict:
    """Which reference validators this host can actually run."""
    return {
        "imsc": {
            "available": check_imsc_validator_available(),
            "version": _imsc_validator_version() if check_imsc_validator_available() else "",
            "install_hint": INSTALL_HINT_IMSC,
        },
        "imf": {
            "available": check_imf_validator_available(),
            "version": os.path.basename(photon_jar_path()) if photon_jar_path() else "",
            "install_hint": INSTALL_HINT_IMF,
        },
        "loudness": {
            "available": check_loudness_validator_available(),
            "version": "ITU-R BS.1770 via FFmpeg loudnorm",
            "install_hint": "FFmpeg is bundled with the installer.",
        },
    }


__all__ = [
    "ValidationReport",
    "PHOTON_JAR_ENV",
    "LOUDNESS_TOLERANCE_LU",
    "check_imsc_validator_available",
    "check_imf_validator_available",
    "check_loudness_validator_available",
    "validate_imsc",
    "validate_imf_package",
    "validate_loudness_measurement",
    "build_reference_tone",
    "expected_tone_lufs",
    "k_weighting_gain_db",
    "validator_status",
]
