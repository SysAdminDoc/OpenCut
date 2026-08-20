"""The suite must run on the stack the project declares.

`check_dependency_matrix.py` resolved the declared lanes but never compared
them to what is installed, so a 10,726-pass baseline could be produced on a
stack that violated four of OpenCut's own constraints - two at major-version
boundaries. A user installing per `pyproject.toml` then executed code paths
the suite had never run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_installed_versions.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_installed_versions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


# A declared security floor that cannot be installed alongside the rest of the
# stack is recorded here with its concrete blocker; anything NOT on this list
# is a new regression and fails the test. `scripts/check_installed_versions.py`
# (and the release smoke step that runs it) still fail on these - that is the
# point of the gate.
KNOWN_ENVIRONMENT_GAPS = {
    # transformers 5.x forces huggingface_hub 1.x, which the pyannote /
    # faster-whisper / diffusers stack in this matrix does not accept yet.
    "transformers": ">=5.3",
}


def test_no_new_declared_constraint_is_violated():
    report = checker.check_installed(["all"])
    unexpected = [
        m for m in report["mismatches"] if m["distribution"] not in KNOWN_ENVIRONMENT_GAPS
    ]
    assert unexpected == [], (
        "the tested stack violates declared constraints that are not recorded "
        "as known gaps: "
        + "; ".join(
            f"{m['distribution']} {m['installed']} vs '{m['declared']}'"
            for m in unexpected
        )
    )


def test_recorded_environment_gaps_do_not_outlive_their_blockers():
    """A gap that has been resolved must be removed from the record."""
    report = checker.check_installed(["all"])
    violated = {m["distribution"] for m in report["mismatches"]}
    # An uninstalled distribution reports neither a mismatch nor a match, so it
    # is no evidence the blocker is gone. Only an installed-and-satisfying
    # distribution proves the gap is stale; otherwise a machine without the
    # optional AI extras would demand the record be deleted on absence alone.
    unproven = set(report["absent"])
    stale = sorted(set(KNOWN_ENVIRONMENT_GAPS) - violated - unproven)
    assert stale == [], (
        f"these distributions now satisfy their declared floors: {stale}. "
        "Drop them from KNOWN_ENVIRONMENT_GAPS so the gate is fully enforced."
    )


def test_the_gate_actually_inspects_something():
    """Guard against a check wired to a source that yields nothing."""
    report = checker.check_installed(["all"])
    assert report["checked"] > 20, report


def test_the_gate_reports_a_violation_it_is_given(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "fixture"',
                'version = "0.0.0"',
                'dependencies = ["pytest<1"]',
                "",
                "[project.optional-dependencies]",
                'extra = ["pytest>=99"]',
            ]
        ),
        encoding="utf-8",
    )

    report = checker.check_installed(["all"], pyproject=pyproject)

    assert report["ok"] is False
    assert [m["distribution"] for m in report["mismatches"]] == ["pytest"]


def test_absent_optional_packages_never_fail_the_gate(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'name = "fixture"',
                'version = "0.0.0"',
                'dependencies = []',
                "",
                "[project.optional-dependencies]",
                'extra = ["a-package-nobody-has>=1"]',
            ]
        ),
        encoding="utf-8",
    )

    report = checker.check_installed(["all"], pyproject=pyproject)

    assert report["ok"] is True
    assert report["absent"] == ["a-package-nobody-has"]


def test_release_smoke_runs_the_installed_version_gate():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")
    assert "step_installed_versions" in smoke
    assert "check_installed_versions.py" in smoke


class TestPySceneDetectSevenPath:
    """PySceneDetect 0.7 is a documented breaking release; exercise it."""

    def test_declared_floor_is_the_installed_major(self):
        scenedetect = pytest.importorskip("scenedetect")
        version = tuple(
            int(part) for part in scenedetect.__version__.split(".")[:2] if part.isdigit()
        )
        assert version >= (0, 7), (
            f"PySceneDetect {scenedetect.__version__} predates the 0.7 API the "
            "project declares; the adapter's 0.7 behaviour is untested here."
        )

    def test_detects_a_real_cut_through_the_zero_seven_api(self, tmp_path):
        pytest.importorskip("scenedetect")
        import subprocess

        from opencut.core.scene_detect import detect_scenes_pyscenedetect
        from opencut.helpers import get_ffmpeg_path

        clip = tmp_path / "two_scenes.mp4"
        result = subprocess.run(
            [
                get_ffmpeg_path(), "-y", "-v", "error",
                "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2,format=yuv420p",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2,format=yuv420p",
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-r", "25", str(clip),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0 or not clip.is_file():
            pytest.skip("FFmpeg could not generate the fixture clip")

        info = detect_scenes_pyscenedetect(str(clip), threshold=27.0, min_scene_length=0.5)

        times = [round(b.time, 1) for b in info.boundaries]
        assert times[0] == 0.0
        assert 2.0 in times, f"the red/blue cut at 2.0s was not detected: {times}"
        assert info.duration == pytest.approx(4.0, abs=0.2)
