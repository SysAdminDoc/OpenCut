"""F324 — strings both panels show must have one canonical source.

The panels keep independent locale namespaces (2,880 CEP keys, 1,928 UXP keys,
26 shared names), which hides how much they overlap: hundreds of identical
English strings sit under different key names. A translator adding a language
would translate each of them twice, and the two copies drift as soon as one
panel is edited. These tests pin the shared registry, the drift gate, and the
propagation path that makes a new language a single translation pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_shared_locales.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("sync_shared_locales", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


class TestRegistryIsPresentAndSane:
    def test_the_canonical_files_exist(self):
        assert tool.SHARED_EN.is_file(), "run scripts/sync_shared_locales.py --regenerate"
        assert tool.KEY_MAP.is_file()

    def test_every_shared_string_owns_keys_in_both_panels(self):
        english = json.loads(tool.SHARED_EN.read_text(encoding="utf-8"))
        key_map = json.loads(tool.KEY_MAP.read_text(encoding="utf-8"))

        assert english, "the shared registry is empty"
        assert set(english) == set(key_map), "the English file and key map disagree"
        for slug, owners in key_map.items():
            assert owners.get("cep"), f"{slug} owns no CEP key"
            assert owners.get("uxp"), f"{slug} owns no UXP key"

    def test_trivial_strings_are_not_coupled(self):
        """Forcing "2x" through a shared key couples panels for no benefit."""
        english = json.loads(tool.SHARED_EN.read_text(encoding="utf-8"))
        for slug, value in english.items():
            assert len(value) >= tool.MIN_SHARED_LENGTH, f"{slug} is too trivial to share"

    def test_a_shared_key_is_not_claimed_twice(self):
        key_map = json.loads(tool.KEY_MAP.read_text(encoding="utf-8"))
        for panel in ("cep", "uxp"):
            seen = set()
            for slug, owners in key_map.items():
                for key in owners.get(panel, []):
                    assert key not in seen, f"{panel}:{key} is owned by two shared strings"
                    seen.add(key)


class TestTheGate:
    def test_the_tree_is_currently_clean(self):
        report = tool.check(verbose=False)
        assert report["ok"], (
            "shared locale drift: "
            f"drifted={report['drifted'][:3]} missing={report['missing'][:3]} "
            f"unregistered={report['unregistered_duplicates'][:3]}"
        )

    def test_drift_between_panels_is_detected(self, tmp_path, monkeypatch):
        """The whole point: one panel edited, the other not."""
        self._stage(tmp_path, monkeypatch, uxp_override={"uxp.a": "CHANGED"})
        report = tool.check(verbose=False)
        assert report["ok"] is False
        assert report["drifted"]

    def test_a_removed_mapped_key_is_reported(self, tmp_path, monkeypatch):
        self._stage(tmp_path, monkeypatch, uxp_override={}, drop_uxp=True)
        report = tool.check(verbose=False)
        assert report["ok"] is False
        assert report["missing"]

    def test_new_unregistered_duplication_is_reported(self, tmp_path, monkeypatch):
        """A new string copied into both panels must be registered, not ignored."""
        shared_text = "A brand new sentence shown by both panels"
        self._stage(
            tmp_path,
            monkeypatch,
            cep_extra={"cep.new": shared_text},
            uxp_extra={"uxp.new": shared_text},
        )
        report = tool.check(verbose=False)
        assert report["ok"] is False
        assert shared_text in report["unregistered_duplicates"]

    def test_trivial_new_duplication_is_ignored(self, tmp_path, monkeypatch):
        self._stage(tmp_path, monkeypatch, cep_extra={"cep.ok": "OK"}, uxp_extra={"uxp.ok": "OK"})
        report = tool.check(verbose=False)
        assert report["ok"] is True

    def _stage(self, tmp_path, monkeypatch, *, cep_extra=None, uxp_extra=None,
               uxp_override=None, drop_uxp=False):
        """Build a miniature two-panel tree with one registered shared string."""
        canonical = "Detect and stage silent segments"
        cep_dir = tmp_path / "cep"
        uxp_dir = tmp_path / "uxp"
        shared_dir = tmp_path / "shared"
        for directory in (cep_dir, uxp_dir, shared_dir):
            directory.mkdir(parents=True)

        cep = {"cep.a": canonical}
        cep.update(cep_extra or {})
        uxp = {} if drop_uxp else {"uxp.a": canonical}
        uxp.update(uxp_override or {})
        uxp.update(uxp_extra or {})

        (cep_dir / "en.json").write_text(json.dumps(cep), encoding="utf-8")
        (uxp_dir / "en.json").write_text(json.dumps(uxp), encoding="utf-8")
        (shared_dir / "en.json").write_text(
            json.dumps({"shared.cep.a": canonical}), encoding="utf-8"
        )
        (shared_dir / "key_map.json").write_text(
            json.dumps({"shared.cep.a": {"cep": ["cep.a"], "uxp": ["uxp.a"]}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(tool, "CEP_LOCALES", cep_dir)
        monkeypatch.setattr(tool, "UXP_LOCALES", uxp_dir)
        monkeypatch.setattr(tool, "SHARED_DIR", shared_dir)
        monkeypatch.setattr(tool, "SHARED_EN", shared_dir / "en.json")
        monkeypatch.setattr(tool, "KEY_MAP", shared_dir / "key_map.json")


class TestPropagation:
    def test_one_translation_reaches_both_panels(self, tmp_path, monkeypatch):
        """This is what "translate each string once" has to mean in practice."""
        cep_dir = tmp_path / "cep"
        uxp_dir = tmp_path / "uxp"
        shared_dir = tmp_path / "shared"
        for directory in (cep_dir, uxp_dir, shared_dir):
            directory.mkdir(parents=True)

        (shared_dir / "key_map.json").write_text(
            json.dumps({"shared.cep.a": {"cep": ["cep.a", "cep.b"], "uxp": ["uxp.a"]}}),
            encoding="utf-8",
        )
        (shared_dir / "fr.json").write_text(
            json.dumps({"shared.cep.a": "Détecter les silences"}), encoding="utf-8"
        )

        monkeypatch.setattr(tool, "CEP_LOCALES", cep_dir)
        monkeypatch.setattr(tool, "UXP_LOCALES", uxp_dir)
        monkeypatch.setattr(tool, "SHARED_DIR", shared_dir)
        monkeypatch.setattr(tool, "KEY_MAP", shared_dir / "key_map.json")

        assert tool.propagate("fr") == 0

        cep = json.loads((cep_dir / "fr.json").read_text(encoding="utf-8"))
        uxp = json.loads((uxp_dir / "fr.json").read_text(encoding="utf-8"))
        assert cep["cep.a"] == "Détecter les silences"
        assert cep["cep.b"] == "Détecter les silences"
        assert uxp["uxp.a"] == "Détecter les silences"

    def test_propagating_a_missing_language_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tool, "SHARED_DIR", tmp_path)
        assert tool.propagate("zz") == 2


def test_release_smoke_runs_the_shared_locale_gate():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")
    assert "sync_shared_locales" in smoke, (
        "the shared-locale gate must run in release smoke or it will rot"
    )


@pytest.mark.parametrize("flag", ["--regenerate", "--check", "--propagate"])
def test_cli_exposes_the_documented_flags(flag):
    assert flag in SCRIPT.read_text(encoding="utf-8")
