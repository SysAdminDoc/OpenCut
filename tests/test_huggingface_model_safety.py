"""Security contract for Hugging Face downloads and custom model code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from opencut.core import model_safety

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_REVISION = "a" * 40


@pytest.mark.parametrize(
    "filename",
    [
        "../escape",
        "nested/../../escape",
        "..\\..\\escape",
        "/etc/passwd",
        "\\Windows\\System32\\escape",
        "C:\\Windows\\System32\\escape",
        "C:drive-relative",
        "//server/share/escape",
        "\\\\server\\share\\escape",
    ],
)
def test_hostile_repository_filenames_are_rejected_cross_platform(filename):
    with pytest.raises(ValueError, match="unsafe Hugging Face filename"):
        model_safety.validate_hf_filename(filename)


def test_snapshot_preflight_rejects_hostile_name_before_disk_mutation(monkeypatch, tmp_path):
    destination = tmp_path / "model"
    download_calls = []

    class FakeHub:
        @staticmethod
        def list_repo_files(*_args, **_kwargs):
            return ["config.json", "C:outside"]

        @staticmethod
        def snapshot_download(*_args, **_kwargs):
            download_calls.append(True)
            destination.mkdir()
            return str(destination)

    monkeypatch.setattr(model_safety, "_load_huggingface_hub", lambda: FakeHub)

    with pytest.raises(ValueError, match="unsafe Hugging Face filename"):
        model_safety.safe_snapshot_download(
            "owner/model",
            revision=PINNED_REVISION,
            local_dir=destination,
        )

    assert download_calls == []
    assert not destination.exists()


def test_snapshot_download_uses_one_immutable_revision(monkeypatch, tmp_path):
    calls = []

    class FakeHub:
        @staticmethod
        def list_repo_files(repo_id, **kwargs):
            calls.append(("list", repo_id, kwargs))
            return ["config.json", "weights/model.safetensors"]

        @staticmethod
        def snapshot_download(repo_id, **kwargs):
            calls.append(("download", repo_id, kwargs))
            return str(tmp_path / "cache")

    monkeypatch.setattr(model_safety, "_load_huggingface_hub", lambda: FakeHub)

    result = model_safety.safe_snapshot_download(
        "owner/model",
        revision=PINNED_REVISION,
        local_dir=tmp_path / "model",
    )

    assert result == str(tmp_path / "cache")
    assert calls[0][2]["revision"] == PINNED_REVISION
    assert calls[1][2]["revision"] == PINNED_REVISION


def test_single_file_download_rejects_path_before_hub_call(monkeypatch):
    calls = []

    class FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs):
            calls.append(kwargs)
            return "should-not-exist"

    monkeypatch.setattr(model_safety, "_load_huggingface_hub", lambda: FakeHub)

    with pytest.raises(ValueError, match="unsafe Hugging Face filename"):
        model_safety.safe_hf_hub_download(
            repo_id="owner/model",
            filename="..\\outside.nemo",
            revision=PINNED_REVISION,
        )

    assert calls == []


@pytest.mark.parametrize("revision", ["", "main", "v1.0", "deadbee", "g" * 40])
def test_mutable_or_incomplete_revision_is_rejected(revision):
    with pytest.raises(ValueError, match="full 40-character commit"):
        model_safety.require_immutable_hf_revision(revision)


def test_remote_code_runs_only_for_reviewed_model_revision_pairs():
    revision = model_safety.REVIEWED_REMOTE_CODE_MODELS["microsoft/Florence-2-base"]

    assert model_safety.reviewed_remote_code_kwargs(
        "microsoft/Florence-2-base",
        revision=revision,
    ) == {"trust_remote_code": True, "revision": revision}

    with pytest.raises(model_safety.ModelSecurityError, match="not reviewed"):
        model_safety.reviewed_remote_code_kwargs("attacker/model", revision=PINNED_REVISION)
    with pytest.raises(model_safety.ModelSecurityError, match="not reviewed"):
        model_safety.reviewed_remote_code_kwargs(
            "MICROSOFT/Florence-2-base",
            revision=revision,
        )
    with pytest.raises(model_safety.ModelSecurityError, match="revision mismatch"):
        model_safety.reviewed_remote_code_kwargs(
            "microsoft/Florence-2-base",
            revision=PINNED_REVISION,
        )


def test_no_loader_can_bypass_the_reviewed_remote_code_policy():
    violations = []
    direct_hub_imports = []
    for path in sorted((REPO_ROOT / "opencut").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "huggingface_hub" for alias in node.names):
                    if relative != "opencut/core/model_safety.py":
                        direct_hub_imports.append(f"{relative}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module == "huggingface_hub":
                imported = {alias.name for alias in node.names}
                guarded_download_apis = {
                    "hf_hub_download",
                    "list_repo_files",
                    "snapshot_download",
                }
                if (
                    imported & guarded_download_apis
                    and relative != "opencut/core/model_safety.py"
                ):
                    direct_hub_imports.append(f"{relative}:{node.lineno}")
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"hf_hub_download", "list_repo_files", "snapshot_download"}
                and relative != "opencut/core/model_safety.py"
            ):
                direct_hub_imports.append(f"{relative}:{node.lineno}")
            for keyword in node.keywords:
                if (
                    keyword.arg == "trust_remote_code"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    violations.append(f"{relative}:{node.lineno}")

    assert direct_hub_imports == []
    assert violations == []


def test_huggingface_hub_security_floor_is_release_126():
    assert model_safety.HUGGINGFACE_HUB_MIN_VERSION == "1.26.0"


@pytest.mark.parametrize("version", ["1.25.9", "1.26.0rc1", "2.0.0"])
def test_runtime_rejects_hub_versions_outside_the_reviewed_release_lane(
    monkeypatch,
    version,
):
    monkeypatch.setattr(model_safety.metadata, "version", lambda _name: version)

    with pytest.raises(model_safety.ModelSecurityError, match="reviewed"):
        model_safety.require_safe_huggingface_hub()


def test_remote_weight_loaders_prefer_safetensors_when_the_repo_supports_it():
    for relative in (
        "opencut/core/matte_birefnet.py",
        "opencut/core/object_removal.py",
        "opencut/core/slate_id.py",
        "opencut/routes/video_core.py",
    ):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "use_safetensors=True" in source, relative
