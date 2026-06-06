"""Distribution documentation guards for Docker onboarding."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readme_docker_commands_match_committed_compose_file():
    readme = _read("README.md")
    assert "docker-compose.gpu.yml" not in readme
    assert "docker compose up opencut-server" in readme
    assert "docker compose --profile gpu up opencut-server-gpu" in readme


def test_docker_run_examples_use_non_root_data_home():
    dockerfile = _read("Dockerfile")
    assert "/root/.opencut" not in dockerfile
    assert "-v opencut-data:/home/opencut/.opencut" in dockerfile


def test_documented_compose_override_files_exist():
    docs = "\n".join(_read(path) for path in ["README.md", "Dockerfile"])
    referenced = set(re.findall(r"docker compose -f\s+([^\s]+)", docs))
    referenced |= set(re.findall(r"docker-compose -f\s+([^\s]+)", docs))
    missing = sorted(path for path in referenced if not (REPO_ROOT / path).is_file())
    assert missing == []


def test_gpu_compose_command_targets_gpu_service_only():
    compose = _read("docker-compose.yml")
    assert not compose.startswith("version:")
    assert "docker compose --profile gpu up opencut-server-gpu" in compose
    assert re.search(r"opencut-server-gpu:\s+build:", compose, re.S)
