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


def test_docker_runtime_is_http_only_by_default():
    dockerfile = _read("Dockerfile")
    compose = _read("docker-compose.yml")
    readme = _read("README.md")

    assert re.search(r"^EXPOSE 5679$", dockerfile, re.M)
    assert "EXPOSE 5679 5680" not in dockerfile
    assert "5680:5680" not in compose
    default_service = compose.split("  # GPU-enabled variant", maxsplit=1)[0]
    assert "5681:5681" not in default_service
    assert "Docker publishes the HTTP API on port 5679" in readme
    assert "does not publish the optional WebSocket 5680 or\nMCP 5681 sidecars by default" in readme


def test_docker_remote_bind_explicitly_opts_into_auth_gate():
    dockerfile = _read("Dockerfile")
    compose = _read("docker-compose.yml")

    assert "ENV OPENCUT_HOST=0.0.0.0" in dockerfile
    assert "ENV OPENCUT_ALLOW_REMOTE=1" in dockerfile
    assert re.search(r"opencut-server:\s+.*?OPENCUT_ALLOW_REMOTE=1", compose, re.S)
    assert re.search(r"opencut-server-gpu:\s+.*?OPENCUT_ALLOW_REMOTE=1", compose, re.S)


def test_every_remote_compose_profile_mounts_the_required_secret_file():
    compose = _read("docker-compose.yml")

    assert "file: ${OPENCUT_REMOTE_AUTH_TOKEN_FILE:?" in compose
    assert "OPENCUT_ALLOW_INSECURE_SECRET_STORAGE" not in compose
    for service in ["opencut-server", "opencut-server-gpu", "opencut-mcp"]:
        block = compose.split(f"  {service}:", maxsplit=1)[1]
        block = re.split(r"\n  [a-zA-Z][\w-]*:", block, maxsplit=1)[0]
        assert "- opencut_remote_auth_token" in block
        assert "OPENCUT_REMOTE_AUTH_TOKEN_FILE=/run/secrets/opencut_remote_auth_token" in block


def test_container_entrypoint_normalizes_secret_without_exporting_value():
    dockerfile = _read("Dockerfile")
    entrypoint = _read("scripts/docker-entrypoint.sh")

    assert 'ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]' in dockerfile
    assert "USER opencut" in dockerfile
    assert "chmod 0400" in entrypoint
    assert '[ -L "$source_path" ]' in entrypoint
    assert 'export OPENCUT_REMOTE_AUTH_TOKEN_FILE="$runtime_path"' in entrypoint
    assert "export OPENCUT_REMOTE_AUTH_TOKEN=" not in entrypoint


def test_mcp_compose_profile_uses_supported_http_arguments_and_backend_url():
    compose = _read("docker-compose.yml")
    mcp_block = compose.split("  opencut-mcp:", maxsplit=1)[1].split("\nvolumes:", maxsplit=1)[0]

    assert '"--http", "--http-bind", "0.0.0.0", "--http-port", "5681"' in mcp_block
    assert "OPENCUT_MCP_BACKEND_URL=http://opencut-server:5679" in mcp_block
    assert 'entrypoint: ["python", "-m", "opencut.mcp_server"]' not in mcp_block


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


def test_dockerfile_uses_tracked_dependency_surface():
    dockerfile = _read("Dockerfile")
    assert "pip install --no-cache-dir --requirement requirements.txt" in dockerfile
    forbidden_fail_open_tokens = ["|| echo", "|| true", "set +e", "Some optional deps failed"]
    for token in forbidden_fail_open_tokens:
        assert token not in dockerfile
    assert "pydub" not in dockerfile
    assert "deep-translator" not in dockerfile


def test_dockerignore_mirrors_sensitive_gitignore_patterns():
    gitignore = _read(".gitignore").splitlines()
    dockerignore = {line.strip() for line in _read(".dockerignore").splitlines() if line.strip()}
    sensitive_patterns = [
        ".env",
        ".env.*",
        "*.key",
        "*.pem",
        "credentials*.json",
        "*.log",
    ]
    missing_from_gitignore = [pattern for pattern in sensitive_patterns if pattern not in gitignore]
    missing_from_dockerignore = [pattern for pattern in sensitive_patterns if pattern not in dockerignore]
    assert missing_from_gitignore == []
    assert missing_from_dockerignore == []


def test_dockerignore_excludes_local_runtime_state():
    dockerignore = {line.strip() for line in _read(".dockerignore").splitlines() if line.strip()}
    required_patterns = [
        ".coverage",
        "htmlcov/",
        ".opencut/",
        "*.db",
        "*.db-*",
        "*.sqlite",
        "*.sqlite3",
        "*.sqlite3-*",
    ]
    missing = [pattern for pattern in required_patterns if pattern not in dockerignore]
    assert missing == []
