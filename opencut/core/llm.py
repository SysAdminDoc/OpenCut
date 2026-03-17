"""
OpenCut LLM Abstraction Module

Shared LLM interface for highlight extraction, video summarization,
and the shorts pipeline. Supports Ollama (local), OpenAI, and Anthropic.

Uses stdlib urllib.request — zero pip dependencies.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "ollama"  # "ollama", "openai", "anthropic"
    model: str = "llama3"
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 2000


@dataclass
class LLMResponse:
    """Response from LLM query."""
    text: str = ""
    provider: str = ""
    model: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------
def _http_json(url, data=None, headers=None, timeout=60):
    """Make an HTTP request and return parsed JSON response."""
    headers = headers or {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    else:
        body = None

    req = urllib.request.Request(url, data=body, headers=headers)
    method = "POST" if body else "GET"
    req.get_method = lambda: method

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {err_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed to {url}: {exc.reason}") from exc


def _query_ollama(prompt, system_prompt, config):
    """Query Ollama local server."""
    from urllib.parse import urlparse
    parsed = urlparse(config.base_url)
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError("Ollama base_url must point to localhost for security")
    url = f"{config.base_url.rstrip('/')}/api/generate"
    body = {
        "model": config.model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    data = _http_json(url, data=body, timeout=180)

    return LLMResponse(
        text=data.get("response", ""),
        provider="ollama",
        model=config.model,
        tokens_used=data.get("eval_count", 0),
    )


def _query_openai(prompt, system_prompt, config):
    """Query OpenAI API."""
    if not config.api_key:
        raise RuntimeError("OpenAI API key is required. Set config.api_key.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    data = _http_json(url, data=body, headers=headers, timeout=60)

    text = ""
    tokens = 0
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        text = json.dumps(data)
    try:
        tokens = data.get("usage", {}).get("total_tokens", 0)
    except Exception:
        pass

    return LLMResponse(
        text=text,
        provider="openai",
        model=config.model,
        tokens_used=tokens,
    )


def _query_anthropic(prompt, system_prompt, config):
    """Query Anthropic API."""
    if not config.api_key:
        raise RuntimeError("Anthropic API key is required. Set config.api_key.")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    body = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt

    data = _http_json(url, data=body, headers=headers, timeout=60)

    text = ""
    tokens = 0
    try:
        text = data["content"][0]["text"]
    except (KeyError, IndexError):
        text = json.dumps(data)
    try:
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
    except Exception:
        pass

    return LLMResponse(
        text=text,
        provider="anthropic",
        model=config.model,
        tokens_used=tokens,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_PROVIDERS = {
    "ollama": _query_ollama,
    "openai": _query_openai,
    "anthropic": _query_anthropic,
}


def query_llm(prompt, config=None, system_prompt="", on_progress=None):
    """
    Send prompt to configured LLM provider.

    Args:
        prompt: User prompt text.
        config: LLMConfig instance. Uses defaults (Ollama) if None.
        system_prompt: Optional system instruction.
        on_progress: Progress callback(pct, msg).

    Returns:
        LLMResponse with the model's reply.
    """
    if config is None:
        config = LLMConfig()

    provider = config.provider.lower().strip()
    handler = _PROVIDERS.get(provider)
    if handler is None:
        return LLMResponse(
            text=f"Unknown LLM provider: '{provider}'. Use 'ollama', 'openai', or 'anthropic'.",
            provider=provider,
            model=config.model,
        )

    if on_progress:
        on_progress(10, f"Querying {provider} ({config.model})...")

    try:
        response = handler(prompt, system_prompt, config)
    except Exception as exc:
        logger.error("LLM query failed (%s/%s): %s", provider, config.model, exc)
        return LLMResponse(
            text=f"LLM error: {exc}",
            provider=provider,
            model=config.model,
        )

    if on_progress:
        on_progress(100, "LLM response received")

    logger.info("LLM response from %s/%s (%d tokens)", provider, config.model, response.tokens_used)
    return response


def check_llm_reachable(config=None):
    """
    Check if configured LLM provider is reachable.

    Returns:
        dict with keys: available (bool), provider (str), models (list), error (str).
    """
    if config is None:
        config = LLMConfig()

    provider = config.provider.lower().strip()
    result = {"available": False, "provider": provider, "models": [], "error": ""}

    if provider == "ollama":
        try:
            models = list_ollama_models(config.base_url)
            result["available"] = True
            result["models"] = models
        except Exception as exc:
            result["error"] = str(exc)

    elif provider == "openai":
        if config.api_key:
            result["available"] = True
            result["models"] = [config.model]
        else:
            result["error"] = "OpenAI API key not configured"

    elif provider == "anthropic":
        if config.api_key:
            result["available"] = True
            result["models"] = [config.model]
        else:
            result["error"] = "Anthropic API key not configured"

    else:
        result["error"] = f"Unknown provider: {provider}"

    return result


def list_ollama_models(base_url="http://localhost:11434"):
    """
    List available Ollama models.

    Args:
        base_url: Ollama server URL.

    Returns:
        List of model name strings.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    data = _http_json(url, timeout=10)

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            models.append(name)

    return models
