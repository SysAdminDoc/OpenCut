"""System model routes registered on the shared system blueprint."""

from .system import (
    _MODELS_CACHE_TTL,
    WHISPER_MODELS_DIR,
    __version__,
    _cache_plan_entry,
    _delete_cache_target,
    _models_cache,
    _models_cache_lock,
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    is_path_within_any,
    jsonify,
    logger,
    os,
    require_csrf,
    require_rate_limit,
    safe_bool,
    safe_error,
    safe_pip_install,
    should_skip_install_in_testing,
    suppress,
    system_bp,
    testing_install_response,
    threading,
    time,
    validate_path,
    verify_destructive_confirm_token,
)


# ---------------------------------------------------------------------------
# Demucs Install
# ---------------------------------------------------------------------------
@system_bp.route("/demucs/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_demucs():
    """Install Demucs for AI audio separation."""
    if should_skip_install_in_testing("install"):
        return testing_install_response("demucs")

    try:
        safe_pip_install("demucs", timeout=600)
        return jsonify({"success": True, "message": "Demucs installed successfully"})
    except RuntimeError as e:
        return safe_error(e, "install_demucs")
    except Exception as e:
        return safe_error(e, "install_demucs")


# ---------------------------------------------------------------------------
# Watermark Install
# ---------------------------------------------------------------------------
@system_bp.route("/watermark/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_watermark():
    """Install watermark removal dependencies."""
    if should_skip_install_in_testing("install"):
        return testing_install_response("watermark")

    try:
        packages = [
            'transformers',
            'simple-lama-inpainting',
            'torch',
            'torchvision',
            'pillow',
            'opencv-python'
        ]

        failed = []
        for pkg in packages:
            try:
                safe_pip_install(pkg, timeout=600)
            except RuntimeError:
                failed.append(pkg)

        if failed:
            from opencut.errors import error_response
            return error_response("INSTALL_FAILED",
                                  f"Failed to install: {', '.join(failed)}",
                                  status=500,
                                  suggestion="Try running as administrator or install manually via pip.")
        return jsonify({"success": True, "message": "Watermark remover installed successfully"})

    except Exception as e:
        return safe_error(e, "install_watermark")


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------
@system_bp.route("/models/list", methods=["GET"])
def list_models():
    """List downloaded AI models and their sizes (cached with 5-min TTL)."""
    now = time.time()
    with _models_cache_lock:
        if _models_cache["data"] is not None and (now - _models_cache["ts"]) < _MODELS_CACHE_TTL:
            return jsonify(_models_cache["data"])

    models = []
    # Check HuggingFace cache
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        for entry in os.listdir(hf_cache):
            path = os.path.join(hf_cache, entry)
            if os.path.isdir(path) and entry.startswith("models--"):
                name = entry.replace("models--", "").replace("--", "/")
                size = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        with suppress(OSError):
                            size += os.path.getsize(os.path.join(root, f))
                models.append({"name": name, "path": path, "size_mb": round(size / (1024 * 1024), 1), "source": "huggingface"})
    # Check torch hub cache
    torch_cache = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub"))
    if os.path.isdir(torch_cache):
        checkpoints = os.path.join(torch_cache, "checkpoints")
        if os.path.isdir(checkpoints):
            for f in os.listdir(checkpoints):
                fp = os.path.join(checkpoints, f)
                if os.path.isfile(fp):
                    size = os.path.getsize(fp) / (1024 * 1024)
                    models.append({"name": f, "path": fp, "size_mb": round(size, 1), "source": "torch"})
    # Check custom whisper models dir
    if WHISPER_MODELS_DIR and os.path.isdir(WHISPER_MODELS_DIR):
        for entry in os.listdir(WHISPER_MODELS_DIR):
            path = os.path.join(WHISPER_MODELS_DIR, entry)
            if os.path.isdir(path):
                size = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        with suppress(OSError):
                            size += os.path.getsize(os.path.join(root, f))
                models.append({"name": "whisper/" + entry, "path": path, "size_mb": round(size / (1024 * 1024), 1), "source": "whisper"})

    result = {"models": models, "total_mb": round(sum(m["size_mb"] for m in models), 1)}
    with _models_cache_lock:
        _models_cache["data"] = result
        _models_cache["ts"] = now
    return jsonify(result)

@system_bp.route("/models/delete", methods=["POST"])
@require_csrf
def delete_model():
    """Delete a downloaded model."""
    try:
        data = get_json_dict()
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "code": "INVALID_INPUT",
            "suggestion": "Send a top-level JSON object in the request body.",
        }), 400
    path = data.get("path", "")
    if path and not isinstance(path, str):
        return jsonify({
            "error": "path must be a string",
            "code": "INVALID_INPUT",
            "suggestion": "Pass the exact model cache file or directory path as a string.",
        }), 400
    path = path.strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    # Validate path for traversal attacks (may be file or directory)
    try:
        path = validate_path(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Security: only allow deletion within known cache directories
    allowed_roots = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    ]
    if WHISPER_MODELS_DIR:
        allowed_roots.append(WHISPER_MODELS_DIR)
    if not is_path_within_any(path, allowed_roots):
        return jsonify({"error": "Cannot delete files outside of model cache directories"}), 403
    if not os.path.isdir(path) and not os.path.isfile(path):
        return jsonify({"error": "Path not found"}), 404

    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    plan_entry = _cache_plan_entry(path, category="model-cache")
    entries = [plan_entry]
    plan = build_destructive_plan(
        "models.delete",
        targets=entries,
        metadata={"route": "/models/delete", "total_bytes": plan_entry["bytes"]},
        reversible=False,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": entries,
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "total_bytes": plan_entry["bytes"],
            "deleted": [],
            "errors": plan_entry.get("errors", []),
        })
    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409

    deleted, delete_error = _delete_cache_target(path)
    errors = list(plan_entry.get("errors", []))
    if delete_error:
        errors.append(delete_error)
    try:
        if errors or not deleted:
            return jsonify({
                "success": False,
                "dry_run": False,
                "plan": entries,
                "destructive_plan": plan,
                "total_bytes": plan_entry["bytes"],
                "deleted": [],
                "errors": errors,
            }), 500
        # Invalidate models cache after deletion
        with _models_cache_lock:
            _models_cache["data"] = None
            _models_cache["ts"] = 0
        return jsonify({
            "success": True,
            "dry_run": False,
            "plan": entries,
            "destructive_plan": plan,
            "total_bytes": plan_entry["bytes"],
            "deleted": [path],
            "errors": errors,
        })
    except Exception as e:
        return safe_error(e, "delete_model")


# ---------------------------------------------------------------------------
# LLM Status & Test
# ---------------------------------------------------------------------------
@system_bp.route("/llm/status", methods=["GET"])
def llm_status():
    """Check LLM provider availability and list Ollama models if running."""
    from opencut.checks import check_ollama_available
    ollama_ok = check_ollama_available()

    ollama_models = []
    if ollama_ok:
        try:
            from opencut.core.llm import list_ollama_models
            ollama_models = list_ollama_models()
        except Exception:
            pass

    return jsonify({
        "ollama": {"available": ollama_ok, "models": ollama_models},
        "openai": {"available": True, "note": "Requires API key"},
        "anthropic": {"available": True, "note": "Requires API key"},
    })


@system_bp.route("/llm/test", methods=["POST"])
@require_csrf
def llm_test():
    """Test LLM connectivity with a simple prompt."""
    data = get_json_dict()

    _VALID_LLM_PROVIDERS = {"ollama", "openai", "anthropic", "gemini"}
    provider = data.get("provider", "ollama").strip().lower()
    if provider not in _VALID_LLM_PROVIDERS:
        return jsonify({"success": False, "error": f"Invalid provider: {provider}. Must be one of: {', '.join(sorted(_VALID_LLM_PROVIDERS))}"}), 400
    model = data.get("model", "").strip()
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()

    try:
        from opencut.core.llm import LLMConfig, query_llm

        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=50,
        )

        response = query_llm("Say 'Hello from OpenCut!' in exactly those words.", config)

        return jsonify({
            "success": True,
            "response": response.text,
            "provider": response.provider,
            "model": response.model,
        })
    except Exception as e:
        return safe_error(e, "llm_test")


# ---------------------------------------------------------------------------
# Update Check (GitHub releases, 1-hour cache)
# ---------------------------------------------------------------------------
_update_cache = {"data": None, "ts": 0}
_update_cache_lock = threading.Lock()
_UPDATE_CACHE_TTL = 3600  # 1 hour


@system_bp.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Return the OpenAPI 3.0 specification for this server."""
    from flask import current_app

    from opencut.openapi import generate_openapi_spec
    return jsonify(generate_openapi_spec(current_app))


@system_bp.route("/system/update-check", methods=["GET"])
def check_for_update():
    """Check GitHub for a newer release. Cached for 1 hour."""
    import json
    import urllib.request

    now = time.time()
    with _update_cache_lock:
        if _update_cache["data"] is not None and (now - _update_cache["ts"]) < _UPDATE_CACHE_TTL:
            return jsonify(_update_cache["data"])

    current = __version__
    result = {
        "current_version": current,
        "latest_version": current,
        "update_available": False,
        "release_url": "https://github.com/SysAdminDoc/OpenCut/releases",
    }

    try:
        url = "https://api.github.com/repos/SysAdminDoc/OpenCut/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "OpenCut"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            _max_bytes = 10 * 1024 * 1024
            raw = resp.read(_max_bytes + 1)
            if len(raw) > _max_bytes:
                raise ValueError(f"GitHub response exceeds {_max_bytes} byte cap")
            data = json.loads(raw.decode("utf-8"))

        tag = data.get("tag_name", "").lstrip("vV")
        html_url = data.get("html_url", result["release_url"])

        if tag:
            result["latest_version"] = tag
            result["release_url"] = html_url

            def _parse_version(v: str) -> tuple:
                """Parse version like '1.16.0' or '1.16.0-rc1' into comparable tuple.

                Keeps only leading integer components, ignoring any -rc/.dev/+build
                suffix. Falls back to (0,) if no digits are present so comparison
                never raises on unexpected inputs.
                """
                parts = []
                for seg in str(v).split("."):
                    # Strip any trailing non-digit suffix (e.g. "0-rc1" -> "0")
                    digits = ""
                    for ch in seg:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        parts.append(int(digits))
                    else:
                        break
                return tuple(parts) if parts else (0,)

            try:
                current_parts = _parse_version(current)
                latest_parts = _parse_version(tag)
                result["update_available"] = latest_parts > current_parts
            except Exception as ve:
                logger.debug("Version compare failed (%s vs %s): %s", current, tag, ve)
    except Exception as exc:
        logger.debug("Update check failed: %s", exc)
        result["error"] = "offline"

    with _update_cache_lock:
        _update_cache["data"] = result
        _update_cache["ts"] = now

    return jsonify(result)
