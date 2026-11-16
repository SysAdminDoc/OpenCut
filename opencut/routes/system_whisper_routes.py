"""System whisper routes registered on the shared system blueprint."""

__all__ = [
    "install_whisper",
    "whisper_settings",
    "whisper_clear_cache",
    "whisper_reinstall",
]

from .system import (
    VALID_WHISPER_MODELS,
    _build_whisper_cache_plan,
    _delete_cache_target,
    _is_cancelled,
    _register_job_process,
    _unregister_job_process,
    _update_job,
    async_job,
    build_destructive_plan,
    destructive_confirmation_required_response,
    forget_capability_facts,
    get_json_dict,
    jsonify,
    load_whisper_settings,
    logger,
    request,
    require_csrf,
    safe_bool,
    save_whisper_settings,
    suppress,
    sys,
    system_bp,
    verify_destructive_confirm_token,
)


# ---------------------------------------------------------------------------
# Install Whisper
# ---------------------------------------------------------------------------
@system_bp.route("/install-whisper", methods=["POST"])
@require_csrf
@async_job("install-whisper", filepath_required=False, rate_limit_key="model_install")
def install_whisper(job_id, filepath, data):
    """Install faster-whisper via pip (on-demand from panel)."""
    backend = data.get("backend", "faster-whisper")
    allowed = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed:
        raise ValueError(f"Unknown backend: {backend}")

    import subprocess as _sp

    try:
        from opencut.core import package_store

        _update_job(job_id, progress=5, message=f"Installing {backend}...")
        logger.info(f"Starting Whisper install: {backend}")

        def _tooling_upgrade():
            """Upgrade the source interpreter's own pip so it finds current wheels."""
            from opencut.security import install_python

            base = [install_python(), "-m", "pip", "install", "--upgrade",
                    "pip", "setuptools", "wheel", "--quiet"]
            for cmd in (base, [*base, "--user"]):
                with suppress(Exception):
                    if _sp.run(cmd, capture_output=True, text=True, timeout=120, check=False).returncode == 0:
                        return

        # Each strategy goes through the add-on package store: a staging
        # folder, a child server that has to import the module, then the merge.
        if backend == "faster-whisper":
            strategies = [
                {
                    "label": "Install tokenizers wheel with faster-whisper",
                    "reqs": ["tokenizers", "faster-whisper"],
                    "pip_args": ["--only-binary", "tokenizers"],
                    "module": "faster_whisper",
                },
                {
                    "label": "Pin older tokenizers with wheel support",
                    "reqs": ["tokenizers>=0.13,<0.20", "faster-whisper"],
                    "pip_args": ["--only-binary", "tokenizers"],
                    "module": "faster_whisper",
                },
                {
                    "label": "Fallback to openai-whisper (no Rust needed)",
                    "reqs": ["openai-whisper"],
                    "module": "whisper",
                    "backend_name": "openai-whisper",
                },
            ]
            if not getattr(sys, "frozen", False):
                # A source checkout's own pip may be too old for current wheels.
                strategies.insert(1, {
                    "label": "Upgrade pip, then install faster-whisper",
                    "pre": _tooling_upgrade,
                    "reqs": ["faster-whisper"],
                    "module": "faster_whisper",
                })
        else:
            strategies = [
                {
                    "label": "Standard install",
                    "reqs": [backend],
                    "module": "whisper" if backend == "openai-whisper" else backend,
                },
            ]

        last_error = ""
        for si, strat in enumerate(strategies):
            if _is_cancelled(job_id):
                return {"backend": backend, "installed": False, "cancelled": True}

            pct_base = int(5 + (si / len(strategies)) * 70)
            _update_job(job_id, progress=pct_base,
                        message=f"Strategy {si+1}/{len(strategies)}: {strat['label']}...")
            logger.info(f"Whisper install strategy {si+1}: {strat['label']}")
            if "pre" in strat:
                strat["pre"]()

            try:
                package_store.install(
                    strat["reqs"],
                    modules=[strat["module"]],
                    pip_args=strat.get("pip_args", ()),
                    timeout=600,
                    report=lambda line, pct=pct_base: _update_job(job_id, progress=pct + 10, message=line),
                    on_process=lambda proc: _register_job_process(job_id, proc),
                )
            except package_store.StoreBusy:
                raise
            except RuntimeError as e:
                last_error = str(e)
                logger.warning(f"Strategy {si+1} failed: {last_error[-300:]}")
                continue
            finally:
                _unregister_job_process(job_id)

            actual_backend = strat.get("backend_name", backend)
            logger.info(f"Whisper installed via strategy {si+1}: {actual_backend}")
            return {"backend": actual_backend, "installed": True}

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

        # All strategies failed
        # Provide a helpful error message about Rust/tokenizers
        helpful = last_error
        if "metadata-generation-failed" in helpful or "rust" in helpful.lower() or "tokenizers" in helpful.lower():
            helpful = (
                "Could not install Whisper. The 'tokenizers' package needs either "
                "a pre-built wheel for your Python version or Rust to compile from source.\n\n"
                "Try one of these:\n"
                "1. Update Python to 3.10-3.12 (pre-built wheels available)\n"
                "2. Install Rust: https://rustup.rs/\n"
                "3. Run manually: pip install openai-whisper\n\n"
                f"Last error:\n{last_error[-200:]}"
            )

        logger.error(f"All whisper install strategies failed. Last error: {last_error[:500]}")
        raise RuntimeError(helpful)
    finally:
        _unregister_job_process(job_id)

# ---------------------------------------------------------------------------
# Whisper Settings & Reinstall
# ---------------------------------------------------------------------------
@system_bp.route("/whisper/settings", methods=["GET", "POST"])
@require_csrf
def whisper_settings():
    """Get or update Whisper settings (CPU mode, default model)."""
    if request.method == "GET":
        settings = load_whisper_settings()
        return jsonify(settings)

    # POST - update settings
    data = get_json_dict() if request.data else {}
    settings = load_whisper_settings()

    if "cpu_mode" in data:
        settings["cpu_mode"] = safe_bool(data["cpu_mode"], False)
    if "model" in data:
        _m = str(data["model"])
        if _m in VALID_WHISPER_MODELS:
            settings["model"] = _m

    save_whisper_settings(settings)
    forget_capability_facts()  # /health reports cpu_mode as whisper_cpu_mode.
    logger.info(f"Whisper settings updated: {settings}")

    return jsonify({"success": True, "settings": settings})


@system_bp.route("/whisper/clear-cache", methods=["POST"])
@require_csrf
def whisper_clear_cache():
    """Clear downloaded Whisper model cache."""
    data = get_json_dict() if request.data else {}
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    cache_plan = _build_whisper_cache_plan()
    plan = build_destructive_plan(
        "whisper.clear_cache",
        targets=cache_plan["entries"],
        metadata={"route": "/whisper/clear-cache", "total_bytes": cache_plan["total_bytes"]},
        reversible=False,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": cache_plan["entries"],
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "total_bytes": cache_plan["total_bytes"],
            "cleared": [],
            "errors": cache_plan["errors"],
            "message": f"Previewed {len(cache_plan['entries'])} cache location(s)",
        })
    if plan["targets"] and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409

    cleared = []
    # The plan entries keep their sizing errors, and those don't fail a clear.
    errors = list(cache_plan["listing_errors"])
    for entry in plan["targets"]:
        path = str(entry.get("path", ""))
        deleted, error = _delete_cache_target(path)
        if deleted:
            cleared.append(path)
        elif error:
            errors.append(error)

    return jsonify({
        "success": len(errors) == 0,
        "dry_run": False,
        "plan": cache_plan["entries"],
        "destructive_plan": plan,
        "total_bytes": cache_plan["total_bytes"],
        "cleared": cleared,
        "errors": errors,
        "message": f"Cleared {len(cleared)} cache location(s)"
    })


@system_bp.route("/whisper/reinstall", methods=["POST"])
@require_csrf
@async_job("reinstall-whisper", filepath_required=False, rate_limit_key="model_install")
def whisper_reinstall(job_id, filepath, data):
    """Complete Whisper reinstall: uninstall, clear cache, reinstall fresh."""
    backend = data.get("backend", "faster-whisper")
    allowed_backends = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed_backends:
        raise ValueError(f"Unknown backend: {backend}")

    cpu_mode = safe_bool(data.get("cpu_mode", False), False)

    try:
        import subprocess as _sp

        from opencut.core import package_store
        from opencut.security import install_python, runtime_security_requirement

        # Refuse a backend the dependency policy blocks before anything is removed.
        runtime_security_requirement(backend)
        _pip_python = install_python()
        frozen = bool(getattr(sys, "frozen", False))

        # Step 1: Uninstall existing packages
        _update_job(job_id, progress=5, message="Uninstalling existing Whisper packages...")
        logger.info("Reinstall: Uninstalling existing packages")

        # A packaged server never loads the system Python's copies, so they
        # belong to the user; the store install below replaces OpenCut's own.
        uninstall_pkgs = [] if frozen else ["faster-whisper", "openai-whisper", "whisperx"]
        for pkg in uninstall_pkgs:
            with suppress(Exception):
                _sp.run(
                    [_pip_python, "-m", "pip", "uninstall", pkg, "-y"],
                    capture_output=True, timeout=60, check=False
                )

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

        # Step 2: Clear model cache
        _update_job(job_id, progress=20, message="Clearing model cache...")
        logger.info("Reinstall: Clearing cache")

        # The same targets Clear cache shows: only the whisper repos in the
        # Hub cache, wherever HF_HUB_CACHE or HF_HOME put it.
        for entry in _build_whisper_cache_plan()["entries"]:
            removed, error = _delete_cache_target(entry["path"])
            if not removed:
                logger.warning("Reinstall: couldn't clear %s", error)

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

        # Step 3: Clear pip cache for these packages
        _update_job(job_id, progress=30, message="Clearing pip cache...")
        with suppress(Exception):
            _sp.run(
                [_pip_python, "-m", "pip", "cache", "remove", "faster_whisper"],
                capture_output=True, timeout=30, check=False
            )
            _sp.run(
                [_pip_python, "-m", "pip", "cache", "remove", "ctranslate2"],
                capture_output=True, timeout=30, check=False
            )

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

        # Step 4: Install fresh through the package store. keep_shared=False
        # replaces the dependencies the store already has, too.
        _update_job(job_id, progress=40, message=f"Installing {backend} fresh...")
        logger.info(f"Reinstall: Installing {backend}")

        if backend == "faster-whisper":
            reqs = ["faster-whisper"]
            if cpu_mode:
                _update_job(job_id, progress=45, message="Installing CPU-optimized version...")
                reqs.append("ctranslate2")
            module = "faster_whisper"
        else:
            reqs = [backend]
            module = "whisper" if backend == "openai-whisper" else backend

        try:
            package_store.install(
                reqs,
                modules=[module],
                pip_args=["--no-cache-dir"],
                keep_shared=False,
                timeout=600,
                report=lambda line: _update_job(job_id, progress=60, message=line),
                on_process=lambda proc: _register_job_process(job_id, proc),
            )
        except package_store.PipFailed as exc:
            if _is_cancelled(job_id):
                return {"backend": backend, "installed": False, "cancelled": True}
            if frozen:
                raise
            raise RuntimeError(
                f"{exc} Try running as administrator or: pip install {backend} --user --force-reinstall"
            ) from exc
        finally:
            _unregister_job_process(job_id)

        # Step 5: Save CPU mode setting
        _update_job(job_id, progress=85, message="Saving settings...")
        settings = load_whisper_settings()
        settings["cpu_mode"] = cpu_mode
        save_whisper_settings(settings)
        forget_capability_facts()

        # The store already had a child server import the module.
        logger.info(f"Whisper reinstalled: {backend}, cpu_mode={cpu_mode}")
        return {"backend": backend, "cpu_mode": cpu_mode, "installed": True}

    finally:
        _unregister_job_process(job_id)
