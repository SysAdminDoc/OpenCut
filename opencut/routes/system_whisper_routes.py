"""System whisper routes registered on the shared system blueprint."""

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
    get_json_dict,
    jsonify,
    load_whisper_settings,
    logger,
    os,
    request,
    require_csrf,
    safe_bool,
    save_whisper_settings,
    shutil,
    suppress,
    sys,
    system_bp,
    threading,
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

    try:
        import subprocess as _sp

        _update_job(job_id, progress=5, message=f"Installing {backend}...")
        logger.info(f"Starting Whisper install: {backend}")

        # Resolve Python executable (frozen builds can't use sys.executable for pip)
        from opencut.security import _find_system_python
        if getattr(sys, "frozen", False):
            _pip_python = _find_system_python() or sys.executable
        else:
            _pip_python = sys.executable

        # Permission fallback: --target to a writable directory
        _target_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
        os.makedirs(_target_dir, exist_ok=True)
        if _target_dir not in sys.path:
            sys.path.append(_target_dir)

        # Build pip commands with permission fallbacks baked in
        def _pip_cmd(pkg, *extra_flags):
            """Return list of pip commands: normal, --user, --target."""
            base = [_pip_python, "-m", "pip", "install"] + list(extra_flags) + [pkg]
            return [
                base + ["--progress-bar", "on"],
                base + ["--user", "--progress-bar", "on"],
                base + ["--target", _target_dir, "--progress-bar", "on"],
            ]

        def _pip_pre(pkg, *extra_flags):
            """Return list of pre-install commands with fallbacks."""
            base = [_pip_python, "-m", "pip", "install"] + list(extra_flags) + [pkg, "--quiet"]
            return [
                base,
                base + ["--user"],
                base + ["--target", _target_dir],
            ]

        # Strategy list - try each in order until one works
        strategies = []
        if backend == "faster-whisper":
            strategies = [
                {
                    "label": "Install tokenizers wheel first, then faster-whisper",
                    "pre_cmds": _pip_pre("tokenizers", "--only-binary", "tokenizers"),
                    "cmds": _pip_cmd("faster-whisper"),
                },
                {
                    "label": "Upgrade pip + prefer binary wheels",
                    "pre_cmds": _pip_pre("pip setuptools wheel", "--upgrade"),
                    "cmds": _pip_cmd("faster-whisper", "--prefer-binary"),
                },
                {
                    "label": "Pin older tokenizers with wheel support",
                    "pre_cmds": _pip_pre("tokenizers>=0.13,<0.20", "--only-binary", "tokenizers"),
                    "cmds": _pip_cmd("faster-whisper"),
                },
                {
                    "label": "Fallback to openai-whisper (no Rust needed)",
                    "cmds": _pip_cmd("openai-whisper"),
                    "verify": [_pip_python, "-c",
                               "import whisper; print('ok')"],
                    "backend_name": "openai-whisper",
                },
            ]
        else:
            strategies = [
                {
                    "label": "Standard install",
                    "cmds": _pip_cmd(backend),
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

            # Run pre-commands if present (try each fallback until one succeeds)
            if "pre_cmds" in strat:
                for pre_cmd in strat["pre_cmds"]:
                    try:
                        pre_result = _sp.run(
                            pre_cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            check=False,
                        )
                        if pre_result.returncode == 0:
                            logger.debug(f"Pre-command succeeded: {' '.join(pre_cmd[:5])}")
                            break
                        logger.debug(f"Pre-command exit {pre_result.returncode}, trying next fallback")
                    except Exception as e:
                        logger.warning(f"Pre-command failed: {e}")

            # Run main install command (try each permission fallback)
            cmds_to_try = strat.get("cmds", [strat["cmd"]] if "cmd" in strat else [])
            pip_ok = False
            lines = []
            for cmd_variant in cmds_to_try:
                try:
                    proc = _sp.Popen(cmd_variant, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
                    _register_job_process(job_id, proc)

                    # Safety timeout: kill pip if it hangs (10 minutes)
                    _pip_timeout = 600
                    _pip_timer = threading.Timer(_pip_timeout, lambda: proc.kill())
                    _pip_timer.daemon = True
                    _pip_timer.start()

                    lines = []
                    for line in proc.stdout:
                        line = line.strip()
                        if line:
                            lines.append(line)
                            lower = line.lower()
                            if "downloading" in lower:
                                _update_job(job_id, progress=pct_base + 10,
                                            message="Downloading packages...")
                            elif "installing" in lower or "building" in lower:
                                _update_job(job_id, progress=pct_base + 20,
                                            message="Installing packages...")
                            elif "successfully installed" in lower:
                                _update_job(job_id, progress=85,
                                            message="Verifying installation...")

                    _pip_timer.cancel()
                    proc.wait(timeout=30)
                    _unregister_job_process(job_id)
                    logger.debug(f"pip exit code: {proc.returncode}")

                    if proc.returncode == 0:
                        pip_ok = True
                        break  # This permission variant worked
                    else:
                        last_error = "\n".join(lines[-8:])
                        logger.warning(f"Strategy {si+1} cmd variant failed (trying next): {last_error[-200:]}")
                        continue  # Try next permission variant (--user, --target)

                except Exception as e:
                    _unregister_job_process(job_id)
                    last_error = str(e)
                    logger.warning(f"Strategy {si+1} cmd variant exception: {e}")
                    continue  # Try next permission variant

            if not pip_ok:
                logger.warning(f"Strategy {si+1} failed all permission variants")
                continue  # Try next strategy

            # Verify import
            try:
                _update_job(job_id, progress=90, message="Verifying import...")

                verify_cmd = strat.get("verify", None)
                actual_backend = strat.get("backend_name", backend)

                if verify_cmd is None:
                    if actual_backend == "faster-whisper":
                        verify_cmd = [_pip_python, "-c", "from faster_whisper import WhisperModel; print('ok')"]
                    elif actual_backend == "openai-whisper":
                        verify_cmd = [_pip_python, "-c", "import whisper; print('ok')"]
                    else:
                        if actual_backend not in allowed:
                            last_error = f"Unknown backend: {actual_backend}"
                            continue
                        verify_cmd = [_pip_python, "-c", f"import {actual_backend}; print('ok')"]

                # Also try importing directly in-process (for --target installs)
                verify = _sp.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                if verify.returncode == 0 and "ok" in verify.stdout:
                    logger.info(f"Whisper installed via strategy {si+1}: {actual_backend}")
                    return {"backend": actual_backend, "installed": True}
                else:
                    last_error = f"Import failed: {verify.stderr[:300]}"
                    logger.warning(f"Strategy {si+1} verify failed: {last_error}")
                    continue

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Strategy {si+1} verify exception: {e}")
                continue

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
    errors = list(cache_plan["errors"])
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

        # Resolve Python executable (frozen builds can't use sys.executable for pip)
        from opencut.security import _find_system_python
        if getattr(sys, "frozen", False):
            _pip_python = _find_system_python() or sys.executable
        else:
            _pip_python = sys.executable

        _target_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
        os.makedirs(_target_dir, exist_ok=True)
        if _target_dir not in sys.path:
            sys.path.append(_target_dir)

        def _run_pip_with_fallback(args, timeout=300):
            """Try pip command, then --user, then --target fallback."""
            base = [_pip_python, "-m", "pip"] + args
            result = None
            for variant in [base, base + ["--user"], base + ["--target", _target_dir]]:
                try:
                    result = _sp.run(
                        variant,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                    if result.returncode == 0:
                        return result
                except Exception:
                    continue
            return result  # Return last result even if failed

        # Step 1: Uninstall existing packages
        _update_job(job_id, progress=5, message="Uninstalling existing Whisper packages...")
        logger.info("Reinstall: Uninstalling existing packages")

        uninstall_pkgs = ["faster-whisper", "openai-whisper", "whisperx"]
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

        cache_paths = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenCut", "models"),
        ]

        for cache_dir in cache_paths:
            if not cache_dir or not os.path.exists(cache_dir):
                continue
            try:
                if "huggingface" in cache_dir:
                    for item in os.listdir(cache_dir):
                        if "whisper" in item.lower():
                            shutil.rmtree(os.path.join(cache_dir, item), ignore_errors=True)
                else:
                    shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:
                pass

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

        # Step 4: Install fresh (with permission fallbacks)
        _update_job(job_id, progress=40, message=f"Installing {backend} fresh...")
        logger.info(f"Reinstall: Installing {backend}")

        if backend == "faster-whisper":
            install_base = ["install", "faster-whisper", "--force-reinstall", "--no-cache-dir"]

            # For CPU mode, also install CPU-only ctranslate2
            if cpu_mode:
                _update_job(job_id, progress=45, message="Installing CPU-optimized version...")
                _run_pip_with_fallback(
                    ["install", "ctranslate2", "--force-reinstall", "--no-cache-dir"],
                    timeout=300
                )
        else:
            install_base = ["install", backend, "--force-reinstall", "--no-cache-dir"]

        # Try each permission variant for the main install
        install_ok = False
        for install_cmd in [
            [_pip_python, "-m", "pip"] + install_base,
            [_pip_python, "-m", "pip"] + install_base + ["--user"],
            [_pip_python, "-m", "pip"] + install_base + ["--target", _target_dir],
        ]:
            proc = _sp.Popen(install_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
            _register_job_process(job_id, proc)

            for line in proc.stdout:
                if _is_cancelled(job_id):
                    proc.kill()
                    _unregister_job_process(job_id)
                    return {"backend": backend, "installed": False, "cancelled": True}
                line = line.strip().lower()
                if "downloading" in line:
                    _update_job(job_id, progress=55, message="Downloading packages...")
                elif "installing" in line:
                    _update_job(job_id, progress=70, message="Installing packages...")

            try:
                proc.wait(timeout=600)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)
            _unregister_job_process(job_id)

            if proc.returncode == 0:
                install_ok = True
                break
            logger.warning("Reinstall pip variant failed, trying next")

        if not install_ok:
            raise RuntimeError(
                "pip install failed -- permission denied on all attempts. "
                "Try running as administrator or: pip install faster-whisper --user --force-reinstall"
            )

        # Step 5: Save CPU mode setting
        _update_job(job_id, progress=85, message="Saving settings...")
        settings = load_whisper_settings()
        settings["cpu_mode"] = cpu_mode
        save_whisper_settings(settings)

        # Step 6: Verify
        _update_job(job_id, progress=90, message="Verifying installation...")

        if backend == "faster-whisper":
            verify_cmd = [_pip_python, "-c", "from faster_whisper import WhisperModel; print('ok')"]
        else:
            verify_cmd = [_pip_python, "-c", "import whisper; print('ok')"]

        verify = _sp.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if verify.returncode == 0 and "ok" in verify.stdout:
            logger.info(f"Whisper reinstalled: {backend}, cpu_mode={cpu_mode}")
            return {"backend": backend, "cpu_mode": cpu_mode, "installed": True}
        else:
            raise RuntimeError(f"Verification failed: {verify.stderr[:200]}")

    finally:
        _unregister_job_process(job_id)
