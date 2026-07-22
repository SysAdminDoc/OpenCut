#!/usr/bin/env python3
"""
OpenCut Installer
Cross-platform setup script. Run: python install.py

Version is read from VERS below — kept in sync with opencut/__init__.py
by ``scripts/sync_version.py``.
"""

import os
import platform
import shutil
import subprocess
import sys

VERS = "1.42.1"
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 14)
CEP_EXT = "com.opencut.panel"
WIN_CEP_DIR = os.path.expandvars(r"%APPDATA%\Adobe\CEP\extensions")
MAC_CEP_DIR = os.path.expanduser("~/Library/Application Support/Adobe/CEP/extensions")
LINUX_CEP_DIR = os.path.expanduser("~/.local/share/Adobe/CEP/extensions")


def banner():
    print()
    print("  ============================================")
    print(f"  OpenCut Installer v{VERS}")
    print("  ============================================")
    print()


def check_python():
    v = sys.version_info
    detected = f"{v.major}.{v.minor}.{v.micro}"
    required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"
    if not (MIN_PYTHON <= v[:2] <= MAX_PYTHON):
        print(f"  [!!] Detected Python {detected}; OpenCut requires Python {required}.")
        print("       Install a supported version from https://www.python.org/downloads/")
        print("       Windows: winget install Python.Python.3.12")
        sys.exit(1)
    print(f"  [OK] Python {detected} satisfies the required Python {required}")


def check_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("  [!!] FFmpeg not found on PATH.")
        print("       Install FFmpeg 8.1.2+ from https://ffmpeg.org/download.html")
        sys.exit(1)

    from opencut.core.ffmpeg_provenance import probe_binary_security

    grade = probe_binary_security(ffmpeg, timeout=5)
    if not grade.get("ok"):
        print(f"  [!!] FFmpeg blocked: {grade.get('version') or 'unknown'}")
        print(f"       {grade.get('reason')}")
        print("       CVE-2026-8461 requires FFmpeg 8.1.2+ or a dated post-fix snapshot.")
        sys.exit(1)

    print(
        f"  [OK] FFmpeg {grade.get('version')} clears the security floor "
        f"({grade.get('lane')} lane)"
    )


def install_deps():
    print("\n  Installing Python dependencies...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, "requirements.txt")
    release_lock = os.path.join(base_dir, "requirements-release-lock.txt")

    # 30-min timeout: enough for slow networks downloading torch/faster-whisper,
    # but bounded so a hung pip mirror doesn't freeze the installer forever.
    pip_timeout = 1800
    # ``--quiet`` was removed — torch + faster-whisper take 5–15 minutes on
    # cold installs and a silent pip looks like a hang to the user.
    # ``--progress-bar on`` keeps the visual feedback even when stdout is
    # piped through subprocess.
    try:
        if os.path.exists(release_lock):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", release_lock,
                 "--prefer-binary", "--progress-bar", "on"],
                check=True, timeout=pip_timeout,
            )
        elif os.path.exists(req_file):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_file,
                 "--prefer-binary", "--progress-bar", "on"],
                check=True, timeout=pip_timeout,
            )
        else:
            lock_file = os.path.join(base_dir, "requirements-lock.txt")
            fallback_req = lock_file if os.path.exists(lock_file) else None
            if fallback_req:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", fallback_req,
                     "--prefer-binary", "--progress-bar", "on"],
                    check=True, timeout=pip_timeout,
                )
            else:
                print("  [!!] No dependency lock or requirements file was found.")
                print("       Re-download the complete OpenCut release and retry.")
                sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"  [!!] pip install timed out after {pip_timeout // 60} minutes.")
        print("       Check your network connection and try again.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"  [!!] pip install failed (exit code {exc.returncode}).")
        sys.exit(1)
    print("  [OK] Dependencies installed")


def install_cep_extension():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ext_src = os.path.join(base_dir, "extension", CEP_EXT)

    if not os.path.isdir(ext_src):
        print("  [!!] Extension source not found, skipping CEP install")
        return

    system = platform.system()
    if system == "Windows":
        cep_dir = WIN_CEP_DIR
    elif system == "Darwin":
        cep_dir = MAC_CEP_DIR
    else:
        cep_dir = LINUX_CEP_DIR

    dest = os.path.join(cep_dir, CEP_EXT)

    os.makedirs(cep_dir, exist_ok=True)

    if os.path.exists(dest):
        shutil.rmtree(dest)
        print("  [OK] Removed previous extension install")

    shutil.copytree(ext_src, dest)
    print(f"  [OK] Extension installed to: {dest}")

    if system == "Windows":
        enable_unsigned_extensions_windows()


def enable_unsigned_extensions_windows():
    """Set PlayerDebugMode registry key for unsigned CEP extensions."""
    try:
        import winreg
        # Adobe CEP reads HKCU\Software\Adobe\CSXS.<integer> (e.g. CSXS.11) —
        # dotted keys like "CSXS.11.0" are never consulted and the unsigned
        # panel silently fails to load. Cover CSXS 7 (CC 2014) through 18
        # (PPro 2025+), mirroring Install.ps1 and OpenCut.iss.
        versions = [str(v) for v in range(7, 19)]
        key_base = r"SOFTWARE\Adobe\CSXS"
        count = 0
        for ver in versions:
            try:
                key_path = f"{key_base}.{ver}"
                key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_WRITE)
                winreg.SetValueEx(key, "PlayerDebugMode", 0, winreg.REG_SZ, "1")
                winreg.CloseKey(key)
                count += 1
            except Exception:
                pass
        if count:
            print(f"  [OK] PlayerDebugMode set for {count} CSXS versions")
        else:
            print("  [!!] Could not set PlayerDebugMode (try running as admin)")
    except ImportError:
        pass


def create_launcher():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()

    if system == "Windows":
        launcher = os.path.join(base_dir, "Start-OpenCut.bat")
        # ``encoding="utf-8"`` so non-ASCII install paths don't corrupt the
        # generated launcher under Windows' system locale (cp1252).
        with open(launcher, "w", encoding="utf-8") as f:
            f.write("@echo off\n")
            f.write("echo.\n")
            f.write(f"echo   OpenCut Server v{VERS}\n")
            f.write("echo   ========================\n")
            f.write("echo   Starting on http://localhost:5679\n")
            f.write("echo   Close this window to stop.\n")
            f.write("echo.\n")
            f.write(f'"{sys.executable}" -m opencut.server\n')
            f.write("pause\n")
        print(f"  [OK] Launcher created: {launcher}")
    else:
        launcher = os.path.join(base_dir, "start-opencut.sh")
        with open(launcher, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write(f'echo "OpenCut Server v{VERS}"\n')
            f.write('echo "Starting on http://localhost:5679"\n')
            f.write(f'"{sys.executable}" -m opencut.server\n')
        os.chmod(launcher, 0o755)
        print(f"  [OK] Launcher created: {launcher}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # The PyTorch attribute is `total_memory`, not `total_mem` —
            # the latter raises AttributeError and the GPU was reported as
            # "unavailable" instead of detected.
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [OK] CUDA GPU: {name} ({vram:.1f}GB VRAM)")
        else:
            print("  [--] No CUDA GPU detected. AI features will use CPU (slower).")
            print("       For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  [--] PyTorch not installed. AI features unavailable until installed.")
    except Exception as exc:
        print(f"  [--] Could not query GPU: {exc}")


def verify():
    print("\n  Verifying installation...")
    errors = []

    try:
        import flask
        print(f"  [OK] Flask {flask.__version__}")
    except ImportError:
        errors.append("flask")

    try:
        from faster_whisper import WhisperModel  # noqa: F401 — import probes availability
        print("  [OK] faster-whisper")
    except ImportError:
        print("  [--] faster-whisper not available (captions won't work)")

    try:
        import cv2
        print(f"  [OK] OpenCV {cv2.__version__}")
    except ImportError:
        print("  [--] OpenCV not available (some video effects limited)")

    if errors:
        print(f"\n  [!!] Missing critical deps: {', '.join(errors)}")
        return False

    print("\n  [OK] Installation verified successfully!")
    return True


def main():
    banner()
    check_python()
    check_ffmpeg()
    install_deps()
    install_cep_extension()
    create_launcher()
    check_gpu()
    verify()

    print()
    print("  ============================================")
    print("  Installation complete!")
    print("  ============================================")
    print()
    print("  Next steps:")
    print("  1. Run Start-OpenCut.bat (or: python -m opencut.server)")
    print("  2. Open Premiere Pro > Window > Extensions > OpenCut")
    print("  3. Select a clip and start editing!")
    print()


if __name__ == "__main__":
    main()
