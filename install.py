#!/usr/bin/env python3
"""
OpenCut Installer v1.0.0-beta
Cross-platform setup script. Run: python install.py
"""

import os
import sys
import shutil
import subprocess
import platform

VERS = "1.0.0-beta"
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
    print(f"  [OK] Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 9):
        print("  [!!] Python 3.9+ required. Please upgrade.")
        sys.exit(1)


def check_ffmpeg():
    if shutil.which("ffmpeg"):
        try:
            r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            ver = r.stdout.split("\n")[0] if r.stdout else "unknown"
            print(f"  [OK] FFmpeg found: {ver[:60]}")
        except Exception:
            print("  [OK] FFmpeg found on PATH")
    else:
        print("  [!!] FFmpeg not found on PATH.")
        print("       Download from https://ffmpeg.org/download.html")
        print("       Or: winget install ffmpeg  |  brew install ffmpeg  |  apt install ffmpeg")
        resp = input("\n  Continue without FFmpeg? (y/n): ").strip().lower()
        if resp != "y":
            sys.exit(1)


def install_deps():
    print("\n  Installing Python dependencies...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(base_dir, "requirements.txt")

    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "click>=8.0", "rich>=13.0", "flask>=3.0", "flask-cors>=4.0",
                               "faster-whisper>=1.0", "opencv-python-headless>=4.8",
                               "Pillow>=10.0", "numpy>=1.24", "--quiet"])
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
        versions = ["11.0", "12.0", "13.0", "14.0", "15.0", "16.0", "17.0", "18.0"]
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
        with open(launcher, "w") as f:
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
        with open(launcher, "w") as f:
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
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  [OK] CUDA GPU: {name} ({vram:.1f}GB VRAM)")
        else:
            print("  [--] No CUDA GPU detected. AI features will use CPU (slower).")
            print("       For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  [--] PyTorch not installed. AI features unavailable until installed.")


def verify():
    print("\n  Verifying installation...")
    errors = []

    try:
        import flask
        print(f"  [OK] Flask {flask.__version__}")
    except ImportError:
        errors.append("flask")

    try:
        from faster_whisper import WhisperModel
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
