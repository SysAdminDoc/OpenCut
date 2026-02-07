# -*- mode: python ; coding: utf-8 -*-
"""
OpenCut Backend Server - PyInstaller Spec
Bundles the Flask backend into a standalone .exe so users
don't need Python installed.

Build with:  pyinstaller build/opencut.spec
Output:      dist/opencut-server/opencut-server.exe
"""

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Collect all opencut submodules
opencut_hiddenimports = collect_submodules('opencut')

# Core dependencies that PyInstaller often misses
extra_hiddenimports = [
    'flask', 'flask.json', 'flask_cors',
    'click', 'rich', 'rich.console', 'rich.table', 'rich.progress',
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
    'logging.handlers',
    'xml.etree.ElementTree',
    'json', 'uuid', 'threading', 'socket', 'tempfile',
    'urllib.request',
]

# Optional: Whisper support (only included if installed)
try:
    import faster_whisper
    extra_hiddenimports += collect_submodules('faster_whisper')
    extra_hiddenimports += ['ctranslate2', 'huggingface_hub', 'tokenizers']
    print("[OpenCut Build] faster-whisper detected - including in build")
except ImportError:
    print("[OpenCut Build] faster-whisper not found - captions will require separate install")

all_hiddenimports = opencut_hiddenimports + extra_hiddenimports

a = Analysis(
    ['../opencut/server.py'],
    pathex=[os.path.abspath('..')],
    binaries=[],
    datas=[],
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy', 'numpy.testing',
        'pytest', 'setuptools', 'pip', 'wheel',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='opencut-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # Show console window for server output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='opencut-server',
)
