# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for OpenCut Server

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all opencut submodules (lazy imports in route handlers)
opencut_hiddenimports = collect_submodules('opencut')

# External deps that are lazily imported inside route handlers.
#
# This list used to be hand-maintained and had drifted from reality: the
# source lazily imports ~90 optional modules while the list named 25, and it
# still carried names the code no longer uses while missing ones it does
# (``transnetv2_pytorch``). Derive it from the source instead, so a new
# optional backend is bundled without anyone remembering to edit this file.
#
# Every optional backend is reached through ``helpers._try_import("name")``,
# which makes that call the authoritative record of what may need bundling.
import ast
import re as _re

_SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
_OPENCUT_DIR = os.path.join(_SPEC_DIR, "opencut")

# Framework and first-party runtime deps that are imported normally rather
# than through _try_import, so the scan below would not see them.
_ALWAYS_BUNDLE = [
    'flask',
    'flask_cors',
    'click',
    'rich',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
]


def _discover_lazy_imports(root):
    """Collect every module name passed to helpers._try_import() in the tree."""
    found = set()
    pattern = _re.compile("_try_import\\(\\s*[\"']([A-Za-z0-9_.]+)[\"']")
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                    source = handle.read()
            except OSError:
                continue
            found.update(pattern.findall(source))
    return found


try:
    external_hiddenimports = sorted(
        set(_ALWAYS_BUNDLE) | _discover_lazy_imports(_OPENCUT_DIR)
    )
except Exception:
    # A frozen build must never fail because discovery hit something odd;
    # fall back to the framework set, which the app cannot start without.
    external_hiddenimports = list(_ALWAYS_BUNDLE)

# Filter to only actually installed packages
valid_imports = []
for mod in external_hiddenimports:
    try:
        __import__(mod)
        valid_imports.append(mod)
    except Exception:
        # ImportError is the common case, but a half-installed optional
        # backend can raise anything at import time; never fail the build.
        pass

all_hiddenimports = opencut_hiddenimports + valid_imports

# Collect runtime JSON data and native DLLs for optional backends.
#
# ``collect_data_files`` is per-subpackage, so naming ``opencut.data`` alone
# silently shipped a build with no ``opencut/_generated`` at all: no build
# error, no import error, a green suite (which only ever runs against the
# source tree), and sixteen manifests missing from the artifact users install.
# Derive the data-bearing subpackages from the source instead, the same way
# the hidden-import list above is derived, so a new one cannot be forgotten.
def _opencut_data_subpackages():
    # The spec is built from the repo root (pathex=['.'], server.py is joined
    # relative), so resolve 'opencut' the same way the rest of this file does.
    root = os.path.abspath('opencut')
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        if '__init__.py' not in filenames:
            continue
        if not any(not name.endswith(('.py', '.pyc')) for name in filenames):
            continue
        rel = os.path.relpath(dirpath, os.path.dirname(root))
        found.append(rel.replace(os.sep, '.'))
    return sorted(found)


extra_datas = []
for _pkg in _opencut_data_subpackages():
    extra_datas += collect_data_files(_pkg)
for pkg in ['ctranslate2', 'faster_whisper']:
    try:
        extra_datas += collect_data_files(pkg)
    except Exception:
        pass

a = Analysis(
    [os.path.join('opencut', 'server.py')],
    pathex=['.'],
    binaries=[],
    datas=extra_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional deps that aren't installed
        'torch', 'torchaudio', 'torchvision',
        'demucs', 'audiocraft',
        'realesrgan', 'gfpgan', 'insightface', 'rembg',
        'onnxruntime', 'onnxruntime_gpu',
        'pyannote', 'whisperx',
        'pedalboard', 'edge_tts', 'kokoro',
        # Exclude dev/test stuff
        'pytest', 'ruff', 'black', 'mypy',
        'tkinter', '_tkinter',
        'matplotlib',
    ],
    noarchive=False,
)

# The Windows OpenCV wheel carries a prebuilt FFmpeg plugin whose ABI predates
# the CVE-2026-8461 floor. OpenCut disables that backend before cv2 import and
# omits the DLL from the frozen payload. Media Foundation remains available.
def _is_opencv_ffmpeg_plugin(entry):
    return any('opencv_videoio_ffmpeg' in str(value).lower() for value in entry[:2])


if sys.platform == 'win32':
    a.binaries = [entry for entry in a.binaries if not _is_opencv_ffmpeg_plugin(entry)]
    a.datas = [entry for entry in a.datas if not _is_opencv_ffmpeg_plugin(entry)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OpenCut-Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=os.path.join('img', 'logo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpenCut-Server',
)
