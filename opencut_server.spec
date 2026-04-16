# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for OpenCut Server

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all opencut submodules (lazy imports in route handlers)
opencut_hiddenimports = collect_submodules('opencut')

# External deps that are lazily imported inside route handlers
external_hiddenimports = [
    'faster_whisper',
    'ctranslate2',
    'huggingface_hub',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'numpy',
    'librosa',
    'pydub',
    'noisereduce',
    'deep_translator',
    'scenedetect',
    'flask',
    'flask_cors',
    'click',
    'rich',
    'soundfile',
    'tokenizers',
    'sentencepiece',
    # v1.3.0 additions
    'mediapipe',
    'auto_editor',
    'transnetv2',
    'resemble_enhance',
]

# Filter to only actually installed packages
valid_imports = []
for mod in external_hiddenimports:
    try:
        __import__(mod)
        valid_imports.append(mod)
    except ImportError:
        pass

all_hiddenimports = opencut_hiddenimports + valid_imports

# Collect runtime JSON data and native DLLs for optional backends.
extra_datas = collect_data_files('opencut.data')
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
