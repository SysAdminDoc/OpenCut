#!/bin/sh
set -eu

OPENCUT_HOME=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
export OPENCUT_HOME

if [ -x "$OPENCUT_HOME/python/bin/python3" ]; then
    PYTHON="$OPENCUT_HOME/python/bin/python3"
    PATH="$OPENCUT_HOME/python/bin:$PATH"
elif [ -x "$OPENCUT_HOME/python/bin/python" ]; then
    PYTHON="$OPENCUT_HOME/python/bin/python"
    PATH="$OPENCUT_HOME/python/bin:$PATH"
elif [ -x "$OPENCUT_HOME/python/python" ]; then
    PYTHON="$OPENCUT_HOME/python/python"
    PATH="$OPENCUT_HOME/python:$PATH"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "OpenCut requires Python 3.9 or later, but no python3 or python executable was found." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
    echo "OpenCut requires Python 3.9 or later." >&2
    exit 1
fi

if [ -d "$OPENCUT_HOME/ffmpeg/bin" ]; then
    PATH="$OPENCUT_HOME/ffmpeg/bin:$PATH"
elif [ -d "$OPENCUT_HOME/ffmpeg" ]; then
    PATH="$OPENCUT_HOME/ffmpeg:$PATH"
fi
export PATH

if [ -d "$OPENCUT_HOME/models" ]; then
    export OPENCUT_BUNDLED=true
    export WHISPER_MODELS_DIR="$OPENCUT_HOME/models/whisper"
    export TORCH_HOME="$OPENCUT_HOME/models/demucs"
    export OPENCUT_FLORENCE_DIR="$OPENCUT_HOME/models/florence"
    export OPENCUT_LAMA_DIR="$OPENCUT_HOME/models/lama"
fi

printf "\n OpenCut Server\n ==============\n Backend output appears below. Press Ctrl-C to stop.\n\n"
exec "$PYTHON" -m opencut.server "$@"
