#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
exec /bin/sh "$SCRIPT_DIR/OpenCut-Server.sh" "$@"
