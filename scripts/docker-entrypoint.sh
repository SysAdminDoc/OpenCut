#!/bin/sh
set -eu

# Compose file-backed secrets are commonly mounted 0444 even when a stricter
# mode is requested. Copy only at container start into an owner-only runtime
# directory; the Python backend then performs its own lstat/open validation.
source_path="${OPENCUT_REMOTE_AUTH_TOKEN_FILE:-}"
if [ -n "$source_path" ]; then
    if [ -L "$source_path" ] || [ ! -f "$source_path" ]; then
        echo "OpenCut auth secret must be a regular non-symlink file." >&2
        exit 78
    fi
    runtime_path="/run/opencut-secrets/remote-auth-token"
    if [ "$source_path" != "$runtime_path" ]; then
        umask 077
        cp --no-dereference "$source_path" "$runtime_path"
        chmod 0400 "$runtime_path"
        export OPENCUT_REMOTE_AUTH_TOKEN_FILE="$runtime_path"
    fi
fi

exec "$@"
