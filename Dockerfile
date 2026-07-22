# ============================================================
# OpenCut Server — Docker Image
# Multi-stage build for minimal final image
#
# Build:       docker build -t opencut .
# Run (CPU):   the image bakes OPENCUT_ALLOW_REMOTE=1, so the server requires a
#              mounted auth-token file and exits otherwise (fail-closed):
#                docker run -d -p 5679:5679 \
#                  -e OPENCUT_REMOTE_AUTH_TOKEN_FILE=/run/secrets/opencut-token \
#                  -v ./opencut-token:/run/secrets/opencut-token:ro \
#                  -v opencut-data:/home/opencut/.opencut opencut
# Run (GPU):   add --gpus all to the command above
# Compose:     docker compose up opencut-server (mounts the same token via the
#              opencut_remote_auth_token secret; see docker-compose.yml)
# Runtime:     publishes the HTTP API only. Optional WebSocket/MCP sidecars
#              are separate opt-in processes, not default container ports.
# ============================================================

# Stage 0: build the exact CVE-floor release from upstream source. Debian's
# distro package can lag the patched point release, so it is never accepted as
# the container's media runtime.
ARG FFMPEG_VERSION=8.1.2
ARG FFMPEG_SHA256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
FROM python:3.12-slim-bookworm AS ffmpeg-build

ARG FFMPEG_VERSION
ARG FFMPEG_SHA256

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libass-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    nasm \
    pkg-config \
    xz-utils \
    yasm \
    && curl -fsSLo /tmp/ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum --check --strict \
    && mkdir /tmp/ffmpeg-src \
    && tar --extract --xz --file /tmp/ffmpeg.tar.xz --directory /tmp/ffmpeg-src --strip-components=1 \
    && cd /tmp/ffmpeg-src \
    && ./configure \
        --prefix=/opt/ffmpeg \
        --disable-debug \
        --disable-doc \
        --disable-ffplay \
        --enable-gpl \
        --enable-version3 \
        --enable-gnutls \
        --enable-libass \
        --enable-libfontconfig \
        --enable-libfreetype \
        --enable-libfribidi \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libvpx \
        --enable-libx264 \
        --enable-libx265 \
    && make -j"$(nproc)" \
    && make install

# Stage 1: Base with Python and the matching runtime libraries.
FROM python:3.12-slim-bookworm AS base

# Development package names are stable across the pinned Debian base and pull
# the corresponding runtime libraries used by the upstream-built binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libass-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libsndfile1 \
    # libgl1 replaces libgl1-mesa-glx in Debian 12+ (bookworm). The old
    # virtual package was dropped and pulling it fails the apt step on
    # newer base images. libgl1 satisfies OpenCV's GL dependency.
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build /opt/ffmpeg /opt/ffmpeg
ENV PATH="/opt/ffmpeg/bin:${PATH}"
RUN ffmpeg -version | grep -F "ffmpeg version 8.1.2" \
    && ffprobe -version | grep -F "ffprobe version 8.1.2"

WORKDIR /app

# Stage 2: Install Python dependencies
FROM base AS deps

COPY requirements.txt requirements-release-lock.txt ./

# Keep the container dependency layer on the committed Python install surface.
# This prevents retired packages from re-entering through Docker and lets pip
# failures fail the image build instead of producing a partial runtime.
RUN python -m pip install --no-cache-dir --require-hashes --requirement requirements-release-lock.txt

# Stage 3: Final image
FROM base AS final

ARG FFMPEG_VERSION
ARG FFMPEG_SHA256

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Project-owned provenance gate records CVE-2026-8461 and both upstream fix
# commits and fails the image build if the resolved binary is ever downgraded.
RUN python scripts/verify_ffmpeg_provenance.py /opt/ffmpeg/bin/ffmpeg \
    --ffprobe /opt/ffmpeg/bin/ffprobe \
    --release \
    --source-url "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    --source-sha256 "${FFMPEG_SHA256}" \
    --build-origin "OpenCut Dockerfile upstream source build for FFmpeg ${FFMPEG_VERSION}" \
    --corresponding-source "Download the archive at source.url, verify source.sha256, then run the recorded configuration with the Debian Bookworm development libraries listed in Dockerfile." \
    --manifest /tmp/ffmpeg-provenance.json \
    && python scripts/release_composition.py \
        --lane linux \
        --artifact /app/opencut \
        --artifact /opt/ffmpeg/bin/ffmpeg \
        --artifact /opt/ffmpeg/bin/ffprobe \
        --ffmpeg-provenance /tmp/ffmpeg-provenance.json \
        --output-dir /app/release-metadata

# Create a stable non-root identity and a private runtime-secret directory.
# The fixed IDs make Compose secret ownership deterministic across hosts.
RUN groupadd --gid 10001 opencut && useradd --uid 10001 --gid 10001 -d /home/opencut -m opencut \
    && mkdir -p /home/opencut/.opencut/packages /home/opencut/.opencut/models /home/opencut/.opencut/plugins /run/opencut-secrets \
    && chmod 0700 /run/opencut-secrets \
    && chown -R opencut:opencut /home/opencut /app
RUN chown opencut:opencut /run/opencut-secrets \
    && chmod 0755 /app/scripts/docker-entrypoint.sh

# Expose the HTTP API. WebSocket and MCP sidecars are not published by default.
EXPOSE 5679

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5679/health || exit 1

# Environment
ENV OPENCUT_BUNDLED=false
ENV PYTHONUNBUFFERED=1
ENV OPENCUT_HOST=0.0.0.0
ENV OPENCUT_PORT=5679
ENV OPENCUT_ALLOW_REMOTE=1
ENV HOME=/home/opencut

# Run as non-root
USER opencut

# Entrypoint validates and normalizes a mounted Compose secret without ever
# copying its value into an environment variable or image layer.
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python", "-m", "opencut.server", "--host", "0.0.0.0", "--port", "5679"]
