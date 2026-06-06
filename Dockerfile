# ============================================================
# OpenCut Server — Docker Image
# Multi-stage build for minimal final image
#
# Build:       docker build -t opencut .
# Run (CPU):   docker run -d -p 5679:5679 -v opencut-data:/home/opencut/.opencut opencut
# Run (GPU):   docker run -d --gpus all -p 5679:5679 -v opencut-data:/home/opencut/.opencut opencut
# Runtime:     publishes the HTTP API only. Optional WebSocket/MCP sidecars
#              are separate opt-in processes, not default container ports.
# ============================================================

# Stage 1: Base with Python and system deps
FROM python:3.12-slim AS base

# System dependencies for FFmpeg, OpenCV, and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    # libgl1 replaces libgl1-mesa-glx in Debian 12+ (bookworm). The old
    # virtual package was dropped and pulling it fails the apt step on
    # newer base images. libgl1 satisfies OpenCV's GL dependency.
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Install Python dependencies
FROM base AS deps

COPY requirements.txt ./

# Keep the container dependency layer on the committed Python install surface.
# This prevents retired packages from re-entering through Docker and lets pip
# failures fail the image build instead of producing a partial runtime.
RUN python -m pip install --no-cache-dir --requirement requirements.txt

# Stage 3: Final image
FROM base AS final

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Create non-root user and data directories
RUN groupadd -r opencut && useradd -r -g opencut -d /home/opencut -m opencut \
    && mkdir -p /home/opencut/.opencut/packages /home/opencut/.opencut/models /home/opencut/.opencut/plugins \
    && chown -R opencut:opencut /home/opencut /app

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
ENV HOME=/home/opencut

# Run as non-root
USER opencut

# Entrypoint
ENTRYPOINT ["python", "-m", "opencut.server"]
CMD ["--host", "0.0.0.0", "--port", "5679"]
