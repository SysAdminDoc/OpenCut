# ============================================================
# OpenCut Server — Docker Image
# Multi-stage build for minimal final image
# ============================================================

# Stage 1: Base with Python and system deps
FROM python:3.12-slim AS base

# System dependencies for FFmpeg, OpenCV, and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Install Python dependencies
FROM base AS deps

COPY requirements.txt pyproject.toml ./
COPY opencut/__init__.py opencut/__init__.py

# Install core dependencies first (cached layer)
RUN pip install --no-cache-dir flask flask-cors click rich

# Install standard optional dependencies
RUN pip install --no-cache-dir \
    faster-whisper \
    opencv-python-headless \
    Pillow \
    numpy \
    librosa \
    pydub \
    noisereduce \
    deep-translator \
    scenedetect[opencv] \
    || echo "Some optional deps failed — continuing"

# Stage 3: Final image
FROM base AS final

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Create data directories
RUN mkdir -p /root/.opencut/packages /root/.opencut/models

# Expose server port
EXPOSE 5679

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5679/health || exit 1

# Environment
ENV OPENCUT_BUNDLED=false
ENV PYTHONUNBUFFERED=1

# Entrypoint
ENTRYPOINT ["python", "-m", "opencut.server"]
CMD ["--host", "0.0.0.0", "--port", "5679"]
