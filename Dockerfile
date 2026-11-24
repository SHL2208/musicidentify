# ==============================================================================
# Dockerfile: AI Music Detector Agent (Production Multi-Stage Build with uv)
# ==============================================================================

# Stage 1: Get official Deno standalone binary (for yt-dlp JavaScript challenges)
FROM denoland/deno:bin AS deno-bin

# Stage 2: Final runtime container
FROM python:3.11-slim-bookworm

# Set metadata
LABEL maintainer="AI Music Detector Team"
LABEL description="AI Music Detector Agent - Forensic Audio Analysis & Multi-tier AI Classification"

# Install official uv binary from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Install required system packages:
# - ffmpeg: Audio extraction, slicing, and conversion (yt-dlp & audio engine)
# - libsndfile1: Required by soundfile & librosa for WAV/audio I/O
# - curl: For container healthchecks and network utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Deno binary from stage 1 (enables yt-dlp to decrypt YouTube JS challenges)
COPY --from=deno-bin /deno /usr/local/bin/deno
RUN chmod +x /usr/local/bin/deno

# Set working directory
WORKDIR /app

# Install dependencies using uv and uv.lock for 100% reproducible and ultra-fast builds
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-cache

# Copy application source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY data/catalogs/ ./data/catalogs/
COPY *.py ./

# Ensure data directories exist for runtime caches and outputs
RUN mkdir -p /app/data/temp /app/data/outputs /app/data/audio_cache /app/data/catalogs

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck for either Streamlit UI or FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || curl -fsS http://localhost:8000/health || exit 1

# Default command: Start Streamlit Web UI
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
