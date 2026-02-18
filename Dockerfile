# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for document parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Set Hugging Face cache directory to a writable location
ENV HF_HOME=/app/model_cache

# Create non-root user FIRST
RUN useradd --create-home appuser

# Create directory and set permissions
RUN mkdir -p /app/model_cache /app/chroma_db /app/uploads && \
    chown -R appuser:appuser /app

# Download model during build to avoid runtime rate limits
COPY scripts/download_model.py /app/scripts/
RUN python /app/scripts/download_model.py && \
    # Ensure appuser owns the cache (again, to be safe)
    chown -R appuser:appuser /app/model_cache

USER appuser

# Expose port for Cloud Run
EXPOSE 8000

# Health check (dynamic port)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8080'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Run FastAPI server (respect PORT env var, default 8080 for Cloud Run)
CMD sh -c "uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8080}"
