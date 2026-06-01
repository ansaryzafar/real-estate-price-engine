FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the install step.
# The install only re-runs when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/
COPY config/ config/

# ── Model artefacts ───────────────────────────────────────────────────────────
# The trained model and preprocessor are baked into the image at build time.
# For production, consider mounting these as a volume or pulling from a model
# registry (MLflow, S3) at startup instead.
COPY models/ models/
COPY data/processed/preprocessor.pkl data/processed/preprocessor.pkl
COPY data/processed/feature_names.json data/processed/feature_names.json

# ── Non-root user (security best practice) ───────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Use 1 worker for a single-container deployment. Scale out with multiple
# container replicas behind a load balancer (e.g., Kubernetes, ECS) rather
# than increasing workers per container — avoids memory duplication of the
# model across workers in the same process.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
