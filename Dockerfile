# Serving image. Expects a trained model in models/ (run `make train` first);
# in a real deployment the model would instead be pulled from object
# storage or a model registry at build or startup time.
FROM python:3.12-slim

# Run as an unprivileged user; created first so COPY can assign ownership.
RUN useradd --create-home appuser

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --chown keeps the files readable regardless of the host's permission bits.
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/

USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
