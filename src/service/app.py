"""FastAPI prediction service.

Run locally with:
    uvicorn src.service.app:app --port 8000

Endpoints:
    POST /predict  - sentence-level sentiment for a review text
    GET  /health   - liveness + model-loaded check (cloud readiness probe)
    GET  /metrics  - request counters, latency percentiles, class distribution
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from src import config
from src.model import SentimentModel, load_model
from src.preprocessing import split_sentences
from src.service.monitoring import MetricsTracker
from src.service.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SentencePrediction,
)

logger = logging.getLogger("sentiment_service")


class JsonFormatter(logging.Formatter):
    """Structured logs so a log aggregator can parse fields without regexes."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def create_app(model_path: Path | None = None) -> FastAPI:
    """App factory; ``model_path`` is injectable for tests."""
    _configure_logging(config.LOG_LEVEL)
    resolved_model_path = model_path or config.MODEL_PATH
    metrics = MetricsTracker(window_size=config.METRICS_WINDOW_SIZE)
    state: dict[str, SentimentModel] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Load once at startup so no request pays deserialization cost.
        state["model"] = load_model(resolved_model_path)
        logger.info(
            "model loaded",
            extra={
                "extra_fields": {
                    "model_name": state["model"].name,
                    "model_path": str(resolved_model_path),
                }
            },
        )
        yield
        state.clear()

    app = FastAPI(title="Review Sentiment Service", version="1.0.0", lifespan=lifespan)

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        model = state.get("model")
        if model is None:
            metrics.record_error()
            raise HTTPException(status_code=503, detail="Model not loaded")

        sentences = split_sentences(request.text)
        if not sentences:
            metrics.record_error()
            raise HTTPException(status_code=422, detail="No sentences found in text")

        # One vectorized call for all sentences of the review.
        probabilities = model.predict_proba(sentences)
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = [model.classes[i] for i in predicted_indices]
        overall = model.classes[int(np.mean(probabilities, axis=0).argmax())]

        latency_ms = (time.perf_counter() - started) * 1000
        metrics.record_request(latency_ms, predicted_labels)
        logger.info(
            "prediction served",
            extra={
                "extra_fields": {
                    "n_sentences": len(sentences),
                    "overall_sentiment": overall,
                    "latency_ms": round(latency_ms, 2),
                }
            },
        )
        return PredictResponse(
            overall_sentiment=overall,
            sentences=[
                SentencePrediction(
                    sentence=sentence,
                    sentiment=label,
                    confidence=round(float(probabilities[i][predicted_indices[i]]), 4),
                )
                for i, (sentence, label) in enumerate(zip(sentences, predicted_labels))
            ],
            model_name=model.name,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        model = state.get("model")
        return HealthResponse(
            status="ok" if model else "unavailable",
            model_loaded=model is not None,
            model_name=model.name if model else None,
        )

    @app.get("/metrics")
    def get_metrics() -> dict:
        return metrics.snapshot()

    return app


app = create_app()
