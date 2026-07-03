import pytest
from fastapi.testclient import TestClient

from src.model import create_model
from src.preprocessing import LABELS
from src.service.app import create_app
from tests.test_model import TRAIN_LABELS, TRAIN_TEXTS


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A client backed by a small model trained on synthetic data."""
    model = create_model("tfidf_logreg", seed=0)
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    model_path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    model.save(model_path)
    with TestClient(create_app(model_path=model_path)) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "tfidf_logreg"


def test_predict_returns_per_sentence_predictions(client):
    response = client.post(
        "/predict", json={"text": "Absolutely loved this book. Terrible ending though."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["overall_sentiment"] in LABELS
    assert body["model_name"] == "tfidf_logreg"
    assert len(body["sentences"]) == 2
    for prediction in body["sentences"]:
        assert prediction["sentiment"] in LABELS
        assert 0.0 <= prediction["confidence"] <= 1.0


def test_predict_empty_text_is_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # fails pydantic min_length validation


def test_predict_whitespace_only_text_is_rejected(client):
    response = client.post("/predict", json={"text": "   \n  "})
    assert response.status_code == 422
    assert response.json()["detail"] == "No sentences found in text"


def test_predict_missing_field_is_rejected(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_metrics_reflect_served_requests(client):
    client.post("/predict", json={"text": "A book I truly loved."})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] >= 1
    assert body["latency_window"]["p99_ms"] >= 0
    assert sum(body["prediction_counts"].values()) >= 1
