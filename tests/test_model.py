import pytest

from src.model import MODEL_REGISTRY, create_model, load_model
from src.preprocessing import LABELS

TRAIN_TEXTS = [
    "absolutely loved this book",
    "wonderful story, great characters",
    "best read of the year",
    "terrible book, waste of money",
    "awful writing, hated it",
    "worst purchase ever",
    "it was okay, nothing special",
    "average story, fine I guess",
    "mediocre but readable",
] * 3
TRAIN_LABELS = (["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3) * 3


@pytest.fixture
def trained_model():
    model = create_model("tfidf_logreg", seed=0)
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return model


def test_create_model_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("does_not_exist")


def test_registry_contains_default_model():
    assert "tfidf_logreg" in MODEL_REGISTRY


def test_predict_returns_known_labels(trained_model):
    predictions = trained_model.predict(["loved it", "hated it"])
    assert len(predictions) == 2
    assert set(predictions) <= set(LABELS)


def test_predict_proba_shape_and_normalization(trained_model):
    probabilities = trained_model.predict_proba(["loved it", "hated it", "okay"])
    assert probabilities.shape == (3, len(LABELS))
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])


def test_save_and_load_roundtrip(trained_model, tmp_path):
    path = tmp_path / "model.joblib"
    trained_model.save(path)
    restored = load_model(path)
    texts = ["absolutely loved this book", "terrible book"]
    assert restored.predict(texts) == trained_model.predict(texts)
    assert restored.name == trained_model.name
