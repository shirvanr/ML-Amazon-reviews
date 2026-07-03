"""Model abstraction and implementations.

The service and training pipeline only depend on the small ``SentimentModel``
interface, so adding a new model (e.g. a fine-tuned transformer) means adding
one subclass and one registry entry - no changes to training, serving or
monitoring code.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SentimentModel(ABC):
    """Interface every sentiment model must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, texts: list[str], labels: list[str]) -> None: ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, n_classes) probability matrix.

        Column order must match ``self.classes``.
        """

    @property
    @abstractmethod
    def classes(self) -> list[str]: ...

    def predict(self, texts: list[str]) -> list[str]:
        probabilities = self.predict_proba(texts)
        return [self.classes[i] for i in probabilities.argmax(axis=1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class TfidfLogisticModel(SentimentModel):
    """TF-IDF features + logistic regression.

    Chosen as the primary model: trains in seconds on 10k reviews, serves in
    well under a millisecond per sentence (comfortably inside the 300ms p99
    budget), and its coefficients are directly inspectable. The task brief
    explicitly de-emphasises model sophistication.
    """

    name = "tfidf_logreg"

    def __init__(self, seed: int = 42):
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),  # bigrams capture simple negation ("not good")
                        min_df=2,
                        sublinear_tf=True,
                        max_features=100_000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        # neutral is half the size of the other classes
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self._pipeline.fit(texts, labels)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipeline.predict_proba(texts)

    @property
    def classes(self) -> list[str]:
        return list(self._pipeline.classes_)


# Registry mapping CLI/config names to implementations. New models are added
# here (e.g. "distilbert": DistilBertModel) and selected via `train.py --model`.
MODEL_REGISTRY: dict[str, type[SentimentModel]] = {
    TfidfLogisticModel.name: TfidfLogisticModel,
}


def create_model(name: str, seed: int = 42) -> SentimentModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed=seed)


def load_model(path: Path) -> SentimentModel:
    model = joblib.load(path)
    if not isinstance(model, SentimentModel):
        raise TypeError(f"Model at {path} is not a SentimentModel")
    return model
