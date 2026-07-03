"""Training pipeline: load data, train, evaluate, persist assets.

Run with:
    python -m src.train [--model tfidf_logreg] [--data data/Books_10k.jsonl]

Writes two assets to the model directory:
    model.joblib   - the trained model
    metadata.json  - evaluation metrics plus everything needed to trace the
                     assets back to how it was produced (model name, data
                     file, seed, package versions, timestamp)
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import load_reviews, make_sentence_dataset, split_reviews
from src.model import MODEL_REGISTRY, create_model
from src.preprocessing import LABELS

logger = logging.getLogger(__name__)


def train_and_evaluate(
    data_path: Path,
    model_name: str,
    model_dir: Path,
    test_size: float,
    seed: int,
) -> dict:
    """Run the full pipeline and return the evaluation metadata."""
    reviews = load_reviews(data_path)
    train_reviews, test_reviews = split_reviews(reviews, test_size=test_size, seed=seed)

    train_texts, train_labels = make_sentence_dataset(train_reviews)
    test_texts, test_labels = make_sentence_dataset(test_reviews)
    logger.info(
        "Sentence datasets built: %d train / %d test sentences",
        len(train_texts),
        len(test_texts),
    )

    model = create_model(model_name, seed=seed)
    start = time.perf_counter()
    model.fit(train_texts, train_labels)
    logger.info("Trained '%s' in %.1fs", model_name, time.perf_counter() - start)

    predictions = model.predict(test_texts)
    report = classification_report(
        test_labels, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(test_labels, predictions, labels=LABELS).tolist()

    logger.info(
        "Evaluation:\n%s",
        classification_report(test_labels, predictions, labels=LABELS, zero_division=0),
    )
    logger.info("Confusion matrix (rows=true, cols=predicted, order=%s):", LABELS)
    for label, row in zip(LABELS, matrix):
        logger.info("  %-9s %s", label, row)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "n_reviews": len(reviews),
        "n_train_sentences": len(train_texts),
        "n_test_sentences": len(test_texts),
        "test_size": test_size,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "metrics": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {label: report[label] for label in LABELS},
            "confusion_matrix": {"labels": LABELS, "matrix": matrix},
        },
    }

    model_path = model_dir / "model.joblib"
    model.save(model_path)
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model to %s and metadata to %s", model_path, metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment classifier")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    metadata = train_and_evaluate(
        data_path=args.data,
        model_name=args.model,
        model_dir=args.model_dir,
        test_size=args.test_size,
        seed=args.seed,
    )
    metrics = metadata["metrics"]
    logger.info(
        "Done. accuracy=%.3f macro_f1=%.3f", metrics["accuracy"], metrics["macro_f1"]
    )


if __name__ == "__main__":
    main()
