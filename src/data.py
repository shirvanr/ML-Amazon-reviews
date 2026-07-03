"""Dataset loading and construction of the sentence-level training data."""

import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocessing import rating_to_sentiment, split_sentences

logger = logging.getLogger(__name__)


def load_reviews(path: Path) -> list[dict]:
    """Load reviews from a JSONL file, keeping only the fields we use.

    Each returned record has: review_id (line number, unique within the file),
    rating (float) and text (str). Records with empty text are skipped.
    """
    reviews = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            record = json.loads(line)
            text = record["text"].strip()
            if not text:
                continue
            reviews.append(
                {
                    "review_id": line_number,
                    "rating": float(record["rating"]),
                    "text": text,
                }
            )
    logger.info("Loaded %d reviews from %s", len(reviews), path)
    return reviews


def split_reviews(
    reviews: list[dict], test_size: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split at the *review* level, stratified by sentiment class.

    Splitting before exploding reviews into sentences ensures sentences from
    the same review never appear in both train and test, which would leak
    author style and vocabulary and inflate evaluation metrics.
    """
    labels = [rating_to_sentiment(r["rating"]) for r in reviews]
    train, test = train_test_split(
        reviews, test_size=test_size, random_state=seed, stratify=labels
    )
    return train, test


def make_sentence_dataset(reviews: list[dict]) -> tuple[list[str], list[str]]:
    """Explode reviews into (sentence, sentiment) pairs.

    Per the task definition, every sentence inherits its review's star rating.
    """
    sentences: list[str] = []
    labels: list[str] = []
    for review in reviews:
        sentiment = rating_to_sentiment(review["rating"])
        for sentence in split_sentences(review["text"]):
            sentences.append(sentence)
            labels.append(sentiment)
    return sentences, labels
