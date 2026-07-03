"""Text preprocessing shared by training and serving.

Keeping sentence splitting and label mapping in one module guarantees the
service applies exactly the same transformations the model was trained on
(no train/serve skew).
"""

import re

# Class labels in a fixed, alphabetical order used across the project.
LABELS = ["negative", "neutral", "positive"]

# Star-rating thresholds for the sentiment mapping (1-2 negative, 3 neutral,
# 4-5 positive). This is the standard mapping for 5-star review data.
_NEGATIVE_MAX = 2
_NEUTRAL = 3

# Sentence boundary: one or more of . ! ? followed by whitespace, or any
# newline run. A regex splitter is deliberately chosen over a statistical
# tokenizer (e.g. NLTK punkt): it has no runtime downloads, is trivially
# testable, and is fast enough to keep serving latency negligible. Known
# limitation: abbreviations like "Mr. Smith" produce an extra split, which
# is acceptable noise for bag-of-words sentiment.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def rating_to_sentiment(rating: float) -> str:
    """Map a 1-5 star rating to a sentiment class."""
    if not 1 <= rating <= 5:
        raise ValueError(f"rating must be within [1, 5], got {rating}")
    if rating <= _NEGATIVE_MAX:
        return "negative"
    if rating <= _NEUTRAL:
        return "neutral"
    return "positive"


def split_sentences(text: str) -> list[str]:
    """Split review text into sentences, dropping empty fragments."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
