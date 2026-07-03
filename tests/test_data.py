import json

import pytest

from src.data import load_reviews, make_sentence_dataset, split_reviews


@pytest.fixture
def sample_jsonl(tmp_path):
    """A small dataset with all ratings represented and one empty review."""
    records = []
    for rating in [1, 2, 3, 4, 5] * 10:
        records.append({"rating": rating, "text": f"Sentence one, {rating} stars. Sentence two."})
    records.append({"rating": 5, "text": "   "})  # should be skipped
    path = tmp_path / "reviews.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_load_reviews_skips_empty_text(sample_jsonl):
    reviews = load_reviews(sample_jsonl)
    assert len(reviews) == 50
    assert all(review["text"] for review in reviews)


def test_load_reviews_assigns_unique_ids(sample_jsonl):
    reviews = load_reviews(sample_jsonl)
    ids = [review["review_id"] for review in reviews]
    assert len(ids) == len(set(ids))


def test_split_reviews_no_overlap_and_stratified(sample_jsonl):
    reviews = load_reviews(sample_jsonl)
    train, test = split_reviews(reviews, test_size=0.2, seed=0)

    train_ids = {review["review_id"] for review in train}
    test_ids = {review["review_id"] for review in test}
    # No review (and therefore no sentence of it) can be in both splits.
    assert train_ids.isdisjoint(test_ids)
    assert len(train) + len(test) == len(reviews)
    # Stratification: the balanced input stays balanced in the test split.
    test_ratings = sorted(review["rating"] for review in test)
    assert test_ratings == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]


def test_make_sentence_dataset_explodes_and_inherits_label():
    reviews = [{"review_id": 0, "rating": 5.0, "text": "Loved it. Truly great!"}]
    sentences, labels = make_sentence_dataset(reviews)
    assert sentences == ["Loved it.", "Truly great!"]
    assert labels == ["positive", "positive"]
