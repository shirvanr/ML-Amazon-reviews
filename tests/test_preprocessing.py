import pytest

from src.preprocessing import rating_to_sentiment, split_sentences


class TestRatingToSentiment:
    @pytest.mark.parametrize(
        "rating,expected",
        [
            (1.0, "negative"),
            (2.0, "negative"),
            (3.0, "neutral"),
            (4.0, "positive"),
            (5.0, "positive"),
        ],
    )
    def test_mapping(self, rating, expected):
        assert rating_to_sentiment(rating) == expected

    @pytest.mark.parametrize("rating", [0.0, 6.0, -1.0])
    def test_out_of_range_raises(self, rating):
        with pytest.raises(ValueError):
            rating_to_sentiment(rating)


class TestSplitSentences:
    def test_multiple_sentences(self):
        text = "Great book. Loved the plot! Would you read it? Yes."
        assert split_sentences(text) == [
            "Great book.",
            "Loved the plot!",
            "Would you read it?",
            "Yes.",
        ]

    def test_single_sentence_without_terminator(self):
        assert split_sentences("no punctuation at all") == ["no punctuation at all"]

    def test_newlines_are_boundaries(self):
        assert split_sentences("First line\nSecond line") == ["First line", "Second line"]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_extra_whitespace_is_stripped(self):
        assert split_sentences("  Good.   Bad.  ") == ["Good.", "Bad."]
