"""Request/response contracts for the prediction service."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="Review text; it is split into sentences server-side.",
        examples=["Great story and characters. The ending felt rushed though."],
    )


class SentencePrediction(BaseModel):
    sentence: str
    sentiment: str
    confidence: float = Field(description="Probability of the predicted class.")


class PredictResponse(BaseModel):
    overall_sentiment: str = Field(
        description="Argmax of the mean class probabilities across sentences."
    )
    sentences: list[SentencePrediction]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
