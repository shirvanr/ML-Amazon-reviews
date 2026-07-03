# Review Sentiment Service

[![CI](https://github.com/shirvanr/ML-Amazon-reviews/actions/workflows/ci.yml/badge.svg)](https://github.com/shirvanr/ML-Amazon-reviews/actions/workflows/ci.yml)

An end-to-end ML system that classifies book-review **sentences** as
`positive`, `neutral` or `negative`, and serves predictions over HTTP within a
p99 latency budget of 300 ms.

Built for the MLOps take-home exercise, using the provided `Books_10k.jsonl`
subset of the Amazon Books reviews dataset (10,000 reviews, balanced across
1–5 stars).

## How it works

```
data/Books_10k.jsonl
        │
        ▼
  src/train.py ── review-level 80/20 split ── sentence explosion ── TF-IDF + LogReg
        │
        ▼
  models/model.joblib + models/metadata.json (metrics, versions, timestamp)
        │
        ▼
  src/service/app.py (FastAPI)
        ├── POST /predict   per-sentence sentiment + review-level aggregate
        ├── GET  /health    liveness / readiness probe
        └── GET  /metrics   latency percentiles + predicted-class distribution
```

Labels are derived from star ratings (1–2 → negative, 3 → neutral,
4–5 → positive) and every sentence inherits its review's rating, as specified
in the task.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
make install            # pip install -r requirements-dev.txt
```

## Running

```bash
make train              # trains and evaluates; writes models/ (~10 s)
make serve              # starts the API on http://127.0.0.1:8000
make latency            # p99 latency check against the running service
make test               # unit + API tests (pytest)
make lint               # ruff
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Great story and characters. The ending felt rushed though."}'
```

```json
{
  "overall_sentiment": "neutral",
  "sentences": [
    {"sentence": "Great story and characters.", "sentiment": "positive", "confidence": 0.5026},
    {"sentence": "The ending felt rushed though.", "sentiment": "neutral", "confidence": 0.8534}
  ],
  "model_name": "tfidf_logreg"
}
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### Docker

```bash
make train              # the image packages the trained model
make docker-build
make docker-run         # serves on http://127.0.0.1:8000
```

## Results

Evaluated on a held-out 20% of reviews (14,177 sentences), fixed seed:

| class    | precision | recall | f1   |
|----------|-----------|--------|------|
| negative | 0.59      | 0.57   | 0.58 |
| neutral  | 0.35      | 0.36   | 0.35 |
| positive | 0.64      | 0.64   | 0.64 |

Accuracy **0.55**, macro-F1 **0.53**. Full report, confusion matrix and
training metadata are written to `models/metadata.json`.

**Why the scores look modest — and why that's expected.** The task assigns
every sentence its review's star rating, which makes the labels inherently
noisy: a 5-star review still contains neutral sentences ("I bought this for
my daughter"), and a 3-star review contains genuinely positive and negative
ones. The confusion matrix confirms this — errors concentrate in and around
the `neutral` class, while the `negative`/`positive` confusion is much lower.
Measured against these labels, even a much heavier model would be capped by
the same noise, which is why effort went into the system rather than the
model (the brief also explicitly de-emphasises model choice).

Latency, measured client-side with real review texts
(`scripts/latency_check.py`, 300 requests after warmup, local machine):

| p50     | p95     | p99     | SLO        |
|---------|---------|---------|------------|
| 2.7 ms  | 3.7 ms  | 4.5 ms  | p99 < 300 ms ✅ |

## Design decisions

- **Model: TF-IDF (1–2 grams) + logistic regression.** Trains in seconds,
  predicts a full review in ~1 ms, and its coefficients are inspectable.
  Bigrams capture simple negation ("not good"); `class_weight="balanced"`
  compensates for `neutral` being half the size of the other classes.
- **Swappable models.** Training and serving depend only on the small
  `SentimentModel` interface (`src/model.py`). Adding e.g. a distilled
  transformer means one subclass and one `MODEL_REGISTRY` entry, selected via
  `python -m src.train --model <name>` — no serving changes.
- **Review-level split before sentence explosion.** Sentences from one review
  never straddle train and test; splitting after explosion would leak author
  style and vocabulary and inflate metrics.
- **One shared preprocessing module.** The same sentence splitter and label
  mapping (`src/preprocessing.py`) are used in training and serving, ruling
  out train/serve skew. The splitter is a small regex — no model downloads,
  trivially testable; it over-splits abbreviations ("Mr. Smith"), which is
  acceptable noise for bag-of-words sentiment.
- **Model loaded once at startup** (FastAPI lifespan); requests only pay for
  vectorize + predict, and all sentences of a review are predicted in one
  vectorized call.
- **Config via environment variables** (`src/config.py`) so the identical
  image runs locally and in the cloud.

## Logging and monitoring

- **Structured JSON logs** — model load events, one line per prediction
  (sentence count, overall sentiment, latency), errors with tracebacks.
  Directly ingestible by any log aggregator.
- **`/metrics`** exposes request/error counts, rolling p50/p95/p99 latency,
  and the distribution of predicted classes. The class distribution is a
  deliberately cheap drift signal: with no ground-truth labels at serving
  time, a shift in predictions is the first hint of input drift or an
  upstream data problem.
- **`/health`** reports whether the model is loaded, suitable as a
  readiness/liveness probe.

## Continuous integration

Every push runs the GitHub Actions workflow (`.github/workflows/ci.yml`) in
two stages:

1. **quality** — ruff lint plus the unit and API test suite.
2. **train-and-serve** — retrains the model from the committed dataset,
   builds the Docker image around the fresh model, starts the container and
   smoke tests `/health` and `/predict` against it.

The second stage is the ML-specific part: it proves the entire pipeline —
data → training → image → serving — is reproducible from a clean checkout on
every change, not just that the unit tests pass.

## Running this in production

What the path to a cloud deployment looks like (the code already supports it):

1. **model storage** — `train.py` writes a self-describing pair
   (`model.joblib` + `metadata.json`); in production these would go to object
   storage or a model registry (Vertex AI Model Registry, MLflow) instead of
   a local folder, and the image would pull the model at startup rather than
   baking it in.
2. **Deployment** — the container is stateless and configured via env vars,
   so it maps directly onto Cloud Run / ECS / Kubernetes with horizontal
   scaling behind a load balancer.
3. **CI/CD** — the included GitHub Actions workflow already lints, tests,
   retrains and smoke tests the serving image on every push; production would
   add pushing the image to a registry and deploying it, plus training as a
   scheduled or data-triggered pipeline step with an automatic quality gate
   (block promotion if macro-F1 regresses against the current production
   model).
4. **Monitoring** — export the `/metrics` counters to Prometheus/Grafana with
   alerts on latency SLO burn and prediction-distribution shift; sample
   predictions to a store for periodic human labelling to measure real
   accuracy over time.

## Limitations and future work

- **Label noise is the accuracy ceiling** — sentence-level labels inherited
  from review ratings are wrong for many individual sentences. A sample of
  human-labelled sentences would both measure and lift that ceiling.
- The regex sentence splitter over-splits abbreviations and decimal numbers;
  a proper tokenizer would be a drop-in change inside `preprocessing.py`.
- A transformer-based model behind the same interface would likely improve
  the `neutral` class, at the cost of GPU serving considerations — worth it
  only after cleaner labels exist.
- `/metrics` is in-process (per-replica, reset on restart); Prometheus
  instrumentation is the production replacement.
- English-only for now; the TF-IDF vocabulary would not transfer to other
  languages.

## Project layout

```
├── .github/workflows/ci.yml    # CI: lint + tests, then train + image + smoke test
├── data/Books_10k.jsonl        # provided dataset (committed so the repo runs end-to-end)
├── src/
│   ├── config.py               # env-var driven settings
│   ├── preprocessing.py        # sentence splitting + label mapping (shared train/serve)
│   ├── data.py                 # loading, review-level split, sentence dataset
│   ├── model.py                # SentimentModel interface, TF-IDF+LogReg, registry
│   ├── train.py                # training pipeline CLI
│   └── service/                # FastAPI app, schemas, metrics tracker
├── scripts/latency_check.py    # client-side p99 verification
├── tests/                      # unit + API tests
├── Dockerfile                  # non-root serving image
└── Makefile                    # install / train / serve / test / lint / latency
```
