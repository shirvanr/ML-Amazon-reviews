"""Central configuration.

All values can be overridden through environment variables so the same code
runs unchanged on a laptop, in Docker, or in a cloud environment.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data / assets
DATA_PATH = Path(os.getenv("DATA_PATH", REPO_ROOT / "data" / "Books_10k.jsonl"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", REPO_ROOT / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
METADATA_PATH = MODEL_DIR / "metadata.json"

# Training
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "tfidf_logreg")

# Service
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Rolling window used for latency percentiles exposed on /metrics.
METRICS_WINDOW_SIZE = int(os.getenv("METRICS_WINDOW_SIZE", "1000"))
