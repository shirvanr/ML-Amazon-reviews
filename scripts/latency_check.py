"""Measure end-to-end /predict latency against the 300ms p99 requirement.

Usage (with the service already running):
    python scripts/latency_check.py [--url http://127.0.0.1:8000] [--requests 300]

Sends real review texts sampled from the dataset so payload sizes match
production-like traffic, then reports client-side latency percentiles.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "Books_10k.jsonl"
SLO_P99_MS = 300


def load_sample_texts(path: Path, n: int, seed: int = 42) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"].strip()
            if text:
                texts.append(text[:5000])
    random.Random(seed).shuffle(texts)
    return texts[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    texts = load_sample_texts(args.data, args.requests + args.warmup)
    latencies_ms = []

    with httpx.Client(base_url=args.url, timeout=10) as client:
        health = client.get("/health").json()
        if not health.get("model_loaded"):
            sys.exit("Service is up but no model is loaded - run training first.")

        for i, text in enumerate(texts):
            started = time.perf_counter()
            response = client.post("/predict", json={"text": text})
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            if i >= args.warmup:  # exclude connection/JIT warmup from the SLO check
                latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    quantiles = statistics.quantiles(latencies_ms, n=100)
    p50, p95, p99 = quantiles[49], quantiles[94], quantiles[98]

    print(f"requests measured : {len(latencies_ms)} (after {args.warmup} warmup)")
    print(f"p50               : {p50:8.2f} ms")
    print(f"p95               : {p95:8.2f} ms")
    print(f"p99               : {p99:8.2f} ms")
    print(f"max               : {latencies_ms[-1]:8.2f} ms")
    print(f"SLO (p99 < {SLO_P99_MS}ms) : {'PASS' if p99 < SLO_P99_MS else 'FAIL'}")
    if p99 >= SLO_P99_MS:
        sys.exit(1)


if __name__ == "__main__":
    main()
