"""Measure end-to-end /predict latency against the 300ms p99 requirement.

Usage (with the service already running):
    python scripts/latency_check.py                  # sequential baseline
    python scripts/latency_check.py --concurrency 50 # p99 under parallel load

Sends real review texts sampled from the dataset so payload sizes match
production-like traffic, then reports client-side latency percentiles and
throughput. The sequential run shows best-case latency; the concurrent run
shows whether the SLO still holds when many clients hit the service at once,
which is the number that actually matters for capacity planning.
"""

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


def run_load(
    client: httpx.Client, texts: list[str], concurrency: int
) -> tuple[list[float], float]:
    """Send one request per text; return per-request latencies and wall time."""

    def send_one(text: str) -> float:
        started = time.perf_counter()
        response = client.post("/predict", json={"text": text})
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return elapsed_ms

    started = time.perf_counter()
    if concurrency == 1:
        latencies_ms = [send_one(text) for text in texts]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            latencies_ms = list(pool.map(send_one, texts))
    wall_seconds = time.perf_counter() - started
    return latencies_ms, wall_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--concurrency", type=int, default=1, help="number of parallel clients"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    texts = load_sample_texts(args.data, args.requests + args.warmup)
    limits = httpx.Limits(max_connections=args.concurrency)

    with httpx.Client(base_url=args.url, timeout=30, limits=limits) as client:
        health = client.get("/health").json()
        if not health.get("model_loaded"):
            sys.exit("Service is up but no model is loaded - run training first.")

        # Sequential warmup excludes connection setup and lazy imports from
        # the SLO measurement.
        run_load(client, texts[: args.warmup], concurrency=1)
        latencies_ms, wall_seconds = run_load(
            client, texts[args.warmup :], concurrency=args.concurrency
        )

    latencies_ms.sort()
    quantiles = statistics.quantiles(latencies_ms, n=100)
    p50, p95, p99 = quantiles[49], quantiles[94], quantiles[98]
    throughput = len(latencies_ms) / wall_seconds

    print(f"requests measured : {len(latencies_ms)} (after {args.warmup} warmup)")
    print(f"concurrency       : {args.concurrency}")
    print(f"throughput        : {throughput:8.1f} req/s")
    print(f"p50               : {p50:8.2f} ms")
    print(f"p95               : {p95:8.2f} ms")
    print(f"p99               : {p99:8.2f} ms")
    print(f"max               : {latencies_ms[-1]:8.2f} ms")
    print(f"SLO (p99 < {SLO_P99_MS}ms) : {'PASS' if p99 < SLO_P99_MS else 'FAIL'}")
    if p99 >= SLO_P99_MS:
        sys.exit(1)


if __name__ == "__main__":
    main()
