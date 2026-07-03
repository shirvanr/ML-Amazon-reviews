"""In-process service metrics.

A deliberately small, dependency-free tracker exposed on /metrics. In a real
deployment this would be replaced by (or exported to) Prometheus; the point
here is that the service observes the two things that matter for an ML
system: request latency against the SLO, and the distribution of predicted
classes over time - a shift in that distribution is a cheap first signal of
input drift or an upstream data problem.
"""

import threading
import time
from collections import Counter, deque


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; expects a sorted, non-empty list."""
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


class MetricsTracker:
    """Thread-safe counters and a rolling latency window."""

    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._latencies_ms: deque[float] = deque(maxlen=window_size)
        self._request_count = 0
        self._error_count = 0
        self._prediction_counts: Counter[str] = Counter()

    def record_request(self, latency_ms: float, predicted_labels: list[str]) -> None:
        with self._lock:
            self._request_count += 1
            self._latencies_ms.append(latency_ms)
            self._prediction_counts.update(predicted_labels)

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            latency_stats = None
            if latencies:
                latency_stats = {
                    "count": len(latencies),
                    "p50_ms": round(percentile(latencies, 0.50), 2),
                    "p95_ms": round(percentile(latencies, 0.95), 2),
                    "p99_ms": round(percentile(latencies, 0.99), 2),
                    "max_ms": round(latencies[-1], 2),
                }
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "request_count": self._request_count,
                "error_count": self._error_count,
                "latency_window": latency_stats,
                "prediction_counts": dict(self._prediction_counts),
            }
