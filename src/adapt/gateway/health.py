"""What `/readyz` checks and what `/metrics` counts.

Ready means that vLLM answers, that it serves the base model and every adapter of
`configs/serve.yaml`, and that the adapter files on disk have the hashes in that file. The hash of
an adapter is worked out again only when its files change, because they are large and `/readyz` is
called often.
"""

import threading
from collections.abc import Sequence
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Histogram

from adapt.client import ModelClient, ModelError
from adapt.eval.runner import ADAPTER_FILES, hash_directory
from adapt.serve import ServeConfig, adapter_path

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
UNKNOWN_PLATFORM = "unknown"


class AdapterHashes:
    """The hash of each adapter folder, worked out again when a file changes."""

    def __init__(self) -> None:
        self._seen: dict[Path, tuple[tuple[int, int], ...]] = {}
        self._hashes: dict[Path, str] = {}
        self._lock = threading.Lock()

    def of(self, path: Path) -> str:
        stamp = tuple(
            (stat.st_mtime_ns, stat.st_size)
            for stat in (path.joinpath(name).stat() for name in ADAPTER_FILES)
        )
        with self._lock:
            if self._seen.get(path) != stamp:
                self._hashes[path] = hash_directory(path, ADAPTER_FILES)
                self._seen[path] = stamp
            return self._hashes[path]


def adapter_problems(serve: ServeConfig, hashes: AdapterHashes) -> list[str]:
    problems = []
    for adapter in serve.adapters:
        path = adapter_path(adapter)
        if not all(path.joinpath(name).is_file() for name in ADAPTER_FILES):
            problems.append(f"adapter {adapter.name} is not on disk")
        elif hashes.of(path) != adapter.sha256:
            problems.append(f"adapter {adapter.name} does not have the hash in serve.yaml")
    return problems


def readiness(serve: ServeConfig, client: ModelClient, hashes: AdapterHashes) -> list[str]:
    """What is wrong, or an empty list when the gateway can serve."""
    problems = []
    try:
        served = client.models()
    except ModelError as error:
        problems.append(f"vLLM cannot be reached ({error.kind})")
    else:
        wanted = [serve.model, *(adapter.name for adapter in serve.adapters)]
        problems += [f"{name} is not served by vLLM" for name in wanted if name not in served]
    return problems + adapter_problems(serve, hashes)


class Metrics:
    """Counters of the gateway, in a registry of its own so that every app starts from zero."""

    def __init__(self, platforms: Sequence[str]) -> None:
        self.registry = CollectorRegistry()
        self._platforms = set(platforms)
        self.requests = Counter(
            "gateway_requests_total",
            "Requests by platform and outcome",
            ["platform", "outcome"],
            registry=self.registry,
        )
        self.reasons = Counter(
            "gateway_review_reasons_total",
            "Documents sent to review, by platform and reason",
            ["platform", "reason"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "gateway_request_duration_seconds",
            "Time to answer a request",
            ["platform"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

    def platform(self, name: str) -> str:
        """A platform the gateway serves, or `unknown`: a caller must not be able to make labels."""
        return name if name in self._platforms else UNKNOWN_PLATFORM

    def observe(self, platform: str, outcome: str, reasons: Sequence[str], seconds: float) -> None:
        label = self.platform(platform)
        self.requests.labels(label, outcome).inc()
        for reason in reasons:
            self.reasons.labels(label, reason).inc()
        self.latency.labels(label).observe(seconds)
