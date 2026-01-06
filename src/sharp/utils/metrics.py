"""Lightweight metrics helpers for timing and counters.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    """Collects timing samples and counters."""

    timings: dict[str, list[float]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def add_time(self, name: str, duration_s: float) -> None:
        """Record a timing sample in seconds."""
        self.timings.setdefault(name, []).append(float(duration_s))

    def inc(self, name: str, n: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + int(n)

    def summarize(self) -> dict[str, dict[str, float]]:
        """Summarize timing samples with mean/p50/p90/total."""
        summary: dict[str, dict[str, float]] = {}
        for name, samples in self.timings.items():
            if not samples:
                continue
            values = np.asarray(samples, dtype=np.float64)
            summary[name] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "total": float(values.sum()),
            }
        return summary
