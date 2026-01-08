"""Lightweight metrics helpers for timing and counters.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

import numpy as np
import torch


@dataclass
class Metrics:
    """Collects timing samples and counters."""

    timings: dict[str, list[float]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    render_timing: RenderTiming | None = None

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


LOGGER = logging.getLogger(__name__)


class RenderTiming:
    """Collect per-frame render timings with CPU and CUDA event support."""

    stage_order = [
        "render_setup",
        "render_pack_inputs",
        "render_h2d_transfer",
        "render_gpu_project_sort",
        "render_gpu_shading",
        "render_gpu_raster_blend",
        "render_d2h_transfer",
        "render_encode_prepare",
        "render_encode_compress",
        "render_encode_write",
        "render_output_encode",
        "render_sync_overhead",
    ]
    encode_stages = [
        "render_encode_prepare",
        "render_encode_compress",
        "render_encode_write",
    ]
    gpu_stages = {
        "render_gpu_project_sort",
        "render_gpu_shading",
        "render_gpu_raster_blend",
    }

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = {}
        self._current_frame: dict[str, float] | None = None
        self._pending_gpu_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            stage: [] for stage in self.gpu_stages
        }
        self._gpu_enabled = torch.cuda.is_available()
        self._logged_gpu_disabled = False

    def start_frame(self) -> None:
        """Start a new frame's timing accumulation."""
        self._current_frame = {stage: 0.0 for stage in self.stage_order}
        self._pending_gpu_events = {stage: [] for stage in self.gpu_stages}

    def _ensure_frame(self) -> None:
        if self._current_frame is None:
            self.start_frame()

    @contextmanager
    def timed_cpu(self, stage: str) -> Iterator[None]:
        """Time a CPU stage using wall-clock."""
        self._ensure_frame()
        start = perf_counter()
        yield
        duration = perf_counter() - start
        if self._current_frame is not None:
            self._current_frame[stage] = self._current_frame.get(stage, 0.0) + duration

    @contextmanager
    def gpu_event_timer(self, stage: str) -> Iterator[None]:
        """Time a GPU stage using CUDA events."""
        self._ensure_frame()
        if not self._gpu_enabled:
            if not self._logged_gpu_disabled:
                LOGGER.info("CUDA not available; GPU stage timings disabled.")
                self._logged_gpu_disabled = True
            yield
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield
        end_event.record()
        self._pending_gpu_events.setdefault(stage, []).append((start_event, end_event))

    def finalize_frame(self) -> None:
        """Finalize the current frame, syncing once to read GPU event timings."""
        if self._current_frame is None:
            return
        sync_overhead = 0.0
        if self._gpu_enabled and any(self._pending_gpu_events.values()):
            sync_start = perf_counter()
            torch.cuda.synchronize()
            sync_overhead = perf_counter() - sync_start
            for stage, events in self._pending_gpu_events.items():
                for start_event, end_event in events:
                    duration = start_event.elapsed_time(end_event) / 1000.0
                    self._current_frame[stage] = self._current_frame.get(stage, 0.0) + duration
        if sync_overhead > 0.0:
            self._current_frame["render_sync_overhead"] += sync_overhead

        encode_total = sum(self._current_frame.get(stage, 0.0) for stage in self.encode_stages)
        if self._current_frame.get("render_output_encode", 0.0) == 0.0 and encode_total > 0.0:
            self._current_frame["render_output_encode"] = encode_total

        for stage in set(self.stage_order) | set(self._current_frame):
            self.timings.setdefault(stage, []).append(self._current_frame.get(stage, 0.0))

        total_stage_order = [
            stage
            for stage in self.stage_order
            if stage not in self.encode_stages and stage != "render_output_encode"
        ] + ["render_output_encode"]
        total_breakdown = sum(self._current_frame.get(stage, 0.0) for stage in total_stage_order)
        self.timings.setdefault("render_total_breakdown", []).append(total_breakdown)
        self._current_frame = None

    def summarize(self) -> dict[str, dict[str, float]]:
        """Summarize render timings with mean/p50/p90/total."""
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
