from __future__ import annotations

import resource
import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import psutil


@dataclass
class ProfileResult:
    wall_time_sec: float
    cpu_time_sec: float
    peak_rss_mb: float
    tracemalloc_peak_mb: float
    energy_estimate_j: float
    avg_cpu_percent: float


class _CpuSampler:
    """Background thread that samples CPU% at ~100ms intervals."""

    def __init__(self):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            self._samples.append(proc.cpu_percent(interval=0.1))

    def stop(self) -> list[float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._samples


def profile(fn: Callable[[], str], tdp_watts: float = 15.0) -> tuple[str, ProfileResult]:
    """Profile a callable that returns a transcription string.

    Args:
        fn: zero-arg callable returning hypothesis text.
        tdp_watts: assumed CPU TDP for energy estimation.

    Returns:
        (hypothesis_text, ProfileResult)
    """
    sampler = _CpuSampler()

    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    cpu_start = time.process_time()
    sampler.start()
    wall_start = time.perf_counter()

    result_text = fn()

    wall_end = time.perf_counter()
    sampler_samples = sampler.stop()
    cpu_end = time.process_time()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_time = wall_end - wall_start
    cpu_time = cpu_end - cpu_start

    # ru_maxrss is in bytes on macOS, KB on Linux
    import platform
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    peak_rss_mb = max(rss_before, rss_after) / divisor

    avg_cpu = sum(sampler_samples) / len(sampler_samples) if sampler_samples else 0.0
    # Energy estimate: fraction of TDP used × wall time
    energy_j = (avg_cpu / 100.0) * tdp_watts * wall_time

    return result_text, ProfileResult(
        wall_time_sec=wall_time,
        cpu_time_sec=cpu_time,
        peak_rss_mb=peak_rss_mb,
        tracemalloc_peak_mb=tracemalloc_peak / (1024 * 1024),
        energy_estimate_j=energy_j,
        avg_cpu_percent=avg_cpu,
    )
