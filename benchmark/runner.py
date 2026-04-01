from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass

import psutil
from tabulate import tabulate

from benchmark.backends.base import BackendBase
from benchmark.dataset.protocol import Sample
from benchmark.metrics import compute_cer, compute_wer, corpus_wer, corpus_cer
from benchmark.profiler import ProfileResult, profile


@dataclass
class SampleResult:
    backend: str
    dataset: str
    sample_id: str
    duration_sec: float
    wall_time_sec: float
    cpu_time_sec: float
    rtf: float
    peak_rss_mb: float
    tracemalloc_peak_mb: float
    energy_estimate_j: float
    avg_cpu_percent: float
    wer: float
    cer: float
    hypothesis: str
    reference: str


def run_benchmark(
    backend: BackendBase,
    samples: list[Sample],
    dataset_name: str,
    warmup: int = 1,
) -> list[SampleResult]:
    results: list[SampleResult] = []

    # Warmup
    for i in range(min(warmup, len(samples))):
        print(f"  [warmup {i+1}/{warmup}] {samples[i].sample_id}", file=sys.stderr)
        backend.transcribe(samples[i].audio_path)

    for i, sample in enumerate(samples):
        print(
            f"  [{i+1}/{len(samples)}] {sample.sample_id} "
            f"({sample.duration_sec:.1f}s)",
            file=sys.stderr,
        )

        hyp, prof = profile(lambda s=sample: backend.transcribe(s.audio_path))

        wer = compute_wer(sample.reference, hyp)
        cer = compute_cer(sample.reference, hyp)
        rtf = prof.wall_time_sec / sample.duration_sec if sample.duration_sec > 0 else 0.0

        results.append(SampleResult(
            backend=backend.name(),
            dataset=dataset_name,
            sample_id=sample.sample_id,
            duration_sec=sample.duration_sec,
            wall_time_sec=prof.wall_time_sec,
            cpu_time_sec=prof.cpu_time_sec,
            rtf=rtf,
            peak_rss_mb=prof.peak_rss_mb,
            tracemalloc_peak_mb=prof.tracemalloc_peak_mb,
            energy_estimate_j=prof.energy_estimate_j,
            avg_cpu_percent=prof.avg_cpu_percent,
            wer=wer,
            cer=cer,
            hypothesis=hyp,
            reference=sample.reference,
        ))

    return results


def collect_hardware_info() -> dict:
    mem = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": None,
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_available_gb": round(mem.available / (1024**3), 1),
        "python_version": platform.python_version(),
    }
    freq = psutil.cpu_freq()
    if freq:
        info["cpu_freq_mhz"] = round(freq.current)
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        info["omp_num_threads"] = int(omp)
    return info


def save_results(results: list[SampleResult], output_path: str) -> None:
    output = {
        "hardware": collect_hardware_info(),
        "samples": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def print_summary(results: list[SampleResult]) -> None:
    backend_name = results[0].backend
    n = len(results)
    avg_rtf = sum(r.rtf for r in results) / n
    avg_wall = sum(r.wall_time_sec for r in results) / n
    max_rss = max(r.peak_rss_mb for r in results)
    avg_energy = sum(r.energy_estimate_j for r in results) / n
    total_duration = sum(r.duration_sec for r in results)
    total_wall = sum(r.wall_time_sec for r in results)

    all_refs = [r.reference for r in results]
    all_hyps = [r.hypothesis for r in results]
    agg_wer = corpus_wer(all_refs, all_hyps) * 100
    agg_cer = corpus_cer(all_refs, all_hyps) * 100

    rows = [[
        backend_name,
        n,
        f"{total_duration:.1f}",
        f"{avg_rtf:.3f}",
        f"{avg_wall:.2f}",
        f"{total_wall:.1f}",
        f"{max_rss:.0f}",
        f"{avg_energy:.2f}",
        f"{agg_wer:.2f}",
        f"{agg_cer:.2f}",
    ]]

    headers = [
        "Backend", "Samples", "Total Audio (s)", "Avg RTF",
        "Avg Wall (s)", "Total Wall (s)", "Peak RSS (MB)",
        "Avg Energy (J)", "WER (%)", "CER (%)",
    ]

    print()
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()
