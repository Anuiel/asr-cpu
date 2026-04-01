#!/usr/bin/env python3
"""CLI entry point for ASR CPU benchmarks."""
from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Whisper CPU Benchmark Suite")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["openai-whisper", "faster-whisper", "onnx", "torch-opt"],
        help="Backend to benchmark",
    )
    parser.add_argument(
        "--dataset",
        default="librispeech",
        help="Dataset name (default: librispeech)",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to prepared dataset directory (with manifest.json)")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-name", default="base", help="Whisper model size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup samples")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--tdp-watts", type=float, default=15.0, help="CPU TDP for energy est.")

    args = parser.parse_args()

    from benchmark.dataset import get_dataset
    from benchmark.backends import get_backend
    from benchmark.runner import run_benchmark, save_results, print_summary

    # Load dataset
    print(f"Loading dataset: {args.dataset} from {args.data_dir}", file=sys.stderr)
    ds = get_dataset(args.dataset)
    samples = ds.load(
        data_dir=args.data_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} samples", file=sys.stderr)

    if not samples:
        print("No samples matched the filters. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Load backend
    print(f"Setting up backend: {args.backend} (model={args.model_name})", file=sys.stderr)
    backend = get_backend(args.backend)
    backend.setup(args.model_name)

    # Run benchmark
    print("Running benchmark...", file=sys.stderr)
    results = run_benchmark(
        backend=backend,
        samples=samples,
        dataset_name=ds.name,
        warmup=args.warmup,
    )

    # Output
    print_summary(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_results(results, args.output)
        print(f"Results saved to {args.output}", file=sys.stderr)

    backend.cleanup()


if __name__ == "__main__":
    main()
