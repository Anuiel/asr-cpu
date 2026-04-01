#!/usr/bin/env python3
"""Download and prepare LibriSpeech data for benchmarking.

Run this ONCE outside Docker, then mount the output directory as a volume.

Usage:
    python prepare_data.py --output ./data/librispeech --split test.clean
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import soundfile as sf
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ASR dataset")
    parser.add_argument("--output", required=True, help="Output directory for prepared data")
    parser.add_argument("--split", default="test.clean", help="Dataset split (default: test.clean)")
    parser.add_argument("--dataset", default="librispeech", choices=["librispeech"],
                        help="Dataset to download")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    wav_dir = os.path.join(args.output, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    if args.dataset == "librispeech":
        download_librispeech(args.split, wav_dir, args.output)


def download_librispeech(split: str, wav_dir: str, output_dir: str):
    print(f"Downloading openslr/librispeech_asr split={split} (streaming)...", file=sys.stderr)
    ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)

    manifest = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        sr = audio["sampling_rate"]
        array = audio["array"]
        duration = len(array) / sr
        sample_id = str(row["id"])

        wav_path = os.path.join(wav_dir, f"{sample_id}.wav")
        if not os.path.exists(wav_path):
            sf.write(wav_path, array, sr)

        manifest.append({
            "sample_id": sample_id,
            "audio_path": f"wav/{sample_id}.wav",  # relative to output dir
            "reference": row["text"],
            "duration_sec": round(duration, 3),
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples...", file=sys.stderr)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done: {len(manifest)} samples saved to {output_dir}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)

    # Print duration stats
    durations = [s["duration_sec"] for s in manifest]
    print(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s", file=sys.stderr)
    print(f"Total audio: {sum(durations)/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
