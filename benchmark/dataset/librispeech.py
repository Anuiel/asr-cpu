from __future__ import annotations

import json
import os

from benchmark.dataset.protocol import DatasetProtocol, Sample


class LibriSpeechDataset(DatasetProtocol):
    @property
    def name(self) -> str:
        return "librispeech"

    def load(
        self,
        *,
        data_dir: str,
        min_duration: float = 0.0,
        max_duration: float | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> list[Sample]:
        manifest_path = os.path.join(data_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run: python prepare_data.py --output {data_dir}"
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        samples: list[Sample] = []
        for entry in manifest:
            duration = entry["duration_sec"]
            if duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue

            audio_path = os.path.join(data_dir, entry["audio_path"])
            samples.append(Sample(
                audio_path=audio_path,
                reference=entry["reference"],
                duration_sec=duration,
                sample_id=entry["sample_id"],
            ))

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples
