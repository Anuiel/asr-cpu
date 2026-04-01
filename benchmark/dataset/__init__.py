from benchmark.dataset.protocol import DatasetProtocol, Sample
from benchmark.dataset.librispeech import LibriSpeechDataset

DATASETS: dict[str, type[DatasetProtocol]] = {
    "librispeech": LibriSpeechDataset,
}


def get_dataset(name: str) -> DatasetProtocol:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]()
