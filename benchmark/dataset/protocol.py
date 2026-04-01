from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Sample:
    audio_path: str
    reference: str
    duration_sec: float
    sample_id: str


class DatasetProtocol(ABC):
    @abstractmethod
    def load(
        self,
        *,
        data_dir: str,
        min_duration: float = 0.0,
        max_duration: float | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> list[Sample]:
        """Load and filter samples from a local prepared directory."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
