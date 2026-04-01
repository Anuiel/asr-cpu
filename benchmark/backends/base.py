from abc import ABC, abstractmethod


class BackendBase(ABC):
    @abstractmethod
    def setup(self, model_name: str) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

    @abstractmethod
    def name(self) -> str: ...

    def cleanup(self) -> None:
        pass
