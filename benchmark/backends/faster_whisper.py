from benchmark.backends.base import BackendBase


class FasterWhisperBackend(BackendBase):
    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "faster-whisper"

    def setup(self, model_name: str) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self._model.transcribe(audio_path)
        return " ".join(seg.text.strip() for seg in segments).strip()


    def cleanup(self) -> None:
        del self._model
        self._model = None
