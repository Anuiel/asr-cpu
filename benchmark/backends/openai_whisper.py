from benchmark.backends.base import BackendBase


class OpenAIWhisperBackend(BackendBase):
    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "openai-whisper"

    def setup(self, model_name: str) -> None:
        import whisper
        self._model = whisper.load_model(model_name, device="cpu")

    def transcribe(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path, fp16=False)
        return result["text"].strip()


    def cleanup(self) -> None:
        del self._model
        self._model = None
