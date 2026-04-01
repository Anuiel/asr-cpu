import torch
from benchmark.backends.base import BackendBase


class TorchOptBackend(BackendBase):
    def __init__(self):
        self._pipe = None

    @property
    def name(self) -> str:
        return "torch-opt"

    def setup(self, model_name: str) -> None:
        import sys
        from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline

        model_id = f"openai/whisper-{model_name}"
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        model = torch.compile(model)

        processor = AutoProcessor.from_pretrained(model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

    def transcribe(self, audio_path: str) -> str:
        result = self._pipe(audio_path, return_timestamps=True)
        return result["text"].strip()


    def cleanup(self) -> None:
        del self._pipe
        self._pipe = None
