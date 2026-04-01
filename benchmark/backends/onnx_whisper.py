from benchmark.backends.base import BackendBase


class OnnxWhisperBackend(BackendBase):
    def __init__(self):
        self._pipe = None

    @property
    def name(self) -> str:
        return "onnx"

    def setup(self, model_name: str) -> None:
        from transformers import AutoProcessor, pipeline
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        model_id = f"openai/whisper-{model_name}"
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=ort_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )

    def transcribe(self, audio_path: str) -> str:
        result = self._pipe(audio_path, return_timestamps=True)
        return result["text"].strip()


    def cleanup(self) -> None:
        del self._pipe
        self._pipe = None
