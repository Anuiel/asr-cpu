from benchmark.backends.base import BackendBase

BACKENDS: dict[str, str] = {
    "openai-whisper": "benchmark.backends.openai_whisper.OpenAIWhisperBackend",
    "faster-whisper": "benchmark.backends.faster_whisper.FasterWhisperBackend",
    "onnx": "benchmark.backends.onnx_whisper.OnnxWhisperBackend",
    "torch-opt": "benchmark.backends.torch_whisper.TorchOptBackend",
}


def get_backend(name: str) -> BackendBase:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    module_path, class_name = BACKENDS[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()
