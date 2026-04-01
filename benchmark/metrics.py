import jiwer
from whisper.normalizers import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    return _normalizer(text)


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    refs_f, hyps_f = zip(*pairs)
    return jiwer.wer(list(refs_f), list(hyps_f))


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0
    refs_f, hyps_f = zip(*pairs)
    return jiwer.cer(list(refs_f), list(hyps_f))
