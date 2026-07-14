"""Language detection for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass

_MIN_CHARS_FOR_DETECTION = 20


@dataclass
class LanguageDetectionResult:
    code: str  # ISO 639-1 code (e.g. "en", "fr"), or "unknown"
    confidence: float  # 0.0-1.0; 0.0 when detection was skipped/failed


def detect_language(text: str) -> LanguageDetectionResult:
    """Best-effort language detection. Never raises — falls back to "unknown"
    on empty/too-short text or a detector failure (e.g. no linguistic features).
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS_FOR_DETECTION:
        return LanguageDetectionResult(code="unknown", confidence=0.0)

    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    # Deterministic results across runs/processes (langdetect samples internally).
    DetectorFactory.seed = 0

    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        return LanguageDetectionResult(code="unknown", confidence=0.0)

    if not candidates:
        return LanguageDetectionResult(code="unknown", confidence=0.0)

    best = candidates[0]
    return LanguageDetectionResult(code=best.lang, confidence=round(best.prob, 4))
