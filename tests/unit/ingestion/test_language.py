from backend.ingestion.language import detect_language


def test_detects_english():
    text = (
        "This banking policy document describes the terms and conditions "
        "applicable to overdraft protection on checking accounts."
    )
    result = detect_language(text)
    assert result.code == "en"
    assert 0.0 < result.confidence <= 1.0


def test_detects_french():
    text = (
        "Ce document decrit les conditions generales applicables aux comptes "
        "courants et aux decouverts autorises pour les clients particuliers."
    )
    result = detect_language(text)
    assert result.code == "fr"
    assert 0.0 < result.confidence <= 1.0


def test_returns_unknown_for_empty_text():
    result = detect_language("")
    assert result.code == "unknown"
    assert result.confidence == 0.0


def test_returns_unknown_for_very_short_text():
    result = detect_language("Hi")
    assert result.code == "unknown"
    assert result.confidence == 0.0


def test_returns_unknown_on_undetectable_text():
    # Digits/punctuation only: no linguistic features for langdetect to work with.
    result = detect_language("12345 !!! 000 --- 6789 %%% 111")
    assert result.code == "unknown"
    assert result.confidence == 0.0
