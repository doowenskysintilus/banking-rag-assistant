from backend.ingestion.cleaning import clean_text


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""


def test_collapses_repeated_whitespace_and_blank_lines():
    raw = "Hello    world\n\n\n\n\nSecond   paragraph"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\n\nSecond paragraph"


def test_strips_page_number_footer_lines():
    raw = "Introduction text.\n\nPage 3 of 10\n\nMore content."
    cleaned = clean_text(raw)
    assert "Page 3 of 10" not in cleaned
    assert "Introduction text." in cleaned
    assert "More content." in cleaned


def test_strips_lone_numeric_footer_line():
    raw = "Some policy text.\n\n42\n\nNext section."
    cleaned = clean_text(raw)
    lines = cleaned.split("\n")
    assert "42" not in lines
    assert "Some policy text." in cleaned
    assert "Next section." in cleaned


def test_removes_control_characters_but_keeps_newlines_and_tabs():
    raw = "Valid\x00text\x07with\x1fcontrol\tchars\nand newline"
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "\x1f" not in cleaned
    assert "\n" in cleaned


def test_normalizes_unicode_variants_nfkc():
    # U+FB03 LATIN SMALL LIGATURE FFI NFKC-normalizes to the plain letters "ffi"
    raw = "eﬃciency"
    cleaned = clean_text(raw)
    assert cleaned == "efficiency"


def test_collapses_long_punctuation_runs():
    raw = "Section one..........Section two"
    cleaned = clean_text(raw)
    assert "..........." not in cleaned
    assert "...." in cleaned


def test_strips_leading_and_trailing_whitespace():
    raw = "   \n  Hello world  \n   "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world"


def test_normalizes_windows_and_mac_line_endings():
    raw = "line one\r\nline two\rline three"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert cleaned == "line one\nline two\nline three"
