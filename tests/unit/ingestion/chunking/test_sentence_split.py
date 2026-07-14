from backend.ingestion.chunking.sentence_split import split_sentences


def test_splits_simple_sentences():
    text = "The Borrower shall repay the loan. Interest accrues daily."
    assert split_sentences(text) == [
        "The Borrower shall repay the loan.",
        "Interest accrues daily.",
    ]


def test_returns_empty_list_for_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_does_not_split_on_single_word_abbreviations():
    text = "Please see Art. 5 for details. The Bank, Inc. shall not be liable."
    sentences = split_sentences(text)
    assert sentences == [
        "Please see Art. 5 for details.",
        "The Bank, Inc. shall not be liable.",
    ]


def test_does_not_split_on_decimal_numbers():
    text = "The ratio must exceed 1.25 at all times. This is tested quarterly."
    sentences = split_sentences(text)
    assert sentences[0] == "The ratio must exceed 1.25 at all times."
    assert len(sentences) == 2


def test_does_not_split_on_multi_part_abbreviations():
    text = "Fees may apply, e.g. overdraft fees. The U.S. rules differ from the U.K. rules."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "e.g." in sentences[0]
    assert "U.S." in sentences[1]
    assert "U.K." in sentences[1]


def test_does_not_split_clause_numbering_mid_sentence():
    text = "Article 5.2 - Prepayment. The Borrower may prepay the facility at any time."
    sentences = split_sentences(text)
    assert sentences[0].startswith("Article 5.2 - Prepayment.")


def test_splits_on_question_and_exclamation_marks():
    text = "Is the facility fully drawn? Yes, it is fully drawn! No further amounts are available."
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_handles_quoted_defined_terms_without_premature_split():
    text = (
        'The Borrower (the "Client") shall notify the Bank. Notice must be in writing.'
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert 'the "Client"' in sentences[0]


def test_preserves_percentage_and_currency_decimals():
    text = "The margin is 2.5 per cent per annum. The fee is $12.50 per transaction."
    sentences = split_sentences(text)
    assert "2.5 per cent" in sentences[0]
    assert "$12.50" in sentences[1]
