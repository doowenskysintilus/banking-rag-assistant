from backend.llm.prompts import HEDGE_PHRASES, INSUFFICIENT_INFO_PHRASE, SYSTEM_PROMPT


def test_insufficient_info_phrase_is_embedded_in_the_system_prompt():
    assert INSUFFICIENT_INFO_PHRASE in SYSTEM_PROMPT


def test_hedge_phrases_include_the_lowercased_insufficient_info_phrase():
    assert INSUFFICIENT_INFO_PHRASE.lower() in HEDGE_PHRASES


def test_hedge_phrases_have_no_duplicates():
    assert len(HEDGE_PHRASES) == len(set(HEDGE_PHRASES))


def test_system_prompt_covers_every_required_section():
    expected_headings = [
        "ABSOLUTE GROUNDING RULE",
        "CITATION REQUIREMENT",
        "NUMERICAL PRECISION",
        "WHEN CONTEXT DOES NOT CONTAIN THE ANSWER",
        "LEGAL AND REGULATORY INTERPRETATION",
        "CONFLICTING OR MULTIPLE SOURCES",
        "CUSTOMER AND ACCOUNT DATA",
        "CONFIDENCE AND HEDGING",
        "STYLE",
        "WORKED EXAMPLES",
    ]
    for heading in expected_headings:
        assert heading in SYSTEM_PROMPT, f"missing section: {heading}"


def test_system_prompt_instructs_never_inventing_source_numbers():
    assert "Never invent a source number" in SYSTEM_PROMPT


def test_system_prompt_includes_a_positive_and_a_negative_worked_example():
    assert "GOOD ANSWER" in SYSTEM_PROMPT
    assert "BAD ANSWER (never do this)" in SYSTEM_PROMPT


def test_system_prompt_does_not_claim_to_be_a_lawyer():
    assert "not a lawyer" in SYSTEM_PROMPT
