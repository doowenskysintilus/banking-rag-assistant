from backend.llm.prompts.templates import (
    build_user_turn,
    is_legal_or_regulatory_question,
)


class TestIsLegalOrRegulatoryQuestion:
    def test_detects_english_legal_keywords(self):
        assert is_legal_or_regulatory_question("Am I legally entitled to a refund?")
        assert is_legal_or_regulatory_question("Is this compliant with the regulation?")
        assert is_legal_or_regulatory_question("Can the bank sue me for this?")

    def test_detects_french_legal_keywords(self):
        assert is_legal_or_regulatory_question("Est-ce légalement obligatoire ?")
        assert is_legal_or_regulatory_question(
            "Cela respecte-t-il la réglementation en vigueur ?"
        )
        assert is_legal_or_regulatory_question("Quelle est la conformité exigée ?")

    def test_plain_banking_question_is_not_flagged(self):
        assert not is_legal_or_regulatory_question(
            "What is the overdraft fee for premium checking?"
        )

    def test_is_case_insensitive(self):
        assert is_legal_or_regulatory_question("IS THIS ILLEGAL?")


class TestBuildUserTurn:
    def test_includes_context_and_question(self):
        turn = build_user_turn(
            "What is the fee?", "CONTEXT:\nsome context", legal_flag=False
        )
        assert "CONTEXT:\nsome context" in turn
        assert "QUESTION:\nWhat is the fee?" in turn

    def test_empty_context_falls_back_to_no_context_text(self):
        turn = build_user_turn("q", "", legal_flag=False)
        assert "No relevant internal documents were found" in turn

    def test_low_confidence_adds_the_low_confidence_addendum(self):
        turn = build_user_turn("q", "ctx", confidence_level="low", legal_flag=False)
        assert "low retrieval confidence" in turn

    def test_medium_confidence_also_adds_the_addendum(self):
        turn = build_user_turn("q", "ctx", confidence_level="medium", legal_flag=False)
        assert "low retrieval confidence" in turn

    def test_high_confidence_does_not_add_the_addendum(self):
        turn = build_user_turn("q", "ctx", confidence_level="high", legal_flag=False)
        assert "low retrieval confidence" not in turn

    def test_explicit_legal_flag_true_adds_the_legal_addendum(self):
        turn = build_user_turn("Completely unrelated question", "ctx", legal_flag=True)
        assert "compliance or legal counsel" in turn

    def test_explicit_legal_flag_false_suppresses_auto_detection(self):
        turn = build_user_turn("Is this legally required?", "ctx", legal_flag=False)
        assert "compliance or legal counsel" not in turn

    def test_legal_flag_defaults_to_auto_detection_when_omitted(self):
        turn = build_user_turn("Is this legally required?", "ctx")
        assert "compliance or legal counsel" in turn

    def test_addenda_appear_before_the_question_section(self):
        turn = build_user_turn("Is this legal?", "ctx", confidence_level="low")
        question_pos = turn.index("QUESTION:")
        assert turn.index("low retrieval confidence") < question_pos
        assert turn.index("compliance or legal counsel") < question_pos
