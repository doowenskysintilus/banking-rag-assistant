from backend.rag.confidence import ConfidenceAssessment
from backend.rag.context_builder import ContextSource
from backend.rag.guardrails import verify_answer

_HIGH = ConfidenceAssessment(score=0.9, level="high")
_LOW = ConfidenceAssessment(score=0.1, level="low")


def _source(index: int = 1) -> ContextSource:
    return ContextSource(
        index=index,
        document_id="d1",
        chunk_id="c1",
        document_filename="policy.pdf",
        header_path="",
        score=0.9,
        excerpt="Overdraft fee is $35 per item.",
    )


class TestCitationValidity:
    def test_valid_citation_within_range_is_not_flagged(self):
        warnings = verify_answer(
            "The fee is $35 [Source 1].",
            "CONTEXT:\nOverdraft fee is $35 per item.",
            [_source()],
            _HIGH,
        )
        assert warnings == []

    def test_citation_index_beyond_available_sources_is_flagged(self):
        warnings = verify_answer(
            "See [Source 2] for details.", "CONTEXT:\nsome context", [_source()], _HIGH
        )
        assert any("fabricated citation" in w for w in warnings)

    def test_citation_index_zero_is_flagged(self):
        warnings = verify_answer(
            "See [Source 0].", "CONTEXT:\nsome context", [_source()], _HIGH
        )
        assert any("fabricated citation" in w for w in warnings)

    def test_no_sources_at_all_means_any_citation_is_flagged(self):
        warnings = verify_answer("See [Source 1].", "CONTEXT:\nsome context", [], _HIGH)
        assert any("fabricated citation" in w for w in warnings)

    def test_repeated_invalid_citation_is_only_flagged_once(self):
        warnings = verify_answer(
            "[Source 5] and again [Source 5].", "CONTEXT:\nctx", [_source()], _HIGH
        )
        assert sum("fabricated citation" in w for w in warnings) == 1


class TestNumericGrounding:
    def test_number_present_verbatim_in_context_is_not_flagged(self):
        warnings = verify_answer(
            "The overdraft fee is $35 [Source 1].",
            "CONTEXT:\nOverdraft fee is $35 per item.",
            [_source()],
            _HIGH,
        )
        assert warnings == []

    def test_number_absent_from_context_is_flagged(self):
        warnings = verify_answer(
            "The overdraft fee is $99 [Source 1].",
            "CONTEXT:\nOverdraft fee is $35 per item.",
            [_source()],
            _HIGH,
        )
        assert any("not found verbatim" in w for w in warnings)

    def test_percentage_absent_from_context_is_flagged(self):
        warnings = verify_answer(
            "The rate is 7.2%.", "CONTEXT:\nThe rate is 4.5%.", [_source()], _HIGH
        )
        assert any("not found verbatim" in w for w in warnings)

    def test_answer_with_no_numbers_triggers_no_numeric_check(self):
        warnings = verify_answer(
            "There is no overdraft fee mentioned.",
            "CONTEXT:\nsome unrelated context.",
            [_source()],
            _HIGH,
        )
        assert not any("not found verbatim" in w for w in warnings)


class TestConfidenceHedging:
    def test_low_confidence_without_hedge_language_is_flagged(self):
        warnings = verify_answer(
            "The fee is exactly $35.", "CONTEXT:\nsome context", [], _LOW
        )
        assert any("no hedging" in w for w in warnings)

    def test_low_confidence_with_hedge_language_is_not_flagged(self):
        warnings = verify_answer(
            "I do not have sufficient information in the provided documents to answer this question.",
            "CONTEXT:\nsome context",
            [],
            _LOW,
        )
        assert not any("no hedging" in w for w in warnings)

    def test_high_confidence_skips_the_hedging_check_entirely(self):
        warnings = verify_answer(
            "The fee is exactly $35 [Source 1].",
            "CONTEXT:\nOverdraft fee is $35.",
            [_source()],
            _HIGH,
        )
        assert not any("no hedging" in w for w in warnings)


class TestLegalOverreach:
    def test_legal_conclusion_language_is_flagged(self):
        warnings = verify_answer(
            "You are legally entitled to a full refund.", "CONTEXT:\nctx", [], _HIGH
        )
        assert any("legal conclusion" in w for w in warnings)

    def test_ordinary_instruction_with_must_is_not_flagged(self):
        warnings = verify_answer(
            "You must complete form X to open the account.", "CONTEXT:\nctx", [], _HIGH
        )
        assert not any("legal conclusion" in w for w in warnings)

    def test_quoting_a_regulatory_requirement_is_not_flagged(self):
        warnings = verify_answer(
            "Institutions shall maintain a CET1 ratio of at least 4.5% [Source 1].",
            "CONTEXT:\nInstitutions shall maintain a CET1 ratio of at least 4.5%.",
            [_source()],
            _HIGH,
        )
        assert not any("legal conclusion" in w for w in warnings)


class TestVerifyAnswerRobustness:
    def test_clean_answer_yields_no_warnings(self):
        warnings = verify_answer(
            "The overdraft fee is $35 per item [Source 1].",
            "CONTEXT:\nOverdraft fee is $35 per item.",
            [_source()],
            _HIGH,
        )
        assert warnings == []

    def test_never_raises_even_if_a_check_blows_up(self, monkeypatch):
        import backend.rag.guardrails as guardrails_module

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(guardrails_module, "_check_citation_validity", _boom)
        warnings = guardrails_module.verify_answer(
            "anything", "ctx", [_source()], _HIGH
        )
        assert warnings == []
