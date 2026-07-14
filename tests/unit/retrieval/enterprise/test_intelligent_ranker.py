from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.documents import Document

from backend.retrieval.enterprise.intelligent_ranker import BoostRule, IntelligentRanker


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class TestNoRules:
    def test_ranks_purely_by_base_score_when_no_rules(self):
        ranker = IntelligentRanker(rules=[])
        documents = [
            Document(page_content="low", metadata={"score": 0.2}),
            Document(page_content="high", metadata={"score": 0.9}),
        ]
        ranked = ranker.rank(documents)
        assert [d.page_content for d in ranked] == ["high", "low"]
        assert ranked[0].metadata["final_score"] == 0.9

    def test_prefers_rerank_score_over_raw_score_when_both_present(self):
        ranker = IntelligentRanker(rules=[])
        documents = [
            Document(page_content="doc", metadata={"score": 0.1, "rerank_score": 0.8})
        ]
        ranked = ranker.rank(documents)
        assert ranked[0].metadata["final_score"] == 0.8

    def test_missing_score_defaults_to_zero(self):
        ranker = IntelligentRanker(rules=[])
        ranked = ranker.rank([Document(page_content="doc", metadata={})])
        assert ranked[0].metadata["final_score"] == 0.0

    def test_does_not_mutate_input_documents(self):
        ranker = IntelligentRanker(rules=[])
        original = Document(page_content="doc", metadata={"score": 0.5})
        ranker.rank([original])
        assert "final_score" not in original.metadata


class TestCustomRules:
    def test_boost_rule_shifts_ranking_order(self):
        rule = BoostRule(
            name="preferred_language",
            weight=1.0,
            signal=lambda m: 1.0 if m.get("language") == "fr" else 0.0,
        )
        ranker = IntelligentRanker(rules=[rule])

        documents = [
            Document(
                page_content="english, higher base score",
                metadata={"score": 0.6, "language": "en"},
            ),
            Document(
                page_content="french, lower base score",
                metadata={"score": 0.5, "language": "fr"},
            ),
        ]
        ranked = ranker.rank(documents)

        assert ranked[0].page_content == "french, lower base score"
        assert ranked[0].metadata["boosts"] == {"preferred_language": 1.0}

    def test_multiple_rules_are_additive(self):
        rules = [
            BoostRule(name="a", weight=0.1, signal=lambda m: 1.0),
            BoostRule(name="b", weight=0.2, signal=lambda m: 1.0),
        ]
        ranker = IntelligentRanker(rules=rules)
        ranked = ranker.rank([Document(page_content="doc", metadata={"score": 0.5})])
        assert ranked[0].metadata["final_score"] == pytest.approx(0.5 + 0.1 + 0.2)


class TestDefaultRules:
    def test_recency_boosts_newer_documents_over_older_ones_with_equal_base_score(self):
        ranker = IntelligentRanker.with_default_rules(
            recency_weight=0.1, recency_half_life_days=30
        )
        documents = [
            Document(
                page_content="old", metadata={"score": 0.5, "created_at": _iso(365)}
            ),
            Document(
                page_content="new", metadata={"score": 0.5, "created_at": _iso(0)}
            ),
        ]
        ranked = ranker.rank(documents)
        assert ranked[0].page_content == "new"

    def test_recency_signal_gracefully_ignores_missing_created_at(self):
        ranker = IntelligentRanker.with_default_rules(recency_weight=0.1)
        ranked = ranker.rank([Document(page_content="doc", metadata={"score": 0.5})])
        assert ranked[0].metadata["boosts"]["recency"] == 0.0

    def test_recency_signal_gracefully_ignores_malformed_created_at(self):
        ranker = IntelligentRanker.with_default_rules(recency_weight=0.1)
        ranked = ranker.rank(
            [
                Document(
                    page_content="doc",
                    metadata={"score": 0.5, "created_at": "not-a-date"},
                )
            ]
        )
        assert ranked[0].metadata["boosts"]["recency"] == 0.0

    def test_structural_context_boosts_chunks_with_header_path(self):
        ranker = IntelligentRanker.with_default_rules(
            recency_weight=0, structural_context_weight=0.1
        )
        documents = [
            Document(page_content="no context", metadata={"score": 0.5}),
            Document(
                page_content="has context",
                metadata={"score": 0.5, "header_path": "Article 5"},
            ),
        ]
        ranked = ranker.rank(documents)
        assert ranked[0].page_content == "has context"

    def test_zero_weight_disables_a_default_rule(self):
        ranker = IntelligentRanker.with_default_rules(
            recency_weight=0, structural_context_weight=0
        )
        ranked = ranker.rank(
            [
                Document(
                    page_content="doc",
                    metadata={"score": 0.5, "header_path": "Article 5"},
                )
            ]
        )
        assert ranked[0].metadata["boosts"] == {}
