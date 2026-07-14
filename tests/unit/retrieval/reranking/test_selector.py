import pytest

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.retrieval.reranking.selector import (
    probe_availability,
    select_provider_name,
)


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


class TestSelectProviderName:
    def test_pinned_provider_is_returned_even_if_unavailable(self):
        settings = _settings(enterprise_reranker_provider="bge")
        assert select_provider_name(settings, availability={"bge": False}) == "bge"

    def test_auto_picks_bge_first_when_available(self):
        settings = _settings(enterprise_reranker_provider="auto")
        availability = {"bge": True, "cross_encoder": True}
        assert select_provider_name(settings, availability) == "bge"

    def test_auto_falls_back_to_cross_encoder_when_bge_unavailable(self):
        settings = _settings(enterprise_reranker_provider="auto")
        availability = {"bge": False, "cross_encoder": True}
        assert select_provider_name(settings, availability) == "cross_encoder"

    def test_auto_raises_when_nothing_is_available(self):
        settings = _settings(enterprise_reranker_provider="auto")
        with pytest.raises(ExternalServiceError):
            select_provider_name(
                settings, availability={"bge": False, "cross_encoder": False}
            )

    def test_auto_treats_missing_keys_as_unavailable(self):
        settings = _settings(enterprise_reranker_provider="auto")
        with pytest.raises(ExternalServiceError):
            select_provider_name(settings, availability={})


class TestProbeAvailability:
    def test_returns_a_bool_for_every_known_provider(self):
        availability = probe_availability(_settings())
        assert set(availability.keys()) == {"bge", "cross_encoder"}
        assert all(isinstance(v, bool) for v in availability.values())
