import pytest

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.retrieval.embeddings.selector import (
    probe_availability,
    select_provider_name,
)


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


class TestSelectProviderName:
    def test_pinned_provider_is_returned_even_if_unavailable(self):
        settings = _settings(embedding_provider="openai")
        assert (
            select_provider_name(settings, availability={"openai": False}) == "openai"
        )

    def test_auto_picks_openai_first_when_available(self):
        settings = _settings(embedding_provider="auto")
        availability = {"openai": True, "bge_large": True, "instructor_xl": True}
        assert select_provider_name(settings, availability) == "openai"

    def test_auto_falls_back_to_bge_large_when_openai_unavailable(self):
        settings = _settings(embedding_provider="auto")
        availability = {"openai": False, "bge_large": True, "instructor_xl": True}
        assert select_provider_name(settings, availability) == "bge_large"

    def test_auto_falls_back_to_instructor_xl_last(self):
        settings = _settings(embedding_provider="auto")
        availability = {"openai": False, "bge_large": False, "instructor_xl": True}
        assert select_provider_name(settings, availability) == "instructor_xl"

    def test_auto_raises_when_nothing_is_available(self):
        settings = _settings(embedding_provider="auto")
        availability = {"openai": False, "bge_large": False, "instructor_xl": False}
        with pytest.raises(ExternalServiceError):
            select_provider_name(settings, availability)

    def test_auto_treats_missing_keys_as_unavailable(self):
        settings = _settings(embedding_provider="auto")
        with pytest.raises(ExternalServiceError):
            select_provider_name(settings, availability={})


class TestProbeAvailability:
    def test_openai_requires_both_api_key_and_importable_package(self):
        settings = _settings(openai_api_key="")
        availability = probe_availability(settings)
        assert availability["openai"] is False

    def test_openai_available_when_key_set_and_package_installed(self):
        settings = _settings(openai_api_key="sk-fake")
        availability = probe_availability(settings)
        # openai is a hard dependency of this project (requirements.txt), so this
        # reflects real installation state rather than a mock.
        assert availability["openai"] is True

    def test_returns_a_bool_for_every_known_provider(self):
        settings = _settings()
        availability = probe_availability(settings)
        assert set(availability.keys()) == {"openai", "bge_large", "instructor_xl"}
        assert all(isinstance(v, bool) for v in availability.values())
