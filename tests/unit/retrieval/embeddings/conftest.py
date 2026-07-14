"""Shared fixtures for embeddings module tests."""

from __future__ import annotations

import pytest

from tests.unit.retrieval.embeddings.fakes import FakeProvider


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
