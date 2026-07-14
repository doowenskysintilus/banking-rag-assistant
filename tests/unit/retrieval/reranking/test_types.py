import pytest

from backend.retrieval.reranking.types import sigmoid


def test_sigmoid_of_zero_is_one_half():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_is_bounded_between_zero_and_one():
    # At extreme magnitudes float64 precision rounds the result to exactly 0.0/1.0
    # (e.g. exp(-100) is far below float64's epsilon relative to 1.0) — that's
    # expected, not a bug, so the bound is inclusive here.
    assert 0.0 <= sigmoid(-100.0) <= 1.0
    assert 0.0 <= sigmoid(100.0) <= 1.0
    # A more moderate magnitude stays strictly within the open interval.
    assert 0.0 < sigmoid(-20.0) < 1.0
    assert 0.0 < sigmoid(20.0) < 1.0


def test_sigmoid_is_monotonically_increasing():
    xs = [-10, -1, 0, 1, 10]
    values = [sigmoid(x) for x in xs]
    assert values == sorted(values)


def test_sigmoid_does_not_overflow_on_extreme_values():
    # Must not raise OverflowError regardless of sign.
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-9)
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-9)
