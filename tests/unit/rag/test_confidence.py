from backend.rag.confidence import assess_confidence


def test_no_scores_is_low_confidence():
    result = assess_confidence([])
    assert result.score == 0.0
    assert result.level == "low"


def test_low_confidence_signal_overrides_scores():
    result = assess_confidence([0.99, 0.95], low_confidence_signal=True)
    assert result.score == 0.0
    assert result.level == "low"


def test_high_scores_yield_high_confidence():
    result = assess_confidence([0.95, 0.9, 0.85])
    assert result.level == "high"
    assert result.score >= 0.75


def test_moderate_scores_yield_medium_confidence():
    result = assess_confidence([0.5, 0.45])
    assert result.level == "medium"


def test_weak_scores_yield_low_confidence():
    result = assess_confidence([0.1, 0.05])
    assert result.level == "low"


def test_single_score_still_works():
    result = assess_confidence([0.9])
    assert result.level == "high"


def test_score_is_bounded_between_zero_and_one():
    result = assess_confidence([1.5, 1.2])  # hypothetical out-of-range inputs
    assert 0.0 <= result.score <= 1.0


def test_top_score_weighted_more_than_the_rest():
    # Same top score, but one case has weaker supporting scores — the combined
    # score should still favor the strong top score, just slightly lower.
    strong_support = assess_confidence([0.9, 0.85, 0.8])
    weak_support = assess_confidence([0.9, 0.1, 0.1])
    assert strong_support.score > weak_support.score
    assert weak_support.score > 0  # the top score still counts for something
