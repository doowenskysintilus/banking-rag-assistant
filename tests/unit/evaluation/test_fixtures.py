from backend.evaluation.fixtures import GOLDEN_SAMPLES


def test_golden_samples_have_unique_ids():
    ids = [s.sample_id for s in GOLDEN_SAMPLES]
    assert len(ids) == len(set(ids))


def test_every_sample_has_non_empty_required_fields():
    for sample in GOLDEN_SAMPLES:
        assert sample.question.strip()
        assert sample.answer.strip()
        assert sample.ground_truth.strip()
        assert sample.contexts
        assert all(c.strip() for c in sample.contexts)


def test_includes_at_least_one_defective_answer_for_the_metrics_to_catch():
    """An evaluation golden set that's all perfect answers hasn't actually
    exercised the metrics it's meant to demonstrate."""
    fabricated = next(
        s for s in GOLDEN_SAMPLES if s.sample_id == "wire-transfer-fabricated-number"
    )
    assert "$60" in fabricated.answer
    assert "$60" not in " ".join(fabricated.contexts)
