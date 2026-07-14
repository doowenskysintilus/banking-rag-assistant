from backend.evaluation.types import EvalSample, MetricScore, SampleEvaluation


def test_score_for_returns_the_matching_metric_score():
    sample_eval = SampleEvaluation(
        sample=EvalSample(
            sample_id="s1", question="q", answer="a", contexts=["c"], ground_truth="g"
        ),
        scores=[MetricScore(metric="faithfulness", score=0.8, provider="ragas")],
    )
    assert sample_eval.score_for("faithfulness") == 0.8


def test_score_for_returns_none_when_metric_absent():
    sample_eval = SampleEvaluation(
        sample=EvalSample(
            sample_id="s1", question="q", answer="a", contexts=["c"], ground_truth="g"
        ),
        scores=[],
    )
    assert sample_eval.score_for("faithfulness") is None
