from dataclasses import dataclass, field

from backend.evaluation.dataset import EvalDatasetBuilder, LabeledQuestion


@dataclass
class _FakeRAGResponse:
    answer: str
    source_texts: list[str] = field(default_factory=list)


class _FakeRAGEngine:
    def __init__(self, responses: dict[str, _FakeRAGResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str, str | None]] = []

    async def answer(
        self, db, question, owner_id, *, conversation_id=None, history=None
    ):
        self.calls.append((question, owner_id, conversation_id))
        return self._responses[question]


async def test_builds_one_eval_sample_per_labeled_question():
    engine = _FakeRAGEngine(
        {
            "What is the overdraft fee?": _FakeRAGResponse(
                answer="It is $35 [Source 1].",
                source_texts=["Overdraft fee is $35 per item."],
            )
        }
    )
    builder = EvalDatasetBuilder(engine)
    questions = [
        LabeledQuestion(
            sample_id="q1",
            question="What is the overdraft fee?",
            ground_truth="$35 per item.",
        )
    ]

    samples = await builder.build(db=None, owner_id="owner-1", questions=questions)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "q1"
    assert sample.question == "What is the overdraft fee?"
    assert sample.answer == "It is $35 [Source 1]."
    assert sample.contexts == ["Overdraft fee is $35 per item."]
    assert sample.ground_truth == "$35 per item."


async def test_calls_the_engine_once_per_question_with_no_conversation():
    engine = _FakeRAGEngine(
        {
            "Q1": _FakeRAGResponse(answer="A1"),
            "Q2": _FakeRAGResponse(answer="A2"),
        }
    )
    builder = EvalDatasetBuilder(engine)
    questions = [
        LabeledQuestion(sample_id="s1", question="Q1", ground_truth="G1"),
        LabeledQuestion(sample_id="s2", question="Q2", ground_truth="G2"),
    ]

    samples = await builder.build(db=None, owner_id="owner-42", questions=questions)

    assert [s.answer for s in samples] == ["A1", "A2"]
    assert engine.calls == [("Q1", "owner-42", None), ("Q2", "owner-42", None)]


async def test_empty_question_list_returns_empty_samples():
    engine = _FakeRAGEngine({})
    builder = EvalDatasetBuilder(engine)
    samples = await builder.build(db=None, owner_id="owner-1", questions=[])
    assert samples == []
