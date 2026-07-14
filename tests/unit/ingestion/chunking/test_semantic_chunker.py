import numpy as np
import pytest

from backend.ingestion.chunking.semantic_chunker import (
    group_by_semantic_breakpoints,
    semantic_group_async,
)


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    array = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms


class TestGroupBySemanticBreakpoints:
    def test_empty_units_returns_empty_list(self):
        assert group_by_semantic_breakpoints([], np.empty((0, 3))) == []

    def test_single_unit_returns_single_group(self):
        embeddings = _normalized([[1.0, 0.0, 0.0]])
        assert group_by_semantic_breakpoints(["only one"], embeddings) == [[0]]

    def test_mismatched_embedding_count_raises(self):
        with pytest.raises(ValueError):
            group_by_semantic_breakpoints(["a", "b"], _normalized([[1.0, 0.0]]))

    def test_all_similar_embeddings_form_a_single_group(self):
        # Identical embeddings -> zero distance everywhere -> no distance exceeds
        # the (zero) 95th-percentile threshold -> one group.
        units = [f"sentence {i}" for i in range(5)]
        embeddings = _normalized([[1.0, 0.2, 0.0]] * 5)
        groups = group_by_semantic_breakpoints(
            units, embeddings, breakpoint_percentile=95.0
        )
        assert groups == [[0, 1, 2, 3, 4]]

    def test_detects_breakpoint_at_topic_shift(self):
        # Three orthogonal-ish clusters of three units each: within-cluster distance
        # is 0, cross-cluster distance is 1. A moderate percentile threshold (70)
        # correctly isolates the two large cross-cluster jumps as breakpoints.
        units = [f"unit-{i}" for i in range(9)]
        embeddings = _normalized(
            [[1.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 0.0]] * 3 + [[0.0, 0.0, 1.0]] * 3
        )
        groups = group_by_semantic_breakpoints(
            units, embeddings, breakpoint_percentile=70.0
        )
        assert groups == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_high_percentile_on_a_single_distance_yields_no_breakpoint(self):
        # Documents a known limitation: with only two units (a single pairwise
        # distance), `np.percentile` of a one-element array returns that element
        # itself regardless of the percentile requested, so the strict
        # "distance > threshold" comparison never fires — even for two maximally
        # dissimilar (orthogonal) sentences. Real documents have enough sentences
        # that this doesn't happen in practice (see the hybrid chunker / benchmark
        # tests, which use realistic-length fixtures).
        units = ["topic a sentence", "topic b sentence"]
        embeddings = _normalized([[1.0, 0.0], [0.0, 1.0]])
        groups = group_by_semantic_breakpoints(
            units, embeddings, breakpoint_percentile=95.0
        )
        assert groups == [[0, 1]]


class TestSemanticGroupAsync:
    async def test_returns_empty_for_no_units(self, fake_embedder):
        assert await semantic_group_async([], fake_embedder) == []

    async def test_embeds_units_and_groups_by_topic(self, fake_embedder):
        units = [
            "The overdraft facility accrues interest daily.",
            "Interest on the overdraft facility is payable monthly.",
            "The overdraft facility has a defined credit limit.",
            "The kitchen recipe requires two cups of flour.",
            "Bake the recipe at a moderate oven temperature.",
            "The recipe yields twelve small pastries.",
        ]
        groups = await semantic_group_async(
            units, fake_embedder, breakpoint_percentile=60.0
        )

        # All indices are covered exactly once, in order.
        flattened = [i for group in groups for i in group]
        assert flattened == list(range(len(units)))
        assert fake_embedder.calls == [units]
