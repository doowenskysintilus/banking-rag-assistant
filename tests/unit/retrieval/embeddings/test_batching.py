import asyncio

import numpy as np
import pytest

from backend.retrieval.embeddings.batching import make_batches, run_batches_in_parallel


class TestMakeBatches:
    def test_splits_into_even_batches(self):
        items = [str(i) for i in range(10)]
        batches = make_batches(items, batch_size=3)
        assert batches == [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"], ["9"]]

    def test_batch_size_larger_than_input_returns_single_batch(self):
        items = ["a", "b"]
        assert make_batches(items, batch_size=10) == [["a", "b"]]

    def test_empty_input_returns_no_batches(self):
        assert make_batches([], batch_size=5) == []

    def test_rejects_non_positive_batch_size(self):
        with pytest.raises(ValueError):
            make_batches(["a"], batch_size=0)
        with pytest.raises(ValueError):
            make_batches(["a"], batch_size=-1)


class TestRunBatchesInParallel:
    async def test_returns_empty_list_for_no_batches(self):
        async def embed_fn(batch):
            raise AssertionError("should never be called")

        assert await run_batches_in_parallel([], embed_fn, max_concurrency=4) == []

    async def test_preserves_batch_order_regardless_of_completion_order(self):
        # Batch 0 finishes last, batch 2 finishes first — results must still come
        # back in the original (input) order.
        delays = {0: 0.03, 1: 0.01, 2: 0.0}

        async def embed_fn(batch: list[str]) -> np.ndarray:
            index = int(batch[0])
            await asyncio.sleep(delays[index])
            return np.full((1, 2), index, dtype="float32")

        batches = [["0"], ["1"], ["2"]]
        results = await run_batches_in_parallel(batches, embed_fn, max_concurrency=3)

        assert [int(r[0, 0]) for r in results] == [0, 1, 2]

    async def test_limits_concurrent_in_flight_calls(self):
        max_observed_concurrency = 0
        current_concurrency = 0
        lock = asyncio.Lock()

        async def embed_fn(batch: list[str]) -> np.ndarray:
            nonlocal max_observed_concurrency, current_concurrency
            async with lock:
                current_concurrency += 1
                max_observed_concurrency = max(
                    max_observed_concurrency, current_concurrency
                )
            await asyncio.sleep(0.02)
            async with lock:
                current_concurrency -= 1
            return np.zeros((1, 1), dtype="float32")

        batches = [[str(i)] for i in range(10)]
        await run_batches_in_parallel(batches, embed_fn, max_concurrency=2)

        assert max_observed_concurrency <= 2

    async def test_a_single_failing_batch_propagates_the_exception(self):
        async def embed_fn(batch: list[str]) -> np.ndarray:
            if batch[0] == "bad":
                raise RuntimeError("boom")
            return np.zeros((1, 1), dtype="float32")

        with pytest.raises(RuntimeError):
            await run_batches_in_parallel(
                [["ok"], ["bad"]], embed_fn, max_concurrency=2
            )
