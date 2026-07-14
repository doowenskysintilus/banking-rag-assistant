"""Batching and bounded-concurrency parallelization for embedding requests.

Kept provider-agnostic: works identically for an I/O-bound API call (OpenAI) and a
CPU/GPU-bound local model forward pass (BGE, Instructor-XL), since both are wrapped
behind the same `async def embed_batch(texts) -> np.ndarray` signature.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import numpy as np


def make_batches(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def run_batches_in_parallel(
    batches: list[list[str]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    max_concurrency: int,
) -> list[np.ndarray]:
    """Runs `embed_fn` over every batch with at most `max_concurrency` in flight at
    once. Results are returned in the same order as `batches` regardless of which
    batch finishes first — `asyncio.gather` preserves input order.
    """
    if not batches:
        return []

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(batch: list[str]) -> np.ndarray:
        async with semaphore:
            return await embed_fn(batch)

    return list(await asyncio.gather(*[_run_one(batch) for batch in batches]))
