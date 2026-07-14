"""Compares reranking strategies — no-rerank baseline vs BGE vs generic
cross-encoder — on the labeled banking examples in `fixtures.py`, reporting
standard IR metrics plus `grounding_precision` (the hallucination-risk proxy).

Run as a script:
    python -m backend.retrieval.reranking.benchmark

A provider is skipped automatically if its model can't actually be loaded in the
current environment (e.g. no network to fetch weights) — the baseline comparison
still runs everywhere.
"""

from __future__ import annotations

import asyncio

from backend.core.config import Settings
from backend.retrieval.reranking.bge_reranker import BGERerankerProvider
from backend.retrieval.reranking.cross_encoder import CrossEncoderRerankerProvider
from backend.retrieval.reranking.evaluation import EvaluationReport, RerankEvaluator
from backend.retrieval.reranking.fixtures import EVAL_EXAMPLES
from backend.retrieval.reranking.types import RerankerProvider

_DEFAULT_K = 5
_DEFAULT_RELEVANCE_THRESHOLD = 0.5


def _default_settings() -> Settings:
    return Settings(secret_key="benchmark-script-placeholder-key")


def _try_load_provider(provider: RerankerProvider) -> RerankerProvider | None:
    try:
        asyncio.run(provider.score("warm up query", ["warm up document"]))
        return provider
    except Exception:
        return None


def run_benchmark(
    k: int = _DEFAULT_K,
    relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
    settings: Settings | None = None,
) -> dict[str, EvaluationReport]:
    settings = settings or _default_settings()
    evaluator = RerankEvaluator(k=k, relevance_threshold=relevance_threshold)

    results: dict[str, EvaluationReport] = {
        "baseline": asyncio.run(
            evaluator.evaluate(EVAL_EXAMPLES, provider_name="baseline")
        )
    }

    candidate_providers: list[RerankerProvider] = [
        BGERerankerProvider(settings),
        CrossEncoderRerankerProvider(settings),
    ]
    for provider in candidate_providers:
        loaded = _try_load_provider(provider)
        if loaded is None:
            continue
        results[loaded.name] = asyncio.run(
            evaluator.evaluate(
                EVAL_EXAMPLES, score_fn=loaded.score, provider_name=loaded.name
            )
        )

    return results


def format_report(results: dict[str, EvaluationReport]) -> str:
    header = (
        f"{'provider':<14} {'n':>3} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'MAP':>6} "
        f"{'nDCG@k':>7} {'grounding':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, report in results.items():
        lines.append(
            f"{name:<14} {report.num_examples:>3} {report.precision_at_k:>6.2f} {report.recall_at_k:>6.2f} "
            f"{report.mrr:>6.2f} {report.map_score:>6.2f} {report.ndcg_at_k:>7.2f} {report.grounding_precision:>10.2f}"
        )
    lines.append("")
    lines.append(
        "Note: baseline's 'grounding' isn't a calibrated confidence signal (no model "
        "behind its scores) — only meaningful for the real reranker providers."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(run_benchmark()))
