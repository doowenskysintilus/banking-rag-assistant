"""Automatic reranker provider selection — mirrors
`backend.retrieval.embeddings.selector`: a pure, dependency-free decision function
plus a real-world availability probe.
"""

from __future__ import annotations

import importlib.util

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError

# BGE first: a reranker trained specifically for this task, and multilingual
# (v2-m3) — a meaningfully stronger default than a generic MS-MARCO cross-encoder
# for a bank operating across languages.
PROVIDER_PRIORITY: tuple[str, ...] = ("bge", "cross_encoder")


def select_provider_name(settings: Settings, availability: dict[str, bool]) -> str:
    if settings.enterprise_reranker_provider != "auto":
        return settings.enterprise_reranker_provider

    for name in PROVIDER_PRIORITY:
        if availability.get(name):
            return name

    raise ExternalServiceError(
        "No reranker provider is available for ENTERPRISE_RERANKER_PROVIDER=auto. "
        "Install sentence-transformers (covers both bge and cross_encoder) or FlagEmbedding."
    )


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_availability(settings: Settings) -> dict[str, bool]:
    return {
        "bge": _is_importable("FlagEmbedding")
        or _is_importable("sentence_transformers"),
        "cross_encoder": _is_importable("sentence_transformers"),
    }
