"""Automatic embedding provider selection.

The selection *decision* (`select_provider_name`) is pure and dependency-free so it
can be unit tested without installing any of the optional ML libraries — callers
supply an `availability` mapping (real probing lives in `probe_availability`, which
does the actual optional imports).
"""

from __future__ import annotations

import importlib.util

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError

# Preference order when EMBEDDING_PROVIDER=auto: a managed API (best quality, zero
# local compute/memory) beats a local model, and among local models BGE-Large is
# smaller/faster to load than Instructor-XL.
PROVIDER_PRIORITY: tuple[str, ...] = ("openai", "bge_large", "instructor_xl")


def select_provider_name(settings: Settings, availability: dict[str, bool]) -> str:
    """Returns the provider name to use, given explicit config and availability.

    If `settings.embedding_provider` names a specific provider, it's returned as-is
    (even if unavailable — the caller will get a clear error when it tries to
    actually build/use that provider, rather than a silent substitution). Only
    "auto" triggers the priority-ordered fallback search.
    """
    if settings.embedding_provider != "auto":
        return settings.embedding_provider

    for name in PROVIDER_PRIORITY:
        if availability.get(name):
            return name

    raise ExternalServiceError(
        "No embedding provider is available for EMBEDDING_PROVIDER=auto. "
        "Configure OPENAI_API_KEY, or ensure sentence-transformers "
        "(for bge_large) or InstructorEmbedding (for instructor_xl) is installed."
    )


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def probe_availability(settings: Settings) -> dict[str, bool]:
    """Real-world availability check: is each provider's dependency importable /
    credential configured? Does NOT load any model — that happens lazily on first
    use inside each provider.
    """
    return {
        "openai": bool(settings.openai_api_key) and _is_importable("openai"),
        "bge_large": _is_importable("sentence_transformers"),
        "instructor_xl": _is_importable("InstructorEmbedding"),
    }
