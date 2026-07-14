"""Shared contract every format-specific extractor implements."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedDocument:
    """Raw output of a format extractor, before cleaning/language detection/hashing."""

    text: str
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    sheet_names: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Extractor(abc.ABC):
    """One implementation per supported file format."""

    @abc.abstractmethod
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        """Extract raw text and structural metadata from a file's raw bytes.

        Implementations should raise `backend.core.exceptions.ValidationAppError`
        for malformed/corrupt input rather than letting a third-party parser
        exception escape.
        """
