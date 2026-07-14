"""Text normalization applied uniformly to every extracted document, regardless
of source format, before language detection / hashing / chunking.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters other than newline (\n), carriage return (\r), and tab (\t).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(
    r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)
_REPEATED_PUNCTUATION_RE = re.compile(r"([.\-_=*#~])\1{4,}")


def _strip_boilerplate_lines(text: str) -> str:
    """Drop lines that are purely page-number/footer artifacts common in PDF exports."""
    kept_lines = [
        line for line in text.split("\n") if not _PAGE_NUMBER_LINE_RE.match(line)
    ]
    return "\n".join(kept_lines)


def clean_text(text: str) -> str:
    """Normalize extracted text into a consistent, chunking-ready form.

    Steps: Unicode NFKC normalization, control-character stripping, boilerplate
    (page-number) line removal, whitespace collapsing, and trimming.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _CONTROL_CHARS_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    normalized = _strip_boilerplate_lines(normalized)
    normalized = _REPEATED_PUNCTUATION_RE.sub(r"\1\1\1\1", normalized)

    normalized = _TRAILING_WHITESPACE_RE.sub("\n", normalized)
    normalized = _MULTI_SPACE_RE.sub(" ", normalized)
    normalized = _MULTI_BLANK_LINE_RE.sub("\n\n", normalized)

    # Strip leading/trailing whitespace on each line, then the whole document.
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()
