"""Structural heading detection for banking contracts, regulations, and financial
reports — the "recursive" split's top level and the source of each chunk's
context-preserving breadcrumb (e.g. "Article 5 > Clause 5.2").

Three heading families are recognized, in order of specificity:
  1. Markdown-style headers ("# ", "## ", ...) — present if the source was HTML/MD.
  2. Numbered legal/financial headings — "Article 5", "Section 3.1", "Clause 7.1.2",
     "Chapter II", "Part 4", "Item 7.", "Note 14", "Schedule A", "§ 12" — the
     numbering depth (dot-segments) refines the hierarchy level below the keyword's
     base level, so "Article 5.2" nests under "Article 5".
  3. ALL-CAPS short lines — a generic fallback heading pattern common in scanned/
     converted financial reports ("RISK FACTORS", "MANAGEMENT'S DISCUSSION").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.ingestion.chunking.types import Section

# Base hierarchy level per keyword family (lower = higher up the document tree).
_KEYWORD_LEVELS: dict[str, int] = {
    "part": 0,
    "title": 0,
    "chapter": 1,
    "section": 2,
    "article": 2,
    "item": 2,
    "§": 2,
    "clause": 3,
    "note": 3,
    "schedule": 3,
    "appendix": 3,
    "annex": 3,
}

_ROMAN_RE = r"[IVXLCDM]+"

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_NUMBERED_HEADING_RE = re.compile(
    r"^[ \t]*(?P<keyword>PART|TITLE|CHAPTER|SECTION|ARTICLE|ITEM|CLAUSE|NOTE|SCHEDULE|APPENDIX|ANNEX|§)"
    r"[ \t]*\.?[ \t]*(?P<number>\d+(?:\.\d+)*|" + _ROMAN_RE + r")"
    r"(?P<rest>[ \t]*[-–—.:)]?[ \t]*[^\n]*)$",
    re.MULTILINE | re.IGNORECASE,
)

# A short line that is (mostly) uppercase letters/digits/punctuation — heuristic
# fallback heading for financial-report style section titles.
_ALL_CAPS_LINE_RE = re.compile(
    r"^[ \t]*([A-Z][A-Z0-9 ,.&'\-:()/]{2,79})[ \t]*$", re.MULTILINE
)


@dataclass
class HeadingMatch:
    title: str  # display text, e.g. "Article 5.2 - Prepayment"
    level: int
    start_char: int
    end_char: int  # end of the heading line itself (body text follows)


def _numbering_depth(number: str) -> int:
    """How many dot-separated segments the heading number has, e.g. "5.2.1" -> 2
    extra levels of nesting beyond the keyword's base level."""
    if re.fullmatch(_ROMAN_RE, number, re.IGNORECASE):
        return 0
    return number.count(".")


def _is_all_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    return all(c.isupper() for c in letters)


def detect_headings(text: str) -> list[HeadingMatch]:
    """Find structural heading lines and assign each a hierarchy level.

    Returns headings sorted by position. Overlapping matches (a numbered heading
    that also happens to be all-caps) are de-duplicated, preferring the more
    specific numbered/markdown match over the generic all-caps fallback.
    """
    headings: list[HeadingMatch] = []
    claimed_lines: set[int] = set()

    for match in _MARKDOWN_HEADING_RE.finditer(text):
        level = len(match.group(1)) - 1
        title = match.group(2).strip()
        headings.append(
            HeadingMatch(
                title=title, level=level, start_char=match.start(), end_char=match.end()
            )
        )
        claimed_lines.add(text.count("\n", 0, match.start()))

    for match in _NUMBERED_HEADING_RE.finditer(text):
        line_no = text.count("\n", 0, match.start())
        if line_no in claimed_lines:
            continue
        keyword = match.group("keyword").lower()
        number = match.group("number")
        rest = (match.group("rest") or "").strip(" \t.-–—:")
        base_level = _KEYWORD_LEVELS.get(keyword, 2)
        depth = (
            0
            if re.fullmatch(_ROMAN_RE, number, re.IGNORECASE)
            else _numbering_depth(number)
        )
        level = base_level + depth

        title = f"{match.group('keyword').strip()} {number}"
        if rest:
            title = f"{title} - {rest}"

        headings.append(
            HeadingMatch(
                title=title, level=level, start_char=match.start(), end_char=match.end()
            )
        )
        claimed_lines.add(line_no)

    for match in _ALL_CAPS_LINE_RE.finditer(text):
        line_no = text.count("\n", 0, match.start())
        if line_no in claimed_lines:
            continue
        line = match.group(1).strip()
        if not _is_all_caps_heading(line):
            continue
        headings.append(
            HeadingMatch(
                title=line, level=1, start_char=match.start(), end_char=match.end()
            )
        )
        claimed_lines.add(line_no)

    headings.sort(key=lambda h: h.start_char)
    return headings


def split_into_sections(text: str) -> list[Section]:
    """Split the full document into contiguous, breadcrumb-tagged sections.

    Every character of `text` belongs to exactly one section — including a leading
    "" (root) section for any preamble text before the first detected heading, and
    (if no headings are found at all) a single section spanning the whole document.
    """
    headings = detect_headings(text)

    if not headings:
        return [Section(breadcrumb="", start_char=0, end_char=len(text), text=text)]

    sections: list[Section] = []
    if headings[0].start_char > 0:
        preamble = text[: headings[0].start_char]
        sections.append(
            Section(
                breadcrumb="",
                start_char=0,
                end_char=headings[0].start_char,
                text=preamble,
            )
        )

    stack: list[HeadingMatch] = []
    for i, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        stack.append(heading)
        breadcrumb = " > ".join(h.title for h in stack)

        seg_start = heading.start_char
        seg_end = headings[i + 1].start_char if i + 1 < len(headings) else len(text)
        sections.append(
            Section(
                breadcrumb=breadcrumb,
                start_char=seg_start,
                end_char=seg_end,
                text=text[seg_start:seg_end],
            )
        )

    return sections
