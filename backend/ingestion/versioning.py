"""Document versioning and logical-identity rules.

Kept free of any database/ORM dependency so the decision logic is trivial to
unit test — callers (backend.services.document_service) fetch the candidate
rows and pass lightweight `DocumentRef` values in.
"""

from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

_SLUG_RE = re.compile(r"[^a-z0-9.]+")


def normalize_document_key(filename: str) -> str:
    """Derive a stable logical identity for a document from its filename.

    Two uploads with the same normalized key are treated as versions of the same
    logical document (e.g. "Overdraft Policy.pdf" and "overdraft_policy.pdf" both
    normalize to "overdraft-policy.pdf"). Case, accents, and separator differences
    are folded away; the extension is preserved since a different format is a
    different artifact even if the underlying policy is the same.
    """
    base_name = PurePosixPath(filename or "untitled").name.strip().lower()
    ascii_name = (
        unicodedata.normalize("NFKD", base_name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_RE.sub("-", ascii_name).strip("-")
    return slug or "untitled"


class IngestionDecision(str, enum.Enum):
    NEW = "new"
    NEW_VERSION = "new_version"
    EXACT_DUPLICATE = "exact_duplicate"


@dataclass(frozen=True)
class DocumentRef:
    id: str
    version: int


@dataclass(frozen=True)
class VersioningResult:
    decision: IngestionDecision
    supersedes: DocumentRef | None = None
    duplicate_of: DocumentRef | None = None

    @property
    def next_version(self) -> int:
        if (
            self.decision is IngestionDecision.NEW_VERSION
            and self.supersedes is not None
        ):
            return self.supersedes.version + 1
        return 1


def decide(
    *,
    exact_duplicate_candidate: DocumentRef | None,
    same_key_current_candidate: DocumentRef | None,
) -> VersioningResult:
    """Decide what to do with a newly-ingested document.

    `exact_duplicate_candidate`: an existing *current* document owned by the same
        user whose content hash matches exactly (byte-for-byte after cleaning).
    `same_key_current_candidate`: an existing *current* document owned by the same
        user whose normalized filename (`document_key`) matches, regardless of
        content — i.e. a re-upload of "the same document" with different content.

    Exact-duplicate takes priority: if the content is identical, it doesn't matter
    whether the filename also matches — there is nothing new to version.
    """
    if exact_duplicate_candidate is not None:
        return VersioningResult(
            decision=IngestionDecision.EXACT_DUPLICATE,
            duplicate_of=exact_duplicate_candidate,
        )

    if same_key_current_candidate is not None:
        return VersioningResult(
            decision=IngestionDecision.NEW_VERSION,
            supersedes=same_key_current_candidate,
        )

    return VersioningResult(decision=IngestionDecision.NEW)
