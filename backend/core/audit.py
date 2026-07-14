"""Tamper-evident audit trail: every `AuditLog` row is chained to the one
before it via a SHA-256 hash (`entry_hash`/`prev_hash`), so retroactively
editing or deleting a past row is detectable (see `verify_audit_chain`) even
though nothing stops someone with raw DB access from doing it.

`append_audit_log` is the single insertion point for `AuditLog` rows —
`backend.rag.audit.RagQueryAuditor` and
`backend.services.auth_service.AuthService` both go through it rather than
constructing `AuditLog` directly, so every row (RAG queries, logins, role
changes, ...) participates in one consistent chain.

Concurrency note: this chain is only strictly linear under a single writer at
a time. Two requests appending concurrently can both read the same "latest"
row before either commits, producing two rows that claim the same
`prev_hash` — a fork, not silent corruption, and `verify_audit_chain` will
flag the resulting broken link for a human to investigate. A fully serialized
chain under concurrent writers would need a DB-level advisory lock around the
read-then-insert; that isn't done here because audit logging must never block
or fail the request it's instrumenting (see the try/except at both call
sites) — a false-positive break report is an acceptable trade for that.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.datetime_utils import ensure_utc
from backend.models.audit_log import AuditLog


def _hash_entry(
    prev_hash: str | None,
    *,
    user_id: str | None,
    action: str,
    resource: str | None,
    detail: str | None,
    ip_address: str | None,
    request_id: str | None,
    created_at: datetime,
) -> str:
    payload = "|".join(
        [
            prev_hash or "",
            user_id or "",
            action,
            resource or "",
            detail or "",
            ip_address or "",
            request_id or "",
            ensure_utc(created_at).isoformat(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _latest_entry(db: AsyncSession) -> AuditLog | None:
    result = await db.execute(
        select(AuditLog)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(1)
    )
    return result.scalars().first()


async def append_audit_log(
    db: AsyncSession,
    *,
    user_id: str | None,
    action: str,
    resource: str | None = None,
    detail: str | None = None,
    ip_address: str | None = None,
    request_id: str | None = None,
) -> AuditLog:
    """Adds a chained `AuditLog` row to `db` — does not commit, so it composes
    with a caller writing other rows (e.g. `RagQueryLog`) in the same
    transaction.
    """
    previous = await _latest_entry(db)
    created_at = datetime.now(timezone.utc)
    prev_hash = previous.entry_hash if previous else None
    entry_hash = _hash_entry(
        prev_hash,
        user_id=user_id,
        action=action,
        resource=resource,
        detail=detail,
        ip_address=ip_address,
        request_id=request_id,
        created_at=created_at,
    )
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        detail=detail,
        ip_address=ip_address,
        request_id=request_id,
        created_at=created_at,
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )
    db.add(entry)
    return entry


async def verify_audit_chain(db: AsyncSession) -> list[str]:
    """Walks the whole `audit_logs` table in insertion order and recomputes
    each row's hash, returning human-readable problems found (empty if the
    chain is intact). `detail` is decrypted transparently on read (see
    `backend.security.encryption.EncryptedString`) before hashing, so this
    reproduces the same hash computed at insert time regardless of
    encryption — encryption protects confidentiality, this checks integrity;
    neither substitutes for the other.
    """
    result = await db.execute(
        select(AuditLog).order_by(AuditLog.created_at.asc(), AuditLog.id.asc())
    )
    rows = result.scalars().all()

    problems: list[str] = []
    expected_prev: str | None = None
    for row in rows:
        if row.prev_hash != expected_prev:
            problems.append(
                f"Row {row.id} ({row.action} @ {row.created_at.isoformat()}): prev_hash does not match the "
                "preceding row's entry_hash — the chain is broken here (a row may have been edited, deleted, "
                "reordered, or this is a concurrent-write fork; see the module docstring)."
            )

        recomputed = _hash_entry(
            row.prev_hash,
            user_id=row.user_id,
            action=row.action,
            resource=row.resource,
            detail=row.detail,
            ip_address=row.ip_address,
            request_id=row.request_id,
            created_at=row.created_at,
        )
        if recomputed != row.entry_hash:
            problems.append(
                f"Row {row.id} ({row.action} @ {row.created_at.isoformat()}): entry_hash does not match its "
                "recomputed content — this row's data was modified after being written."
            )

        expected_prev = row.entry_hash
    return problems
