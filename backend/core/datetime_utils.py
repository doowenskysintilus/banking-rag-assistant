"""SQLite (unlike Postgres) doesn't actually preserve timezone-awareness
across a round trip: a `DateTime(timezone=True)` column stores a plain ISO
string with no UTC offset, so a value read back after a genuine reload from
the DB comes back *naive* even though it was written as UTC-aware — same
wall-clock instant, but naive vs. aware values raise `TypeError` when compared
directly, and format differently via `.isoformat()`.

Every datetime this app ever writes to the DB is already UTC (see the several
`default=lambda: datetime.now(timezone.utc)` column defaults across
`backend.models`) — `ensure_utc` treats a naive value as already *being* UTC,
just missing the label, rather than guessing at some other timezone. Use this
before comparing or hashing any datetime that came from a DB column and might
have made a real round trip (as opposed to still being the same in-memory
object from the request that wrote it).
"""

from __future__ import annotations

from datetime import datetime, timezone


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
