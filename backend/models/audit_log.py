import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base
from backend.security.encryption import EncryptedString


class AuditLog(Base):
    """Immutable, tamper-evident trail of sensitive actions, required for banking
    compliance reviews.

    Tamper-evidence: `entry_hash` chains every row to the one before it (see
    `backend.core.audit.append_audit_log`/`verify_audit_chain`) — editing or
    deleting a past row breaks the chain from that point forward, which
    `verify_audit_chain` detects. `detail` is stored encrypted at rest (see
    `backend.security.encryption`) in addition to being PII-redacted before
    it ever gets here (see `backend.security.pii`) — redaction limits what's
    written, encryption protects what is.

    `created_at` is generated in application code (not `server_default`)
    specifically so its value is known *before* the row is hashed and
    inserted — a DB-side default would only be assigned after INSERT, too
    late to include in `entry_hash`.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # PII-redacted (see backend.security.pii) *and* encrypted at rest (see
    # backend.security.encryption.EncryptedString) human-readable detail.
    detail: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Tamper-evidence chain — see backend.core.audit.
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
