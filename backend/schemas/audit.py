from datetime import datetime

from pydantic import BaseModel

from backend.schemas.common import ORMModel


class AuditLogEntry(ORMModel):
    id: str
    user_id: str | None
    action: str
    resource: str | None
    detail: str | None
    ip_address: str | None
    created_at: datetime


class AuditLogList(BaseModel):
    items: list[AuditLogEntry]
    total: int


class AuditChainVerification(BaseModel):
    intact: bool
    checked_entries: int
    problems: list[str]
