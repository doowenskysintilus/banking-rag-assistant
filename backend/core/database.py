"""Async SQLAlchemy engine/session setup."""

from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import ConnectionPoolEntry

from backend.core.config import get_settings

settings = get_settings()

_engine_kwargs: dict = {
    "echo": settings.debug and not settings.is_production,
    "future": True,
}
if settings.database_url.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(settings.database_url, **_engine_kwargs)

if settings.database_url.startswith("sqlite"):
    # SQLite ignores FOREIGN KEY ... ON DELETE CASCADE unless explicitly enabled per
    # connection. Required for `passive_deletes=True` relationships to cascade correctly.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(
        dbapi_connection, connection_record: ConnectionPoolEntry
    ) -> None:  # noqa: ANN001, ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


async def init_db() -> None:
    """Create tables if they do not exist. Use Alembic migrations in production instead."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
