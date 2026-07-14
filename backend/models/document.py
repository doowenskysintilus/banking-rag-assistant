import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.PENDING
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Populated synchronously at upload time by the ingestion pipeline (extraction +
    # cleaning); the background task chunks/embeds this text rather than re-parsing
    # the original file.
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Deduplication / versioning.
    document_key: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    content_simhash: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    previous_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    near_duplicate_of_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )

    # Extraction metadata.
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    doc_metadata: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-encoded DocumentMetadata

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Row id of this chunk's embedding inside the vector store (FAISS index position).
    vector_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    # Breadcrumb in the source document's heading hierarchy (e.g. "Article 5 > Clause
    # 5.2"), populated by backend.ingestion.chunking. Empty string if the document
    # (or chunking strategy) has no detectable structure.
    header_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")
