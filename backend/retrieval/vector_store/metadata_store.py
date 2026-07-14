"""Persisted sidecar store: vector_id -> {metadata, text}.

FAISS itself only knows vectors and integer IDs — this is where the rest of what
the vector store needs (filterable metadata, and the original text for lexical/
hybrid search) actually lives. Persisted as a single JSON file next to the FAISS
index, written atomically (temp file + rename) so a crash mid-save can't corrupt it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, TypedDict


class VectorRecord(TypedDict):
    metadata: dict[str, Any]
    text: str | None


class MetadataStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[int, VectorRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._records = {int(vector_id): record for vector_id, record in raw.items()}

    def save(self) -> None:
        with self._lock:
            serializable = {
                str(vector_id): record for vector_id, record in self._records.items()
            }
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(serializable), encoding="utf-8")
            tmp_path.replace(self._path)

    def set(self, vector_id: int, metadata: dict[str, Any], text: str | None) -> None:
        with self._lock:
            self._records[vector_id] = {"metadata": metadata, "text": text}

    def delete(self, vector_id: int) -> None:
        with self._lock:
            self._records.pop(vector_id, None)

    def get(self, vector_id: int) -> VectorRecord | None:
        return self._records.get(vector_id)

    def metadata_of(self, vector_id: int) -> dict[str, Any]:
        record = self._records.get(vector_id)
        return record["metadata"] if record else {}

    def text_of(self, vector_id: int) -> str | None:
        record = self._records.get(vector_id)
        return record["text"] if record else None

    def known_ids(self) -> list[int]:
        return list(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)
