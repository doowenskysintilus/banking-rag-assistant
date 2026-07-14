"""A minimal in-memory fake of the parts of the `pinecone` SDK that
`PineconeVectorStore` uses — enough surface to unit test the store's logic
without a real Pinecone account or network access.
"""

from __future__ import annotations

import numpy as np

from backend.retrieval.vector_store.filters import matches


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a, dtype="float64"), np.array(b, dtype="float64")
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1.0
    return float(np.dot(a_arr, b_arr) / denom)


class FakeMatch:
    def __init__(self, id: str, score: float, metadata: dict) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata


class FakeQueryResponse:
    def __init__(self, matches: list[FakeMatch]) -> None:
        self.matches = matches


class FakeVectorRecord:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class FakeFetchResponse:
    def __init__(self, vectors: dict[str, FakeVectorRecord]) -> None:
        self.vectors = vectors


class FakeServerlessSpec:
    def __init__(self, cloud: str, region: str) -> None:
        self.cloud = cloud
        self.region = region


class FakeIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.store: dict[str, dict] = {}
        self.list_batch_size = 100

    def upsert(self, vectors: list[dict], namespace: str | None = None) -> None:
        for item in vectors:
            self.store[item["id"]] = {
                "values": item["values"],
                "metadata": dict(item.get("metadata") or {}),
            }

    def update(
        self, id: str, set_metadata: dict | None = None, namespace: str | None = None
    ) -> None:
        record = self.store.setdefault(id, {"values": [0.0] * self.dim, "metadata": {}})
        if set_metadata:
            record["metadata"].update(set_metadata)

    def delete(self, ids: list[str], namespace: str | None = None) -> None:
        for vector_id in ids:
            self.store.pop(vector_id, None)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict | None = None,
        include_metadata: bool = True,
        namespace: str | None = None,
    ) -> FakeQueryResponse:
        candidates = []
        for vector_id, record in self.store.items():
            if filter and not matches(record["metadata"], filter):
                continue
            score = _cosine(vector, record["values"])
            candidates.append((vector_id, score, record["metadata"]))
        candidates.sort(key=lambda c: c[1], reverse=True)
        top = candidates[:top_k]
        return FakeQueryResponse(
            [FakeMatch(vid, score, dict(metadata)) for vid, score, metadata in top]
        )

    def fetch(self, ids: list[str], namespace: str | None = None) -> FakeFetchResponse:
        vectors = {}
        for vector_id in ids:
            if vector_id in self.store:
                vectors[vector_id] = FakeVectorRecord(
                    dict(self.store[vector_id]["metadata"])
                )
        return FakeFetchResponse(vectors)

    def list(self, namespace: str | None = None):
        ids = list(self.store.keys())
        for i in range(0, len(ids), self.list_batch_size):
            yield ids[i : i + self.list_batch_size]


class FakeIndexList:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return self._names


class FakePineconeClient:
    """One "connection" to the fake remote project. `indexes`/`create_index_calls`
    are shared (not copied) so multiple client instances — e.g. from constructing
    two `PineconeVectorStore`s in a row, simulating a process restart — see the
    same remote state, exactly like reconnecting to a real Pinecone project would.
    """

    def __init__(
        self,
        api_key: str | None = None,
        indexes: dict[str, FakeIndex] | None = None,
        create_index_calls: list[dict] | None = None,
    ) -> None:
        self.api_key = api_key
        self.indexes = indexes if indexes is not None else {}
        self.create_index_calls = (
            create_index_calls if create_index_calls is not None else []
        )

    def list_indexes(self) -> FakeIndexList:
        return FakeIndexList(list(self.indexes.keys()))

    def create_index(self, name: str, dimension: int, metric: str, spec) -> None:
        self.indexes[name] = FakeIndex(dimension)
        self.create_index_calls.append(
            {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        )

    def Index(self, name: str) -> FakeIndex:
        return self.indexes[name]


class FakePineconeProject:
    """Test harness: a fake remote Pinecone project that persists across however
    many `Pinecone(api_key=...)` client constructions happen during a test —
    monkeypatch `pinecone.Pinecone` to `project.connect` to use it.
    """

    def __init__(self) -> None:
        self.indexes: dict[str, FakeIndex] = {}
        self.create_index_calls: list[dict] = []
        self.clients: list[FakePineconeClient] = []

    def connect(self, api_key: str | None = None) -> FakePineconeClient:
        client = FakePineconeClient(
            api_key=api_key,
            indexes=self.indexes,
            create_index_calls=self.create_index_calls,
        )
        self.clients.append(client)
        return client
