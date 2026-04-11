"""L2 — Semantic Memory (Qdrant).

Permanent expertise retrieval with hybrid search and significance
score decay.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from aura9.core.config import get

COLLECTIONS = [
    "expertise",
    "documentation",
    "skill_library",
    "past_missions",
    "failure_analysis",
]


class QdrantMemory:
    """Async Qdrant wrapper for L2 semantic memory."""

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        url = get("memory.qdrant.url", "http://localhost:6333")
        self._client = AsyncQdrantClient(url=url)
        logger.info("L2 Qdrant connected — {}", url)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> AsyncQdrantClient:
        assert self._client is not None, "QdrantMemory not connected"  # noqa: S101
        return self._client

    async def ensure_collections(self) -> None:
        """Create any missing collections with the configured vector size."""
        dim_key = "memory.qdrant.collections_dimensions"
        dimensions = get(dim_key, get("model.embedding.dimensions", 768))
        existing = {c.name for c in await self.client.get_collections()}
        for name in COLLECTIONS:
            if name not in existing:
                await self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: {}", name)

    async def upsert(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        await self.client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        results = await self.client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results.points
        ]

    async def is_healthy(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False
