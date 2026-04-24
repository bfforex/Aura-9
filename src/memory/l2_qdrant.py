"""L2 Qdrant Memory — hybrid semantic search with dense + sparse vectors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from loguru import logger

COLLECTIONS = ["expertise", "documentation", "skill_library", "past_missions", "failure_analysis"]
EMBEDDING_DIM = 768


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    payload: dict[str, Any]
    collection: str


class QdrantMemory:
    """L2 vector memory backed by Qdrant with hybrid search."""

    def __init__(self, qdrant_client, ollama_client=None) -> None:
        self._qd = qdrant_client
        self._ollama = ollama_client

    async def initialize_collections(self) -> None:
        """Create all 5 collections with dense + sparse vector params."""
        try:
            from qdrant_client.models import (  # noqa: PLC0415
                Distance,
                SparseVectorParams,
                VectorParams,
            )

            for collection in COLLECTIONS:
                existing = [c.name for c in (await self._qd.get_collections()).collections]
                if collection in existing:
                    logger.debug(f"L2: collection {collection} already exists")
                    continue

                await self._qd.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "text_dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={"text_sparse": SparseVectorParams()},
                )
                logger.info(f"L2: created collection {collection}")
        except Exception as exc:
            logger.error(f"L2: initialize_collections failed: {exc}")

    async def hybrid_search(
        self, collection: str, query: str, top_k: int = 5
    ) -> list[SearchResult]:
        """Hybrid search using RRF (Reciprocal Rank Fusion)."""
        try:
            embedding = await self._embed(query)
            if not embedding:
                return []

            from qdrant_client.models import (  # noqa: PLC0415
                Prefetch,
                Query,
                SparseVector,
            )

            # Dense prefetch
            dense_prefetch = Prefetch(
                query=embedding,
                using="text_dense",
                limit=20,
            )

            # Sparse query (simple keyword-based sparse vector)
            sparse_indices, sparse_values = self._make_sparse_vector(query)
            sparse_prefetch = Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="text_sparse",
                limit=20,
            )

            results = await self._qd.query_points(
                collection_name=collection,
                prefetch=[dense_prefetch, sparse_prefetch],
                query=Query(fusion="rrf"),
                limit=top_k,
            )

            return [
                SearchResult(
                    id=str(r.id),
                    score=float(r.score),
                    text=r.payload.get("text", "") if r.payload else "",
                    payload=r.payload or {},
                    collection=collection,
                )
                for r in results.points
            ]
        except Exception as exc:
            logger.warning(f"L2: hybrid_search failed: {exc}")
            return []

    async def upsert(self, collection: str, text: str, payload: dict[str, Any]) -> str:
        """Embed text and upsert into collection. Returns the point ID."""
        try:
            embedding = await self._embed(text)
            if not embedding:
                return ""

            from datetime import datetime  # noqa: PLC0415

            from qdrant_client.models import PointStruct  # noqa: PLC0415

            point_id = str(uuid.uuid4())
            payload["text"] = text
            payload["promoted_at"] = datetime.now(UTC).isoformat()

            await self._qd.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"text_dense": embedding},
                        payload=payload,
                    )
                ],
            )
            return point_id
        except Exception as exc:
            logger.error(f"L2: upsert to {collection} failed: {exc}")
            return ""

    def compute_significance(self, r: float, f: float, pin_bonus: float = 0.0) -> float:
        """Compute significance score: S = (R * F) + pin_bonus."""
        return max(0.0, min(2.0, r * f + pin_bonus))

    def compute_r(self, retrievals: int, max_retrievals: int) -> float:
        """Compute recency score R. Division-by-zero safe."""
        if max_retrievals <= 0:
            return 0.0
        return min(1.0, retrievals / max_retrievals)

    def format_results(self, results: list[SearchResult], max_tokens: int = 8192) -> str:
        """Format search results as context string."""
        lines = []
        total_tokens = 0
        token_budget = max_tokens

        for i, r in enumerate(results):
            line = f"[L2-{i + 1} | score={r.score:.2f} | {r.collection}] {r.text}"
            # Rough token estimate
            est_tokens = int(len(line.split()) * 1.3)
            if total_tokens + est_tokens > token_budget:
                break
            lines.append(line)
            total_tokens += est_tokens

        return "\n".join(lines)

    async def _embed(self, text: str) -> list[float]:
        if self._ollama:
            try:
                return await self._ollama.embed(text)
            except Exception as exc:
                logger.warning(f"L2: embed failed: {exc}")
        return []

    def _make_sparse_vector(self, text: str) -> tuple[list[int], list[float]]:
        """Create a simple sparse vector from word hashes."""
        words = text.lower().split()
        freq: dict[int, float] = {}
        for word in words:
            idx = abs(hash(word)) % 30000
            freq[idx] = freq.get(idx, 0.0) + 1.0
        if not freq:
            return [0], [1.0]
        indices = list(freq.keys())
        values = [freq[i] for i in indices]
        return indices, values
