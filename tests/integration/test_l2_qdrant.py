"""Integration tests for L2 Qdrant Memory using mock client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.l2_qdrant import COLLECTIONS, QdrantMemory, SearchResult


@pytest.fixture
def mock_ollama_embed():
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1] * 768)
    return client


@pytest.fixture
def l2(mock_qdrant, mock_ollama_embed):
    return QdrantMemory(qdrant_client=mock_qdrant, ollama_client=mock_ollama_embed)


@pytest.mark.integration
class TestL2CollectionInit:
    @pytest.mark.asyncio
    async def test_initialize_creates_collections(self, l2, mock_qdrant):
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant.create_collection = AsyncMock()

        await l2.initialize_collections()

        assert mock_qdrant.create_collection.call_count == len(COLLECTIONS)

    @pytest.mark.asyncio
    async def test_skip_existing_collections(self, l2, mock_qdrant):
        existing = [MagicMock(name="expertise"), MagicMock(name="documentation")]
        for m in existing:
            m.name = m.name
        mock_qdrant.get_collections.return_value = MagicMock(collections=existing)
        mock_qdrant.create_collection = AsyncMock()

        await l2.initialize_collections()

        # Should only create the missing ones
        assert mock_qdrant.create_collection.call_count < len(COLLECTIONS)


@pytest.mark.integration
class TestL2HybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, l2, mock_qdrant):
        mock_point = MagicMock()
        mock_point.id = "abc-123"
        mock_point.score = 0.95
        mock_point.payload = {"text": "Python async programming", "collection": "expertise"}
        mock_qdrant.query_points = AsyncMock(return_value=MagicMock(points=[mock_point]))

        results = await l2.hybrid_search("expertise", "python async", top_k=5)

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.95, abs=0.01)

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_on_error(self, l2, mock_qdrant):
        mock_qdrant.query_points = AsyncMock(side_effect=Exception("connection failed"))

        results = await l2.hybrid_search("expertise", "query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_no_embedding(self, l2, mock_ollama_embed):
        mock_ollama_embed.embed = AsyncMock(return_value=[])

        results = await l2.hybrid_search("expertise", "query", top_k=5)
        assert results == []


@pytest.mark.integration
class TestL2Upsert:
    @pytest.mark.asyncio
    async def test_upsert_returns_id(self, l2, mock_qdrant):
        mock_qdrant.upsert = AsyncMock()

        point_id = await l2.upsert("expertise", "Python is great", {"tag": "test"})
        assert point_id  # Should return a non-empty string

    @pytest.mark.asyncio
    async def test_upsert_calls_embed(self, l2, mock_ollama_embed, mock_qdrant):
        mock_qdrant.upsert = AsyncMock()

        await l2.upsert("expertise", "test content", {})
        mock_ollama_embed.embed.assert_called_once_with("test content")


@pytest.mark.integration
class TestL2SignificanceScore:
    def test_significance_formula(self):
        l2 = QdrantMemory(qdrant_client=None)
        # S = (R * F) + pin_bonus
        score = l2.compute_significance(r=0.8, f=0.7, pin_bonus=0.0)
        assert score == pytest.approx(0.56, abs=0.01)

    def test_significance_with_pin_bonus(self):
        l2 = QdrantMemory(qdrant_client=None)
        score = l2.compute_significance(r=0.5, f=0.5, pin_bonus=1.0)
        assert score == pytest.approx(1.25, abs=0.01)

    def test_compute_r_division_safe(self):
        l2 = QdrantMemory(qdrant_client=None)
        assert l2.compute_r(0, 0) == 0.0
        assert l2.compute_r(5, 10) == pytest.approx(0.5, abs=0.01)
        assert l2.compute_r(15, 10) == 1.0  # Capped at 1.0


@pytest.mark.integration
class TestL2FormatResults:
    def test_format_basic(self):
        l2 = QdrantMemory(qdrant_client=None)
        results = [
            SearchResult(
                id="1", score=0.9, text="Python asyncio guide", payload={}, collection="expertise"
            ),
            SearchResult(
                id="2", score=0.8, text="Redis caching patterns",
                payload={}, collection="documentation"
            ),
        ]
        formatted = l2.format_results(results)
        assert "[L2-1 | score=0.90 | expertise]" in formatted
        assert "[L2-2 | score=0.80 | documentation]" in formatted

    def test_format_respects_token_limit(self):
        l2 = QdrantMemory(qdrant_client=None)
        # Create results that would exceed token budget
        results = [
            SearchResult(
                id=str(i), score=0.9, text="word " * 1000, payload={}, collection="expertise"
            )
            for i in range(20)
        ]
        formatted = l2.format_results(results, max_tokens=100)
        # Should be truncated
        lines = formatted.strip().split("\n")
        assert len(lines) < 20
