"""Unit tests for the Memory Router."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.memory.memory_router import MemoryRouter, _extract_entities


@pytest.mark.unit
class TestEntityExtraction:
    def test_extract_email(self):
        entities = _extract_entities("Contact user@example.com for details")
        types = [e["type"] for e in entities]
        assert "EMAIL" in types

    def test_extract_url(self):
        entities = _extract_entities("Visit https://example.com/path for info")
        types = [e["type"] for e in entities]
        assert "URL" in types

    def test_extract_date(self):
        entities = _extract_entities("Due date is 2024-12-31")
        types = [e["type"] for e in entities]
        assert "DATE" in types

    def test_extract_version(self):
        entities = _extract_entities("Install version v2.4.1")
        types = [e["type"] for e in entities]
        assert "VERSION" in types

    def test_no_entities(self):
        entities = _extract_entities("Hello world how are you today")
        assert isinstance(entities, list)

    def test_multiple_entities(self):
        entities = _extract_entities("Email user@test.com about v1.2 at 2024-01-15")
        types = {e["type"] for e in entities}
        assert len(types) >= 2


@pytest.mark.unit
class TestMemoryRouterClassification:
    def _make_router(self):
        l1 = AsyncMock()
        l1.set_scratchpad = AsyncMock()
        l1.add_turn = AsyncMock()
        l1.set_asd_state = AsyncMock()
        l2 = AsyncMock()
        l2.upsert = AsyncMock()
        l3 = AsyncMock()
        l3.create_entity = AsyncMock()
        return MemoryRouter(l1=l1, l2=l2, l3=l3), l1, l2, l3

    @pytest.mark.asyncio
    async def test_ephemeral_routes_to_l1_only(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route("content", "scratchpad", "sess-123")
        assert decision == "L1_ONLY"
        l1.set_scratchpad.assert_called_once()
        l2.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_routes_to_l1(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route(
            "user said hello", "turn", "sess-123", metadata={"role": "user"}
        )
        assert decision == "L1_ONLY"
        l1.add_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_factual_knowledge_low_confidence_l1_only(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route("fact", "factual_knowledge", "sess-123", confidence=0.7)
        assert decision == "L1_ONLY"
        l2.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_factual_knowledge_high_confidence_l1_l2(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route("fact", "factual_knowledge", "sess-123", confidence=0.9)
        assert decision == "L1_L2"
        l2.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_asd_state_direct_write(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route('{"asd_update": {}}', "asd_state", "sess-123")
        assert decision == "ASD_STATE_DIRECT"
        l1.set_asd_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_relationship_l1_l3(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route(
            "User john@example.com owns v2.4.0", "entity_relationship", "sess-123"
        )
        assert decision in ("L1_L3", "L1_ONLY")

    @pytest.mark.asyncio
    async def test_unknown_type_discard(self):
        router, l1, l2, l3 = self._make_router()
        decision = await router.route("content", "unknown_type", "sess-123")
        assert decision == "DISCARD"
