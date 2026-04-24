"""Shared test fixtures for Aura-9."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import fakeredis.aioredis
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False


@pytest.fixture
def fake_redis():
    """FakeRedis async client."""
    if not _FAKEREDIS_AVAILABLE:
        pytest.skip("fakeredis not available")
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def mock_qdrant():
    """Mock QdrantClient."""
    client = AsyncMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    return client


@pytest.fixture
def mock_ollama():
    """Mock AsyncOllamaClient."""
    client = AsyncMock()
    client.chat = AsyncMock(return_value={"message": {"content": "test response"}})
    client.embed = AsyncMock(return_value=[0.1] * 768)
    client.unload_model = AsyncMock()
    client.load_model = AsyncMock()
    client.check_model_loaded = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_pynvml(monkeypatch):
    """Mock pynvml for TAIS tests."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlDeviceGetHandleByIndex = MagicMock()
    mock.nvmlDeviceGetTemperature = MagicMock(return_value=65)
    mock.NVML_TEMPERATURE_GPU = 0
    monkeypatch.setitem(__import__("sys").modules, "pynvml", mock)
    return mock
