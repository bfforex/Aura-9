"""IPC subscriber — async generator for Redis pub/sub messages."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger


async def subscribe(
    channel: str, redis_client
) -> AsyncGenerator[dict[str, Any], None]:
    """Subscribe to a Redis pub/sub channel, yielding decoded payloads."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message.get("data", b"")
                if isinstance(data, bytes):
                    data = data.decode()
                try:
                    yield json.loads(data)
                except json.JSONDecodeError as exc:
                    logger.warning(f"IPC subscriber: decode error on {channel}: {exc}")
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
