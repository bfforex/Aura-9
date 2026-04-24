"""IPC publisher — publishes messages to Redis pub/sub channels."""

from __future__ import annotations

import json


async def publish(channel: str, payload: dict, redis_client) -> None:
    """Publish a payload dict to a Redis pub/sub channel."""
    await redis_client.publish(channel, json.dumps(payload))
