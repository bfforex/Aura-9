"""Memory Router (MR-1).

Routes incoming data to the appropriate memory tier:
  L1 Redis  — live state / ephemeral working data
  L2 Qdrant — factual knowledge / reusable insight
  L3 FalkorDB — entity relationships
"""

from __future__ import annotations

import enum
from typing import Any

from loguru import logger


class Destination(enum.StrEnum):
    L1 = "L1_REDIS"
    L2 = "L2_QDRANT"
    L3 = "L3_FALKORDB"
    DISCARD = "DISCARD"


def classify(data: dict[str, Any]) -> list[Destination]:
    """Decide where a piece of data should be routed.

    The classification mirrors the decision tree from the spec:
      1. Live state / ephemeral → L1
      2. Factual knowledge / reusable insight → L2 + L3 (extract entities)
      3. Relationship-only → L3
      4. None of the above → Discard
    """
    destinations: list[Destination] = []

    data_type = data.get("type", "")

    # Live state and ephemeral working data always go to L1
    if data_type in ("turn", "tool_result", "scratchpad", "asd_state"):
        destinations.append(Destination.L1)
        return destinations

    # Factual knowledge → L2 + entity extraction to L3
    if data_type in ("knowledge", "expertise", "documentation", "skill", "mission_postmortem"):
        destinations.append(Destination.L2)
        if data.get("entities"):
            destinations.append(Destination.L3)
        return destinations

    # Relationship-only data → L3
    if data_type in ("relationship", "dependency", "entity_link"):
        destinations.append(Destination.L3)
        return destinations

    logger.debug("MR-1 discarding data of type={}", data_type)
    return [Destination.DISCARD]
