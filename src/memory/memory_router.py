"""Memory Router (MR-1) — classifies and routes content to appropriate memory tiers."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Content types routed to L1 only
_EPHEMERAL_TYPES = {"tool_output_raw", "scratchpad", "continuation", "turn"}

# Regex-based entity extractors
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("URL", re.compile(r"https?://[^\s]+")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")),
    ("CODE_ID", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),
    ("VERSION", re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b", re.I)),
    ("FILE_PATH", re.compile(r"(?:\.{0,2}/)?(?:\w+/)+\w+\.\w+")),
    ("CURRENCY", re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")),
    ("PERCENTAGE", re.compile(r"\b\d+(?:\.\d+)?%\b")),
]


def _extract_entities(text: str) -> list[dict[str, str]]:
    """Extract entities from text using regex patterns."""
    entities = []
    for entity_type, pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entities.append({"type": entity_type, "value": match.group()})
    return entities


class MemoryRouter:
    """MR-1: Routes content to appropriate memory tiers based on classification."""

    def __init__(self, l1=None, l2=None, l3=None) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3

        # Prometheus metrics (lazy)
        self._metrics_ok = False

    def _init_metrics(self) -> None:
        if self._metrics_ok:
            return
        try:
            from src.observability.metrics import MR1_ROUTING_DECISIONS  # noqa: PLC0415
            self._metric_decisions = MR1_ROUTING_DECISIONS
            self._metrics_ok = True
        except Exception:  # noqa: S110
            pass

    async def route(
        self,
        content: str,
        content_type: str,
        session_id: str,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Route content to the appropriate memory tier(s).

        Returns the routing decision label.
        """
        import time  # noqa: PLC0415
        t0 = time.monotonic()

        decision = await self._route_internal(
            content, content_type, session_id, confidence, metadata or {}
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._record_metric(content_type, decision, elapsed_ms)
        return decision

    async def _route_internal(
        self,
        content: str,
        content_type: str,
        session_id: str,
        confidence: float,
        metadata: dict[str, Any],
    ) -> str:
        # ASD state: bypass MR-1 — direct dual-write
        if content_type == "asd_state":
            if self._l1:
                await self._l1.set_asd_state(content)
            return "ASD_STATE_DIRECT"

        # Ephemeral → L1 only
        if content_type in _EPHEMERAL_TYPES:
            if self._l1:
                if content_type == "turn":
                    await self._l1.add_turn(session_id, metadata.get("role", "user"), content)
                elif content_type == "scratchpad":
                    await self._l1.set_scratchpad(session_id, content)
            return "L1_ONLY"

        # Factual knowledge / reusable insight → L1 + L2 (if confidence > 0.85)
        if content_type in ("factual_knowledge", "reusable_insight"):
            if self._l1:
                await self._l1.set_scratchpad(session_id, content)
            if self._l2 and confidence > 0.85:
                collection = metadata.get("collection", "expertise")
                await self._l2.upsert(collection, content, {"session_id": session_id, **metadata})
                return "L1_L2"
            return "L1_ONLY"

        # Entity relationship → L1 + L3
        if content_type == "entity_relationship":
            entities = _extract_entities(content)
            if self._l1:
                await self._l1.set_scratchpad(session_id, content)
            if self._l3 and entities:
                for entity in entities[:5]:  # Limit to 5 entities per call
                    await self._l3.create_entity(
                        entity["type"],
                        {"value": entity["value"], "source": content[:200]},
                        session_id,
                    )
                return "L1_L3"
            return "L1_ONLY"

        # Default → DISCARD
        logger.debug(f"MemoryRouter: DISCARD content_type={content_type}")
        return "DISCARD"

    def _record_metric(self, content_type: str, decision: str, latency_ms: float) -> None:
        self._init_metrics()
        if self._metrics_ok:
            try:
                self._metric_decisions.labels(
                    content_type=content_type, decision=decision
                ).inc()
            except Exception:  # noqa: S110
                pass
