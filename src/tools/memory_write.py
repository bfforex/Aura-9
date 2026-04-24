"""Memory write tool — routes content to the Memory Router."""

from __future__ import annotations

import time

from src.tools.base import ToolResult


async def memory_write(
    content: str,
    content_type: str,
    tags: list[str] | None = None,
    session_id: str | None = None,
    memory_router=None,
    confidence: float = 0.0,
) -> ToolResult:
    """Write content to memory via the Memory Router."""
    t0 = time.monotonic()
    try:
        if memory_router is None:
            return ToolResult(success=False, output=None, error="Memory router not available")

        metadata = {"tags": tags or []}
        decision = await memory_router.route(
            content=content,
            content_type=content_type,
            session_id=session_id or "",
            confidence=confidence,
            metadata=metadata,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolResult(
            success=True,
            output={"decision": decision},
            execution_time_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolResult(success=False, output=None, error=str(exc), execution_time_ms=elapsed_ms)
