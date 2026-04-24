"""Qdrant search tool."""

from __future__ import annotations

import time

from src.tools.base import ToolResult


async def qdrant_search(
    query: str,
    collection: str = "expertise",
    top_k: int = 5,
    l2_memory=None,
) -> ToolResult:
    """Search L2 Qdrant memory using hybrid search."""
    t0 = time.monotonic()
    try:
        if l2_memory is None:
            return ToolResult(success=False, output=None, error="L2 memory not available")

        results = await l2_memory.hybrid_search(collection, query, top_k=top_k)
        formatted = l2_memory.format_results(results)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolResult(success=True, output=formatted, execution_time_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolResult(success=False, output=None, error=str(exc), execution_time_ms=elapsed_ms)
