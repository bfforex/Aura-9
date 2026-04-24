"""Base tool types and registry for Aura-9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0


class ToolRegistry:
    """Registry for tools and synthesized skills."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Any, dict]] = {}  # name → (func, schema)

    def register(self, name: str, func: Any, schema: dict) -> None:
        """Register a tool or skill."""
        self._tools[name] = (func, schema)

    def get(self, name: str) -> Any | None:
        """Look up a tool function by name. Returns callable or None."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def list_tools(self) -> list[dict]:
        """Return all registered tool schemas."""
        return [
            {"name": name, "schema": schema}
            for name, (_, schema) in self._tools.items()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools
