"""L3 — Relational Memory (FalkorDB).

Structural map of entities, tasks, tools, and the ASD shadow graph.
"""

from __future__ import annotations

from typing import Any

from falkordb import FalkorDB as FalkorDBDriver
from loguru import logger

from aura9.core.config import get


class FalkorMemory:
    """FalkorDB wrapper for L3 relational memory and ASD shadow."""

    def __init__(self) -> None:
        self._driver: FalkorDBDriver | None = None
        self._graph_name: str = get("memory.falkordb.graph_name", "aura9_graph")

    def connect(self) -> None:
        host = get("memory.falkordb.host", "localhost")
        port = get("memory.falkordb.port", 6380)
        self._driver = FalkorDBDriver(host=host, port=port)
        logger.info("L3 FalkorDB connected — {}:{}", host, port)

    def close(self) -> None:
        if self._driver:
            self._driver = None

    @property
    def graph(self) -> Any:
        assert self._driver is not None, "FalkorMemory not connected"  # noqa: S101
        return self._driver.select_graph(self._graph_name)

    def ensure_schema(self) -> None:
        """Apply the core graph schema indices."""
        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.id)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.id)",
            "CREATE INDEX IF NOT EXISTS FOR (tl:Tool) ON (tl.id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Skill) ON (s.id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (sn:StateNode) ON (sn.task_id)",
            "CREATE INDEX IF NOT EXISTS FOR (mn:MemoryNode) ON (mn.id)",
        ]
        g = self.graph
        for stmt in indices:
            try:
                g.query(stmt)
            except Exception:
                logger.debug("Index may already exist: {}", stmt)
        logger.info("L3 schema indices ensured")

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        return self.graph.query(cypher, params or {})

    def is_healthy(self) -> bool:
        try:
            self.graph.query("RETURN 1")
            return True
        except Exception:
            return False
