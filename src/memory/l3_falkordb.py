"""L3 FalkorDB Memory — knowledge graph for entity relationships."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from loguru import logger

RETRY_QUEUE_TTL = 3600   # 1 hour
MAX_ATTEMPTS = 3
RETRY_INTERVAL = 60      # seconds

try:
    import falkordb as _falkordb_mod  # noqa: F401
    _FALKORDB_AVAILABLE = True
except ImportError:
    _FALKORDB_AVAILABLE = False
    logger.warning("L3: falkordb package not available — shadow write mode active")


class FalkorDBMemory:
    """L3 knowledge graph memory backed by FalkorDB."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6380,
        graph_name: str = "aura9",
        redis_client=None,
    ) -> None:
        self._host = host
        self._port = port
        self._graph_name = graph_name
        self._redis = redis_client
        self._graph = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to FalkorDB."""
        if not _FALKORDB_AVAILABLE:
            logger.warning("L3: FalkorDB not available — operating in shadow mode")
            return

        try:
            import falkordb  # noqa: PLC0415
            db = falkordb.FalkorDB(host=self._host, port=self._port)
            self._graph = db.select_graph(self._graph_name)
            self._connected = True
            logger.info(f"L3: connected to FalkorDB at {self._host}:{self._port}")
        except Exception as exc:
            logger.warning(f"L3: FalkorDB connection failed: {exc} — shadow mode active")
            self._connected = False

    async def create_entity(
        self, entity_type: str, properties: dict, session_id: str
    ) -> str:
        """Create a node in the knowledge graph. Returns entity ID."""
        entity_id = str(uuid.uuid4())
        props = {**properties, "entity_id": entity_id, "session_id": session_id,
                 "created_at": datetime.now(UTC).isoformat()}

        cypher = self._build_create_node_cypher(entity_type, props)

        if self._connected and self._graph is not None:
            try:
                self._graph.query(cypher)
                return entity_id
            except Exception as exc:
                logger.warning(f"L3: create_entity failed: {exc} — queuing retry")

        await self._queue_retry(
            operation="CREATE_NODE",
            cypher=cypher,
            session_id=session_id,
        )
        return entity_id

    async def create_relationship(
        self,
        from_id: str,
        rel_type: str,
        to_id: str,
        properties: dict | None = None,
    ) -> None:
        """Create a relationship edge between two nodes."""
        props = properties or {}
        props["created_at"] = datetime.now(UTC).isoformat()

        prop_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
        cypher = (
            f"MATCH (a {{entity_id: {json.dumps(from_id)}}}), "
            f"(b {{entity_id: {json.dumps(to_id)}}}) "
            f"CREATE (a)-[r:{rel_type} {{{prop_str}}}]->(b)"
        )

        if self._connected and self._graph is not None:
            try:
                self._graph.query(cypher)
                return
            except Exception as exc:
                logger.warning(f"L3: create_relationship failed: {exc} — queuing retry")

        await self._queue_retry(
            operation="CREATE_EDGE",
            cypher=cypher,
            session_id="",
        )

    async def get_entity(self, entity_id: str) -> dict | None:
        """Retrieve a node by entity_id."""
        if not self._connected or self._graph is None:
            return None
        try:
            cypher = f"MATCH (n {{entity_id: {json.dumps(entity_id)}}}) RETURN n LIMIT 1"
            result = self._graph.query(cypher)
            if result.result_set:
                return dict(result.result_set[0][0].properties)
        except Exception as exc:
            logger.warning(f"L3: get_entity failed: {exc}")
        return None

    async def update_entity(self, entity_id: str, properties: dict) -> None:
        """Update properties of an existing node."""
        set_clauses = ", ".join(f"n.{k} = {json.dumps(v)}" for k, v in properties.items())
        cypher = (
            f"MATCH (n {{entity_id: {json.dumps(entity_id)}}}) "
            f"SET {set_clauses}"
        )

        if self._connected and self._graph is not None:
            try:
                self._graph.query(cypher)
                return
            except Exception as exc:
                logger.warning(f"L3: update_entity failed: {exc} — queuing retry")

        await self._queue_retry(
            operation="UPDATE_NODE",
            cypher=cypher,
            session_id="",
        )

    async def drain_retry_queue(self) -> int:
        """Process pending writes from the retry queue. Returns number processed."""
        if not self._redis:
            return 0

        processed = 0
        while True:
            val = await self._redis.lpop("falkordb:retry_queue")
            if not val:
                break

            raw = val.decode() if isinstance(val, bytes) else val
            try:
                item = json.loads(raw)
            except Exception:  # noqa: S112
                continue

            if item.get("retry_count", 0) >= MAX_ATTEMPTS:
                logger.warning(f"L3: retry item exceeded max attempts: {item.get('write_id')}")
                continue

            if self._connected and self._graph is not None:
                try:
                    self._graph.query(item["cypher"])
                    processed += 1
                    continue
                except Exception as exc:
                    logger.warning(f"L3: drain retry failed: {exc}")

            # Re-queue with incremented retry count
            item["retry_count"] = item.get("retry_count", 0) + 1
            await self._queue_retry_item(item)

        return processed

    async def _queue_retry(self, operation: str, cypher: str, session_id: str) -> None:
        item = {
            "write_id": str(uuid.uuid4()),
            "queued_at": datetime.now(UTC).isoformat(),
            "operation": operation,
            "cypher": cypher,
            "session_id": session_id,
            "retry_count": 0,
        }
        await self._queue_retry_item(item)

    async def _queue_retry_item(self, item: dict) -> None:
        if not self._redis:
            return
        try:
            await self._redis.rpush("falkordb:retry_queue", json.dumps(item))
        except Exception as exc:
            logger.error(f"L3: failed to push to retry queue: {exc}")

    @staticmethod
    def _build_create_node_cypher(entity_type: str, props: dict) -> str:
        prop_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
        return f"CREATE (n:{entity_type} {{{prop_str}}})"
