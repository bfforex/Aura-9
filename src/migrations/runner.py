"""Migration runner — applies FalkorDB graph schema migrations."""

from __future__ import annotations

from loguru import logger

try:
    import falkordb as _falkordb_mod  # noqa: F401
    _FALKORDB_AVAILABLE = True
except ImportError:
    _FALKORDB_AVAILABLE = False


_MIGRATIONS = [
    # Create indexes and constraints for all node types
    "CREATE INDEX ON :Task(task_id)",
    "CREATE INDEX ON :Session(session_id)",
    "CREATE INDEX ON :Skill(skill_id)",
    "CREATE INDEX ON :Entity(entity_id)",
    "CREATE INDEX ON :TaskCompletion(task_id)",
]

_AGENT_SINGLETON = (
    "MERGE (a:Agent {agent_id: 'aura9-v2.4'}) "
    "ON CREATE SET a.created_at = timestamp(), a.version = '2.4.0'"
)


class MigrationRunner:
    """Applies FalkorDB graph schema migrations."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6380,
        graph_name: str = "aura9",
    ) -> None:
        self._host = host
        self._port = port
        self._graph_name = graph_name

    def run(self) -> None:
        """Apply all pending migrations."""
        if not _FALKORDB_AVAILABLE:
            logger.warning("MigrationRunner: falkordb not available — skipping graph migrations")
            self._print_migrations()
            return

        try:
            import falkordb  # noqa: PLC0415
            db = falkordb.FalkorDB(host=self._host, port=self._port)
            graph = db.select_graph(self._graph_name)

            for migration in _MIGRATIONS:
                try:
                    graph.query(migration)
                    logger.info(f"Migration applied: {migration[:60]}")
                except Exception as exc:
                    # Many will fail if already applied (idempotent)
                    logger.debug(f"Migration note: {exc}")

            # Agent singleton
            try:
                graph.query(_AGENT_SINGLETON)
                logger.info("Agent singleton created/verified")
            except Exception as exc:
                logger.debug(f"Agent singleton: {exc}")

            logger.info("All migrations complete")

        except Exception as exc:
            logger.error(f"MigrationRunner: connection failed: {exc}")
            self._print_migrations()

    def _print_migrations(self) -> None:
        """Print migrations that would be applied (dry run)."""
        logger.info("Dry-run migrations:")
        for m in _MIGRATIONS:
            logger.info(f"  Would apply: {m}")
