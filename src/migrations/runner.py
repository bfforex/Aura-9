"""Migration runner — applies FalkorDB graph schema migrations."""

from __future__ import annotations

import glob
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

try:
    import falkordb as _falkordb_mod  # noqa: F401
    _FALKORDB_AVAILABLE = True
except ImportError:
    _FALKORDB_AVAILABLE = False

_MIGRATIONS_DIR = Path(__file__).parent

_AGENT_SINGLETON = (
    "MERGE (a:Agent {agent_id: 'aura9-v2.4'}) "
    "ON CREATE SET a.created_at = timestamp(), a.version = '2.4.0'"
)


class MigrationRunner:
    """Applies FalkorDB graph schema migrations.

    Scans ``src/migrations/*.cypher`` files, tracks applied migrations in a
    ``(:Migration {version, applied_at})`` node, and only applies unapproved ones.
    """

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

            applied = self._get_applied_versions(graph)
            cypher_files = sorted(
                glob.glob(str(_MIGRATIONS_DIR / "*.cypher"))
            )

            if not cypher_files:
                logger.info("MigrationRunner: no .cypher files found")
            else:
                for filepath in cypher_files:
                    version = os.path.basename(filepath)
                    if version in applied:
                        logger.debug(f"Migration already applied: {version}")
                        continue

                    content = Path(filepath).read_text(encoding="utf-8")
                    # Execute each non-empty, non-comment statement
                    for statement in self._split_statements(content):
                        try:
                            graph.query(statement)
                            logger.info(f"Migration statement applied: {statement[:80]}")
                        except Exception as exc:
                            logger.debug(f"Migration note ({version}): {exc}")

                    self._mark_applied(graph, version)
                    logger.info(f"Migration complete: {version}")

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

    @staticmethod
    def _get_applied_versions(graph) -> set[str]:
        """Return set of already-applied migration version strings."""
        try:
            result = graph.query("MATCH (m:Migration) RETURN m.version")
            return {row[0] for row in result.result_set if row}
        except Exception:
            return set()

    @staticmethod
    def _mark_applied(graph, version: str) -> None:
        applied_at = datetime.now(UTC).isoformat()
        try:
            # Use JSON-encoded strings to safely escape special characters
            import json  # noqa: PLC0415
            safe_version = json.dumps(version)
            safe_applied_at = json.dumps(applied_at)
            graph.query(
                f"MERGE (m:Migration {{version: {safe_version}}}) "
                f"ON CREATE SET m.applied_at = {safe_applied_at}"
            )
        except Exception as exc:
            logger.warning(f"MigrationRunner: could not record migration {version}: {exc}")

    @staticmethod
    def _split_statements(content: str) -> list[str]:
        """Split a Cypher file into individual statements, ignoring comments."""
        statements = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                statements.append(stripped)
        return statements

    def _print_migrations(self) -> None:
        """Print migrations that would be applied (dry run)."""
        cypher_files = sorted(glob.glob(str(_MIGRATIONS_DIR / "*.cypher")))
        logger.info("Dry-run migrations:")
        for f in cypher_files:
            logger.info(f"  Would apply: {os.path.basename(f)}")

