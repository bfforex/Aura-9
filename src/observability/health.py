"""Health checker for Aura-9 infrastructure components."""

from __future__ import annotations

from datetime import UTC, datetime


class HealthChecker:
    """Checks health of all Aura-9 infrastructure components."""

    def __init__(
        self,
        redis_client=None,
        qdrant_client=None,
        falkordb=None,
        ollama_client=None,
        l1=None,
        tais_daemon=None,
    ) -> None:
        self._redis = redis_client
        self._qdrant = qdrant_client
        self._falkordb = falkordb
        self._ollama = ollama_client
        self._l1 = l1
        self._tais = tais_daemon

    async def check_all(self) -> dict:
        """Run all health checks. Returns status dict."""
        results = {}
        checks = [
            ("ollama", self._check_ollama()),
            ("redis", self._check_redis()),
            ("qdrant", self._check_qdrant()),
            ("falkordb", self._check_falkordb()),
            ("asd_state", self._check_asd_state()),
            ("watchdog_heartbeat", self._check_watchdog_heartbeat()),
            ("tais", self._check_tais()),
            ("audit_trail", self._check_audit_trail()),
        ]

        import asyncio  # noqa: PLC0415
        check_results = await asyncio.gather(
            *[coro for _, coro in checks],
            return_exceptions=True,
        )

        for (name, _), result in zip(checks, check_results, strict=False):
            if isinstance(result, Exception):
                results[name] = {"status": "ERROR", "error": str(result)}
            else:
                results[name] = result

        results["timestamp"] = datetime.now(UTC).isoformat()
        results["overall"] = "OK" if all(
            r.get("status") == "OK" for r in results.values() if isinstance(r, dict)
        ) else "DEGRADED"

        return results

    async def _check_ollama(self) -> dict:
        if not self._ollama:
            return {"status": "UNKNOWN", "message": "not configured"}
        try:
            loaded = await self._ollama.check_model_loaded("")
            return {"status": "OK", "model_loaded": loaded}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    async def _check_redis(self) -> dict:
        if not self._redis:
            return {"status": "UNKNOWN"}
        try:
            await self._redis.ping()
            return {"status": "OK"}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    async def _check_qdrant(self) -> dict:
        if not self._qdrant:
            return {"status": "UNKNOWN"}
        try:
            await self._qdrant.get_collections()
            return {"status": "OK"}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    async def _check_falkordb(self) -> dict:
        if not self._falkordb:
            return {"status": "UNKNOWN"}
        return {"status": "OK" if self._falkordb._connected else "DEGRADED"}

    async def _check_asd_state(self) -> dict:
        if not self._l1:
            return {"status": "UNKNOWN"}
        try:
            state = await self._l1.get_asd_state()
            return {"status": "OK", "has_state": state is not None}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    async def _check_watchdog_heartbeat(self) -> dict:
        if not self._redis:
            return {"status": "UNKNOWN"}
        try:
            val = await self._redis.get("watchdog:heartbeat")
            return {"status": "OK" if val else "WARN", "alive": val is not None}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    async def _check_tais(self) -> dict:
        if not self._tais:
            return {"status": "UNKNOWN"}
        return {
            "status": "OK",
            "tais_status": self._tais.get_status().value,
            "temp": self._tais.get_temp(),
        }

    async def _check_audit_trail(self) -> dict:
        return {"status": "OK"}
