"""MCP Gateway — routes tool calls to registered MCP servers."""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from loguru import logger

from src.tools.base import ToolResult

_TIER_AUTO = "auto"
_TIER_AUTO_LOG = "auto_with_log"
_TIER_HUMAN_GATE = "human_gate"


class MCPGateway:
    """Routes tool calls to MCP servers with tier enforcement."""

    def __init__(
        self,
        credentials_path: str = "./secrets/mcp-credentials.yaml",
        l1=None,
        sanitizer=None,
        financial_gate=None,
        human_gate=None,
        audit_trail=None,
    ) -> None:
        self._creds_path = credentials_path
        self._l1 = l1
        self._sanitizer = sanitizer
        self._financial_gate = financial_gate
        self._human_gate = human_gate
        self._audit = audit_trail
        self._servers: dict[str, dict] = {}
        self._disabled_servers: set[str] = set()
        self._daily_limits: dict[str, int] = {}
        self._load_credentials()

    def _load_credentials(self) -> None:
        path = Path(self._creds_path)
        if not path.exists():
            logger.debug(f"MCP: credentials file not found at {path}")
            return
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self._servers = data.get("servers", {})
            self._daily_limits = {
                sid: cfg.get("daily_limit", 100)
                for sid, cfg in self._servers.items()
            }
            logger.info(f"MCP: loaded {len(self._servers)} server(s)")
        except Exception as exc:
            logger.warning(f"MCP: failed to load credentials: {exc}")

    async def call(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
        session_id: str,
    ) -> ToolResult:
        """Execute a tool call against an MCP server."""
        t0 = time.monotonic()

        if server_id in self._disabled_servers:
            return ToolResult(success=False, output=None, error=f"Server {server_id} is disabled")

        # Financial gate check
        if self._financial_gate and self._financial_gate.check(server_id, tool_name, arguments):
            approved = await self._financial_gate.request_confirmation({
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "session_id": session_id,
            })
            if not approved:
                return ToolResult(success=False, output=None, error="Financial gate: not approved")

        # Sanitize inputs
        if self._sanitizer:
            arguments = self._sanitizer.sanitize(arguments)

        # Check daily limit
        if self._l1:
            count = await self._l1.increment_mcp_calls(server_id)
            limit = self._daily_limits.get(server_id, 100)
            if count > limit:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Daily call limit ({limit}) exceeded for {server_id}",
                )

        # Tier enforcement
        server_cfg = self._servers.get(server_id, {})
        tier = server_cfg.get("tier", _TIER_AUTO)

        if tier == _TIER_HUMAN_GATE and self._human_gate:
            response = await self._human_gate.request(
                question=f"MCP Tier3 call: {server_id}.{tool_name}({arguments})",
                context="External write operation requires human approval",
                task_id="",
                session_id=session_id,
            )
            if not response.approved:
                return ToolResult(success=False, output=None, error="Human gate: not approved")

        if tier == _TIER_AUTO_LOG and self._audit:
            await self._audit.write(
                event_type="MCP_TIER2_CALL",
                data={"server_id": server_id, "tool_name": tool_name},
                session_id=session_id,
            )

        # Execute the call via JSON-RPC 2.0 when a URL is available
        url = server_cfg.get("url")
        api_key = server_cfg.get("api_key", "")

        if url:
            elapsed_ms = (time.monotonic() - t0) * 1000
            try:
                import httpx  # noqa: PLC0415

                timeout = server_cfg.get("timeout_seconds", 30)
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                payload = {
                    "jsonrpc": "2.0",
                    "method": tool_name,
                    "params": arguments,
                    "id": 1,
                }

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.info(f"MCP: {server_id}.{tool_name} completed in {elapsed_ms:.0f}ms")

                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    return ToolResult(
                        success=False,
                        output=None,
                        error=msg,
                        execution_time_ms=elapsed_ms,
                    )

                return ToolResult(
                    success=True,
                    output=data.get("result"),
                    execution_time_ms=elapsed_ms,
                )

            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.error(f"MCP: {server_id}.{tool_name} failed: {exc}")
                return ToolResult(
                    success=False,
                    output=None,
                    error=str(exc),
                    execution_time_ms=elapsed_ms,
                )

        # No URL configured — return stub OK (server not yet registered)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"MCP: calling {server_id}.{tool_name} (no URL — stub response)")
        return ToolResult(
            success=True,
            output={"server_id": server_id, "tool_name": tool_name, "result": "OK"},
            execution_time_ms=elapsed_ms,
        )

    async def get_daily_count(self, server_id: str) -> int:
        """Get today's call count for a server."""
        if self._l1:
            from datetime import date  # noqa: PLC0415
            today = date.today().isoformat()
            val = await self._l1._r.get(f"mcp:calls:{server_id}:{today}")
            return int(val) if val else 0
        return 0

    def disable_server(self, server_id: str) -> None:
        self._disabled_servers.add(server_id)
        logger.info(f"MCP: disabled server {server_id}")

    def enable_server(self, server_id: str) -> None:
        self._disabled_servers.discard(server_id)
        logger.info(f"MCP: enabled server {server_id}")
